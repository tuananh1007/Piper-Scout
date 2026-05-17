"""Segmentation node — ROS 2 Humble port of stem_grasp_ros1/scripts/segmentation_node.py.

Faithful port preserving every parameter name, every topic name, and the
full inference pipeline:

  - YOLO-seg path (ultralytics)
  - Grounded-SAM path (GroundingDINO + SAM) — optional, requires checkpoints
  - HSV green fallback
  - Frangi vesselness fusion (thin-feature recall booster)
  - Depth pre- / post-filter (range gating)
  - Mask dilation (thicken thin masks)
  - Instance selection: highest_conf or closest_to_base
  - Stationary gate via joint_states
  - Target caption channel (fruit/flower) → /stem_grasp/target_mask and
    /stem_grasp/target_point in camera optical frame
  - Debug overlay image

The only behavioral changes vs the ROS 1 source are mechanical ROS 1→2 swaps:
  - rospy.Time(0) ↔ rclpy.time.Time()
  - CameraInfo.K is now lowercase: msg.k[]
  - logwarn_throttle replaced by a simple manual throttle map
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None  # type: ignore[assignment]
    _YOLO_OK = False

try:
    from skimage.filters import frangi as _frangi
    _FRANGI_OK = True
except Exception:
    _frangi = None  # type: ignore[assignment]
    _FRANGI_OK = False


class _Throttle:
    """Tiny replacement for rospy.logwarn_throttle / loginfo_throttle.

    rclpy has no built-in throttle; this keeps a per-key timestamp map.
    """

    def __init__(self) -> None:
        self._t: dict[str, float] = {}

    def ok(self, key: str, period_sec: float) -> bool:
        now = time.monotonic()
        last = self._t.get(key, 0.0)
        if now - last >= period_sec:
            self._t[key] = now
            return True
        return False


class StemSegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__("stem_grasp_segmentation")

        # --- Parameters (mirrors every ROS 1 ~param) -----------------------
        p = self._declare_params(
            publish_debug=True,
            conf_threshold=0.35,
            mask_mode="yolo_seg",
            yolo_model_path="",
            yolo_conf=0.35,
            yolo_iou=0.45,
            yolo_imgsz=960,
            yolo_mask_threshold=0.25,
            yolo_stem_class_id=0,
            text_prompt="stem",
            enable_context_prompt=False,
            context_prompt_fallback_to_base=True,
            context_prompt_clauses=["hanging fruit", "pot"],
            grounded_sam_extra_captions=[],
            target_caption="",
            target_min_pixels=30,
            enable_depth_prefilter=False,
            enable_depth_postfilter=True,
            segmentation_min_depth_m=0.05,
            segmentation_max_depth_m=1.20,
            mask_dilate_kernel=3,
            mask_dilate_iterations=1,
            enable_vesselness_fusion=False,
            vessel_sigma_min=1.0,
            vessel_sigma_max=4.0,
            vessel_sigma_step=1.0,
            vessel_threshold=0.05,
            vessel_roi_dilate_px=25,
            vessel_downsample=2,
            hsv_lower=[30, 30, 30],
            hsv_upper=[95, 255, 255],
            instance_selection_mode="highest_conf",
            selection_frame="base_link",
            camera_optical_frame="camera_color_optical_frame",
            depth_topic="/camera/depth/image_rect_raw",
            color_topic="/camera/color/image_raw",
            camera_info_topic="/camera/color/camera_info",
            joint_states_topic="/joint_states",
            min_valid_depth_pixels=50,
            torch_device="auto",
            grounded_sam_every_n_frames=8,
            grounded_sam_caption_mode="best",
            gdino_config="",
            gdino_checkpoint="",
            gdino_box_threshold=0.9,
            gdino_text_threshold=0.9,
            sam_model_type="vit_b",
            sam_checkpoint="",
            grasp_uv_timeout_sec=2.0,
            segment_only_when_stationary=True,
            stationary_joint_vel_threshold=0.02,
            joint_states_timeout_sec=0.75,
            stationary_settle_sec=0.12,
        )
        self.p = p
        self.throttle = _Throttle()
        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # HSV bounds → uint8 vectors
        self.hsv_lower = self._coerce_hsv(self.p["hsv_lower"], default=[30, 30, 30])
        self.hsv_upper = self._coerce_hsv(self.p["hsv_upper"], default=[95, 255, 255])

        self.instance_selection_mode = str(self.p["instance_selection_mode"]).strip().lower()
        if self.instance_selection_mode not in ("highest_conf", "closest_to_base"):
            self.get_logger().warn(
                f"invalid instance_selection_mode={self.instance_selection_mode!r}, "
                "using highest_conf"
            )
            self.instance_selection_mode = "highest_conf"

        # Inference device
        self._device = self._resolve_device(str(self.p["torch_device"]))

        # Camera state
        self._fx = self._fy = self._cx = self._cy = None
        self.latest_depth: Optional[Image] = None
        self.latest_depth_stamp: Optional[RclpyTime] = None

        # Mask caches
        self._cached_mask: Optional[np.ndarray] = None
        self._cached_conf: float = 0.0
        self._latest_target_mask: Optional[np.ndarray] = None
        self._cached_target_mask: Optional[np.ndarray] = None
        self._frame_count = 0

        # Stationary gate
        self.latest_joint_state_stamp: Optional[RclpyTime] = None
        self.last_motion_time: RclpyTime = self.get_clock().now()
        self._prev_joint_positions: Optional[np.ndarray] = None
        self._prev_joint_time: Optional[RclpyTime] = None

        # Grasp candidate uv cache (legacy debug)
        self.latest_grasp_uv: Optional[Tuple[float, float]] = None
        self.latest_grasp_uv_stamp: Optional[RclpyTime] = None

        # Models
        self.yolo = None
        self._gdino_ready = False
        self._gdino_model = None
        self._sam_predictor = None
        self._torch = None
        self._gdino_load_image = None
        self._gdino_predict = None
        self._box_convert = None

        if self.p["mask_mode"] == "yolo_seg":
            self.yolo = self._init_yolo()
            if self.yolo is None:
                self.get_logger().warn("YOLO requested but unavailable; segmentation disabled.")
                self.p["mask_mode"] = "disabled"
        elif self.p["mask_mode"] == "grounded_sam":
            self._init_grounded_sam()
            if not self._gdino_ready:
                self.get_logger().warn(
                    "Grounded-SAM requested but unavailable; segmentation disabled."
                )
                self.p["mask_mode"] = "disabled"

        # --- Publishers ----------------------------------------------------
        self.mask_pub = self.create_publisher(Image, "/stem_grasp/mask", 1)
        self.debug_pub = self.create_publisher(Image, "/stem_grasp/debug_image", 1)
        self.conf_pub = self.create_publisher(Float32, "/stem_grasp/stem_conf", 1)
        self.centroid_pub = self.create_publisher(
            PointStamped, "/stem_grasp/mask_centroid", 1
        )
        self.target_mask_pub = self.create_publisher(
            Image, "/stem_grasp/target_mask", 1
        )
        self.target_point_pub = self.create_publisher(
            PointStamped, "/stem_grasp/target_point", 1
        )
        self.combined_centroid_pub = self.create_publisher(
            PointStamped, "/stem_grasp/combined_centroid", 1
        )

        # --- Subscribers ---------------------------------------------------
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_rel = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            Image, self.p["color_topic"], self.image_cb, qos,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Image, self.p["depth_topic"], self.depth_cb, qos,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            CameraInfo, self.p["camera_info_topic"], self.camera_info_cb, qos,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            PointStamped, "/stem_grasp/grasp_candidate_uv", self.grasp_uv_cb, qos,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            JointState, self.p["joint_states_topic"], self.joint_state_cb, qos_rel,
            callback_group=self.cb_group,
        )

        self.get_logger().info(
            f"stem_grasp_ros2 segmentation node ready "
            f"(mode={self.p['mask_mode']}, device={self._device})"
        )

    # ------------------------------------------------------------------ params
    def _declare_params(self, **defaults) -> dict:
        out = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            out[name] = self.get_parameter(name).value
        return out

    @staticmethod
    def _coerce_hsv(raw, default) -> np.ndarray:
        try:
            if isinstance(raw, str):
                raw = [x.strip() for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]
            arr = np.asarray([int(float(x)) for x in raw], dtype=np.uint8)
        except Exception:
            arr = np.asarray(default, dtype=np.uint8)
        if arr.shape != (3,):
            arr = np.asarray(default, dtype=np.uint8)
        return arr

    def _resolve_device(self, requested: str) -> str:
        device = requested.strip().lower()
        if device in ("auto", "gpu"):
            try:
                import torch
                if torch.cuda.is_available():
                    self.get_logger().info(
                        f"segmentation auto-selected cuda:0 "
                        f"(torch={torch.__version__}, cuda={torch.version.cuda})"
                    )
                    return "cuda:0"
            except Exception:
                pass
            return "cpu"
        return requested

    # ----------------------------------------------------------------- models
    def _init_yolo(self):
        if not _YOLO_OK:
            self.get_logger().warn("ultralytics is not installed in this environment.")
            return None
        path = str(self.p["yolo_model_path"])
        if not path:
            self.get_logger().warn("yolo_model_path is empty.")
            return None
        if not Path(path).exists():
            self.get_logger().warn(f"YOLO model not found: {path}")
            return None
        try:
            model = YOLO(path)
            self.get_logger().info(f"loaded YOLO model from {path}")
            return model
        except Exception as exc:
            self.get_logger().warn(f"failed loading YOLO: {exc}")
            return None

    def _init_grounded_sam(self) -> None:
        try:
            import importlib
            import torch
            import torchvision
            gdino_inf = importlib.import_module("groundingdino.util.inference")
            sam_module = importlib.import_module("segment_anything")
            torchvision_ops = importlib.import_module("torchvision.ops")

            self.get_logger().info(
                f"Grounded-SAM stack: torch={torch.__version__}, "
                f"torchvision={torchvision.__version__}, "
                f"cuda={getattr(torch.version, 'cuda', None)}"
            )

            gconfig = str(self.p["gdino_config"])
            gckpt = str(self.p["gdino_checkpoint"])
            sam_type = str(self.p["sam_model_type"])
            sam_ckpt = str(self.p["sam_checkpoint"])
            if not gconfig or not gckpt or not sam_ckpt:
                self.get_logger().warn("Grounded-SAM checkpoint paths missing.")
                return

            self._torch = torch
            self._gdino_load_image = gdino_inf.load_image
            self._gdino_predict = gdino_inf.predict
            self._box_convert = torchvision_ops.box_convert
            self._gdino_model = gdino_inf.load_model(gconfig, gckpt, device=self._device)

            sam = sam_module.sam_model_registry[sam_type](checkpoint=sam_ckpt)
            sam.to(device=self._device)
            self._sam_predictor = sam_module.SamPredictor(sam)
            self._gdino_ready = True
            self.get_logger().info(f"Grounded-SAM ready on {self._device}")
        except Exception as exc:
            self.get_logger().warn(
                f"failed to init Grounded-SAM: {exc}\n{traceback.format_exc()}"
            )

    # ----------------------------------------------------------- captions
    def _grounded_sam_captions(self) -> list[str]:
        base = str(self.p["text_prompt"]).strip() or "stem"
        captions: list[str] = []
        clauses = list(self.p["context_prompt_clauses"]) if isinstance(
            self.p["context_prompt_clauses"], (list, tuple)
        ) else []
        if bool(self.p["enable_context_prompt"]) and clauses:
            for c in clauses:
                captions.append(f"{base} with {c}")
            if len(clauses) > 1:
                captions.append(f"{base} with {' and '.join(clauses)}")
        if (not captions) or bool(self.p["context_prompt_fallback_to_base"]):
            captions.append(base)
        for extra in list(self.p["grounded_sam_extra_captions"] or []):
            captions.append(str(extra))
        dedup, seen = [], set()
        for c in captions:
            k = c.strip().lower()
            if k and k not in seen:
                seen.add(k)
                dedup.append(c)
        return dedup

    # ---------------------------------------------------- segmentation paths
    def _segment_grounded_sam(self, bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        self._latest_target_mask = None
        if not self._gdino_ready:
            return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
            cv2.imwrite(temp_path, bgr)

            image_source, image_tensor = self._gdino_load_image(temp_path)
            box_thr = float(self.p["gdino_box_threshold"])
            txt_thr = float(self.p["gdino_text_threshold"])
            mode = str(self.p["grounded_sam_caption_mode"]).strip().lower()
            if mode not in ("best", "union"):
                mode = "best"

            per_caption = []
            for caption in self._grounded_sam_captions():
                boxes, logits, _ = self._gdino_predict(
                    model=self._gdino_model,
                    image=image_tensor,
                    caption=caption,
                    box_threshold=box_thr,
                    text_threshold=txt_thr,
                    device=self._device,
                )
                if boxes is None or len(boxes) == 0:
                    continue
                per_caption.append((caption, boxes, logits))

            if not per_caption:
                return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0

            if mode == "union":
                active = per_caption
            else:
                best_idx = max(
                    range(len(per_caption)),
                    key=lambda i: float(self._torch.max(per_caption[i][2]).item()),
                )
                active = [per_caption[best_idx]]

            best_boxes = self._torch.cat([d[1] for d in active], dim=0)
            best_logits = self._torch.cat([d[2] for d in active], dim=0)
            if best_boxes is None or len(best_boxes) == 0:
                return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0

            h, w, _ = image_source.shape
            boxes_scaled = best_boxes * self._torch.Tensor([w, h, w, h])
            xyxy = self._box_convert(
                boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy"
            ).to(self._device)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._sam_predictor.set_image(rgb)
            transformed = self._sam_predictor.transform.apply_boxes_torch(xyxy, rgb.shape[:2])
            masks, _, _ = self._sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )

            mask_list = [
                masks[i, 0].detach().cpu().numpy().astype(np.uint8) * 255
                for i in range(masks.shape[0])
            ]
            conf_list = (
                [float(best_logits[i].item()) for i in range(len(best_logits))]
                if len(best_logits) > 0
                else [0.0] * len(mask_list)
            )

            target_caption = str(self.p["target_caption"]).strip()
            if target_caption:
                tmask = self._segment_grounded_sam_target_mask(
                    bgr, image_source, image_tensor,
                    target_caption, box_thr, txt_thr,
                )
                if tmask is not None:
                    self._latest_target_mask = tmask

            return self._select_instance_mask(mask_list, conf_list)
        except Exception as exc:
            if self.throttle.ok("gsam_fail", 2.0):
                self.get_logger().warn(f"Grounded-SAM failed: {exc}")
            return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _segment_grounded_sam_target_mask(
        self, bgr, image_source, image_tensor, caption, box_thr, txt_thr
    ):
        caption = str(caption).strip()
        if not caption:
            return None
        boxes, logits, _ = self._gdino_predict(
            model=self._gdino_model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_thr,
            text_threshold=txt_thr,
            device=self._device,
        )
        if boxes is None or len(boxes) == 0:
            return None
        h, w, _ = image_source.shape
        boxes_scaled = boxes * self._torch.Tensor([w, h, w, h])
        xyxy = self._box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").to(self._device)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._sam_predictor.set_image(rgb)
        transformed = self._sam_predictor.transform.apply_boxes_torch(xyxy, rgb.shape[:2])
        masks, _, _ = self._sam_predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed, multimask_output=False,
        )
        target = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for i in range(masks.shape[0]):
            m = masks[i, 0].detach().cpu().numpy().astype(np.uint8) * 255
            if m.shape != bgr.shape[:2]:
                m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            target = cv2.bitwise_or(target, m)
        if int(np.count_nonzero(target)) < int(self.p["target_min_pixels"]):
            return None
        return target

    def _segment_yolo(self, bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.yolo is None:
            return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0
        predict_kwargs = dict(
            source=bgr,
            conf=float(self.p["yolo_conf"]),
            iou=float(self.p["yolo_iou"]),
            device=self._device,
            verbose=False,
        )
        if int(self.p["yolo_imgsz"]) > 0:
            predict_kwargs["imgsz"] = int(self.p["yolo_imgsz"])
        try:
            results = self.yolo.predict(**predict_kwargs)
        except Exception as exc:
            if self.throttle.ok("yolo_fail", 2.0):
                self.get_logger().warn(f"YOLO failed: {exc}")
            return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0
        if not results:
            return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0
        result = results[0]
        if result.masks is None or result.masks.data is None:
            return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0
        masks = result.masks.data.cpu().numpy()
        classes = (
            result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes is not None and result.boxes.cls is not None
            else []
        )
        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None and result.boxes.conf is not None
            else []
        )
        thr = float(self.p["yolo_mask_threshold"])
        stem_class = int(self.p["yolo_stem_class_id"])
        mc, cc = [], []
        for idx, m in enumerate(masks):
            cid = int(classes[idx]) if len(classes) > idx else -1
            det = float(confs[idx]) if len(confs) > idx else 0.0
            if cid != stem_class:
                continue
            mc.append((m > thr).astype(np.uint8) * 255)
            cc.append(det)
        if not mc:
            return np.zeros(bgr.shape[:2], dtype=np.uint8), 0.0
        mask, best_conf = self._select_instance_mask(mc, cc)
        if mask.shape != bgr.shape[:2]:
            mask = cv2.resize(
                mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        return mask, best_conf

    def _segment_hsv(self, bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        if int(np.count_nonzero(mask)) == 0:
            return mask, 0.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        conf = float(np.count_nonzero(mask)) / float(mask.shape[0] * mask.shape[1])
        return mask, min(conf * 50.0, 1.0)

    # ------------------------------------------------------ mask refinement
    def _apply_mask_dilation(self, mask: np.ndarray) -> np.ndarray:
        k = int(self.p["mask_dilate_kernel"])
        it = int(self.p["mask_dilate_iterations"])
        if k <= 0 or it <= 0 or int(np.count_nonzero(mask)) == 0:
            return mask
        k |= 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask, kernel, iterations=it)

    def _vesselness_refine(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not bool(self.p["enable_vesselness_fusion"]):
            return mask
        if not _FRANGI_OK:
            if self.throttle.ok("frangi_missing", 60.0):
                self.get_logger().warn(
                    "enable_vesselness_fusion=true but skimage.filters.frangi unavailable"
                )
            return mask
        if int(np.count_nonzero(mask)) == 0:
            return mask
        try:
            ds = max(int(self.p["vessel_downsample"]), 1)
            h, w = bgr.shape[:2]
            if ds > 1:
                bgr_s = cv2.resize(bgr, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
                mask_s = cv2.resize(mask, (w // ds, h // ds), interpolation=cv2.INTER_NEAREST)
            else:
                bgr_s, mask_s = bgr, mask
            green = bgr_s[:, :, 1].astype(np.float32) / 255.0
            inverted = 1.0 - green
            s_min = max(float(self.p["vessel_sigma_min"]), 0.5)
            s_max = max(float(self.p["vessel_sigma_max"]), s_min)
            s_step = max(float(self.p["vessel_sigma_step"]), 0.5)
            sigmas = np.arange(s_min, s_max + 1e-6, s_step)
            if len(sigmas) == 0:
                sigmas = [s_min]
            response = _frangi(inverted, sigmas=sigmas, black_ridges=False)
            roi_k = max(int(self.p["vessel_roi_dilate_px"]) // ds, 1) | 1
            roi_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (roi_k, roi_k))
            roi = cv2.dilate(mask_s, roi_kernel, iterations=1) > 0
            vessel = (
                (response > float(self.p["vessel_threshold"])) & roi
            ).astype(np.uint8) * 255
            if ds > 1:
                vessel = cv2.resize(vessel, (w, h), interpolation=cv2.INTER_NEAREST)
            return cv2.bitwise_or(mask, vessel)
        except Exception as exc:
            if self.throttle.ok("frangi_fail", 5.0):
                self.get_logger().warn(f"vesselness refinement failed: {exc}")
            return mask

    @staticmethod
    def _make_masks_mutually_exclusive(mask_a, mask_b):
        if mask_a is None or mask_b is None:
            return mask_a, mask_b
        a = np.asarray(mask_a, dtype=np.uint8)
        b = np.asarray(mask_b, dtype=np.uint8)
        if a.shape != b.shape:
            return a, b
        overlap = np.logical_and(a > 0, b > 0)
        if not np.any(overlap):
            return a, b
        a_excl = cv2.bitwise_and(a, cv2.bitwise_not(b))
        return a_excl, b

    # ------------------------------------------------------------ instance
    def _select_instance_mask(self, mc, cc) -> Tuple[np.ndarray, float]:
        if not mc:
            return np.zeros((1, 1), dtype=np.uint8), 0.0
        if self.instance_selection_mode == "closest_to_base":
            idx = self._closest_mask_index(mc)
            if idx is not None:
                return mc[idx], float(cc[idx]) if len(cc) > idx else 0.0
            if self.throttle.ok("close_fallback", 2.0):
                self.get_logger().warn(
                    "closest_to_base requested but depth/TF not ready; falling back"
                )
        idx = int(np.argmax(np.asarray(cc, dtype=np.float32))) if cc else 0
        return mc[idx], float(cc[idx]) if cc else 0.0

    def _closest_mask_index(self, mc) -> Optional[int]:
        if self.latest_depth is None or self._fx is None or self._fy is None:
            return None
        try:
            depth = self.bridge.imgmsg_to_cv2(
                self.latest_depth, desired_encoding="passthrough"
            )
        except Exception as exc:
            if self.throttle.ok("depth_decode", 2.0):
                self.get_logger().warn(f"depth decode failed: {exc}")
            return None
        depth = depth.astype(np.float32)
        if np.nanmax(depth) > 10.0:
            depth = depth / 1000.0
        best_idx, best_dist = None, float("inf")
        for idx, m in enumerate(mc):
            mb = m > 0
            if depth.shape[:2] != mb.shape[:2]:
                continue
            z = depth[mb]
            valid = (
                np.isfinite(z)
                & (z >= float(self.p["segmentation_min_depth_m"]))
                & (z <= float(self.p["segmentation_max_depth_m"]))
            )
            if int(np.count_nonzero(valid)) < int(self.p["min_valid_depth_pixels"]):
                continue
            zm = float(np.median(z[valid]))
            c = self._mask_centroid(m)
            if c is None:
                continue
            u, v = c
            x = (u - self._cx) * zm / self._fx
            y = (v - self._cy) * zm / self._fy
            p_base = self._point_camera_to_frame(
                np.array([x, y, zm], dtype=np.float32),
                str(self.p["selection_frame"]),
            )
            if p_base is None:
                continue
            d = float(np.linalg.norm(p_base))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def _point_camera_to_frame(self, point_xyz, target_frame):
        camera_frame = str(self.p["camera_optical_frame"])
        if target_frame == camera_frame:
            return point_xyz.astype(np.float32)
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame, camera_frame, RclpyTime(), Duration(seconds=0.15)
            )
        except Exception as exc:
            if self.throttle.ok(f"tf_{target_frame}", 2.0):
                self.get_logger().warn(
                    f"TF lookup failed ({camera_frame} -> {target_frame}): {exc}"
                )
            return None
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        # Quaternion-to-matrix (avoid pulling in scipy here; same math as ROS 1)
        qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
        norm = qx * qx + qy * qy + qz * qz + qw * qw
        if norm < 1e-12:
            return None
        inv = 1.0 / norm
        qx, qy, qz, qw = qx * inv, qy * inv, qz * inv, qw * inv
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        rot = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        trans = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float32)
        return rot @ point_xyz.astype(np.float32) + trans

    @staticmethod
    def _mask_centroid(mask):
        pts = np.argwhere(mask > 0)
        if len(pts) == 0:
            return None
        return float(np.mean(pts[:, 1])), float(np.mean(pts[:, 0]))

    @staticmethod
    def _largest_connected_component_mask(mask):
        if mask is None:
            return None
        m = np.asarray(mask, dtype=np.uint8)
        if m.size == 0 or int(np.count_nonzero(m)) == 0:
            return None
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (m > 0).astype(np.uint8), connectivity=8
        )
        if n <= 1:
            return m
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(m, dtype=np.uint8)
        out[labels == largest] = 255
        return out

    # ------------------------------------------------------------ depth
    def _depth_range_mask(self, target_shape=None):
        if self.latest_depth is None:
            return None
        try:
            depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="passthrough")
        except Exception:
            return None
        depth = depth.astype(np.float32)
        if np.nanmax(depth) > 10.0:
            depth = depth / 1000.0
        valid = (
            np.isfinite(depth)
            & (depth >= float(self.p["segmentation_min_depth_m"]))
            & (depth <= float(self.p["segmentation_max_depth_m"]))
        )
        if target_shape is not None and valid.shape[:2] != target_shape:
            valid = (
                cv2.resize(
                    valid.astype(np.uint8),
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        return valid

    # ----------------------------------------------------------- callbacks
    def depth_cb(self, msg: Image) -> None:
        self.latest_depth = msg
        self.latest_depth_stamp = RclpyTime.from_msg(msg.header.stamp)

    def camera_info_cb(self, msg: CameraInfo) -> None:
        if len(msg.k) < 6:
            return
        fx, fy, cx, cy = float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy

    def grasp_uv_cb(self, msg: PointStamped) -> None:
        self.latest_grasp_uv = (float(msg.point.x), float(msg.point.y))
        self.latest_grasp_uv_stamp = RclpyTime.from_msg(msg.header.stamp)

    def joint_state_cb(self, msg: JointState) -> None:
        now = self.get_clock().now()
        stamp = RclpyTime.from_msg(msg.header.stamp) if msg.header.stamp else now
        self.latest_joint_state_stamp = stamp

        max_speed = 0.0
        if msg.velocity and len(msg.velocity) > 0:
            max_speed = float(np.max(np.abs(np.asarray(msg.velocity, dtype=np.float32))))
        elif msg.position and len(msg.position) > 0:
            positions = np.asarray(msg.position, dtype=np.float32)
            if self._prev_joint_positions is not None and self._prev_joint_time is not None:
                dt = (stamp - self._prev_joint_time).nanoseconds * 1e-9
                if dt > 1e-3 and positions.shape == self._prev_joint_positions.shape:
                    vel = np.abs((positions - self._prev_joint_positions) / dt)
                    max_speed = float(np.max(vel)) if vel.size > 0 else 0.0
            self._prev_joint_positions = positions
            self._prev_joint_time = stamp

        if max_speed > float(self.p["stationary_joint_vel_threshold"]):
            self.last_motion_time = now

    def _robot_is_moving(self) -> bool:
        if not bool(self.p["segment_only_when_stationary"]):
            return False
        now = self.get_clock().now()
        if self.latest_joint_state_stamp is None:
            if self.throttle.ok("no_joint_states", 5.0):
                self.get_logger().warn("motion gate on but no joint states yet")
            return False
        age = (now - self.latest_joint_state_stamp).nanoseconds * 1e-9
        if age > float(self.p["joint_states_timeout_sec"]):
            if self.throttle.ok("joint_state_stale", 5.0):
                self.get_logger().warn(f"joint states stale ({age:.3f}s)")
            return False
        since = (now - self.last_motion_time).nanoseconds * 1e-9
        return since < float(self.p["stationary_settle_sec"])

    # ----------------------------------------------- main image_cb pipeline
    def image_cb(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge failed: {exc}")
            return

        self._frame_count += 1

        # Depth prefilter
        depth_keep = None
        bgr_for_seg = bgr
        if bool(self.p["enable_depth_prefilter"]):
            depth_keep = self._depth_range_mask(target_shape=bgr.shape[:2])
            if depth_keep is not None:
                bgr_for_seg = bgr.copy()
                bgr_for_seg[~depth_keep] = 0

        # Motion gate
        if self._robot_is_moving():
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            stem_conf = 0.0
            out = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            out.header = msg.header
            self.mask_pub.publish(out)
            self.conf_pub.publish(Float32(data=stem_conf))
            if bool(self.p["publish_debug"]):
                dbg = bgr.copy()
                cv2.putText(
                    dbg, "segmentation paused: robot moving",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                )
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))
            return

        # Inference
        mode = str(self.p["mask_mode"])
        if mode == "yolo_seg":
            mask, det_conf = self._segment_yolo(bgr_for_seg)
            stem_conf = float(det_conf)
            if stem_conf <= 0.0:
                stem_conf = float(np.count_nonzero(mask)) / float(mask.shape[0] * mask.shape[1])
        elif mode == "grounded_sam":
            every = max(int(self.p["grounded_sam_every_n_frames"]), 1)
            run_now = (self._frame_count % every == 0) or (self._cached_mask is None)
            if run_now:
                mask, det_conf = self._segment_grounded_sam(bgr_for_seg)
                self._cached_mask = mask.copy()
                self._cached_conf = float(det_conf)
                self._cached_target_mask = (
                    self._latest_target_mask.copy()
                    if self._latest_target_mask is not None
                    else None
                )
            else:
                mask = (
                    self._cached_mask.copy()
                    if self._cached_mask is not None
                    else np.zeros(bgr.shape[:2], dtype=np.uint8)
                )
                det_conf = self._cached_conf
            stem_conf = float(det_conf)
            if stem_conf <= 0.0:
                stem_conf = float(np.count_nonzero(mask)) / float(mask.shape[0] * mask.shape[1])
        elif mode == "hsv_green":
            mask, det_conf = self._segment_hsv(bgr_for_seg)
            stem_conf = float(det_conf)
        else:
            if self.throttle.ok("bad_mode", 5.0):
                self.get_logger().warn(f"unsupported mask_mode={mode}; disabled")
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            stem_conf = 0.0

        # Refinement
        mask = self._vesselness_refine(bgr, mask)
        mask = self._apply_mask_dilation(mask)

        if bool(self.p["enable_depth_postfilter"]):
            post_keep = (
                depth_keep
                if depth_keep is not None
                else self._depth_range_mask(target_shape=bgr.shape[:2])
            )
            if post_keep is not None:
                mask[~post_keep] = 0

        if stem_conf < float(self.p["conf_threshold"]) * 0.05:
            mask[:] = 0
            stem_conf = 0.0

        if mode == "grounded_sam" and self._cached_target_mask is not None:
            mask, self._cached_target_mask = self._make_masks_mutually_exclusive(
                mask, self._cached_target_mask
            )
            self._cached_target_mask = self._apply_mask_dilation(self._cached_target_mask)

        centroid = self._mask_centroid(mask)

        out = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        out.header = msg.header
        self.mask_pub.publish(out)
        self.conf_pub.publish(Float32(data=stem_conf))
        self._publish_centroid(msg.header, centroid)
        self._publish_target(msg.header)
        self._publish_combined_centroid(msg.header, mask, self._cached_target_mask)

        if bool(self.p["publish_debug"]):
            self._publish_debug(bgr, mask, stem_conf, centroid)

    # ----------------------------------------------------------- publishers
    def _publish_centroid(self, header, centroid):
        if centroid is None:
            return
        u, v = centroid
        ps = PointStamped()
        ps.header = header
        ps.point.x, ps.point.y, ps.point.z = u, v, 0.0
        self.centroid_pub.publish(ps)

    def _publish_combined_centroid(self, header, mask, target_mask):
        if mask is None and target_mask is None:
            return
        if mask is not None and target_mask is not None:
            combined = cv2.bitwise_or(mask, target_mask)
        elif mask is not None:
            combined = mask
        else:
            combined = target_mask
        c = self._mask_centroid(combined)
        if c is None:
            return
        u, v = c
        ps = PointStamped()
        ps.header = header
        ps.point.x, ps.point.y, ps.point.z = u, v, 0.0
        self.combined_centroid_pub.publish(ps)

    def _publish_target(self, header):
        if not str(self.p["target_caption"]).strip():
            return
        tmask = self._cached_target_mask
        if tmask is None or int(np.count_nonzero(tmask)) < int(self.p["target_min_pixels"]):
            return
        try:
            tmsg = self.bridge.cv2_to_imgmsg(tmask, encoding="mono8")
            tmsg.header = header
            self.target_mask_pub.publish(tmsg)
        except Exception as exc:
            if self.throttle.ok("target_pub_fail", 5.0):
                self.get_logger().warn(f"target_mask publish failed: {exc}")
        if self._fx is None or self.latest_depth is None:
            return
        cluster = self._largest_connected_component_mask(tmask)
        if cluster is None or int(np.count_nonzero(cluster)) < int(self.p["target_min_pixels"]):
            return
        c = self._mask_centroid(cluster)
        if c is None:
            return
        u, v = c
        try:
            depth = self.bridge.imgmsg_to_cv2(
                self.latest_depth, desired_encoding="passthrough"
            )
        except Exception:
            return
        depth = depth.astype(np.float32)
        if np.nanmax(depth) > 10.0:
            depth = depth / 1000.0
        ui, vi = int(round(u)), int(round(v))
        hd, wd = depth.shape[:2]
        u0, u1 = max(0, ui - 5), min(wd, ui + 6)
        v0, v1 = max(0, vi - 5), min(hd, vi + 6)
        patch = depth[v0:v1, u0:u1]
        valid = (
            np.isfinite(patch)
            & (patch >= float(self.p["segmentation_min_depth_m"]))
            & (patch <= float(self.p["segmentation_max_depth_m"]))
        )
        if not np.any(valid):
            return
        z = float(np.median(patch[valid]))
        x = (u - self._cx) * z / self._fx
        y = (v - self._cy) * z / self._fy
        ps = PointStamped()
        ps.header.stamp = header.stamp
        ps.header.frame_id = str(self.p["camera_optical_frame"])
        ps.point.x, ps.point.y, ps.point.z = float(x), float(y), float(z)
        self.target_point_pub.publish(ps)

    def _publish_debug(self, bgr, mask, stem_conf, centroid):
        dbg = bgr.copy()
        if np.count_nonzero(mask) > 0:
            overlay = dbg.copy()
            overlay[mask > 0] = (0, 180, 0)
            dbg = cv2.addWeighted(overlay, 0.28, dbg, 0.72, 0.0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, contours, -1, (0, 255, 0), 2)
        if centroid is not None:
            cv2.drawMarker(
                dbg, (int(centroid[0]), int(centroid[1])),
                (0, 255, 255), cv2.MARKER_CROSS, 20, 2,
            )
        tmask = self._cached_target_mask
        if tmask is not None and int(np.count_nonzero(tmask)) > 0:
            overlay_t = dbg.copy()
            overlay_t[tmask > 0] = (0, 255, 255)
            dbg = cv2.addWeighted(overlay_t, 0.28, dbg, 0.72, 0.0)
            tc, _ = cv2.findContours(tmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg, tc, -1, (0, 255, 255), 2)
        cv2.putText(
            dbg, f"conf={stem_conf:.3f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StemSegmentationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
