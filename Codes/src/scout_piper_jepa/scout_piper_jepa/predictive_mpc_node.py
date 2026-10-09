"""C3: whole-body MPC with the Piper-JEPA visibility cost (P3B.11).

The geometry-only MPC node (scout_piper_whole_body_mpc ``WholeBodyMpcNode``,
same parameters, same node name ``whole_body_mpc`` so ``whole_body_mpc.yaml``
applies) plus, in the same process (dense features never leave it, ROADMAP
Phase 2A rule):

  * the Stage A target memory on the colour stream (re-grounded by a mask on
    /piper_jepa/init_mask, as target_state_node);
  * the Stage B predictor (``jepa_model``: a ``TorchACPredictor`` checkpoint
    from ``scout_piper_jepa.train``; empty = persistence, i.e. no prediction),
    stepped every ``jepa_stride`` MPC steps; ``jepa_stride`` 0 (default) takes
    it from the checkpoint's training frame interval (``step_s`` × ``rate_hz``),
    else 2;
  * the Stage C ``JepaVisibilityCost`` in the MPC cost, anchored on the
    target's metric position when depth gives one; the camera pose of each
    planned state is link6 (FK) times the link6 → camera transform read once
    from TF (the hand-eye calibration).

Without an initialised, visible target the cost adds nothing, so the node
behaves exactly like the geometry-only MPC (C2).

    ros2 run scout_piper_jepa predictive_mpc_node --ros-args \\
        --params-file $(ros2 pkg prefix scout_piper_whole_body_mpc)/share/scout_piper_whole_body_mpc/config/whole_body_mpc.yaml \\
        -p jepa_model:=runs/e3/P3.pt -p encoder:=vjepa -p hub_entry:=vjepa2_vit_large -p image_size:=256

Extra output: /piper_jepa/mpc_context (std_msgs/String JSON: target status,
whether the cost is active, last J_vis / J_id range).
"""

from __future__ import annotations

import json
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from scout_piper_whole_body_mpc.controller_node import WholeBodyMpcNode

from .encoder import make_encoder
from .image_codec import image_to_numpy
from .predictive_cost import JepaVisibilityCost, VisibilityCostWeights
from .predictor import PersistencePredictor, StateConditionedPredictor
from .target_memory import TargetMemory, TargetMemoryConfig


def _quat_to_T(tr) -> np.ndarray:
    q, t = tr.rotation, tr.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class _PersistenceStates:
    """StatePredictor with no prediction: the current features at every step."""

    def __init__(self, stride: int):
        self.stride = stride
        self.p = PersistencePredictor()

    def predict(self, Z_hist, X):
        n = len(range(self.stride, np.shape(X)[1], self.stride))
        return np.repeat(np.asarray(Z_hist)[None, -1:], len(X), 0).repeat(n, 1)


class PredictiveMpcNode(WholeBodyMpcNode):
    def __init__(self) -> None:
        super().__init__()
        self.declare_parameters("", [
            ("jepa_model", ""), ("jepa_device", "cuda"), ("jepa_stride", 0),   # 0 = from the checkpoint
            ("w_vis", 1.0), ("w_id", 1.0),
            ("encoder", "color_patch"), ("hub_entry", ""), ("image_size", 384), ("encoder_device", "cuda"),
            ("rgb_topic", "/camera/color/image_raw"),
            ("depth_topic", "/camera/aligned_depth_to_color/image_raw"),
            ("camera_info_topic", "/camera/color/camera_info"),
            ("init_mask_topic", "/piper_jepa/init_mask"),
            ("flange_frame", "piper_link6"),
            ("anchor_tolerance_m", 0.03),
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        kw = {}
        if p("encoder") != "color_patch":
            kw = {"device": p("encoder_device"), "image_size": int(p("image_size")),
                  **({"hub_entry": p("hub_entry")} if p("hub_entry") else {})}
        self.encoder = make_encoder(p("encoder"), **kw)
        stride = int(p("jepa_stride"))
        if p("jepa_model"):
            from .torch_predictor import TorchACPredictor, predictor_stride  # noqa: PLC0415 — needs torch
            net = TorchACPredictor.load(p("jepa_model"), device=p("jepa_device"))
            fit, warning = predictor_stride(net.cfg.step_s, self.model.dt)
            if warning:
                self.get_logger().warn(warning)
            if stride <= 0:
                stride = fit
            elif net.cfg.step_s > 0 and stride != fit:
                self.get_logger().warn(f"jepa_stride {stride} x {self.model.dt:.3f} s does not match the "
                                       f"training frame interval {net.cfg.step_s:.3f} s (jepa_stride {fit})")
            self.get_logger().info(f"predictor step: {stride} MPC steps ({stride * self.model.dt:.2f} s)")
            self.history = net.history
            self.grid_hw = tuple(net.cfg.grid_hw)
            predictor = StateConditionedPredictor(net, self.model.kin.tcp, stride)
        else:
            self.get_logger().warn("jepa_model empty: persistence predictor (the cost sees no prediction)")
            self.history, self.grid_hw = 1, None
            stride = stride if stride > 0 else 2
            predictor = _PersistenceStates(stride)
        self.memory = TargetMemory(TargetMemoryConfig())
        self.vis: Optional[JepaVisibilityCost] = None
        self.vis_args = dict(predictor=predictor, w=VisibilityCostWeights(w_vis=float(p("w_vis")),
                                                                           w_id=float(p("w_id"))),
                             stride=stride)
        self.flange = p("flange_frame")
        self.anchor_tol = float(p("anchor_tolerance_m"))
        self.T_flange_cam: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.cam_frame: Optional[str] = None
        self.clip = deque(maxlen=2)
        self.Z_hist = deque(maxlen=max(self.history, 1))
        self.pending_mask: Optional[np.ndarray] = None
        self.initialized = False
        self.p_world: Optional[np.ndarray] = None
        self.u_now: Optional[np.ndarray] = None
        from tf2_ros import Buffer, TransformListener  # noqa: PLC0415
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        be = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, p("rgb_topic"), self._rgb, be)
        self.create_subscription(Image, p("depth_topic"), self._depth, be)
        self.create_subscription(CameraInfo, p("camera_info_topic"), self._info, be)
        self.create_subscription(Image, p("init_mask_topic"), self._mask, 2)
        self.pub_ctx = self.create_publisher(String, "/piper_jepa/mpc_context", 5)
        self.get_logger().info("predictive MPC (C3): visibility cost active once the target is grounded")

    # ------------------------------------------------------------- inputs
    def _info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.cam_frame = msg.header.frame_id

    def _depth(self, msg: Image) -> None:
        d = image_to_numpy(msg).astype(np.float32)
        self.depth = d / 1000.0 if msg.encoding == "16UC1" else d

    def _mask(self, msg: Image) -> None:
        self.pending_mask = image_to_numpy(msg) > 0

    def _camera_pose(self, X: np.ndarray) -> np.ndarray:
        return self.model.world_frames(np.asarray(X, float))[..., 6, :, :] @ self.T_flange_cam

    def _flange_cam(self) -> Optional[np.ndarray]:
        if self.T_flange_cam is None and self.cam_frame:
            try:
                tr = self.tf_buf.lookup_transform(self.flange, self.cam_frame, rclpy.time.Time())
                self.T_flange_cam = _quat_to_T(tr.transform)
            except Exception:  # noqa: BLE001
                return None
        return self.T_flange_cam

    def _rgb(self, msg: Image) -> None:
        rgb = image_to_numpy(msg)
        rgb = rgb[..., ::-1] if msg.encoding == "bgr8" else rgb
        self.clip.append(np.ascontiguousarray(rgb))
        Z = self.encoder.encode(list(self.clip))
        if self.grid_hw is not None and tuple(Z.shape[:2]) != self.grid_hw:
            self.get_logger().error(f"encoder grid {Z.shape[:2]} != predictor grid {self.grid_hw}: "
                                    "train the predictor on this encoder", throttle_duration_sec=10.0)
            return
        if self.pending_mask is not None:
            try:
                self.memory.initialize(Z, self.pending_mask)
                self.initialized, self.p_world = True, None
                self.get_logger().info("target memory (re)initialised from mask")
            except ValueError as exc:
                self.get_logger().warn(f"init mask rejected: {exc}")
            self.pending_mask = None
        self.Z_hist.append(Z)
        if not self.initialized or self.K is None:
            return
        T_wc = None
        if self.base is not None and self.q is not None and self._flange_cam() is not None:
            T_wc = self._camera_pose(np.r_[self.base, self.q])
        s = self.memory.update(Z, image_hw=rgb.shape[:2], depth=self.depth, K=self.K, T_world_cam=T_wc)
        if s.visible:
            self.u_now = s.u_mean
            if s.p_world is not None and (self.p_world is None
                                          or np.linalg.norm(s.p_world - self.p_world) < self.anchor_tol):
                self.p_world = s.p_world
        active = s.status != "lost" and self.u_now is not None
        if active:
            if self.vis is None:
                self.vis = JepaVisibilityCost(image_hw=rgb.shape[:2], K=self.K,
                                              camera_pose=self._camera_pose if self.T_flange_cam is not None else None,
                                              **self.vis_args)
                self.extra_terms = [self.vis]
            self.vis.set_context(np.stack(self.Z_hist), self.memory.r, self.u_now, p_world=self.p_world)
        elif self.vis is not None:
            self.vis.clear()
        terms = self.vis.last_terms if self.vis is not None else {}
        self.pub_ctx.publish(String(data=json.dumps({
            "status": s.status, "visible": bool(s.visible), "cost_active": bool(active),
            "anchored": self.p_world is not None,
            **{k: [round(float(np.min(v)), 3), round(float(np.max(v)), 3)] for k, v in terms.items()}})))


def main(args=None) -> None:
    import signal  # noqa: PLC0415

    from rclpy.signals import SignalHandlerOptions  # noqa: PLC0415

    from scout_piper_whole_body_mpc.controller_node import _interrupt  # noqa: PLC0415
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = PredictiveMpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motion()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
