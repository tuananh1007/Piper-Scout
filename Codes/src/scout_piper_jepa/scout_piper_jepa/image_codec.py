"""sensor_msgs/Image <-> numpy without cv_bridge.

Works on anything with ``height``, ``width``, ``encoding``, ``step``,
``is_bigendian`` and ``data`` attributes, so it is used both by the live node
and by the rosbag exporter (and is testable without ROS).
"""

from __future__ import annotations

import numpy as np

_DTYPES = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4), "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1),
    "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1), "32FC1": (np.float32, 1),
}


def image_to_numpy(msg) -> np.ndarray:
    if msg.encoding not in _DTYPES:
        raise ValueError(f"unsupported encoding {msg.encoding!r}")
    dtype, ch = _DTYPES[msg.encoding]
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    px = row[:, : msg.width * ch * dt.itemsize].copy().view(dt)
    arr = px.reshape(msg.height, msg.width, ch).astype(dtype)
    return arr[..., 0] if ch == 1 else arr


def numpy_to_mono8(a: np.ndarray):
    """[0,1] float array -> sensor_msgs/Image mono8 (ROS import is lazy)."""
    from sensor_msgs.msg import Image  # noqa: PLC0415
    a8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    msg = Image()
    msg.height, msg.width = a8.shape
    msg.encoding, msg.step, msg.is_bigendian = "mono8", a8.shape[1], 0
    msg.data = a8.tobytes()
    return msg
