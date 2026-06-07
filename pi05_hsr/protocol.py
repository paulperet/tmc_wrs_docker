#!/usr/bin/env python
"""Shared contract between the host policy server and the in-container ROS bridge.

Both processes run in different Python interpreters (the server in the lerobot
venv on the host, the bridge in the ROS Noetic container) but import THIS file so
the joint ordering and wire format can never drift apart. Keep it dependency
light: stdlib + numpy + cv2 only (both environments have all three).

The ordering below is copied verbatim from the training dataset's meta/info.json
(robot_type "hsr"), so observations and actions line up with what pi05 expects.
"""
import base64

import numpy as np

try:
    import cv2  # present on the host venv; may be absent in a minimal ROS container
    HAVE_CV2 = True
except Exception:  # noqa: BLE001
    HAVE_CV2 = False

# --- observation.state, shape [8] (absolute joint positions, radians/metres) ---
STATE_JOINTS = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "hand_motor_joint",
    "head_pan_joint",
    "head_tilt_joint",
]

# --- action, shape [11] ---
# indices 0..7 are ABSOLUTE joint targets (same order/units as the state above),
# indices 8..10 are a per-step DELTA of the omni base in the base frame.
ACTION_NAMES = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "hand_motor_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "base_x",   # delta [m]
    "base_y",   # delta [m]
    "base_t",   # delta [rad]
]

ARM_JOINTS = ACTION_NAMES[0:5]
GRIPPER_JOINT = ACTION_NAMES[5]
HEAD_JOINTS = ACTION_NAMES[6:8]
BASE_SLICE = slice(8, 11)

# Image keys the model consumes, mapped to the wire field names below.
HEAD_IMAGE_KEY = "observation.image.head"
HAND_IMAGE_KEY = "observation.image.hand"

# Resolution the training frames were stored at (H, W). Both cameras are resized
# to this on the client so preprocessing matches the dataset exactly.
IMAGE_HW = (480, 640)

FPS = 30
CHUNK_SIZE = 50

# HTTP endpoints
EP_HEALTH = "/health"
EP_RESET = "/reset"
EP_PREDICT = "/predict_chunk"


def encode_image(img_rgb: np.ndarray, quality: int = 95) -> dict:
    """RGB uint8 HxWx3 -> a small JSON-safe envelope.

    Uses JPEG when cv2 is available (≈40 KB/frame), otherwise falls back to raw
    base64 bytes so the bridge works even in a ROS container without OpenCV.
    """
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    h, w = img_rgb.shape[:2]
    if HAVE_CV2:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return {"fmt": "jpg", "b64": base64.b64encode(buf.tobytes()).decode("ascii")}
    return {"fmt": "raw", "h": int(h), "w": int(w),
            "b64": base64.b64encode(img_rgb.tobytes()).decode("ascii")}


def decode_image(env: dict) -> np.ndarray:
    """Inverse of encode_image -> RGB uint8 HxWx3."""
    raw = base64.b64decode(env["b64"])
    if env.get("fmt") == "raw":
        return np.frombuffer(raw, dtype=np.uint8).reshape(env["h"], env["w"], 3).copy()
    # JPEG path always decodes on the server side, which has cv2.
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
