#!/usr/bin/env python
"""pi05-HSR policy server (runs on the HOST, in the lerobot venv).

Loads the fine-tuned pi05 checkpoint and serves action chunks over HTTP so the
ROS bridge inside the Docker container can drive the simulated HSR. The model
runs natively on the host (Apple MPS / CUDA / CPU) because the ROS Noetic
container ships Python 3.8 and cannot host modern PyTorch.

Run (from the repo root):

    PI05_REPO=paulprt/pi05-hsr-5k \
    ../lerobot/.venv/bin/python pi05_hsr/policy_server.py

Endpoints:
    GET  /health          -> {status, device, repo, ...}
    POST /reset           -> clears the policy's internal queue/state
    POST /predict_chunk   -> {head,hand: b64 JPEG, state:[8], task} -> {chunk:[50][11]}

The returned chunk is fully UN-normalized (raw robot units): indices 0..7 are
absolute joint targets, 8..10 are base deltas. See protocol.py for the layout.
"""
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
import protocol as P  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference  # noqa: E402
from lerobot.policies.pi05 import PI05Policy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
log = logging.getLogger("pi05-server")

REPO = os.environ.get("PI05_REPO", "paulprt/pi05-hsr-5k")
DEFAULT_TASK = os.environ.get("PI05_TASK", "pick up the object")
HOST = os.environ.get("PI05_HOST", "0.0.0.0")
PORT = int(os.environ.get("PI05_PORT", "8000"))
WARMUP = os.environ.get("PI05_WARMUP", "1") == "1"


def pick_device() -> str:
    if os.environ.get("PI05_DEVICE"):
        return os.environ["PI05_DEVICE"]
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
log.info(f"loading {REPO} on device={DEVICE} ...")
_t0 = time.time()
_cfg = PreTrainedConfig.from_pretrained(REPO)
_cfg.device = DEVICE
_cfg.compile_model = False  # max-autotune torch.compile needs CUDA; run eager on MPS/CPU
policy = PI05Policy.from_pretrained(REPO, config=_cfg)
policy.to(DEVICE)
policy.eval()
# Fewer flow-matching denoising steps => faster inference (useful on CPU), slight
# quality trade-off. Default keeps the checkpoint's trained value.
if os.environ.get("PI05_INFERENCE_STEPS"):
    policy.config.num_inference_steps = int(os.environ["PI05_INFERENCE_STEPS"])
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=REPO,
    preprocessor_overrides={"device_processor": {"device": DEVICE}},
)
N_STEPS = int(policy.config.n_action_steps)
log.info(f"loaded in {time.time() - _t0:.1f}s; chunk={policy.config.chunk_size} n_action_steps={N_STEPS}")

_lock = threading.Lock()


def _infer_chunk(head_rgb: np.ndarray, hand_rgb: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
    """Run the policy once and return an un-normalized [N_STEPS, 11] action chunk."""
    obs = {
        P.HEAD_IMAGE_KEY: head_rgb,
        P.HAND_IMAGE_KEY: hand_rgb,
        "observation.state": state.astype(np.float32),
    }
    with torch.inference_mode():
        batch = prepare_observation_for_inference(obs, torch.device(DEVICE), task, "hsr")
        batch = preprocessor(batch)
        chunk = policy.predict_action_chunk(batch)[:, :N_STEPS]  # [1, N, 11] (normalized)
        # Un-normalize per timestep through the postprocessor (the exact path
        # predict_action() uses for a single action, applied to each step).
        steps = [
            np.asarray(postprocessor(chunk[:, t]).float().cpu()).reshape(-1)
            for t in range(chunk.shape[1])
        ]
    return np.stack(steps, axis=0)  # [N, 11]


if WARMUP:
    try:
        log.info("warming up (first MPS/CUDA graph build is slow)...")
        _wt = time.time()
        _dummy = (np.random.rand(*P.IMAGE_HW, 3) * 255).astype(np.uint8)
        _infer_chunk(_dummy, _dummy.copy(), np.zeros(8, np.float32), DEFAULT_TASK)
        policy.reset()
        log.info(f"warmup done in {time.time() - _wt:.1f}s")
    except Exception as e:  # noqa: BLE001
        log.warning(f"warmup failed (continuing): {e}")

# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #
app = FastAPI(title="pi05-HSR policy server")


class PredictRequest(BaseModel):
    head: dict           # image envelope (see protocol.encode_image), RGB
    hand: dict           # image envelope, RGB
    state: list[float]   # length 8, STATE_JOINTS order
    task: str | None = None


class PredictResponse(BaseModel):
    chunk: list[list[float]]   # [N_STEPS][11], un-normalized
    infer_ms: float


@app.get(P.EP_HEALTH)
def health():
    return {
        "status": "ok",
        "repo": REPO,
        "device": DEVICE,
        "default_task": DEFAULT_TASK,
        "n_action_steps": N_STEPS,
        "num_inference_steps": int(policy.config.num_inference_steps),
        "action_names": P.ACTION_NAMES,
        "state_joints": P.STATE_JOINTS,
    }


@app.post(P.EP_RESET)
def reset():
    with _lock:
        policy.reset()
    return {"status": "reset"}


@app.post(P.EP_PREDICT, response_model=PredictResponse)
def predict_chunk(req: PredictRequest):
    head = P.decode_image(req.head)
    hand = P.decode_image(req.hand)
    state = np.asarray(req.state, dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"state must be length 8, got {state.shape}")
    task = req.task or DEFAULT_TASK
    t0 = time.time()
    with _lock:
        chunk = _infer_chunk(head, hand, state, task)
    return PredictResponse(chunk=chunk.tolist(), infer_ms=(time.time() - t0) * 1000.0)


def main():
    log.info(f"serving on http://{HOST}:{PORT}  (task default: {DEFAULT_TASK!r})")
    uvicorn.run(app, host=HOST, port=PORT, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
