#!/usr/bin/env python
"""Smoke test: load the pi05-HSR checkpoint and run one inference on this host.

Run with the lerobot venv, e.g.:

    ../lerobot/.venv/bin/python pi05_hsr/smoke_test.py

It validates (before we wire up ROS):
  1. PI05Policy.from_pretrained loads on MPS (or CUDA/CPU).
  2. predict_action_chunk produces a [1, 50, 11] chunk and how long that takes.
  3. Unnormalizing the whole chunk in one reshaped call == the canonical
     per-step select_action + postprocessor path (so the server may return a
     full chunk and let the ROS client pop it one action at a time).
"""
import os
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference
from lerobot.policies.pi05 import PI05Policy

REPO = os.environ.get("PI05_REPO", "paulprt/pi05-hsr-5k")
TASK = os.environ.get("PI05_TASK", "pick up the object")


def pick_device() -> str:
    if os.environ.get("PI05_DEVICE"):
        return os.environ["PI05_DEVICE"]
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dummy_obs() -> dict:
    return {
        "observation.image.head": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "observation.image.hand": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
    }


_FIXED_OBS = dummy_obs()  # reuse one observation so the two paths are comparable


def build_batch(pre, device):
    obs = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in _FIXED_OBS.items()}
    obs = prepare_observation_for_inference(obs, torch.device(device), TASK, "hsr")
    return pre(obs)


def main():
    device = pick_device()
    print(f"[smoke] device={device} repo={REPO}")

    cfg = PreTrainedConfig.from_pretrained(REPO)
    cfg.device = device
    cfg.compile_model = False  # max-autotune torch.compile needs CUDA; run eager on MPS/CPU
    t0 = time.time()
    policy = PI05Policy.from_pretrained(REPO, config=cfg)
    policy.to(device)
    policy.eval()
    print(f"[smoke] policy loaded in {time.time() - t0:.1f}s; chunk_size={policy.config.chunk_size}")

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=REPO,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # --- canonical single-action path (what predict_action() does) ---
    torch.manual_seed(0)
    policy.reset()
    a_single = post(policy.select_action(build_batch(pre, device)))
    a_single = np.asarray(a_single).reshape(-1).astype(np.float32)

    # --- full-chunk path, unnormalized in one reshaped call ---
    torch.manual_seed(0)
    policy.reset()
    t0 = time.time()
    chunk = policy.predict_action_chunk(build_batch(pre, device))  # [1, 50, 11]
    dt = time.time() - t0
    b, t, d = chunk.shape
    chunk_un = post(chunk.reshape(b * t, d)).reshape(b, t, d)
    chunk0 = np.asarray(chunk_un[0, 0].float().cpu()).reshape(-1).astype(np.float32)

    print(f"[smoke] chunk shape={tuple(chunk.shape)} inference={dt * 1000:.0f} ms "
          f"(~{dt / t * 1000:.1f} ms/step amortized over {t} steps)")
    print(f"[smoke] action dim={d} (expect 11)")
    np.set_printoptions(precision=3, suppress=True)
    print(f"[smoke] unnormalized action[0]      = {chunk0}")
    print(f"[smoke]   arm(5)+grip+head(2)        = {chunk0[:8]}")
    print(f"[smoke]   base delta (x,y,theta)     = {chunk0[8:]}")

    max_abs = float(np.max(np.abs(a_single - chunk0)))
    ok = np.allclose(a_single, chunk0, atol=1e-3)
    print(f"[smoke] chunk[0] vs select_action max|Δ|={max_abs:.2e} -> "
          f"{'MATCH ✓ (chunk path valid)' if ok else 'MISMATCH ✗ (use per-step postprocess)'}")

    print("[smoke] DONE")


if __name__ == "__main__":
    main()
