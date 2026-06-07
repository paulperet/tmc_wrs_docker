#!/usr/bin/env python
"""Sanity-check a running pi05-HSR policy server without ROS or the simulator.

Posts a synthetic observation (random images + zero state) and prints the
returned action chunk. Use it after starting policy_server.py to confirm the
host side works before launching the in-container bridge.

    ../lerobot/.venv/bin/python pi05_hsr/check_server.py [--url http://localhost:8000]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import protocol as P  # noqa: E402


def _post(url, obj, timeout=60):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("PI05_SERVER_URL", "http://localhost:8000"))
    ap.add_argument("--task", default=os.environ.get("PI05_TASK", ""))
    args = ap.parse_args()

    print(f"[check] GET {args.url}{P.EP_HEALTH}")
    health = _get(args.url + P.EP_HEALTH)
    print("[check] health:", json.dumps(health))
    assert health.get("status") == "ok"

    head = (np.random.rand(*P.IMAGE_HW, 3) * 255).astype(np.uint8)
    hand = (np.random.rand(*P.IMAGE_HW, 3) * 255).astype(np.uint8)
    payload = {
        "head": P.encode_image(head),
        "hand": P.encode_image(hand),
        "state": [0.0] * len(P.STATE_JOINTS),
        "task": args.task or None,
    }
    t0 = time.time()
    out = _post(args.url + P.EP_PREDICT, payload)
    rtt = (time.time() - t0) * 1000.0

    chunk = np.asarray(out["chunk"], dtype=np.float32)
    print(f"[check] /predict_chunk -> chunk shape={chunk.shape} "
          f"server_infer={out['infer_ms']:.0f}ms round_trip={rtt:.0f}ms")
    assert chunk.shape == (health["n_action_steps"], len(P.ACTION_NAMES)), chunk.shape
    np.set_printoptions(precision=3, suppress=True)
    print(f"[check] action[0] = {chunk[0]}")
    print(f"[check]   arm+grip+head = {chunk[0][:8]}")
    print(f"[check]   base delta     = {chunk[0][8:]}")
    print(f"[check] /reset ->", _post(args.url + P.EP_RESET, {}))
    print("[check] SERVER OK ✓")


if __name__ == "__main__":
    main()
