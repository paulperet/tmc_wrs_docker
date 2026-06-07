#!/usr/bin/env bash
# Start the pi05-HSR policy server on the HOST (Apple MPS / CUDA / CPU).
#
#   ./pi05_hsr/run_server.sh
#
# Override with env vars, e.g. PI05_REPO, PI05_TASK, PI05_PORT, PI05_DEVICE,
# PI05_INFERENCE_STEPS, PI05_VENV_PY.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
VENV_PY="${PI05_VENV_PY:-$REPO_ROOT/../lerobot/.venv/bin/python}"

export PI05_REPO="${PI05_REPO:-paulprt/pi05-hsr-10k-128}"
export PI05_TASK="${PI05_TASK:-pick up the object}"
export PI05_PORT="${PI05_PORT:-8000}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
# The checkpoint + paligemma backbone are already in the HF cache; staying offline
# avoids a Hub network check that can hang model loading.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: lerobot venv python not found at: $VENV_PY" >&2
  echo "Set PI05_VENV_PY to your lerobot venv python (the one with torch+lerobot)." >&2
  exit 1
fi

echo "Starting pi05 server  repo=$PI05_REPO  task='$PI05_TASK'  port=$PI05_PORT"
echo "(first start loads a ~3B model + warms up; expect ~1-2 min)"
exec "$VENV_PY" "$HERE/policy_server.py"
