# Driving the simulated HSR with a π0.5 (pi05) VLA

Run your LeRobot-trained **pi05** checkpoint (`paulprt/pi05-hsr-5k`) as a closed-loop
controller for the Toyota HSR inside the `tmc_wrs_docker` Gazebo simulator.

The model runs **on the host** (Apple MPS / CUDA / CPU) because the sim's ROS Noetic
containers ship old Python; a thin ROS node inside a container streams observations to
it and executes the returned actions:

```
   ┌──────────────── HOST (macOS, Apple MPS) ─────────────────┐
   │  policy_server.py  (../lerobot/.venv)                     │
   │    loads paulprt/pi05-hsr-5k                              │
   │    POST /predict_chunk : obs -> action chunk [50 x 11]    │
   └───────────────▲──────────────────────────┬───────────────┘
                   │ HTTP host.docker.internal:8000
   ┌───────────────┴──────────────────────────▼───────────────┐
   │  hsr_bridge_node.py  (workspace container, ROS Noetic)    │
   │    head + hand cameras + /hsrb/joint_states  ->  obs      │
   │    chunk -> arm/head/gripper traj cmds + base velocity    │
   └───────────────────────────┬──────────────────────────────┘
                               ROS │  (master = simulator:11311)
                        ┌──────────▼──────────┐
                        │    HSR in Gazebo    │
                        └─────────────────────┘
```

**Contract** (from the checkpoint, encoded in [`protocol.py`](protocol.py)):

| input | shape | content |
|---|---|---|
| `observation.image.head` | 480×640×3 RGB | head RGB-D sensor color image |
| `observation.image.hand` | 480×640×3 RGB | hand/gripper camera |
| `observation.state` | `[8]` | `arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, hand_motor, head_pan, head_tilt` |
| **action** | `[11]` | those 8 joints (**absolute** targets) + `base_x, base_y, base_t` (**per-step delta**) @ 30 Hz |

---

## 1. Install / one-time setup

**Prerequisites**
- **Docker Desktop** (macOS), and the simulator images pulled (`./pull-images.sh`, see the repo root README).
- A **LeRobot venv on the host** with `torch` + `lerobot` — this package assumes `../lerobot/.venv`
  (override with `PI05_VENV_PY`). Verify: `../lerobot/.venv/bin/python -c "import torch, lerobot"`.
- The checkpoint **`paulprt/pi05-hsr-5k`** in your HF cache (already present if you trained/downloaded it).

**macOS workspace fix (already included).** The stock `workspace` container crash-loops on macOS
(its entrypoint runs `chown -R /workspace`, which fails on bind-mounted `.git` files under `bash -e`).
[`../docker-compose.override.yml`](../docker-compose.override.yml) bind-mounts a patched entrypoint
([`workspace-entrypoint.sh`](workspace-entrypoint.sh)) that skips the chown. It is auto-merged by
`docker-compose` — nothing to do but keep both files in place.

**Sanity-check the model loads** (optional, ~1–2 min):
```sh
../lerobot/.venv/bin/python pi05_hsr/smoke_test.py     # expect: chunk shape=(1,50,11) ... MATCH ✓
```

---

## 2. Run

Three terminals. (1) sim, (2) policy server on the host, (3) the bridge inside a container.

### Terminal 1 — simulator
```sh
docker-compose up          # CPU;  or: docker-compose -f docker-compose.nvidia.yml up
```
Then:
- Open **http://localhost:3000** (Gazebo) and press **▶ Play** — *cameras and joints only publish while playing.*
- This also starts the dev container: **IDE http://localhost:3001**, **Jupyter http://localhost:3002**.

### Terminal 2 — policy server (host)
```sh
./pi05_hsr/run_server.sh
# in another shell, confirm it works without ROS:
../lerobot/.venv/bin/python pi05_hsr/check_server.py        # expect: SERVER OK ✓
```
First start loads the ~3B model + warms up (~1–2 min). It runs offline from your HF cache.

### Terminal 3 — the bridge (in the `workspace` container)
Open a terminal in the **IDE at http://localhost:3001** (ROS + `ROS_MASTER_URI` are already set there),
or `docker exec -it tmc_wrs_docker-workspace-1 bash && source /opt/ros/noetic/setup.bash`. Then:
```sh
# 1) dry-run first — prints the actions it WOULD send, drives nothing:
python3 /workspace/pi05_hsr/hsr_bridge_node.py --task "pick up the can" --dry-run

# 2) when the dry-run looks sane, drive the robot for real:
python3 /workspace/pi05_hsr/hsr_bridge_node.py --task "pick up the can"
```
`Ctrl-C` stops the node and zeroes the base. `/workspace` is the live repo mount, so edits on the
host appear here immediately — no copying.

> **Safety:** start with `--dry-run`; keep clamps conservative until you trust the policy on your scene
> (e.g. `PI05_MAX_JOINT_STEP=0.05 PI05_BASE_VX_MAX=0.1 python3 …`); bring groups up one at a time with
> `PI05_ENABLE_BASE=0` / `PI05_ENABLE_ARM=0` etc.

<details>
<summary>Alternative: run the bridge in the <b>simulator</b> container (no /workspace mount)</summary>

```sh
docker cp pi05_hsr 37b4fcf33ebb:/root/pi05_hsr      # use your simulator container id
docker exec -it 37b4fcf33ebb bash -lc '
  source /opt/ros/noetic/setup.bash; export ROS_MASTER_URI=http://localhost:11311
  python3 /root/pi05_hsr/hsr_bridge_node.py --task "pick up the can" --dry-run'
```
This is a *copy*, so re-`docker cp` after edits. Prefer the workspace container above.
</details>

---

## 3. Connect to / view the cameras

The bridge **auto-connects two cameras** to the policy — exactly the two it was trained on:

| model input | ROS topic | what it is |
|---|---|---|
| `observation.image.head` | `/hsrb/head_rgbd_sensor/rgb/image_rect_color` | head RGB (≈30 Hz) |
| `observation.image.hand` | `/hsrb/hand_camera/image_raw` | hand/gripper cam (≈7 Hz) |

Other cameras exist but are **not** fed to the model: depth
(`/hsrb/head_rgbd_sensor/depth_registered/image`), head center
(`/hsrb/head_center_camera/image_raw`), stereo L/R
(`/hsrb/head_l_stereo_camera/image_rect_color`, `…r_stereo…`).

To **see** any camera (do this in the `workspace` container — it has the GUI tools; the sim container
does not):

**A. Inline in your Mac browser — Jupyter** (`http://localhost:3002`): open
`notebooks/5_recognition.ipynb` and run
```python
import rospy; from utils import *; import matplotlib.pyplot as plt
rospy.init_node("view"); rgbd = RGBD()
plt.imshow(rgbd.get_image())          # head RGB camera (also has depth/laser examples)
```

**B. Live, switch between cameras — `rqt_image_view`** (in the IDE terminal at `http://localhost:3001`):
```sh
rqt_image_view
```
The window opens **inside the Gazebo/noVNC screen at http://localhost:3000** (it renders on display `:0`);
use its dropdown to pick a topic. Point it at the head + hand topics above to see precisely the policy's input.

**C. Cameras + 3D together — `rviz`:** run `rviz`, then *Add → Image*, set the topic. Also shows in `localhost:3000`.

List what's actually publishing:
```sh
rostopic list | grep -E "image_(raw|rect_color)$"
rostopic hz /hsrb/head_rgbd_sensor/rgb/image_rect_color      # confirm it's flowing (sim must be playing)
```

---

## Configuration (environment variables)

**Server** (host):

| var | default | meaning |
|---|---|---|
| `PI05_REPO` | `paulprt/pi05-hsr-5k` | HF checkpoint |
| `PI05_TASK` | `pick up the object` | default language instruction |
| `PI05_DEVICE` | auto `mps`>`cuda`>`cpu` | inference device |
| `PI05_PORT` | `8000` | HTTP port |
| `PI05_INFERENCE_STEPS` | 10 (checkpoint) | fewer = faster, lower quality (good on CPU) |
| `PI05_VENV_PY` | `../lerobot/.venv/bin/python` | python used by `run_server.sh` |

**Bridge** (container):

| var | default | meaning |
|---|---|---|
| `PI05_SERVER_URL` | `http://host.docker.internal:8000` | where the server is |
| `PI05_TASK` | "" → server default | instruction (or use `--task`) |
| `PI05_HEAD_TOPIC` | `/hsrb/head_rgbd_sensor/rgb/image_rect_color` | head camera |
| `PI05_HAND_TOPIC` | `/hsrb/hand_camera/image_raw` | hand camera |
| `PI05_JOINTS_TOPIC` | `/hsrb/joint_states` | joint state |
| `PI05_ARM_TOPIC` / `PI05_HEAD_TOPIC_CMD` / `PI05_GRIPPER_TOPIC` | `/hsrb/*_trajectory_controller/command` | controller commands |
| `PI05_BASE_TOPIC` | `/hsrb/command_velocity` | base velocity |
| `PI05_ENABLE_ARM/HEAD/GRIPPER/BASE` | `1` | per-group on/off |
| `PI05_BASE_VX_MAX/VY_MAX/WZ_MAX` | `0.2/0.2/0.5` | base speed clamps (m/s, m/s, rad/s) |
| `PI05_MAX_JOINT_STEP` | `0.15` | max joint move per tick (`0` disables) |
| `PI05_FPS` | `30` | control rate |
| `PI05_PREFETCH_AT` | `25` | request next chunk when queue ≤ this |

---

## How control works

- The server returns a fully **un-normalized** 50-step action chunk per request.
- The bridge owns a queue, executes one action per 30 Hz tick, and **prefetches** the next chunk in a
  background thread once ~25 actions remain, replacing the stale tail (receding horizon). On MPS a chunk
  infers in ~0.8–1.7 s, which is hidden by prefetch.
- Action indices 0–7 → **absolute** position targets streamed to the arm/gripper/head trajectory
  controllers (`time_from_start=0.3 s`). Indices 8–10 → a **per-step base delta** converted to an
  instantaneous `Twist` (`v = delta × fps`), clamped. When the queue is briefly empty the bridge holds
  (zero base velocity + re-asserts the last posture).

---

## Troubleshooting

- **Bridge can't reach the server.** Server up? (`curl localhost:8000/health`). On *Linux* Docker,
  `host.docker.internal` may not resolve — add `extra_hosts: ["host.docker.internal:host-gateway"]`
  to the `workspace` service, or set `PI05_SERVER_URL=http://<host-ip>:8000`. macOS Desktop works as-is.
- **"waiting for first … messages" forever.** Press **▶ Play** in Gazebo; check `rostopic list` and set
  the matching `PI05_*_TOPIC`. (Bridge `loginfo` goes to stdout — run with `python3 -u` if piping.)
- **Server hangs on load / connection refused.** It loads offline from the HF cache; `run_server.sh`
  sets `HF_HUB_OFFLINE=1` to avoid a Hub network check that can stall loading.
- **`workspace` container keeps exiting.** That's the macOS chown crash — make sure
  `docker-compose.override.yml` + `workspace-entrypoint.sh` are present, then `docker-compose up -d workspace`.
- **Robot jerks / overshoots.** Lower `PI05_MAX_JOINT_STEP`, `PI05_BASE_*_MAX`, or raise `PI05_TRAJ_TIME`.
- **Inference too slow (CPU).** `PI05_INFERENCE_STEPS=5` and/or `PI05_FPS=10`.

---

## Files

| file | runs where | purpose |
|---|---|---|
| [`protocol.py`](protocol.py) | both | shared joint order + wire format (single source of truth) |
| [`policy_server.py`](policy_server.py) | host | loads pi05, serves `/predict_chunk` |
| [`hsr_bridge_node.py`](hsr_bridge_node.py) | container | ROS ↔ policy control loop |
| [`run_server.sh`](run_server.sh) | host | launch the server (MPS, offline) |
| [`check_server.py`](check_server.py) | host | verify the server without ROS |
| [`smoke_test.py`](smoke_test.py) | host | verify the checkpoint loads + infers |
| [`workspace-entrypoint.sh`](workspace-entrypoint.sh) | — | patched entrypoint bind-mounted into the workspace container (macOS fix) |
