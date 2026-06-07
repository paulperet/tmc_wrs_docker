#!/usr/bin/env python
"""pi05-HSR ROS bridge (runs INSIDE the simulator's `workspace` container).

It closes the loop between the HSR and the pi05 policy server on the host:

    cameras + joint_states  --->  [this node]  --(HTTP)-->  policy_server.py
                                       |                          |
                                       |  <----- action chunk ----+
                                       v
            arm / head / gripper trajectory commands  +  base velocity

Run it in the container's IDE terminal (http://localhost:3001), after starting
the simulation in Gazebo and the policy server on the host:

    python /workspace/pi05_hsr/hsr_bridge_node.py --task "pick up the can"

Everything below the CONFIG block is generic; topic/controller names and safety
limits are env-overridable so you can adapt without editing code. Use --dry-run
first to watch the actions it *would* send.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time

try:
    import urllib.request as urlreq
except ImportError:  # pragma: no cover
    import urllib2 as urlreq  # type: ignore

import json

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    import cv2  # only used to resize cameras to the training resolution, if available
    HAVE_CV2 = True
except Exception:  # noqa: BLE001
    HAVE_CV2 = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P  # noqa: E402

# --------------------------------------------------------------------------- #
# CONFIG  (override any of these with environment variables)
# --------------------------------------------------------------------------- #
def _env(name, default):
    return os.environ.get(name, default)

def _envf(name, default):
    return float(os.environ.get(name, default))

def _envb(name, default):
    return os.environ.get(name, "1" if default else "0") == "1"

SERVER_URL = _env("PI05_SERVER_URL", "http://host.docker.internal:8000")
HTTP_TIMEOUT = _envf("PI05_HTTP_TIMEOUT", 30.0)

# Sensor topics (inputs)
HEAD_IMG_TOPIC = _env("PI05_HEAD_TOPIC", "/hsrb/head_rgbd_sensor/rgb/image_rect_color")
HAND_IMG_TOPIC = _env("PI05_HAND_TOPIC", "/hsrb/hand_camera/image_raw")
JOINT_STATES_TOPIC = _env("PI05_JOINTS_TOPIC", "/hsrb/joint_states")

# Controller command topics (outputs)
ARM_CMD_TOPIC = _env("PI05_ARM_TOPIC", "/hsrb/arm_trajectory_controller/command")
HEAD_CMD_TOPIC = _env("PI05_HEAD_TOPIC_CMD", "/hsrb/head_trajectory_controller/command")
GRIPPER_CMD_TOPIC = _env("PI05_GRIPPER_TOPIC", "/hsrb/gripper_controller/command")
BASE_VEL_TOPIC = _env("PI05_BASE_TOPIC", "/hsrb/command_velocity")

# Per-group enable flags (handy for bringing the robot up piece by piece)
ENABLE_ARM = _envb("PI05_ENABLE_ARM", True)
ENABLE_HEAD = _envb("PI05_ENABLE_HEAD", True)
ENABLE_GRIPPER = _envb("PI05_ENABLE_GRIPPER", True)
ENABLE_BASE = _envb("PI05_ENABLE_BASE", True)

# Safety limits
BASE_VX_MAX = _envf("PI05_BASE_VX_MAX", 0.2)   # m/s
BASE_VY_MAX = _envf("PI05_BASE_VY_MAX", 0.2)   # m/s
BASE_WZ_MAX = _envf("PI05_BASE_WZ_MAX", 0.5)   # rad/s
MAX_JOINT_STEP = _envf("PI05_MAX_JOINT_STEP", 0.15)  # rad/m per control tick; 0 disables

# Timing / horizon
FPS = int(_envf("PI05_FPS", P.FPS))
TRAJ_TIME = _envf("PI05_TRAJ_TIME", 0.3)       # time_from_start for streamed points [s]
# Request the next chunk while ~25 actions remain. Inference is ~0.7s (≈21 steps
# @30Hz) on MPS, so the fresh chunk lands before the queue drains -> smooth 30Hz.
PREFETCH_AT = int(_envf("PI05_PREFETCH_AT", 25))

ARM_CMD_JOINTS = list(P.ARM_JOINTS)
HEAD_CMD_JOINTS = list(P.HEAD_JOINTS)
GRIPPER_CMD_JOINTS = [P.GRIPPER_JOINT]


class Pi05HsrBridge:
    def __init__(self, task: str, dry_run: bool):
        self.task = task
        self.dry_run = dry_run

        self._lock = threading.Lock()
        self._head = None          # latest RGB np.uint8 HxWx3
        self._hand = None
        self._joint_pos = {}       # name -> position
        self._queue = collections.deque()
        self._fetching = False
        self._last_cmd = None      # last 11-vector actually commanded (for hold + rate limit)
        self._n_infer = 0
        self._last_infer_ms = 0.0
        self._next_fetch_ok = 0.0  # backoff gate so failed fetches don't spin at 30 Hz
        self._stop = False

        # Publishers
        self.pub_arm = rospy.Publisher(ARM_CMD_TOPIC, JointTrajectory, queue_size=1)
        self.pub_head = rospy.Publisher(HEAD_CMD_TOPIC, JointTrajectory, queue_size=1)
        self.pub_grip = rospy.Publisher(GRIPPER_CMD_TOPIC, JointTrajectory, queue_size=1)
        self.pub_base = rospy.Publisher(BASE_VEL_TOPIC, Twist, queue_size=1)

        # Subscribers
        rospy.Subscriber(HEAD_IMG_TOPIC, Image, self._head_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(HAND_IMG_TOPIC, Image, self._hand_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(JOINT_STATES_TOPIC, JointState, self._joints_cb, queue_size=1)

    # --- subscriber callbacks ------------------------------------------------
    def _to_rgb(self, msg):
        """sensor_msgs/Image -> RGB uint8 HxWx3 (no cv_bridge dependency)."""
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        enc = (msg.encoding or "rgb8").lower()
        if enc == "rgb8":
            rgb = arr[..., :3]
        elif enc == "bgr8":
            rgb = arr[..., 2::-1]
        elif enc == "rgba8":
            rgb = arr[..., :3]
        elif enc == "bgra8":
            rgb = arr[..., [2, 1, 0]]
        elif enc in ("mono8", "8uc1"):
            rgb = np.repeat(arr[..., :1], 3, axis=2)
        else:
            rgb = arr[..., :3]  # best effort
        if HAVE_CV2 and rgb.shape[:2] != P.IMAGE_HW:
            rgb = cv2.resize(rgb, (P.IMAGE_HW[1], P.IMAGE_HW[0]), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _head_cb(self, msg):
        with self._lock:
            self._head = self._to_rgb(msg)

    def _hand_cb(self, msg):
        with self._lock:
            self._hand = self._to_rgb(msg)

    def _joints_cb(self, msg):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_pos[name] = pos

    # --- observation / inference --------------------------------------------
    def _snapshot_obs(self):
        with self._lock:
            if self._head is None or self._hand is None:
                return None
            missing = [j for j in P.STATE_JOINTS if j not in self._joint_pos]
            if missing:
                return None
            head = self._head.copy()
            hand = self._hand.copy()
            state = [float(self._joint_pos[j]) for j in P.STATE_JOINTS]
        return head, hand, state

    def _fetch_worker(self):
        ok = False
        try:
            snap = self._snapshot_obs()
            if snap is None:
                return
            head, hand, state = snap
            payload = json.dumps({
                "head": P.encode_image(head),
                "hand": P.encode_image(hand),
                "state": state,
                "task": self.task,
            }).encode("utf-8")
            req = urlreq.Request(
                SERVER_URL + P.EP_PREDICT, data=payload,
                headers={"Content-Type": "application/json"},
            )
            t0 = time.time()
            with urlreq.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            chunk = out["chunk"]
            with self._lock:
                self._queue.clear()
                self._queue.extend(np.asarray(a, dtype=np.float32) for a in chunk)
                self._n_infer += 1
                self._last_infer_ms = out.get("infer_ms", (time.time() - t0) * 1000.0)
            ok = True
        except Exception as e:  # noqa: BLE001
            rospy.logwarn_throttle(2.0, "policy fetch failed: %s" % e)
        finally:
            with self._lock:
                self._fetching = False
                if not ok:
                    self._next_fetch_ok = time.time() + 1.0  # back off on failure

    def _maybe_fetch(self):
        with self._lock:
            if self._fetching or len(self._queue) > PREFETCH_AT:
                return
            if self._head is None or self._hand is None or time.time() < self._next_fetch_ok:
                return
            self._fetching = True
        threading.Thread(target=self._fetch_worker, daemon=True).start()

    # --- action execution ----------------------------------------------------
    def _traj_msg(self, joint_names, positions):
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.joint_names = list(joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.time_from_start = rospy.Duration(TRAJ_TIME)
        jt.points = [pt]
        return jt

    def _rate_limit(self, target):
        """Clamp absolute joint targets (indices 0..7) to MAX_JOINT_STEP from the last command."""
        target = np.asarray(target, dtype=np.float32).copy()
        if MAX_JOINT_STEP > 0 and self._last_cmd is not None:
            lo = self._last_cmd[:8] - MAX_JOINT_STEP
            hi = self._last_cmd[:8] + MAX_JOINT_STEP
            target[:8] = np.clip(target[:8], lo, hi)
        return target

    def _execute(self, action):
        action = self._rate_limit(action)
        if ENABLE_ARM and not self.dry_run:
            self.pub_arm.publish(self._traj_msg(ARM_CMD_JOINTS, action[0:5]))
        if ENABLE_GRIPPER and not self.dry_run:
            self.pub_grip.publish(self._traj_msg(GRIPPER_CMD_JOINTS, action[5:6]))
        if ENABLE_HEAD and not self.dry_run:
            self.pub_head.publish(self._traj_msg(HEAD_CMD_JOINTS, action[6:8]))
        # base: per-step delta -> instantaneous velocity command
        vx = float(np.clip(action[8] * FPS, -BASE_VX_MAX, BASE_VX_MAX))
        vy = float(np.clip(action[9] * FPS, -BASE_VY_MAX, BASE_VY_MAX))
        wz = float(np.clip(action[10] * FPS, -BASE_WZ_MAX, BASE_WZ_MAX))
        if ENABLE_BASE and not self.dry_run:
            tw = Twist()
            tw.linear.x, tw.linear.y, tw.angular.z = vx, vy, wz
            self.pub_base.publish(tw)
        self._last_cmd = action
        if self.dry_run:
            rospy.loginfo_throttle(
                0.5, "[dry-run] arm=%s grip=%.3f head=%s base(v=%.2f,%.2f w=%.2f)" % (
                    np.round(action[0:5], 3), action[5], np.round(action[6:8], 3), vx, vy, wz))

    def _hold(self):
        """No fresh action available: stop the base and re-assert the last posture."""
        if ENABLE_BASE and not self.dry_run:
            self.pub_base.publish(Twist())  # zero velocity
        if self._last_cmd is not None and not self.dry_run:
            a = self._last_cmd
            if ENABLE_ARM:
                self.pub_arm.publish(self._traj_msg(ARM_CMD_JOINTS, a[0:5]))
            if ENABLE_GRIPPER:
                self.pub_grip.publish(self._traj_msg(GRIPPER_CMD_JOINTS, a[5:6]))
            if ENABLE_HEAD:
                self.pub_head.publish(self._traj_msg(HEAD_CMD_JOINTS, a[6:8]))

    # --- main loop -----------------------------------------------------------
    def spin(self):
        self._wait_for_inputs()
        try:
            urlreq.urlopen(urlreq.Request(SERVER_URL + P.EP_RESET, data=b"{}",
                           headers={"Content-Type": "application/json"}), timeout=5.0)
        except Exception as e:  # noqa: BLE001
            rospy.logwarn("could not reach policy server at %s (%s)" % (SERVER_URL, e))

        rate = rospy.Rate(FPS)
        rospy.loginfo("pi05 bridge running @ %d Hz -> %s  task=%r  dry_run=%s  (Ctrl-C to stop)"
                      % (FPS, SERVER_URL, self.task, self.dry_run))
        try:
            while not self._stop and not rospy.is_shutdown():
                self._maybe_fetch()
                with self._lock:
                    action = self._queue.popleft() if self._queue else None
                    qlen = len(self._queue)
                if action is not None:
                    self._execute(action)
                else:
                    self._hold()
                    rospy.loginfo_throttle(1.0, "waiting for action chunk (infer #%d, %.0f ms) ..."
                                           % (self._n_infer, self._last_infer_ms))
                rospy.loginfo_throttle(5.0, "queue=%d infers=%d last_infer=%.0fms"
                                       % (qlen, self._n_infer, self._last_infer_ms))
                rate.sleep()
        except (KeyboardInterrupt, rospy.ROSInterruptException):
            pass
        finally:
            self.stop()

    def stop(self):
        """Zero the base velocity and mark shutdown (idempotent)."""
        if self._stop:
            return
        self._stop = True
        rospy.loginfo("stopping: zeroing base velocity")
        if not self.dry_run:
            for _ in range(5):
                try:
                    self.pub_base.publish(Twist())
                except Exception:  # noqa: BLE001
                    break
                time.sleep(0.03)

    def _wait_for_inputs(self):
        published = dict(rospy.get_published_topics())
        for label, topic in [("head image", HEAD_IMG_TOPIC), ("hand image", HAND_IMG_TOPIC),
                             ("joint states", JOINT_STATES_TOPIC)]:
            if topic not in published:
                rospy.logwarn("expected %s topic '%s' is not advertised yet. "
                              "Check `rostopic list` and set the matching PI05_*_TOPIC env var."
                              % (label, topic))
        rospy.loginfo("waiting for first head/hand/joint_states messages ...")
        t0 = time.time()
        while not rospy.is_shutdown():
            if self._snapshot_obs() is not None:
                rospy.loginfo("observations ready (waited %.1fs)" % (time.time() - t0))
                return
            if time.time() - t0 > 20.0:
                with self._lock:
                    miss = [j for j in P.STATE_JOINTS if j not in self._joint_pos]
                    have_imgs = (self._head is not None, self._hand is not None)
                rospy.logwarn_throttle(
                    5.0, "still waiting: head=%s hand=%s missing_joints=%s"
                    % (have_imgs[0], have_imgs[1], miss))
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser(description="pi05-HSR ROS bridge")
    ap.add_argument("--task", default=os.environ.get("PI05_TASK", ""),
                    help="language instruction for the policy (defaults to server's task)")
    ap.add_argument("--server", default=None, help="override policy server URL")
    ap.add_argument("--dry-run", action="store_true", help="log actions without commanding the robot")
    args = ap.parse_args()
    if args.server:
        global SERVER_URL
        SERVER_URL = args.server

    # disable_signals=True so Ctrl-C raises KeyboardInterrupt in our control loop
    # for a prompt, clean shutdown (rospy's default handler can hang a custom loop,
    # especially under sim time).
    rospy.init_node("pi05_hsr_bridge", disable_signals=True)
    bridge = Pi05HsrBridge(task=args.task, dry_run=args.dry_run)
    try:
        bridge.spin()
    except KeyboardInterrupt:
        bridge.stop()
    finally:
        # Guarantee the process exits even if ROS/daemon threads linger.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
