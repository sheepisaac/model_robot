#!/usr/bin/env python3
"""OpenCV all-in-one console for the physical UGV, RoArm and L515.

The GUI keeps rendering while a worker thread performs the blocking robot
protocol calls.  Every physical move is a bounded calibration transaction:
capture an L515 reference, move, stop/verify, capture feedback and visual
odometry, update the world model, and append one JSONL record.

Physical movement is disabled unless both ``--execute`` and the same explicit
acknowledgement used by ``live_robot_calibration.py`` are supplied.
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = PROJECT_ROOT / "calibration"
for import_path in (PROJECT_ROOT, CALIBRATION_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from calibration.live_robot_calibration import (  # noqa: E402
    LiveCalibration, MOTION_ACK, load_config, parse_surface_command,
    pose_to_joints)
from robot_models import RobotCameraOdometry  # noqa: E402
from robot_models.transforms import pose2d_transform  # noqa: E402
from visualization.opencv_motion_demo import (  # noqa: E402
    CommandError, MotionDemo, _put_text)


LIVE_WINDOW_NAME = "autoNBV live robot control + visualization"


def parse_live_command(text, default_drive_duration_s=0.6):
    """Parse GUI commands, adding a safe duration to ``go L R``.

    The original UGV controller surface uses ``go L R``.  Live operation must
    never leave a velocity latched indefinitely, so this console converts that
    spelling into a bounded pulse.  ``go L R DURATION`` remains available.
    """
    fields = str(text).strip().split()
    if not fields:
        raise ValueError("empty command")
    keyword = fields[0].lower()
    if keyword in {"probe", "reference", "zero", "reset", "clear", "help",
                   "quit", "exit", "stop"}:
        if len(fields) != 1:
            raise ValueError(f"{keyword} takes no arguments")
        return {"kind": "quit" if keyword == "exit" else keyword,
                "text": text}
    if keyword == "go" and len(fields) == 3:
        duration = float(default_drive_duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("default drive duration must be positive")
        text = f"{text} {duration:.6g}"
    command = parse_surface_command(text)
    numeric = [value for key, value in command.items()
               if key not in {"kind", "text", "relative_delta"}]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("all motion values must be finite")
    return command


def compose_planar_pose(base_pose, relative_transform):
    """Compose a visual-odometry base increment with the current world pose."""
    relative = np.asarray(relative_transform, dtype=float).reshape(4, 4)
    world = pose2d_transform(base_pose).dot(relative)
    return np.array([world[0, 3], world[1, 3],
                     math.atan2(world[1, 0], world[0, 0])], dtype=float)


class LiveRobotConsole(MotionDemo):
    """Hardware-backed variant of the existing OpenCV model visualizer."""

    def __init__(self, session, execute, output_path,
                 default_drive_duration_s=0.6, **view_options):
        self.session = session
        self.execute_enabled = bool(execute)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_drive_duration_s = float(default_drive_duration_s)
        self.command_queue = queue.Queue(maxsize=1)
        self.event_queue = queue.Queue()
        self.stop_worker = threading.Event()
        self.busy = False
        self.last_arm_feedback = None
        self.last_metrics = "Waiting for first measurement."
        self.connection_state = "CONNECTED / READ ONLY"
        self.sequence = 0
        self._animation_kind = None
        self._animation_until = 0.0
        self._motion_base_start = None
        self._motion_joints_start = None
        nominal_scale = float(session.config["ugv"].get(
            "nominal_mps_per_command", 1.0))
        self.nominal_mps_per_command = nominal_scale
        super().__init__(session.model, **view_options)

        # Read all services before accepting GUI input.  This also gives the
        # visualizer a real arm pose instead of the simulation default.
        initial = self.session.probe()
        self._apply_probe(initial, preserve_base=False)
        self.connection_state = ("CONNECTED / MOTION ENABLED" if execute else
                                 "CONNECTED / READ ONLY")
        self.status = ("Live services ready. Type help for commands." if execute
                       else "Read-only live view. Restart with --execute to move.")
        self.worker = threading.Thread(
            target=self._worker_loop, name="live-robot-command", daemon=True)
        self.worker.start()

    def _append_jsonl(self, record):
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _worker_loop(self):
        while not self.stop_worker.is_set():
            try:
                command = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if command is None:
                self.command_queue.task_done()
                break
            self.event_queue.put({"phase": "started", "command": command})
            started_ns = time.time_ns()
            try:
                kind = command["kind"]
                if kind == "ugv":
                    record = self.session.run_ugv(
                        command, require_visual_odometry=False)
                elif kind == "arm":
                    # Four bare numbers intentionally preserve the existing
                    # ctrl_armSimple.py behavior: send one direct absolute
                    # target.  Firmware IK, joint clamps and post-move feedback
                    # verification still apply, but calibration step-size
                    # limits do not.
                    record = self.session.run_arm(
                        command, allow_large_absolute=True,
                        require_visual_odometry=False)
                elif kind == "arm_delta":
                    record = self.session.run_arm(
                        command, require_visual_odometry=False)
                elif kind == "probe":
                    record = {"kind": "probe", "probe": self.session.probe()}
                elif kind == "reference":
                    record = {"kind": "reference",
                              "reference": self.session.vo.reset_reference()}
                elif kind == "stop":
                    self.session._safe_stop()
                    record = {"kind": "stop", "stopped": True}
                else:
                    raise RuntimeError(f"unsupported worker command: {kind}")
                record.update({"timestamp_ns": time.time_ns(),
                               "started_ns": started_ns,
                               "sequence": self.sequence, "success": True})
                self.sequence += 1
                self._append_jsonl(record)
                self.event_queue.put({"phase": "completed", "record": record})
            except Exception as exc:
                try:
                    self.session._safe_stop()
                except Exception:
                    pass
                record = {"timestamp_ns": time.time_ns(),
                          "started_ns": started_ns,
                          "sequence": self.sequence,
                          "success": False, "command": command,
                          "error": str(exc)}
                # A VO-quality rejection can occur after hardware already
                # moved.  Recover fresh firmware state so the display does not
                # keep pretending the arm is at the requested target.
                try:
                    record["recovery_probe"] = self.session.probe()
                except Exception as recovery_exc:
                    record["recovery_error"] = str(recovery_exc)
                self.sequence += 1
                self._append_jsonl(record)
                self.event_queue.put({"phase": "failed", "record": record})
            finally:
                self.command_queue.task_done()

    def _apply_probe(self, probe, preserve_base=True):
        arm = probe["arm"]
        joints = pose_to_joints(self.model, arm)
        base = self.odometry.base_pose.copy() if preserve_base else np.zeros(3)
        if preserve_base:
            self.last_update = self.odometry.update_absolute(
                base_pose=base, joint_angles=joints)
            self._append_trail()
        else:
            self.odometry = RobotCameraOdometry(self.model, base, joints)
            self.base_trail = deque(
                [base[:2].copy()], maxlen=self.trail_length)
            self.camera_trail = deque(
                [self.odometry.camera_pose[:2, 3].copy()],
                maxlen=self.trail_length)
        self.target_joints = joints.copy()
        self.wheel_velocity[:] = 0.0
        self.last_arm_feedback = dict(arm)
        vo = probe.get("visual_odometry", {})
        self.last_metrics = "VO ready={} | reference={} | intrinsics={}".format(
            vo.get("ready"), vo.get("has_reference"),
            (vo.get("intrinsics") or {}).get("status", "UNKNOWN"))

    def _apply_record(self, record):
        kind = record["kind"]
        self.wheel_velocity[:] = 0.0
        self._animation_kind = None
        if kind == "probe":
            self._apply_probe(record["probe"], preserve_base=True)
            self.status = "All three live services responded."
            return
        if kind == "reference":
            reference = record["reference"]
            self.status = "L515 VO reference reset: {} features.".format(
                reference.get("features", "?"))
            return
        if kind == "stop":
            self.status = "UGV stop command completed."
            return

        old_base = self.odometry.base_pose.copy()
        if kind == "ugv":
            comparison_record = record["nominal_vs_visual"]
            transform_key = ("measured" if
                             record.get("visual_odometry_available", True)
                             else "predicted")
            relative = comparison_record[transform_key]["matrix"]
            base_at_start = (self._motion_base_start if
                             self._motion_base_start is not None else old_base)
            new_base = compose_planar_pose(base_at_start, relative)
            arm_feedback = record["arm_after"]
        else:
            new_base = old_base
            arm_feedback = record["firmware_after"]
        joints = pose_to_joints(self.model, arm_feedback)
        if (self._motion_base_start is not None and
                self._motion_joints_start is not None):
            # Discard the animated preview state before creating the measured
            # delta shown in telemetry.  Trails may retain the preview, but
            # last dBase/dCam must describe the real transaction.
            self.odometry = RobotCameraOdometry(
                self.model, self._motion_base_start,
                self._motion_joints_start)
        self.last_update = self.odometry.update_absolute(
            base_pose=new_base, joint_angles=joints)
        self.target_joints = joints.copy()
        self.last_arm_feedback = dict(arm_feedback)
        self._motion_base_start = None
        self._motion_joints_start = None
        self._append_trail()

        vo = record.get("visual_odometry")
        inliers = int(vo.get("inliers", 0)) if vo is not None else 0
        rmse = (float(vo.get("reprojection_rmse_px", math.nan))
                if vo is not None else math.nan)
        if kind == "ugv":
            if record.get("visual_odometry_available", True):
                fit = record["fit_row"]
                self.last_metrics = (
                    "UGV VO: {:+.1f} mm, yaw {:+.2f} deg | inliers={} | "
                    "RMSE={:.2f}px".format(
                        1000.0 * float(fit["distance_m"]),
                        math.degrees(float(fit["yaw_delta_rad"])),
                        inliers, rmse))
                self.status = (
                    "UGV pulse completed and stopped; pose updated from L515.")
            else:
                predicted = record["nominal_vs_visual"]["predicted"]
                self.last_metrics = (
                    "UGV nominal update={:+.1f}mm, {:+.2f}deg | "
                    "VO unavailable: {}".format(
                        1000.0 * float(predicted["distance_m"]),
                        math.degrees(float(predicted["yaw_rad"])),
                        record.get("visual_odometry_error", "unknown error")))
                self.status = (
                    "UGV pulse completed and stopped; VO unavailable, so the "
                    "display used nominal motion.")
        else:
            target = record["command"]
            error_mm = float(np.linalg.norm([
                float(target[key]) - float(arm_feedback[key])
                for key in ("x", "y", "z")]))
            if record.get("visual_odometry_available", True):
                residual = record["model_vs_visual"]["residual"]
                self.last_metrics = (
                    "ARM target error={:.1f}mm | model/VO residual={:.1f}mm, "
                    "{:.2f}deg | inliers={} RMSE={:.2f}px".format(
                        error_mm, 1000.0 * float(residual["distance_m"]),
                        math.degrees(float(residual["rotation_rad"])),
                        inliers, rmse))
                self.status = (
                    "RoArm command verified; model updated from feedback.")
            else:
                vo_error = record.get("visual_odometry_error", "unknown error")
                self.last_metrics = (
                    "ARM target error={:.1f}mm | VO unavailable: {}".format(
                        error_mm, vo_error))
                self.status = (
                    "RoArm command verified from firmware; VO measurement "
                    "was unavailable.")

    def _start_animation(self, command):
        kind = command["kind"]
        self._animation_kind = kind
        if kind in {"ugv", "arm", "arm_delta"}:
            self._motion_base_start = self.odometry.base_pose.copy()
            self._motion_joints_start = self.odometry.joint_angles.copy()
        if kind == "ugv":
            self.wheel_velocity[:] = [
                command["left"] * self.nominal_mps_per_command,
                command["right"] * self.nominal_mps_per_command]
            self._animation_until = time.monotonic() + command["duration"]
        elif kind in {"arm", "arm_delta"}:
            try:
                if kind == "arm_delta" and self.last_arm_feedback is not None:
                    pose = {
                        "x": self.last_arm_feedback["x"] + command["dx"],
                        "y": self.last_arm_feedback["y"] + command["dy"],
                        "z": self.last_arm_feedback["z"] + command["dz"],
                        "t": self.last_arm_feedback["t"] + command["dt"],
                    }
                else:
                    pose = command
                self.target_joints = pose_to_joints(self.model, pose)
            except Exception:
                # The hardware worker performs the authoritative fresh-feedback
                # validation.  Keep the last pose if preview IK cannot be drawn.
                self.target_joints = self.odometry.joint_angles.copy()

    def _drain_events(self):
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            phase = event["phase"]
            if phase == "started":
                command = event["command"]
                self._start_animation(command)
                self.status = "Executing: " + command.get("text", command["kind"])
                self.status_is_error = False
            elif phase == "completed":
                self.busy = False
                self._apply_record(event["record"])
                self.status_is_error = False
            elif phase == "failed":
                self.busy = False
                self.wheel_velocity[:] = 0.0
                self._animation_kind = None
                self._motion_base_start = None
                self._motion_joints_start = None
                record = event["record"]
                if "recovery_probe" in record:
                    self._apply_probe(record["recovery_probe"],
                                      preserve_base=True)
                self.status = "ERROR: " + record["error"]
                self.status_is_error = True

    def update(self, dt):
        self._drain_events()
        if self._animation_kind == "ugv":
            if time.monotonic() >= self._animation_until:
                self.wheel_velocity[:] = 0.0
            super().update(dt)
        elif self._animation_kind in {"arm", "arm_delta"}:
            super().update(dt)

    def _queue_hardware_command(self, command):
        if command["kind"] not in {"probe", "reference"} and not self.execute_enabled:
            raise CommandError(
                "motion disabled; restart with --execute and acknowledgement")
        if self.busy:
            raise CommandError(
                "a bounded command is already running; wait for completion")
        try:
            self.command_queue.put_nowait(command)
        except queue.Full as exc:
            raise CommandError("command worker is busy") from exc
        self.busy = True
        self.status = "Queued: " + command.get("text", command["kind"])
        self.status_is_error = False

    def execute(self, text):
        try:
            command = parse_live_command(text, self.default_drive_duration_s)
        except (ValueError, OverflowError) as exc:
            raise CommandError(str(exc)) from exc
        kind = command["kind"]
        if kind == "quit":
            self.running = False
        elif kind == "help":
            self.status = ("go L R [SEC] | X Y Z T (direct absolute) | "
                           "arm_delta dX dY dZ dT | "
                           "stop | probe | reference | zero | clear | quit")
            self.status_is_error = False
        elif kind == "clear":
            self.base_trail.clear()
            self.camera_trail.clear()
            self._append_trail()
            self.status = "Visualization trails cleared."
            self.status_is_error = False
        elif kind in {"zero", "reset"}:
            if self.busy:
                raise CommandError("cannot reset the display during a live command")
            joints = self.odometry.joint_angles.copy()
            self.odometry = RobotCameraOdometry(
                self.model, np.zeros(3), joints)
            self.target_joints = joints.copy()
            self.base_trail = deque(maxlen=self.trail_length)
            self.camera_trail = deque(maxlen=self.trail_length)
            self._append_trail()
            self.status = "Display world origin reset; hardware was not moved."
            self.status_is_error = False
        else:
            self._queue_hardware_command(command)

    def render(self, size=(1280, 900)):
        image = super().render(size)
        width, height = size

        # Replace the simulation banner while retaining the proven 2D/3D
        # renderer and the gripper/camera geometry.
        cv2.rectangle(image, (0, 0), (width, 52), (25, 29, 34), -1)
        _put_text(image, "autoNBV LIVE UGV + RoArm + L515 console",
                  (22, 34), 0.72, (235, 240, 245), 2)
        live_color = ((80, 210, 120) if self.execute_enabled else
                      (80, 180, 255))
        _put_text(image, self.connection_state,
                  (width - 380, 34), 0.50, live_color, 2)

        cv2.rectangle(image, (20, 610), (width - 20, height - 18),
                      (48, 55, 63), -1)
        _put_text(image,
                  "Commands: go .06 .06 [0.6] | 10 0 100 3 | "
                  "arm_delta 0 0 10 0 | stop | help",
                  (38, 642), 0.50, (180, 215, 240))
        _put_text(image, self.status, (38, 677), 0.48,
                  (80, 100, 255) if self.status_is_error else
                  (110, 225, 145), 2)
        cv2.rectangle(image, (36, 700), (width - 38, 754),
                      (18, 21, 25), -1)
        prompt = ("[BUSY] " if self.busy else "") + "> " + self.input_buffer
        if int(time.monotonic() * 2.0) % 2 == 0:
            prompt += "_"
        _put_text(image, prompt[-105:], (50, 735), 0.62,
                  (245, 245, 245), 2)
        _put_text(image, self.last_metrics, (38, 790), 0.45,
                  (180, 215, 240))
        arm = self.last_arm_feedback
        if arm is not None:
            _put_text(image,
                      "firmware arm: x {:+.2f} y {:+.2f} z {:+.2f} mm  "
                      "t {:+.3f} rad".format(
                          arm["x"], arm["y"], arm["z"], arm["t"]),
                      (38, 820), 0.45, (215, 220, 225))
        _put_text(image,
                  "UGV pulses and arm_delta use YAML step limits; bare X Y Z T "
                  "is one direct absolute arm command. Esc safely exits.",
                  (38, 854), 0.42, (145, 153, 162))
        return image

    def close(self):
        self.running = False
        self.stop_worker.set()
        try:
            self.command_queue.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=15.0)
        self.session.close()


def run_live(console):
    cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_WINDOW_NAME, 1280, 900)
    previous = time.monotonic()
    try:
        while console.running:
            now = time.monotonic()
            console.update(now - previous)
            previous = now
            cv2.imshow(LIVE_WINDOW_NAME, console.render())
            console.handle_key(cv2.waitKeyEx(16))
            if cv2.getWindowProperty(LIVE_WINDOW_NAME,
                                     cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        console.close()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="live calibration YAML")
    parser.add_argument("--execute", action="store_true",
                        help="allow commands typed in the GUI to move hardware")
    parser.add_argument("--acknowledge-motion", default="")
    parser.add_argument("--output", default="results/live_control.jsonl",
                        help="append every command/result to this JSONL")
    parser.add_argument("--default-drive-duration", type=float, default=0.6,
                        help="seconds used when typing only 'go L R'")
    parser.add_argument("--view-azimuth-deg", type=float, default=-45.0)
    parser.add_argument("--gripper-sweep-deg", type=float, default=90.0)
    args = parser.parse_args(argv)
    if args.execute and args.acknowledge_motion != MOTION_ACK:
        parser.error(
            f"physical motion also requires --acknowledge-motion {MOTION_ACK}")
    if not 0.1 <= args.default_drive_duration <= 1.5:
        parser.error("--default-drive-duration must be in [0.1, 1.5] seconds")

    config = load_config(args.config)
    configured_max = float(config["safety"]["max_ugv_duration_s"])
    if args.default_drive_duration > configured_max:
        parser.error(
            "default drive duration exceeds safety.max_ugv_duration_s")
    session = LiveCalibration(config, execute=args.execute)
    try:
        console = LiveRobotConsole(
            session, args.execute, args.output,
            default_drive_duration_s=args.default_drive_duration,
            arm_speed_rad_s=float(config["arm"].get("speed", 0.3)),
            max_wheel_speed=float(config["safety"]["max_abs_ugv_command"]),
            gripper_orientation="vertical",
            gripper_sweep_deg=args.gripper_sweep_deg,
            view_azimuth_deg=args.view_azimuth_deg)
    except Exception:
        session.close()
        raise
    run_live(console)


if __name__ == "__main__":
    main()
