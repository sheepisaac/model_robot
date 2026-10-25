#!/usr/bin/env python3
"""Safely compare live UGV/RoArm motion with the integrated model and L515 VO.

The command list accepts the same surface syntax as the existing controllers:
``go L R DURATION`` for the UGV and ``X Y Z T`` for the arm.  The command type
is inferred automatically.  Physical motion requires two explicit CLI flags.
"""

import argparse
import json
import math
from pathlib import Path
import socket
import sys
import time

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_models import IntegratedRobotCameraModel
from robot_models.transforms import invert_transform, rotation_angle
from robot_models.ugv import UGVModel
from calibrate_hand_eye import calibrate as calibrate_hand_eye
from calibrate_hand_eye import calibration_residuals


MOTION_ACK = "I_HAVE_CLEARED_THE_ROBOT"
STOP_COMMAND = '{"T":1,"L":0.0,"R":0.0}'


class JsonLineClient:
    def __init__(self, host, port, timeout=5.0):
        self.endpoint = f"{host}:{port}"
        self.socket = socket.create_connection((host, int(port)), timeout=timeout)
        self.socket.settimeout(timeout)
        self.stream = self.socket.makefile("rwb")
        hello = self.request({"type": "hello", "version": 1}, "hello")
        if hello.get("version") != 1:
            raise RuntimeError(f"{self.endpoint}: incompatible VO service")
        self.hello = hello

    def request(self, message, expected):
        self.stream.write((json.dumps(message, separators=(",", ":")) +
                           "\n").encode())
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise ConnectionError(f"{self.endpoint}: connection closed")
        reply = json.loads(line)
        if reply.get("type") == "error":
            raise RuntimeError(f"{self.endpoint}: {reply.get('error')}")
        if reply.get("type") != expected:
            raise RuntimeError(f"{self.endpoint}: expected {expected}, got {reply}")
        return reply

    def reset_reference(self):
        return self.request({"type": "reset_reference"}, "reference")

    def estimate(self):
        return self.request({"type": "estimate"}, "estimate")

    def status(self):
        return self.request({"type": "status"}, "status")

    def close(self):
        try:
            self.stream.close()
        finally:
            self.socket.close()


def parse_surface_command(text):
    fields = str(text).strip().split()
    if not fields:
        raise ValueError("empty command")
    if fields[0].lower() == "go":
        if len(fields) != 4:
            raise ValueError("UGV calibration syntax is: go LEFT RIGHT DURATION")
        left, right, duration = map(float, fields[1:])
        return {"kind": "ugv", "left": left, "right": right,
                "duration": duration, "text": text}
    if len(fields) == 4:
        x, y, z, t = map(float, fields)
        return {"kind": "arm", "x": x, "y": y, "z": z, "t": t,
                "text": text}
    raise ValueError("use 'go L R DURATION' or 'X Y Z T'")


def yaw_from_transform(transform):
    return math.atan2(float(transform[1, 0]), float(transform[0, 0]))


def transform_summary(transform):
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    return {"matrix": transform.tolist(),
            "translation_m": transform[:3, 3].tolist(),
            "distance_m": float(np.linalg.norm(transform[:3, 3])),
            "rotation_rad": rotation_angle(np.eye(3), transform[:3, :3]),
            "yaw_rad": yaw_from_transform(transform)}


def comparison(predicted, measured):
    residual = invert_transform(predicted).dot(measured)
    return {"predicted": transform_summary(predicted),
            "measured": transform_summary(measured),
            "residual": transform_summary(residual)}


def pose_to_joints(model, position):
    return model.roarm.inverse_kinematics({
        "x_mm": position["x"], "y_mm": position["y"],
        "z_mm": position["z"], "t": position["t"]})


def firmware_distance_delta(before, after, unit):
    scale = {"m": 1.0, "cm": 0.01}.get(unit)
    if scale is None:
        raise RuntimeError("configured firmware odometry unit is not m or cm")
    return ((float(after["odl"]) - float(before["odl"])) * scale,
            (float(after["odr"]) - float(before["odr"])) * scale)


class LiveCalibration:
    def __init__(self, config, execute=False):
        self.config = config
        self.execute = execute
        self.drive = None
        self.arm = None
        self.vo = None
        control_root = Path(config["control_root"]).expanduser()
        if not control_root.is_absolute():
            control_root = (PROJECT_ROOT / control_root).resolve()
        if not control_root.is_dir():
            raise FileNotFoundError(f"control_root does not exist: {control_root}")
        sys.path.insert(0, str(control_root))
        from drive_master_sync import DriveMaster
        from master_arm import Master
        self.DriveMaster, self.ArmMaster = DriveMaster, Master
        model_path = Path(config["model_config"])
        if not model_path.is_absolute():
            model_path = (PROJECT_ROOT / model_path).resolve()
        self.model = IntegratedRobotCameraModel.from_yaml(model_path)
        robot = config["robot"]
        host = str(robot["host"])
        timeout = float(robot.get("timeout_s", 5.0))
        samples = int(robot.get("sync_samples", 5))
        try:
            self.drive = self.DriveMaster(
                [f"{host}:{int(robot.get('drive_port', 50220))}"],
                timeout, samples)
            self.arm = self.ArmMaster(
                [f"{host}:{int(robot.get('arm_port', 50210))}"],
                timeout, samples)
            self.vo = JsonLineClient(
                host, int(robot.get("vo_port", 50240)), timeout)
            if len(self.drive.robots) != 1 or len(self.arm.robots) != 1:
                raise RuntimeError(
                    "live calibration currently requires exactly one robot")
            if not self.arm.robots[0].arm_feedback_supported:
                raise RuntimeError(
                    "deploy the updated slave_arm.py before calibration")
        except Exception:
            self.close()
            raise

    def drive_feedback(self):
        return next(iter(self.drive.telemetry(1.5).values()))["feedback"]

    def arm_feedback(self):
        reply, _ = self.arm.robots[0].request(
            {"type": "arm_feedback", "timeout": 1.5}, "arm_feedback")
        return reply["position"]

    def probe(self):
        return {"mode": "read_only_probe", "timestamp_ns": time.time_ns(),
                "drive": self.drive_feedback(), "arm": self.arm_feedback(),
                "visual_odometry": self.vo.status(),
                "model_calibration_status": self.model.calibration_status}

    def _safe_stop(self):
        if not self.execute or self.drive is None:
            return
        try:
            self.drive.dispatch([(0.0, "calibration_stop", STOP_COMMAND)],
                                0.35, report=False)
        except Exception as exc:
            print(f"WARNING: explicit UGV stop failed: {exc}", file=sys.stderr)

    def _check_ugv_limits(self, command):
        limits = self.config["safety"]
        maximum = float(limits["max_abs_ugv_command"])
        if max(abs(command["left"]), abs(command["right"])) > maximum:
            raise ValueError(f"UGV command exceeds safety limit {maximum}")
        if not 0.1 <= command["duration"] <= float(limits["max_ugv_duration_s"]):
            raise ValueError("UGV duration is outside the configured safety range")

    def _check_arm_limits(self, before, command):
        limits = self.config["safety"]
        xyz_step = np.linalg.norm([
            command["x"] - before["x"], command["y"] - before["y"],
            command["z"] - before["z"]])
        if xyz_step > float(limits["max_arm_xyz_step_mm"]):
            raise ValueError(f"arm Cartesian step {xyz_step:.1f} mm exceeds limit")
        if abs(command["t"] - before["t"]) > float(limits["max_arm_t_step_rad"]):
            raise ValueError("arm t step exceeds configured safety limit")
        pose_to_joints(self.model, command)

    def run_ugv(self, command):
        self._check_ugv_limits(command)
        arm_before = self.arm_feedback()
        joints = pose_to_joints(self.model, arm_before)
        telemetry_before = self.drive_feedback()
        reference = self.vo.reset_reference()
        left, right, duration = (command["left"], command["right"],
                                 command["duration"])
        start = json.dumps({"T": 1, "L": left, "R": right},
                           separators=(",", ":"))
        try:
            self.drive.dispatch([(0.0, "start", start),
                                 (duration, "stop", STOP_COMMAND)],
                                0.5, report=True)
        finally:
            self._safe_stop()
        time.sleep(float(self.config["sampling"].get("settle_s", 0.5)))
        telemetry_after = self.drive_feedback()
        arm_after = self.arm_feedback()
        vo = self.vo.estimate()
        if np.linalg.norm([arm_after[key] - arm_before[key]
                           for key in ("x", "y", "z")]) > 3.0:
            raise RuntimeError("arm moved during UGV-only trial")

        T_C0_C1 = np.asarray(vo["T_C0_C1"], dtype=float)
        T_BC = self.model.compute_camera_pose(np.zeros(3), joints)
        T_B0_B1_visual = T_BC.dot(T_C0_C1).dot(invert_transform(T_BC))
        dl, dr = firmware_distance_delta(
            telemetry_before, telemetry_after,
            self.model.ugv.parameters.firmware_odometry_unit)
        encoder_pose = self.model.ugv.integrate_odometry(
            np.zeros(3), {"delta_left_m": dl, "delta_right_m": dr})
        from robot_models.transforms import pose2d_transform
        T_encoder = pose2d_transform(encoder_pose)
        nominal_mps = float(self.config["ugv"].get(
            "nominal_mps_per_command", 1.0))
        expected_pose = self.model.ugv.predict_motion(
            np.zeros(3), left * nominal_mps, right * nominal_mps, duration)
        T_expected = pose2d_transform(expected_pose)
        visual_yaw = yaw_from_transform(T_B0_B1_visual)
        visual_distance = math.copysign(
            float(np.linalg.norm(T_B0_B1_visual[:2, 3])),
            float(T_B0_B1_visual[0, 3]))
        return {"kind": "ugv", "command": command,
                "reference": reference, "visual_odometry": vo,
                "firmware_before": telemetry_before,
                "firmware_after": telemetry_after,
                "firmware_wheel_delta_m": [dl, dr],
                "nominal_vs_visual": comparison(T_expected, T_B0_B1_visual),
                "encoder_vs_visual": comparison(T_encoder, T_B0_B1_visual),
                "fit_row": {"v_left": left, "v_right": right,
                            "duration": duration, "distance_m": visual_distance,
                            "yaw_delta_rad": visual_yaw},
                "arm_before": arm_before, "arm_after": arm_after}

    def run_arm(self, command):
        before = self.arm_feedback()
        self._check_arm_limits(before, command)
        telemetry_before = self.drive_feedback()
        joints_before = pose_to_joints(self.model, before)
        reference = self.vo.reset_reference()
        speed = min(float(self.config["arm"].get("speed", 0.35)),
                    float(self.config["safety"]["max_arm_speed"]))
        arm_json = json.dumps({"T": 104, "x": command["x"],
                               "y": command["y"], "z": command["z"],
                               "t": command["t"], "spd": speed},
                              separators=(",", ":"))
        verification = {"target": {key: command[key]
                                    for key in ("x", "y", "z", "t")},
                        "timeout": 10.0, "feedback_start_delay": 1.0,
                        "xyz_tolerance": 8.0, "t_tolerance": 0.15,
                        "stable_samples": 2}
        results = self.arm.dispatch(
            [(0.0, "arm_position", arm_json, verification)], 0.5, report=True)
        if not (results[0].get("verification") or {}).get("success"):
            raise RuntimeError("arm target verification failed")
        time.sleep(float(self.config["sampling"].get("settle_s", 0.5)))
        after = self.arm_feedback()
        telemetry_after = self.drive_feedback()
        dl, dr = firmware_distance_delta(
            telemetry_before, telemetry_after,
            self.model.ugv.parameters.firmware_odometry_unit)
        if max(abs(dl), abs(dr)) > 0.005:
            raise RuntimeError("UGV moved during arm-only trial")
        joints_after = pose_to_joints(self.model, after)
        T_before = self.model.compute_camera_pose(np.zeros(3), joints_before)
        T_after = self.model.compute_camera_pose(np.zeros(3), joints_after)
        T_BE_before = self.model.end_pose_world(np.zeros(3), joints_before)
        T_BE_after = self.model.end_pose_world(np.zeros(3), joints_after)
        T_predicted = invert_transform(T_before).dot(T_after)
        vo = self.vo.estimate()
        T_visual = np.asarray(vo["T_C0_C1"], dtype=float)
        return {"kind": "arm", "command": command, "reference": reference,
                "visual_odometry": vo,
                "model_vs_visual": comparison(T_predicted, T_visual),
                "firmware_before": before, "firmware_after": after,
                "T_BE_before": T_BE_before.tolist(),
                "T_BE_after": T_BE_after.tolist(),
                "ugv_wheel_delta_m": [dl, dr]}

    def run_commands(self, commands, output_path):
        if not self.execute:
            raise RuntimeError("run_commands requires explicit physical-motion mode")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records, ugv_rows, hand_eye_segments = [], [], []
        hand_eye_segment = None

        def finish_hand_eye_segment():
            nonlocal hand_eye_segment
            if hand_eye_segment and len(hand_eye_segment["samples"]) >= 4:
                hand_eye_segments.append(hand_eye_segment)
            hand_eye_segment = None

        with output_path.open("a", encoding="utf-8") as handle:
            for index, text in enumerate(commands):
                command = parse_surface_command(text)
                print(f"[{index + 1}/{len(commands)}] {command['kind']}: {text}",
                      flush=True)
                try:
                    record = (self.run_ugv(command) if command["kind"] == "ugv"
                              else self.run_arm(command))
                    record.update({"timestamp_ns": time.time_ns(),
                                   "sequence": index, "success": True})
                    if "fit_row" in record:
                        ugv_rows.append(record["fit_row"])
                        finish_hand_eye_segment()
                    elif record["kind"] == "arm":
                        before_t = float(record["firmware_before"]["t"])
                        after_t = float(record["firmware_after"]["t"])
                        if (hand_eye_segment is None or
                                abs(before_t - hand_eye_segment["fixed_t_rad"]) > 0.03 or
                                abs(after_t - hand_eye_segment["fixed_t_rad"]) > 0.03):
                            finish_hand_eye_segment()
                            if abs(after_t - before_t) <= 0.03:
                                hand_eye_segment = {
                                    "fixed_t_rad": 0.5 * (before_t + after_t),
                                    "T_Ci_Cstart": np.eye(4),
                                    "samples": [{
                                        "T_BG": record["T_BE_before"],
                                        "T_CT": np.eye(4).tolist(),
                                    }],
                                }
                        if hand_eye_segment is not None:
                            step = np.asarray(
                                record["visual_odometry"]["T_C1_C0"], dtype=float)
                            cumulative = step.dot(
                                hand_eye_segment["T_Ci_Cstart"])
                            hand_eye_segment["T_Ci_Cstart"] = cumulative
                            hand_eye_segment["samples"].append({
                                "T_BG": record["T_BE_after"],
                                "T_CT": cumulative.tolist(),
                            })
                except Exception as exc:
                    self._safe_stop()
                    record = {"timestamp_ns": time.time_ns(), "sequence": index,
                              "command": command, "success": False,
                              "error": str(exc)}
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    raise
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)
        finish_hand_eye_segment()
        fits = {"ugv": None, "fixed_t_hand_eye": None}
        if len(ugv_rows) >= 3:
            try:
                fitted = UGVModel.estimate_motion_parameters(ugv_rows).parameters
                fits["ugv"] = {
                    "status": "CANDIDATE_REQUIRES_REPEAT_VALIDATION",
                    "effective_track_width_m": fitted.effective_track_width_m,
                    "left_scale_mps_per_command": fitted.left_scale,
                    "right_scale_mps_per_command": fitted.right_scale,
                    "sample_count": len(ugv_rows)}
            except ValueError as exc:
                fits["ugv"] = {"status": "REJECTED", "reason": str(exc),
                               "sample_count": len(ugv_rows)}
        hand_eye_candidates = []
        for segment in hand_eye_segments:
            try:
                T_EC, singular = calibrate_hand_eye(segment["samples"])
                residuals = calibration_residuals(segment["samples"], T_EC)
                hand_eye_candidates.append({
                    "status": "CANDIDATE_REQUIRES_REPEAT_VALIDATION",
                    "fixed_t_rad": segment["fixed_t_rad"],
                    "T_EC": T_EC.tolist(),
                    "rotation_singular_values": singular.tolist(),
                    "validation": residuals,
                    "sample_count": len(segment["samples"]),
                })
            except ValueError as exc:
                hand_eye_candidates.append({"status": "REJECTED",
                                            "reason": str(exc),
                                            "sample_count": len(segment["samples"])})
        if hand_eye_candidates:
            accepted = [item for item in hand_eye_candidates
                        if "validation" in item]
            fits["fixed_t_hand_eye"] = (
                min(accepted, key=lambda item:
                    item["validation"]["translation_rmse_m"])
                if accepted else hand_eye_candidates[0])
        candidate_path = output_path.with_suffix(".candidates.yaml")
        with candidate_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump({"source_jsonl": str(output_path),
                            "calibration_candidates": fits},
                           handle, sort_keys=False)
        return records, fits

    def close(self):
        self._safe_stop()
        if self.execute and self.drive is not None:
            try:
                self.drive.release_control()
            except Exception as exc:
                print("WARNING: clean drive release failed; disconnect safety "
                      f"stop will be used: {exc}", file=sys.stderr)
        # Close the drive before the arm. If clean release failed, the drive
        # slave deliberately emits global T=0 and the still-running arm
        # watchdog gets a chance to restore holding torque.
        for resource in (self.vo, self.drive, self.arm):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception:
                pass
            if resource is self.drive and self.arm is not None:
                time.sleep(0.35)


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("live calibration YAML must contain a mapping")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--probe", action="store_true",
                        help="read all three services without motor movement")
    parser.add_argument("--execute", action="store_true",
                        help="allow plan commands to move physical hardware")
    parser.add_argument("--acknowledge-motion", default="")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.execute and args.acknowledge_motion != MOTION_ACK:
        parser.error(f"physical motion also requires --acknowledge-motion {MOTION_ACK}")
    if not args.probe and not args.execute:
        parser.error("choose --probe or explicitly enable --execute")
    session = LiveCalibration(config, execute=args.execute)
    try:
        if args.probe:
            print(json.dumps(session.probe(), indent=2))
            return
        commands = config.get("commands", [])
        if not commands:
            raise RuntimeError("the plan has no commands")
        output = args.output or config.get("output_jsonl", "results/live_calibration.jsonl")
        records, fits = session.run_commands(commands, output)
        print(json.dumps({"records": len(records), "calibration_candidates": fits,
                          "output": str(output)}, indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    main()
