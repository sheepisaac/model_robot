"""Unified UGV02 - RoArm-M2-S - moving-gripper L515 pose model."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

from .camera import CameraIntrinsics, L515CameraModel
from .gripper import GripperCameraMount
from .roarm import RoArmModel
from .transforms import (invert_transform, pose2d_transform, rotation_angle,
                         transform_from_config, validate_transform)
from .ugv import UGVModel, UGVParameters


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    joint_angles: Optional[np.ndarray]
    reason: str = ""
    position_error_m: float = math.inf
    orientation_error_rad: float = math.inf
    calibrated: bool = False


class IntegratedRobotCameraModel:
    """Compose the mobile manipulator and its camera mount.

    For the user's EEMode-0 hardware, the L515 is attached to the moving
    gripper member:

    ``T_WC = T_WB @ T_BA @ T_AE(q) @ T_EG(t) @ T_GC``

    A legacy fixed ``T_EC`` remains supported for EEMode 1 and for older
    configuration files.  No calibration state is inferred from numerical
    values alone.
    """

    def __init__(self, roarm: RoArmModel, ugv: UGVModel,
                 camera: L515CameraModel, T_BA, T_EC=None,
                 t_ba_status: str = "UNCALIBRATED",
                 t_ec_status: str = "UNCALIBRATED",
                 camera_mount: Optional[GripperCameraMount] = None):
        self.roarm = roarm
        self.ugv = ugv
        self.camera = camera
        self.T_BA = validate_transform(np.asarray(T_BA, dtype=float)).copy()
        self.t_ba_status = str(t_ba_status).upper()
        if camera_mount is None:
            if T_EC is None:
                raise ValueError("T_EC or camera_mount is required")
            camera_mount = GripperCameraMount.fixed(
                validate_transform(np.asarray(T_EC, dtype=float)),
                status=t_ec_status)
        if not isinstance(camera_mount, GripperCameraMount):
            raise TypeError("camera_mount must be GripperCameraMount")
        self.camera_mount = camera_mount
        # Backward-compatible reference transform.  Dynamic callers must use
        # camera_mount.camera_transform(t) rather than treating this as fixed.
        self.T_EC = camera_mount.camera_transform(
            camera_mount.reference_t_rad).copy()
        self.t_ec_status = ("CALIBRATED" if camera_mount.calibrated else
                            "UNCALIBRATED")

    @property
    def calibration_status(self):
        return {
            "T_BA": self.t_ba_status,
            "gripper_motion": self.camera_mount.motion_status,
            "T_GC": self.camera_mount.t_gc_status,
            "l515_intrinsics": self.camera.status,
            "ugv": "CALIBRATED" if self.ugv.parameters.calibrated else
                   "UNCALIBRATED",
        }

    @property
    def calibrated(self) -> bool:
        return (self.t_ba_status == "CALIBRATED" and
                self.camera_mount.calibrated and
                self.camera.calibrated)

    @classmethod
    def from_yaml(cls, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        arm_config = config.get("roarm", {})
        ugv_config = config.get("ugv", {})
        camera_config = config["l515"]
        extrinsics = config["extrinsics"]
        roarm = RoArmModel(eemode=int(arm_config.get("eemode", 0)))
        ugv = UGVModel(UGVParameters.from_mapping(ugv_config))
        camera = L515CameraModel(
            CameraIntrinsics.from_mapping(camera_config),
            status=camera_config.get("status", "UNCALIBRATED"))
        t_ba = extrinsics["T_BA"]
        t_ba_status = (t_ba.get("status", "UNCALIBRATED")
                       if isinstance(t_ba, dict) else "UNCALIBRATED")

        if "gripper_camera" in config:
            if "T_GC" not in extrinsics:
                raise ValueError("moving gripper config requires extrinsics.T_GC")
            mount = GripperCameraMount.from_config(
                config["gripper_camera"], extrinsics["T_GC"])
            return cls(roarm, ugv, camera, transform_from_config(t_ba),
                       t_ba_status=t_ba_status, camera_mount=mount)

        # Backward-compatible fixed-camera configuration.
        t_ec = extrinsics["T_EC"]
        t_ec_status = (t_ec.get("status", "UNCALIBRATED")
                       if isinstance(t_ec, dict) else "UNCALIBRATED")
        return cls(roarm, ugv, camera, transform_from_config(t_ba),
                   transform_from_config(t_ec), t_ba_status, t_ec_status)

    def end_pose_world(self, base_pose: Sequence[float],
                       joint_angles: Sequence[float]) -> np.ndarray:
        return (pose2d_transform(base_pose).dot(self.T_BA).dot(
            self.roarm.forward_kinematics(joint_angles)))

    def gripper_pose_world(self, base_pose: Sequence[float],
                           joint_angles: Sequence[float]) -> np.ndarray:
        joints = self.roarm._joints(joint_angles)
        return self.end_pose_world(base_pose, joints).dot(
            self.camera_mount.gripper_transform(joints[3]))

    def compute_camera_pose(self, base_pose: Sequence[float],
                            joint_angles: Sequence[float]) -> np.ndarray:
        joints = self.roarm._joints(joint_angles)
        return self.end_pose_world(base_pose, joints).dot(
            self.camera_mount.camera_transform(joints[3]))

    predict_camera_pose = compute_camera_pose

    def predict_camera_motion(self, base_pose, joint_angles, base_motion,
                              joint_motion):
        before = self.compute_camera_pose(base_pose, joint_angles)
        new_base = self.ugv.predict_motion(
            base_pose, float(base_motion["left"]), float(base_motion["right"]),
            float(base_motion["duration"]))
        new_joints = np.asarray(joint_angles, dtype=float) + np.asarray(
            joint_motion, dtype=float)
        after = self.compute_camera_pose(new_base, new_joints)
        return {"before": before, "after": after, "relative":
                invert_transform(before).dot(after), "base_pose": new_base,
                "joint_angles": new_joints}

    def _calibration_failure(self):
        missing = [name for name, status in self.calibration_status.items()
                   if name != "ugv" and status != "CALIBRATED"]
        return "camera pose is not calibrated; missing: " + ", ".join(missing)

    def _evaluate_camera_solution(self, target, base_pose, joints,
                                  position_tolerance_m,
                                  orientation_tolerance_rad):
        achieved = self.compute_camera_pose(base_pose, joints)
        position_error = float(np.linalg.norm(
            achieved[:3, 3] - target[:3, 3]))
        orientation_error = rotation_angle(achieved[:3, :3], target[:3, :3])
        score = (position_error / max(position_tolerance_m, 1e-12) +
                 orientation_error / max(orientation_tolerance_rad, 1e-12))
        return score, position_error, orientation_error

    def check_camera_view_reachability(self, target_view,
                                       base_pose=(0.0, 0.0, 0.0),
                                       allow_uncalibrated: bool = False,
                                       position_tolerance_m: float = 0.01,
                                       orientation_tolerance_rad: float = 0.35):
        target = validate_transform(np.asarray(target_view, dtype=float))
        if not self.calibrated and not allow_uncalibrated:
            return ReachabilityResult(
                False, None, self._calibration_failure(), calibrated=False)

        T_WB = pose2d_transform(base_pose)
        T_AW = invert_transform(self.T_BA).dot(invert_transform(T_WB))

        if self.camera_mount.dynamic and self.roarm.eemode == 0:
            candidates = self.camera_mount.candidate_t_values(
                self.roarm.joint_limits[3])
        else:
            candidates = self.camera_mount.candidate_t_values(
                self.roarm.joint_limits[3], count=1)

        best = None
        failure_reasons = []
        for t_rad in candidates:
            try:
                T_EC = self.camera_mount.camera_transform(t_rad)
            except ValueError as exc:
                failure_reasons.append(str(exc))
                continue
            target_end = T_AW.dot(target).dot(invert_transform(T_EC))
            ik = self.roarm.solve_ik(
                target_end,
                tool_angle=float(t_rad) if self.roarm.eemode == 0 else None,
                position_tolerance_m=position_tolerance_m)
            if not ik.success:
                failure_reasons.append(ik.reason)
                continue
            score, position_error, orientation_error = (
                self._evaluate_camera_solution(
                    target, base_pose, ik.joint_angles,
                    position_tolerance_m, orientation_tolerance_rad))
            candidate = (score, ik, position_error, orientation_error)
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None:
            reason = "no gripper/camera kinematic solution"
            if failure_reasons:
                reason += ": " + failure_reasons[0]
            return ReachabilityResult(False, None, reason,
                                      calibrated=self.calibrated)

        _, ik, position_error, orientation_error = best
        reachable = (position_error <= position_tolerance_m and
                     orientation_error <= orientation_tolerance_rad)
        reason = ""
        if not reachable:
            if (ik.orientation_limited and
                    position_error <= position_tolerance_m):
                reason = ("position reachable, but EEMode 0 cannot satisfy the "
                          "requested camera orientation")
            else:
                reason = "camera pose residual exceeds configured tolerance"
        elif ik.orientation_limited:
            reason = ("pose matches, but EEMode 0 orientation is constrained by "
                      "the arm and moving-gripper mechanism")
        return ReachabilityResult(reachable, ik.joint_angles, reason,
                                  position_error, orientation_error,
                                  self.calibrated)
