"""Unified UGV02 - RoArm-M2-S - L515 pose model."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

from .camera import CameraIntrinsics, L515CameraModel
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
    """Compose ``T_WC = T_WB @ T_BA @ T_AE(q) @ T_EC``.

    ``T_BA`` maps the RoArm firmware frame into the UGV base frame. ``T_EC``
    maps L515 camera coordinates into the modeled arm end frame.  Neither is
    silently treated as safe when its status is UNCALIBRATED.
    """

    def __init__(self, roarm: RoArmModel, ugv: UGVModel,
                 camera: L515CameraModel, T_BA, T_EC,
                 t_ba_status: str = "UNCALIBRATED",
                 t_ec_status: str = "UNCALIBRATED"):
        self.roarm = roarm
        self.ugv = ugv
        self.camera = camera
        self.T_BA = validate_transform(np.asarray(T_BA, dtype=float)).copy()
        self.T_EC = validate_transform(np.asarray(T_EC, dtype=float)).copy()
        self.t_ba_status = str(t_ba_status).upper()
        self.t_ec_status = str(t_ec_status).upper()

    @property
    def calibrated(self) -> bool:
        return self.t_ba_status == "CALIBRATED" and self.t_ec_status == "CALIBRATED"

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
        ugv = UGVModel(UGVParameters(
            wheel_diameter_m=float(ugv_config.get("wheel_diameter_m", 0.0800)),
            encoder_pulses_per_rev=int(ugv_config.get("encoder_pulses_per_rev", 660)),
            effective_track_width_m=float(ugv_config.get(
                "effective_track_width_m", 0.172)),
            left_scale=float(ugv_config.get("left_scale", 1.0)),
            right_scale=float(ugv_config.get("right_scale", 1.0)),
            calibrated=bool(ugv_config.get("calibrated", False))))
        camera = L515CameraModel(CameraIntrinsics.from_mapping(camera_config))
        t_ba = extrinsics["T_BA"]
        t_ec = extrinsics["T_EC"]
        return cls(roarm, ugv, camera, transform_from_config(t_ba),
                   transform_from_config(t_ec), t_ba.get("status", "UNCALIBRATED")
                   if isinstance(t_ba, dict) else "UNCALIBRATED",
                   t_ec.get("status", "UNCALIBRATED")
                   if isinstance(t_ec, dict) else "UNCALIBRATED")

    def compute_camera_pose(self, base_pose: Sequence[float],
                            joint_angles: Sequence[float]) -> np.ndarray:
        T_WB = pose2d_transform(base_pose)
        T_AE = self.roarm.forward_kinematics(joint_angles)
        return T_WB.dot(self.T_BA).dot(T_AE).dot(self.T_EC)

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

    def check_camera_view_reachability(self, target_view,
                                       base_pose=(0.0, 0.0, 0.0),
                                       allow_uncalibrated: bool = False,
                                       position_tolerance_m: float = 0.01,
                                       orientation_tolerance_rad: float = 0.35):
        target = validate_transform(np.asarray(target_view, dtype=float))
        if not self.calibrated and not allow_uncalibrated:
            return ReachabilityResult(
                False, None, "T_BA/T_EC are not calibrated; refusing safety claim",
                calibrated=False)
        T_WB = pose2d_transform(base_pose)
        target_end = (invert_transform(self.T_BA).dot(invert_transform(T_WB)).dot(
            target).dot(invert_transform(self.T_EC)))
        ik = self.roarm.solve_ik(target_end,
                                 position_tolerance_m=position_tolerance_m)
        if not ik.success:
            return ReachabilityResult(False, ik.joint_angles, ik.reason,
                                      ik.position_error_m, calibrated=self.calibrated)
        achieved = self.compute_camera_pose(base_pose, ik.joint_angles)
        position_error = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
        orientation_error = rotation_angle(achieved[:3, :3], target[:3, :3])
        # EEMode 0 cannot command arbitrary end orientation. Preserve the
        # positional solution but make the orientation limitation explicit.
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
            reason = ("pose matches, but EEMode 0 orientation is passive rather "
                      "than independently controllable")
        return ReachabilityResult(reachable, ik.joint_angles, reason,
                                  position_error, orientation_error,
                                  self.calibrated)
