"""Optional bridge from the mathematical model to existing autoNBV types.

This module computes commands only.  It never opens a serial port or sends a
vcrobot command, preserving the existing motion safety boundary.
"""

from pathlib import Path

import numpy as np

from robot_models import IntegratedRobotCameraModel


class AutoNBVRobotModel:
    def __init__(self, calibration_path):
        self.calibration_path = Path(calibration_path)
        self.model = IntegratedRobotCameraModel.from_yaml(self.calibration_path)

    @property
    def hardware_ready(self):
        return self.model.calibrated and self.model.ugv.parameters.calibrated

    def camera_pose_from_joints(self, base_pose, joint_angles):
        return self.model.compute_camera_pose(base_pose, joint_angles)

    def camera_pose_from_arm_feedback(self, base_pose, arm_pose):
        """Convert vcrobot ``x,y,z,t`` feedback through IK, then return T_WC."""
        target = {
            "x_mm": float(arm_pose["x"]),
            "y_mm": float(arm_pose["y"]),
            "z_mm": float(arm_pose["z"]),
            "t": float(arm_pose["t"]),
        }
        joints = self.model.roarm.inverse_kinematics(target)
        return self.model.compute_camera_pose(base_pose, joints)

    def arm_command_for_camera_view(self, target_camera_pose, base_pose,
                                    allow_uncalibrated=False):
        """Return an existing vcrobot-compatible XYZT dictionary or raise."""
        result = self.model.check_camera_view_reachability(
            np.asarray(target_camera_pose, dtype=float), base_pose=base_pose,
            allow_uncalibrated=allow_uncalibrated)
        if not result.reachable:
            raise ValueError(result.reason)
        pose = self.model.roarm.firmware_pose(result.joint_angles)
        return {"x": float(pose[0]), "y": float(pose[1]),
                "z": float(pose[2]), "t": float(pose[3])}
