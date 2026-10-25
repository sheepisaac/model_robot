"""Stateful UGV, arm and camera motion accounting.

This module is an odometry *model*, not a localization sensor.  Wheel feedback
is integrated in SE(2), arm feedback is converted through FK/IK, and the
calibrated transform chain propagates both changes to the L515 pose.
"""

from dataclasses import dataclass
import math
from typing import Dict, Sequence

import numpy as np

from .transforms import invert_transform, pose2d_transform, rotation_angle


@dataclass(frozen=True)
class PoseDelta:
    """Relative transform and compact magnitude summary."""

    transform: np.ndarray
    translation: np.ndarray
    distance_m: float
    rotation_rad: float

    @classmethod
    def between(cls, before, after):
        relative = invert_transform(before).dot(after)
        translation = relative[:3, 3].copy()
        return cls(relative, translation, float(np.linalg.norm(translation)),
                   rotation_angle(np.eye(3), relative[:3, :3]))

    def to_dict(self):
        return {"transform": self.transform.tolist(),
                "translation_m": self.translation.tolist(),
                "distance_m": self.distance_m,
                "rotation_rad": self.rotation_rad,
                "rotation_deg": math.degrees(self.rotation_rad)}


@dataclass(frozen=True)
class OdometryUpdate:
    base_pose_before: np.ndarray
    base_pose_after: np.ndarray
    base_delta_body: np.ndarray
    joint_angles_before: np.ndarray
    joint_angles_after: np.ndarray
    joint_delta: np.ndarray
    camera_pose_before: np.ndarray
    camera_pose_after: np.ndarray
    camera_from_base: PoseDelta
    camera_from_arm: PoseDelta
    camera_total: PoseDelta
    sources: Dict[str, str]
    calibrated: bool

    def to_dict(self):
        return {
            "base_pose_before": self.base_pose_before.tolist(),
            "base_pose_after": self.base_pose_after.tolist(),
            "base_delta_body": self.base_delta_body.tolist(),
            "joint_angles_before": self.joint_angles_before.tolist(),
            "joint_angles_after": self.joint_angles_after.tolist(),
            "joint_delta": self.joint_delta.tolist(),
            "camera_pose_before": self.camera_pose_before.tolist(),
            "camera_pose_after": self.camera_pose_after.tolist(),
            "camera_from_base": self.camera_from_base.to_dict(),
            "camera_from_arm": self.camera_from_arm.to_dict(),
            "camera_total": self.camera_total.to_dict(),
            "sources": dict(self.sources),
            "calibrated": self.calibrated,
        }


class RobotCameraOdometry:
    """Accumulate feedback and report base/arm/camera increments.

    The state starts from an explicit base pose and four arm joint angles.
    Encoder-count and cumulative-distance methods need one baseline call before
    they can produce an update.  `None` from that first call means "baseline
    stored", not a failed measurement.
    """

    def __init__(self, model, base_pose: Sequence[float],
                 joint_angles: Sequence[float]):
        self.model = model
        self.base_pose = model.ugv._state(base_pose).copy()
        self.joint_angles = model.roarm._joints(joint_angles).copy()
        if not model.roarm.check_joint_limits(self.joint_angles):
            raise ValueError("initial joint angles violate configured limits")
        self._encoder_counts = None
        self._wheel_distances = None

    @property
    def camera_pose(self):
        return self.model.compute_camera_pose(self.base_pose, self.joint_angles)

    def _resolve_joints(self, joint_angles=None, arm_pose=None):
        if joint_angles is not None and arm_pose is not None:
            raise ValueError("provide joint_angles or arm_pose, not both")
        if arm_pose is not None:
            target = {"x_mm": float(arm_pose["x"]),
                      "y_mm": float(arm_pose["y"]),
                      "z_mm": float(arm_pose["z"]),
                      "t": float(arm_pose["t"])}
            joints = self.model.roarm.inverse_kinematics(target)
        elif joint_angles is not None:
            joints = self.model.roarm._joints(joint_angles)
        else:
            joints = self.joint_angles.copy()
        if not self.model.roarm.check_joint_limits(joints):
            raise ValueError("updated joint angles violate configured limits")
        return joints

    def _commit(self, new_base, new_joints, sources):
        old_base = self.base_pose.copy()
        old_joints = self.joint_angles.copy()
        new_base = self.model.ugv._state(new_base).copy()
        new_joints = self.model.roarm._joints(new_joints).copy()

        camera_before = self.model.compute_camera_pose(old_base, old_joints)
        camera_base_only = self.model.compute_camera_pose(new_base, old_joints)
        camera_arm_only = self.model.compute_camera_pose(old_base, new_joints)
        camera_after = self.model.compute_camera_pose(new_base, new_joints)

        relative_base = invert_transform(pose2d_transform(old_base)).dot(
            pose2d_transform(new_base))
        base_delta = np.array([
            relative_base[0, 3], relative_base[1, 3],
            math.atan2(relative_base[1, 0], relative_base[0, 0])], dtype=float)

        update = OdometryUpdate(
            old_base, new_base, base_delta, old_joints, new_joints,
            new_joints - old_joints, camera_before, camera_after,
            PoseDelta.between(camera_before, camera_base_only),
            PoseDelta.between(camera_before, camera_arm_only),
            PoseDelta.between(camera_before, camera_after),
            dict(sources), self.model.calibrated)
        self.base_pose = new_base
        self.joint_angles = new_joints
        return update

    def update_absolute(self, base_pose=None, joint_angles=None, arm_pose=None):
        new_base = self.base_pose if base_pose is None else base_pose
        new_joints = self._resolve_joints(joint_angles, arm_pose)
        return self._commit(new_base, new_joints,
                            {"base": "absolute" if base_pose is not None else "unchanged",
                             "arm": "xyzt_ik" if arm_pose is not None else
                             ("joint_feedback" if joint_angles is not None else "unchanged")})

    def update_wheel_velocity(self, v_left: float, v_right: float, dt: float,
                              joint_angles=None, arm_pose=None):
        new_base = self.model.ugv.integrate_odometry(
            self.base_pose, {"v_left": v_left, "v_right": v_right}, dt=dt)
        return self._commit(new_base, self._resolve_joints(joint_angles, arm_pose),
                            {"base": "wheel_velocity", "arm": self._arm_source(
                                joint_angles, arm_pose)})

    def update_wheel_distances(self, delta_left_m: float, delta_right_m: float,
                               joint_angles=None, arm_pose=None):
        new_base = self.model.ugv.integrate_odometry(
            self.base_pose, {"delta_left_m": delta_left_m,
                             "delta_right_m": delta_right_m})
        return self._commit(new_base, self._resolve_joints(joint_angles, arm_pose),
                            {"base": "wheel_distance", "arm": self._arm_source(
                                joint_angles, arm_pose)})

    def update_encoder_counts(self, left_count: float, right_count: float,
                              joint_angles=None, arm_pose=None):
        current = np.array([float(left_count), float(right_count)])
        if self._encoder_counts is None:
            self._encoder_counts = current
            return None
        delta = current - self._encoder_counts
        self._encoder_counts = current
        return self.update_wheel_distances(
            self.model.ugv.encoder_counts_to_distance(delta[0]),
            self.model.ugv.encoder_counts_to_distance(delta[1]),
            joint_angles=joint_angles, arm_pose=arm_pose)

    def update_cumulative_wheel_distances(self, left_m: float, right_m: float,
                                          joint_angles=None, arm_pose=None):
        """Update from cumulative physical wheel distances in metres.

        Do not pass the Waveshare ROS firmware's raw ``odl``/``odr`` fields to
        this method: those fields are integer centimetres.  Use
        :meth:`update_firmware_odometry` for raw firmware feedback.
        """
        current = np.array([float(left_m), float(right_m)])
        if not np.all(np.isfinite(current)):
            raise ValueError("cumulative wheel distances must be finite")
        if self._wheel_distances is None:
            self._wheel_distances = current
            return None
        delta = current - self._wheel_distances
        self._wheel_distances = current
        return self.update_wheel_distances(
            delta[0], delta[1], joint_angles=joint_angles, arm_pose=arm_pose)

    def update_firmware_odometry(self, odl, odr, joint_angles=None,
                                 arm_pose=None):
        """Update from raw cumulative ``odl``/``odr`` firmware fields.

        The supplied ``ugv_base_ros`` firmware publishes these values as
        integer centimetres (``long(en_odom_* * 100)``).  Other firmware
        profiles must declare their unit explicitly; an unknown contract is
        rejected rather than silently interpreted as metres.
        """
        unit = self.model.ugv.parameters.firmware_odometry_unit
        if unit is None:
            raise ValueError(
                "UGV firmware odometry unit is unknown; select a verified "
                "firmware_profile or configure firmware_odometry_unit")
        scale = {"m": 1.0, "cm": 0.01}[unit]
        update = self.update_cumulative_wheel_distances(
            float(odl) * scale, float(odr) * scale,
            joint_angles=joint_angles, arm_pose=arm_pose)
        if update is None:
            return None
        sources = dict(update.sources)
        sources["base"] = "firmware_odometry_{}".format(unit)
        return OdometryUpdate(
            update.base_pose_before, update.base_pose_after,
            update.base_delta_body, update.joint_angles_before,
            update.joint_angles_after, update.joint_delta,
            update.camera_pose_before, update.camera_pose_after,
            update.camera_from_base, update.camera_from_arm,
            update.camera_total, sources, update.calibrated)

    @staticmethod
    def _arm_source(joint_angles, arm_pose):
        if arm_pose is not None:
            return "xyzt_ik"
        if joint_angles is not None:
            return "joint_feedback"
        return "unchanged"
