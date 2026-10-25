"""RoArm-M2-S kinematics ported from the official ESP32 firmware.

The arm frame A is deliberately the firmware Cartesian-command frame: its
origin is on the base-yaw axis at shoulder height, +X is forward at base angle
zero, +Y is left, and +Z is up.  The physical base-link-to-A offset belongs in
``T_BA`` and must be measured on the assembled robot.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .transforms import make_transform


@dataclass(frozen=True)
class RoArmParameters:
    # Official RoArm-M2_config.h values, converted from mm to m.
    base_to_shoulder_m: float = 0.12606
    l2_a_m: float = 0.23682
    l2_b_m: float = 0.03000
    l3_a_mode0_m: float = 0.28015
    l3_b_mode0_m: float = 0.00173
    l3_a_mode1_m: float = 0.21599
    l3_b_mode1_m: float = 0.0
    l4_a_m: float = 0.06785
    l4_b_m: float = 0.00598
    eoat_a_m: float = 0.0
    eoat_b_m: float = 0.0
    joint_min_rad: Tuple[float, float, float, float] = (
        -math.pi, -math.pi / 2.0, -512.0 * 2.0 * math.pi / 4096.0,
        700.0 * 2.0 * math.pi / 4096.0)
    joint_max_rad: Tuple[float, float, float, float] = (
        math.pi, math.pi / 2.0, 2047.0 * 2.0 * math.pi / 4096.0,
        3396.0 * 2.0 * math.pi / 4096.0)


@dataclass(frozen=True)
class IKResult:
    success: bool
    joint_angles: Optional[np.ndarray]
    reason: str = ""
    position_error_m: float = math.inf
    orientation_limited: bool = False
    clamped: bool = False


class RoArmModel:
    """Firmware-compatible 4-DOF model for EEMode 0 (gripper) or 1 (wrist)."""

    def __init__(self, eemode: int = 0, parameters: Optional[RoArmParameters] = None):
        if int(eemode) not in (0, 1):
            raise ValueError("eemode must be 0 (gripper) or 1 (wrist)")
        self.eemode = int(eemode)
        self.parameters = parameters or RoArmParameters()

    @property
    def joint_limits(self) -> np.ndarray:
        return np.array([self.parameters.joint_min_rad,
                         self.parameters.joint_max_rad], dtype=float).T

    def _lengths(self):
        p = self.parameters
        l2 = math.hypot(p.l2_a_m, p.l2_b_m)
        t2 = math.atan2(p.l2_b_m, p.l2_a_m)
        if self.eemode == 0:
            l3a, l3b = p.l3_a_mode0_m, p.l3_b_mode0_m
        else:
            l3a, l3b = p.l3_a_mode1_m, p.l3_b_mode1_m
        l3 = math.hypot(l3a, l3b)
        t3 = math.atan2(l3b, l3a)
        lea, leb = p.eoat_a_m + p.l4_a_m, p.eoat_b_m + p.l4_b_m
        le = math.hypot(lea, leb)
        te = math.atan2(leb, lea)
        return l2, t2, l3, t3, le, te

    @staticmethod
    def _joints(joint_angles: Sequence[float]) -> np.ndarray:
        joints = np.asarray(joint_angles, dtype=float).reshape(-1)
        if joints.size != 4 or not np.all(np.isfinite(joints)):
            raise ValueError("joint_angles must contain four finite radians")
        return joints

    def check_joint_limits(self, joint_angles: Sequence[float], atol: float = 1e-9) -> bool:
        joints = self._joints(joint_angles)
        limits = self.joint_limits
        return bool(np.all(joints >= limits[:, 0] - atol) and
                    np.all(joints <= limits[:, 1] + atol))

    def firmware_pose(self, joint_angles: Sequence[float]) -> np.ndarray:
        """Return the firmware ``[x_mm, y_mm, z_mm, t_rad]`` feedback values."""
        q_base, q_shoulder, q_elbow, q_wrist = self._joints(joint_angles)
        l2, t2, l3, t3, le, te = self._lengths()
        theta2 = math.pi / 2.0 - (q_shoulder + t2)
        # The EEMode=0 expression intentionally omits t3, exactly as firmware.
        theta3 = math.pi / 2.0 - (q_elbow + q_shoulder +
                                  (t3 if self.eemode == 1 else 0.0))
        radial = l2 * math.cos(theta2) + l3 * math.cos(theta3)
        height = l2 * math.sin(theta2) + l3 * math.sin(theta3)
        if self.eemode == 1:
            theta_e = (3.0 * math.pi / 2.0 - q_wrist - te -
                       q_shoulder - q_elbow)
            radial += le * math.cos(theta_e)
            height += le * math.sin(theta_e)
            tool_angle = q_wrist + q_shoulder + q_elbow - math.pi / 2.0
        else:
            tool_angle = q_wrist
        return np.array([1000.0 * radial * math.cos(q_base),
                         1000.0 * radial * math.sin(q_base),
                         1000.0 * height, tool_angle], dtype=float)

    def forward_kinematics(self, joint_angles: Sequence[float]) -> np.ndarray:
        """Return ``T_AE`` in metres.

        Position is an exact port of ``RoArmM2_computePosbyJointRad``.  The
        firmware does not publish an end-frame rotation, so the rotation below
        is the documented geometric frame: +X follows the distal/tool link,
        +Y is tangential about the base axis, and +Z completes a right-handed
        frame.  EEMode 0's gripper command does not rotate this frame.
        """
        q_base, q_shoulder, q_elbow, q_wrist = self._joints(joint_angles)
        pose = self.firmware_pose((q_base, q_shoulder, q_elbow, q_wrist))
        _, _, _, t3, _, te = self._lengths()
        if self.eemode == 1:
            link_pitch = (3.0 * math.pi / 2.0 - q_wrist - te -
                          q_shoulder - q_elbow)
        else:
            link_pitch = math.pi / 2.0 - q_shoulder - q_elbow - t3
        x_axis = np.array([math.cos(q_base) * math.cos(link_pitch),
                           math.sin(q_base) * math.cos(link_pitch),
                           math.sin(link_pitch)])
        y_axis = np.array([-math.sin(q_base), math.cos(q_base), 0.0])
        z_axis = np.cross(x_axis, y_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        return make_transform(rotation, pose[:3] / 1000.0)

    def joint_positions(self, joint_angles: Sequence[float]) -> np.ndarray:
        """Return shoulder, elbow, distal-link and end points for proxy plots."""
        q_base, q_shoulder, q_elbow, q_wrist = self._joints(joint_angles)
        l2, t2, l3, t3, le, te = self._lengths()
        radial_dir = np.array([math.cos(q_base), math.sin(q_base), 0.0])
        shoulder = np.zeros(3)
        theta2 = math.pi / 2.0 - (q_shoulder + t2)
        elbow = shoulder + radial_dir * (l2 * math.cos(theta2))
        elbow[2] += l2 * math.sin(theta2)
        theta3 = math.pi / 2.0 - q_shoulder - q_elbow - (
            t3 if self.eemode == 1 else 0.0)
        distal = elbow + radial_dir * (l3 * math.cos(theta3))
        distal[2] += l3 * math.sin(theta3)
        if self.eemode == 0:
            end = distal
        else:
            theta_e = 3.0 * math.pi / 2.0 - q_wrist - te - q_shoulder - q_elbow
            end = distal + radial_dir * (le * math.cos(theta_e))
            end[2] += le * math.sin(theta_e)
        return np.vstack((shoulder, elbow, distal, end))

    @staticmethod
    def _acos(value: float) -> float:
        if value < -1.0 - 1e-9 or value > 1.0 + 1e-9:
            raise ValueError("target is outside planar linkage workspace")
        return math.acos(max(-1.0, min(1.0, value)))

    def _simple_linkage_ik(self, radial: float, height: float):
        l2, t2, l3, t3, _, _ = self._lengths()
        if radial < 0.0:
            raise ValueError("radial distance must be non-negative")
        distance2 = radial * radial + height * height
        distance = math.sqrt(distance2)
        if distance < 1e-9:
            raise ValueError("target at arm origin is singular")
        lam = math.atan2(height, radial)
        psi = self._acos((l2 * l2 + distance2 - l3 * l3) /
                         (2.0 * l2 * distance)) + t2
        alpha = math.pi / 2.0 - lam - psi
        omega = self._acos((l3 * l3 + distance2 - l2 * l2) /
                           (2.0 * distance * l3))
        beta = psi + omega - t3
        delta = math.pi / 2.0 - alpha - beta
        return alpha, beta, delta

    def _target_xyzt(self, target_pose, tool_angle: Optional[float]):
        orientation_limited = False
        if isinstance(target_pose, Mapping):
            scale = 0.001 if any(key.endswith("_mm") for key in target_pose) else 1.0
            if scale == 0.001:
                x, y, z = (float(target_pose[key]) * scale
                           for key in ("x_mm", "y_mm", "z_mm"))
            else:
                x, y, z = (float(target_pose[key]) for key in ("x", "y", "z"))
            default_angle = (math.pi if self.eemode == 0 else
                             float(tool_angle or 0.0))
            angle = float(target_pose.get(
                "t", target_pose.get("tool_angle_rad", default_angle)))
            return x, y, z, angle, orientation_limited
        value = np.asarray(target_pose, dtype=float)
        if value.shape == (4, 4):
            x, y, z = value[:3, 3]
            radial = np.array([math.cos(math.atan2(y, x)),
                               math.sin(math.atan2(y, x)), 0.0])
            link_x = value[:3, 0]
            link_pitch = math.atan2(link_x[2], float(np.dot(link_x, radial)))
            if self.eemode == 1:
                _, _, _, _, _, te = self._lengths()
                angle = math.pi - link_pitch - te
            else:
                angle = float(math.pi if tool_angle is None else tool_angle)
                orientation_limited = True
            return float(x), float(y), float(z), angle, orientation_limited
        flat = value.reshape(-1)
        if flat.size not in (3, 4):
            raise ValueError("target_pose must be a 3/4-vector, mapping, or 4x4 matrix")
        default_angle = math.pi if self.eemode == 0 else float(tool_angle or 0.0)
        angle = float(flat[3]) if flat.size == 4 else default_angle
        return float(flat[0]), float(flat[1]), float(flat[2]), angle, orientation_limited

    def solve_ik(self, target_pose, tool_angle: Optional[float] = None,
                 position_tolerance_m: float = 0.004) -> IKResult:
        try:
            x, y, z, target_t, orientation_limited = self._target_xyzt(
                target_pose, tool_angle)
            base = math.atan2(y, x)
            radial = math.hypot(x, y)
            if self.eemode == 1:
                _, _, _, _, le, te = self._lengths()
                delta_x = le * math.cos(te + target_t - math.pi)
                delta_z = le * math.sin(te + target_t - math.pi)
                if radial - delta_x <= 1e-9:
                    radial_without_tool = 0.0
                else:
                    radial_without_tool = radial - delta_x
                shoulder, elbow, wrist_buffer = self._simple_linkage_ik(
                    radial_without_tool, z + delta_z)
                wrist = wrist_buffer + target_t
            else:
                shoulder, elbow, _ = self._simple_linkage_ik(radial, z)
                wrist = target_t
            joints = np.array([base, shoulder, elbow, wrist], dtype=float)
            if not self.check_joint_limits(joints):
                return IKResult(False, joints, "solution violates firmware joint limits",
                                orientation_limited=orientation_limited)
            achieved = self.forward_kinematics(joints)[:3, 3]
            error = float(np.linalg.norm(achieved - np.array([x, y, z])))
            if error > position_tolerance_m:
                return IKResult(False, joints,
                                "firmware FK/IK convention residual exceeds tolerance",
                                error, orientation_limited)
            return IKResult(True, joints, "", error, orientation_limited)
        except (KeyError, TypeError, ValueError) as exc:
            return IKResult(False, None, str(exc))

    def inverse_kinematics(self, target_pose, tool_angle: Optional[float] = None) -> np.ndarray:
        result = self.solve_ik(target_pose, tool_angle=tool_angle)
        if not result.success:
            raise ValueError(result.reason)
        return result.joint_angles.copy()

    def solve_ik_firmware(self, target_pose, tool_angle: Optional[float] = None):
        """Emulate coordinate-command acceptance and per-servo constraints.

        Firmware computes IK first and then constrains each servo in the joint
        control functions. It does not reject a finite IK solution merely
        because one of its joints is outside that range. This method preserves
        that command behaviour and reports the pose reached after clamping.
        ``solve_ik`` remains the strict reachability check.
        """
        result = self.solve_ik(target_pose, tool_angle=tool_angle)
        if result.success:
            return result
        if result.joint_angles is None:
            return result
        try:
            x, y, z, _, orientation_limited = self._target_xyzt(
                target_pose, tool_angle)
            limits = self.joint_limits
            joints = np.clip(result.joint_angles, limits[:, 0], limits[:, 1])
            clamped_mask = np.abs(joints - result.joint_angles) > 1e-10
            achieved = self.forward_kinematics(joints)[:3, 3]
            error = float(np.linalg.norm(achieved - np.array([x, y, z])))
            joint_names = ("base", "shoulder", "elbow", "hand")
            names = [name for name, changed in zip(joint_names, clamped_mask)
                     if changed]
            reason = "firmware servo clamp applied: " + ", ".join(names)
            return IKResult(True, joints, reason, error,
                            orientation_limited, bool(names))
        except (KeyError, TypeError, ValueError) as exc:
            return IKResult(False, None, str(exc))

    def check_reachability(self, target_pose, tool_angle: Optional[float] = None) -> bool:
        return self.solve_ik(target_pose, tool_angle=tool_angle).success

    is_reachable = check_reachability

    def sample_workspace(self, num_samples: int, seed: int = 0) -> np.ndarray:
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        rng = np.random.RandomState(seed)
        limits = self.joint_limits
        joints = rng.uniform(limits[:, 0], limits[:, 1], size=(int(num_samples), 4))
        return np.vstack([self.forward_kinematics(q)[:3, 3] for q in joints])
