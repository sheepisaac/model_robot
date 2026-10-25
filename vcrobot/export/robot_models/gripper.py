"""Camera mounting model for the moving RoArm EEMode-0 gripper member.

The stock RoArm FK intentionally stops at frame E because the firmware treats
the fourth EEMode-0 value as a gripper command.  On this robot the L515 is
attached to the moving member, so the remaining transform is

    T_EC(t) = T_EG(t) @ T_GC

This module keeps that custom mechanism separate from the firmware-compatible
arm model.  Nominal motion parameters must never be mistaken for calibration.
"""

import math
from typing import Mapping, Sequence

import numpy as np

from .transforms import make_transform, transform_from_config, validate_transform


def _finite_scalar(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(-1)
    if axis.size != 3 or not np.all(np.isfinite(axis)):
        raise ValueError("rotation axis must contain three finite values")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("rotation axis must be non-zero")
    x, y, z = axis / norm
    angle = _finite_scalar(angle, "angle")
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [c + x * x * one_minus_c,
         x * y * one_minus_c - z * s,
         x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s,
         c + y * y * one_minus_c,
         y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s,
         z * y * one_minus_c + x * s,
         c + z * z * one_minus_c],
    ], dtype=float)


class GripperCameraMount:
    """Return the end-to-camera transform for a gripper command ``t``.

    Supported modes:

    ``fixed``
        Backward-compatible fixed ``T_EC``.
    ``revolute``
        Continuous one-axis nominal/calibrated model.  The moving frame is
        ``T_EG_reference @ Rot(axis, radians_per_t * (t-reference_t))``.
    ``lookup``
        Discrete calibrated ``t -> T_EG`` samples.  Only a sample within
        ``lookup_tolerance_rad`` is accepted; no physically unjustified SE(3)
        interpolation is performed.
    """

    MODES = {"fixed", "revolute", "lookup"}

    def __init__(self, mode="fixed", T_GC=None, T_EG_reference=None,
                 reference_t_rad=math.pi, axis=(0.0, 1.0, 0.0),
                 radians_per_t=1.0, samples=None,
                 lookup_tolerance_rad=1e-3,
                 motion_status="UNCALIBRATED",
                 t_gc_status="UNCALIBRATED"):
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError("camera mount mode must be fixed, revolute, or lookup")
        self.T_GC = validate_transform(
            np.eye(4) if T_GC is None else np.asarray(T_GC, dtype=float)).copy()
        self.T_EG_reference = validate_transform(
            np.eye(4) if T_EG_reference is None else
            np.asarray(T_EG_reference, dtype=float)).copy()
        self.reference_t_rad = _finite_scalar(reference_t_rad, "reference_t_rad")
        self.axis = np.asarray(axis, dtype=float).reshape(-1)
        if self.axis.size != 3 or not np.all(np.isfinite(self.axis)):
            raise ValueError("axis must contain three finite values")
        axis_norm = float(np.linalg.norm(self.axis))
        if axis_norm <= 1e-12:
            raise ValueError("axis must be non-zero")
        self.axis = self.axis / axis_norm
        self.radians_per_t = _finite_scalar(radians_per_t, "radians_per_t")
        self.lookup_tolerance_rad = _finite_scalar(
            lookup_tolerance_rad, "lookup_tolerance_rad")
        if self.lookup_tolerance_rad < 0.0:
            raise ValueError("lookup_tolerance_rad must be non-negative")
        self.motion_status = str(motion_status).upper()
        self.t_gc_status = str(t_gc_status).upper()

        parsed_samples = []
        for sample in samples or ():
            t_rad, transform = sample
            parsed_samples.append((
                _finite_scalar(t_rad, "sample t_rad"),
                validate_transform(np.asarray(transform, dtype=float)).copy()))
        parsed_samples.sort(key=lambda item: item[0])
        if len({item[0] for item in parsed_samples}) != len(parsed_samples):
            raise ValueError("lookup sample t values must be unique")
        if self.mode == "lookup" and not parsed_samples:
            raise ValueError("lookup mode requires at least one t-to-T_EG sample")
        self.samples = tuple(parsed_samples)

    @property
    def dynamic(self):
        return self.mode != "fixed"

    @property
    def calibrated(self):
        return (self.motion_status == "CALIBRATED" and
                self.t_gc_status == "CALIBRATED")

    def gripper_transform(self, t_rad):
        """Return ``T_EG(t)``."""
        t_rad = _finite_scalar(t_rad, "t_rad")
        if self.mode == "fixed":
            return self.T_EG_reference.copy()
        if self.mode == "revolute":
            angle = self.radians_per_t * (t_rad - self.reference_t_rad)
            rotation = make_transform(_axis_rotation(self.axis, angle))
            return self.T_EG_reference.dot(rotation)

        nearest = min(self.samples, key=lambda item: abs(item[0] - t_rad))
        if abs(nearest[0] - t_rad) > self.lookup_tolerance_rad:
            raise ValueError(
                "t={:.6f} rad has no calibrated lookup sample within {:.6f} rad"
                .format(t_rad, self.lookup_tolerance_rad))
        return nearest[1].copy()

    def camera_transform(self, t_rad):
        """Return ``T_EC(t) = T_EG(t) @ T_GC``."""
        return self.gripper_transform(t_rad).dot(self.T_GC)

    def candidate_t_values(self, joint_limits, count=65):
        """Return deterministic ``t`` candidates for camera-pose IK."""
        limits = np.asarray(joint_limits, dtype=float).reshape(2)
        if self.mode == "lookup":
            return np.array([item[0] for item in self.samples
                             if limits[0] <= item[0] <= limits[1]], dtype=float)
        if self.mode == "fixed":
            return np.array([min(max(self.reference_t_rad, limits[0]), limits[1])])
        values = np.linspace(limits[0], limits[1], max(3, int(count)))
        reference = min(max(self.reference_t_rad, limits[0]), limits[1])
        return np.unique(np.append(values, reference))

    @classmethod
    def fixed(cls, T_EC, status="UNCALIBRATED", reference_t_rad=math.pi):
        return cls(mode="fixed", T_GC=T_EC,
                   reference_t_rad=reference_t_rad,
                   motion_status="CALIBRATED", t_gc_status=status)

    @classmethod
    def from_config(cls, config: Mapping, t_gc_config):
        config = dict(config or {})
        mode = str(config.get("mode", "fixed")).lower()
        t_gc_status = (str(t_gc_config.get("status", "UNCALIBRATED")).upper()
                       if isinstance(t_gc_config, Mapping) else "UNCALIBRATED")
        T_GC = transform_from_config(t_gc_config)
        reference = transform_from_config(
            config.get("T_EG_reference", np.eye(4)))
        samples = []
        for sample in config.get("samples", []):
            if "t_rad" not in sample or "T_EG" not in sample:
                raise ValueError("each gripper lookup sample needs t_rad and T_EG")
            samples.append((float(sample["t_rad"]),
                            transform_from_config(sample["T_EG"])))
        return cls(
            mode=mode,
            T_GC=T_GC,
            T_EG_reference=reference,
            reference_t_rad=config.get("reference_t_rad", math.pi),
            axis=config.get("axis", (0.0, 1.0, 0.0)),
            radians_per_t=config.get("radians_per_t", 1.0),
            samples=samples,
            lookup_tolerance_rad=config.get("lookup_tolerance_rad", 1e-3),
            motion_status=config.get("status", "UNCALIBRATED"),
            t_gc_status=t_gc_status,
        )

