"""Calibratable planar UGV02 differential/skid-steer approximation."""

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class UGVParameters:
    wheel_diameter_m: float = 0.0800
    encoder_pulses_per_rev: int = 660
    effective_track_width_m: float = 0.172
    left_scale: float = 1.0
    right_scale: float = 1.0
    calibrated: bool = False


class UGVModel:
    """Exact constant-twist integration for the initial skid-steer model."""

    def __init__(self, parameters: Optional[UGVParameters] = None):
        self.parameters = parameters or UGVParameters()
        if self.parameters.effective_track_width_m <= 0.0:
            raise ValueError("effective_track_width_m must be positive")

    @staticmethod
    def _state(value: Iterable[float]) -> np.ndarray:
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.size != 3 or not np.all(np.isfinite(result)):
            raise ValueError("state must be finite [x, y, yaw]")
        return result

    @staticmethod
    def normalize_yaw(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def body_twist(self, v_left: float, v_right: float):
        left = self.parameters.left_scale * float(v_left)
        right = self.parameters.right_scale * float(v_right)
        linear = 0.5 * (right + left)
        angular = (right - left) / self.parameters.effective_track_width_m
        return linear, angular

    def predict_motion(self, state: Iterable[float], v_left: float,
                       v_right: float, duration: float) -> np.ndarray:
        state = self._state(state)
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be finite and non-negative")
        linear, angular = self.body_twist(v_left, v_right)
        x, y, yaw = state
        if abs(angular) < 1e-10:
            x += linear * duration * math.cos(yaw)
            y += linear * duration * math.sin(yaw)
        else:
            new_yaw = yaw + angular * duration
            radius = linear / angular
            x += radius * (math.sin(new_yaw) - math.sin(yaw))
            y -= radius * (math.cos(new_yaw) - math.cos(yaw))
        yaw = self.normalize_yaw(yaw + angular * duration)
        return np.array([x, y, yaw], dtype=float)

    def integrate_odometry(self, state: Iterable[float], wheel_feedback,
                           dt: Optional[float] = None) -> np.ndarray:
        """Integrate velocity feedback or cumulative wheel distances.

        A mapping with ``v_left``/``v_right`` requires ``dt``.  A mapping with
        ``delta_left_m``/``delta_right_m`` is interpreted as wheel travel over
        one sample and does not use scale factors a second time.
        """
        if isinstance(wheel_feedback, Mapping):
            if "delta_left_m" in wheel_feedback:
                left = float(wheel_feedback["delta_left_m"])
                right = float(wheel_feedback["delta_right_m"])
                return self._integrate_distances(state, left, right)
            left = float(wheel_feedback["v_left"])
            right = float(wheel_feedback["v_right"])
        else:
            left, right = np.asarray(wheel_feedback, dtype=float).reshape(2)
        if dt is None:
            raise ValueError("dt is required for wheel velocity feedback")
        return self.predict_motion(state, left, right, dt)

    def _integrate_distances(self, state, left_distance, right_distance):
        p = self.parameters
        # Distances from encoder feedback are already physical measurements.
        duration = 1.0
        temporary = UGVModel(UGVParameters(
            wheel_diameter_m=p.wheel_diameter_m,
            encoder_pulses_per_rev=p.encoder_pulses_per_rev,
            effective_track_width_m=p.effective_track_width_m,
            left_scale=1.0, right_scale=1.0, calibrated=p.calibrated))
        return temporary.predict_motion(state, left_distance, right_distance, duration)

    def encoder_counts_to_distance(self, counts: float) -> float:
        return (float(counts) / self.parameters.encoder_pulses_per_rev *
                math.pi * self.parameters.wheel_diameter_m)

    @classmethod
    def estimate_motion_parameters(cls, calibration_dataset: Sequence[Mapping]):
        """Fit left/right speed scales and effective track width.

        Rows require ``v_left``, ``v_right``, ``duration``, ``distance_m`` and
        ``yaw_delta_rad``.  ``distance_m`` is signed forward arc length.
        The linear equation estimates speed scales; non-degenerate turns then
        estimate the effective track width by least squares.
        """
        rows = list(calibration_dataset)
        if len(rows) < 3:
            raise ValueError("at least three calibration rows are required")
        design, target = [], []
        for row in rows:
            dt = float(row["duration"])
            if dt <= 0.0:
                raise ValueError("duration must be positive")
            design.append([0.5 * float(row["v_left"]) * dt,
                           0.5 * float(row["v_right"]) * dt])
            target.append(float(row["distance_m"]))
        scales, _, rank, _ = np.linalg.lstsq(np.asarray(design),
                                             np.asarray(target), rcond=None)
        if rank < 2 or np.any(scales <= 0.0):
            raise ValueError("calibration data cannot identify positive wheel scales")
        numerators, yaws = [], []
        for row in rows:
            dt = float(row["duration"])
            numerator = (scales[1] * float(row["v_right"]) -
                         scales[0] * float(row["v_left"])) * dt
            yaw = float(row["yaw_delta_rad"])
            if abs(numerator) > 1e-6 and abs(yaw) > 1e-6:
                numerators.append(numerator)
                yaws.append(yaw)
        if not numerators:
            raise ValueError("turning samples are required to identify track width")
        numerators = np.asarray(numerators)
        yaws = np.asarray(yaws)
        inverse_track = float(np.dot(numerators, yaws) /
                              np.dot(numerators, numerators))
        if inverse_track <= 0.0:
            raise ValueError("fitted track width is non-positive; check wheel signs")
        parameters = UGVParameters(effective_track_width_m=1.0 / inverse_track,
                                   left_scale=float(scales[0]),
                                   right_scale=float(scales[1]), calibrated=True)
        return cls(parameters)
