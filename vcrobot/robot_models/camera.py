"""ROS- and device-independent RealSense-compatible camera geometry."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from .transforms import transform_points, validate_transform


_BROWN_MODELS = {"brown_conrady", "modified_brown_conrady",
                 "inverse_brown_conrady"}


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str = "none"
    coeffs: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_mapping(cls, value):
        coeffs = tuple(float(item) for item in value.get("coeffs", [0.0] * 5))
        if len(coeffs) != 5:
            raise ValueError("camera coeffs must contain five values")
        return cls(int(value["width"]), int(value["height"]),
                   float(value["fx"]), float(value["fy"]),
                   float(value["ppx"]), float(value["ppy"]),
                   str(value.get("model", "none")).lower(), coeffs)


def validate_camera_intrinsics(intrinsics: CameraIntrinsics) -> bool:
    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("intrinsics must be CameraIntrinsics")
    values = [intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy]
    if intrinsics.width <= 0 or intrinsics.height <= 0:
        raise ValueError("camera resolution must be positive")
    if not all(math.isfinite(item) for item in values + list(intrinsics.coeffs)):
        raise ValueError("camera parameters must be finite")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    supported = {"none"}.union(_BROWN_MODELS)
    if intrinsics.model not in supported:
        raise ValueError("unsupported distortion model: {}".format(intrinsics.model))
    return True


def _brown_forward(x: float, y: float, coeffs):
    k1, k2, p1, p2, k3 = coeffs
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    return (x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y))


def _brown_inverse(xd: float, yd: float, coeffs):
    x, y = xd, yd
    for _ in range(12):
        estimate_x, estimate_y = _brown_forward(x, y, coeffs)
        x += xd - estimate_x
        y += yd - estimate_y
    return x, y


def project_point(point_3d: Iterable[float], intrinsics: CameraIntrinsics) -> np.ndarray:
    validate_camera_intrinsics(intrinsics)
    x, y, z = np.asarray(point_3d, dtype=float).reshape(3)
    if not np.all(np.isfinite([x, y, z])) or z <= 0.0:
        raise ValueError("point must be finite and in front of the camera")
    x, y = x / z, y / z
    if intrinsics.model in ("brown_conrady", "modified_brown_conrady"):
        x, y = _brown_forward(x, y, intrinsics.coeffs)
    elif intrinsics.model == "inverse_brown_conrady":
        x, y = _brown_inverse(x, y, intrinsics.coeffs)
    return np.array([x * intrinsics.fx + intrinsics.ppx,
                     y * intrinsics.fy + intrinsics.ppy], dtype=float)


def deproject_pixel(pixel: Iterable[float], depth: float,
                    intrinsics: CameraIntrinsics) -> np.ndarray:
    validate_camera_intrinsics(intrinsics)
    u, v = np.asarray(pixel, dtype=float).reshape(2)
    depth = float(depth)
    if not np.all(np.isfinite([u, v, depth])) or depth <= 0.0:
        raise ValueError("pixel and depth must be finite; depth must be positive")
    x = (u - intrinsics.ppx) / intrinsics.fx
    y = (v - intrinsics.ppy) / intrinsics.fy
    if intrinsics.model in ("brown_conrady", "modified_brown_conrady"):
        x, y = _brown_inverse(x, y, intrinsics.coeffs)
    elif intrinsics.model == "inverse_brown_conrady":
        x, y = _brown_forward(x, y, intrinsics.coeffs)
    return np.array([depth * x, depth * y, depth], dtype=float)


def transform_depth_to_color(point, extrinsics) -> np.ndarray:
    return transform_points(validate_transform(extrinsics), np.asarray(point, dtype=float))


class L515CameraModel:
    """Projection wrapper; no librealsense or physical camera is required."""

    def __init__(self, intrinsics: CameraIntrinsics,
                 status: str = "UNCALIBRATED"):
        validate_camera_intrinsics(intrinsics)
        self.intrinsics = intrinsics
        self.status = str(status).upper()

    @property
    def calibrated(self) -> bool:
        return self.status == "CALIBRATED"

    def project(self, point_3d):
        return project_point(point_3d, self.intrinsics)

    project_point = project

    def deproject(self, pixel, depth):
        return deproject_pixel(pixel, depth, self.intrinsics)

    deproject_pixel = deproject

    @staticmethod
    def transform_depth_to_color(point, extrinsics):
        return transform_depth_to_color(point, extrinsics)

    def visible(self, point_3d, margin_px: float = 0.0) -> bool:
        try:
            pixel = self.project(point_3d)
        except ValueError:
            return False
        return bool(margin_px <= pixel[0] < self.intrinsics.width - margin_px and
                    margin_px <= pixel[1] < self.intrinsics.height - margin_px)
