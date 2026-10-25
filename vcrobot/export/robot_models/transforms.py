"""Small, dependency-light SE(3) helpers.

Convention: ``T_XY`` maps points expressed in frame Y into frame X.  Vectors
are column vectors and transforms compose left-to-right, e.g. ``T_WB @ T_BC``.
Angles are radians and translations are metres.
"""

import math
from typing import Iterable

import numpy as np


def _vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError("{} must contain {} finite values".format(name, size))
    return result


def skew(vector: Iterable[float]) -> np.ndarray:
    x, y, z = _vector(vector, 3, "vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the fixed-axis roll/pitch/yaw rotation ``Rz(yaw)Ry(pitch)Rx(roll)``."""
    return rotation_z(yaw).dot(rotation_y(pitch)).dot(rotation_x(roll))


def make_transform(rotation=None, translation=None) -> np.ndarray:
    result = np.eye(4, dtype=float)
    if rotation is not None:
        result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    if translation is not None:
        result[:3, 3] = _vector(translation, 3, "translation")
    validate_transform(result)
    return result


def pose2d_transform(state: Iterable[float]) -> np.ndarray:
    x, y, yaw = _vector(state, 3, "state")
    return make_transform(rotation_z(yaw), (x, y, 0.0))


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = validate_transform(transform)
    rotation = transform[:3, :3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T.dot(transform[:3, 3])
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = validate_transform(transform)
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1
    points = points.reshape(-1, 3)
    result = points.dot(transform[:3, :3].T) + transform[:3, 3]
    return result[0] if single else result


def validate_rotation(rotation: np.ndarray, atol: float = 1e-7) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T.dot(rotation), np.eye(3), atol=atol):
        raise ValueError("rotation matrix is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=atol):
        raise ValueError("rotation matrix determinant must be +1")
    return rotation


def validate_transform(transform: np.ndarray, atol: float = 1e-7) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("invalid homogeneous transform bottom row")
    validate_rotation(transform[:3, :3], atol=atol)
    return transform


def rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    delta = validate_rotation(left).T.dot(validate_rotation(right))
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return math.acos(cosine)


def transform_from_config(value) -> np.ndarray:
    """Parse a 4x4 matrix or ``{translation_m, rpy_deg/rpy_rad}`` mapping."""
    if isinstance(value, dict):
        if "matrix" in value:
            return validate_transform(np.asarray(value["matrix"], dtype=float))
        translation = value.get("translation_m", (0.0, 0.0, 0.0))
        if "rpy_deg" in value:
            rpy = np.radians(_vector(value["rpy_deg"], 3, "rpy_deg"))
        else:
            rpy = _vector(value.get("rpy_rad", (0.0, 0.0, 0.0)), 3, "rpy_rad")
        return make_transform(rpy_matrix(*rpy), translation)
    return validate_transform(np.asarray(value, dtype=float))


def transform_to_config(transform: np.ndarray):
    transform = validate_transform(transform)
    return [[float(item) for item in row] for row in transform]
