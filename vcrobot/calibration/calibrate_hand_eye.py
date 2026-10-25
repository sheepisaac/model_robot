"""Estimate fixed-t eye-in-hand T_EC and reject weak motion sets.

For the moving-camera gripper this is an intermediate calibration result.
Deriving T_GC additionally requires a calibrated T_EG at the fixed t value.
"""

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_models.transforms import (rotation_angle, transform_to_config,
                                     validate_transform)


def _rotation_vector(rotation):
    vector, _ = cv2.Rodrigues(rotation)
    return vector.reshape(3)


def validate_observability(transforms, minimum_rotation_deg=5.0):
    if len(transforms) < 4:
        raise ValueError("at least four diverse poses are required")
    relative_vectors = []
    for index in range(1, len(transforms)):
        relative = np.linalg.inv(transforms[0]).dot(transforms[index])
        vector = _rotation_vector(relative[:3, :3])
        if np.linalg.norm(vector) >= math.radians(minimum_rotation_deg):
            relative_vectors.append(vector)
    if len(relative_vectors) < 3:
        raise ValueError("insufficient rotational diversity for hand-eye calibration")
    matrix = np.vstack(relative_vectors)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] < 0.02:
        raise ValueError("hand-eye rotations are nearly rank deficient")
    return singular


def calibrate(samples):
    T_BG = [validate_transform(np.asarray(item["T_BG"], dtype=float))
            for item in samples]
    T_CT = [validate_transform(np.asarray(item["T_CT"], dtype=float))
            for item in samples]
    singular = validate_observability(T_BG)
    rotation, translation = cv2.calibrateHandEye(
        [item[:3, :3] for item in T_BG], [item[:3, 3] for item in T_BG],
        [item[:3, :3] for item in T_CT], [item[:3, 3] for item in T_CT],
        method=cv2.CALIB_HAND_EYE_TSAI)
    T_EC = np.eye(4)
    T_EC[:3, :3] = rotation
    T_EC[:3, 3] = np.asarray(translation).reshape(3)
    return validate_transform(T_EC), singular


def calibration_residuals(samples, T_EC):
    """Measure fixed-target consistency after hand-eye calibration.

    For each sample, ``T_BG @ T_EC @ T_CT`` should produce the same target
    pose in the base frame.  The returned residuals validate the result; they
    are separate from the numerical calibration itself.
    """
    T_EC = validate_transform(np.asarray(T_EC, dtype=float))
    target_poses = []
    for item in samples:
        T_BG = validate_transform(np.asarray(item["T_BG"], dtype=float))
        T_CT = validate_transform(np.asarray(item["T_CT"], dtype=float))
        target_poses.append(T_BG.dot(T_EC).dot(T_CT))
    translations = np.vstack([item[:3, 3] for item in target_poses])
    translation_center = np.mean(translations, axis=0)
    translation_errors = np.linalg.norm(
        translations - translation_center, axis=1)
    reference_rotation = target_poses[0][:3, :3]
    rotation_errors = np.array([
        rotation_angle(reference_rotation, item[:3, :3])
        for item in target_poses])
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_rmse_deg": float(math.degrees(
            np.sqrt(np.mean(rotation_errors ** 2)))),
        "rotation_max_deg": float(math.degrees(np.max(rotation_errors))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_yaml")
    parser.add_argument("output_yaml")
    parser.add_argument("--max-translation-rmse-m", type=float, default=0.01)
    parser.add_argument("--max-rotation-rmse-deg", type=float, default=2.0)
    args = parser.parse_args(argv)
    with Path(args.input_yaml).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result, singular = calibrate(config["hand_eye"]["samples"])
    residuals = calibration_residuals(config["hand_eye"]["samples"], result)
    accepted = (
        residuals["translation_rmse_m"] <= args.max_translation_rmse_m and
        residuals["rotation_rmse_deg"] <= args.max_rotation_rmse_deg)
    output = {"extrinsics": {"T_EC_at_fixed_t": {
        "status": "CALIBRATED" if accepted else "REJECTED",
        "matrix": transform_to_config(result),
        "source": str(args.input_yaml),
        "fixed_t_rad": config["hand_eye"].get("fixed_t_rad"),
        "rotation_singular_values": [float(item) for item in singular],
        "validation": residuals,
        "thresholds": {
            "max_translation_rmse_m": args.max_translation_rmse_m,
            "max_rotation_rmse_deg": args.max_rotation_rmse_deg,
        }}}}
    with Path(args.output_yaml).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)


if __name__ == "__main__":
    main()
