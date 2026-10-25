"""Estimate eye-in-hand T_EC with OpenCV and reject weak motion sets."""

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

from robot_models.transforms import transform_to_config, validate_transform


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_yaml")
    parser.add_argument("output_yaml")
    args = parser.parse_args(argv)
    with Path(args.input_yaml).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result, singular = calibrate(config["hand_eye"]["samples"])
    output = {"extrinsics": {"T_EC": {"status": "CALIBRATED",
                                        "matrix": transform_to_config(result),
                                        "source": str(args.input_yaml),
                                        "rotation_singular_values":
                                        [float(item) for item in singular]}}}
    with Path(args.output_yaml).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)


if __name__ == "__main__":
    main()
