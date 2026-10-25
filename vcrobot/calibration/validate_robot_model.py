"""Run deterministic numerical checks against a calibration YAML."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_models import IntegratedRobotCameraModel
from robot_models.transforms import invert_transform


def validate(path):
    model = IntegratedRobotCameraModel.from_yaml(path)
    # Avoid a joint-limit boundary: the manufacturer FK and IK use the small
    # L3 offset angle differently, so a nominal boundary pose may round just
    # outside the declared firmware limit.
    q = np.array([0.2, -0.15, 0.8, math.pi])
    T = model.compute_camera_pose((0.0, 0.0, 0.0), q)
    inverse_error = float(np.max(np.abs(invert_transform(T).dot(T) - np.eye(4))))
    firmware_pose = model.roarm.firmware_pose(q)
    ik = model.roarm.solve_ik({"x_mm": firmware_pose[0],
                               "y_mm": firmware_pose[1],
                               "z_mm": firmware_pose[2],
                               "t": firmware_pose[3]})
    return {"calibrated": model.calibrated,
            "calibration_status": model.calibration_status,
            "camera_mount_mode": model.camera_mount.mode,
            "ugv_firmware_profile": model.ugv.parameters.firmware_profile,
            "ugv_firmware_odometry_unit":
            model.ugv.parameters.firmware_odometry_unit,
            "ugv_encoder_pulses_per_rev":
            model.ugv.parameters.encoder_pulses_per_rev,
            "transform_inverse_max_error": inverse_error,
            "fk_ik_success": ik.success,
            "fk_ik_reason": ik.reason,
            "fk_ik_position_error_m": ik.position_error_m,
            "safe_for_hardware_view_claims": model.calibrated and
            model.ugv.parameters.calibrated}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.config), indent=2))


if __name__ == "__main__":
    main()
