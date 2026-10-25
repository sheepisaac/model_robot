import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "calibration"
for path in (ROOT, CALIBRATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from live_robot_calibration import parse_surface_command
from rgbd_visual_odometry import (estimate_pose_from_correspondences,
                                  invert_transform)


class CommandParsingTests(unittest.TestCase):
    def test_controller_surface_syntax_is_classified(self):
        ugv = parse_surface_command("go 0.08 -0.06 1.2")
        arm = parse_surface_command("10 0 100 3")
        self.assertEqual(ugv["kind"], "ugv")
        self.assertAlmostEqual(ugv["duration"], 1.2)
        self.assertEqual(arm["kind"], "arm")
        self.assertEqual(arm["t"], 3.0)

    def test_ugv_duration_is_required(self):
        with self.assertRaises(ValueError):
            parse_surface_command("go 0.1 0.1")


class RGBDOdometryTests(unittest.TestCase):
    def test_metric_pnp_transform_convention(self):
        rng = np.random.default_rng(7)
        points = np.column_stack((rng.uniform(-0.5, 0.5, 80),
                                  rng.uniform(-0.35, 0.35, 80),
                                  rng.uniform(1.0, 2.5, 80)))
        camera = np.array([[610.0, 0.0, 319.5],
                           [0.0, 608.0, 239.5],
                           [0.0, 0.0, 1.0]])
        rvec = np.array([0.015, -0.03, 0.02])
        translation = np.array([0.04, -0.015, 0.025])
        image, _ = cv2.projectPoints(points, rvec, translation, camera, None)
        image = image.reshape(-1, 2)
        image += rng.normal(0.0, 0.15, image.shape)
        estimated, inliers, rmse = estimate_pose_from_correspondences(
            points, image, camera)
        expected_rotation, _ = cv2.Rodrigues(rvec)
        self.assertGreaterEqual(len(inliers), 70)
        self.assertLess(rmse, 0.5)
        np.testing.assert_allclose(estimated[:3, :3], expected_rotation,
                                   atol=2e-3)
        np.testing.assert_allclose(estimated[:3, 3], translation, atol=3e-3)
        np.testing.assert_allclose(estimated.dot(invert_transform(estimated)),
                                   np.eye(4), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
