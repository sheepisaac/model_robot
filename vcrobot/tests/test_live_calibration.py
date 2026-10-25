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

from live_robot_calibration import (firmware_distance_delta,
                                    firmware_odometry_available,
                                    parse_surface_command, pose_to_joints,
                                    resolve_arm_command)
from robot_models import IntegratedRobotCameraModel
from rgbd_visual_odometry import (estimate_pose_from_correspondences,
                                  invert_transform)
from slave_v4l2_odometry import V4L2_FOURCC, decode_frame, parse_distortion


class CommandParsingTests(unittest.TestCase):
    def test_controller_surface_syntax_is_classified(self):
        ugv = parse_surface_command("go 0.08 -0.06 1.2")
        arm = parse_surface_command("10 0 100 3")
        relative = parse_surface_command("arm_delta 0 0 10 0")
        self.assertEqual(ugv["kind"], "ugv")
        self.assertAlmostEqual(ugv["duration"], 1.2)
        self.assertEqual(arm["kind"], "arm")
        self.assertEqual(arm["t"], 3.0)
        self.assertEqual(relative["kind"], "arm_delta")
        self.assertEqual(relative["dz"], 10.0)

    def test_relative_arm_command_uses_fresh_feedback(self):
        request = parse_surface_command("arm_delta 0 0 10 0")
        resolved = resolve_arm_command(
            {"x": 8.3344, "y": -0.0384, "z": 193.8203, "t": 3.4944},
            request)
        self.assertEqual(resolved["kind"], "arm")
        self.assertAlmostEqual(resolved["x"], 8.3344)
        self.assertAlmostEqual(resolved["z"], 203.8203)
        self.assertAlmostEqual(resolved["t"], 3.4944)
        self.assertEqual(resolved["relative_delta"]["dz"], 10.0)

    def test_ugv_duration_is_required(self):
        with self.assertRaises(ValueError):
            parse_surface_command("go 0.1 0.1")

    def test_general_driver_feedback_explicitly_has_no_odometry(self):
        general = {"T": 1001, "L": 0.0, "R": 0.0,
                   "odometry_available": False}
        ros_before = {"T": 1001, "odl": 10, "odr": 20,
                      "odometry_available": True}
        ros_after = {"T": 1001, "odl": 13, "odr": 18,
                     "odometry_available": True}
        self.assertFalse(firmware_odometry_available(general))
        self.assertTrue(firmware_odometry_available(ros_before, ros_after))
        self.assertEqual(firmware_distance_delta(
            ros_before, ros_after, "cm"), (0.03, -0.02))

    def test_live_feedback_on_firmware_clamp_is_modelled(self):
        model = IntegratedRobotCameraModel.from_yaml(
            ROOT / "configs" / "robot_calibration.template.yaml")
        joints = pose_to_joints(model, {
            "x": 42.56144571, "y": -0.195866703,
            "z": 47.80462851, "t": 3.140058673,
        })
        self.assertTrue(model.roarm.check_joint_limits(joints))
        achieved = model.roarm.firmware_pose(joints)
        self.assertLess(np.linalg.norm(
            achieved[:3] - [42.56144571, -0.195866703, 47.80462851]), 4.0)


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


class V4L2FrameTests(unittest.TestCase):
    def test_z16_fourcc_preserves_the_required_trailing_space(self):
        self.assertEqual(V4L2_FOURCC["Z16"], "0x2036315a")

    def test_grey_and_little_endian_z16_frames_are_decoded(self):
        grey = decode_frame(bytes([1, 2, 3, 4, 5, 6]), 3, 2, "GREY")
        np.testing.assert_array_equal(grey, [[1, 2, 3], [4, 5, 6]])
        raw_depth = np.array([1, 256, 1000, 65535], dtype="<u2").tobytes()
        depth = decode_frame(raw_depth, 2, 2, "Z16")
        np.testing.assert_array_equal(depth, [[1, 256], [1000, 65535]])

    def test_frame_size_and_distortion_validation(self):
        with self.assertRaises(ValueError):
            decode_frame(b"too short", 2, 2, "Z16")
        np.testing.assert_array_equal(parse_distortion("0,0,0,0,0"),
                                      np.zeros(5))
        with self.assertRaises(ValueError):
            parse_distortion("0,0,0")


if __name__ == "__main__":
    unittest.main()
