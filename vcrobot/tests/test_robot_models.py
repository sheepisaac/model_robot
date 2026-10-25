import math
import unittest
from pathlib import Path

import numpy as np

from calibration.calibrate_hand_eye import calibrate, calibration_residuals
from autonbv.robot_model import AutoNBVRobotModel
from robot_models import (CameraIntrinsics, GripperCameraMount,
                          IntegratedRobotCameraModel, L515CameraModel,
                          RoArmModel, RobotCameraOdometry, UGVModel,
                          UGVParameters)
from robot_models.camera import deproject_pixel, project_point
from robot_models.transforms import (invert_transform, make_transform,
                                     pose2d_transform, rotation_angle,
                                     rotation_z, rpy_matrix)


ROOT = Path(__file__).resolve().parents[1]


class TransformTests(unittest.TestCase):
    def test_inverse_and_pose2d(self):
        transform = pose2d_transform([1.0, -2.0, 0.7])
        self.assertTrue(np.allclose(invert_transform(transform).dot(transform),
                                    np.eye(4), atol=1e-12))
        self.assertTrue(np.allclose(transform[:3, :3], rotation_z(0.7)))

    def test_invalid_rotation_is_rejected(self):
        with self.assertRaises(ValueError):
            make_transform(np.zeros((3, 3)), [0, 0, 0])


class RoArmTests(unittest.TestCase):
    def test_firmware_fk_reference_expression_mode0(self):
        model = RoArmModel(eemode=0)
        q = [0.0, 0.0, math.pi / 2.0, math.pi]
        pose = model.firmware_pose(q)
        # Independently evaluated from RoArmM2_computePosbyJointRad.
        l3 = math.hypot(280.15, 1.73)
        expected = np.array([30.0 + l3, 0.0, 236.82, math.pi])
        self.assertTrue(np.allclose(pose, expected, atol=1e-6))

    def test_fk_ik_fk_roundtrip_mode0(self):
        model = RoArmModel(eemode=0)
        q = np.array([0.2, -0.15, 0.8, math.pi])
        target = model.firmware_pose(q)
        result = model.solve_ik({"x_mm": target[0], "y_mm": target[1],
                                 "z_mm": target[2], "t": target[3]})
        self.assertTrue(result.success, result.reason)
        achieved = model.firmware_pose(result.joint_angles)
        self.assertLess(np.linalg.norm(achieved[:3] - target[:3]), 4.0)

    def test_fk_ik_fk_roundtrip_mode1(self):
        model = RoArmModel(eemode=1)
        q = np.array([0.1, -0.2, 0.7, 2.0])
        target = model.firmware_pose(q)
        result = model.solve_ik({"x_mm": target[0], "y_mm": target[1],
                                 "z_mm": target[2], "t": target[3]})
        self.assertTrue(result.success, result.reason)
        self.assertTrue(np.allclose(result.joint_angles, q, atol=1e-10))

    def test_joint_limits_and_unreachable(self):
        model = RoArmModel()
        self.assertFalse(model.check_joint_limits([0, 0, 0, 0]))
        self.assertFalse(model.check_reachability([2.0, 0.0, 0.0, 0.0]))

    def test_joint_limits_match_firmware_servo_clamps(self):
        model = RoArmModel()
        limits = model.joint_limits
        self.assertAlmostEqual(limits[0, 0], -math.pi)
        self.assertAlmostEqual(limits[0, 1], math.pi)
        self.assertAlmostEqual(limits[1, 0], -math.pi / 2.0)
        self.assertAlmostEqual(limits[1, 1], math.pi / 2.0)
        self.assertAlmostEqual(limits[2, 0], -math.pi / 4.0)
        self.assertGreater(limits[2, 1], 3.13)

    def test_firmware_compatible_ik_clamps_but_accepts_real_command(self):
        model = RoArmModel()
        target = {"x_mm": 10, "y_mm": 0, "z_mm": 100, "t": 3}
        self.assertFalse(model.solve_ik(target).success)
        result = model.solve_ik_firmware(target)
        self.assertTrue(result.success, result.reason)
        self.assertTrue(result.clamped)
        self.assertIn("shoulder", result.reason)
        self.assertAlmostEqual(result.joint_angles[1], -math.pi / 2.0)
        self.assertAlmostEqual(result.joint_angles[3], 3.0)
        self.assertGreater(result.position_error_m, 0.0)

    def test_workspace_is_deterministic(self):
        model = RoArmModel()
        left = model.sample_workspace(10, seed=7)
        right = model.sample_workspace(10, seed=7)
        self.assertEqual(left.shape, (10, 3))
        self.assertTrue(np.array_equal(left, right))


class UGVTests(unittest.TestCase):
    def test_straight_reverse_and_rotation(self):
        model = UGVModel()
        self.assertTrue(np.allclose(model.predict_motion([0, 0, 0], 0.1, 0.1, 2),
                                    [0.2, 0.0, 0.0]))
        self.assertTrue(np.allclose(model.predict_motion([0, 0, 0], -0.1, -0.1, 2),
                                    [-0.2, 0.0, 0.0]))
        rotated = model.predict_motion([0, 0, 0], -0.1, 0.1, 1)
        self.assertAlmostEqual(rotated[0], 0.0)
        self.assertAlmostEqual(rotated[1], 0.0)
        self.assertAlmostEqual(rotated[2], 0.2 / 0.172)

    def test_parameter_fit_on_synthetic_data(self):
        truth_left, truth_right, truth_track = 0.94, 1.06, 0.19
        rows = []
        for left, right, duration in [(0.1, 0.1, 2), (-0.1, -0.1, 1),
                                      (-0.1, 0.1, 1), (0.05, 0.13, 1.5)]:
            distance = 0.5 * (truth_left * left + truth_right * right) * duration
            yaw = (truth_right * right - truth_left * left) / truth_track * duration
            rows.append({"v_left": left, "v_right": right, "duration": duration,
                         "distance_m": distance, "yaw_delta_rad": yaw})
        fitted = UGVModel.estimate_motion_parameters(rows).parameters
        self.assertAlmostEqual(fitted.left_scale, truth_left, places=10)
        self.assertAlmostEqual(fitted.right_scale, truth_right, places=10)
        self.assertAlmostEqual(fitted.effective_track_width_m, truth_track, places=10)

    def test_firmware_profiles_select_encoder_resolution_and_feedback_unit(self):
        ros = UGVParameters.from_mapping({"firmware_profile": "ugv_base_ros"})
        general = UGVParameters.from_mapping(
            {"firmware_profile": "ugv_base_general"})
        self.assertEqual(ros.encoder_pulses_per_rev, 660)
        self.assertEqual(ros.firmware_odometry_unit, "cm")
        self.assertEqual(general.encoder_pulses_per_rev, 1650)
        self.assertIsNone(general.firmware_odometry_unit)
        with self.assertRaises(ValueError):
            UGVParameters.from_mapping({"firmware_profile": "ugv_base_ros",
                                        "encoder_pulses_per_rev": 1650})


class CameraTests(unittest.TestCase):
    def test_projection_roundtrip(self):
        intrinsics = CameraIntrinsics(640, 480, 610, 608, 319.5, 239.5)
        point = np.array([0.12, -0.08, 1.4])
        pixel = project_point(point, intrinsics)
        restored = deproject_pixel(pixel, point[2], intrinsics)
        self.assertTrue(np.allclose(restored, point, atol=1e-12))

    def test_brown_projection_roundtrip(self):
        intrinsics = CameraIntrinsics(640, 480, 610, 608, 319.5, 239.5,
                                     "brown_conrady", (0.02, -0.01, 0.001, 0.002, 0.0))
        point = np.array([0.25, 0.1, 1.2])
        restored = deproject_pixel(project_point(point, intrinsics), point[2], intrinsics)
        self.assertTrue(np.allclose(restored, point, atol=1e-8))


class IntegratedTests(unittest.TestCase):
    @staticmethod
    def model(status="UNCALIBRATED"):
        intrinsics = CameraIntrinsics(640, 480, 600, 600, 319.5, 239.5)
        return IntegratedRobotCameraModel(
            RoArmModel(), UGVModel(), L515CameraModel(intrinsics, status),
            np.eye(4), np.eye(4), status, status)

    def test_pose_composition(self):
        model = self.model()
        q = [0.0, 0.0, math.pi / 2, math.pi]
        expected = pose2d_transform([1, 2, 0.3]).dot(
            model.roarm.forward_kinematics(q))
        actual = model.compute_camera_pose([1, 2, 0.3], q)
        self.assertTrue(np.allclose(actual, expected))

    def test_uncalibrated_model_cannot_claim_safe_reachability(self):
        model = self.model()
        target = model.compute_camera_pose([0, 0, 0],
                                           [0, 0, math.pi / 2, math.pi])
        result = model.check_camera_view_reachability(target)
        self.assertFalse(result.reachable)
        self.assertIn("not calibrated", result.reason)

    def test_mode0_rejects_unattainable_camera_orientation(self):
        model = self.model(status="CALIBRATED")
        target = model.compute_camera_pose([0, 0, 0],
                                           [0, 0, 0.8, math.pi])
        target[:3, :3] = target[:3, :3].dot(rotation_z(math.pi / 2))
        result = model.check_camera_view_reachability(target)
        self.assertFalse(result.reachable)
        self.assertIn("cannot satisfy", result.reason)

    def test_template_load_and_adapter(self):
        adapter = AutoNBVRobotModel(ROOT / "configs" /
                                    "robot_calibration.template.yaml")
        self.assertFalse(adapter.hardware_ready)
        pose = adapter.camera_pose_from_joints([0, 0, 0],
                                               [0, 0, math.pi / 2, math.pi])
        self.assertEqual(pose.shape, (4, 4))

    def test_moving_gripper_changes_camera_pose_with_t(self):
        intrinsics = CameraIntrinsics(640, 480, 600, 600, 319.5, 239.5)
        T_GC = make_transform(translation=[0.1, 0.0, 0.0])
        mount = GripperCameraMount(
            mode="revolute", T_GC=T_GC,
            reference_t_rad=math.pi, axis=[0, 1, 0], radians_per_t=1.0,
            motion_status="CALIBRATED", t_gc_status="CALIBRATED")
        model = IntegratedRobotCameraModel(
            RoArmModel(), UGVModel(),
            L515CameraModel(intrinsics, "CALIBRATED"), np.eye(4),
            t_ba_status="CALIBRATED", camera_mount=mount)
        low_t = [0.0, 0.0, math.pi / 2.0, math.pi - 0.3]
        high_t = [0.0, 0.0, math.pi / 2.0, math.pi + 0.3]
        low_pose = model.compute_camera_pose([0, 0, 0], low_t)
        high_pose = model.compute_camera_pose([0, 0, 0], high_t)
        self.assertLess(high_pose[2, 3], low_pose[2, 3])
        result = model.check_camera_view_reachability(low_pose)
        self.assertTrue(result.reachable, result.reason)

    def test_lookup_mount_requires_a_calibrated_t_sample(self):
        first = np.eye(4)
        second = make_transform(translation=[0.0, 0.0, -0.02])
        mount = GripperCameraMount(
            mode="lookup", T_GC=np.eye(4),
            samples=[(2.0, first), (3.0, second)],
            lookup_tolerance_rad=0.01,
            motion_status="CALIBRATED", t_gc_status="CALIBRATED")
        self.assertTrue(np.allclose(mount.gripper_transform(3.005), second))
        with self.assertRaises(ValueError):
            mount.gripper_transform(2.5)

    def test_intrinsics_are_part_of_integrated_calibration_state(self):
        intrinsics = CameraIntrinsics(640, 480, 600, 600, 319.5, 239.5)
        model = IntegratedRobotCameraModel(
            RoArmModel(), UGVModel(), L515CameraModel(intrinsics),
            np.eye(4), np.eye(4), "CALIBRATED", "CALIBRATED")
        self.assertFalse(model.calibrated)
        self.assertEqual(model.calibration_status["l515_intrinsics"],
                         "UNCALIBRATED")


class OdometryTests(unittest.TestCase):
    @staticmethod
    def odometry():
        model = IntegratedTests.model(status="CALIBRATED")
        return RobotCameraOdometry(
            model, [0.0, 0.0, 0.0],
            [0.0, 0.0, math.pi / 2.0, math.pi])

    def test_base_and_camera_increment_from_wheel_velocity(self):
        odometry = self.odometry()
        update = odometry.update_wheel_velocity(0.1, 0.1, 1.0)
        self.assertTrue(np.allclose(update.base_delta_body, [0.1, 0.0, 0.0]))
        self.assertAlmostEqual(update.camera_from_base.distance_m, 0.1)
        self.assertAlmostEqual(update.camera_from_arm.distance_m, 0.0)
        self.assertAlmostEqual(update.camera_total.distance_m, 0.1)
        self.assertEqual(update.sources["base"], "wheel_velocity")

    def test_arm_and_camera_increment_from_joint_feedback(self):
        odometry = self.odometry()
        target = np.array([0.1, -0.1, 0.8, math.pi])
        update = odometry.update_absolute(joint_angles=target)
        self.assertTrue(np.allclose(update.base_delta_body, 0.0))
        self.assertTrue(np.allclose(update.joint_delta,
                                    target - update.joint_angles_before))
        self.assertAlmostEqual(update.camera_from_base.distance_m, 0.0)
        self.assertGreater(update.camera_from_arm.distance_m, 0.0)
        self.assertTrue(np.allclose(update.camera_from_arm.transform,
                                    update.camera_total.transform))

    def test_encoder_counts_require_baseline_and_integrate_one_revolution(self):
        odometry = self.odometry()
        self.assertIsNone(odometry.update_encoder_counts(1000, 1000))
        update = odometry.update_encoder_counts(1660, 1660)
        expected = math.pi * 0.08
        self.assertAlmostEqual(update.base_delta_body[0], expected)
        self.assertAlmostEqual(update.base_delta_body[1], 0.0)
        self.assertAlmostEqual(update.base_delta_body[2], 0.0)

    def test_firmware_odl_odr_centimetres_and_xyzt_feedback(self):
        intrinsics = CameraIntrinsics(640, 480, 600, 600, 319.5, 239.5)
        parameters = UGVParameters.from_mapping({
            "firmware_profile": "ugv_base_ros", "calibrated": True})
        model = IntegratedRobotCameraModel(
            RoArmModel(), UGVModel(parameters),
            L515CameraModel(intrinsics, "CALIBRATED"),
            np.eye(4), np.eye(4), "CALIBRATED", "CALIBRATED")
        odometry = RobotCameraOdometry(
            model, [0.0, 0.0, 0.0],
            [0.0, 0.0, math.pi / 2.0, math.pi])
        self.assertIsNone(
            odometry.update_firmware_odometry(200, 300))
        target_joints = np.array([0.1, -0.1, 0.8, math.pi])
        x, y, z, t = odometry.model.roarm.firmware_pose(target_joints)
        update = odometry.update_firmware_odometry(
            210, 310, arm_pose={"x": x, "y": y, "z": z, "t": t})
        self.assertAlmostEqual(update.base_delta_body[0], 0.1)
        self.assertEqual(update.sources["base"], "firmware_odometry_cm")
        self.assertEqual(update.sources["arm"], "xyzt_ik")
        achieved = odometry.model.roarm.firmware_pose(update.joint_angles_after)
        self.assertLess(np.linalg.norm(achieved[:3] - [x, y, z]), 4.0)
        self.assertGreater(update.camera_total.distance_m, 0.0)


class CalibrationTests(unittest.TestCase):
    def test_synthetic_hand_eye_recovers_known_transform(self):
        truth = make_transform(rpy_matrix(0.12, -0.18, 0.25),
                               [0.035, -0.012, 0.08])
        target = make_transform(rpy_matrix(0.2, 0.1, -0.15),
                                [0.45, 0.05, 0.22])
        poses = [(0, 0, 0, 0, 0, 0),
                 (0.1, -0.2, 0.3, .1, .0, .05),
                 (-.2, .15, -.1, .0, .12, .02),
                 (.25, .1, .2, -.08, .05, .08),
                 (-.1, -.25, .18, .04, -.1, .04),
                 (.18, -.12, -.22, -.05, -.04, .06)]
        samples = []
        for roll, pitch, yaw, x, y, z in poses:
            T_BG = make_transform(rpy_matrix(roll, pitch, yaw), [x, y, z])
            T_CT = invert_transform(T_BG.dot(truth)).dot(target)
            samples.append({"T_BG": T_BG.tolist(), "T_CT": T_CT.tolist()})
        estimate, _ = calibrate(samples)
        self.assertLess(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]), 1e-10)
        self.assertLess(rotation_angle(estimate[:3, :3], truth[:3, :3]), 1e-10)
        residuals = calibration_residuals(samples, estimate)
        self.assertLess(residuals["translation_rmse_m"], 1e-10)
        self.assertLess(residuals["rotation_rmse_deg"], 1e-8)


if __name__ == "__main__":
    unittest.main()
