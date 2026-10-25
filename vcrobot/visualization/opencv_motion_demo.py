"""Interactive OpenCV motion demo for the UGV/RoArm/L515 model.

This is deliberately simulation-only: it never imports pyserial and never
sends a command to the physical robot.  Click the OpenCV window, type a command
and press Enter.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_models import IntegratedRobotCameraModel, RobotCameraOdometry
from robot_models.transforms import pose2d_transform, transform_points


WINDOW_NAME = "autoNBV motion-model demo (SIMULATION ONLY)"
INITIAL_JOINTS = np.array([0.0, 0.0, math.pi / 2.0, math.pi])


class CommandError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedCommand:
    action: str
    values: tuple = ()


def parse_command(text):
    """Parse the two existing vcrobot command styles plus demo commands."""
    parts = text.strip().lower().split()
    if not parts:
        raise CommandError("empty command")
    if len(parts) == 1 and parts[0] in {
            "stop", "reset", "clear", "help", "quit", "exit"}:
        action = "quit" if parts[0] == "exit" else parts[0]
        return ParsedCommand(action)
    if parts[0] == "go":
        if len(parts) != 3:
            raise CommandError("usage: go LEFT_MPS RIGHT_MPS")
        try:
            values = tuple(float(value) for value in parts[1:])
        except ValueError as exc:
            raise CommandError("wheel speeds must be numbers") from exc
        if not all(math.isfinite(value) for value in values):
            raise CommandError("wheel speeds must be finite")
        return ParsedCommand("go", values)
    if len(parts) == 4:
        try:
            values = tuple(float(value) for value in parts)
        except ValueError as exc:
            raise CommandError("arm command must be: X_MM Y_MM Z_MM T_RAD") from exc
        if not all(math.isfinite(value) for value in values):
            raise CommandError("arm values must be finite")
        return ParsedCommand("arm", values)
    raise CommandError("use 'go L R' or 'X Y Z T'; type help for more")


def _put_text(image, text, position, scale=0.48, color=(225, 230, 235),
              thickness=1):
    cv2.putText(image, str(text), tuple(map(int, position)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def _polyline(image, points, color, thickness=1, closed=False):
    points = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) >= 2:
        cv2.polylines(image, [points], closed, color, thickness, cv2.LINE_AA)


class MotionDemo:
    def __init__(self, model, arm_speed_rad_s=1.0, max_wheel_speed=1.0,
                 trail_length=1500, gripper_orientation="vertical",
                 gripper_sweep_deg=90.0, view_azimuth_deg=-45.0):
        self.model = model
        self.arm_speed_rad_s = float(arm_speed_rad_s)
        self.max_wheel_speed = float(max_wheel_speed)
        self.trail_length = int(trail_length)
        if gripper_orientation not in ("vertical", "lateral"):
            raise ValueError("gripper_orientation must be vertical or lateral")
        self.gripper_orientation = gripper_orientation
        self.gripper_sweep_deg = float(gripper_sweep_deg)
        self.view_azimuth_deg = float(view_azimuth_deg)
        self.input_buffer = ""
        self.history = []
        self.history_index = 0
        self.status = "Click this window, type a command, then press Enter."
        self.status_is_error = False
        self.running = True
        self.reset()

    def reset(self):
        self.odometry = RobotCameraOdometry(
            self.model, [0.0, 0.0, 0.0], INITIAL_JOINTS)
        self.wheel_velocity = np.zeros(2)
        self.target_joints = INITIAL_JOINTS.copy()
        self.last_update = None
        self.base_trail = deque(maxlen=self.trail_length)
        self.camera_trail = deque(maxlen=self.trail_length)
        self._append_trail()
        self.status = "Simulation reset."
        self.status_is_error = False

    def _append_trail(self):
        self.base_trail.append(self.odometry.base_pose[:2].copy())
        self.camera_trail.append(self.odometry.camera_pose[:2, 3].copy())

    def _gripper_fraction(self):
        limits = self.model.roarm.joint_limits[3]
        fraction = ((self.odometry.joint_angles[3] - limits[0]) /
                    (limits[1] - limits[0]))
        return float(np.clip(fraction, 0.0, 1.0))

    def _gripper_angle_deg(self):
        """Vertical proxy: small t moves up and large t moves down."""
        return 0.5 * self.gripper_sweep_deg * (
            1.0 - 2.0 * self._gripper_fraction())

    def _gripper_segments(self, T_WE):
        """Return colored 3D line segments for the selected mount direction."""
        end = T_WE[:3, 3]
        tool_x, tool_y, tool_z = (T_WE[:3, index] for index in range(3))
        palm = end - 0.020 * tool_x
        segments = [(palm, end, (70, 190, 235))]
        if self.gripper_orientation == "vertical":
            angle = math.radians(self._gripper_angle_deg())
            moving_direction = math.cos(angle) * tool_x + math.sin(angle) * tool_z
            segments.append((end, end + 0.065 * tool_x, (180, 190, 200)))
            segments.append((end, end + 0.065 * moving_direction,
                             (60, 235, 235)))
        else:
            gap = 0.010 + (1.0 - self._gripper_fraction()) * 0.070
            bases = np.array([palm - 0.5 * gap * tool_y,
                              palm + 0.5 * gap * tool_y])
            tips = bases + 0.060 * tool_x
            segments.append((bases[0], bases[1], (60, 235, 235)))
            segments.extend((base, tip, (60, 235, 235))
                            for base, tip in zip(bases, tips))
        return segments

    def execute(self, text):
        command = parse_command(text)
        if command.action == "go":
            if max(abs(value) for value in command.values) > self.max_wheel_speed:
                raise CommandError(
                    "demo speed limit is +/-{:.2f} m/s".format(
                        self.max_wheel_speed))
            self.wheel_velocity[:] = command.values
            self.status = "UGV velocity set: L={:.3f}, R={:.3f} m/s".format(
                *command.values)
        elif command.action == "stop":
            self.wheel_velocity[:] = 0.0
            self.status = "UGV stopped; arm target is unchanged."
        elif command.action == "arm":
            x_mm, y_mm, z_mm, t_rad = command.values
            result = self.model.roarm.solve_ik_firmware({
                "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm, "t": t_rad})
            if not result.success:
                raise CommandError("arm target rejected: " + result.reason)
            self.target_joints = result.joint_angles.copy()
            if result.clamped:
                self.status = (
                    "RoArm accepted with {}; reached-position error={:.1f} mm"
                    .format(result.reason, 1000.0 * result.position_error_m))
            else:
                self.status = (
                    "RoArm target accepted; IK residual={:.2f} mm".format(
                        1000.0 * result.position_error_m))
        elif command.action == "reset":
            self.reset()
            return
        elif command.action == "clear":
            self.base_trail.clear()
            self.camera_trail.clear()
            self._append_trail()
            self.status = "Trails cleared."
        elif command.action == "help":
            self.status = "go L R | stop | X_MM Y_MM Z_MM T_RAD | reset | clear | quit"
        elif command.action == "quit":
            self.running = False
            return
        self.status_is_error = False

    def submit_input(self):
        text = self.input_buffer.strip()
        self.input_buffer = ""
        if not text:
            return
        self.history.append(text)
        self.history = self.history[-100:]
        self.history_index = len(self.history)
        try:
            self.execute(text)
        except CommandError as exc:
            self.status = "ERROR: " + str(exc)
            self.status_is_error = True

    def update(self, dt):
        dt = max(0.0, min(float(dt), 0.1))
        joint_error = self.target_joints - self.odometry.joint_angles
        max_step = self.arm_speed_rad_s * dt
        joint_step = np.clip(joint_error, -max_step, max_step)
        next_joints = self.odometry.joint_angles + joint_step
        self.last_update = self.odometry.update_wheel_velocity(
            self.wheel_velocity[0], self.wheel_velocity[1], dt,
            joint_angles=next_joints)
        if (np.linalg.norm(self.last_update.base_delta_body) > 1e-10 or
                np.linalg.norm(self.last_update.joint_delta) > 1e-10):
            self._append_trail()

    @staticmethod
    def _world_to_pixel(points, center, panel, pixels_per_metre):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        x0, y0, x1, y1 = panel
        panel_center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        shifted = points - np.asarray(center, dtype=float)
        return np.column_stack((
            panel_center[0] + shifted[:, 0] * pixels_per_metre,
            panel_center[1] - shifted[:, 1] * pixels_per_metre)).astype(int)

    def _project_3d(self, points, center, panel, pixels_per_metre):
        """Orthographically project world XYZ into an isometric OpenCV view."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        x0, y0, x1, y1 = panel
        relative = points - np.asarray(center, dtype=float).reshape(1, 3)
        azimuth = math.radians(self.view_azimuth_deg)
        elevation = math.radians(27.0)
        screen_right = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
        screen_up = np.array([
            -math.cos(azimuth) * math.sin(elevation),
            -math.sin(azimuth) * math.sin(elevation),
            math.cos(elevation)])
        horizontal = relative.dot(screen_right)
        vertical = relative.dot(screen_up)
        origin = np.array([(x0 + x1) / 2.0 - 8.0, y1 - 48.0])
        return np.column_stack((
            origin[0] + horizontal * pixels_per_metre,
            origin[1] - vertical * pixels_per_metre)).astype(int)

    def _draw_top_view(self, image, panel):
        x0, y0, x1, y1 = panel
        cv2.rectangle(image, (x0, y0), (x1, y1), (60, 67, 74), 1)
        _put_text(image, "TOP VIEW (world XY, follows UGV)", (x0 + 12, y0 + 23),
                  0.52, (130, 210, 255))
        center = self.odometry.base_pose[:2]
        scale = 145.0

        # Half-metre world grid.
        half_x = (x1 - x0) / (2.0 * scale)
        half_y = (y1 - y0) / (2.0 * scale)
        gx0 = math.floor((center[0] - half_x) / 0.5) * 0.5
        gy0 = math.floor((center[1] - half_y) / 0.5) * 0.5
        for gx in np.arange(gx0, center[0] + half_x + 0.5, 0.5):
            points = self._world_to_pixel(
                [[gx, center[1] - half_y], [gx, center[1] + half_y]],
                center, panel, scale)
            cv2.line(image, tuple(points[0]), tuple(points[1]), (40, 46, 52), 1)
        for gy in np.arange(gy0, center[1] + half_y + 0.5, 0.5):
            points = self._world_to_pixel(
                [[center[0] - half_x, gy], [center[0] + half_x, gy]],
                center, panel, scale)
            cv2.line(image, tuple(points[0]), tuple(points[1]), (40, 46, 52), 1)

        if len(self.base_trail) > 1:
            _polyline(image, self._world_to_pixel(
                self.base_trail, center, panel, scale), (100, 170, 245), 2)
        if len(self.camera_trail) > 1:
            _polyline(image, self._world_to_pixel(
                self.camera_trail, center, panel, scale), (90, 100, 245), 1)

        base = self.odometry.base_pose
        c, s = math.cos(base[2]), math.sin(base[2])
        rotation = np.array([[c, -s], [s, c]])
        body_local = np.array([
            [-0.16, -0.12], [0.16, -0.12], [0.16, 0.12], [-0.16, 0.12]])
        body_world = body_local.dot(rotation.T) + base[:2]
        body_px = self._world_to_pixel(body_world, center, panel, scale)
        cv2.fillConvexPoly(image, body_px, (95, 112, 126), cv2.LINE_AA)
        _polyline(image, np.vstack((body_px, body_px[0])), (210, 220, 228), 2)

        heading = np.array([base[:2], base[:2] + 0.25 * np.array([c, s])])
        heading_px = self._world_to_pixel(heading, center, panel, scale)
        cv2.arrowedLine(image, tuple(heading_px[0]), tuple(heading_px[1]),
                        (70, 230, 150), 2, cv2.LINE_AA, tipLength=0.25)

        T_WB = pose2d_transform(base)
        arm_world = transform_points(
            T_WB.dot(self.model.T_BA),
            self.model.roarm.joint_positions(self.odometry.joint_angles))
        arm_px = self._world_to_pixel(arm_world[:, :2], center, panel, scale)
        _polyline(image, arm_px, (80, 210, 250), 4)
        for point in arm_px:
            cv2.circle(image, tuple(point), 5, (50, 150, 240), -1, cv2.LINE_AA)

        # In EEMode 0, q4/t drives the gripper rather than end-frame rotation.
        if self.model.roarm.eemode == 0:
            T_WE = T_WB.dot(self.model.T_BA).dot(
                self.model.roarm.forward_kinematics(
                    self.odometry.joint_angles))
            for start, finish, color in self._gripper_segments(T_WE):
                segment_px = self._world_to_pixel(
                    [start[:2], finish[:2]], center, panel, scale)
                cv2.line(image, tuple(segment_px[0]), tuple(segment_px[1]),
                         color, 4, cv2.LINE_AA)

        camera = self.odometry.camera_pose
        camera_line = np.array([
            camera[:3, 3], camera[:3, 3] + 0.18 * camera[:3, 2]])
        camera_px = self._world_to_pixel(camera_line[:, :2], center, panel, scale)
        cv2.circle(image, tuple(camera_px[0]), 6, (80, 80, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(image, tuple(camera_px[0]), tuple(camera_px[1]),
                        (80, 80, 255), 2, cv2.LINE_AA, tipLength=0.3)

    def _draw_3d_view(self, image, panel):
        x0, y0, x1, y1 = panel
        cv2.rectangle(image, (x0, y0), (x1, y1), (60, 67, 74), 1)
        _put_text(image, "3D VIEW (world XYZ, follows UGV)", (x0 + 12, y0 + 23),
                  0.52, (130, 210, 255))
        scale = 500.0
        base = self.odometry.base_pose
        center = np.array([base[0], base[1], 0.0])
        T_WB = pose2d_transform(base)

        # Ground grid in the fixed world frame.
        grid_extent = 0.28
        grid_values = np.linspace(-grid_extent, grid_extent, 7)
        for value in grid_values:
            horizontal = np.array([
                [base[0] - grid_extent, base[1] + value, 0.0],
                [base[0] + grid_extent, base[1] + value, 0.0]])
            vertical = np.array([
                [base[0] + value, base[1] - grid_extent, 0.0],
                [base[0] + value, base[1] + grid_extent, 0.0]])
            horizontal_px = self._project_3d(horizontal, center, panel, scale)
            vertical_px = self._project_3d(vertical, center, panel, scale)
            cv2.line(image, tuple(horizontal_px[0]), tuple(horizontal_px[1]),
                     (42, 48, 54), 1, cv2.LINE_AA)
            cv2.line(image, tuple(vertical_px[0]), tuple(vertical_px[1]),
                     (42, 48, 54), 1, cv2.LINE_AA)

        # World X/Y/Z axes at the current UGV position.
        axes = ((np.array([0.22, 0.0, 0.0]), "X", (80, 90, 255)),
                (np.array([0.0, 0.22, 0.0]), "Y", (80, 220, 100)),
                (np.array([0.0, 0.0, 0.22]), "Z", (255, 170, 70)))
        origin_px = self._project_3d([center], center, panel, scale)[0]
        for offset, label, color in axes:
            endpoint = center + offset
            endpoint_px = self._project_3d(
                [endpoint], center, panel, scale)[0]
            cv2.arrowedLine(image, tuple(origin_px), tuple(endpoint_px), color,
                            2, cv2.LINE_AA, tipLength=0.18)
            _put_text(image, label, endpoint_px + np.array([4, -3]),
                      0.40, color, 2)

        # UGV proxy cuboid.
        footprint = [(-0.16, -0.12), (0.16, -0.12),
                     (0.16, 0.12), (-0.16, 0.12)]
        body_local = np.array(
            [[x, y, z] for z in (0.0, 0.07) for x, y in footprint])
        body_world = transform_points(T_WB, body_local)
        body_px = self._project_3d(body_world, center, panel, scale)
        body_edges = ((0, 1), (1, 2), (2, 3), (3, 0),
                      (4, 5), (5, 6), (6, 7), (7, 4),
                      (0, 4), (1, 5), (2, 6), (3, 7))
        for first, second in body_edges:
            cv2.line(image, tuple(body_px[first]), tuple(body_px[second]),
                     (190, 198, 205), 2, cv2.LINE_AA)

        arm_world = transform_points(
            T_WB.dot(self.model.T_BA),
            self.model.roarm.joint_positions(self.odometry.joint_angles))
        arm_px = self._project_3d(arm_world, center, panel, scale)
        _polyline(image, arm_px, (80, 210, 250), 5)
        for point in arm_px:
            cv2.circle(image, tuple(point), 6, (50, 150, 240), -1, cv2.LINE_AA)

        # Actual 3D gripper proxy at the modeled tool frame.
        if self.model.roarm.eemode == 0:
            T_WE = T_WB.dot(self.model.T_BA).dot(
                self.model.roarm.forward_kinematics(
                    self.odometry.joint_angles))
            for start, finish, color in self._gripper_segments(T_WE):
                segment_px = self._project_3d(
                    [start, finish], center, panel, scale)
                cv2.line(image, tuple(segment_px[0]), tuple(segment_px[1]),
                         color, 4, cv2.LINE_AA)

        camera_world = self.odometry.camera_pose
        camera_line = np.array([
            camera_world[:3, 3],
            camera_world[:3, 3] + 0.15 * camera_world[:3, 2]])
        camera_px = self._project_3d(camera_line, center, panel, scale)
        cv2.circle(image, tuple(camera_px[0]), 6, (80, 80, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(image, tuple(camera_px[0]), tuple(camera_px[1]),
                        (80, 80, 255), 2, cv2.LINE_AA, tipLength=0.3)

        # Magnified front-view proxy so small physical jaw changes remain
        # readable even when the full robot is zoomed out.
        inset = (x1 - 145, y0 + 34, x1 - 12, y0 + 145)
        ix0, iy0, ix1, iy1 = inset
        cv2.rectangle(image, (ix0, iy0), (ix1, iy1), (75, 82, 89), 1)
        _put_text(image, "GRIPPER PROXY", (ix0 + 7, iy0 + 18), 0.36,
                  (70, 225, 235))
        center_y = (iy0 + iy1) // 2 + 8
        palm_x = ix0 + 25
        tip_x = ix1 - 16
        if self.gripper_orientation == "vertical":
            angle_deg = self._gripper_angle_deg()
            angle = math.radians(angle_deg)
            length = min(tip_x - palm_x, 68)
            moving_tip = (int(round(palm_x + length * math.cos(angle))),
                          int(round(center_y - length * math.sin(angle))))
            cv2.line(image, (palm_x, center_y), (palm_x + length, center_y),
                     (180, 190, 200), 4, cv2.LINE_AA)
            cv2.line(image, (palm_x, center_y), moving_tip,
                     (60, 235, 235), 5, cv2.LINE_AA)
            cv2.circle(image, (palm_x, center_y), 5,
                       (70, 190, 235), -1, cv2.LINE_AA)
            direction = "UP" if angle_deg > 1e-6 else (
                "DOWN" if angle_deg < -1e-6 else "LEVEL")
            _put_text(image, "{} {:+.1f} deg".format(direction, angle_deg),
                      (ix0 + 23, iy1 - 8), 0.32, (210, 220, 225))
        else:
            gap = 0.010 + (1.0 - self._gripper_fraction()) * 0.070
            gap_px = int(round(8.0 + (gap - 0.010) / 0.070 * 42.0))
            upper_y = center_y - gap_px // 2
            lower_y = center_y + gap_px // 2
            cv2.line(image, (palm_x, upper_y), (palm_x, lower_y),
                     (60, 235, 235), 4, cv2.LINE_AA)
            cv2.line(image, (palm_x, upper_y), (tip_x, upper_y),
                     (60, 235, 235), 5, cv2.LINE_AA)
            cv2.line(image, (palm_x, lower_y), (tip_x, lower_y),
                     (60, 235, 235), 5, cv2.LINE_AA)
            _put_text(image, "{:.1f} mm".format(1000.0 * gap),
                      (ix0 + 40, iy1 - 8), 0.34, (210, 220, 225))

    def _draw_telemetry(self, image, origin):
        x, y = origin
        base = self.odometry.base_pose
        q = self.odometry.joint_angles
        camera = self.odometry.camera_pose[:3, 3]
        rows = [
            "wheel m/s   L {:+.3f}   R {:+.3f}".format(*self.wheel_velocity),
            "base        x {:+.3f}  y {:+.3f}  yaw {:+.1f} deg".format(
                base[0], base[1], math.degrees(base[2])),
            "joints rad  " + "  ".join("{:+.3f}".format(value) for value in q),
            "gripper     t {:+.3f} rad  vertical {:+.1f} deg (proxy)".format(
                q[3], self._gripper_angle_deg())
            if self.gripper_orientation == "vertical" else
            "gripper     t {:+.3f} rad  lateral {:.0f}%".format(
                q[3], 100.0 * self._gripper_fraction()),
            "camera m    x {:+.3f}  y {:+.3f}  z {:+.3f}".format(*camera),
        ]
        if self.last_update is not None:
            delta = self.last_update
            rows.extend([
                "last dBase [{:+.4f}, {:+.4f}, {:+.3f} deg]".format(
                    delta.base_delta_body[0], delta.base_delta_body[1],
                    math.degrees(delta.base_delta_body[2])),
                "last dCam  {:.4f} m / {:.3f} deg".format(
                    delta.camera_total.distance_m,
                    math.degrees(delta.camera_total.rotation_rad)),
            ])
        for index, row in enumerate(rows):
            _put_text(image, row, (x, y + 24 * index), 0.47)

    def render(self, size=(1280, 800)):
        width, height = size
        image = np.full((height, width, 3), (25, 29, 34), dtype=np.uint8)
        _put_text(image, "autoNBV UGV + RoArm + L515 motion model",
                  (22, 34), 0.72, (235, 240, 245), 2)
        calibration = "CALIBRATED" if self.model.calibrated else "UNCALIBRATED"
        _put_text(image, "SIMULATION ONLY | " + calibration,
                  (width - 335, 34), 0.52,
                  (80, 210, 120) if self.model.calibrated else (80, 180, 255), 2)

        self._draw_top_view(image, (20, 55, 825, 585))
        self._draw_3d_view(image, (845, 55, width - 20, 405))
        self._draw_telemetry(image, (860, 438))

        cv2.rectangle(image, (20, 610), (width - 20, height - 18),
                      (48, 55, 63), -1)
        _put_text(image, "Commands: go 0.1 0.1 | stop | 10 0 100 3 | "
                  "reset | clear | quit", (38, 642), 0.54, (180, 215, 240))
        _put_text(image, self.status, (38, 677), 0.50,
                  (80, 100, 255) if self.status_is_error else (110, 225, 145), 2)
        cv2.rectangle(image, (36, 700), (width - 38, 754), (18, 21, 25), -1)
        prompt = "> " + self.input_buffer
        if int(time.monotonic() * 2.0) % 2 == 0:
            prompt += "_"
        _put_text(image, prompt[-105:], (50, 735), 0.65, (245, 245, 245), 2)
        _put_text(image, "Esc also exits. Up/Down recall command history.",
                  (38, 783), 0.42, (145, 153, 162))
        return image

    def handle_key(self, key):
        if key < 0:
            return
        # Common Windows/Linux OpenCV key codes for arrows.
        if key in (2490368, 65362):
            if self.history and self.history_index > 0:
                self.history_index -= 1
                self.input_buffer = self.history[self.history_index]
            return
        if key in (2621440, 65364):
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.input_buffer = self.history[self.history_index]
            else:
                self.history_index = len(self.history)
                self.input_buffer = ""
            return
        ascii_key = key & 0xFF
        if ascii_key == 27:
            self.running = False
        elif ascii_key in (10, 13):
            self.submit_input()
        elif ascii_key in (8, 127):
            self.input_buffer = self.input_buffer[:-1]
        elif 32 <= ascii_key <= 126 and len(self.input_buffer) < 120:
            self.input_buffer += chr(ascii_key)


def run_live(demo):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 800)
    previous = time.monotonic()
    try:
        while demo.running:
            now = time.monotonic()
            demo.update(now - previous)
            previous = now
            cv2.imshow(WINDOW_NAME, demo.render())
            demo.handle_key(cv2.waitKeyEx(16))
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


def run_smoke_test(demo, output):
    demo.execute("go 0.1 0.12")
    demo.execute("10 0 100 5")
    for _ in range(120):
        demo.update(1.0 / 60.0)
    if demo.gripper_orientation == "vertical":
        assert demo._gripper_angle_deg() < 0.0  # large t must move down
    image = demo.render()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError("OpenCV failed to write " + str(output))
    print(output.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenCV command-input animation for the integrated model")
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "robot_calibration.template.yaml")
    parser.add_argument("--arm-speed", type=float, default=1.0,
                        help="maximum simulated joint speed [rad/s]")
    parser.add_argument("--max-wheel-speed", type=float, default=1.0,
                        help="absolute demo wheel-speed limit [m/s]")
    parser.add_argument("--gripper-orientation",
                        choices=("vertical", "lateral"), default="vertical",
                        help="vertical matches the user's 90-degree rotated mount")
    parser.add_argument("--gripper-sweep-deg", type=float, default=90.0,
                        help="visual proxy sweep from small-t up to large-t down")
    parser.add_argument("--view-azimuth-deg", type=float, default=-45.0,
                        help="3D camera azimuth; -45 is the requested +90-deg view rotation")
    parser.add_argument("--smoke-test", metavar="OUTPUT_PNG",
                        help="render a deterministic frame without opening a window")
    args = parser.parse_args(argv)
    if (args.arm_speed <= 0.0 or args.max_wheel_speed <= 0.0 or
            not 0.0 < args.gripper_sweep_deg <= 180.0):
        parser.error("speeds must be positive and gripper sweep in (0, 180]")
    model = IntegratedRobotCameraModel.from_yaml(args.config)
    demo = MotionDemo(model, args.arm_speed, args.max_wheel_speed,
                      gripper_orientation=args.gripper_orientation,
                      gripper_sweep_deg=args.gripper_sweep_deg,
                      view_azimuth_deg=args.view_azimuth_deg)
    if args.smoke_test:
        run_smoke_test(demo, args.smoke_test)
    else:
        run_live(demo)


if __name__ == "__main__":
    main()
