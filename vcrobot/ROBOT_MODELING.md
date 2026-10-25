# Integrated robot-camera model

This directory adds a ROS-free, NumPy-based mathematical model for the UGV02,
RoArm-M2-S and RealSense L515. It does not send motion commands. The existing
`autonbv.motion.VCrobotTCPMotion` remains the only hardware-control boundary.

## Coordinate contract

All core translations are metres and angles are radians. `T_XY` maps a point
from frame Y into frame X, using column-vector composition:

```text
T_WC = T_WB @ T_BA @ T_AE(q) @ T_EC
```

- W: world/map frame, X forward, Y left, Z up.
- B: planar UGV body frame, X forward, Y left, Z up.
- A: RoArm firmware Cartesian frame. Its origin is on the yaw axis at shoulder
  height; X is forward at base angle zero, Y is left, Z is up.
- E: modeled distal/tool frame.
- C: L515 optical frame, X right, Y down, Z forward.

The firmware frame A is not the ROS URDF `base_link`. The URDF places the base
yaw joint at approximately `[0.010, 0, 0.123059]` m, while firmware declares
`L1=126.06 mm` and returns XYZT without adding L1. Therefore the physical
UGV-to-firmware transform belongs in measured `T_BA`; it is not guessed here.

## Inspected source facts

RoArm values are taken from `RoArm-M2_config.h`: L2 components 236.82/30.00 mm,
EEMode-0 L3 components 280.15/1.73 mm, EEMode-1 L3 length 215.99 mm, and default
wrist extension 67.85/5.98 mm. Joint limits reproduce the control-function
clamps: base +/- pi, shoulder +/- pi/2, elbow servo steps 512..3071 (about
-pi/4..pi), and hand steps 700..3396 (about 1.074..5.209 rad). The FK position
and the manufacturer IK branch are ported from
`RoArm-M2_module.h`. The firmware's EEMode-0 FK omits the small L3 offset angle
that its IK uses; tests retain a 4 mm tolerance rather than hiding this source
inconsistency. In EEMode 0 the fourth servo is a gripper, so arbitrary camera
orientation is not independently controllable.

UGV nominal values come from `ugv_config.h`: 80 mm wheel diameter, 660 encoder
pulses/rev and 172 mm track width for `mainType=2` (UGV Rover). `T=1` L/R
commands feed the wheel-speed PID in m/s. The six-wheel skid-steer chassis is
only approximated as a differential drive; effective track width and left/right
scale factors remain uncalibrated until fitted from repeated motion logs.

L515 projection uses the librealsense optical convention. Device-specific
intrinsics, distortion coefficients, depth scale and depth-to-colour extrinsics
must be queried from the actual camera. The template values are synthetic.

The supplied official source folders are downloaded archives without `.git`
metadata. `references/reference_manifest.yaml` records this limitation and
pins hashes for the inspected files; it does not invent commit SHAs.

## Usage

```python
import numpy as np
from robot_models import IntegratedRobotCameraModel

model = IntegratedRobotCameraModel.from_yaml(
    "configs/robot_calibration.template.yaml")
T_WC = model.compute_camera_pose(
    base_pose=[0.0, 0.0, 0.0],
    joint_angles=[0.0, 0.0, np.pi / 2.0, np.pi],
)
```

The template loads for simulation, but `check_camera_view_reachability()`
refuses to make a hardware-safety claim until both extrinsics are explicitly
marked `CALIBRATED`. The autoNBV bridge in `autonbv/robot_model.py` converts a
validated camera target back to the existing vcrobot `x,y,z,t` command format.

## Incremental odometry and camera motion

`RobotCameraOdometry` keeps the latest UGV pose and RoArm joint state. Each
feedback update reports the body-frame UGV increment, four joint increments,
and the resulting L515 change. It also evaluates two counterfactual camera
changes so the base contribution and arm contribution can be inspected
separately. These two contributions are not generally additive because rigid
transform composition is non-commutative.

```python
import numpy as np
from robot_models import IntegratedRobotCameraModel, RobotCameraOdometry

model = IntegratedRobotCameraModel.from_yaml(
    "configs/robot_calibration.template.yaml")
odometry = RobotCameraOdometry(
    model,
    base_pose=[0.0, 0.0, 0.0],
    joint_angles=[0.0, 0.0, np.pi / 2.0, np.pi],
)

# vcrobot UGV feedback publishes cumulative left/right travel as odl/odr (m).
# The first sample establishes a baseline and therefore returns None.
odometry.update_cumulative_wheel_distances(odl, odr)
update = odometry.update_cumulative_wheel_distances(
    next_odl, next_odr,
    arm_pose={"x": x_mm, "y": y_mm, "z": z_mm, "t": t_rad},
)

print(update.base_delta_body)              # [forward, left, yaw] in B_old
print(update.joint_delta)                  # four joint changes in radians
print(update.camera_from_base.distance_m)  # isolated base contribution
print(update.camera_from_arm.distance_m)   # isolated arm contribution
print(update.camera_total.rotation_rad)    # actual combined camera change
```

`update_encoder_counts()` accepts cumulative raw wheel counts instead.
`update_wheel_velocity()` accepts left/right m/s and a time interval, while
`update_absolute()` can consume an externally estimated UGV pose. RoArm
feedback can be supplied either as four joint radians or the existing vcrobot
firmware `x,y,z,t` values; XYZT is converted back through the firmware IK.

This is wheel/kinematic odometry, not an absolute localization system. It will
accumulate skid, wheel slip, encoder scale and timing errors. The isolated and
total camera relative transforms are expressed in the previous camera frame.
Their numerical accuracy also depends on synchronized feedback and calibrated
`T_BA`, `T_EC`, wheel scales and effective track width. The report's
`calibrated` field remains false until both extrinsics are marked calibrated.

## OpenCV interactive motion demo

`visualization/opencv_motion_demo.py` is a simulation-only test tool. It does
not import pyserial, open a serial port, or move physical hardware. The window
itself accepts keyboard input after it has focus:

```text
go 0.1 0.1       continuously drive left/right wheels in m/s
stop              stop the UGV (the arm keeps its current target)
100 0 480 4.1     set RoArm target as x_mm y_mm z_mm t_rad
reset             reset UGV pose, arm pose and trails
clear             clear trails without changing pose
quit              close the demo
```

Run it from WSL with WSLg/OpenCV GUI support:

```bash
cd /mnt/c/Research/autoNBV/model/vcrobot
python3 visualization/opencv_motion_demo.py
```

Click the window before typing. The top view follows the UGV and shows the UGV,
arm, a gripper-motion proxy, L515 optical direction and motion trails. The demo
defaults to the user's 90-degree rotated gripper mounting: small EEMode-0 `t`
values move the yellow member upward and large `t` values move it downward.
The currently unmeasured travel is rendered as a linear +/-45-degree proxy.
The right panel is
an OpenCV-rendered orthographic 3D world view with an XYZ ground grid, UGV
cuboid, arm links, gripper and L515 optical axis. Live telemetry includes UGV pose, joints,
gripper proxy angle, camera position, and the most recent base/camera increments.
Its default azimuth rotates the previous view clockwise by 90 degrees in the XY
plane, making world +X project toward the viewer. This is a display-camera
change only and does not alter FK, IK, odometry, or any robot coordinates. A
different view can be selected with `--view-azimuth-deg DEGREES`.

To compare against the stock lateral mounting, run with
`--gripper-orientation lateral`. Once measured endpoint angles are available,
replace the default proxy range with `--gripper-sweep-deg DEGREES`.
This changes the displayed gripper member only; `T_EC` remains a fixed
tool-to-camera transform. If the L515 is physically attached to that moving
member, its `t`-dependent camera transform must be calibrated separately.

The real firmware accepts some Cartesian targets whose IK angles exceed a
servo range and silently clamps the servo command. The demo mirrors that
behaviour: for example, `10 0 100 3` is accepted but reports the shoulder clamp
and the resulting reached-position error. Strict planning and reachability
checks still use `solve_ik()` and reject such a target; only this command-style
demo uses `solve_ik_firmware()`.

Up/Down recalls command history; Esc exits. If OpenCV is missing on Raspberry
Pi OS, install the distribution's `python3-opencv` package. A headless render
check is also available:

```bash
python3 visualization/opencv_motion_demo.py \
  --smoke-test opencv_motion_demo_preview.png
```

## Validation and export

```bash
python3 -m unittest discover -s tests -v
python3 calibration/validate_robot_model.py \
  configs/robot_calibration.template.yaml
python3 visualization/visualize_robot.py \
  configs/robot_calibration.template.yaml --output robot_model.png
python3 scripts/export_robot_models.py
```

The export contains only `robot_models`, NumPy/PyYAML-compatible code and a
calibration template. It has no ROS, MoveIt, CUDA, PyTorch, visualization or
physical device dependency.

## Remaining measurements

1. Confirm the installed RoArm firmware revision and active EEMode.
2. Measure/calibrate `T_BA` for the actual UGV mount.
3. Calibrate `T_EC` with diverse poses; the provided script rejects weak
   rotation sets, which are a real risk for this 4-DOF arm.
4. Export L515 colour/depth intrinsics, distortion, depth scale and stream
   extrinsics from the connected device.
5. Fit UGV left/right speed scales and effective track width using repeated
   straight, reverse and turning trials with an external pose reference.
6. Verify camera/arm clearance and collision envelopes on the physical robot;
   the plot uses proxy geometry and is not a collision checker.
