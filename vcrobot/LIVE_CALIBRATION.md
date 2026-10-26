# Live robot/model calibration

This workflow connects the mathematical model to the existing `vcrobot`
masters and robot-side slaves. It compares three independent motion estimates:

1. the command and integrated mathematical model;
2. UGV encoder or RoArm Cartesian firmware feedback;
3. metric L515 RGB-D visual odometry.

The L515 estimate is the IMU substitute. ORB features are deprojected with the
reference depth image, then a RANSAC PnP solve estimates the current camera
pose in metres. It is suitable for short, stopped-between-samples calibration
motions. It is not a global localization system and cannot eliminate drift,
texture loss, moving-object errors, or skid-steer slip on its own.

## Data flow and safety boundary

```text
PC live_robot_calibration.py
  +-- TCP/50220 -> drive_sync.py -> UGV serial + T=1001 odl/odr feedback
  +-- TCP/50210 -> slave_arm.py  -> RoArm serial + T=1051 x/y/z/t feedback
  `-- TCP/50240 -> slave_rgbd_odometry.py or slave_v4l2_odometry.py
                    -> L515 only (never motors)
```

Physical motion is disabled unless the master is launched with both
`--execute` and the exact acknowledgement token. The template has an empty
command list. Each UGV trial schedules a normal stop, repeats the stop in a
`finally` block, and the existing drive slave retains its disconnect watchdog.
Hard limits in the YAML reject oversized wheel commands, durations, arm steps,
tool-angle steps, and arm speed.

Live RoArm feedback is converted with the firmware-compatible IK path. This
allows a measured pose already resting on a servo clamp to be represented by
the model; post-command Cartesian feedback verification still rejects a target
the physical arm did not reach.

On a normal stopped shutdown the master requests a checked clean release, so
the drive connection can close without the global `T=0` that also releases arm
torque. The slave refuses that release if a job is pending or either wheel is
not at zero; an abnormal disconnect therefore keeps the original emergency
`T=0` behavior.

Do not rely on software limits as collision protection. Put the robot on a
clear, level test area, keep an emergency-stop method within reach, begin below
the template limits, and supervise every trial.

## 1. Deploy the updated services

On the robot, update these existing control files from
`C:/Research/vcrobot/scripts/ctrl_scripts`:

- `drive_sync.py`: adds fresh `T=1001` UGV telemetry requests;
- `slave_arm.py`: adds a read-only Cartesian feedback request.

Start the normal UGV and arm slaves exactly as before. Their default ports are
50220 and 50210. The ROS firmware emits `odl` and `odr` as cumulative integer
centimetres. The General Driver's valid T=1001 response omits those fields;
live calibration then records firmware encoder comparison as unavailable and
uses L515 visual odometry as the metric UGV reference. It does not synthesize
encoder measurements.

Copy both files below to one directory on the Raspberry Pi:

```text
calibration/rgbd_visual_odometry.py
calibration/slave_rgbd_odometry.py
```

Install the camera dependencies using the RealSense packages appropriate for
the Pi image, plus NumPy and OpenCV. Then start the camera service:

```bash
python3 slave_rgbd_odometry.py --listen 0.0.0.0 --port 50240
```

### V4L2-only L515 service (no ROS or librealsense)

On the tested L515, V4L2 enumerates registered 8-bit IR on `/dev/video0`, Z16
depth on `/dev/video2`, packed confidence on `/dev/video4`, and YUYV colour on
`/dev/video6`. The V4L2 service deliberately uses IR plus depth, so it does not
need colour-to-depth alignment:

The depth FourCC is `'Z16 '` with a significant trailing space. The service
passes its unambiguous numeric representation `0x2036315a` to `v4l2-ctl`;
plain `pixelformat=Z16` is rejected by this driver.

```bash
sudo apt install -y v4l-utils python3-numpy python3-opencv

python3 slave_v4l2_odometry.py \
  --listen 0.0.0.0 --port 50240 \
  --ir-device /dev/video0 --depth-device /dev/video2 \
  --width 480 --height 640 --fps 30 \
  --depth-scale-m ACTUAL_METRES_PER_Z16_UNIT \
  --fx ACTUAL_IR_FX --fy ACTUAL_IR_FY \
  --ppx ACTUAL_IR_PPX --ppy ACTUAL_IR_PPY
```

The width and height above intentionally follow the raw V4L2 profiles reported
by this camera, rather than silently assuming librealsense's orientation. Use
intrinsics for that exact raw GREY profile. V4L2 provides pixels but does not
provide the RealSense factory calibration, so the service requires explicit
depth scale and IR intrinsics. Guessed or template values must not be used to
accept fitted robot parameters. The service labels them
`MANUAL_REQUIRES_VALIDATION` in every hello/status response.

`slave_v4l2_odometry.py` uses the same protocol and port as the pyrealsense2
service. No PC master or live-calibration YAML change is required. Only one of
the two odometry services may run at a time.

Only one process can own the L515 pipeline. Stop the regular image-acquisition
slave while this calibration service is running.

## 2. Configure and run the read-only probe

Copy the safe template and replace only the robot IP first:

```bash
cd /mnt/c/Research/autoNBV/model/vcrobot
cp configs/live_calibration.template.yaml configs/live_calibration.local.yaml
```

The local file should normally remain uncommitted because it identifies a
specific robot/network. Probe all feedback paths without a movement command:

```bash
python3 calibration/live_robot_calibration.py \
  configs/live_calibration.local.yaml --probe
```

A successful result contains UGV `odl/odr`, arm `x/y/z/t`, L515 intrinsics and
the model calibration status. Resolve any missing service or telemetry error
before putting commands in the plan.

## 3. Add small calibration commands

The command type is inferred from the same input shapes used by the existing
controllers. UGV trials additionally require a duration so the stop can be
scheduled before motion begins:

```yaml
commands:
  - "go 0.08 0.08 1.0"
  - "go -0.06 0.06 0.8"
  - "go 0.06 -0.06 0.8"
  - "10 0 100 3.0"
```

For a safe incremental arm test that cannot become stale when watchdog or
torque recovery changes the held pose, use fresh-feedback-relative syntax:

```yaml
commands:
  - "arm_delta 0 0 10 0"  # current x/y/t, current z + 10 mm
```

The relative target is resolved only after reading arm feedback immediately
before the trial, then passes through the same Cartesian-step, tool-angle,
firmware IK, speed, and post-motion feedback checks as an absolute target.

For a useful UGV fit, collect repeated straight, left-turn, and right-turn
motions. For fixed-tool-angle eye-in-hand calibration, collect at least four
consecutive arm motions with the same `t` and diverse arm rotations. Commands
that change `t` are still logged and compared with the dynamic gripper model,
but are excluded from the fixed-`t` hand-eye fit.

After physically clearing the work area, run:

```bash
python3 calibration/live_robot_calibration.py \
  configs/live_calibration.local.yaml \
  --execute \
  --acknowledge-motion I_HAVE_CLEARED_THE_ROBOT
```

## Outputs and acceptance

The JSONL log records every command, raw feedback, timestamps, L515 quality
metrics, predicted and measured transforms, and their residual transform. The
sidecar `*.candidates.yaml` contains:

- UGV left/right metres-per-second-per-command scales and effective track width;
- a fixed-`t` hand-eye `T_EC` candidate when the arm motion set is observable.

Candidates are deliberately labelled `CANDIDATE_REQUIRES_REPEAT_VALIDATION`.
Do not copy them into `robot_calibration.template.yaml` after one run. Repeat
the sequence in both directions, reject high reprojection/model residuals, and
validate on a held-out motion sequence before marking any field calibrated.

The moving gripper still needs several validated fixed-`t` datasets to derive
the final `t -> T_EG(t)` lookup and `T_GC`; a single fixed-`t` hand-eye result
cannot identify both independently.

## Live all-in-one OpenCV console

`visualization/live_robot_console.py` connects to the same three services and
combines guarded physical control, firmware feedback, L515 visual odometry,
the 2D base trail, and the 3D arm/gripper/camera model in one OpenCV window.
It appends each completed or failed transaction to
`results/live_control.jsonl` by default.

Read-only connection and visualization:

```bash
python3 visualization/live_robot_console.py \
  configs/live_calibration.local.yaml
```

After clearing the physical workspace, enable typed motion commands:

```bash
python3 visualization/live_robot_console.py \
  configs/live_calibration.local.yaml \
  --execute \
  --acknowledge-motion I_HAVE_CLEARED_THE_ROBOT \
  --output results/live_control.jsonl
```

Supported window commands are:

```text
go 0.06 0.06             # bounded 0.6-second pulse, then automatic stop
go 0.06 -0.06 0.8        # explicit duration in seconds
100 0 480 4.1            # one direct absolute RoArm target, as ctrl_armSimple
arm_delta 0 0 10 0       # fresh-feedback-relative RoArm move
stop                      # send a normal UGV stop when the worker is idle
probe                     # refresh all read-only service feedback
reference                 # reset the L515 VO reference frame
zero                      # reset display world origin; do not move hardware
clear                     # clear visualization trails
quit                      # safe shutdown
```

Only one bounded transaction is accepted at a time. `go L R` is deliberately
converted to a short pulse rather than a latched velocity; its duration can be
changed with `--default-drive-duration` but cannot exceed the YAML safety
limit. Closing the window waits for the active bounded transaction and sends a
final normal stop before releasing the drive connection.

The four-number `X Y Z T` form deliberately preserves the existing
`ctrl_armSimple.py` behavior: it sends one direct absolute firmware target,
even when the difference from the current pose exceeds the conservative
calibration step limits. Firmware-compatible IK/joint clamps and post-command
Cartesian feedback verification still apply. Use `arm_delta` for incremental
calibration motion; it continues to enforce `max_arm_xyz_step_mm` and
`max_arm_t_step_rad`. The non-interactive `live_robot_calibration.py` command
plan also retains both limits.

In the interactive console, a verified RoArm command or a completed bounded
UGV pulse remains successful when the L515 rejects the corresponding frame
(for example, a textureless image with no ORB descriptors). The JSONL record
then contains `visual_odometry_available: false`, the sensor error, and the
model-predicted transform. The OpenCV arm is updated from firmware feedback;
the UGV display falls back to nominal command motion because the General
Driver has no wheel odometry. Batch calibration still requires VO and aborts
that sample, because silently fitting a calibration without a metric
observation would be invalid.
