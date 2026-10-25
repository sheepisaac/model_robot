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

On a normal stopped shutdown the master requests a checked clean release, so
the drive connection can close without the global `T=0` that also releases arm
torque. The slave refuses that release if a job is pending or either wheel is
not at zero; an abnormal disconnect therefore keeps the original emergency
`T=0` behavior.

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
50220 and 50210. The installed UGV firmware must be the ROS profile that emits
`odl` and `odr` as cumulative integer centimetres.

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
