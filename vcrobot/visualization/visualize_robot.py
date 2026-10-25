"""Matplotlib proxy visualization of UGV, RoArm and L515 optical axis."""

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; registers old Matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_models import IntegratedRobotCameraModel
from robot_models.transforms import transform_points


def plot_model(model, base_pose, joints, output=None):
    T_WB = np.eye(4)
    T_WB[:3, :3] = [[math.cos(base_pose[2]), -math.sin(base_pose[2]), 0.0],
                     [math.sin(base_pose[2]), math.cos(base_pose[2]), 0.0],
                     [0.0, 0.0, 1.0]]
    T_WB[:3, 3] = [base_pose[0], base_pose[1], 0.0]
    arm_points = transform_points(T_WB.dot(model.T_BA),
                                  model.roarm.joint_positions(joints))
    camera = model.compute_camera_pose(base_pose, joints)
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    # Conservative proxy footprint; not a collision model.
    body = np.array([[-0.16, -0.12, 0], [0.16, -0.12, 0],
                     [0.16, 0.12, 0], [-0.16, 0.12, 0], [-0.16, -0.12, 0]])
    body = transform_points(T_WB, body)
    axis.plot(body[:, 0], body[:, 1], body[:, 2], color="black", label="UGV proxy")
    axis.plot(arm_points[:, 0], arm_points[:, 1], arm_points[:, 2], "o-",
              linewidth=3, label="RoArm")
    optical_end = camera[:3, 3] + 0.15 * camera[:3, 2]
    axis.plot([camera[0, 3], optical_end[0]], [camera[1, 3], optical_end[1]],
              [camera[2, 3], optical_end[2]], color="tab:red", linewidth=3,
              label="L515 +Z optical axis")
    axis.set_xlabel("world X [m]")
    axis.set_ylabel("world Y [m]")
    axis.set_zlabel("world Z [m]")
    if hasattr(axis, "set_box_aspect"):
        axis.set_box_aspect((1, 1, 1))
    axis.legend()
    axis.set_title("UNCALIBRATED" if not model.calibrated else "CALIBRATED")
    if output:
        fig.savefig(output, dpi=160, bbox_inches="tight")
    else:
        plt.show()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output")
    parser.add_argument("--joints", nargs=4, type=float,
                        default=[0.0, 0.0, math.pi / 2.0, math.pi])
    args = parser.parse_args(argv)
    plot_model(IntegratedRobotCameraModel.from_yaml(args.config),
               (0.0, 0.0, 0.0), args.joints, args.output)


if __name__ == "__main__":
    main()
