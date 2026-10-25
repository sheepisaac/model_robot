"""Create the ROS/GPU-free Raspberry Pi deployment directory."""

import argparse
from pathlib import Path
import shutil


FILES = ["__init__.py", "transforms.py", "roarm.py", "ugv.py", "camera.py",
         "gripper.py", "integrated.py", "odometry.py"]


def export(project_root, destination, calibration=None):
    project_root = Path(project_root).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = project_root / "robot_models"
    for name in FILES:
        shutil.copy2(str(source / name), str(destination / name))
    calibration = (Path(calibration).resolve() if calibration else
                   project_root / "configs" / "robot_calibration.template.yaml")
    if not calibration.is_file():
        raise FileNotFoundError("calibration file not found: {}".format(calibration))
    shutil.copy2(str(calibration), str(destination / "calibration.yaml"))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", default=None)
    parser.add_argument(
        "--calibration", default=None,
        help="configuration to export (defaults to the uncalibrated template)")
    args = parser.parse_args(argv)
    root = Path(args.project_root)
    destination = (Path(args.destination) if args.destination else
                   root / "export" / "robot_models")
    print(export(root, destination, args.calibration))


if __name__ == "__main__":
    main()
