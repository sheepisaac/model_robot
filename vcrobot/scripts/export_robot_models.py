"""Create the ROS/GPU-free Raspberry Pi deployment directory."""

import argparse
from pathlib import Path
import shutil


FILES = ["__init__.py", "transforms.py", "roarm.py", "ugv.py", "camera.py",
         "integrated.py", "odometry.py"]


def export(project_root, destination):
    project_root = Path(project_root).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = project_root / "robot_models"
    for name in FILES:
        shutil.copy2(str(source / name), str(destination / name))
    shutil.copy2(str(project_root / "configs" / "robot_calibration.template.yaml"),
                 str(destination / "calibration.yaml"))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", default=None)
    args = parser.parse_args(argv)
    root = Path(args.project_root)
    destination = (Path(args.destination) if args.destination else
                   root / "export" / "robot_models")
    print(export(root, destination))


if __name__ == "__main__":
    main()
