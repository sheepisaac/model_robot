"""Fit the lightweight UGV model from a CSV and write a YAML result."""

import argparse
import csv
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_models.ugv import UGVModel


def load_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"v_left", "v_right", "duration", "distance_m", "yaw_delta_rad"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError("CSV is missing columns: {}".format(", ".join(sorted(missing))))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_yaml")
    args = parser.parse_args(argv)
    model = UGVModel.estimate_motion_parameters(load_rows(args.input_csv))
    p = model.parameters
    output = {"ugv": {"wheel_diameter_m": p.wheel_diameter_m,
                      "encoder_pulses_per_rev": p.encoder_pulses_per_rev,
                      "effective_track_width_m": p.effective_track_width_m,
                      "left_scale": p.left_scale, "right_scale": p.right_scale,
                      "calibrated": True, "source": str(args.input_csv)}}
    with Path(args.output_yaml).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)


if __name__ == "__main__":
    main()
