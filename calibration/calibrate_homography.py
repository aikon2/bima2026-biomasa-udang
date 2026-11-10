"""Compute the image-px -> plane-cm homography from >= 4 marker point pairs.

Place markers with known positions (cm) on the measurement plane (e.g. the
feeding tray, in water), note their pixel positions in a captured frame, and
list them in a CSV with columns u,v,x_cm,y_cm.

Usage: python calibration/calibrate_homography.py --points markers.csv [--dry-run]
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_yaml, resolve_path  # noqa: E402
from src.geometry import compute_homography, reprojection_errors_cm  # noqa: E402

LOGGER = logging.getLogger("calibrate_homography")

DEFAULT_OUT = "config/calibration.yaml"
POINT_COLUMNS = ("u", "v", "x_cm", "y_cm")
CONFIG_HEADER = ("# Pixel -> measurement-plane scale. cm_per_pixel is the fallback when\n"
                 "# homography is null. homography written by calibrate_homography.py.\n")


def _parse_point_row(row: dict, line_number: int) -> List[float]:
    try:
        return [float(row[column]) for column in POINT_COLUMNS]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid point row at line {}: {}".format(line_number, error))


def load_point_pairs(csv_path) -> Tuple[np.ndarray, np.ndarray]:
    """Read u,v,x_cm,y_cm rows (blank lines skipped) into Nx2 px and Nx2 cm arrays."""
    values = []
    with open(str(resolve_path(csv_path)), "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not any((value or "").strip() for value in row.values()):
                continue
            values.append(_parse_point_row(row, reader.line_num))
    points = np.asarray(values, dtype=np.float64).reshape(-1, len(POINT_COLUMNS))
    return points[:, :2], points[:, 2:]


def write_homography(homography: Sequence[Sequence[float]], config_path) -> Path:
    """Set `homography` in the calibration YAML, keeping the other keys."""
    target = resolve_path(config_path)
    config = load_yaml(target) if target.is_file() else {}
    config["homography"] = [[float(value) for value in row] for row in homography]
    with open(str(target), "w") as config_file:
        config_file.write(CONFIG_HEADER)
        config_file.write(yaml.safe_dump(config, default_flow_style=None))
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute px->cm plane homography.")
    parser.add_argument("--points", required=True,
                        help="CSV with columns u,v,x_cm,y_cm (>= 4 rows).")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Calibration YAML to update.")
    parser.add_argument("--dry-run", action="store_true", help="Print only, do not write.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    try:
        points_px, points_cm = load_point_pairs(args.points)
        homography = compute_homography(points_px, points_cm)
    except (OSError, ValueError) as error:
        LOGGER.error("Homography calibration failed: %s", error)
        return 1
    errors_cm = reprojection_errors_cm(homography, points_px, points_cm)
    print("Homography (px -> cm):")
    for row in homography:
        print("  [{}]".format(", ".join("{:.6g}".format(value) for value in row)))
    print("Reprojection error: mean {:.3f} cm, max {:.3f} cm over {} points".format(
        float(errors_cm.mean()), float(errors_cm.max()), errors_cm.size))
    if args.dry_run:
        return 0
    print("Wrote {}".format(write_homography(homography, args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
