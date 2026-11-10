"""Fit the shrimp length-weight model W = a * L^b from measured samples.

Log-log OLS (numpy only, runs on Jetson) with the Sprugel (1983) bias
correction CF = exp(SEE^2 / 2) applied to a, because back-transforming a
log-scale fit otherwise underestimates the mean weight.

Usage: python calibration/calibrate_ab.py --csv my_samples.csv [--dry-run] [--force]
"""
import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_yaml, resolve_cli_path, resolve_path  # noqa: E402

LOGGER = logging.getLogger("calibrate_ab")

MIN_CALIBRATION_SAMPLES = 10
RECOMMENDED_CALIBRATION_SAMPLES = 30
MIN_FIT_SAMPLES = 3
FIT_PARAMETERS = 2
DEFAULT_CSV = "calibration/calibration_data.example.csv"
DEFAULT_OUT = "config/biomass.yaml"
LENGTH_COLUMN = "length_cm"
WEIGHT_COLUMN = "weight_g"
OUTPUT_PRECISION = "{:.6g}"
# Two-sided 95% Student-t critical values for df = 1..30 (normal approx beyond).
T_CRITICAL_95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
                 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
                 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
Z_CRITICAL_95 = 1.96


def _t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= len(T_CRITICAL_95):
        return T_CRITICAL_95[degrees_of_freedom - 1]
    return Z_CRITICAL_95


def _parse_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def load_samples(csv_path) -> Tuple[np.ndarray, np.ndarray]:
    """Read length/weight columns; blank lines skipped, unparsable cells -> NaN."""
    lengths, weights = [], []
    with open(str(resolve_path(csv_path)), "r", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if not any((value or "").strip() for value in row.values()):
                continue
            lengths.append(_parse_float(row.get(LENGTH_COLUMN)))
            weights.append(_parse_float(row.get(WEIGHT_COLUMN)))
    return np.asarray(lengths, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _clean_samples(lengths_cm, weights_g) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths_cm, dtype=np.float64)
    weights = np.asarray(weights_g, dtype=np.float64)
    valid = np.isfinite(lengths) & np.isfinite(weights) & (lengths > 0) & (weights > 0)
    dropped = int(lengths.size - valid.sum())
    if dropped:
        LOGGER.warning("Dropped %d blank/invalid/non-positive rows", dropped)
    return lengths[valid], weights[valid]


def _check_sample_size(count: int, force: bool) -> None:
    if count < MIN_FIT_SAMPLES:
        raise ValueError("Need at least {} valid samples to fit, got {}".format(
            MIN_FIT_SAMPLES, count))
    if count < MIN_CALIBRATION_SAMPLES and not force:
        raise ValueError("Only {} valid samples (< {}); use force=True / --force".format(
            count, MIN_CALIBRATION_SAMPLES))
    if count < RECOMMENDED_CALIBRATION_SAMPLES:
        LOGGER.warning("Only %d samples; >= %d are recommended", count,
                       RECOMMENDED_CALIBRATION_SAMPLES)


def fit_length_weight(lengths_cm, weights_g, force: bool = False) -> Dict[str, object]:
    """Log-log OLS fit of W = a * L^b with Sprugel-corrected a."""
    lengths, weights = _clean_samples(lengths_cm, weights_g)
    count = int(lengths.size)
    _check_sample_size(count, force)
    log_length, log_weight = np.log(lengths), np.log(weights)
    slope, intercept = np.polyfit(log_length, log_weight, 1)
    residuals = log_weight - (slope * log_length + intercept)
    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((log_weight - log_weight.mean()) ** 2))
    degrees_of_freedom = count - FIT_PARAMETERS
    see = math.sqrt(sse / degrees_of_freedom)
    correction = math.exp(see ** 2 / 2.0)
    spread = float(np.sum((log_length - log_length.mean()) ** 2))
    b_se = see / math.sqrt(spread) if spread > 0 else float("inf")
    margin = _t_critical_95(degrees_of_freedom) * b_se
    a_raw = math.exp(intercept)
    return {
        "a": a_raw * correction, "a_raw": a_raw, "b": float(slope), "cf": correction,
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"), "n": count,
        "see": see, "b_se": b_se, "b_ci95": (float(slope) - margin, float(slope) + margin),
    }


def write_biomass_config(fit: Dict[str, object], out_path) -> Path:
    """Write a, b, a_raw, cf, n, r2 to YAML, preserving an existing pond: section."""
    target = resolve_path(out_path)
    existing = load_yaml(target) if target.is_file() else {}
    lines = ["# Written by calibration/calibrate_ab.py (a includes Sprugel CF)."]
    for key in ("a", "b", "a_raw", "cf", "r2"):
        lines.append("{}: {}".format(key, OUTPUT_PRECISION.format(fit[key])))
    lines.append("n: {}".format(int(fit["n"])))
    text = "\n".join(lines) + "\n"
    if existing.get("pond"):
        text += yaml.safe_dump({"pond": existing["pond"]}, default_flow_style=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(str(target), "w") as out_file:
        out_file.write(text)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit shrimp length-weight model W = a*L^b.")
    parser.add_argument("--csv", type=resolve_cli_path, default=str(resolve_path(DEFAULT_CSV)),
                        help="CSV with length_cm,weight_g columns (relative to the current directory).")
    parser.add_argument("--out", type=resolve_cli_path, default=str(resolve_path(DEFAULT_OUT)),
                        help="Output biomass YAML.")
    parser.add_argument("--force", action="store_true",
                        help="Allow fewer than {} samples.".format(MIN_CALIBRATION_SAMPLES))
    parser.add_argument("--dry-run", action="store_true", help="Print the fit, do not write.")
    return parser.parse_args()


def print_fit(fit: Dict[str, object]) -> None:
    low, high = fit["b_ci95"]
    print("=== Length-weight calibration (W = a * L^b) ===")
    print("n      = {}".format(fit["n"]))
    print("a      = {:.6g}  (a_raw = {:.6g}, Sprugel CF = {:.4f})".format(
        fit["a"], fit["a_raw"], fit["cf"]))
    print("b      = {:.4f}  (SE = {:.4f}, 95% CI = [{:.4f}, {:.4f}])".format(
        fit["b"], fit["b_se"], low, high))
    print("R^2    = {:.4f}  (log-log), SEE = {:.4f}".format(fit["r2"], fit["see"]))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    try:
        lengths, weights = load_samples(args.csv)
        fit = fit_length_weight(lengths, weights, force=args.force)
    except (OSError, ValueError) as error:
        LOGGER.error("Calibration failed: %s", error)
        return 1
    print_fit(fit)
    if args.dry_run:
        return 0
    print("Wrote {}".format(write_biomass_config(fit, args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
