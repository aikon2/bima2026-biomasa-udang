"""Evaluate predicted shrimp length and weight against manual ground truth.

Reports n, MAE, RMSE, MAPE, R^2 and bias (mean of pred - gt) separately for
length and weight (numpy only, runs on Jetson). Optionally warns when
evaluation rows duplicate the length-weight calibration rows (data leakage).

Usage: python evaluation/evaluate.py --csv my_eval.csv [--calibration-csv cal.csv]
                                     [--json-out metrics.json]
"""
import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve_cli_path, resolve_path  # noqa: E402

LOGGER = logging.getLogger("evaluate")

DEFAULT_CSV = "evaluation/data.example.csv"
PERCENT = 100.0
MIN_R2_SAMPLES = 2
LEAKAGE_DECIMALS = 3
JSON_INDENT = 2
# (target name, ground-truth column, prediction column, unit)
TARGETS = (("length", "length_gt_cm", "length_pred_cm", "cm"),
           ("weight", "weight_gt_g", "weight_pred_g", "g"))
EVAL_LEAKAGE_COLUMNS = ("length_gt_cm", "weight_gt_g")
CALIBRATION_LEAKAGE_COLUMNS = ("length_cm", "weight_g")
METRIC_COLUMNS = (("n", "n", "{:d}"), ("MAE", "mae", "{:.3f}"), ("RMSE", "rmse", "{:.3f}"),
                  ("MAPE%", "mape_pct", "{:.2f}"), ("R2", "r2", "{:.3f}"),
                  ("bias", "bias", "{:+.3f}"))


def _parse_float(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(csv_path) -> List[Dict[str, str]]:
    """Read CSV rows as dicts, skipping fully blank lines."""
    with open(str(resolve_path(csv_path)), "r", newline="") as csv_file:
        return [row for row in csv.DictReader(csv_file)
                if any((value or "").strip() for value in row.values())]


def _paired_values(rows: Sequence[dict], gt_column: str, pred_column: str):
    gt = np.asarray([_parse_float(row.get(gt_column)) for row in rows], dtype=np.float64)
    pred = np.asarray([_parse_float(row.get(pred_column)) for row in rows], dtype=np.float64)
    valid = np.isfinite(gt) & np.isfinite(pred)
    return gt[valid], pred[valid]


def _r2(gt: np.ndarray, residuals: np.ndarray) -> Optional[float]:
    if gt.size < MIN_R2_SAMPLES:
        return None
    sst = float(np.sum((gt - gt.mean()) ** 2))
    if sst <= 0:
        return None
    return 1.0 - float(np.sum(residuals ** 2)) / sst


def compute_metrics(ground_truth: Sequence[float], predicted: Sequence[float]) -> dict:
    """Error metrics; MAPE excludes gt == 0 rows; R^2 None for n < 2 or constant gt."""
    gt = np.asarray(ground_truth, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    count = int(gt.size)
    if count == 0:
        return {"n": 0, "n_mape": 0, "mae": None, "rmse": None, "mape_pct": None,
                "r2": None, "bias": None}
    residuals = pred - gt
    nonzero = gt != 0
    mape = None
    if np.any(nonzero):
        mape = float(np.mean(np.abs(residuals[nonzero] / gt[nonzero])) * PERCENT)
    return {
        "n": count,
        "n_mape": int(np.count_nonzero(nonzero)),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(math.sqrt(np.mean(residuals ** 2))),
        "mape_pct": mape,
        "r2": _r2(gt, residuals),
        "bias": float(np.mean(residuals)),
    }


def evaluate_rows(rows: Sequence[dict]) -> Dict[str, Optional[dict]]:
    """Metrics per target; None when the target's columns are absent."""
    columns = set(rows[0]) if rows else set()
    results = {}
    for name, gt_column, pred_column, _ in TARGETS:
        if gt_column not in columns or pred_column not in columns:
            results[name] = None
            continue
        gt, pred = _paired_values(rows, gt_column, pred_column)
        results[name] = compute_metrics(gt, pred)
    return results


def _row_key(row: dict, columns: Sequence[str]):
    values = [_parse_float(row.get(column)) for column in columns]
    if not all(math.isfinite(value) for value in values):
        return None
    return tuple(round(value, LEAKAGE_DECIMALS) for value in values)


def find_leakage(evaluation_rows: Sequence[dict],
                 calibration_rows: Sequence[dict]) -> List[int]:
    """Indices of evaluation rows whose (length, weight) ground truth is a calibration row."""
    calibration_keys = set(_row_key(row, CALIBRATION_LEAKAGE_COLUMNS)
                           for row in calibration_rows)
    calibration_keys.discard(None)
    return [index for index, row in enumerate(evaluation_rows)
            if _row_key(row, EVAL_LEAKAGE_COLUMNS) in calibration_keys]


def format_table(results: Dict[str, Optional[dict]]) -> str:
    header = ["target"] + [label for label, _, _ in METRIC_COLUMNS]
    lines = ["{:<12}".format(header[0]) + "".join("{:>10}".format(h) for h in header[1:])]
    for name, _, _, unit in TARGETS:
        metrics = results.get(name)
        if metrics is None:
            continue
        cells = ["-" if metrics[key] is None else template.format(metrics[key])
                 for _, key, template in METRIC_COLUMNS]
        label = "{} ({})".format(name, unit)
        lines.append("{:<12}".format(label) + "".join("{:>10}".format(c) for c in cells))
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted shrimp length/weight against ground truth.")
    parser.add_argument("--csv", type=resolve_cli_path, default=str(resolve_path(DEFAULT_CSV)),
                        help="CSV with length_gt_cm,length_pred_cm and/or "
                             "weight_gt_g,weight_pred_g columns (default: %(default)s).")
    parser.add_argument("--calibration-csv", type=resolve_cli_path, default=None,
                        help="Length-weight calibration CSV (length_cm,weight_g); warns if "
                             "evaluation rows duplicate it (data leakage).")
    parser.add_argument("--json-out", type=resolve_cli_path, default=None, help="Also write metrics as JSON here.")
    return parser.parse_args(argv)


def _check_leakage(rows: Sequence[dict], calibration_csv: Optional[str]) -> List[int]:
    if not calibration_csv:
        return []
    leaked = find_leakage(rows, load_rows(calibration_csv))
    if leaked:
        LOGGER.warning("%d evaluation rows duplicate calibration rows (leakage): %s",
                       len(leaked), leaked)
    return leaked


def _write_json(report: dict, json_out: str) -> None:
    target = resolve_path(json_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(str(target), "w") as json_file:
        json.dump(report, json_file, indent=JSON_INDENT, sort_keys=True)
    LOGGER.info("Metrics written to %s", target)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        rows = load_rows(args.csv)
        leaked = _check_leakage(rows, args.calibration_csv)
    except (OSError, csv.Error) as error:
        LOGGER.error("Evaluation failed: %s", error)
        return 1
    results = evaluate_rows(rows)
    if all(metrics is None for metrics in results.values()):
        LOGGER.error("No length_gt_cm/length_pred_cm or weight_gt_g/weight_pred_g columns")
        return 1
    print(format_table(results))
    if args.json_out:
        report = {"csv": str(resolve_path(args.csv)), "metrics": results,
                  "leakage_rows": leaked}
        _write_json(report, args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
