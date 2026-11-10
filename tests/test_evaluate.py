import json
import math

import pytest

from evaluation.evaluate import (compute_metrics, evaluate_rows, find_leakage, load_rows,
                                 main)
from src.config import REPO_ROOT

EXAMPLE_CSV = REPO_ROOT / "evaluation" / "data.example.csv"
CALIBRATION_EXAMPLE_CSV = REPO_ROOT / "calibration" / "calibration_data.example.csv"


def test_perfect_predictions():
    metrics = compute_metrics([5.0, 8.0, 11.0], [5.0, 8.0, 11.0])
    assert metrics["n"] == 3
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape_pct"] == 0.0
    assert metrics["bias"] == 0.0
    assert metrics["r2"] == 1.0


def test_known_values_hand_computed():
    # errors (pred - gt): +1, -1, +2 ; gt = 10, 20, 40
    metrics = compute_metrics([10.0, 20.0, 40.0], [11.0, 19.0, 42.0])
    assert metrics["mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(6.0 / 3.0))
    assert metrics["mape_pct"] == pytest.approx((0.1 + 0.05 + 0.05) / 3.0 * 100.0)
    assert metrics["bias"] == pytest.approx(2.0 / 3.0)
    gt_mean = 70.0 / 3.0
    sst = sum((value - gt_mean) ** 2 for value in (10.0, 20.0, 40.0))
    assert metrics["r2"] == pytest.approx(1.0 - 6.0 / sst)


def test_zero_ground_truth_excluded_from_mape_only():
    metrics = compute_metrics([0.0, 10.0], [1.0, 11.0])
    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["mape_pct"] == pytest.approx(10.0)
    assert metrics["n_mape"] == 1


def test_small_samples_return_none_instead_of_crashing():
    single = compute_metrics([10.0], [11.0])
    assert single["n"] == 1
    assert single["r2"] is None
    assert single["mae"] == pytest.approx(1.0)
    empty = compute_metrics([], [])
    assert empty["n"] == 0
    assert empty["mae"] is None and empty["r2"] is None and empty["mape_pct"] is None


def test_constant_ground_truth_gives_no_r2():
    assert compute_metrics([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])["r2"] is None


def test_evaluate_rows_skips_blank_cells_and_missing_columns():
    rows = [{"length_gt_cm": "10", "length_pred_cm": "11"},
            {"length_gt_cm": "12", "length_pred_cm": ""},
            {"length_gt_cm": "14", "length_pred_cm": "13"}]
    results = evaluate_rows(rows)
    assert results["length"]["n"] == 2
    assert results["weight"] is None


def test_leakage_detection_flags_identical_rows():
    evaluation_rows = [{"length_gt_cm": "8.1", "weight_gt_g": "7.6"},
                       {"length_gt_cm": "10.0", "weight_gt_g": "9.9"}]
    calibration_rows = [{"length_cm": "8.10", "weight_g": "7.6"},
                        {"length_cm": "9.4", "weight_g": "9.0"}]
    assert find_leakage(evaluation_rows, calibration_rows) == [0]


def test_example_data_has_new_columns_and_no_leakage():
    rows = load_rows(EXAMPLE_CSV)
    assert len(rows) >= 3
    assert {"length_gt_cm", "length_pred_cm", "weight_gt_g", "weight_pred_g"} <= set(rows[0])
    assert find_leakage(rows, load_rows(CALIBRATION_EXAMPLE_CSV)) == []


def test_cli_writes_json(tmp_path, capsys):
    json_out = tmp_path / "metrics.json"
    exit_code = main(["--csv", str(EXAMPLE_CSV), "--calibration-csv",
                      str(CALIBRATION_EXAMPLE_CSV), "--json-out", str(json_out)])
    assert exit_code == 0
    saved = json.loads(json_out.read_text())
    assert saved["metrics"]["length"]["n"] >= 3
    assert saved["leakage_rows"] == []
    assert "MAE" in capsys.readouterr().out


def test_cli_missing_file_returns_error(tmp_path):
    assert main(["--csv", str(tmp_path / "missing.csv")]) == 1
