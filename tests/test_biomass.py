import math

import pytest

from src.biomass import summarize_session, weight_g

SUMMARY_KEYS = {"n_detected", "n_measured", "abw_g", "size_pcs_per_kg", "cv_weight_pct",
                "length_mean_cm", "length_std_cm", "sampled_biomass_g", "pond_biomass_kg"}


def test_weight_g_power_law():
    assert weight_g(10, 0.01, 3.0) == pytest.approx(10.0)


@pytest.mark.parametrize("length", [None, 0, -1.5, float("nan"), float("inf")])
def test_weight_g_invalid_length_is_none(length):
    assert weight_g(length, 0.01, 3.0) is None


def test_summarize_session_metrics():
    weights = [10.0, 20.0, 30.0]
    summary = summarize_session([8.0, 10.0, 12.0], weights, n_detected=5)
    assert set(summary) == SUMMARY_KEYS
    assert summary["n_detected"] == 5
    assert summary["n_measured"] == 3
    assert summary["abw_g"] == pytest.approx(20.0)
    assert summary["size_pcs_per_kg"] == pytest.approx(50.0)
    assert summary["cv_weight_pct"] == pytest.approx(50.0)  # sample std 10 / mean 20
    assert summary["length_mean_cm"] == pytest.approx(10.0)
    assert summary["length_std_cm"] == pytest.approx(2.0)
    assert summary["sampled_biomass_g"] == pytest.approx(60.0)
    assert summary["pond_biomass_kg"] is None


def test_summarize_session_ignores_unmeasured_entries():
    summary = summarize_session([8.0, None, float("nan")], [10.0, None, float("nan")], 3)
    assert summary["n_measured"] == 1
    assert summary["abw_g"] == pytest.approx(10.0)
    assert summary["cv_weight_pct"] is None


def test_summarize_session_pond_biomass():
    pond = {"stocked_pcs": 100000, "survival_rate": 0.8}
    summary = summarize_session([10.0, 12.0], [15.0, 25.0], 2, pond_cfg=pond)
    assert summary["pond_biomass_kg"] == pytest.approx(20.0 * 100000 * 0.8 / 1000.0)


def test_summarize_session_empty_input():
    summary = summarize_session([], [], 0, pond_cfg={"stocked_pcs": 1000, "survival_rate": 0.9})
    assert summary["n_measured"] == 0
    assert summary["abw_g"] is None
    assert summary["size_pcs_per_kg"] is None
    assert summary["cv_weight_pct"] is None
    assert summary["length_mean_cm"] is None
    assert summary["sampled_biomass_g"] == 0.0
    assert summary["pond_biomass_kg"] is None
    assert not math.isnan(summary["sampled_biomass_g"])
