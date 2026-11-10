import math

import numpy as np
import pytest
import yaml

from calibration.calibrate_ab import (MIN_CALIBRATION_SAMPLES, fit_length_weight, load_samples,
                                      write_biomass_config)

TRUE_A = 0.008
TRUE_B = 3.05
NOISE_SIGMA = 0.1
SAMPLE_COUNT = 40
SEED = 42
FIT_KEYS = {"a", "a_raw", "b", "cf", "r2", "n", "b_se", "b_ci95"}


def synthetic_samples(count=SAMPLE_COUNT):
    rng = np.random.RandomState(SEED)
    lengths = rng.uniform(5.0, 15.0, count)
    weights = TRUE_A * lengths ** TRUE_B * np.exp(rng.normal(0.0, NOISE_SIGMA, count))
    return lengths, weights


def test_fit_recovers_parameters_with_sprugel_correction():
    lengths, weights = synthetic_samples()
    fit = fit_length_weight(lengths, weights)
    assert FIT_KEYS <= set(fit)
    assert fit["n"] == SAMPLE_COUNT
    assert fit["b"] == pytest.approx(TRUE_B, abs=0.1)
    slope, intercept = np.polyfit(np.log(lengths), np.log(weights), 1)
    residuals = np.log(weights) - (slope * np.log(lengths) + intercept)
    see = math.sqrt(np.sum(residuals ** 2) / (SAMPLE_COUNT - 2))
    assert fit["cf"] == pytest.approx(math.exp(see ** 2 / 2.0))
    assert fit["cf"] >= 1.0
    assert fit["a_raw"] == pytest.approx(math.exp(intercept))
    assert fit["a"] == pytest.approx(fit["a_raw"] * fit["cf"])
    assert 0.9 < fit["r2"] <= 1.0
    low, high = fit["b_ci95"]
    assert low < fit["b"] < high
    assert fit["b_se"] > 0


def test_fit_drops_invalid_rows():
    lengths, weights = synthetic_samples()
    dirty_lengths = np.concatenate([lengths, [np.nan, 0.0, -2.0, 10.0]])
    dirty_weights = np.concatenate([weights, [5.0, 5.0, 5.0, np.nan]])
    assert fit_length_weight(dirty_lengths, dirty_weights)["n"] == SAMPLE_COUNT


def test_fit_refuses_small_sample_without_force():
    lengths, weights = synthetic_samples(MIN_CALIBRATION_SAMPLES - 1)
    with pytest.raises(ValueError):
        fit_length_weight(lengths, weights)
    assert fit_length_weight(lengths, weights, force=True)["n"] == MIN_CALIBRATION_SAMPLES - 1


def test_load_samples_skips_blank_and_invalid(tmp_path):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("length_cm,weight_g\n8.1,7.6\n\n9.4,\nabc,3\n10.2,11.5\n\n")
    lengths, weights = load_samples(csv_path)
    assert len(lengths) == len(weights) == 4
    fit_input_valid = np.isfinite(lengths) & np.isfinite(weights)
    assert int(fit_input_valid.sum()) == 2


def test_write_biomass_config_preserves_pond(tmp_path):
    out_path = tmp_path / "biomass.yaml"
    out_path.write_text("a: 0.1\nb: 2.0\npond:\n  stocked_pcs: 50000\n  survival_rate: 0.85\n")
    lengths, weights = synthetic_samples()
    fit = fit_length_weight(lengths, weights)
    write_biomass_config(fit, out_path)
    written = yaml.safe_load(out_path.read_text())
    assert written["pond"] == {"stocked_pcs": 50000, "survival_rate": 0.85}
    assert written["a"] == pytest.approx(fit["a"], rel=1e-4)
    assert written["b"] == pytest.approx(fit["b"], rel=1e-4)
    assert {"a_raw", "cf", "n", "r2"} <= set(written)
