"""Length-weight model and farmer-relevant session statistics."""
import logging
import math
from typing import Dict, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

GRAMS_PER_KG = 1000.0
PERCENT = 100.0
SAMPLE_STD_DDOF = 1


def _is_valid_positive(value) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def weight_g(length_cm, a: float, b: float) -> Optional[float]:
    """Weight from W = a * L^b; None for missing, non-positive or non-finite length."""
    if not _is_valid_positive(length_cm):
        return None
    return float(a) * float(length_cm) ** float(b)


def _valid_values(values: Sequence) -> np.ndarray:
    return np.asarray([float(value) for value in values if _is_valid_positive(value)],
                      dtype=np.float64)


def _sample_std(values: np.ndarray) -> Optional[float]:
    if values.size <= SAMPLE_STD_DDOF:
        return None
    return float(np.std(values, ddof=SAMPLE_STD_DDOF))


def _pond_biomass_kg(abw_g: Optional[float], pond_cfg: Optional[dict]) -> Optional[float]:
    if abw_g is None or not pond_cfg:
        return None
    stocked = float(pond_cfg["stocked_pcs"])
    survival = float(pond_cfg["survival_rate"])
    return abw_g * stocked * survival / GRAMS_PER_KG


def summarize_session(lengths_cm: Sequence, weights_g: Sequence, n_detected: int,
                      pond_cfg: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """Aggregate measured shrimp into ABW, size (pcs/kg), CV% and biomass estimates.

    Invalid entries (None/NaN/<=0) are ignored. Std/CV use the sample (ddof=1)
    estimator and are None with fewer than two values. Without tracking the same
    shrimp may appear in several frames, so counts are sample counts.
    """
    lengths = _valid_values(lengths_cm)
    weights = _valid_values(weights_g)
    abw_g = float(np.mean(weights)) if weights.size else None
    weight_std = _sample_std(weights)
    cv_weight_pct = None
    if abw_g is not None and weight_std is not None:
        cv_weight_pct = weight_std / abw_g * PERCENT
    return {
        "n_detected": int(n_detected),
        "n_measured": int(weights.size),
        "abw_g": abw_g,
        "size_pcs_per_kg": GRAMS_PER_KG / abw_g if abw_g else None,
        "cv_weight_pct": cv_weight_pct,
        "length_mean_cm": float(np.mean(lengths)) if lengths.size else None,
        "length_std_cm": _sample_std(lengths),
        "sampled_biomass_g": float(np.sum(weights)),
        "pond_biomass_kg": _pond_biomass_kg(abw_g, pond_cfg),
    }
