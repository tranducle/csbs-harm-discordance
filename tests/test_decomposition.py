"""The nesting decomposition must hold exactly on any nested pair of indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csbs_harm.decomposition import decompose


def _nested_sample(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 500
    practice = rng.integers(0, 2, n)
    identified = (rng.random(n) < np.where(practice == 1, 0.3, 0.5)).astype(float)
    harmful = identified * (rng.random(n) < 0.4)
    return pd.DataFrame({
        "x1_staff_awareness_test": practice,
        "y1_identified_incident": identified,
        "y2_identified_harmful_incident": harmful,
        "survey_weight": rng.uniform(0.2, 3.0, n),
    })


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_identity_holds(seed):
    parts = decompose(_nested_sample(seed), "x1_staff_awareness_test", 1e-12)
    total = parts["nesting_component"] + parts["severity_component"]
    assert parts["delta_unadjusted"] == pytest.approx(total, abs=1e-12)
    assert parts["rd_y2_unadjusted"] - parts["rd_y1_unadjusted"] == pytest.approx(parts["delta_unadjusted"])


def test_identity_failure_is_reported():
    df = _nested_sample(7)
    with pytest.raises(AssertionError):
        decompose(df, "x1_staff_awareness_test", -1.0)
