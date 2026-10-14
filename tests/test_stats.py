"""Unit tests for the weighted statistics and stacking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csbs_harm.stats import (
    find_term,
    kish_effective_sample_size,
    stack_outcomes,
    weighted_correlation,
    weighted_mean,
)


def test_weighted_mean_skips_missing_pairs():
    assert weighted_mean([1.0, 0.0, np.nan], [1.0, 3.0, 5.0]) == pytest.approx(0.25)


def test_kish_effective_sample_size_equals_n_for_equal_weights():
    assert kish_effective_sample_size([2.0] * 10) == pytest.approx(10.0)
    assert kish_effective_sample_size([1.0, 3.0]) < 2.0


def test_weighted_correlation_of_identical_series_is_one():
    s = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0])
    assert weighted_correlation(s, s, pd.Series([1, 2, 1, 2, 1])) == pytest.approx(1.0)


def test_find_term_accepts_either_ordering():
    assert find_term(["metric:x"], "metric", "x") == "metric:x"
    assert find_term(["x:metric"], "metric", "x") == "x:metric"
    with pytest.raises(KeyError):
        find_term(["other"], "metric", "x")


def test_stack_outcomes_doubles_rows_and_keeps_clusters():
    df = pd.DataFrame({
        "record_id": ["a", "b"], "survey_year": [2024, 2025], "survey_weight": [1.0, 2.0],
        "x1_staff_awareness_test": [0, 1], "x2_security_monitoring": [1, 0],
        "size_cat": [1, 2], "sector_cat": [3, 4],
    })
    stacked = stack_outcomes(df, [0.0, 1.0], [0.0, 0.0])
    assert len(stacked) == 4
    assert stacked["record_id"].nunique() == 2
    assert stacked["metric"].tolist() == [0, 0, 1, 1]
    assert stacked["outcome"].tolist() == [0.0, 1.0, 0.0, 0.0]
