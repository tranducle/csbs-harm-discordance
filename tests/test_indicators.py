"""Unit tests for the survey-item recodes and the incident indicators."""

from __future__ import annotations

import pandas as pd

from csbs_harm.indicators import (
    IMPACT_SUBSTANTIVE,
    OUTCOME_SUBSTANTIVE,
    binary_01,
    categorical_valid,
    derive_harmful_incident,
    derive_impactful_incident,
    recode_trained,
)


def _frame(rows):
    """Build a frame with every outcome and impact item defaulting to 0."""
    base = {c: 0 for c in OUTCOME_SUBSTANTIVE + ["outcome9", "outcome10"]}
    base.update({c: 0 for c in IMPACT_SUBSTANTIVE + ["impact11", "impact12"]})
    return pd.DataFrame([{**base, **row} for row in rows])


def test_binary_and_trained_recodes_map_other_codes_to_missing():
    assert binary_01(pd.Series([0, 1, 2, -97, 997])).isna().tolist() == [False, False, True, True, True]
    trained = recode_trained(pd.Series([1, 2, -97, 997]))
    assert trained.iloc[0] == 1.0 and trained.iloc[1] == 0.0
    assert trained.iloc[2:].isna().all()


def test_categorical_valid_keeps_only_allowed_codes():
    out = categorical_valid(pd.Series([1, 2, 5, 99]), {1, 2, 3, 4})
    assert out.iloc[:2].tolist() == [1.0, 2.0]
    assert out.iloc[2:].isna().all()


def test_harmful_incident_requires_identified_incident_and_substantive_outcome():
    df = _frame([
        {"type_comb1": 0},
        {"type_comb1": 1, "outcome3": 1},
        {"type_comb1": 1, "outcome10": 1},
        {"type_comb1": 1, "outcome9": 1},
        {"type_comb1": 9},
    ])
    out = derive_harmful_incident(df)
    assert out.iloc[:4].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert pd.isna(out.iloc[4])


def test_harmful_incident_never_exceeds_identified_incident():
    df = _frame([{"type_comb1": 0, "outcome1": 1}, {"type_comb1": 1, "outcome1": 1}])
    harmful = derive_harmful_incident(df)
    identified = binary_01(df["type_comb1"])
    assert (harmful <= identified).all()


def test_impactful_incident_handles_none_and_dont_know():
    df = _frame([
        {"type_comb1": 0},
        {"type_comb1": 1, "impact2": 1},
        {"type_comb1": 1, "impact11": 1},
        {"type_comb1": 1, "impact12": 1},
        {"type_comb1": 1},
    ])
    out = derive_impactful_incident(df)
    assert out.iloc[:3].tolist() == [0.0, 1.0, 0.0]
    assert pd.isna(out.iloc[3])
    assert out.iloc[4] == 0.0
