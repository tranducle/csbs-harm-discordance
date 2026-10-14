"""Derivation of the business-level indicators from CSBS survey items."""

from __future__ import annotations

from typing import List, Set

import pandas as pd

# outcome9 ("Don't know") and outcome10 ("None of these") are not consequences.
OUTCOME_SUBSTANTIVE: List[str] = [f"outcome{i}" for i in list(range(1, 9)) + [11, 12, 13]]
IMPACT_SUBSTANTIVE: List[str] = [f"impact{i}" for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]]


def binary_01(series: pd.Series) -> pd.Series:
    """Map codes 0 and 1 to 0.0 and 1.0 and every other code to missing."""
    out = pd.Series(pd.NA, index=series.index, dtype="Float64")
    out.loc[series.eq(0)] = 0.0
    out.loc[series.eq(1)] = 1.0
    return out


def recode_trained(series: pd.Series) -> pd.Series:
    """Recode the training item: 1 (yes) to 1.0, 2 (no) to 0.0, anything else missing."""
    out = pd.Series(pd.NA, index=series.index, dtype="Float64")
    out.loc[series.eq(1)] = 1.0
    out.loc[series.eq(2)] = 0.0
    return out


def categorical_valid(series: pd.Series, allowed: Set[int]) -> pd.Series:
    """Keep codes in ``allowed`` as floats and set every other code to missing."""
    out = pd.Series(pd.NA, index=series.index, dtype="Float64")
    mask = series.isin(sorted(allowed))
    out.loc[mask] = series.loc[mask].astype(float)
    return out


def derive_harmful_incident(df: pd.DataFrame) -> pd.Series:
    """Business reports at least one identified incident with at least one negative outcome.

    The value is 0 when no incident was identified, and for businesses with an identified
    incident it is 1 if any substantive outcome item equals 1, else 0.
    """
    identified = binary_01(df["type_comb1"])
    out = pd.Series(pd.NA, index=df.index, dtype="Float64")
    out.loc[identified.eq(0)] = 0.0
    incident = identified.eq(1)
    out.loc[incident] = df.loc[incident, OUTCOME_SUBSTANTIVE].eq(1).any(axis=1).astype(float)
    return out


def derive_impactful_incident(df: pd.DataFrame) -> pd.Series:
    """Business reports at least one identified incident with a substantive impact item.

    The value is 0 when no incident was identified and 0 for an explicit "none of these".
    It is missing when the only impact answer is "don't know".
    """
    identified = binary_01(df["type_comb1"])
    out = pd.Series(pd.NA, index=df.index, dtype="Float64")
    out.loc[identified.eq(0)] = 0.0

    incident_index = df.index[identified.eq(1)]
    if len(incident_index) == 0:
        return out

    substantive = df.loc[incident_index, IMPACT_SUBSTANTIVE]
    any_substantive = substantive.eq(1).any(axis=1)
    none_selected = df.loc[incident_index, "impact11"].eq(1)
    dont_know_only = df.loc[incident_index, "impact12"].eq(1) & ~any_substantive
    all_zero = substantive.eq(0).all(axis=1)

    out.loc[incident_index[any_substantive.to_numpy()]] = 1.0
    out.loc[incident_index[((~any_substantive) & none_selected).to_numpy()]] = 0.0
    out.loc[incident_index[((~any_substantive) & (~dont_know_only) & all_zero).to_numpy()]] = 0.0
    return out
