"""Weighted statistics and regression helpers shared by the analyses."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

STACK_COLUMNS = (
    "record_id",
    "survey_year",
    "survey_weight",
    "x1_staff_awareness_test",
    "x2_security_monitoring",
    "size_cat",
    "sector_cat",
)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    """Weighted mean that skips pairs with a missing value or weight."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(w))
    return float(np.average(x[keep], weights=w[keep]))


def kish_effective_sample_size(weights: Iterable[float]) -> float:
    """Kish effective sample size, (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    return float((w.sum() ** 2) / np.square(w).sum())


def weighted_correlation(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    """Weighted Pearson correlation over rows where all three inputs are present."""
    keep = x.notna() & y.notna() & weights.notna()
    xv = x.loc[keep].astype(float).to_numpy()
    yv = y.loc[keep].astype(float).to_numpy()
    wt = weights.loc[keep].astype(float).to_numpy()
    mx = np.average(xv, weights=wt)
    my = np.average(yv, weights=wt)
    cov = np.average((xv - mx) * (yv - my), weights=wt)
    vx = np.average((xv - mx) ** 2, weights=wt)
    vy = np.average((yv - my) ** 2, weights=wt)
    return float(cov / math.sqrt(vx * vy))


def stack_outcomes(
    df: pd.DataFrame,
    first: Sequence[float],
    second: Sequence[float],
    columns: Sequence[str] = STACK_COLUMNS,
) -> pd.DataFrame:
    """Stack two outcomes for the same businesses into one long frame.

    Rows with ``metric == 0`` carry ``first`` and rows with ``metric == 1`` carry ``second``.
    The business identifier ``record_id`` is kept so that standard errors can be clustered.
    ``columns`` lists the covariates carried into both copies.
    """
    left = df[list(columns)].copy()
    left["metric"] = 0
    left["outcome"] = np.asarray(first, dtype=float)
    right = df[list(columns)].copy()
    right["metric"] = 1
    right["outcome"] = np.asarray(second, dtype=float)
    stacked = pd.concat([left, right], ignore_index=True)
    stacked["metric"] = stacked["metric"].astype(int)
    return stacked


def find_term(params_index: Iterable[str], first: str, second: str) -> str:
    """Return the interaction term ``first:second`` in whichever order patsy wrote it."""
    names = list(params_index)
    for candidate in (f"{first}:{second}", f"{second}:{first}"):
        if candidate in names:
            return candidate
    raise KeyError(f"Interaction term not found for {first}, {second}")


def linear_contrast(result, terms: Dict[str, float]) -> Dict[str, float]:
    """Estimate, standard error, 95% interval and p-value of a linear combination of coefficients."""
    names = list(result.params.index)
    vector = np.zeros(len(names), dtype=float)
    for term, value in terms.items():
        vector[names.index(term)] = value
    test = result.t_test(vector)
    interval = np.asarray(test.conf_int(alpha=0.05)).reshape(-1, 2)[0]
    return {
        "estimate": float(np.asarray(test.effect).reshape(-1)[0]),
        "se": float(np.asarray(test.sd).reshape(-1)[0]),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
    }
