"""Exact unadjusted decomposition of the discordance between two nested indicators."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from csbs_harm.stats import weighted_mean


def decompose(df: pd.DataFrame, practice: str, tolerance: float) -> Dict[str, float]:
    """Decompose the unadjusted discordance for one practice.

    Every harmful incident is also an identified incident, so for a practice status ``X``
    ``P(harm | X) = p(X) * s(X)`` with ``p(X) = P(identified | X)`` and
    ``s(X) = P(harm | identified, X)``. With ``p0, p1, s0, s1`` the values at ``X = 0`` and
    ``X = 1``, the difference of risk differences satisfies exactly

        RD(harm) - RD(identified) = (s1 - 1)(p1 - p0) + p0 (s1 - s0).

    The first term is the nesting component and the second the conditional-severity component.

    Args:
        df: Business-level frame with the weighted indicators.
        practice: Name of the binary practice column.
        tolerance: Largest accepted absolute error of the identity.

    Returns:
        The weighted proportions, both risk differences, and the two components.

    Raises:
        AssertionError: If the identity fails beyond ``tolerance``.
    """
    parts: Dict[str, float] = {}
    for level in (0, 1):
        group = df[df[practice] == level]
        p = weighted_mean(group["y1_identified_incident"], group["survey_weight"])
        q = weighted_mean(group["y2_identified_harmful_incident"], group["survey_weight"])
        incident = group[group["y1_identified_incident"] == 1]
        s = weighted_mean(incident["y2_identified_harmful_incident"], incident["survey_weight"])
        parts.update({f"p{level}": p, f"q{level}": q, f"s{level}": s})

    rd_identified = parts["p1"] - parts["p0"]
    rd_harmful = parts["q1"] - parts["q0"]
    nesting = (parts["s1"] - 1.0) * rd_identified
    severity = parts["p0"] * (parts["s1"] - parts["s0"])
    delta = rd_harmful - rd_identified
    error = abs(delta - (nesting + severity))
    if error > tolerance:
        raise AssertionError(f"Identity failed for {practice}: {delta} vs {nesting + severity}")
    return {
        **parts,
        "rd_y1_unadjusted": rd_identified,
        "rd_y2_unadjusted": rd_harmful,
        "delta_unadjusted": delta,
        "nesting_component": nesting,
        "severity_component": severity,
        "identity_abs_error": error,
    }
