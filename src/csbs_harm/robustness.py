"""Model-fitting helpers for the robustness analyses."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from csbs_harm.stats import linear_contrast, stack_outcomes, weighted_mean

logger = logging.getLogger(__name__)

X1 = "x1_staff_awareness_test"
X2 = "x2_security_monitoring"
HARM = "y2_identified_harmful_incident"
IDENTIFIED = "y1_identified_incident"

ROBUSTNESS_STACK_COLUMNS: Tuple[str, ...] = (
    "record_id", "survey_year", "survey_weight", X1, X2, "size_cat", "size_ms_ml", "sector_cat",
)
_BOOTSTRAP_FAILURES = (np.linalg.LinAlgError, ValueError, FloatingPointError, PerfectSeparationError)


def find_interaction(index, variable: str) -> str:
    """Return the ``metric`` by ``variable`` interaction term in either patsy ordering."""
    for candidate in (f"metric:{variable}", f"{variable}:metric"):
        if candidate in index:
            return candidate
    raise KeyError(f"Missing metric interaction for {variable}")


def find_threeway(index, variable: str, factor_token: str) -> str:
    """Return the single three-way term involving ``metric``, ``variable`` and ``factor_token``."""
    hits = [
        name for name in index
        if "metric" in name and variable in name and factor_token in name and name.count(":") >= 2
    ]
    if len(hits) != 1:
        raise KeyError(f"Expected one three-way term for {variable} and {factor_token}, got {hits}")
    return hits[0]


def _fit_stacked(stacked: pd.DataFrame, formula: str):
    """Weighted least squares on the stacked frame with business-clustered standard errors."""
    model = smf.wls(formula, data=stacked, weights=stacked["_fit_weight"])
    result = model.fit(cov_type="cluster", cov_kwds={"groups": stacked["record_id"], "use_correction": True})
    return model, result


def fit_lpm(
    df: pd.DataFrame,
    first: Sequence[float],
    second: Sequence[float],
    formula: str,
    weights: Sequence[float],
    run_id: str,
    family: str,
    variant: str,
    changed_factor: str,
) -> Dict[str, Any]:
    """Fit the stacked linear probability model and return one registry row.

    Args:
        df: Business-level frame.
        first: Outcome stacked as ``metric == 0``.
        second: Outcome stacked as ``metric == 1``.
        formula: Model formula using ``outcome`` and ``metric``.
        weights: One weight per business, repeated for both stacked copies.
        run_id: Identifier of the run in the registry.
        family: Robustness family.
        variant: Variant within the family.
        changed_factor: What differs from the primary specification.

    Returns:
        Registry row with design diagnostics and, for each practice in the formula,
        the discordance estimate, standard error, interval and p-value.
    """
    stacked = stack_outcomes(df, first, second, ROBUSTNESS_STACK_COLUMNS)
    stacked["_fit_weight"] = np.tile(np.asarray(weights, dtype=float), 2)
    model, result = _fit_stacked(stacked, formula)
    pred = np.asarray(result.predict(stacked), dtype=float)
    scaled = np.asarray(model.exog) * np.sqrt(np.asarray(stacked["_fit_weight"]))[:, None]

    row: Dict[str, Any] = {
        "run_id": run_id,
        "family": family,
        "variant": variant,
        "changed_factor": changed_factor,
        "n_business": int(len(df)),
        "n_stacked": int(len(stacked)),
        "n_clusters": int(stacked["record_id"].nunique()),
        "rank": int(np.linalg.matrix_rank(model.exog)),
        "p_columns": int(model.exog.shape[1]),
        "condition_number": float(np.linalg.cond(scaled)),
        "prediction_out_of_range_rate": float(((pred < 0) | (pred > 1)).mean()),
    }
    for label, variable in (("x1", X1), ("x2", X2)):
        values = {"estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
        if variable in formula:
            values = linear_contrast(result, {find_interaction(result.params.index, variable): 1.0})
        row.update({
            f"{label}_estimate": values["estimate"],
            f"{label}_se": values["se"],
            f"{label}_ci_low": values["ci_low"],
            f"{label}_ci_high": values["ci_high"],
            f"{label}_p": values["p_value"],
        })
    return row


def fit_heterogeneity(
    df: pd.DataFrame, factor: str, factor_token: str, formula: str, run_id: str, family: str, variant: str
) -> Dict[str, Any]:
    """Formal test of whether the discordance differs across levels of ``factor``."""
    stacked = stack_outcomes(df, df[IDENTIFIED].to_numpy(), df[HARM].to_numpy(), ROBUSTNESS_STACK_COLUMNS)
    stacked["_fit_weight"] = np.tile(df["survey_weight"].to_numpy(dtype=float), 2)
    _, result = _fit_stacked(stacked, formula)
    term1 = find_threeway(result.params.index, X1, factor_token)
    term2 = find_threeway(result.params.index, X2, factor_token)
    c1 = linear_contrast(result, {term1: 1.0})
    c2 = linear_contrast(result, {term2: 1.0})
    reject, p_adj, _, _ = multipletests([c1["p_value"], c2["p_value"]], alpha=0.05, method="holm")
    row: Dict[str, Any] = {
        "run_id": run_id, "family": family, "variant": variant, "factor": factor,
        "n_business": int(len(df)),
    }
    for i, (label, term, contrast) in enumerate((("x1", term1, c1), ("x2", term2, c2))):
        row.update({
            f"{label}_term": term,
            f"{label}_estimate": contrast["estimate"],
            f"{label}_se": contrast["se"],
            f"{label}_ci_low": contrast["ci_low"],
            f"{label}_ci_high": contrast["ci_high"],
            f"{label}_p": contrast["p_value"],
            f"{label}_holm_p": float(p_adj[i]),
            f"{label}_holm_reject": bool(reject[i]),
        })
    return row


def derive_harm_leave_one_out(df: pd.DataFrame, items: Sequence[str], omitted: str) -> np.ndarray:
    """Harmful-incident indicator rebuilt without one outcome item."""
    kept = [item for item in items if item != omitted]
    identified = df[IDENTIFIED].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    incident = identified == 1.0
    any_positive = df.loc[incident, kept].eq(1).any(axis=1).to_numpy(dtype=bool)
    out[incident] = any_positive.astype(float)
    return out


def fit_logit_model(df: pd.DataFrame, outcome: str):
    """Weighted logistic model with the primary adjustment set."""
    formula = f"{outcome} ~ {X1} + {X2} + C(survey_year) + C(size_cat) + C(sector_cat)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.glm(formula, data=df, family=sm.families.Binomial(), freq_weights=df["survey_weight"])
        return model.fit(maxiter=100, disp=0)


def standardized_discrete_change(model, df: pd.DataFrame, variable: str) -> float:
    """Weighted average change in predicted probability when ``variable`` moves from 0 to 1."""
    treated, control = df.copy(), df.copy()
    treated[variable] = 1.0
    control[variable] = 0.0
    p1 = np.asarray(model.predict(treated), dtype=float)
    p0 = np.asarray(model.predict(control), dtype=float)
    return weighted_mean(p1 - p0, df["survey_weight"])


def logit_point_estimates(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Marginal effects on each indicator and their difference, for both practices."""
    identified_fit = fit_logit_model(df, IDENTIFIED)
    harm_fit = fit_logit_model(df, HARM)
    out: Dict[str, Dict[str, float]] = {}
    for variable, label in ((X1, "x1"), (X2, "x2")):
        y1_ame = standardized_discrete_change(identified_fit, df, variable)
        y2_ame = standardized_discrete_change(harm_fit, df, variable)
        out[label] = {"y1_ame": y1_ame, "y2_ame": y2_ame, "discordance": y2_ame - y1_ame}
    return out


def bootstrap_logit(df: pd.DataFrame, replicates: int, seed: int, progress_every: int = 200):
    """Business-level bootstrap within survey year of the logistic marginal effects.

    Returns:
        A frame with one row per successful replicate and the number of failed replicates.
    """
    rng = np.random.default_rng(seed)
    year_indices = {
        int(year): np.asarray(idx, dtype=int) for year, idx in df.groupby("survey_year", sort=True).groups.items()
    }
    rows, failures = [], 0
    for b in range(replicates):
        sampled = []
        for year in sorted(year_indices):
            idx = year_indices[year]
            sampled.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
        resampled = df.loc[sampled].reset_index(drop=True)
        try:
            point = logit_point_estimates(resampled)
            rows.append({
                "replicate": b + 1,
                "x1_y1_ame": point["x1"]["y1_ame"],
                "x1_y2_ame": point["x1"]["y2_ame"],
                "x1_discordance": point["x1"]["discordance"],
                "x2_y1_ame": point["x2"]["y1_ame"],
                "x2_y2_ame": point["x2"]["y2_ame"],
                "x2_discordance": point["x2"]["discordance"],
            })
        except _BOOTSTRAP_FAILURES as exc:
            failures += 1
            logger.debug("bootstrap replicate %d failed: %s", b + 1, exc)
        if progress_every and (b + 1) % progress_every == 0:
            logger.info("bootstrap %d/%d success=%d failures=%d", b + 1, replicates, len(rows), failures)
    return pd.DataFrame(rows), failures


def bootstrap_summary(point: Dict[str, Dict[str, float]], boot: pd.DataFrame) -> Dict[str, Any]:
    """Percentile interval, standard error and sign agreement of the bootstrap discordance."""
    out: Dict[str, Any] = {"replicates_success": int(len(boot))}
    for label in ("x1", "x2"):
        values = boot[f"{label}_discordance"].dropna().to_numpy(dtype=float)
        out[label] = {
            "point_y1_ame": point[label]["y1_ame"],
            "point_y2_ame": point[label]["y2_ame"],
            "point_discordance": point[label]["discordance"],
            "bootstrap_ci_low": float(np.quantile(values, 0.025)),
            "bootstrap_ci_high": float(np.quantile(values, 0.975)),
            "bootstrap_se": float(np.std(values, ddof=1)),
            "bootstrap_same_sign_share": float(np.mean(np.sign(values) == np.sign(point[label]["discordance"]))),
        }
    return out
