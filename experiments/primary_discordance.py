"""Primary analysis: do two security practices differ in their association with identified versus harmful incidents?

Usage:
    python experiments/primary_discordance.py

Fits one weighted linear probability model on the stacked outcomes (identified incident and
harmful incident for the same businesses), with standard errors clustered by business. The
metric-by-practice interaction is the discordance. Holm adjustment is applied across the two
practices. Writes four files to results/primary/.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csbs_harm.common import (  # noqa: E402
    CONFIG_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    read_json,
    sha256_file,
    verify_sha256,
    write_json,
)
from csbs_harm.stats import (  # noqa: E402
    find_term,
    kish_effective_sample_size,
    linear_contrast,
    stack_outcomes,
    weighted_correlation,
    weighted_mean,
)

logger = logging.getLogger("primary_discordance")

OUT_DIR = RESULTS_DIR / "primary"
PRIMARY_REQUIRED = [
    "survey_weight", "survey_year", "size_cat", "sector_cat",
    "x1_staff_awareness_test", "x2_security_monitoring",
    "y1_identified_incident", "y2_identified_harmful_incident",
]


def design_summary(df: pd.DataFrame, input_sha256: str) -> Dict[str, Any]:
    """Weighted prevalences, effective sample sizes and the practice correlation."""
    weight = df["survey_weight"]
    summary: Dict[str, Any] = {
        "input_sha256": input_sha256,
        "business_n": int(len(df)),
        "year": {},
        "pooled": {
            "x1_prevalence_weighted": weighted_mean(df["x1_staff_awareness_test"], weight),
            "x2_prevalence_weighted": weighted_mean(df["x2_security_monitoring"], weight),
            "y1_prevalence_weighted": weighted_mean(df["y1_identified_incident"], weight),
            "y2_prevalence_weighted": weighted_mean(df["y2_identified_harmful_incident"], weight),
            "x1_x2_weighted_corr": weighted_correlation(
                df["x1_staff_awareness_test"], df["x2_security_monitoring"], weight
            ),
        },
    }
    for year, group in df.groupby("survey_year", sort=True):
        w = group["survey_weight"]
        summary["year"][str(int(year))] = {
            "n": int(len(group)),
            "weight_sum": float(w.sum()),
            "weight_mean": float(w.mean()),
            "kish_ess": kish_effective_sample_size(w),
            "x1_prevalence_weighted": weighted_mean(group["x1_staff_awareness_test"], w),
            "x2_prevalence_weighted": weighted_mean(group["x2_security_monitoring"], w),
            "y1_prevalence_weighted": weighted_mean(group["y1_identified_incident"], w),
            "y2_prevalence_weighted": weighted_mean(group["y2_identified_harmful_incident"], w),
        }
    return summary


def main() -> None:
    """Run the primary analysis and write its result files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = read_json(CONFIG_DIR / "primary_analysis.json")
    input_path = REPO_ROOT / config["input_csv"]
    input_sha256 = verify_sha256(input_path, config["input_sha256_prefix"], "primary business table")
    df = pd.read_csv(input_path)

    if len(df) != config["expected_business_n"]:
        raise AssertionError(f"Expected {config['expected_business_n']} businesses, got {len(df)}")
    if df["record_id"].duplicated().any():
        raise AssertionError("Duplicate record_id in primary data")
    if df[PRIMARY_REQUIRED].isna().any().any():
        raise AssertionError("Missing primary analysis field")

    design = design_summary(df, input_sha256)

    stacked = stack_outcomes(df, df["y1_identified_incident"], df["y2_identified_harmful_incident"])
    if len(stacked) != config["expected_stacked_n"]:
        raise AssertionError(f"Expected {config['expected_stacked_n']} stacked rows, got {len(stacked)}")
    if stacked["record_id"].nunique() != config["expected_business_n"]:
        raise AssertionError("Unexpected cluster count")

    formula = config["formula"]
    model = smf.wls(formula, data=stacked, weights=stacked["survey_weight"])
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": stacked["record_id"], "use_correction": True},
    )

    # Independent least-squares solution as a cross-check of the fitted coefficients.
    X = np.asarray(model.exog, dtype=float)
    y = np.asarray(model.endog, dtype=float)
    sqrt_w = np.sqrt(np.asarray(stacked["survey_weight"], dtype=float))
    beta_numpy, _, _, _ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
    max_beta_diff = float(np.max(np.abs(beta_numpy - np.asarray(result.params, dtype=float))))
    rank = int(np.linalg.matrix_rank(X))
    condition_number = float(np.linalg.cond(X * sqrt_w[:, None]))

    pred = np.asarray(result.predict(stacked), dtype=float)
    out_of_range = (pred < 0.0) | (pred > 1.0)

    x1_term, x2_term = "x1_staff_awareness_test", "x2_security_monitoring"
    x1_int = find_term(result.params.index, "metric", x1_term)
    x2_int = find_term(result.params.index, "metric", x2_term)

    x1_diff = linear_contrast(result, {x1_int: 1.0})
    x2_diff = linear_contrast(result, {x2_int: 1.0})
    holm = multipletests([x1_diff["p_value"], x2_diff["p_value"]], alpha=config["alpha"], method="holm")
    x1_diff["holm_p"], x1_diff["holm_reject"] = float(holm[1][0]), bool(holm[0][0])
    x2_diff["holm_p"], x2_diff["holm_reject"] = float(holm[1][1]), bool(holm[0][1])

    results = {
        "analysis": config["analysis"],
        "input_csv": config["input_csv"],
        "input_sha256": input_sha256,
        "formula": formula,
        "n_business": int(len(df)),
        "n_stacked": int(len(stacked)),
        "n_clusters": int(stacked["record_id"].nunique()),
        "design_matrix_columns": list(model.exog_names),
        "design_matrix_rank": rank,
        "design_matrix_p": int(X.shape[1]),
        "condition_number": condition_number,
        "max_abs_numpy_statsmodels_beta_diff": max_beta_diff,
        "prediction_min": float(pred.min()),
        "prediction_max": float(pred.max()),
        "prediction_out_of_range_n": int(out_of_range.sum()),
        "prediction_out_of_range_rate": float(out_of_range.mean()),
        "primary_terms": {"x1_interaction_term": x1_int, "x2_interaction_term": x2_int},
        "effects": {
            "x1_y1_rd": linear_contrast(result, {x1_term: 1.0}),
            "x1_y2_rd": linear_contrast(result, {x1_term: 1.0, x1_int: 1.0}),
            "x1_discordance": x1_diff,
            "x2_y1_rd": linear_contrast(result, {x2_term: 1.0}),
            "x2_y2_rd": linear_contrast(result, {x2_term: 1.0, x2_int: 1.0}),
            "x2_discordance": x2_diff,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "patsy": patsy.__version__,
        },
    }

    coefficients = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "se_cluster": result.bse.values,
        "p_value": result.pvalues.values,
        "ci_low": result.conf_int()[0].values,
        "ci_high": result.conf_int()[1].values,
    })
    stacked_summary = {
        "rows": int(len(stacked)),
        "clusters": int(stacked["record_id"].nunique()),
        "metric_counts": {str(k): int(v) for k, v in stacked["metric"].value_counts().sort_index().items()},
        "outcome_by_metric": {
            str(int(k)): {
                "n": int(len(g)),
                "positive": int(g["outcome"].sum()),
                "weighted_mean": weighted_mean(g["outcome"], g["survey_weight"]),
            }
            for k, g in stacked.groupby("metric")
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "design_summary.json", design)
    write_json(OUT_DIR / "primary_results.json", results)
    write_json(OUT_DIR / "stacked_summary.json", stacked_summary)
    coefficients.to_csv(OUT_DIR / "primary_coefficients.csv", index=False)

    logger.info("Discordance, practice 1 (staff awareness test): %.4f (p=%.4g, Holm p=%.4g)",
                x1_diff["estimate"], x1_diff["p_value"], x1_diff["holm_p"])
    logger.info("Discordance, practice 2 (security monitoring): %.4f (p=%.4g, Holm p=%.4g)",
                x2_diff["estimate"], x2_diff["p_value"], x2_diff["holm_p"])
    logger.info("rank %d of %d; numpy cross-check max |diff| %.2e; out-of-range predictions %.4f",
                rank, X.shape[1], max_beta_diff, float(out_of_range.mean()))


if __name__ == "__main__":
    main()
