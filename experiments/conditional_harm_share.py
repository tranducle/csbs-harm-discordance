"""Secondary analysis: harm share among businesses with an identified incident.

Usage:
    python experiments/conditional_harm_share.py

This analysis was run after the primary and robustness results were known and is exploratory.
Conditioning on identified incidents is selection-sensitive: a practice that changes which
incidents are noticed also changes who enters the conditional sample. The estimates describe
associations in that sample and are not evidence for or against the primary discordance result.

Specification:
- sample: businesses with an identified incident;
- outcome: the harmful-incident indicator (the conditional harm share);
- main model: weighted linear probability model with the primary adjustment set (both practices,
  survey year, size, sector), official weights, business-clustered standard errors;
- estimator check: weighted logistic model with average marginal effects standardised over the
  conditional sample, and a business-level bootstrap within survey year;
- multiplicity: Benjamini-Hochberg across the two practice p-values of the main model;
- descriptive: weighted conditional harm share by practice status.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csbs_harm.common import CONFIG_DIR, REPO_ROOT, RESULTS_DIR, read_json, verify_sha256, write_json  # noqa: E402

logger = logging.getLogger("conditional_harm_share")

OUT_DIR = RESULTS_DIR / "conditional_harm_share"


def weighted_share(y: pd.Series, w: pd.Series) -> float:
    """Weighted mean of a binary indicator."""
    return float(np.average(y.astype(float), weights=w.astype(float)))


def fit_lpm(data: pd.DataFrame, formula: str):
    """Weighted linear probability model with business-clustered standard errors."""
    model = smf.wls(formula, data=data, weights=data["survey_weight"])
    return model.fit(cov_type="cluster", cov_kwds={"groups": data["record_id"], "use_correction": True})


def logistic_ames(data: pd.DataFrame, formula: str, design_info, practices) -> Dict[str, float]:
    """Weighted average marginal effect of each practice from one weighted logistic fit."""
    y, _ = patsy.dmatrices(formula, data, return_type="dataframe")
    X = patsy.build_design_matrices([design_info], data, return_type="dataframe")[0]
    fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=data["survey_weight"]).fit()
    out: Dict[str, float] = {}
    for practice in practices:
        treated, control = data.copy(), data.copy()
        treated[practice] = 1
        control[practice] = 0
        X1 = patsy.build_design_matrices([design_info], treated, return_type="dataframe")[0]
        X0 = patsy.build_design_matrices([design_info], control, return_type="dataframe")[0]
        diff = np.asarray(fit.predict(X1)) - np.asarray(fit.predict(X0))
        out[practice] = float(np.average(diff, weights=data["survey_weight"]))
    return out


def bootstrap_ames(data: pd.DataFrame, formula: str, design_info, practices, seed: int, replicates: int):
    """Business-level bootstrap within survey year for the logistic marginal effects."""
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {p: [] for p in practices}
    failures = 0
    groups = [g for _, g in data.groupby("survey_year", sort=True)]
    for _ in range(replicates):
        sample = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups], ignore_index=True)
        try:
            ames = logistic_ames(sample, formula, design_info, practices)
        except (np.linalg.LinAlgError, ValueError, PerfectSeparationError) as exc:
            failures += 1
            logger.warning("bootstrap replicate failed: %s", exc)
            continue
        for practice in practices:
            draws[practice].append(ames[practice])
    out: Dict[str, Dict[str, Any]] = {}
    for practice, values in draws.items():
        arr = np.asarray(values, dtype=float)
        out[practice] = {
            "bootstrap_ci_low": float(np.quantile(arr, 0.025)),
            "bootstrap_ci_high": float(np.quantile(arr, 0.975)),
            "bootstrap_se": float(arr.std(ddof=1)),
            "replicates_success": int(arr.size),
        }
    out["_meta"] = {"replicates_requested": replicates, "replicates_failed": failures}
    return out


def main() -> None:
    """Run the conditional harm-share analysis and write its result files."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = read_json(CONFIG_DIR / "secondary_analyses.json")
    spec = config["conditional_harm_share"]
    practices = tuple(config["practices"])
    formula = spec["formula"]

    input_path = REPO_ROOT / config["input_csv"]
    input_sha = verify_sha256(input_path, config["input_sha256_prefix"], "primary business table")
    df = pd.read_csv(input_path)
    if len(df) != config["expected_business_n"]:
        raise AssertionError(f"Expected {config['expected_business_n']} businesses, got {len(df)}")

    cond = df.loc[df["y1_identified_incident"] == 1].copy().reset_index(drop=True)
    if (cond["y2_identified_harmful_incident"] > cond["y1_identified_incident"]).any():
        raise AssertionError("Nesting violated: harmful incident without an identified incident")
    logger.info("conditional sample n=%d, harmful=%d", len(cond), int(cond["y2_identified_harmful_incident"].sum()))

    w = cond["survey_weight"]
    y = cond["y2_identified_harmful_incident"]
    descriptive: Dict[str, Any] = {
        "n_conditional": int(len(cond)),
        "n_harmful": int(y.sum()),
        "n_by_year": {str(int(k)): int(v) for k, v in cond["survey_year"].value_counts().sort_index().items()},
        "weighted_harm_share_all": weighted_share(y, w),
    }
    for practice in practices:
        for level in (0, 1):
            mask = cond[practice] == level
            descriptive[f"weighted_harm_share_{practice}_{level}"] = weighted_share(y[mask], w[mask])
            descriptive[f"n_{practice}_{level}"] = int(mask.sum())
            descriptive[f"n_harmful_{practice}_{level}"] = int(y[mask].sum())

    lpm = fit_lpm(cond, formula)
    pred = np.asarray(lpm.predict(cond), dtype=float)
    ci = lpm.conf_int()
    lpm_out: Dict[str, Dict[str, Any]] = {
        p: {
            "estimate": float(lpm.params[p]),
            "se": float(lpm.bse[p]),
            "ci_low": float(ci.loc[p, 0]),
            "ci_high": float(ci.loc[p, 1]),
            "p_value": float(lpm.pvalues[p]),
        }
        for p in practices
    }
    bh = multipletests([lpm_out[p]["p_value"] for p in practices], alpha=spec["alpha"], method="fdr_bh")
    for i, practice in enumerate(practices):
        lpm_out[practice]["bh_q"] = float(bh[1][i])
        lpm_out[practice]["bh_reject"] = bool(bh[0][i])

    _, X = patsy.dmatrices(formula, cond, return_type="dataframe")
    design_info = X.design_info
    point = logistic_ames(cond, formula, design_info, practices)
    boot = bootstrap_ames(cond, formula, design_info, practices, spec["bootstrap_seed"], spec["bootstrap_replicates"])
    logit_out: Dict[str, Any] = {p: {"point_ame": point[p], **boot[p]} for p in practices}
    logit_out["_meta"] = {**boot["_meta"], "seed": spec["bootstrap_seed"]}

    results = {
        "status_label": "secondary exploratory analysis, run after the primary and robustness results",
        "boundary": (
            "Conditioning on identified incidents is selection-sensitive; not causal; "
            "not supporting evidence for the primary discordance result."
        ),
        "input_csv": config["input_csv"],
        "input_sha256": input_sha,
        "formula": formula,
        "descriptive": descriptive,
        "lpm": lpm_out,
        "lpm_diagnostics": {
            "n": int(lpm.nobs),
            "n_clusters": int(cond["record_id"].nunique()),
            "prediction_min": float(pred.min()),
            "prediction_max": float(pred.max()),
            "prediction_out_of_range_rate": float(((pred < 0) | (pred > 1)).mean()),
            "condition_number": float(np.linalg.cond(np.asarray(lpm.model.exog))),
        },
        "logistic_ame": logit_out,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "patsy": patsy.__version__,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "conditional_harm_share_results.json", results)
    pd.DataFrame({
        "term": lpm.params.index,
        "estimate": lpm.params.values,
        "se_cluster": lpm.bse.values,
        "p_value": lpm.pvalues.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
    }).to_csv(OUT_DIR / "conditional_harm_share_lpm_coefficients.csv", index=False)
    for practice in practices:
        logger.info("%s: LPM %.4f (p=%.4g); logistic AME %.4f [%.4f, %.4f]",
                    practice, lpm_out[practice]["estimate"], lpm_out[practice]["p_value"],
                    point[practice], boot[practice]["bootstrap_ci_low"], boot[practice]["bootstrap_ci_high"])


if __name__ == "__main__":
    main()
