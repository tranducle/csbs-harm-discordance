"""Robustness analyses for the primary discordance result.

Usage:
    python experiments/robustness_checks.py            # full run, 2,000 bootstrap replicates
    python experiments/robustness_checks.py smoke      # 20 bootstrap replicates, for a quick check

Run experiments/primary_discordance.py first: the sign-stability table compares every variant
with the primary estimates in results/primary/primary_results.json.

Families: logistic marginal standardisation with a bootstrap, year-specific and formal year
heterogeneity, weight sensitivity, mutual-adjustment and adjustment-set ablations,
leave-one-outcome-item-out, an impact-based consequence definition, a phishing-specificity
check, size subsets and heterogeneity, and leave-one-sector-out.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy
import statsmodels

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csbs_harm.common import CONFIG_DIR, REPO_ROOT, RESULTS_DIR, read_json, sha256_file, verify_sha256, write_json  # noqa: E402
from csbs_harm.robustness import (  # noqa: E402
    HARM,
    IDENTIFIED,
    X1,
    X2,
    bootstrap_logit,
    bootstrap_summary,
    derive_harm_leave_one_out,
    fit_heterogeneity,
    fit_lpm,
    logit_point_estimates,
)

logger = logging.getLogger("robustness_checks")

SMOKE_REPLICATES = 20


def _fit(d: pd.DataFrame, formula: str, run_id: str, family: str, variant: str, factor: str,
         second=None, weights=None) -> Dict[str, Any]:
    """Fit one variant with the identified incident as the first outcome."""
    return fit_lpm(
        d,
        d[IDENTIFIED].to_numpy(),
        d[HARM].to_numpy() if second is None else second,
        formula,
        d["survey_weight"].to_numpy() if weights is None else weights,
        run_id, family, variant, factor,
    )


def run_lpm_variants(df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fit every stacked linear probability variant, in registry order."""
    primary = config["primary_formula"]
    controls = "C(survey_year) + C(size_cat) + C(sector_cat)"
    rows = [_fit(df, primary, "R00_PRIMARY_REPRO", "primary_reproduction", "primary", "none")]

    for year in (2024, 2025):
        d = df.loc[df["survey_year"].eq(year)].copy()
        formula = f"outcome ~ metric * ({X1} + {X2} + C(size_cat) + C(sector_cat))"
        rows.append(_fit(d, formula, f"R10_YEAR_{year}", "temporal_stability", str(year), "survey_year_subset"))

    rows.append(_fit(df, primary, "R20_UNWEIGHTED", "weight_sensitivity", "unweighted", "weight",
                     weights=np.ones(len(df), dtype=float)))
    normalized = df["survey_weight"] / df.groupby("survey_year")["survey_weight"].transform("mean")
    rows.append(_fit(df, primary, "R21_WEIGHT_MEAN1_YEAR", "weight_sensitivity", "within_year_mean_one",
                     "weight_scaling", weights=normalized.to_numpy()))

    ablations = [
        (f"outcome ~ metric * ({X1} + {controls})", "A10_X1_ONLY", "mutual_adjustment_ablation", "x1_only", "remove_x2"),
        (f"outcome ~ metric * ({X2} + {controls})", "A11_X2_ONLY", "mutual_adjustment_ablation", "x2_only", "remove_x1"),
        (f"outcome ~ metric * ({X1} + {X2} + C(survey_year))", "A20_YEAR_ONLY", "adjustment_set_ablation",
         "year_only", "remove_size_and_sector"),
        (f"outcome ~ metric * ({X1} + {X2} + C(survey_year) + C(size_cat))", "A21_YEAR_SIZE",
         "adjustment_set_ablation", "year_plus_size", "remove_sector"),
    ]
    for formula, run_id, family, variant, factor in ablations:
        rows.append(_fit(df, formula, run_id, family, variant, factor))

    items = config["y2_substantive_items"]
    for omitted in items:
        rows.append(_fit(df, primary, f"A30_Y2_DROP_{omitted.upper()}", "y2_leave_one_item_out",
                         f"drop_{omitted}", f"y2_omit_{omitted}",
                         second=derive_harm_leave_one_out(df, items, omitted)))

    d4 = df.loc[df["y4_identified_impactful_incident"].notna()].copy()
    rows.append(_fit(d4, primary, "A40_Y4_CONSEQUENCE", "consequence_definition", "y1_vs_y4", "replace_y2_with_y4",
                     second=d4["y4_identified_impactful_incident"].to_numpy()))
    rows.append(_fit(df, primary, "R30_PHISHING_SPECIFICITY", "phishing_specificity",
                     "broad_vs_nonphishing_incident", "second_indicator",
                     second=df["y1_nonphishing_incident"].to_numpy()))

    for group_value, label in ((1.0, "micro_small"), (2.0, "medium_large")):
        dg = df.loc[df["size_ms_ml"].eq(group_value)].copy()
        formula = f"outcome ~ metric * ({X1} + {X2} + C(survey_year) + C(sector_cat))"
        rows.append(_fit(dg, formula, f"R40_SIZE_{label.upper()}", "size_stability", label, "size_subset"))

    for sector in sorted(df["sector_cat"].dropna().unique()):
        ds = df.loc[~df["sector_cat"].eq(sector)].copy()
        rows.append(_fit(ds, primary, f"R50_DROP_SECTOR_{int(sector)}", "leave_one_sector_out",
                         f"drop_sector_{int(sector)}", "sector_exclusion"))
    return rows


def run_heterogeneity(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Formal year and size heterogeneity of the discordance."""
    year_formula = (
        f"outcome ~ metric * ({X1} + {X2} + C(survey_year) + C(size_cat) + C(sector_cat))"
        f" + metric:{X1}:C(survey_year) + metric:{X2}:C(survey_year)"
    )
    size_formula = (
        f"outcome ~ metric * ({X1} + {X2} + C(survey_year) + C(size_ms_ml) + C(sector_cat))"
        f" + metric:{X1}:C(size_ms_ml) + metric:{X2}:C(size_ms_ml)"
    )
    return {
        "year": fit_heterogeneity(df, "survey_year", "C(survey_year)[T.2025]", year_formula,
                                  "R12_YEAR_HETEROGENEITY", "temporal_stability", "formal_2025_vs_2024"),
        "size": fit_heterogeneity(df, "size_ms_ml", "C(size_ms_ml)[T.2.0]", size_formula,
                                  "R42_SIZE_HETEROGENEITY", "size_stability", "medium_large_vs_micro_small"),
    }


def sign_stability(registry: pd.DataFrame, primary_results: Dict[str, Any]) -> pd.DataFrame:
    """Mark whether each variant keeps the sign of the primary discordance."""
    primary_x1 = float(primary_results["effects"]["x1_discordance"]["estimate"])
    primary_x2 = float(primary_results["effects"]["x2_discordance"]["estimate"])
    table = registry.loc[~registry["family"].eq("phishing_specificity")].copy()
    for label, reference in (("x1", primary_x1), ("x2", primary_x2)):
        table[f"{label}_same_sign_as_primary"] = np.where(
            table[f"{label}_estimate"].notna(),
            np.sign(table[f"{label}_estimate"]) == np.sign(reference),
            np.nan,
        )
    return table


def main() -> None:
    """Run the robustness analyses and write their result files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    mode = "smoke" if smoke else "official"
    config_path = CONFIG_DIR / "robustness.json"
    config = read_json(config_path)

    input_path = REPO_ROOT / config["input_csv"]
    input_sha = verify_sha256(input_path, config["input_sha256_prefix"], "primary business table")
    df = pd.read_csv(input_path)
    if len(df) != config["expected_business_n"]:
        raise AssertionError("Unexpected primary N")

    primary_path = REPO_ROOT / config["primary_results"]
    if not primary_path.exists():
        raise FileNotFoundError(f"{primary_path} not found. Run experiments/primary_discordance.py first.")
    primary_results = read_json(primary_path)
    if input_sha != primary_results["input_sha256"]:
        raise AssertionError("Primary input hash changed since the primary analysis was run")

    out_dir = RESULTS_DIR / "robustness" / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.DataFrame(run_lpm_variants(df, config))
    registry.to_csv(out_dir / "robustness_lpm_registry.csv", index=False)
    heterogeneity = run_heterogeneity(df)
    write_json(out_dir / "formal_heterogeneity.json", heterogeneity)

    point = logit_point_estimates(df)
    replicates = SMOKE_REPLICATES if smoke else int(config["logistic_bootstrap_replicates"])
    boot, failures = bootstrap_logit(df, replicates, int(config["bootstrap_seed"]), 5 if smoke else 200)
    boot.to_csv(out_dir / "logit_bootstrap.csv", index=False)
    summary = bootstrap_summary(point, boot)
    summary.update({
        "replicates_requested": replicates,
        "replicates_failed": int(failures),
        "success_rate": float(len(boot) / replicates) if replicates else 0.0,
        "seed": int(config["bootstrap_seed"]),
        "mode": mode,
    })
    write_json(out_dir / "logit_marginal_standardization.json", summary)
    if summary["success_rate"] < config["logistic_min_success_rate"]:
        logger.warning("bootstrap success rate %.3f is below the required %.2f",
                       summary["success_rate"], config["logistic_min_success_rate"])

    sign_stability(registry, primary_results).to_csv(out_dir / "robustness_sign_stability.csv", index=False)
    write_json(out_dir / "run_metadata.json", {
        "mode": mode,
        "config_sha256": sha256_file(config_path),
        "input_sha256": input_sha,
        "primary_result_sha256": sha256_file(primary_path),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "lpm_runs": int(len(registry)),
        "logit_bootstrap_requested": replicates,
        "logit_bootstrap_success": int(len(boot)),
        "logit_bootstrap_failures": int(failures),
    })
    logger.info("%d linear variants; bootstrap %d/%d succeeded; results in %s",
                len(registry), len(boot), replicates, out_dir.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()
