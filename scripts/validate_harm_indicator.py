"""Validate the reconstructed harmful-incident indicator in both CSBS waves.

The 2025 release carries an official derived variable (outcome_any), so the indicator can be
matched record by record. The 2024 release has no such variable, so 2024 is checked in three
other ways: item routing and coding, an independent re-derivation against the processed file,
and reproduction of the published headline statistics.

Published reference values, DSIT Cyber Security Breaches Survey 2025 statistical release
(UKDA SN 9404 documentation, "Trends over time"): 16% of businesses in 2025 compared with 13%
in 2024. The 2024 release (UKDA SN 9285 documentation, section on outcomes) adds a base of
1,111 businesses and 32% for large businesses. The Q56A item list is identical in the 2024 and
2025 questionnaires (technical reports, Q56A_OUTCOME); the UKDA data dictionaries carry older
labels for outcome6 and outcome11.

Usage:
    python scripts/validate_harm_indicator.py

Requires the raw files from DATA.md and the output of scripts/prepare_csbs_data.py. Writes
results/validation/harm_indicator_benchmark.json and exits with status 1 if any check fails.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csbs_harm.common import PROCESSED_DIR, RESULTS_DIR, relative_to_repo, write_json  # noqa: E402
from csbs_harm.indicators import OUTCOME_SUBSTANTIVE  # noqa: E402
from csbs_harm.preprocessing import BUSINESS_QUESTIONNAIRE_CODE, RAW_FILES  # noqa: E402

logger = logging.getLogger("validate_harm_indicator")

PROCESSED = PROCESSED_DIR / "csbs_2024_2025_harmonized_full.csv"
OUT_PATH = RESULTS_DIR / "validation" / "harm_indicator_benchmark.json"
ALL_ITEMS = [f"outcome{i}" for i in range(1, 14)]
LARGE_CODE = 4
PUBLISHED = {
    2024: {"base_businesses": 1111, "business_pct": 13, "large_business_pct": 32},
    2025: {"business_pct": 16},
}
ROUNDING_TOLERANCE_PP = 0.5


def _weighted_pct(values: pd.Series, weights: pd.Series) -> float:
    """Weighted percentage of true values."""
    return float(100 * np.average(values.astype(float), weights=weights))


def check_year(year: int, processed: pd.DataFrame) -> Dict[str, object]:
    """Run every check for one survey wave."""
    raw = pd.read_stata(RAW_FILES[year], convert_categoricals=False)
    incident = raw["type_comb1"].eq(1)
    asked = (raw[ALL_ITEMS] != -1).any(axis=1)
    substantive = raw[OUTCOME_SUBSTANTIVE].eq(1).any(axis=1)
    none_or_dk = raw[["outcome9", "outcome10"]].eq(1).any(axis=1)
    weights = raw["weight"]
    business = raw["questtype"].eq(BUSINESS_QUESTIONNAIRE_CODE)

    fresh = pd.Series(np.nan, index=raw.index)
    fresh[raw["type_comb1"].eq(0)] = 0.0
    fresh[incident] = substantive[incident].astype(float)
    proc = processed[processed["survey_year"] == year].sort_values("source_row").reset_index(drop=True)
    proc_harm = proc["y2_identified_harmful_incident"]
    both = proc_harm.notna() & fresh.notna()

    base = business & incident
    result: Dict[str, object] = {
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(proc)),
        "incident_asked": int((asked & incident).sum()),
        "incident_not_asked": int((~asked & incident).sum()),
        "asked_without_incident": int((asked & ~incident).sum()),
        "none_or_dk_with_consequence": int((none_or_dk & substantive & incident).sum()),
        "incident_with_no_code": int((~substantive & ~none_or_dk & incident).sum()),
        "processed_vs_fresh_mismatch": int((proc_harm[both] != fresh[both]).sum()),
        "processed_vs_fresh_missing_pattern_equal": bool((proc_harm.isna() == fresh.isna()).all()),
        "business_incident_base": int(base.sum()),
        "business_pct": round(_weighted_pct(substantive[base], weights[base]), 2),
    }
    large = base & raw["sizeb"].eq(LARGE_CODE)
    result["large_business_pct"] = round(_weighted_pct(substantive[large], weights[large]), 2)
    if "outcome_any" in raw.columns:
        valid = raw["outcome_any"].isin([0, 1]) & incident
        result["official_outcome_any_records"] = int(valid.sum())
        result["official_outcome_any_mismatch"] = int(
            (raw.loc[valid, "outcome_any"].astype(int) != substantive[valid].astype(int)).sum()
        )
    else:
        result["official_outcome_any_records"] = 0
        result["official_outcome_any_note"] = "variable not present in this release"

    published = PUBLISHED[year]
    result["published"] = published
    checks = {
        "routing_clean": result["incident_not_asked"] == 0,
        "coding_clean": result["none_or_dk_with_consequence"] == 0 and result["incident_with_no_code"] == 0,
        "processed_matches_fresh": result["processed_vs_fresh_mismatch"] == 0
        and result["processed_vs_fresh_missing_pattern_equal"],
        "business_pct_matches_published": abs(result["business_pct"] - published["business_pct"])
        <= ROUNDING_TOLERANCE_PP,
    }
    if "base_businesses" in published:
        checks["base_matches_published"] = result["business_incident_base"] == published["base_businesses"]
    if "large_business_pct" in published:
        checks["large_pct_matches_published"] = (
            abs(result["large_business_pct"] - published["large_business_pct"]) <= ROUNDING_TOLERANCE_PP
        )
    if result["official_outcome_any_records"]:
        checks["official_outcome_any_match"] = result["official_outcome_any_mismatch"] == 0
    result["checks"] = checks
    return result


def main() -> None:
    """Run both waves, write the JSON report, and exit non-zero if any check fails."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    processed = pd.read_csv(PROCESSED, low_memory=False)
    report: Dict[str, object] = {str(year): check_year(year, processed) for year in RAW_FILES}
    failed = [
        f"{year}:{name}"
        for year, res in report.items()
        for name, ok in res["checks"].items()  # type: ignore[index]
        if not ok
    ]
    report["status"] = "PASS" if not failed else "FAIL"
    report["failed_checks"] = failed
    write_json(OUT_PATH, report)
    for year in RAW_FILES:
        res = report[str(year)]
        logger.info(
            "%s: base %s, business %.2f%% (published %s%%), large %.2f%%, checks %s",
            year, res["business_incident_base"], res["business_pct"],  # type: ignore[index]
            res["published"]["business_pct"], res["large_business_pct"],  # type: ignore[index]
            "all pass" if all(res["checks"].values()) else res["checks"],  # type: ignore[index]
        )
    logger.info("STATUS %s; report %s", report["status"], relative_to_repo(OUT_PATH))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
