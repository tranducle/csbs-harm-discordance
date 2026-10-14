"""Harmonise the 2024 and 2025 CSBS business files into one analysis table."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from csbs_harm.common import RAW_DIR
from csbs_harm.indicators import (
    binary_01,
    categorical_valid,
    derive_harmful_incident,
    derive_impactful_incident,
    recode_trained,
)

logger = logging.getLogger(__name__)

RAW_FILES: Dict[int, Path] = {
    2024: RAW_DIR / "UKDA-9285-stata" / "stata" / "stata13" / "csbs_2024_archive_data_public.dta",
    2025: RAW_DIR / "UKDA-9404-stata" / "stata" / "stata13" / "csbs_2025_archive_data_public.dta",
}

RAW_KEEP: List[str] = (
    [
        "weight", "questtype", "sizeb", "sector_comb2",
        "ident11", "ident14", "ident_comb", "audit",
        "trained", "Step2", "Step8", "type_comb1", "type_comb2",
    ]
    + [f"outcome{i}" for i in range(1, 14)]
    + [f"impact{i}" for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]]
)

DERIVED_KEEP: List[str] = [
    "survey_year", "source_row", "record_id", "survey_weight", "is_business",
    "x1_staff_awareness_test", "x2_security_monitoring", "ident_any",
    "trained_binary", "step2_engagement_training", "step8_logging_monitoring",
    "size_cat", "size_ms_ml", "sector_cat",
    "y1_identified_incident", "y1_nonphishing_incident",
    "y2_identified_harmful_incident", "y4_identified_impactful_incident",
    "primary_complete", "primary_exclusion_reason",
]

REQUIRED_PRIMARY: List[str] = [
    "survey_weight", "x1_staff_awareness_test", "x2_security_monitoring",
    "size_cat", "sector_cat", "y1_identified_incident", "y2_identified_harmful_incident",
]

EXPECTED_FULL_ROWS = 7269
EXPECTED_BUSINESS_ROWS = 4180
EXPECTED_PRIMARY_ROWS = 4179

BUSINESS_QUESTIONNAIRE_CODE = 1

VARIABLE_MAPPING: Dict[str, str] = {
    "x1_staff_awareness_test": "ident14, 1 Yes, 0 No, all other codes missing",
    "x2_security_monitoring": "ident11, 1 Yes, 0 No, all other codes missing",
    "y1_identified_incident": "type_comb1, 1 Yes, 0 No, all other codes missing",
    "y1_nonphishing_incident": "type_comb2, 1 Yes, 0 No, all other codes missing",
    "y2_identified_harmful_incident": (
        "0 if Y1=0; if Y1=1 then 1 iff any outcome1-8 or outcome11-13 equals 1, else 0. "
        "This reproduces official 2025 outcome_any exactly."
    ),
    "y4_identified_impactful_incident": (
        "0 if Y1=0; if Y1=1 then 1 iff any impact1-5,7-10,13-14 equals 1; "
        "0 for explicit None of these; missing for Don't know-only."
    ),
    "trained_binary": "trained: 1 Yes -> 1; 2 No -> 0; 2024 -97 and 2025 997 treated missing",
    "survey_weight": "official annual weight retained unchanged; no cross-year renormalization",
}


def _exclusion_reasons(df: pd.DataFrame) -> List[str]:
    """Label each row as complete, not a business, or missing specific required columns."""
    missing = df[REQUIRED_PRIMARY].isna().to_numpy()
    is_business = df["is_business"].to_numpy() == 1
    reasons: List[str] = []
    for row_missing, business in zip(missing, is_business):
        if not business:
            reasons.append("not_primary_business")
            continue
        names = [c for c, flag in zip(REQUIRED_PRIMARY, row_missing) if flag]
        reasons.append("complete" if not names else "missing:" + ",".join(names))
    return reasons


def transform_year(year: int, path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Harmonise one survey year.

    Args:
        year: Survey year, 2024 or 2025.
        path: Stata file of the public archive release for that year.

    Returns:
        The harmonised frame (all respondents) and a dictionary of quality metrics.

    Raises:
        RuntimeError: If a required source column is missing.
        AssertionError: If row preservation, weights, nesting or the official 2025
            ``outcome_any`` cross-check fails.
    """
    raw = pd.read_stata(path, convert_categoricals=False)
    missing_cols = [c for c in RAW_KEEP if c not in raw.columns]
    if missing_cols:
        raise RuntimeError(f"{year} missing required columns: {missing_cols}")

    df = raw[RAW_KEEP].copy()
    df.insert(0, "survey_year", year)
    df.insert(1, "source_row", np.arange(len(df), dtype=np.int64))
    df.insert(2, "record_id", [f"{year}:{i:06d}" for i in range(len(df))])

    df["survey_weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df.loc[~np.isfinite(df["survey_weight"]) | df["survey_weight"].le(0), "survey_weight"] = np.nan

    df["is_business"] = df["questtype"].eq(BUSINESS_QUESTIONNAIRE_CODE).astype(int)
    df["x1_staff_awareness_test"] = binary_01(df["ident14"])
    df["x2_security_monitoring"] = binary_01(df["ident11"])
    df["ident_any"] = binary_01(df["ident_comb"])
    df["trained_binary"] = recode_trained(df["trained"])
    df["step2_engagement_training"] = binary_01(df["Step2"])
    df["step8_logging_monitoring"] = binary_01(df["Step8"])
    df["size_cat"] = categorical_valid(df["sizeb"], {1, 2, 3, 4})
    df["size_ms_ml"] = categorical_valid(df["sizeb"].replace({1: 1, 2: 1, 3: 2, 4: 2}), {1, 2})
    df["sector_cat"] = categorical_valid(df["sector_comb2"], set(range(1, 14)))

    df["y1_identified_incident"] = binary_01(df["type_comb1"])
    df["y1_nonphishing_incident"] = binary_01(df["type_comb2"])
    df["y2_identified_harmful_incident"] = derive_harmful_incident(df)
    df["y4_identified_impactful_incident"] = derive_impactful_incident(df)

    complete = df["is_business"].eq(1)
    for column in REQUIRED_PRIMARY:
        complete &= df[column].notna()
    df["primary_complete"] = complete.astype(int)
    df["primary_exclusion_reason"] = _exclusion_reasons(df)

    if len(df) != len(raw):
        raise AssertionError(f"{year} row preservation failed")
    if df["record_id"].duplicated().any():
        raise AssertionError(f"{year} record_id duplicate")
    if df["survey_weight"].isna().sum() != 0:
        raise AssertionError(f"{year} invalid weights")

    valid_pair = df["y1_identified_incident"].notna() & df["y2_identified_harmful_incident"].notna()
    if (df.loc[valid_pair, "y2_identified_harmful_incident"] > df.loc[valid_pair, "y1_identified_incident"]).any():
        raise AssertionError(f"{year} harmful incident cannot exceed identified incident")

    official_mismatch = None
    if year == 2025 and "outcome_any" in raw.columns:
        official = raw["outcome_any"]
        valid = official.isin([0, 1]) & df["y2_identified_harmful_incident"].notna()
        derived = df.loc[valid, "y2_identified_harmful_incident"].astype(int)
        official_mismatch = int((derived.to_numpy() != official.loc[valid].astype(int).to_numpy()).sum())
        if official_mismatch != 0:
            raise AssertionError(f"2025 outcome_any validation mismatch: {official_mismatch}")

    business = df["is_business"].eq(1)
    metrics: Dict[str, object] = {
        "raw_rows": int(len(raw)),
        "harmonized_rows": int(len(df)),
        "business_rows": int(df["is_business"].sum()),
        "primary_complete_rows": int(df["primary_complete"].sum()),
        "weight_missing": int(df["survey_weight"].isna().sum()),
        "weight_nonpositive": int((df["survey_weight"].fillna(0) <= 0).sum()),
        "y1_missing": int(df["y1_identified_incident"].isna().sum()),
        "y2_missing": int(df["y2_identified_harmful_incident"].isna().sum()),
        "y4_missing": int(df["y4_identified_impactful_incident"].isna().sum()),
        "business_x1_positive": int((business & df["x1_staff_awareness_test"].eq(1)).sum()),
        "business_x2_positive": int((business & df["x2_security_monitoring"].eq(1)).sum()),
        "business_y1_positive": int((business & df["y1_identified_incident"].eq(1)).sum()),
        "business_y2_positive": int((business & df["y2_identified_harmful_incident"].eq(1)).sum()),
        "business_y4_positive": int((business & df["y4_identified_impactful_incident"].eq(1)).sum()),
        "outcome_any_mismatch_2025": official_mismatch,
        "trained_missing_after_recode": int(df["trained_binary"].isna().sum()),
    }
    logger.info("%d: %d rows, %d businesses, %d primary-complete",
                year, metrics["raw_rows"], metrics["business_rows"], metrics["primary_complete_rows"])
    return df, metrics


def combine_years(frames: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenate the yearly frames and derive the business and primary subsets.

    Returns:
        ``(full, business_all, primary)``.

    Raises:
        AssertionError: If a row count differs from the expected sample accounting.
    """
    full = pd.concat(frames, axis=0, ignore_index=True)[DERIVED_KEEP + RAW_KEEP]
    if len(full) != EXPECTED_FULL_ROWS:
        raise AssertionError(f"Expected {EXPECTED_FULL_ROWS} harmonized rows, got {len(full)}")
    if full["record_id"].duplicated().any():
        raise AssertionError("Cross-year record_id duplicate")

    business_all = full.loc[full["is_business"].eq(1)].copy()
    primary = business_all.loc[business_all["primary_complete"].eq(1)].copy()
    if len(business_all) != EXPECTED_BUSINESS_ROWS:
        raise AssertionError(f"Expected {EXPECTED_BUSINESS_ROWS} business rows, got {len(business_all)}")
    if len(primary) != EXPECTED_PRIMARY_ROWS:
        raise AssertionError(f"Expected {EXPECTED_PRIMARY_ROWS} primary-complete rows, got {len(primary)}")
    return full, business_all, primary
