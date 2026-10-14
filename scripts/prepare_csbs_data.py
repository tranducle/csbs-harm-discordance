"""Build the harmonised analysis tables from the 2024 and 2025 CSBS archive files.

Usage:
    python scripts/prepare_csbs_data.py

Reads the two Stata files described in DATA.md from data/raw/ and writes three CSV files
and a manifest to data/processed/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csbs_harm.common import PROCESSED_DIR, relative_to_repo, sha256_file, write_json  # noqa: E402
from csbs_harm.preprocessing import (  # noqa: E402
    RAW_FILES,
    VARIABLE_MAPPING,
    combine_years,
    transform_year,
)

logger = logging.getLogger("prepare_csbs_data")

DISTRIBUTION_COLUMNS = [
    "x1_staff_awareness_test", "x2_security_monitoring",
    "y1_identified_incident", "y1_nonphishing_incident",
    "y2_identified_harmful_incident", "y4_identified_impactful_incident",
    "size_cat", "sector_cat",
]
CHECKED_BINARY_COLUMNS = [
    "x1_staff_awareness_test", "x2_security_monitoring",
    "y1_identified_incident", "y2_identified_harmful_incident",
]


def _distributions(primary: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Count the categories of every analysis column by survey year."""
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for year in (2024, 2025):
        subset = primary.loc[primary["survey_year"].eq(year)]
        out[str(year)] = {
            col: {str(k): int(v) for k, v in subset[col].value_counts(dropna=False).sort_index().items()}
            for col in DISTRIBUTION_COLUMNS
        }
    return out


def _roundtrip_check(primary_path: Path, expected_rows: int) -> None:
    """Re-read the primary CSV and confirm row count, unique ids and binary domains."""
    roundtrip = pd.read_csv(primary_path)
    if len(roundtrip) != expected_rows:
        raise AssertionError("Primary CSV roundtrip row count mismatch")
    if roundtrip["record_id"].duplicated().any():
        raise AssertionError("Primary CSV roundtrip duplicate record_id")
    for column in CHECKED_BINARY_COLUMNS:
        values = set(roundtrip[column].dropna().unique().tolist())
        if not values.issubset({0.0, 1.0}):
            raise AssertionError(f"Primary CSV domain invalid for {column}: {values}")


def main() -> None:
    """Harmonise both years, write the tables and the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_manifest, frames, year_metrics = {}, [], {}
    for year, path in RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. See DATA.md for how to obtain it.")
        raw_manifest[str(year)] = {
            "path": relative_to_repo(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        frame, metrics = transform_year(year, path)
        frames.append(frame)
        year_metrics[str(year)] = metrics

    full, business_all, primary = combine_years(frames)
    outputs = {
        "csbs_2024_2025_harmonized_full.csv": full,
        "csbs_2024_2025_business_all.csv": business_all,
        "csbs_2024_2025_primary_business.csv": primary,
    }
    for name, frame in outputs.items():
        frame.to_csv(PROCESSED_DIR / name, index=False)

    exclusions = business_all["primary_exclusion_reason"].value_counts(dropna=False).to_dict()
    manifest = {
        "preprocessing_version": "1.0",
        "raw_files": raw_manifest,
        "year_metrics": year_metrics,
        "combined_metrics": {
            "harmonized_full_rows": int(len(full)),
            "business_rows_before_primary_complete_filter": int(len(business_all)),
            "primary_complete_rows": int(len(primary)),
            "primary_exclusion_counts": {str(k): int(v) for k, v in exclusions.items()},
            "duplicate_record_ids_full": int(full["record_id"].duplicated().sum()),
            "duplicate_record_ids_primary": int(primary["record_id"].duplicated().sum()),
            "primary_weight_missing": int(primary["survey_weight"].isna().sum()),
            "primary_weight_nonpositive": int((primary["survey_weight"] <= 0).sum()),
            "primary_year_counts": {
                str(k): int(v) for k, v in primary["survey_year"].value_counts().sort_index().items()
            },
            "primary_y1_positive": int(primary["y1_identified_incident"].eq(1).sum()),
            "primary_y2_positive": int(primary["y2_identified_harmful_incident"].eq(1).sum()),
            "primary_y4_positive": int(primary["y4_identified_impactful_incident"].eq(1).sum()),
        },
        "mapping": VARIABLE_MAPPING,
        "distributions_primary": _distributions(primary),
        "outputs": {
            name: {
                "path": relative_to_repo(PROCESSED_DIR / name),
                "size_bytes": (PROCESSED_DIR / name).stat().st_size,
                "sha256": sha256_file(PROCESSED_DIR / name),
            }
            for name in outputs
        },
    }
    manifest_path = PROCESSED_DIR / "csbs_2024_2025_preprocessing_manifest.json"
    write_json(manifest_path, manifest)

    _roundtrip_check(PROCESSED_DIR / "csbs_2024_2025_primary_business.csv", len(primary))
    print(json.dumps({
        "status": "PASS",
        "full_rows": len(full),
        "business_rows": len(business_all),
        "primary_rows": len(primary),
        "exclusions": {str(k): int(v) for k, v in exclusions.items()},
        "manifest": relative_to_repo(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
