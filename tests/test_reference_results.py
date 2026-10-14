"""Check the processed data and the published reference results.

These tests need the processed analysis table (see DATA.md and scripts/prepare_csbs_data.py)
and are skipped when it has not been built.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from csbs_harm.common import PROCESSED_DIR, REPO_ROOT, sha256_file
from csbs_harm.decomposition import decompose

PRIMARY_CSV = PROCESSED_DIR / "csbs_2024_2025_primary_business.csv"
REFERENCE = REPO_ROOT / "results" / "reference"

needs_data = pytest.mark.skipif(not PRIMARY_CSV.exists(), reason="processed data not built")


@needs_data
def test_primary_table_checksum():
    assert sha256_file(PRIMARY_CSV).startswith("57df72b0")


@needs_data
def test_primary_table_shape_and_nesting():
    df = pd.read_csv(PRIMARY_CSV)
    assert len(df) == 4179
    assert df["record_id"].is_unique
    assert (df["y2_identified_harmful_incident"] <= df["y1_identified_incident"]).all()


@needs_data
def test_decomposition_matches_reference():
    df = pd.read_csv(PRIMARY_CSV)
    reference = json.loads((REFERENCE / "discordance_decomposition.json").read_text(encoding="utf-8"))
    for practice, expected in reference["practices"].items():
        actual = decompose(df, practice, 1e-12)
        assert actual["delta_unadjusted"] == pytest.approx(expected["delta_unadjusted"], abs=1e-12)
        assert actual["nesting_component"] == pytest.approx(expected["nesting_component"], abs=1e-12)


def test_reference_primary_estimates():
    results = json.loads((REFERENCE / "primary_results.json").read_text(encoding="utf-8"))
    assert results["n_business"] == 4179
    assert round(100 * results["effects"]["x1_discordance"]["estimate"], 2) == -5.06
    assert round(100 * results["effects"]["x2_discordance"]["estimate"], 2) == -14.98
