"""Descriptive decomposition of the unadjusted discordance (no model, no inference).

Usage:
    python experiments/discordance_decomposition.py

Every harmful incident is also an identified incident, so the difference between the two risk
differences splits exactly into a nesting component and a conditional-severity component. The
identity is documented in csbs_harm.decomposition. The adjusted estimates from
experiments/primary_discordance.py remain the reported results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csbs_harm.common import CONFIG_DIR, REPO_ROOT, RESULTS_DIR, read_json, verify_sha256, write_json  # noqa: E402
from csbs_harm.decomposition import decompose  # noqa: E402

logger = logging.getLogger("discordance_decomposition")

OUT_PATH = RESULTS_DIR / "conditional_harm_share" / "discordance_decomposition.json"


def main() -> None:
    """Decompose the unadjusted discordance for both practices and write the result."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = read_json(CONFIG_DIR / "secondary_analyses.json")
    tolerance = config["discordance_decomposition"]["identity_tolerance"]

    input_path = REPO_ROOT / config["input_csv"]
    sha = verify_sha256(input_path, config["input_sha256_prefix"], "primary business table")
    df = pd.read_csv(input_path)
    out = {
        "status_label": "post hoc descriptive decomposition; unadjusted weighted proportions; no inference",
        "input_sha256": sha,
        "identity": "RD(Y2) - RD(Y1) = (s1 - 1)(p1 - p0) + p0 (s1 - s0)",
        "practices": {p: decompose(df, p, tolerance) for p in config["practices"]},
    }
    write_json(OUT_PATH, out)
    for practice, parts in out["practices"].items():
        logger.info("%s: delta %.4f = nesting %.4f + severity %.4f",
                    practice, parts["delta_unadjusted"], parts["nesting_component"], parts["severity_component"])


if __name__ == "__main__":
    main()
