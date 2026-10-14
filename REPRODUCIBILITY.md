# Reproducing the results

## Environment

The reference run used Python 3.9.6 on macOS with the versions pinned in `requirements.txt`: numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, statsmodels 0.14.6 and patsy 1.0.3.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every command below is run from the repository root. Paths are resolved relative to the repository, or to the directory named in the environment variable `CSBS_HARM_ROOT` if it is set.

## Steps

1. **Get the data.** Follow [DATA.md](DATA.md) so that both Stata files sit under `data/raw/`.
2. **Unit tests.** `pytest` runs without the data. Tests that need the processed table are skipped until step 3 has run.
3. **Prepare the analysis tables.**
   ```bash
   python scripts/prepare_csbs_data.py
   ```
   This writes three CSV files and a manifest to `data/processed/`. The script stops if a row count differs from the expected 7,269 respondents, 4,180 businesses or 4,179 complete cases.
4. **Validate the harmful-incident indicator.**
   ```bash
   python scripts/validate_harm_indicator.py
   ```
   Expected: `STATUS PASS`. The indicator matches the official 2025 `outcome_any` variable on every record, and the weighted shares reproduce the published 13% (2024) and 16% (2025).
5. **Primary analysis.**
   ```bash
   python experiments/primary_discordance.py
   ```
   Expected discordance: -0.0506 for the staff awareness test and -0.1498 for security monitoring.
6. **Robustness analyses.** These read the primary results, so run step 5 first.
   ```bash
   python experiments/robustness_checks.py smoke
   python experiments/robustness_checks.py
   ```
   The first command runs 20 bootstrap replicates and finishes quickly. The second runs the full 2,000 replicates and takes about six minutes. Both write 37 linear-model variants.
7. **Secondary analyses.**
   ```bash
   python experiments/conditional_harm_share.py
   python experiments/discordance_decomposition.py
   ```
   The first includes a 2,000-replicate bootstrap and takes about three minutes. These analyses were run after the primary and robustness results were known, so they are exploratory.
8. **Tests against the reference.** Run `pytest` again. With the processed table present, it also checks the table checksum and the decomposition against `results/reference/`.

## Determinism

All random draws use `numpy.random.default_rng` with seed 20271012, set in `configs/`. The bootstrap resamples businesses within survey year. With the pinned package versions the reference run reproduces exactly:

- the three processed CSV files, byte for byte;
- the coefficient CSV of the primary model, byte for byte;
- every number in the JSON result files.

Other package versions may change results in the last digits.

## Where each output comes from

| Output | Produced by | Configuration |
| - | - | - |
| Processed tables, manifest | `scripts/prepare_csbs_data.py` | none |
| Indicator validation | `scripts/validate_harm_indicator.py` | none |
| Primary estimates | `experiments/primary_discordance.py` | `configs/primary_analysis.json` |
| Robustness registry, bootstrap, heterogeneity | `experiments/robustness_checks.py` | `configs/robustness.json` |
| Conditional harm share | `experiments/conditional_harm_share.py` | `configs/secondary_analyses.json` |
| Decomposition of the discordance | `experiments/discordance_decomposition.py` | `configs/secondary_analyses.json` |

## Decomposition identity

Write p for the share of businesses with an identified incident and s for the share of those whose incident was harmful, each at practice status 1 and 0. Because harmful incidents are a subset of identified incidents, the difference between the two risk differences satisfies exactly

RD(harmful) - RD(identified) = (s1 - 1)(p1 - p0) + p0 (s1 - s0).

The first term is the nesting component and the second the conditional-severity component. `tests/test_decomposition.py` checks the identity on simulated nested data.
