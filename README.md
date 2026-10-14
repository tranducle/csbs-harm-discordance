# csbs-harm-discordance

Code for an analysis of the UK Cyber Security Breaches Survey (CSBS). It asks whether two security practices, a staff awareness test and security monitoring, are associated in the same way with two business-level outcomes:

- **identified incident**: the business reports at least one cyber breach or attack it identified in the past 12 months;
- **harmful incident**: the business reports at least one identified incident that had a negative outcome (any substantive item of the survey's outcome question).

Every harmful incident is also an identified incident, so the two outcomes are nested. The analysis estimates the *discordance*, the practice's association with the harmful incident minus its association with the identified incident, using survey-weighted linear probability models on 4,179 businesses from the 2024 and 2025 waves.

The associations are descriptive. The repository makes no causal claim and does not estimate true incident prevalence.

## Main results

| Practice | Discordance (percentage points) | 95% CI | Holm-adjusted p |
| - | - | - | - |
| Staff awareness test | -5.06 | -9.92 to -0.20 | 0.041 |
| Security monitoring | -14.98 | -19.15 to -10.81 | < 0.001 |

Both practices are more weakly associated with harmful incidents than with identified incidents. The full outputs of the reference run are in [results/reference](results/reference).

## Repository layout

```
src/csbs_harm/     reusable code: indicators, preprocessing, weighted statistics,
                   decomposition, robustness model helpers
experiments/       one runner per analysis (primary, robustness, conditional harm share,
                   discordance decomposition)
scripts/           data preparation and indicator validation
configs/           analysis specifications: formulas, seeds, expected sample sizes, checksums
tests/             unit tests and checks against the reference results
results/reference/ aggregate outputs of the reference run
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

The survey microdata are not part of this repository. [DATA.md](DATA.md) explains how to obtain them from the UK Data Service. After placing them under `data/raw/`, run the pipeline described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md):

```bash
python scripts/prepare_csbs_data.py
python scripts/validate_harm_indicator.py
python experiments/primary_discordance.py
python experiments/robustness_checks.py
python experiments/conditional_harm_share.py
python experiments/discordance_decomposition.py
```

## Data and licence

The code is released under the MIT licence. The CSBS data are governed by the UK Data Service End User Licence and must not be redistributed, so no survey records are included here.

## Citation

Citation details will be added when the article is available.
