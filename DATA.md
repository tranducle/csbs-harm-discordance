# Data access

The analysis uses two public-archive releases of the Cyber Security Breaches Survey, a UK government survey of businesses commissioned by the Department for Science, Innovation and Technology. The files are distributed by the UK Data Service under an End User Licence. Registration is free, but the data may not be redistributed, so this repository contains code only.

| Wave | UK Data Service study | DOI |
| - | - | - |
| 2024 | SN 9285, Cyber Security Breaches Survey, 2024 | 10.5255/UKDA-SN-9285-1 |
| 2025 | SN 9404, Cyber Security Breaches Survey, 2025 | 10.5255/UKDA-SN-9404-1 |

## Obtaining the files

1. Register with the UK Data Service and accept the End User Licence for each study.
2. Download the **Stata** version of each study and unzip it into `data/raw/`.
3. Check that these two files exist:

```
data/raw/UKDA-9285-stata/stata/stata13/csbs_2024_archive_data_public.dta
data/raw/UKDA-9404-stata/stata/stata13/csbs_2025_archive_data_public.dta
```

The `data/` directory is excluded from version control.

## Checksums of the files used

The reference run used files with these SHA-256 digests. A different digest means a different release of the study, and results may differ.

| File | SHA-256 |
| - | - |
| csbs_2024_archive_data_public.dta | de7238376d47828a09d61fe5955547616808d6483ebfc96a80c115a0567763df |
| csbs_2025_archive_data_public.dta | 6e796f0023d76f2b8b4b9ded2fef97249e0e8bdf7d09638eeb66603e3eb980d7 |

`scripts/prepare_csbs_data.py` writes three tables to `data/processed/`. Their digests in the reference run were:

| File | SHA-256 |
| - | - |
| csbs_2024_2025_harmonized_full.csv | 89f03aa05d2a62028bf5fbe52a7ad0971f92a300fa59c5d537156b68f1d6942b |
| csbs_2024_2025_business_all.csv | f3dadc34221714628995f070f2b7960903a9ab6aca5e32f6cc2fa3a38730233e |
| csbs_2024_2025_primary_business.csv | 57df72b01f8ad9e08d81a1b2235b4407f569afa3ca365ddd101e64ea1844881b |

The analysis scripts refuse to run if the primary table does not match the prefix recorded in `configs/`.

## Sample accounting

| Step | Rows |
| - | - |
| Respondents in both waves | 7,269 (3,434 in 2024, 3,835 in 2025) |
| Business questionnaires | 4,180 |
| Businesses with every analysis variable present | 4,179 |

One business is dropped because its identified-incident indicator is missing. Official survey weights are used unchanged, with a separate weight for each wave.

## Variables

| Derived variable | Source items | Coding |
| - | - | - |
| x1_staff_awareness_test | ident14 | 1 yes, 0 no, other codes missing |
| x2_security_monitoring | ident11 | 1 yes, 0 no, other codes missing |
| y1_identified_incident | type_comb1 | 1 yes, 0 no, other codes missing |
| y2_identified_harmful_incident | outcome1 to outcome8, outcome11 to outcome13 | 0 if no identified incident; otherwise 1 if any listed outcome is 1 |
| y4_identified_impactful_incident | impact items | used in one robustness check; "none" counts as 0, "don't know" only is missing |
| y1_nonphishing_incident | type_comb2 | used in the phishing-specificity check |
| size_cat, sector_cat | sizeb, sector_comb2 | validated category codes |

The harmful-incident indicator reproduces the official `outcome_any` variable in the 2025 release exactly. `scripts/validate_harm_indicator.py` checks this and compares both waves with published figures.
