# TEMPO-3 data-only information admission

The first structured harder timing information admission failed its declared deleted-view coverage gate. No encoder or classifier head was fitted, and no encoder quality comparison was generated.

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains/offsets and 3-tick channel gaps · views: intact and extra 30% coordinate deletion.

| Engineering master | View | Valid / total | Analytic conditional accuracy % | Coverage % | Full-population correctness % | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 80383 | intact | 1024/1024 | 100.00 | 100.0000 | 100.0000 | PASS |
| 80383 | deleted | 950/1024 | 100.00 | 92.7734 | 92.7734 | FAIL |
| 81484 | intact | 1024/1024 | 100.00 | 100.0000 | 100.0000 | PASS |
| 81484 | deleted | 900/1024 | 100.00 | 87.8906 | 87.8906 | FAIL |

Each engineering cohort contains 512 TRAIN and 512 separate VALIDATION source pairs (1,024 rows per split). Only the 1,024 VALIDATION rows per view above determine this information gate. The observed-only analytic rule has zero fitted heads. Its labels are used for scoring only; they are not rule inputs.

The frozen thresholds are intact accuracy/coverage ≥99%/99% and deleted accuracy/coverage ≥98%/95%, independently for both masters. Every supported row was correct. Deleted views retained 950 and 900 rows, below the required 973 of 1,024; their conditional 100% does not overcome insufficient coverage. Unsupported rows remain abstentions rather than being discarded from the population.

This failure describes the declared six-observation, same-feature analytic support rule. It does not establish that every unsupported row lacks timing information, and there are no model results to compare. The C3/H32/F3 signal/missingness recipe, prospective card, compiler log, binary and failed record remain preserved. The ordinal complexity label is unchanged.

A possible next data-only specification is to use independently legal three-time triples within each channel and average all cross-channel feature combinations. Positive gains and constant offsets preserve the noiseless determinant sign. That proposal requires a separately specified diagnostic/card and fresh admission; it is not implemented or measured by this report, and it does not lower the existing support/coverage gates.

Evidence: [failed information record](../output/runs/rpb-structured-hard-timing/information-admission/info-M7iGiK/information-record.json), [frozen prospective card](../code/evaluation/cards/structured_hard_timing_comparison_v1.md), and [durable metadata](results/structured_hard_timing_information_v1.json). The information record SHA-256 is `404ec6e0ef2134bf5a99c483358fa2c8f1c95b5f2ae2bb4237764fa7ac45a1e6`. Its source closure is `10a63b2437d9ffd9ae733df95b8741660eccf5577d512621d583b1014a9498d3`; card SHA-256 is `022c7257a3983e8c5230a787e6129963d4e9967333523ea6686f5adb70da1a11`. Source and receipt bytes were checked before and after this metadata-only reporting. No tensor archive, model, head or audit rerun occurred during report creation.
