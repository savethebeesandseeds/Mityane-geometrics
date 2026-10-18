# RPB-v9 native-view agreement validation diagnostic

Date: 2026-10-07. Measured and independently audited; not promoted.
Protocol: `native-view-agreement-validation-v1`; policy 1.2; known development only.

The native-view agreement candidate is a severe failed experiment: all six frozen numerical guards fail, native linear timing accuracy deteriorates, and original reconstruction becomes unstable across the five-run aggregate. All five runs are retained. The measurements establish failure without identifying a cause; no coefficient or budget rescue is attempted. RPB-v4 remains active; this known-VALIDATION result performs no selection, fresh confirmation or promotion.

Five known masters 4404/5505/6606/7707/8808 use timing (`lag_sign`), 128 TRAIN source pairs / 256 examples and 64 known VALIDATION pairs / 128 examples per master. Five new actual-CUDA encoder runs are compared with ten retained v4/v7 runs. Each has 512 updates, batch 8, 225,805 parameters and 16 equivalent presentations with replacement. Saved points and head repetitions are not independent encoder runs.

RPB-v4 reconstructs from ordinary context. RPB-v7 reconstructs from context with 0.15 extra training deletion on every attempt. RPB-v9 keeps ordinary-view reconstruction and adds a same-row native agreement term (weight 0.05) from a repaired 0.15-deleted student view to a detached ordinary view, plus student variance safeguard (weight 0.01, target 0.5, epsilon 1e-4). Point0 legal TRAIN CUDA native32 population scales are fitted once, with floor 1e-6, solely inside auxiliary losses. It uses no EMA, projector or new module. Original queries, targets, support, scaler and row/mask streams remain fixed. Geometry is C3/H32/F3, patch 8, mixer 1 and global mode 2. Every encoder serves native32 and decodes solely through it; no PCA follows an encoder.

Ridge penalty 1 and tanh16 / Adam 0.01 / 100 updates stay fixed. All three declared head repetitions are retained with width-paired seeds 2701/2802/2903. Size32 heads have 66 / 562 parameters; raw576 heads have 1,154 / 9,266. Means average repetitions within master, then masters equally.

Encoder training, point0 calibration encoding and fixed-query reconstruction run on CUDA. Frozen checkpoint feature extraction, head fitting/scoring and the independent arithmetic audit run on CPU. The GPU table covers the stated encoder update loop, not those CPU evaluation stages.

Raw576 contains 288 observed TRAIN-scaled values plus 288 observation flags. Its per-channel/feature observed TRAIN scaler uses floor 1e-8. Standalone PCA32 is fitted on this raw TRAIN input; it is not applied to any encoder export. Observation scales and outer maps fit once per input; the two heads fit separately for each declared repetition. All fit only ordinary TRAIN and are reused for both validation views.

## Intact known VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.69 | 50.83 | 100.00 |
| PCA only — no encoder | 32 | 49.06 | 70.99 | 100.00 |
| RPB-v4 | 32 | 93.59 | 95.99 | 100.00 |
| RPB-v7 | 32 | 97.97 | 98.96 | 100.00 |
| RPB-v9 | 32 | 59.22 | 58.12 | 100.00 |

The deletion view is the exact cached v7 mask, byte-preserved for all tags and budgets under its original `context-lighter-validation-v1` namespace. No new corruption was drawn, repaired or used for fitting. Raw/PCA scores and fitted assets are identical across versions; their repeated copies are not independent data.

| Master | RPB-v4 linear / neural % | RPB-v7 linear / neural % | RPB-v9 linear / neural % | v9 − v7 linear, pp |
| --- | ---: | ---: | ---: | ---: |
| 4404 | 98.44 / 100.00 | 100.00 / 100.00 | 57.03 / 50.00 | -42.96875 |
| 5505 | 91.41 / 92.45 | 93.75 / 96.09 | 71.88 / 65.10 | -21.87500 |
| 6606 | 100.00 / 100.00 | 100.00 / 100.00 | 48.44 / 46.35 | -51.56250 |
| 7707 | 89.06 / 94.79 | 97.66 / 100.00 | 68.75 / 62.24 | -28.90625 |
| 8808 | 89.06 / 92.71 | 98.44 / 98.70 | 50.00 / 66.93 | -48.43750 |

RPB-v9 minus v7 mean effects are -38.75000000 pp linear and -40.83333333 pp neural. Worst-master linear accuracy is 48.43750% versus 93.75000%. Linear decreases occur on masters 4404, 5505, 6606, 7707, 8808.

## Fixed additional 30% deletion on known VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 52.03 | 48.65 | 100.00 |
| PCA only — no encoder | 32 | 49.84 | 54.79 | 100.00 |
| RPB-v4 | 32 | 84.38 | 91.46 | 100.00 |
| RPB-v7 | 32 | 95.16 | 96.25 | 100.00 |
| RPB-v9 | 32 | 57.03 | 55.99 | 100.00 |

The deletion view is the exact cached v7 mask, byte-preserved for all tags and budgets under its original `context-lighter-validation-v1` namespace. No new corruption was drawn, repaired or used for fitting. Raw/PCA scores and fitted assets are identical across versions; their repeated copies are not independent data.

| Master | RPB-v4 linear / neural % | RPB-v7 linear / neural % | RPB-v9 linear / neural % | v9 − v7 linear, pp |
| --- | ---: | ---: | ---: | ---: |
| 4404 | 68.75 / 94.01 | 98.44 / 100.00 | 50.78 / 50.52 | -47.65625 |
| 5505 | 89.06 / 89.58 | 94.53 / 93.75 | 65.62 / 65.89 | -28.90625 |
| 6606 | 97.66 / 99.48 | 98.44 / 97.92 | 48.44 / 43.23 | -50.00000 |
| 7707 | 86.72 / 89.58 | 92.97 / 100.00 | 68.75 / 62.24 | -24.21875 |
| 8808 | 79.69 / 84.64 | 91.41 / 89.58 | 51.56 / 58.07 | -39.84375 |

RPB-v9 minus v7 mean effects are -38.12500000 pp linear and -40.26041667 pp neural. Worst-master linear accuracy is 48.43750% versus 91.40625%. Linear decreases occur on masters 4404, 5505, 6606, 7707, 8808.

## Reconstruction and training

Error is standardized fixed-query MAE on original TRAIN and intact VALIDATION using the original four patch queries and equal cell → channel → example reductions. Classification deletion does not redefine reconstruction targets. v4/v7 timing is retained; v9 timing is a new synchronized two-encode update loop including every-update CPU evidence capture. Point0 CUDA calibration is timed separately. Equal updates are not equal compute, and these measurements are not a kernel-only or inference benchmark.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 512 | 0.067593 | 0.071309 | 13.85 |
| RPB-v7 | 512 | 0.081179 | 0.083630 | 15.69 |
| RPB-v9 | 512 | 29.194490 | 29.371655 | 40.47 |

Mean v9 calibration time is 1.9954 seconds, outside the update-loop table. Calibration evaluates the frozen point0 encoder under original observed TRAIN support and computes float64 population standard deviations before saving float32 loss-only scales.


| Master | v4 TRAIN / VAL MAE | v7 TRAIN / VAL MAE | v9 TRAIN / VAL MAE | v9 GPU seconds |
| --- | ---: | ---: | ---: | ---: |
| 4404 | 0.072697 / 0.075390 | 0.076529 / 0.078533 | 142.226385 / 143.103198 | 39.68 |
| 5505 | 0.059514 / 0.064164 | 0.079698 / 0.083926 | 0.926542 / 0.924284 | 39.26 |
| 6606 | 0.062540 / 0.065255 | 0.087755 / 0.090371 | 0.998337 / 1.005158 | 39.20 |
| 7707 | 0.080216 / 0.085018 | 0.082668 / 0.085783 | 0.921267 / 0.923803 | 39.12 |
| 8808 | 0.063000 / 0.066718 | 0.079243 / 0.079538 | 0.899916 / 0.901835 | 45.09 |

The following components average all 512 sampled training updates, not the fixed-query MAE above. The total is ordinary Huber + 0.05 agreement + 0.01 variance; these are objective diagnostics, not new readout scores.

| Master | Ordinary Huber | Agreement | Variance | Total | Calibration seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4404 | 15.010216 | 221776427.207550 | 0.003538 | 11088836.467413 | 1.5035 |
| 5505 | 0.517389 | 4115.806210 | 0.000774 | 206.307713 | 2.0882 |
| 6606 | 0.627142 | 9128.489999 | 0.000650 | 457.051661 | 2.2725 |
| 7707 | 0.520022 | 3115.654139 | 0.000531 | 156.302737 | 1.8937 |
| 8808 | 0.576378 | 25741.518239 | 0.000938 | 1287.652323 | 2.2190 |

## Frozen decision and evidence

The joint question requires no lower mean and worst-master native ridge than v7 in both views, equal coverage, and mean TRAIN/VALIDATION MAE each no worse than paired v4. These are development guards, not consumer acceptance.

| Guard | Result |
| --- | --- |
| intact mean ridge no lower than v7 | Failed |
| intact worst master ridge no lower than v7 | Failed |
| validation-dropout-030 mean ridge no lower than v7 | Failed |
| validation-dropout-030 worst master ridge no lower than v7 | Failed |
| train MAE no worse than v4 | Failed |
| validation MAE no worse than v4 | Failed |

**Joint numeric guards: failed.** No TEST/stress, model selection, fresh confirmation or promotion occurred. A failed joint guard stops this mechanism; coefficients, rate, budget, heads or source subsets cannot rescue the frozen result.

The [completed capsule](../../../output/runs/rpb-native-view-agreement-validation/native-view-agreement-validation-glrBZE/) contains the [frozen card](../../../output/runs/rpb-native-view-agreement-validation/native-view-agreement-validation-glrBZE/results/native-view-agreement-validation-card.json), [head scores](../../../output/runs/rpb-native-view-agreement-validation/native-view-agreement-validation-glrBZE/results/readouts/report.json), [queries](../../../output/runs/rpb-native-view-agreement-validation/native-view-agreement-validation-glrBZE/results/report.json), [all paired intervals](../../../output/runs/rpb-native-view-agreement-validation/native-view-agreement-validation-glrBZE/results/paired-validation-report.json) and both parity ledgers. Exactly 326 permitted parent roles were hash-bound and preserved. The positive matrix contains 45 TRAIN-fitted pipelines per method, 135 supported total, 45 native old/cache parity records and 60 comparison records / 120 head effects. Separate candidate development fitting has 30 native pipelines. All views reuse their ordinary TRAIN maps/heads; native assets contain no PCA.

The [independent audit](../../../output/runs/rpb-native-view-agreement-validation/audit-tools/run-glrBZE-v2/validation.json) passed 74,721,050 checks. It replays TRAIN statistics/maps, probe inputs/logits/predictions, cached source-paired masks, fixed-query arithmetic, saved CUDA calibration statistics and every update's saved ordinary Huber, agreement, variance, support and weighted total. New saved float32 component replay uses prospectively declared 2e-6 absolute plus 2e-5 relative tolerance; historical head/query checks stay unchanged. Original whole-patch std::shuffle, CUDA encoding/autodiff, detachment, finite gradients and AdamW updates remain coordinated source/admission-bound, not CPU reexecuted. Bootstrap draws/bounds are not regenerated independently; finite ordered bounds, source populations, recipes and point estimates are checked. Intervals are conditional within master, not uncertainty across five encoder runs, and no bounds are averaged.

| Identity | SHA256 |
| --- | --- |
| Orchestration source | `3439df37b420c0b4f7ade3787f85befa77624b66a27904d446ed824315abda0e` |
| Training producer | `3439df37b420c0b4f7ade3787f85befa77624b66a27904d446ed824315abda0e` |
| Core writer | `e464f7d71bbf511457a3385dd796099a923808240d80a5e061950825f97f8d4d` |
| Passed admission | `d342f2c5d184948aecaae00201367fbe61e6a1b2149b9bac8ddfe652529770ba` |
| Frozen human card | `1f33eff0e6ca3e3241fd972fdc855900c535defa330f4eaec7effdf0c90d1c2e` |
| Capsule inventory | `5d8ae964a2d05fd68aaf2fb5c6173e5faaea35e68fbf4a3807b7f04afa0781dc` |
| Independent auditor | `8955026bdeb9361a3dd79adfe9b08255149e9f6f3fe2e671a230915138407435` |
| Passed validation | `c0a96603d9b5e38e619841a5027088a383ce76f8a376409b1dd63b71fe8946c5` |

The [durable numerical summary](../../../doc/results/native_view_agreement_validation_v1.json) retains exact scores, every master and head repetition, cost scopes, guards and evidence identities. Large archives remain local under `output/`.

The first coordinated admission failed compilation before CUDA or quality access because of a mixed-const pointer initializer list. Its failure and source inventory were preserved; the repair uses an explicit const-pointer array without numerical changes. Before measurement, peer review corrected the new auditor's stale v8 input filter and v7 output-path protection, then a reporting-only inherited v1 release-status field. The first announced seal is preserved; final seal changes only the two reviewed release flags. All older readers and result artifacts remain unchanged.

The [preserved audit replay bundle](../../../output/runs/rpb-native-view-agreement-validation/audit-tools/replay-bundle-glrBZE-20261007T081453Z-7b0a1fdf/README.md) contains the exact auditor/helper sources and unchanged passed validation record. It preserves source-only replay dependencies, not the data capsules.

No further candidate is implemented here. The evidence is confined to known synthetic timing cohorts. A failed joint guard cannot be rescued by changing coefficients, rate, budget, head repetition or source subset; even a pass supplies no automatic TEST access or promotion.
