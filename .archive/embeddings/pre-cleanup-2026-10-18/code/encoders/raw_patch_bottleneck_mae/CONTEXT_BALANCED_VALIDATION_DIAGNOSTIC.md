# RPB-v8 balanced context validation diagnostic

Date: 2026-10-07. Measured and independently audited; not promoted.
Protocol: `context-balanced-validation-v1`; policy 1.2; known development only.

The balanced mixture fails the frozen joint development question. Mean native linear timing accuracy falls relative to v7 in both known VALIDATION views, while mean original TRAIN/VALIDATION reconstruction errors are higher than both v4 and v7. Intact worst-master linear accuracy ties v7; deletion worst-master accuracy declines. RPB-v4 remains active, and this result closes the local schedule/rate/budget tuning sequence.

Five known masters 4404/5505/6606/7707/8808 use timing (`lag_sign`), 128 TRAIN source pairs / 256 examples and 64 known VALIDATION pairs / 128 examples per master. Five new actual-CUDA encoder runs are compared with ten retained v4/v7 runs. Each has 512 updates, batch 8, 225,805 parameters and 16 equivalent presentations with replacement. Saved points and head repetitions are not independent encoder runs.

RPB-v4 has ordinary context. RPB-v7 uses 0.15 extra training deletion on every attempt. RPB-v8 keeps the same inference architecture and alternates unchanged ordinary attempts with the existing 0.30 policy: exactly 256 of each at absolute indices 0..511. Original queries, targets, support, scaler, objective and streams remain fixed. Geometry is C3/H32/F3, patch 8, mixer 1 and global mode 2. Every encoder serves native32 and decodes solely through it; no PCA follows an encoder.

Ridge penalty 1 and tanh16 / Adam 0.01 / 100 updates stay fixed. All three declared head repetitions are retained with width-paired seeds 2701/2802/2903. Size32 heads have 66 / 562 parameters; raw576 heads have 1,154 / 9,266. Means average repetitions within master, then masters equally.

Raw576 contains 288 observed TRAIN-scaled values plus 288 observation flags. Its per-channel/feature observed TRAIN scaler uses floor 1e-8. Standalone PCA32 is fitted on this raw TRAIN input; it is not applied to any encoder export. Observation scales and outer maps fit once per input; the two heads fit separately for each declared repetition. All fit only ordinary TRAIN and are reused for both validation views.

## Intact known VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.69 | 50.83 | 100.00 |
| PCA only — no encoder | 32 | 49.06 | 70.99 | 100.00 |
| RPB-v4 | 32 | 93.59 | 95.99 | 100.00 |
| RPB-v7 | 32 | 97.97 | 98.96 | 100.00 |
| RPB-v8 | 32 | 95.94 | 97.76 | 100.00 |

The deletion view is the exact cached v7 mask, byte-preserved for all tags and budgets under its original `context-lighter-validation-v1` namespace. No new corruption was drawn, repaired or used for fitting. Raw/PCA scores and fitted assets are identical across versions; their repeated copies are not independent data.

| Master | RPB-v4 linear / neural % | RPB-v7 linear / neural % | RPB-v8 linear / neural % | v8 − v7 linear, pp |
| --- | ---: | ---: | ---: | ---: |
| 4404 | 98.44 / 100.00 | 100.00 / 100.00 | 94.53 / 99.74 | -5.46875 |
| 5505 | 91.41 / 92.45 | 93.75 / 96.09 | 93.75 / 95.05 | +0.00000 |
| 6606 | 100.00 / 100.00 | 100.00 / 100.00 | 99.22 / 100.00 | -0.78125 |
| 7707 | 89.06 / 94.79 | 97.66 / 100.00 | 93.75 / 96.35 | -3.90625 |
| 8808 | 89.06 / 92.71 | 98.44 / 98.70 | 98.44 / 97.66 | +0.00000 |

RPB-v8 minus v7 mean effects are -2.03125000 pp linear and -1.19791667 pp neural. Worst-master linear accuracy is 93.75000% versus 93.75000%. Linear decreases occur on masters 4404, 6606, 7707.

## Fixed additional 30% deletion on known VALIDATION

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 52.03 | 48.65 | 100.00 |
| PCA only — no encoder | 32 | 49.84 | 54.79 | 100.00 |
| RPB-v4 | 32 | 84.38 | 91.46 | 100.00 |
| RPB-v7 | 32 | 95.16 | 96.25 | 100.00 |
| RPB-v8 | 32 | 91.72 | 94.22 | 100.00 |

The deletion view is the exact cached v7 mask, byte-preserved for all tags and budgets under its original `context-lighter-validation-v1` namespace. No new corruption was drawn, repaired or used for fitting. Raw/PCA scores and fitted assets are identical across versions; their repeated copies are not independent data.

| Master | RPB-v4 linear / neural % | RPB-v7 linear / neural % | RPB-v8 linear / neural % | v8 − v7 linear, pp |
| --- | ---: | ---: | ---: | ---: |
| 4404 | 68.75 / 94.01 | 98.44 / 100.00 | 88.28 / 94.27 | -10.15625 |
| 5505 | 89.06 / 89.58 | 94.53 / 93.75 | 92.19 / 91.67 | -2.34375 |
| 6606 | 97.66 / 99.48 | 98.44 / 97.92 | 96.09 / 100.00 | -2.34375 |
| 7707 | 86.72 / 89.58 | 92.97 / 100.00 | 87.50 / 92.19 | -5.46875 |
| 8808 | 79.69 / 84.64 | 91.41 / 89.58 | 94.53 / 92.97 | +3.12500 |

RPB-v8 minus v7 mean effects are -3.43750000 pp linear and -2.03125000 pp neural. Worst-master linear accuracy is 87.50000% versus 91.40625%. Linear decreases occur on masters 4404, 5505, 6606, 7707.

## Reconstruction and training

Error is standardized fixed-query MAE on original TRAIN and intact VALIDATION using the original four patch queries and equal cell → channel → example reductions. Classification deletion does not redefine reconstruction targets. v4/v7 timing is retained; v8 timing is new synchronized loop time, not a repeated cost or inference benchmark.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 512 | 0.067593 | 0.071309 | 13.85 |
| RPB-v7 | 512 | 0.081179 | 0.083630 | 15.69 |
| RPB-v8 | 512 | 0.085022 | 0.087620 | 14.79 |

| Master | v4 TRAIN / VAL MAE | v7 TRAIN / VAL MAE | v8 TRAIN / VAL MAE | v8 GPU seconds |
| --- | ---: | ---: | ---: | ---: |
| 4404 | 0.072697 / 0.075390 | 0.076529 / 0.078533 | 0.087894 / 0.088996 | 14.45 |
| 5505 | 0.059514 / 0.064164 | 0.079698 / 0.083926 | 0.079000 / 0.080151 | 14.74 |
| 6606 | 0.062540 / 0.065255 | 0.087755 / 0.090371 | 0.088950 / 0.093342 | 13.57 |
| 7707 | 0.080216 / 0.085018 | 0.082668 / 0.085783 | 0.092676 / 0.094648 | 15.53 |
| 8808 | 0.063000 / 0.066718 | 0.079243 / 0.079538 | 0.076587 / 0.080960 | 15.69 |

## Frozen decision and evidence

The joint question requires no lower mean and worst-master native ridge than v7 in both views, equal coverage, and mean TRAIN/VALIDATION MAE each no worse than paired v4. These are development guards, not consumer acceptance.

| Guard | Result |
| --- | --- |
| intact mean ridge no lower than v7 | Failed |
| intact worst master ridge no lower than v7 | Passed |
| validation-dropout-030 mean ridge no lower than v7 | Failed |
| validation-dropout-030 worst master ridge no lower than v7 | Failed |
| train MAE no worse than v4 | Failed |
| validation MAE no worse than v4 | Failed |

**Joint numeric guards: failed.** No TEST/stress, model selection, fresh confirmation or promotion occurred. Preserve this result and stop the local rate/schedule/budget tuning sequence rather than waive a failed guard or tune heads.

The [completed capsule](../../../output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV/) contains the [frozen card](../../../output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV/results/context-balanced-validation-card.json), [head scores](../../../output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV/results/readouts/report.json), [queries](../../../output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV/results/report.json), [all paired intervals](../../../output/runs/rpb-context-balanced-validation/balanced-validation-99UTEV/results/paired-validation-report.json) and both parity ledgers. Exactly 326 permitted parent roles were hash-bound and preserved. The positive matrix contains 45 TRAIN-fitted pipelines per method, 135 supported total, 45 native old/cache parity records and 60 comparison records / 120 head effects. Separate candidate development fitting has 30 native pipelines. All views reuse their ordinary TRAIN maps/heads; native assets contain no PCA.

The [independent audit](../../../output/runs/rpb-context-balanced-validation/audit-tools/prepared-v8-v2-20261007T044507Z-d4d0c369/measured-audit/validation.json) passed 67,099,959 checks. It replays TRAIN statistics/maps, probe inputs/logits/predictions, cached source-paired masks and query arithmetic, and binds frozen source/admission/inventories. It does not rerun encoder/head optimization or deserialize GPU models. Coordinated actual-CUDA admission supplies source/log-bound initialization, gradient, update and continuous-state evidence. Bootstrap draws/bounds are not regenerated independently; finite ordered bounds, source populations, recipes and point estimates are checked. Intervals are conditional within master, not uncertainty across five encoder runs, and no bounds are averaged.

| Identity | SHA256 |
| --- | --- |
| Orchestration source | `78ab6ffdb0a3edf141e0ad81e58fe10f36e276d4b7d8af1288859004b5f0f38c` |
| Training producer | `3532deae194828f2640ff4fadacbcd94bbccfb8af6eaf37182cb8137ae84cf41` |
| Core writer | `073ec84b7c781cf48a578e58a4c9602c6b765c9c23f6065f27338780be8a1835` |
| Passed admission | `87b3142227e86d43732b22ff6f7bcb3a87334f18f6180e15620ed42f6d5f1421` |
| Frozen human card | `b9455d7cfb3e1dc5a97abb147abaed3cfa85c72b49b2949b0f896a4c75f31c13` |
| Capsule inventory | `e0cdfbc729c8d989e16bb5e8515b5c5d73b7b445bdf91d2d938d88e09c969ac7` |
| Independent auditor | `e436a63f60b3215bc8cda6f545a500c4af93b5d60f3fe5cae78ec9b5e019da49` |
| Passed validation | `a78b231b0a17553030d058c9b26608b6b49d3a8447cc165ca5e8136685427ef0` |

The [durable numerical summary](../../../doc/results/context_balanced_validation_v1.json) retains exact scores, every master and head repetition, cost scopes, guards and evidence identities. Large archives remain local under `output/`.

The [preserved audit replay bundle](../../../output/runs/rpb-context-balanced-validation/audit-tools/replay-bundle-99UTEV-20261007T051527Z-262d74bf/REPLAY.md) contains the exact auditor/helper sources and unchanged passed validation record. It preserves source-only replay dependencies, not the data capsules.

No further candidate is implemented here. The evidence is confined to known synthetic timing cohorts; stop tuning this local schedule sequence and retain RPB-v4 as the active reference until a separately frozen broader generalization question is defined.
