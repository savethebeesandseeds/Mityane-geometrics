# RPB-v7 decoder calibration diagnostic

Date: 2026-10-08. Measured; independent arithmetic audit passed.
Protocol: `v7-decoder-calibration-v1`; policy 1.2; known development cohorts.

The fixed 128-update decoder-only recovery question **passed its numerical guards**. The encoder, buffers, original TRAIN scaler and native 32-number serving output remain exact. This is a reconstruction sufficiency diagnosis; it creates no improved embedding, new model tag, selection or promotion.

Five retained RPB-v7 parents use masters 4404/5505/6606/7707/8808 on timing (`lag_sign`). Each has 256 TRAIN examples from 128 source pairs and 128 known VALIDATION examples from 64 pairs. Opposite-label variants share a source. There are five new decoder trajectories, ten saved stages and no new encoder runs. Original encoder training remains 512 updates at batch 8 on CUDA (NVIDIA RTX A2000, 8 GB). Natural missingness is 0.10, C3/H32/F3, patch 8, width 64, three temporal blocks, four heads, feed-forward 256, aligned mixer 1, global mode 2, native32 and decoder width 128.

## Cached intact timing quality on known VALIDATION

**No new classification experiment. Every quality score in this table is historical and cached.** RPB-v4 is the ordinary learned global32 bottleneck. RPB-v7 uses the same inference architecture with 0.15 extra training-context deletion. Decoder calibration changes neither serving code nor encoder outputs; the saved exact invariants justify retaining its scores. No classifier or normalization fit is rerun here, and no PCA follows an encoder.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.69 | 50.83 | 100 |
| PCA only — no encoder | 32 | 49.06 | 70.99 | 100 |
| RPB-v4 | 32 | 93.59 | 95.99 | 100 |
| RPB-v7 | 32 | 97.97 | 98.96 | 100 |

Native cached scores average all three fixed heads per master, then the five masters equally. The linear recipe is ridge penalty 1; the neural recipe is 16 tanh units, Adam 0.01 and 100 updates, with repetitions 2701/2802/2903 and equal-width paired seeds. Native32/PCA32 have 66 linear and 562 neural parameters; Raw576 has 1,154 and 9,266. Raw/PCA numbers are reused from published summary prose; native scores come from the permitted old point/validation JSONs. These repetitions are not independently trained encoders.

## Frozen representation, fresh decoder optimization

All 225,805 model parameters remain on CUDA. Exactly 214,277 non-decoder parameters and every buffer are frozen. Only six decoder tensors, totalling 11,528 parameters in the position/channel embeddings and two decoder linear layers, receive updates. They start from the saved RPB-v7 values with a fresh decoder-only AdamW optimizer: learning rate 0.001, weight decay 0.0001, clipping 1, threads 1, dropout 0 and exactly 128 unskipped updates. Absolute attempts 512–639 retain the original row, artificial-mask and Torch streams. Extra context deletion is absent in this diagnosis. The encoder runs in evaluation/no-gradient mode and its exact native32 is detached before decode.

The original observed support O, artificial whole-patch mask A, target Q = O & A, eligibility and hierarchical Huber delta 1 remain fixed. Reconstruction uses the unchanged four patch-query banks, TRAIN-fitted standardized targets and equal cell → channel → example reductions. All TRAIN/VALIDATION examples and requested observed targets retain full coverage at both stages.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 512 encoder | 0.067593 | 0.071309 | 13.85 retained |
| RPB-v7 | 512 encoder | 0.081179 | 0.083630 | 15.69 retained |
| RPB-v7 | 512 encoder + 128 decoder | 0.054039 | 0.056989 | 15.69 retained + 2.09 new decoder |

Errors are standardized fixed-query MAE, not the last minibatch optimization loss. Original encoder seconds are retained historical means; new decoder seconds are a synchronized update-loop wall scope including CPU evidence capture, excluding parent load, query/export, scoring and audit. They are not kernel-only GPU seconds and are not an equal-compute benchmark. Each new decoder samples 1,024 row exposures with replacement, four equivalent presentations of the 256 TRAIN examples, not four guaranteed epochs. Typed calibration assets compose with an immutable parent and cannot masquerade as ordinary encoder resume.

## Every master and fixed recovery guards

| Master | v4 TRAIN / VALIDATION MAE | v7 before TRAIN / VALIDATION MAE | v7 after TRAIN / VALIDATION MAE | After − v4 TRAIN / VALIDATION | Decoder loop seconds | CUDA / invariant guard |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4404 | 0.072697 / 0.075390 | 0.076529 / 0.078533 | 0.054265 / 0.055833 | -0.018432 / -0.019558 | 1.97 | Passed |
| 5505 | 0.059514 / 0.064164 | 0.079698 / 0.083926 | 0.054353 / 0.059321 | -0.005160 / -0.004843 | 2.11 | Passed |
| 6606 | 0.062540 / 0.065255 | 0.087755 / 0.090371 | 0.054549 / 0.057135 | -0.007991 / -0.008120 | 2.08 | Passed |
| 7707 | 0.080216 / 0.085018 | 0.082668 / 0.085783 | 0.053683 / 0.057841 | -0.026533 / -0.027176 | 2.00 | Passed |
| 8808 | 0.063000 / 0.066718 | 0.079243 / 0.079538 | 0.053342 / 0.054817 | -0.009658 / -0.011902 | 2.29 | Passed |

The frozen recovery rule requires mean TRAIN MAE ≤ 0.0675931436195981 and mean VALIDATION MAE ≤ 0.07130906590008225, plus equal full support, preserved inputs/sources, unchanged encoder and actual finite CUDA decoder updates. Per-master differences are descriptive; the card fixes the aggregate thresholds and does not create a best-master or budget-selection rule.

| Frozen guard | Result |
| --- | --- |
| Mean TRAIN MAE at or below v4 | Passed |
| Mean VALIDATION MAE at or below v4 | Passed |
| Equal full support | Passed |
| Encoder / buffers / scaler / native exact | Passed |
| Finite CUDA decoder-only updates | Passed |

All five masters improve both errors relative to their uncalibrated v7 parent and paired v4 reference. This pass establishes reconstruction sufficiency under the declared decoder route and additional optimization budget; it does not establish improved classification or generalization. The earlier reconstruction deficit therefore does not by itself establish loss of information in the frozen representation. No extra decoder budget, rate, capacity, head or seed is selected to rescue it. No TEST/stress payload was accessed and the earlier fixed512 version dispositions remain unchanged.

## Evidence and audit scope

The [frozen card](../../../code/evaluation/cards/v7_decoder_calibration_v1.md) binds 68 v7 and 17 v4 input roles, 85 total. All role/path associations are admitted before the first payload hash; allowed bytes are tied to parent inventories and checked before/after. Stage 0 has exact same-CUDA saved v7 query/support parity. Both stages preserve the original targets/support and the paired v4 query geometry.

| Identity | SHA256 / value |
| --- | --- |
| Frozen card | `5ae05a3b5ec253d1743842f82fc28b45e0f20c8c84da1b58bf4ab9c514d292e9` |
| Calibration orchestration source | `fb5aef0d015b9ca373bc7a52a24ada6cb86bde3f9ac0e451d7a419c684d8de4d` |
| Admission passed record | `cdee6112143bf4dc26d80239cc88f2555f1cf4e5e6a68bd4fb21768b34f20852` |
| Admission log | `1166d23bd5894835116b31fb6edef1028eec7047008bdb00e5bebcf1b69bec8b` |
| Measured report | `3fdb8cbe48a7b0026d906f776745b68f607368f02c830d80cfcc55ce06c2c3c3` |
| Capsule inventory | `51400954bc2e9cae483de2e2a784e83f93482b02a888a3bdf95581d207967442` |
| Capsule files / bytes, excluding inventory | 227 / 130,624,183 |
| Independent reader | `bf85b364ad9b41d4d6abc23756dd2d0b216ba877d71377b99a786ce37a0a6628` |
| Independent audit status | passed |
| Passed validation JSON | `dfc8d96b0f1657d22c7e2f8dbd6ae53637f62f3d627d3763d4f5f47ef2808cea` |
| Independent checks | 25,450,287 |

The [new machine report](../../../output/runs/rpb-v7-decoder-calibration/decoder-calibration-jXsW20/results/report.json) retains every master and stage. The [capsule inventory](../../../output/runs/rpb-v7-decoder-calibration/decoder-calibration-jXsW20/artifact-integrity.json) pins original measurement/source/admission evidence. The tracked [durable summary JSON](../../../doc/results/v7_decoder_calibration_v1.json) records exact metadata identities, cached quality lineage and all guard calculations.

The [independent arithmetic audit](../../../output/runs/rpb-decoder-calibration/audit-tools/run-jXsW20-v2/validation.json) passed 25,450,287 checks with 115 CPU archive decodes across the closed 85-role inputs and ten new points. Its CPU reader checks saved reconstruction reductions, masks, targets, scaler identities, row counters, state/optimizer witnesses and input/source bindings. It does not execute CUDA encoding, autodiff, fresh AdamW or classifiers; those mechanisms are constrained by the preserved production source and actual-CUDA admission. Original std::shuffle choices are structurally/source-bound. No across-master confidence interval or new classification uncertainty is inferred from this diagnosis.

The first admission [rZA1Ma](../../../output/runs/rpb-v7-decoder-calibration/admission/admission-rZA1Ma/build-and-tests.log) failed compilation before any quality payload was opened. Its log and source evidence remain preserved. Two const-reference fixes repaired that compile-only failure without a model, card or numerical change; the source-bound actual CUDA admission then passed before measurement.

The [preserved source-only replay bundle](../../../output/runs/rpb-decoder-calibration/audit-tools/replay-bundle-jXsW20-20261008T094909Z-f32057cd/README.md) retains 71 captured producer sources, the exact sealed reader, its sole CPU archive-reader source, the unchanged passed validation and explicit bindings. It has no tensor/model copies and performed no imports, fixture or audit reruns. Its original data capsules and pinned helper path remain dependencies.

| Replay identity | SHA256 / value |
| --- | --- |
| Bundle inventory | `20279aa73d6c6bad15a24457ba61100c06a36f66f6123bb67ff8150aa4919fd4` |
| Source manifest | `464010d177eddfa6d8eaadcff7405b71fec70337fa8a7bb091a74d9d4e804471` |
| Audit binding | `1e17cecd5c7bac054d65e3c4ccc8653687bd5c13956f8153c46c7e9409621ddb` |
| Bundle files / bytes, excluding inventory | 90 / 1,307,790 |
