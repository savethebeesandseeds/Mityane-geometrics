# Content-only ordered relation development evidence

New RPB-v22.alt-01 quality training is complete. The prospective two-cohort screen passes 54/54 conditions; the separately allocated third known cohort passes 36/36. All three trajectories and every declared readout are retained. This is development evidence, with no TEST, market result, historical replacement or automatic serving-default promotion.

## What changed

RPB-v22.alt-01 learns from the ordered content moments introduced in RPB-v21, while removing the separate additive support-fraction inputs. Masks still choose valid endpoints and normalize covariance. The new 20 × 1944 zero-start projection adds 38,880 parameters; the RPB-v20 backbone, decoder, native32 size and final 12 fixed spectral coordinates are preserved. The only objective remains masked waveform Huber reconstruction. Timing labels never enter its optimizer.

The [prospective card](../code/evaluation/cards/configurable_content_relation_v1.md) preserves the exact saved TEMPO-5 observations, TRAIN scaler rule, row/mask/RNG streams, 512 updates and fixed heads. [The protocol](../code/evaluation/protocols/configurable_content_relation_v1/README.md) records CUDA admission, typed checkpoint reload, initialized byte parity, exact fixed coordinates and independent saved-array publication.

RPB-v20.alt-01 is the matched saved compact spectral encoder. RPB-v22.alt-01 is the separate content-only relation design. Size 32 below means the complete export; size 20 means its first 20 learned coordinates, a diagnostic subset.

## Equal means across all three completed cohorts

These are descriptive equal means of masters 941101/941102/941103, retaining all three head repetitions within each cohort. They do not replace the separately frozen screen and third-stage decisions. RPB-v20 measurements are reused saved controls; no saved v20 quality instance or control classifier was rerun or refitted to obtain these scores.

**TEMPO-5 · complexity 5/5:** slow lead/lag beneath fast rhythms, nine grouped features and structured distractors. Intact VALIDATION; all three known cohorts, 512 encoder updates, batch size 8 and three fixed-head repetitions.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 94.66 | 94.31 | 100.00 |
| RPB-v22.alt-01 | 32 | 97.14 | 95.96 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.74 | 50.17 | 100.00 |
| RPB-v22.alt-01 | 20 | 63.54 | 62.98 | 100.00 |
| Raw data — no encoder | 6912 | 49.61 | 50.69 | 100.00 |

The raw row reuses the original baseline scores; it is not a new fit or a new baseline arithmetic measurement.

**TEMPO-5 · complexity 5/5:** slow lead/lag beneath fast rhythms, nine grouped features and structured distractors. VALIDATION under additional 30% coordinate erasure; all three known cohorts, 512 encoder updates, batch size 8 and three fixed-head repetitions.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 90.10 | 88.72 | 100.00 |
| RPB-v22.alt-01 | 32 | 91.28 | 91.06 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.61 | 49.91 | 100.00 |
| RPB-v22.alt-01 | 20 | 63.02 | 63.06 | 100.00 |
| Raw data — no encoder | 6912 | 49.22 | 50.35 | 100.00 |

The raw row reuses the original baseline scores; it is not a new fit or a new baseline arithmetic measurement.

Initialized learned20 controls average 50.13%/49.78% intact and 50.91%/50.69% deleted. Training adds 13.41/13.19 percentage points intact and 12.11/12.37 deleted under the unchanged heads. Exact initialized candidate values and validity justified initialized-head reuse.

The complete encoder stays much stronger than the learned subset. This is useful learned-coordinate progress, while the preserved spectral prior remains a substantial part of the complete representation. Different widths change head capacity and initialization, so complete32 versus learned20 is a sufficiency diagnostic rather than an exact causal contribution. Same-width initialized/trained learned20 is the cleaner training comparison.

**TEMPO-5 · complexity 5/5:** original hidden-patch queries under fixed TRAIN scales, all three known cohorts; lower hierarchical standardized MAE is better. GPU seconds are synchronized training-loop means per trajectory.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 512 | 0.78245 | 0.79860 | 20.8 |
| RPB-v22.alt-01 | 0 | 0.82233 | 0.82499 | 0.0 |
| RPB-v22.alt-01 | 512 | 0.78334 | 0.80402 | 24.7 |

Candidate/reference MAE ratios are 1.001129 TRAIN and 1.006788 VALIDATION. Reconstruction is slightly less accurate than the saved trained v20 control, within the declared 1.02 mean/1.05 per-cohort protection margins. The initial 0-update row has no training-loop work; its CUDA extraction/query work is reported separately.

## Screen, then third development check

The first two known cohorts passed the unchanged joint 54-condition gate before master 941103 was allocated. The third cohort then passed its own 36 conditions. It was already known from earlier development, so it is not fresh unseen confirmation. No best checkpoint, source, repetition or budget was selected.

### First two cohorts: 941101/941102

Gate: **PASS**, 54/54 conditions.

**TEMPO-5 · complexity 5/5:** structured slow timing beneath fast patterns and distractors; Intact VALIDATION, masters 941101/941102, equal means across the fixed head repetitions.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 95.12 | 94.92 | 100.00 |
| RPB-v22.alt-01 | 32 | 97.46 | 96.55 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.80 | 50.26 | 100.00 |
| RPB-v22.alt-01 | 20 | 63.28 | 61.52 | 100.00 |

**TEMPO-5 · complexity 5/5:** structured slow timing beneath fast patterns and distractors; additional 30% erasure VALIDATION, masters 941101/941102, equal means across the fixed head repetitions.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 88.67 | 87.89 | 100.00 |
| RPB-v22.alt-01 | 32 | 90.04 | 89.71 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.80 | 49.87 | 100.00 |
| RPB-v22.alt-01 | 20 | 61.91 | 62.17 | 100.00 |

### Third cohort: 941103

Gate: **PASS**, 36/36 conditions.

**TEMPO-5 · complexity 5/5:** structured slow timing beneath fast patterns and distractors; Intact VALIDATION, masters 941103, equal means across the fixed head repetitions.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 93.75 | 93.10 | 100.00 |
| RPB-v22.alt-01 | 32 | 96.48 | 94.79 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.61 | 50.00 | 100.00 |
| RPB-v22.alt-01 | 20 | 64.06 | 65.89 | 100.00 |

**TEMPO-5 · complexity 5/5:** structured slow timing beneath fast patterns and distractors; additional 30% erasure VALIDATION, masters 941103, equal means across the fixed head repetitions.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 32 | 92.97 | 90.36 | 100.00 |
| RPB-v22.alt-01 | 32 | 93.75 | 93.75 | 100.00 |
| RPB-v20.alt-01 | 20 | 49.22 | 50.00 | 100.00 |
| RPB-v22.alt-01 | 20 | 65.23 | 64.84 | 100.00 |

On the matched first two cohorts, complete32 deletion scores improve from the preserved RPB-v21.alt-01 65.43%/78.06% to RPB-v22.alt-01 90.04%/89.71%. This comparison reuses saved v21 results and includes two cohorts on both sides. Comparing that two-cohort v21 mean directly with the three-cohort v22 mean would mix populations. The result supports this architectural change on known TEMPO-5 evidence; it does not prove that support inputs alone caused every earlier failure.

Some individual learned20 scores remain below 60%: the screen threshold applies to the declared two-cohort means. All per-cohort/repetition decisions and all gate inequalities remain in the exact saved stage summaries.

## Budgets, masks and costs

Each cohort has 512 TRAIN rows/256 paired source groups and 256 VALIDATION rows/128 disjoint groups. Geometry is C3/H128/F9/P8/W64/D32. Each new encoder trains once on CUDA:0 float32 with B8 and 512 completed AdamW updates (learning rate 0.001, weight decay 0.0001, clipping 1), patch masking 25% and original context erasure 15%. This is 8 equivalent row presentations, not 8 complete shuffled epochs.

Heads remain ridge penalty 1 and tanh 16 / Adam 0.01/100 updates, repetitions 2701/2802/2903. Complete32 heads have 66/562 parameters; learned20 heads have 42/370. Their weights and TRAIN normalizers are fitted separately. Head repetitions are not encoder retrainings; deterministic ridge repetitions are not independent uncertainty sources. No post-encoder PCA, raw/PCA fitting, saved v21 encoder execution, saved quality encoder replay or control refit occurred.

Reconstruction uses the same 16 leave-one-patch-out contexts per example at 0/512. All query/readout coverage is 100%. Per-view features and original query arrays are captured on CUDA; CPU performs the declared classifiers and saved arithmetic. The following costs are descriptive, including transfer/output where named, rather than a controlled throughput benchmark.

**TEMPO-5, complexity 5/5:** slow timing beneath fast rhythms, nine features and
structured distractors; descriptive costs for the three known development runs.

| Cohort | GPU training s | Features + transfer/I/O s | Queries + transfer/I/O s | Checkpoint/binding s | CPU heads + readout I/O s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 941101 | 24.632 | 7.480 | 85.750 | 1.514 | 9.475 |
| 941102 | 24.908 | 7.181 | 83.157 | 2.701 | 7.465 |
| 941103 | 24.703 | 9.210 | 101.985 | 2.693 | 6.532 |

Total synchronized candidate training: 74.243 seconds across three trajectories. Classifier/readout I/O time is separate from independent publisher checks; neither runs an encoder on CPU.

## Evidence and interpretation

The completed capsule is `output/runs/rpb-v22-content-relation/admission-53b82q`, disposition `third-development-pass`. Artificial GPU engineering and negative admission/source/card/early-third guards passed before quality. Actual new-projection gradients and weight changes are retained for every trajectory, along with typed state reload and frozen prior checks.

Across the stages, the independent publishers checked 19,273,393 conditions, 18 new fitted head pairs/36 classifiers, 45 reused fitted pairs, 54 candidate prediction archives and 72 paired comparisons. Initial equality covers 98,304 values; fixed-coordinate equality covers 73,728 values at both retained points, with validity checked separately.

Original capacity, branch and completed v21 artifact inventories passed before/after preservation checks. The [durable capture](results/configurable_content_relation_v1/completed/) keeps 233 exact nonbinary files and its own SHA256 manifest. Tensor archives/checkpoints and compiled binaries remain in ignored local output; none were deleted, overwritten or relabelled.

Source fingerprint: `a9e625650b4d40b2658211d1f3f8f5b06a17d71b79ec4f22f491f02460198519`. Card SHA256: `53b0bcc2dc1b30f0b8060448d151def9a4305ba89f22039c5ad79eaca5d5d349`. Complete artifact inventory SHA256: `d0f1c6c014a4d348fb98316205815eefc59e0a3265ad0b459f04044102f6af41`.

The [joint saved record](results/configurable_content_relation_v1.json) preserves both exact stage summary objects and their original file hashes, descriptive combined means, budgets and completion. The [instance registry](configurable_content_relation_embedding_versions.json) binds each 0/512 checkpoint hash to its cohort, TRAIN scaler identity, resolved settings, source/card/config and stage disposition.

The separate [engineering audit](results/configurable_content_relation_v1/engineering-audit.json)
reconciles the actual CUDA pilot. The [independent quality audit](results/configurable_content_relation_v1/quality-audit.json)
checks 25,961 conditions and reproduces all 90 prospective numeric gates without
running models or fitting heads. The actual CLI rejection probes are retained
with their [script](results/configurable_content_relation_v1/prequality/negative-guards.py)
and exact logs in the durable capsule capture.

Conditional source-group intervals measure uncertainty under each saved fitted head and cohort. They are not retraining uncertainty, causal decoder attribution or proof of market generalization. The result establishes useful learned timing coordinates and protected complete-embedding quality on this known synthetic task. Reconstruction remains imperfect, and arbitrary masks/history lengths, fresh data and real markets remain separate work.

## Repository boundary and continuation

The portable v22 API belongs to the encoder; this runner/data challenge/evaluation
belongs to cuwacunu_embedding. The optional v22 API is now mirrored into
[cuwacunu](../../cuwacunu_dev/code/encoders/rpb_content/README.md), with
[six exact file pairs](../../cuwacunu_dev/doc/CONTENT_ENCODER_SYNC.json) and a passed
focused CUDA input/mask/checkpoint test. Selected v20 settings remain preserved;
these development results do not select a market checkpoint or promote a default.
The original test succeeded; a shell-wrapper footer reported a carriage-return
error afterward. Its separate verification receipt and exact durable capture
preserve both outcomes, with no duplicate CUDA run.

Next: preserve this learned-coordinate and deletion-robustness gain, then use a separately frozen fresh-data confirmation before claiming broader quality. Keep unchanged heads and matched v20 controls. A future architecture or objective change receives its own tag/card/output rather than rescuing or overwriting this completed recipe.
