# Temporal architecture screen

New CUDA architecture/objective runs; known TRAIN/VALIDATION development data.
TRAIN256 examples/128 pairs and VALIDATION128/64 pairs per run. B8,512 updates,
native32; fixed Ridge1 and tanh16/Adam .01/100, three head repetitions.
No PCA after either encoder. Existing baselines reuse saved metadata.

**RPB-v14** adds generic signed/symmetric cross-channel temporal products before
the learned32-number bottleneck. **RPB-v15** keeps the original input path and
asks native32 to predict label-free temporal relations during training.

Untrained scores measure the built-in prior; they are not evidence of learning.

## Two-cohort screen

Masters: 75272, 76373.

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.95 | 52.60 | 100 |
| PCA only — no encoder | 32 | 47.66 | 51.43 | 100 |
| RPB-v10.alt-05 | 32 | 50.00 | 53.26 | 100 |
| RPB-v14 | 32 | 96.88 | 95.05 | 100 |
| RPB-v15 | 32 | 52.34 | 52.34 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v14 | 32 | 100.00 | 100.00 | 100 |
| RPB-v15 | 32 | 53.52 | 50.78 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.22 | 51.43 | 100 |
| PCA only — no encoder | 32 | 48.05 | 51.30 | 100 |
| RPB-v10.alt-05 | 32 | 54.30 | 54.56 | 100 |
| RPB-v14 | 32 | 78.12 | 78.52 | 100 |
| RPB-v15 | 32 | 52.73 | 52.47 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v14 | 32 | 81.64 | 89.71 | 100 |
| RPB-v15 | 32 | 48.05 | 50.52 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v14 |512| 0.682767 | 0.728000 | 26.17 |
| RPB-v15 |512| 0.765474 | 0.786182 | 18.23 |

## Five known cohorts

Masters: 75272, 76373, 77474, 78575, 79676.

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.25 | 50.73 | 100 |
| PCA only — no encoder | 32 | 48.44 | 51.15 | 100 |
| RPB-v10.alt-05 | 32 | 52.81 | 54.90 | 100 |
| RPB-v14 | 32 | 97.03 | 95.10 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · intact VALIDATION · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v14 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · trained512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.53 | 50.78 | 100 |
| PCA only — no encoder | 32 | 48.75 | 50.68 | 100 |
| RPB-v10.alt-05 | 32 | 53.75 | 54.32 | 100 |
| RPB-v14 | 32 | 72.97 | 75.21 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · VALIDATION with extra30% deletion · untrained0.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v14 | 32 | 80.31 | 82.71 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable delay, gains/offsets and three-tick channel gaps · fixed masked TRAIN/intact VALIDATION queries, standardized MAE.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v14 |512| 0.685666 | 0.725161 | 24.82 |

The screen gate is75% in both fixed heads and both views on each of the
two starting cohorts, with100% coverage. Retain all failures and initial
scores. Additional known cohorts are development checks after screening,
not fresh-data confirmation. No automatic promotion or reference replacement.

- RPB-v14: passes the predeclared continuation gate.
- RPB-v15: fails the predeclared continuation gate.

RPB-v14 demonstrates a useful architectural prior on this structured timing
challenge, rather than a learned improvement: untrained intact scores are100%,
while the trained five-cohort means are97.03%/95.10%. Additional deletion is
harder, and training lowers the initial80.31%/82.71% to72.97%/75.21%.
The remaining three cohorts were inspected only after the two-cohort gate;
these five-known-cohort numbers are development evidence, not unseen confirmation.
Waveform MAE is also worse than the reused early-v10 reference's0.644282/0.682958.
The waveform objective can reduce timing utility even when useful relations
are explicitly available. Preserve this version as a measured candidate.

RPB-v15 remains near52% on its two-cohort screen. Stop its expansion under this
card; no coefficient, classifier or budget rescue. RPB-v16 is a separately
frozen follow-up with dedicated20 shape+12 grouped time-odd coordinates; its
[separate card](../../evaluation/cards/partitioned_relation_screen_v1.md)
records the allocation and risks before quality training.

Seven new CUDA trajectories,3,584 updates total, were run once: five v14 and
two v15. Retained0/512 points yield42 native exports,42 fixed-head pipelines
(84 heads) and14 original-query writers/56 necessary masked forwards. No old
encoder reruns, baseline/head refits, PCA fits, CPU encoder execution, TEST or
stress access. GPU training seconds are synchronized training-loop wall time
including minibatch preparation/transfers, excluding readout fitting, native
export, masked-query export and artifact writing; these are descriptive timings.

Actual CUDA primitive and serialized-loader admission passed. The initial
v15 test compile failed because it accessed a private backbone member; its
repaired admission is retained separately. The first saved-only checker assumed
Python Torch, which this container does not provide. The completed v14 quality
run was preserved and checked by the additive standard-library reader, using
the already verified archive/math modules, without model reruns or installs.
All saved checks passed:2,331,315 checks/72 CPU archives for each two-cohort
run and3,496,782/108 for v14's three remaining cohorts. Model/optimizer CUDA
archives are not loaded by this reader. Failed attempts remain on disk.
