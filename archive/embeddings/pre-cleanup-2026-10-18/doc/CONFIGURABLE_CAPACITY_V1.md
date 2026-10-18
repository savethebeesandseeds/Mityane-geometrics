# Configurable encoder capacity: TEMPO-5

New synthetic development experiment, completed12 October2026. The configurable
RPB-v20 runner reads the existing core file/settings module and saves resolved
geometry, CUDA checkpoints, native exports, reconstruction queries and provenance.
Evaluation consumes the saved exports through the existing shared fixed heads;
it does not import the encoder or market loader. No PCA follows an encoder.

Per run: TRAIN512 rows from256 independent source pairs, VALIDATION256 rows
from128 independent source pairs. Three fresh dataset masters941101/941102/941103
are shared across profiles; each profile trains three separate encoders, giving
nine trajectories and three independent dataset cohorts. The deleted view reuses
the same validation sources. Each encoder receives512 completed AdamW updates,
batch8, on an NVIDIA RTX A2000 8GB Laptop GPU, CUDA0/float32. No skipped updates.
4096 sampled training rows are eight equivalent presentations, not eight complete epochs.

Protocol [configurable-capacity-v1](../code/evaluation/cards/configurable_capacity_v1.md)
was fixed before quality generation. Linear head: ridge penalty1. Neural head:
16 tanh hidden units, Adam.01,100 updates. Three fixed repetitions2701/2802/2903
are averaged within each cohort, then the three cohort means are averaged.
Each head is fitted once on TRAIN and reused for both validation views. At size32,
the linear/neural heads have66/562 fitted parameters; at64,130/1074; raw
6912 inputs use13826/110642. Parenthesized ranges below are cohort-mean ranges,
not confidence intervals. Conditional source-group intervals and paired
differences are in the [saved reader report](results/configurable_capacity_v1/readouts/report.json).

**RPB-v20.alt-01:** compact backbone W64, embedding D32:20 learned and12 fixed
spectral coordinates. **RPB-v20.alt-02:** W128/D32; changes backbone width only.
**RPB-v20.alt-03:** W64/D64; changes embedding size to52 learned plus12 fixed.
All profiles use C3/H128/F9/P8, three layers, four heads, FF256 and decoder128.
These tags name fresh instance bundles; they do not replace older weights.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features
and structured distractors · **complexity5/5** · intact VALIDATION, updates512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 6912 | 49.6 (48.8–50.4) | 50.7 (49.6–52.7) | 100.0 |
| PCA only — no encoder | 32 | 49.6 (48.8–50.4) | 50.4 (49.3–52.0) | 100.0 |
| PCA only — no encoder | 64 | 50.5 (50.0–51.6) | 49.5 (48.8–50.3) | 100.0 |
| RPB-v20.alt-01 | 32 | 94.7 (93.8–95.3) | 94.3 (93.1–95.8) | 100.0 |
| RPB-v20.alt-02 | 32 | 94.1 (92.2–96.1) | 93.8 (91.8–96.2) | 100.0 |
| RPB-v20.alt-03 | 64 | 94.8 (92.6–96.5) | 93.4 (92.6–94.8) | 100.0 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features
and structured distractors · **complexity5/5** · extra30% coordinate deletion
VALIDATION, updates512; heads remain fitted on intact TRAIN.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 6912 | 49.2 (45.7–51.2) | 50.3 (49.9–51.0) | 100.0 |
| PCA only — no encoder | 32 | 49.7 (48.8–50.8) | 50.0 (48.0–52.1) | 100.0 |
| PCA only — no encoder | 64 | 51.0 (50.4–51.6) | 49.0 (48.2–50.1) | 100.0 |
| RPB-v20.alt-01 | 32 | 90.1 (87.5–93.0) | 88.7 (86.7–90.4) | 100.0 |
| RPB-v20.alt-02 | 32 | 89.3 (88.3–90.6) | 88.5 (86.5–89.8) | 100.0 |
| RPB-v20.alt-03 | 64 | 89.1 (87.5–91.4) | 88.5 (86.1–90.4) | 100.0 |

Initial0 controls are already strong: intact Linear/Neural scores are
94.01%/93.23% for alt-01,94.01%/92.84% for alt-02 and93.23%/92.40% for alt-03.
Alt-01's mean training gains are0.65/1.09 percentage points intact and0.78/0.82
deleted. Thus most timing strength predates learning. Fixed coordinates are
exactly unchanged, and every exported learned TRAIN coordinate changes.
Initial controls alone cannot attribute performance between the fixed prior
and random initialized learned coordinates; that requires a separate ablation.
Both points and all scored views remain in the
[unaltered published tables](results/configurable_capacity_v1/report.md) and
[machine-readable record](results/configurable_capacity_v1.json).

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features
and structured distractors · **complexity5/5** · fixed leave-one-P8-patch-out
TRAIN/intact VALIDATION queries; standardized MAE, lower is better.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v20.alt-01 | 512 | 0.7825 | 0.7986 | 20.8 |
| RPB-v20.alt-02 | 512 | 0.7821 | 0.8000 | 21.6 |
| RPB-v20.alt-03 | 512 | 0.7752 | 0.7950 | 27.9 |

These errors use fixed original hidden-target queries, not the training minibatch
loss. Mean initial VALIDATION MAE is0.8250/0.8265/0.8226 for alt-01/02/03.
Optimization uses hierarchical waveform Huber(delta1), AdamW.001, weight decay.0001,
gradient clipping1, patch masking25% and context coordinate deletion15%.
Mean Huber loss over the final32 updates is0.4100/0.4086/0.4037; all full traces
are retained. No label-dependent encoder objective or auxiliary loss is used.

GPU training seconds are mean synchronized training-loop time per trajectory.
Additional mean native export costs are4.81/4.93/6.59 seconds and fixed query
evaluation costs55.09/58.69/76.48 seconds for alt-01/02/03, including transfers
and I/O. Query evaluation ran once on CUDA at both checkpoint points. Across
all runs, training totals211.0 seconds and query evaluation570.8 seconds.
CPU raw preprocessing/PCA and fixed heads total114.0 seconds; audits reuse saved
arrays. There is no CPU encoder replay. These descriptive costs are not a
controlled performance benchmark.

Registered/trainable parameters are243641/239753,594809/590921 and260729/256841;
each has3888 frozen parameters. Training allocator peaks allocated/reserved
are40.10/56,44.09/66 and40.29/56 MiB, excluding process/context memory.
All4608 updates are finite and supported. Matched per-master scalers, sampling,
mask seeds/digests, target populations and context deletion counts agree across
the three configurations. Typed checkpoint reload and overwrite protection pass.
The independent saved-reader audit replays81 fitted head pairs (162 classifiers);
the publisher checks243 method/repetition/view decision records and both query points.
After completion, all815 artifact-manifest entries are independently rehashed.
Saved-only data review also reproduces all2654208 deletion-mask coordinates
from the declared counter stream, with29.95% of originally visible coordinates
removed. It repeats no encoder, training, head fitting or data generation.

The observed-only analytic information check supports and correctly answers all
3072 TRAIN/intact/deleted decisions before quality models are constructed. It
knows the declared signal family and informative feature groups; its100% score
is a solvability witness, not an encoder result. Engineering checks also admit
H256 data, but H256 encoder quality remains unmeasured. More history can expose
more cycles; tensor size alone does not establish greater learning difficulty.

Retain alt-01 as the compact continuation configuration. Alt-02 uses2.44 times
as many parameters without a consistent final classification gain. Alt-03 slightly
improves reconstruction, but doubles the served size, trains34% longer and loses
mean deleted accuracy. Every paired B/C-versus-A source-group95% interval spans
zero. This fixed512 comparison does not establish that larger capacity can never
help. No promotion or change to the formal RPB-v4 reference follows.

The [next direction](CONTINUATION_2026-10-12_AFTER_CONFIGURABLE_CAPACITY.md) is to
identify the learned branch's contribution from saved features, check history
geometry on H256, then target feature-specific information if needed. Keep heads
fixed and propose each new measurement before using quality sources. No TEST
or real-market data was accessed here.

Full local artifacts: `output/runs/rpb-v20-capacity/admission-0gN4uD`.
The committed [evidence directory](results/configurable_capacity_v1) preserves
source/card/settings captures, reports, loss traces, receipts and the full
artifact manifest. Checkpoints, inputs, queries and fitted tensor archives remain
in that ignored local capsule; they are not included in Git. Source identity:
`8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1`.
The earlier failed compile capsule `admission-xViaQs` is retained locally; SDK
namespace/initializer corrections preceded successful admission and quality runs.
