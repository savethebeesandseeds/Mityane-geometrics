# Matched-target TRAIN magnitude support v1

Prospective development protocol: `matched-target-gain-v1`. Freeze
the final card, complete source closure, actual CUDA admission and reviewed
independent reader before generating quality data. This is one fixed comparison,
with no rate, magnitude, loss, head or budget search.

## Question and groups

Does broader magnitude support during timing TRAIN improve the native embedding
without sacrificing original-query reconstruction or amplitude accessibility?
The whole preceding learning curve reduced reconstruction error without a
consistent timing linear-head improvement. Both paths have now been measured
at all declared budgets; this experiment does not select a favorable point.

**RPB-v10.alt-02** is a fresh early-mixing control. **RPB-v11** is early mixing
with broader TRAIN magnitude support. Their inference architecture is identical:
pretemporal aligned channel mixer, separate independent local temporal pass,
mode2/mixer1, native32, 225,805 parameters, dropout0, same global decoder.
The candidate changes only its closed TRAIN view. Preserve all existing groups,
including original RPB-v7. No automatic promotion occurs.

Timing masters: 41140,42241,43342,44443,45544. Separate amplitude masters:
46645,47746,48847,49948,51049, paired in that order. Generate each task/cohort
once with 128 TRAIN source pairs and 64 VALIDATION source pairs (256/128 rows),
C3/H32/F3, patch8, semantic IDs0/1/2, unitless features, interval1, endpoint31,
CPU float64 legal observations, natural missingness0.10 and zero absent storage.
Prefix IDs with this protocol/task. TRAIN and VALIDATION sources are disjoint;
there are no TEST/stress inputs or historical quality payload inputs.

## Closed candidate training view

Both factories receive the same label-free original timing TRAIN observations
and source order under `matched-target-gain-v1/lag_sign`. Each factory clones
these and fits its original TRAIN scaler once before applying any gain; require
the two scaler states to agree byte for byte. Amplitude never fits
an encoder or encoder scaler. Initial parameters, buffers, original scaler,
configuration and original-input CUDA exports must agree exactly.

Sort distinct TRAIN source IDs lexically, assign ranks starting0, and require
two repeated rows per source without inspecting labels. Define independent
stream `0x6761696e2d763131` (7449351180577550641). Use the existing counter law:
`bits=counter_seed(actual_training_seed,source_rank,stream)>>11`,
`u=bits*0x1p-53`, `g=exp((2*u-1)*0.693147180559945309417232121458176568)`.
The nearest float64 LN2 and operation order are fixed. One static g in [0.5,2)
is shared by all rows from that source, across every attempt. Gain assignment
does not consume row, mask, context or Torch RNG and is stable to row ordering.

At sampled-row selection, candidate-only CPU float64 legal values become
`where(observed,g*x,0)`. Apply the unchanged original scaler exactly once and
then convert to CUDA float32. The same normalized gained batch supplies visible
input and Q reconstruction targets. The affine law is `(g*x-mu)/sigma`; simply
multiplying centered values is forbidden. No gain reaches the decoder as side
information, and no clean/absent values are accessed. Gain scales observed noise
too, preserving SNR; it is not the amplitude generator's fixed-noise law.

Both keep coordinate0.15 context deletion, original Q/eligibility/hierarchical
Huber1, deterministic E-only repair, batch8, AdamW0.001/weight-decay0.0001,
gradient clip1, one thread, full per-attempt trace, exactly512 completed CUDA
updates and attempt cap1024. Abort on any skip or exhaustion. Ten trajectories
give5,120 updates/40,960 sampled row exposures; each has4,096 exposures, or16
equivalent presentations of256 TRAIN rows, not guaranteed epochs. Extra
decoder-only calibration is0. No checkpoint reload-resume is introduced.

Control retains `rpb-training-context-deletion-015-v1`. Candidate has
combined policy `rpb-training-context-deletion-015-source-gain-v1`, with the deletion component
separately recorded. Ordinary/legacy resume and snapshot adapters must reject
the unfamiliar protocol/policy. Quality inference and fixed-query reconstruction
apply no gain or training deletion, using the preserved original scaler.

## Fixed evaluation

Each task has intact VALIDATION and one common additional0.30 coordinate erasure
view with no support repair. Timing uses `stream_seed(timing_master,
0x6d67763174696d30ULL)` with namespace
`matched-target-gain-v1/lag_sign/validation-coordinate-dropout`; amplitude uses
`stream_seed(amplitude_master,0x6d677631616d7030ULL)` and
`matched-target-gain-v1/amplitude/validation-coordinate-dropout`. Keep the
existing source-paired physical-grid law. All six methods share observations,
source order, labels, erasure masks and fixed heads.

Methods are raw576 (288 values+288 mask), mask288, PCA32 only, one byte-proved
shared untrained early32 control, native RPB-v10.alt-02@512 and RPB-v11@512.
PCA follows raw data only. Raw and PCA share one TRAIN outer normalization;
head-specific TRAIN normalization remains unchanged. Three repetitions
2701/2802/2903 use Ridge penalty1 and tanh16/Adam0.01/100 updates; seeds follow
the existing repetition/width law. Fit each head once on ordinary TRAIN and
reuse for both VALIDATION views. Width32 has66 linear/562 neural parameters;
raw576 has1,154/9,266, mask288 has578/4,658.

Expected180 pipelines/360 heads, including90 native pipelines,10 driver raw
outer fits and40 helper outer fits. Preserve unsupported statuses, coverage,
all five cohorts/three repetitions and within-master paired source intervals.
No across-task averaging or averaging interval endpoints; repetitions do not
multiply the number of independently trained encoders.

Retain both models'0/512 checkpoints and typed companions. Before initial-head
fitting, compare both actual point0 exports on every task's TRAIN/intact/deleted
surface. Preserve candidate counterpart archives separately as exact parity
evidence. There are90 unique quality native exports plus30 additional actual
initial-counterpart verification exports, or120 full native forwards. This
necessary CUDA pairing is disclosed; no CPU encoder forward repeats it.
Positive timing TRAIN/intact VALIDATION reconstruction adds20 query writers
and80 required Q-masked CUDA forwards, using the unchanged four original patch
query banks. No point0/amplitude reconstruction is included.

## Admission, archives and independent audit

Artificial CUDA admission proves disabled-policy parity to the existing early
trainer; common full initialization/scaler; paired and row-order-stable gains;
nonzero-mean affine correctness; exact absent zeros and unchanged masks/Q/E;
gained input/target identity and finite CUDA gradients/actual updates; exact
direct versus segmented model/buffer/AdamW/counter/gain state; immutable
artificial0/2/4 snapshots; and source/scaler/policy/namespace rejection. Quality
retains0/512 snapshots separately. Preserve failed gates.

Save original TRAIN raw/scaler, manifest identity, lexical source ranks/top53
integers/float64 uniforms/gains and row association. Candidate companions retain
every512 update's sampled row indices, attempted index and actual normalized
float32 target batch. Independent CPU arithmetic reconstructs all gained
targets from original raw+frozen scaler, checks zero absent storage, association
and exact counters, then unchanged saved native/support/query/head arithmetic.
It runs no encoder, autograd, optimizer, head fitting or PCA/SVD.

Checkpoint bodies are byte-bound; CPU state witnesses and actual CUDA admission
establish state semantics, without pretending to replay AdamW independently.
Freeze schema/whole role matrix/tolerances with artificial fixtures before
quality. Gain exp uses absolute2e-15+relative2e-15; existing float64 replay uses
2e-9+2e-9 and float32 replay2e-6+2e-5. Discrete support, ranks, top53, source
association, zero storage, saved-logit argmax and equality witnesses are exact.
Any reader execution release changes only reviewed flags with byte proof.

## Decision and reporting

Report both timing views and amplitude separately using the agreed quality
table. Short descriptions sit beside tables. Training tables use original
fixed-query standardized MAE and synchronized cumulative CUDA-loop seconds;
retain Huber/gradient/target-count traces separately. Gain intentionally changes
the training target variance, so sampled loss is not a common quality metric.
Separate loop, full native inference, necessary query/parity inference,
binding/I/O, CPU baseline/head/bootstrap and audit costs. Pure kernel time
remains unmeasured unless instrumented.

Primary joint direction requires candidate timing linear means and worst-cohort
scores to be no worse in BOTH views, common coverage unchanged, and original
TRAIN/VALIDATION query MAE means no worse. Report each pass/fail literally;
retain all per-cohort tradeoffs and secondary neural/amplitude results. A failed
joint direction stops this recipe without magnitude/rate/budget rescue. Even a
pass is development evidence, requiring a separately frozen confirmation before
promotion. Strong untrained amplitude accuracy limits learning-credit claims.

No seed dropping, head tuning, native PCA, TEST/stress, selected budget or
automatic promotion. A favorable result supports this fixed broader TRAIN view;
it does not establish an internal causal bottleneck or consumer invariance.
