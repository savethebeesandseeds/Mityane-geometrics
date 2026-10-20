# V15 objective comparison: prospective protocol

Declared 19 October 2026 (Asia/Dubai), before building, training or opening new
confirmation values. The owner approved V15 with “ok, let's do it” after the
V4-based objective proposal. The local resumption and calendar-only data
declaration are under
`.temp/embeddings/runs/cpb-v15-objective-001/attempt-001/`.
This is a new experiment with its own source, checkpoint and result identities.
The saved V14 TRAIN diagnosis remains an unexecuted draft, deferred by this
choice. It is not an experimental prerequisite or a result supporting V15.

## Question and controlled change

Does replacing V4's variance-weighted partial Gaussian-NLL mean gradient with
unit Huber1 improve the information accessible to fixed heads at native36,
without losing trend or hard timing retention? Control A is fresh V4. Candidate
B is fresh V15: unit Huber(delta1) mean supervision plus unit Gaussian variance
calibration on detached mean error. V4's detached mean4 input to its variance
projection is preserved. The calibration route reaches only the variance head;
the mean Huber route reaches the mean/backbone. A single AdamW and a single
global clip1 act on all learned parameters in both arms. Gaussian NLL remains a
separate reporting score. Its value is not the V15 training objective.

Architecture, parameter registration, initialization draws, pooling and fixed
spectral branch match V4: learned20 + spectral12 + mean4, 75 learned tensors,
293672 learned values, 3888 fixed values, 297560 total. There is no added
uncertainty embedding. All weights, optimizer state and TRAIN statistics start
fresh. No old trained encoder, normalization or feature archive supplies state.

## Frozen execution and geometry

Three paired encoder initialization seeds are 945101, 945102 and 945103.
Within each pair row/mask/context streams are seed+1000/+2000/+3000. Each arm
has 2048 successful updates, B8, at most4096 attempts, AdamW lr0.001,
weight_decay0.0001, global_clip1, mask25%, context deletion15%, no dropout.
The run order is 945101 A/B, 945102 B/A, 945103 A/B, sequentially on the shared
GPU. Actual typed continuation at512 must preserve parameters, moments,
counters and stream identities. Initial0 and final2048 are both saved. Final
2048 is the sole outcome checkpoint; there is no checkpoint, seed, coefficient,
head, mask or budget rescue. Failure is retained and does not authorize a new
scientific trajectory. A repaired source needs a new retained generation and
explicitly stated disposition of the failed attempt.

Preprocessing and analytic-bank calibration use January TRAIN only. Data,
frozen heads, temporal support and matching are described in [README](README.md)
and bind to exact runtime pins. The representation is the unprojected native36
vector, with no PCA or newly fitted geometry. The fitted feature scaling is a
head dependency; Euclidean/cosine neighborhood claims are outside this study.
Market ridge1 and tanh16/Adam0.01/100-step heads have reps2701/2802/2903.
Their D36 uint64 seeds remain 13088411117524848790,10018485049408248344,
9083751155097190455. Those three fits are head repetitions, not encoder seeds.
All four past-window targets and all TEMPO views/modes are reported.

All six trajectories finish before development extraction/readout scoring.
Each saved initial/final checkpoint must pass actual cutoff/suffix/internal
causality proof. All saved head arrays, TRAIN normalizers, ridge equations,
head predictions, metrics, masks and row/source association receive independent
native arithmetic audit. The retained January query waveform witnesses cover
the first8 of each112/32 query population; their waveform arithmetic is audited,
while full-population means are checked from the producer's saved row metrics.
That does not establish independent waveform reconstruction for every January
query. February query witnesses cover all32 probes. The complete six final checkpoint/head generations
must then be sealed before confirmation extraction or scoring. No confirmation
target, feature, prediction, score or loss is exposed to the human until then.

February2024 BTC/ETH/SOL Spot1m is confirmation-only. Its32 H256-disjoint
calendar-selected probes and exact format/license constraints were declared
before download. Encoder history is128; targets use the frozen past-window
definition and historical F32-loader then F64 precision, first checked against
all saved January112/32 labels. The same TRAIN-fitted head arrays and scalers
predict February; no February fit, selection, row replacement or tuning.
Fixed-query February reconstruction is descriptive, with the original32 query
Boolean geometries, newly recomputed support and no reconstruction gate.

## Prospective outcomes and interpretation

Primary outcomes are February final-checkpoint **linear R2 for past volatility
and past activity**, candidate minus paired control, separately. Report all
three paired differences, their median and range, and all head repetitions.
Each target's linear ridge is deterministic; duplicate head-repetition ridge
scores do not increase the encoder sample count. No pooled composite, best seed,
significance claim or optional stopping is used.

The initial screen is called **promising** only if each primary target has a
positive median paired difference and improves in at least two of the three
pairs, the median February linear trend difference is at least -0.01 R2, and
every candidate retains all eight frozen TEMPO hard-timing checks in January
development. This declared0.01 trend tolerance applies only to the new screen;
it changes no historical protection floor. Otherwise the result is mixed,
negative or inconclusive, with exact failing conditions recorded. A failed
engineering or numerical admission is invalid execution, not a negative
scientific result. Undefined R2, missing pair, support failure or corruption
prevents a promising verdict.

Secondary reports include January development and February all four targets'
linear/neural R2, MAE, RMSE, support and three-seed variation; initial/final
changes; all TEMPO masters, scaler modes, intact/deleted coverage and accuracy;
TRAIN/known-validation/fixed-confirmation query MAE and Gaussian NLL; trained
parameter movement, clipping/objective witnesses, and actual synchronized
compute cost. January has been used repeatedly and is development evidence.
February scores compare the new paired arms on February; they cannot replace
or be directly ranked against old January scores.

The exact22 strongest V3–V13 predicates in
`.archive/embeddings/pre-cleanup-2026-10-18/doc/causal_fresh_v14/goal-protocol.json`
are retained byte-for-byte as a separate January protection report for each
candidate seed. All earlier leaders remain reference results. V4's historical
19/22 used its original floors; V14's3/22 used stronger floors. Those counts
remain incomparable as a ranking scale. Formal advancement still requires all
applicable frozen protection predicates plus trustworthy complete evidence.
This experiment performs no automatic promotion, catalog application or
Mityane model selection. Even a promising result is a three-seed screen on one
additional month, not an established general-purpose best encoder.

## Operation contract and linked evidence record

**OR-05 / CPB-past-readout v1:** input is an observation embedding with exact
checkpoint, native shape, mask, temporal support, TRAIN preprocessing and saved
head identity. Output is four scalar past-window estimates in the frozen
target units, with validity and source IDs. Required properties are accessible
past volatility/trend/activity/relation information at bounded head capacity.
Dependencies are source observations, target definitions and fixed TRAIN-fitted
heads. The claim is predictive association on declared populations. Invalid
support, zero target variance, unavailable artifacts or failed audit requires
abstention. The primary controlled baseline is fresh paired V4; random initial
encoders are an additional within-source baseline. Historical raw/target
baselines remain historical context and are not rescored solely for this study.

**OR-09 / CPB-fixed-query v1:** input additionally includes fixed masked query
geometry and the model-owned decoder. Output is standardized masked MAE and
Gaussian NLL with eligible example/channel/cell counts. The property tested is
reconstruction through the declared bottleneck under the specified masks.
Dependencies are TRAIN normalization, observation support and decoder state.
Unsupported/nonfinite/empty queries abstain. Lower reconstruction loss alone
does not imply better readouts or forecasting. TEMPO is a scoped OR-05
synthetic transfer test with fixed generated cohorts and target labels.

The Requirements-and-Evidence Record is this prospective protocol plus the
ignored actual input manifest, resumption, independent source/protocol reviews,
build/CUDA receipts, per-subrun outputs, independent arithmetic receipts,
six-run seal and final report. Hypothesis, controls, metrics and failure rules
are fixed above; numerical results and verdict are pending. Exact source and
binary pins, seeds, environment and elapsed compute belong in the final record.
Independent replay of saved arrays verifies their arithmetic; it does not claim
a second model implementation, optimizer replay or independently audited
bootstrap confidence intervals. No OR-11 forecasting, identified intervention
effect, decision policy or downstream application utility is established here.

## Evidence preservation and runtime boundary

Use the existing managed container through root `container.ps1` and named tasks.
All new implementation and audits use C++, Node and Bash; no Python fallback.
Publisher format/terms and checksums must be saved and reviewed before import.
Only local personal noncommercial non-production research is authorized. Raw
datasets, checkpoints and generated outputs stay ignored; no publication or
commercial use follows from this run. Preserve every failed attempt and all
historical archived bytes. No container/image/dependency or ownership change,
cleanup, commit or push is part of this experiment.
