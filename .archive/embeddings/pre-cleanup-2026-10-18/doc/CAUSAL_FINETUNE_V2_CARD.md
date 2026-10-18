<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# CPB-v2.market-01: causal end-to-end adaptation

Declared 15 October 2026 before compilation, new encoder updates or quality
testing. Protocol `causal-finetune-v2`. The user requested renewed causal encoder
research after publishing the integration update, while keeping CPB-v1 as the
baseline. The publication is complete: integration `486886c361fe124b97659ba16c22857aae303d6d`
and research `e6bdd56ea16ddbb82444283e984158948e69032e` are on their existing main
remotes. The application remains on CPB-v1.market-01 throughout this experiment.

## Hypothesis and single change

CPB-v1 uses trained weights imported from the earlier noncausal computation,
with zero subsequent causal training. The experiment asks whether adapting the
learned temporal encoder and reconstruction decoder to their corrected causal
execution improves retention. CPB-v2 keeps native36, C3/H128/F9/P8, architecture,
forward arithmetic, raw-F64 handling, cutoff policy, attention mask, banks,
normalization and Gaussian waveform reconstruction objective unchanged.

The change is trainability: all learned parent encoder/decoder parameters and
the change projection/residual decoder/variance blocks receive gradients.
Literal fixed spectral weights remain frozen. CPB-v1's separate inference-only
wrapper and its parent's NoGrad paths must not silently prevent learning.
There is no auxiliary loss, task-label supervision, synthetic training or
architecture/width change. Parent32 may change after learning; it must not be
described as an exact preserved prefix after training.

This is **fine-tuning from CPB-v1**, not training from scratch. Every initial
parameter and buffer, input scaler and bank normalizer must match the exact
baseline checkpoint. Baseline checkpoints/results remain immutable. The source
checkpoint is local in integration:

`output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/proof/inference.pt`

SHA-256 `54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895`,
1,290,592 bytes. The machine card binds that file, frozen v1 source, saved input
artifacts and exact quality floors before measurement.

## Version ownership and execution

CPB-v2 owns a complete physical copy under
`code/encoders/causal_patch_bottleneck_mae/versions/v2/`, with its own namespace,
public header, private implementation, configuration, tests and provenance.
No mutable v1/RPB implementation include, alias, symlink or execution dependency
is permitted. Initializing from the pinned v1 artifact reads saved tensor data;
it does not run the old encoder and is distinct from resuming a v2 training state.
Use separate v2 inference/training kinds and causal-update/source provenance.
Do not edit evaluated v1 or add a second encoder to active `cuwacunu_dev`.

Use only the existing GPU-enabled `cuwacunu` container. Research source will be
staged as a hash-verified, exclusive source capsule inside its filesystem because
the research checkout is not mounted there. Read existing `/cuwacunu` inputs
without mutation, keep builds in named container directories, retain exclusive
run outputs and copy them back into ignored research output storage. No new
container, mount, dependency installation or host project execution is authorized.

## One bounded training trajectory

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute candles, nine features,
recent128 historical-window targets · **complexity 4/5** · existing saved TRAIN
selection only. The existing 4,096 H256-legal TRAIN anchors cover January 1–21;
the encoder consumes their recent128 windows. Evaluation uses the same 112 TRAIN
and 32 January 22–28 known chronological VALIDATION probes. January 29–31 remains
unused. These are retention probes, not forecasting tests.

Exactly 2,048 new causal updates, batch8, at most 4,096 attempts. Use the existing
25% hidden-query and 15% extra-context-deletion policy, eligibility/skips and
absolute-counter random streams. AdamW learning rate0.001, weight decay0.0001,
global gradient clip1. Seeds942001 initialization (all learned tensors are then
overwritten by the exact warm start), 942002 rows, 942003 query masks, 942004
context and 942005 saved reconstruction queries. Fresh optimizer/moments/counters
begin at causal update0; inherited parent/branch updates are reported separately.
Dropout stays0. No scaler or bank calibration refit.

Keep the same frozen input identity
`rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969` and bank identity
`rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476`.
Use original raw record doubles for the change bank; preserve the parent path's
declared rawF64→F32→CPUF64 saved-scaler→CUDAF32 rounding.

Save initial/final typed inference artifacts and TRAIN states at512/1024/2048.
Select only the final2048 checkpoint for quality, regardless of reconstruction
or quality behavior. No rate, width, seed, budget, head or checkpoint rescue.
Preserve failures and partial states without relabeling them as completed runs.

Report Gaussian NLL and its components separately from standardized MAE on the
unchanged saved TRAIN/VALIDATION hidden reconstruction query plans. Measure
synchronized encoder compute time separately from loading, checkpoint writing,
proof, feature extraction and readout fitting.

## Mandatory engineering admission

Before quality, prove all of the following:

- Exact initialization of all model/scaler/bank tensors; initial native36 matches
  saved CPB-v1 feature witnesses. Reuse initial fixed-head results only after
  parity; no initial head refit or baseline replay.
- Actual CUDA inputs/loss/finite nonzero gradients and parameter movement in
  temporal attention, learned parent projection/pooling, parent decoder and
  change/variance blocks. The fixed spectral prior and both normalization states
  remain bit-exact. Remove inherited zero-start-head assumptions, which are
  invalid for this nonzero warm start.
- Training and inference crop values **and masks** before value validation,
  scaling, banks, random masks or target/loss construction. No target/support
  reduction sees later observations. Temporal attention remains causally masked.
- Pre/post-training suffix invariance across all128 cutoffs, inside-patch cases,
  long/mixed histories, missing support, channel permutations and future
  NaN/Inf/overflow poison. Include independent internal temporal-state prefix
  interventions, deterministic repeat controls and legal-past positive controls.
- Typed interrupted/resumed next-update parity: model, ordered optimizer
  parameters/moments, attempts/completed counts and row/mask/context streams.
  Reject wrong kind, source, geometry, recipe, normalization and corrupt/reset
  optimizer state. Initialization is not resume. Do not reuse the old four-block
  frozen-parent optimizer guard for end-to-end training.

Engineering witnesses may use a separate bounded fixture trajectory. Their
updates and artifacts are distinct from the single quality-training trajectory.
Capture exact source/configuration and executable/dependency hashes before each
attempt; retain failed attempts and do not alter admitted code in place.

## Fixed final quality measurement

MKT-HIST-2 uses saved targets/populations and the unchanged native36 regression
heads: ridge penalty1; 16-tanh neural, Adam0.01/100 updates; repetitions2701/2802/2903
with dimension-derived seeds13088411117524848790,10018485049408248344,
9083751155097190455. D36 regression heads contain148/660 fitted parameters.
Fit only final v2 heads once on TRAIN. Reuse historical and initial v1 scores.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, grouped nine-feature
distractors · **complexity 5/5** · existing cohorts941101/941102/941103, each
512 TRAIN/256 VALIDATION examples (256/128 paired source groups). Use both frozen
input-scaler policies (`synthetic_train`, `market_scaler`), intact and saved extra
30%-deleted validation views; market bank normalization is unchanged in both.
The input manifest is SHA-256
`fdd8b04917b1d4580fea760e4c6d2dedf8a9d47b14271758174da86859d62ccb`.
Use the identical fixed classification readouts, repetitions and width-derived
seeds as CPB-v1 (D36 heads74/626 parameters). Fit each final head once on its
permitted TRAIN features; reuse baseline scores. No post-encoder PCA, head
tuning, seed selection or old encoder training/inference replay.

Four market task tables and four TEMPO-5 policy/view tables remain separate,
with full-precision repetitions/means, physical market errors, coverage and
conditional source-group intervals in durable detail. Follow
`RESULTS_REPORTING_STANDARD.md`; do not combine tasks into an encoder accuracy.

## Advancement criteria, frozen before testing

All engineering gates are mandatory. A candidate qualifies as an improved
baseline only if every condition below passes (arithmetic tolerance1e-12,
which does not relax any substantive margin):

1. Original nine market conditions, strict native36 eight floors, all three
   positive neural-trend fits and full coverage, with the existing numeric floors.
2. Each of CPB-v1's eight market linear/mean-neural scores is retained, and
   mean neural trend is strictly higher than0.24368774940638713.
3. Each of the eight TEMPO-5 policy/view/head means reaches the greater of its
   CPB-v1 mean and the existing native36 retention floor, with full coverage.

Exact per-condition floors and baseline values are copied into the adjacent
machine card. A failed recipe remains a preserved result and does not displace
CPB-v1 or alter these criteria. No outcome is predicted by this declaration.
