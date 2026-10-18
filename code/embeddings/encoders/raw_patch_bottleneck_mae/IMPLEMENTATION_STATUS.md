# RPB-MAE implementation and validation history

This file preserves engineering stages recorded from 6 October 2026, through
the balanced-view RPB-v8 milestone below. It is not the current status of the
model family. RPB-v20 is implemented separately with a completed capacity
quality record. See [current status](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CURRENT_STATUS.md)
and the [configurable API](CONFIGURABLE_SPECTRAL.md). Original measurements and
their protocol semantics remain unchanged.

Date: 2026-10-06

**Active experimental version: RPB-v4 — Learned global bottleneck.** Use
[stable version tags](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md) and the
[results reporting standard](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/RESULTS_REPORTING_STANDARD.md).
The [next advance plan](NEXT_ADVANCE.md) focuses on native32 outputs and fixed
heads, with PCA only on raw data. The new shared archive evaluator completed
[phase-1 controls](NATIVE_BASELINE_COMPARISON.md) without encoder retraining or
TEST access. The separate shared native-only training/selection runner now
completes [phase 2](NATIVE_CURVE_ADVANCE.md): CUDA correctness and focused tests
passed, 512 updates were selected on native linear VALIDATION before fresh TEST,
and the independent audit passed 70,619,069 checks. Timing TEST means are 94.01%
linear / 96.09% neural, at 100% coverage. Historical global/curve drivers retain
their compressed selection protocols. More training improves reconstruction
without improving timing validation; phase 3 targets pooling consistency and
moderate missingness with one encoder change and fixed heads.
The sections below preserve the validation chronology; older designs are
archived references rather than routine retraining candidates.

The first independent-channel candidate is implemented alongside the working
MTF-JEPA-MAE-VICReg baseline. This records engineering checks and bounded
development comparisons; it is not a promotion decision under the
[shared evaluation policy](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_EVALUATION_POLICY.md).
Commands and archive semantics are in the [README](README.md).
The original comparisons are archived protocol-v1 evidence. Later sections
record protocol-v2 and separate reconstruction/stress tracks. Current shared-driver
ownership and protocol-v2 limits are documented in the
[integration guide](../../evaluation/README.md) and
[implementation review](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EVALUATION_IMPLEMENTATION_REVIEW.md).

## Implemented scope

- Raw ordered patches, shared channel-independent temporal attention,
  position-aware compact local vectors and equal-valid-channel global pooling.
- Frozen observed-only float64 scaling before neural float32 conversion,
  explicit semantic IDs, visibility versus reconstruction support, endpoint
  cropping and uniform sampling checks. The crop helper requires an endpoint on
  the sample grid; the archive API expects already cropped histories.
- Bottleneck-only masked Huber decoding, hierarchical loss, invalid zero exports
  and separate attempted/completed update counters with bounded skipped batches.
- Versioned raw/scaler/checkpoint/export archives, dataset/schema identities,
  source fingerprints, AdamW continuation and atomic periodic recovery.
- Independent build targets/binaries and encoder-owned configuration/tests.
  Shared feature fitting/scoring does not depend on either encoder class.
- Paired source-group synthetic tasks, train-only scaling/normalization/PCA,
  fixed ridge and secondary nonlinear probes, raw/descriptor/mask controls,
  untrained and optionally freshly trained frozen encoder controls.

The original comparison's Makefile source manifest fingerprinted both encoder
dependencies, shared headers/source, configuration, build rules and dependency
lock. Current model and evaluation builds have separate source manifests.
Scaler/dataset
FNV checksums are reproducibility identifiers, not cryptographic proofs.
Checkpoint-producing and inference source identities are distinct export fields.

## Original implementation container checks

All builds, tests and training use the existing `cuwacunu_embedding` Linux
container, staged LibTorch 2.6.0+cu124 and named build sessions. No container,
package or host toolchain was added. Baseline implementation/configuration and
its previous archive format are preserved.
During the original implementation, hash comparison confirmed all 42 preexisting
baseline/compatibility/shared implementation, configuration and test files
remained unchanged. This describes that stage, not the later source-ownership
and shared-evaluator refactors.

| Check | Result |
| --- | --- |
| Baseline regression suite and original extraction parity | Passed |
| Baseline CPU/CUDA train, resume and batched export | Passed |
| Baseline real interrupted-process periodic recovery | Passed |
| RPB preprocessing, numerical, masking and core model tests | Passed, including off-grid endpoint rejection and future invariance |
| Shared feature-harness fitting/protocol checks | Passed, including finite-fit rejection and bounded diagnostic parity |
| RPB tagged archives, exact CPU continuation, skip recovery and path preservation | Passed |
| RPB CPU/CUDA CLI and interrupted-process periodic recovery | Passed, including CUDA checkpoint inference on CPU |
| Independent minimum comparison | Completed four tasks, seed 101, fresh baseline four updates |
| Fresh-encoder comparison | Completed four tasks for seeds 101/202/303, baseline eight and RPB twenty updates |
| Saved-checkpoint evaluation with its frozen scaler | Completed four tasks, seed 101; no model/scaler refit |

The fixed-batch core optimization check reduced masked Huber from `0.533232` to
`0.0665845`. That verifies optimization through the served vector; it says
nothing about held-out representation usefulness.

The private build sessions are `implementation-baseline` and
`rpb-implementation`. Persisted smoke/evaluation artifacts are under
`output/runs/<session>`; compiler products stay inside the container at
`/opt/cuwacunu_embedding/build/<session>`.

## Archived protocol-v1 bounded comparison

The [minimum report](../../output/runs/rpb-implementation/minimum-uz8IWb/results/report.json)
contains four runs, 64 measured surface/tier comparisons and four unsupported
compressions. The [fresh-encoder report](../../output/runs/rpb-implementation/rpb-evaluation-RCiWaW/results/report.json)
contains twelve runs, 288 measured comparisons and twelve unsupported
compressions. Each run uses 32 training, 16 validation and 32 testing source
pairs (64/32/64 rows). Every raw oracle scored 100%; mask-only ridge accuracy
was 50%. Unsupported mask compression to 36 dimensions correctly reports
insufficient training rank rather than padding a result.

The following are mean test accuracies over the three development seeds for
the fixed ridge probe at twelve global dimensions. Baseline vectors are natively
twelve-dimensional and the original matched tier retained those vectors without
PCA; RPB vectors use train-fit PCA from 32 to twelve dimensions. Protocol v2 now
applies PCA to every matched surface, including equal-width inputs, so these
scores must retain their original protocol attribution.

| Task | Untrained baseline | Fresh baseline | Untrained RPB | Trained RPB |
| --- | ---: | ---: | ---: | ---: |
| Reversal | 100.0% | 100.0% | 100.0% | 100.0% |
| Level | 100.0% | 100.0% | 100.0% | 100.0% |
| Amplitude | 100.0% | 100.0% | 94.8% | 94.8% |
| Lag sign | 47.9% | 47.9% | 54.7% | 47.4% |

All these surfaces had full validity coverage in this fixture. Local concatenated
RPB vectors at 36 PCA dimensions scored 62.5% untrained and 59.4% trained on lag
with the separate nonlinear probe. These modest exploratory values establish
neither lag recovery nor a learned advantage. The first three tasks are largely
saturated before training and cannot discriminate successful representation
learning in this budget.

The pretraining budgets are deliberately small and do not match compute or
observation exposure: the baseline uses its full 64-row training partition per
update, while RPB uses replacement minibatches of eight. Input preparation is
also architecture-specific and disclosed in the assets. The reports check
protocol/integration and provide a starting point, not a fair final training
comparison or permission to select the test-set winner. They include validation
scores, frozen probes/PCA/scalers, source manifests, grouped uncertainty and
shuffled-label controls.

The recorded source fingerprint is
`2ba521d6bf3a24daf10b056ebb26888c112e25d776c8448bea11dfd84557e171`.
Generated reports/archives are retained locally and excluded from Git. A clean
checkout will not contain those local report links until the runs are reproduced.
The [saved-checkpoint smoke report](../../output/runs/rpb-implementation/frozen-checkpoint-smoke/report.json)
uses the CPU smoke checkpoint's saved model/scaler and records the same inference
source fingerprint.

The following are historical protocol-v1 reproduction commands. They require
archived source and scripts matching the recorded fingerprint, built inside the
managed container in the `rpb-implementation` session. Current v2 source,
scripts and `embedding_evaluate` use a different protocol and do not reproduce
these archived scores:

```bash
env EMBEDDING_BIN=/opt/cuwacunu_embedding/build/rpb-implementation/feature_harness \
  EMBEDDING_RUN_ROOT=/embedding/output/runs/rpb-implementation \
  bash code/embeddings/scripts/evaluate-minimum.sh --seeds 101 --train-pairs 32 \
    --validation-pairs 16 --test-pairs 32 --baseline-steps 4
env EMBEDDING_BIN=/opt/cuwacunu_embedding/build/rpb-implementation/embedding_raw_patch_bottleneck_mae \
  EMBEDDING_RUN_ROOT=/embedding/output/runs/rpb-implementation \
  bash code/embeddings/scripts/evaluate-rpb-mae.sh --seeds 101,202,303 --train-pairs 32 \
    --validation-pairs 16 --test-pairs 32 --baseline-steps 8 --rpb-steps 20
```

## Shared evaluation refactor status

Evaluation orchestration now lives in the shared engine, with registration in
`code/embeddings/evaluation/` and an encoder-owned RPB adapter. The ordinary
`embedding_raw_patch_bottleneck_mae` binary owns synthetic/prepare/train/embed;
it has no evaluation command and no baseline/evaluator linkage. Build the
`evaluation` target and use the separate `embedding_evaluate` executable for
comparisons. The [README](README.md) contains current commands.

The controlled-pairs-v2 run saves an instantiated card before generation and
fitting. That card fixes C/H/F, schema IDs/units, task and label budgets,
matched widths, seeds and explicit compared-method pairs. A companion experiment
plan is required to freeze the complete encoder configuration and training budget.
The adapter's fit
input exposes only permitted training observations and declared metadata;
labels, hidden clean signals and held-out observations stay in the evaluator.
RPB global surfaces use any observed-valid channel; typed
`*_channel_concatenation` surfaces use independent local vectors in declared
semantic order and require every constituent channel to be valid.

Matched tiers always use train-fit PCA, even at equal native and target width.
Insufficient valid training support or numerical rank is an unsupported fit.
Each declared pair has a fixed common-valid population, independent of unrelated
providers. Fresh baseline controls are fitted only on permitted training
observations; the shared driver supplies no verified historical
checkpoint/preprocessing association.

Refactor container validation passed shared dummy-provider and fitting tests,
both model/workflow regressions, original-reference baseline extraction parity,
binary dependency inspection, both-adapter development comparisons, frozen RPB
checkpoint inference and a repeated independent-geometry card. The dated
[shared review](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EVALUATION_IMPLEMENTATION_REVIEW.md) records actual
artifacts, source identities and scope. The original results above remain v1;
no quality promotion follows from the refactor.

## Held-out decoder reliance

The separate [reconstruction record](DECODER_RELIANCE.md) contains a frozen
development recipe and [12 measured runs](../../output/runs/rpb-implementation/reconstruction-2jehCR/results/report.json).
Each trained encoder and independent metadata-only decoder received 128 updates
on the same training source groups and mask streams. All test populations had
full coverage. Every task/seed favored intact vectors over metadata-only
decoding and degraded under valid source-group shuffling, with positive
within-run paired intervals. This establishes reliance on signal values in
these synthetic histories, without establishing classification/lag-probe
quality, robustness or consumer acceptance.

Shared dummy-provider and RPB adapter tests passed, including hidden-storage
and mutation isolation, exact invalid zeros, saved metadata encoder weights,
numerical summary guards and ordinary checkpoint trainability. A saved main
checkpoint resumed through the ordinary CLI for an additional update and
exported embeddings. Restored trainable float32 inference differed by at most
`7.45058e-8`; matching frozen gradient flags restored exact parity. Both satisfy
the existing workflow tolerance. Scaler identity and raw precision were exact.

The small repeated recipe produced identical reports/cards. Existing full and
minimum feature-evaluation paths still pass, and their binary dependencies
remain separate. The [validation manifest](../../output/runs/rpb-implementation/reconstruction-validation-HZp50W/validation.json)
records source identities, unchanged-file hashes and retained artifacts.

## Stronger frozen-probe comparison

The [frozen recipe and measured results](FROZEN_PROBE_ADVANCE.md) record 12
fresh 128-update development runs and 48 fixed trained-versus-untrained pairs.
All requested pairs have full test coverage: 128 rows from 64 independent
source groups. Legal raw oracles score 100%; native and twelve-dimensional
mask-only probes score 50%. Twelve mask-only PCA36 fits remain explicitly
unsupported because their training rank is 31.

The focal lag-sign comparison uses PCA36 channel concatenation and ridge.
Trained accuracies are 52.34%, 48.44% and 52.34% for seeds 101/202/303; all
three paired trained-minus-untrained intervals cross zero. Native concatenation
with the fixed nonlinear probe scores 60.94%, 83.59% and 90.62%, respectively,
and has positive within-run training-effect intervals. The nonlinear result is
diagnostic evidence of relational accessibility, not a rescue of the primary
compact linear claim or proof that compression alone causes its weakness.

The feature adapter now persists actual training settings/seeds, counters and
the adapter producer identity. Its untrained model records no pretraining and
zero weight updates while retaining separate permitted scaler-fit provenance.
The new adapter test and shared feature/reconstruction tests pass. A fixed
seed202 lag run repeats exactly, and all 25 feature/tier entries in the earlier
four-update regression report are unchanged. Existing core, baseline and shared
source/configuration files remain unchanged in this stage. See the
[validation manifest](../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/validation.json)
for checks and source identities.

## Aligned channel-mixer ablation

The optional `channel_mixer_layers` setting defaults to zero. Enabled mode
scatters visible temporal states by original patch index, attends across
observed channels and pools to separately named contextual exports. Local/global
outputs retain their independent computation. Reconstruction uses exactly the
contextual D32 vector; absent channels remain zero/invalid. Raw/scaler schemas
are reusable, and checkpoint/export semantics identify the active architecture.
The [implementation and measured comparison](CHANNEL_MIXER_ADVANCE.md) document
lessons from the baseline's contextual mixing and token-level decoder.

The fixed 12-run comparison contains 84 measured pairs with full test coverage.
Primary lag/PCA36 ridge accuracy is 90.63%, 76.56% and 63.28% for the mixer,
versus 52.34%, 48.44% and 52.34% for the independent model. All three
mixer-minus-independent nominal within-run intervals are positive. Native
nonlinear results vary, including a negative seed202 effect; this is compact
linear accessibility evidence in this synthetic recipe, not universal superiority.
The mixer has additional parameters/compute and its pre-mixer local branch
shares contextually trained weights without a separate objective.

Container checks pass original-position alignment, semantic permutations,
observed-support safety, hidden-value/gradient isolation, exact export decoding,
both adapters, exact CPU resume and CPU/CUDA CLI operation. A genuine saved
pre-mixer checkpoint retains its exports within the existing `1e-6` tolerance
and resumes one additional update; original archives are byte-preserved.
Contextual saved-model parity has maximum discrepancy `1.19209e-7` with restored
trainable flags and zero with matched frozen flags. All 168 prior independent
and control feature/tier records, all twelve source/value/mask manifests and
the 25-entry baseline/independent regression remain exact. A fixed seed202 lag
repeat is exact. All 68 other checked preexisting code/configuration files,
including every baseline/shared file, remain unchanged. Ordinary model and
minimum evaluator binary dependencies remain separate. See the
[validation manifest](../../output/runs/rpb-implementation/channel-mixer-dc13a548aa/validation.json).

## Fresh seeds and fixed-readout stress

The [completed fresh-seed record](FRESH_SEEDS_AND_STRESS.md) retains a pre-score
recipe and 20 task/seed runs with five new seeds. Primary compact lag ridge
means are 79.69% versus 48.75%; four of five intervals are positive, while seed704
crosses zero. The pooled-global result is weaker. The 240 testing-only stress
views reuse the same trained models/scalers/PCA/readouts; strong deletion and
temporal gaps erode lag accuracy despite high observed coverage. Complete channel
concatenation correctly abstains if any required channel is absent, and all
signal surfaces abstain when every observation is absent.

Shared stress primitives, orchestration and both RPB adapters pass in the managed
container. Entire seed502 base/stress JSON repeats exactly. Previous seed202's
30 feature entries/seven pairs and the baseline's 25 entries/three pairs remain
exact; ordinary reports/fitted assets are unchanged with stress on/off. All 56
encoder code/test/config/script files remain unchanged, and 1,685 prior artifact
files are byte-preserved. The shared sweep also validates the first encoder.
See the [validation manifest](../../output/runs/rpb-implementation/fresh-stress-24846234a7/validation.json).

## GPU learning curve

The [completed duration experiment](LEARNING_CURVE.md) verifies actual CUDA
parameters, inputs, losses, optimizer updates and saved-checkpoint reconstruction
before training on the existing RTX A2000 GPU. Six continuous paths complete
12,288 total updates with immutable checkpoints at 0/128/512/2,048. The shared
driver chooses 128 from validation before generating fresh test cohorts.

Primary compact lag ridge validation means for the mixer decline from 76.82%
to 70.31% to 60.42%, although fixed-target validation reconstruction improves.
Fresh selected-budget compact ridge means are 72.92% versus 51.82%, with two
positive conditional paired intervals and one crossing zero. Native-width
nonlinear probes show that longer training does improve some lag access;
pooled-global access remains weak. This does not support training duration alone
as the remedy for the compact/global benchmark.

Shared generation/curve checks and both-mode CUDA adapter tests pass. An
[independent audit](../../output/runs/rpb-implementation/learning-curve-49fd5f8227/validation.json)
passes 12,132 assertions covering counters, actual CUDA placement, immutable
snapshots/readouts, source separation, selection chronology and full expected
coverage. Existing encoder code/configuration and earlier fresh/stress reports
remain unchanged. Evaluation ownership stays in shared code; new CUDA training
and checkpoint callbacks stay in the RPB adapter. This synthetic development
card does not establish consumer acceptance or uncertainty over retraining.

## Optional global bottleneck and compression diagnostics

The [experiment record](GLOBAL_BOTTLENECK_ADVANCE.md) adds
`global_bottleneck_mode=1` (valid-channel mean) and `2` (semantic-ordered learned
pooling). Their sole reconstruction input is the exact served global `[B,D]`
vector. Mode 0 preserves the previous per-channel route, module initialization,
canonical settings and checkpoint semantics. Six archived mode-0 checkpoints
reproduce 24 exact training/validation feature surfaces; their artifacts are
byte-preserved. Core/workflow and all six mixer/global CUDA continuation
combinations pass, including frozen checkpoint reconstruction parity. The old
per-channel reconstruction card explicitly rejects the new global routes.

Shared projection and bottleneck protocol tests pass without encoder includes.
The archive-only projection diagnostic completes all 18 inputs, four widths and
three repetitions. A fresh CUDA comparison completes 36 continuous paths,
18,432 updates and 108 checkpoints, selecting 128 from validation before fresh
testing. Learned-global PCA12 lag ridge means are 78.65% versus 66.67% for the
current mixer; two conditional paired intervals favor learned and one crosses
zero. Native32 nonlinear means are 80.73% versus 57.29%. Reconstruction worsens,
and heavy corruption remains problematic. The independent main audit passes
153,929 assertions. These observations do not promote a default or establish
consumer acceptance. Both subsequent source cohorts finish 512 updates with
exact nested training and shared held-out/stress arrays. Learned pooling with
128 pairs reaches 100% native-global nonlinear and 98.70% PCA12 nonlinear lag
accuracy, while PCA12 ridge remains 76.56%. Fixed 64-row probe refits do not
show a uniform encoder/scaler-only source-diversity benefit. The full three-run
audit passes 433,165 assertions. No auxiliary objective or default promotion is
introduced; the record retains every comparison and tradeoff.

## Remaining evidence and extensions

The minimum runner is an initial development driver. The shared policy's complete
acceptance process still needs consumer task definitions, coverage/abstention
requirements, independent confirmation seeds and paired encoder differences.
It does not select hyperparameters or assert superiority from its final test
scores. A surface with insufficient valid training rows cannot support a probe;
the original driver rejected such a fit, while protocol v2 reports unsupported
fits explicitly. Wider real-data missingness/coverage protocols remain future work.

Held-out reconstruction reliance against a separately trained metadata-only
decoder and support-compatible shuffled vectors is measured for the fully
observed controlled track. The feature tracks now cover light coordinate
missingness and a fixed-readout structured synthetic sweep. Domain data and
confirmation still require separate declared experiments.

The aligned channel-mixer development ablation is complete. The next evidence
needs consumer confirmation contracts/data and resource measurements before a
consumer decision. Five fresh development seeds and fixed-readout structured
missingness are measured. Its contextual decoder
uses the tested exact-export contract; the independent model's earlier
quantitative decoder-reliance results do not automatically apply to this variant.

The current core exports local/global observed summaries and optional observed
contextual summaries. Inferred missing-channel summaries, multiscale/frequency features,
prediction heads, EMA/JEPA/VICReg objectives, external pretrained controls and
specialist banks remain separate experiments. The original architecture
specification describes that broader direction.

## Historical milestone: balanced context views

**RPB-v8 — Global bottleneck with balanced context views** is implemented and
measured under the frozen [balanced-view card](../../evaluation/cards/context_balanced_validation_v1.md).
It alternates 256 ordinary and 256 deletion-0.30 attempts at 512 completed updates,
with unchanged mode2/mixer1/native32 architecture, original targets/loss and heads.
The [five-master diagnostic](CONTEXT_BALANCED_VALIDATION_DIAGNOSTIC.md) reports
mean native linear accuracy of 95.93750% intact and
91.71875% under the exact saved v7 additional-deletion view,
at 100% coverage. Mean TRAIN/VALIDATION MAE is
0.085022/0.087620.
The predeclared joint development guard **failed**. Independent archive audit v2
passed 67,099,959 checks; actual CUDA admission and prior-policy regressions passed.
V8 is not promoted, and RPB-v4 remains active. No TEST/stress, checkpoint selection,
head tuning or post-encoder PCA occurred. Five new v8 trajectories use ten
retained v4/v7 references; all five masters and both heads remain in the report.

Earlier sections retain their historical recipes and chronology. See the
[durable result summary](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/results/context_balanced_validation_v1.json)
for this completed known-VALIDATION diagnostic.
