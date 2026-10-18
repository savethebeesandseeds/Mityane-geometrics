# Evaluation separation and policy review

Historical review of the evaluator refactor and its original development tracks.
For the configurable RPB-v20 API and latest completed capacity quality
record, read [current status](CURRENT_STATUS.md) and the
[evaluation integration guide](../code/evaluation/README.md). RPB-v4 remains
the formal evaluated reference; the dated validation below retains its original
scope and is not a fresh audit of every later protocol.

Date: 2026-10-07

The agreed [reporting standard](RESULTS_REPORTING_STANDARD.md) and
[version registry](EMBEDDING_VERSIONS.md) retain RPB-v4 as the formal evaluated reference.
Primary comparisons use native exports without PCA afterward, fixed
heads, and standalone raw/PCA controls. This is policy version 1.2. Existing
drivers/cards remain pinned to their historical protocols. The new shared
[archive readout](../code/evaluation/cards/archive_readout_v1.md) passed its
focused container tests and completed the
[native baseline comparison](../code/encoders/raw_patch_bottleneck_mae/NATIVE_BASELINE_COMPARISON.md)
using TRAIN/VALIDATION inputs only, with fixed heads and no encoder retraining.
The separate [native-only training/selection curve](../code/encoders/raw_patch_bottleneck_mae/NATIVE_CURVE_ADVANCE.md)
is now implemented and measured under native-curve-v1. Focused generic and CUDA
gate tests passed; independent artifacts passed 70,619,069 checks. Selection
precedes all fresh TEST generation, controls fit once, and checkpoint/readout
assets are reused without TEST/stress fitting. The
[next advance](../code/encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md) records
the measured gaps and completed fixed training-view diagnostics. This is development
evidence; consumer acceptance contracts remain unimplemented.

RPB-MAE is an encoder governed by the shared
[evaluation policy](EMBEDDING_EVALUATION_POLICY.md). Its architecture-specific
tests validate its declared behavior and optimization. Frozen-feature utility,
comparators, budgets, support populations and decisions belong to evaluation.

The initial implementation shared the probe math but placed orchestration in
the RPB folder, instantiated the baseline there, and linked that driver into
RPB training/export. The refactor separates those responsibilities:

| Boundary | Current owner |
| --- | --- |
| Raw patch encoder, masked reconstruction, training/checkpoints | RPB model/workflow |
| Baseline model and historical evaluator | Baseline folder, preserving its existing behavior |
| Exact preprocessing, model state and frozen exports | Each encoder's `evaluation_adapter.h/.cpp` |
| Exact bottleneck decoding and independently fitted metadata control | Encoder-owned `reconstruction_adapter.h/.cpp` |
| Label-free fit/extraction interface | [Shared provider contract](../code/shared/include/embedding/shared/feature_evaluation.h) |
| Splits/cards, fitting, paired populations, scores and reports | [Shared engine](../code/shared/src/feature_evaluation.cpp) and feature-harness math |
| Native archived exports versus raw/PCA-only controls, with TRAIN-only readouts | [Shared archive readout](../code/shared/src/archive_readout.cpp), no encoder dependency |
| Native continuous training/selection, fit-once controls, fresh TEST and fixed-readout stress | [Shared native curve engine](../code/shared/src/native_curve.cpp); label-free curve/reconstruction factories, no encoder dependency |
| Actual CUDA/checkpoint/exact global bottleneck correctness gate | [Encoder-owned native gate](../code/encoders/raw_patch_bottleneck_mae/src/native_curve_gate.cpp) |
| Testing-only missingness masks and frozen-readout robustness sidecars | [Shared stress engine](../code/shared/src/feature_stress.cpp) |
| Held-out targets, latent interventions, reconstruction metrics and exchange-block effects | [Shared reconstruction engine](../code/shared/src/reconstruction_evaluation.cpp) |
| Registered adapters and CLI options | [Independent evaluation integration](../code/evaluation/README.md) |

The ordinary RPB binary no longer links the baseline, probe library or evaluation
runner. Comparisons run through the separate evaluator directly. Shared engine
tests use dummy providers and compile without
encoder include paths; this demonstrates that reuse is an actual dependency
boundary.

## Policy alignment

| Policy area | Implemented boundary or remaining scope |
| --- | --- |
| Evidence layers/ownership, sections 1–2 | Model correctness and optimization remain encoder-owned; development reports are distinct from acceptance |
| Cards, section 3 | Instantiated versioned cards own geometry, dimensions, tasks, seeds and compared-method pairs |
| Correctness gates, section 4 | RPB gates remain in its tests; adapter outputs validate features/support/schema; each new encoder needs its applicable gates |
| Fitting isolation, section 5 | Fit factories receive training observations/metadata only; labels, clean references and held-out observations stay in the engine |
| Surfaces/dimensions, section 6 | Typed global/concatenation/control declarations; semantic order/support persisted; matched widths belong to the card |
| Controls/probes, section 7 | Common raw/mask controls and fixed probes; adapter-owned descriptors/untrained/fresh-frozen controls; missing historical association disclosed |
| Coverage, section 8 | Unsupported fits, conditional/full-population stress scores and coverage recorded; pair-specific common populations do not expand when unrelated providers are added |
| Diagnostics, section 9 | Shared valid-row diagnostics; RPB structural tests and the separate held-out development decoder-reliance track are validated |
| Uncertainty, section 10 | Paired source-group feature effects and two-source exchange-block reconstruction effects; across-training-run confirmation analysis remains pending |
| Acceptance/cost, section 11 | Not implemented by this development driver; no accepted outcome may be inferred |
| Domains/external tracks, section 12 | Interface supports new adapters; domain datasets/cards, future labels and external overlap audits still require implementation |
| Artifacts/maintenance, section 13 | Policy/protocol/card versions and frozen assets persist; v1 archived reports remain unchanged |

Domain-specific outage and subgroup/full-population acceptance rules,
true per-channel probes, cost measurements, preregistered confirmation thresholds
and hierarchical uncertainty remain incomplete. Reusable evaluation does not mean every policy
requirement has already been implemented.

The new development driver is protocol v2 because surface names, matched-tier
PCA, unsupported-fit reporting and compared populations changed. These changes
cannot be applied retroactively to the original v1 measurements.

## Reuse contract

A new embedding supplies an adapter returning named feature arrays, validity,
surface semantics and provenance. Evaluation owns the legal data/support, card,
split/fitting/probe rules and reports. It does not need the encoder's loss
function, teacher state or internal token layout.

Use [the integration guide](../code/evaluation/README.md) to add an adapter and
run either encoder under the same declared card. Registration is an explicit
integration step; current CLI registration contains baseline and RPB-MAE.

## Validation

Validation ran in the existing managed development container with the pinned
LibTorch bundle. The source was frozen after coordinating the concurrent
baseline namespace cleanup.

| Check | Result |
| --- | --- |
| Shared engine, dummy providers only | Passed: label-free fit contract, typed arbitrary names, stable splits/pairs with an extra mutating zero-valid provider, unsupported fits/rank, archive fidelity and declared-version rejection |
| Shared fitting/protocol math | Passed |
| Final baseline regression and extraction parity | Passed in `implementation-baseline`: model, workflow/resume, numerics, masking, historical evaluator, shared helpers and original-reference parameters/exports/masks/losses/gradients |
| Final RPB core/workflow regression | Passed in `rpb-implementation`: preprocessing/numerics/masking/model, exact CPU resume, scaler/schema/archive/export, skip recovery and path preservation |
| Binary dependency inspection | Passed: ordinary RPB contains no baseline/probe/shared-engine symbols; minimum evaluator contains no RPB symbols |
| Both adapters, four tasks, three seeds | [Completed 12 runs and 36 measured pairs](../output/runs/rpb-implementation/rpb-evaluation-9nX9MV/results/report.json); four fresh updates per encoder, wiring/development evidence only |
| Minimum evaluator without RPB | [Passed four tasks](../output/runs/rpb-implementation/minimum-3KYXIk/results/report.json): four fresh baseline updates; saved provenance matches finalized sources |
| Frozen RPB checkpoint | [Passed](../output/runs/rpb-implementation/rpb-evaluation-YYkgwc/results/report.json): saved model/scaler used, no evaluation refit |
| Independent input geometry | [Passed](../output/runs/rpb-implementation/rpb-evaluation-fRIgiE/results/report.json): both adapters on C=1/F=1, semantic ID 17 and units `volts` |
| Repeated independent-card run | [Passed](../output/runs/rpb-implementation/rpb-evaluation-vuYsqx/results/report.json): report, card and text manifests/provenance match exactly |

The completed feature-refactor report's source fingerprint is
`45985efcbd9aa9088d680702ec32bfc51fd0d834140ea4ee7c180a1bf4bdf2ae`;
ordinary RPB at that stage was
`7bc08d72f3e4e52827edef26bf487a66e5f61839336741195e828dd8732756d4`.
The full comparison measured 288 feature/tier entries and reported 12 unsupported
mask-only 36-dimensional PCA fits explicitly. These are support failures, not
zero-scored learned outputs.

Baseline/control measurements and all observed-data/mask split checksums were
identical with and without the RPB provider. The final minimum evaluator's source
fingerprint is
`bdd099da204b3136195f91700603c2a508806375abe7285f24248de9250b29e6`.
Its provenance-bearing objects were refreshed after a concurrent source-edit
overlap; these final hashes were checked against the current source manifests.

The concurrent source-ownership cleanup separately checked the relocated
baseline CPU/CUDA train/resume/export, CPU serving of CUDA checkpoints, periodic
recovery and historical evaluation runner. Its
[validation record](../output/runs/source-ownership/validation.json) also records
old-checkpoint schema/mask equality and float32 exports within `rtol=1e-5`,
`atol=1e-6` (maximum absolute difference `2.3842e-7`), rather than bitwise export
equality with the older binary.

These generated artifacts are retained locally under `output/` and excluded from
Git. A clean checkout needs to reproduce the runs to resolve those report links.
The feature checks establish reusable evaluation wiring and regression
protection; they do not establish representation improvement or consumer acceptance.

## Reconstruction development track

The [controlled-reconstruction-v1 card](../code/evaluation/cards/controlled_reconstruction_v1.md)
adds a separate reusable contract for visible-observation encoding, exact
compact-vector decoding, raw-unit predictions and a separately fitted
metadata-only control. Fit receives training observations/metadata only. The
shared engine owns hidden targets, disjoint source swaps, zero interventions,
training-fit metric scaling, common support and within-run block uncertainty.
The ordinary RPB binary and minimum harness do not link this engine.

The [12-run report](../output/runs/rpb-implementation/reconstruction-2jehCR/results/report.json)
uses 128 completed updates for both primary and metadata-only models,
four tasks and three development seeds. All 24 test comparisons against
metadata-only decoding or shuffled vectors favor the intact vector with
positive 95% paired intervals. Every testing run has 64 independent source
groups, 32 exchange blocks and full target coverage. This is decoder reliance
on the fully observed synthetic track, not evidence that lag-sign probes,
real-data usefulness or consumer acceptance improved.

Container checks passed the shared dummy-provider engine, RPB adapter assets,
raw/scaler/configuration identity and restored checkpoint trainability. An
ordinary CLI checkpoint resumed one additional update and exported embeddings.
Loaded trainable float32 inference has maximum discrepancy `7.45058e-8` from
the frozen provider; matching gradient flags gives exact parity. The existing
workflow tolerance is `rtol=1e-6, atol=1e-6`. The metadata encoder's parameters
remain exactly equal to matched initialization.

The repeated small recipe has identical reports/cards. Existing feature
evaluation still passes in the [full evaluator](../output/runs/rpb-implementation/rpb-evaluation-HjyLml/results/report.json)
and [minimum harness](../output/runs/rpb-implementation/minimum-BsVx20/results/report.json).
All 68 checked preexisting model/workflow/configuration/shared source/test files
are unchanged. Binary symbol inspection retains the ordinary-RPB and minimum
dependency boundaries.

At the reconstruction stage, full evaluator source was
`23a5d59409ea250fa3cfe3245d9357a659a7c5dcea9c0d9b6b5943211d81255f`;
ordinary RPB is
`b154a385a95edcf350b2d0dce38c1c8cb35f56003be28bc535bd4dbf89bf187a`;
minimum harness is
`f67a7208f8aa53cc0ec5c585e658d7818283c23a2ea54a7b92b5b08e8bab5e5b`.
The ordinary checkpoint fingerprint identifies its core writer. Companion
training provenance records the full adapter producer and recipe separately.
The [validation manifest](../output/runs/rpb-implementation/reconstruction-validation-HZp50W/validation.json)
records source manifests, preserved-file hashes and retained checks.

Domain outage contracts, costs, domain datasets, consumer thresholds and across-training-run
confirmation remain pending.

## Fixed-readout stress development track

The [versioned stress card](../code/evaluation/cards/fixed_readout_stress_v1.md)
adds optional testing-only coordinate deletion, temporal blackouts, semantic
channel outages and all-absent views to any registered feature provider.
`--stress-sweep fixed-readout-v1` retains the ordinary trained model/scalers,
feature normalization, PCA and readouts; there is no refitting under corruption.
The generic sidecars report exact masks/retention/source groups, per-channel
support, conditional accuracy, full-population correctness including abstentions,
and paired conditional/full-population effects. Base v2 reports retain their
original meaning and default behavior.

The [five-fresh-seed RPB record](../code/encoders/raw_patch_bottleneck_mae/FRESH_SEEDS_AND_STRESS.md)
completes 20 task/seed runs and 240 stress views with unchanged 128-update recipes.
Compact channel-concatenation lag ridge means are 79.69% versus 48.75%; four
primary intervals are positive and one crosses zero. Pooled-global evidence is
weaker, and heavy deletion/temporal gaps erode the benefit. This is development
replication, not consumer confirmation.

Managed-container tests verify immutable fits, intact prediction parity,
fit/save-once provider behavior, clone isolation, source-shared label-independent
nested masks, hidden-value zeroing, unsupported base fits and null conditional
scores without support. Both original and RPB encoders run through the same
generic sweep. Ordinary baseline/independent measurements, checkpoints and
fitted probe assets remain exact with stress enabled; the earlier mixer run and
a fresh full stress replay retain exact scores/populations/sidecars. All 56
encoder code/test/config/script files remain unchanged and 1,685 archived
artifact files are byte-preserved. See the
[validation capsule](../output/runs/rpb-implementation/fresh-stress-24846234a7/validation.json).

Domain outage definitions, consumer coverage/abstention thresholds, resource
measurements and across-training/multiplicity confirmation remain pending.

## Stronger frozen-feature development track

The [RPB frozen-probe record](../code/encoders/raw_patch_bottleneck_mae/FROZEN_PROBE_ADVANCE.md)
contains a recipe frozen before generation/fitting and a completed 12-run,
48-pair comparison. Each fresh RPB model receives 128 updates, and its own
untrained control shares initialization and permitted training-fit scaling.
The primary PCA36 channel-concatenation lag-sign ridge effects have intervals
crossing zero in all three seeds. Diagnostic native96 concatenation with the
fixed nonlinear probe exposes lag information and yields positive within-run
training-effect intervals. This establishes neither a compact linear advantage
nor consumer acceptance. All pairs have complete coverage; raw oracles score
100%, mask-only measured probes score 50%, and twelve rank31 mask PCA36 fits
are explicitly unsupported.

The generic feature engine already owns both probe-family paired effects;
no shared/model math changed. The RPB feature adapter now records resolved
training settings, actual fit seeds, attempted/completed counts and producer
identity. Initialized weights explicitly record no pretraining/zero updates;
their permitted scaler fit is separate provenance. Supplied checkpoints retain
their own settings and disclose unrecorded historical training producers.
Frozen feature assets are distinct from ordinary resumable checkpoints.

The experiment's companion plan captures the full configuration and overrides
before fitting, because the generic card alone omits provider training budgets.
The new adapter test and shared feature/reconstruction tests pass. A seed202
lag repeat has identical per-run reports/manifests, and the earlier four-update
regression retains all 25 feature/tier entries exactly. Hash checks preserve
80 other preexisting code/configuration files; the adapter is the only modified
existing C++ source. The Makefile adds its dedicated test target.

For this stage, full evaluator source is
`1254bf5782d43abe86f47ea12fe15e14efee09af02b9de0c2eb0de25b54fb188`;
ordinary RPB is
`b5a6bb8bd127e98fb4ce0105a107bb70b3ae2af6f8283cd762c412b485f09b9f`;
minimum harness is
`3c99b6a12ec436650c38cdbd072094e74d721d4fa231379aee43d737e19b86a2`.
The [validation manifest](../output/runs/rpb-implementation/frozen-probe-8b6e501bee/validation.json)
retains artifacts and checks. Across-training uncertainty, multiplicity-adjusted
confirmation and real data remain open. The subsequent fixed-readout track
above covers structured synthetic outages.

## Aligned channel-mixer development variant

The [mixer record](../code/encoders/raw_patch_bottleneck_mae/CHANNEL_MIXER_ADVANCE.md)
adds optional original-patch-aligned channel attention to RPB, preserving its
independent outputs and exact compact-vector decoder. Evaluation registers the
same adapter twice with distinct `rpb`/`rpb_mixer` namespaces. Contextual
global/concatenation surfaces declare observed support and semantic order;
the shared split/PCA/probe/scoring engines are unchanged.

The frozen comparison completes twelve runs and 84 measured pairs. The primary
PCA36 lag-sign ridge accuracies are 90.63%, 76.56%, 63.28% for the contextual
variant versus 52.34%, 48.44%, 52.34% for the independent control. All three
nominal within-run paired intervals favor the mixer. Native nonlinear gains
vary; parameters/compute are not matched. This supports compact linear
accessibility in the controlled development recipe, without consumer promotion
or a baseline-superiority claim.

Container checks pass both modes' model/support/gradient contracts, workflow
resume/export, feature/reconstruction adapters and CPU/CUDA CLI operations.
A preserved pre-mixer checkpoint re-exports within the existing `1e-6` tolerance
and resumes while originals remain byte-identical. All 168 prior independent/
control feature records and twelve source/value/mask manifests are exact;
the 25-entry baseline/independent regression and seed202 lag repeat are exact.
All checked baseline/shared code is unchanged. Ordinary model/minimum binary
symbol boundaries remain intact, and the minimum binary rejects mixer registration.

This stage's full evaluator source is
`45b298fe18183a0ba121cba8a97cc03c5c8155d137405dffcd2a2d0bb8d6031e`;
ordinary RPB is
`dfdbe30014906bab141b8a1ab89d06f86ff82847c15834affd312adc4108292a`;
minimum harness is
`f46adc78331e6bc329f74eba55bd433d909be7be927d7b2c6879de1b521d7201`.
The [validation manifest](../output/runs/rpb-implementation/channel-mixer-dc13a548aa/validation.json)
retains recipe snapshots, provenance and checks. Quantitative contextual decoder
reliance, confirmation contracts/data and resource costs remain open. Structured
synthetic outages are measured in the subsequent fresh-seed/stress record above.
