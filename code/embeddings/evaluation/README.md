# Reusable embedding evaluation

Completed-run audit and repair utilities live only in the immutable
[historical tools archive](../../../archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/tools/).
The six legacy protocol entry points that published repository-global summaries
or registry changes are retired. Their original implementations remain archived;
new publication needs its own versioned protocol and destination. Capsule-local
publication arithmetic remains available to its owning protocols.

Start with [the current research guide](../../../doc/EMBEDDINGS.md). Encoder
research remains paused at the saved V14 handoff; this directory's historical
recipes and continuation notes do not resume experiments or metadata work.
The maintained layer composes encoder adapters with shared evaluation engines
in [`code/shared/representation/`](../../shared/representation/README.md).
Forecasting baselines and common-observable readouts are owned separately by
[`code/operations/forecasting/evaluation/`](../../operations/forecasting/README.md).

The inherited protocol descriptions below retain their original scientific
scope. Current execution uses the repository-root Makefile/task runner inside
`mityane-geometrics`; frozen source identities and original records remain in
[the pre-cleanup archive](../../../archive/embeddings/README.md). Historical
reproduction requires its own path mapping and protocol admission.

## Historical protocol context

The implemented RPB-v20 API and its
[configurable capacity record](../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONFIGURABLE_CAPACITY_V1.md)
belong to an earlier synthetic development stage.
The [capacity producer/reader](protocols/configurable_capacity_v1/README.md)
is an experiment-specific entry point separate from the historical
`embedding_evaluate` registry. It trains v20 on CUDA and reuses this repository's
shared fixed heads on saved native features; it imports no market loader.
The [v20 instance extension](../../../archive/embeddings/pre-cleanup-2026-10-18/doc/configurable_capacity_embedding_versions.json)
identifies the exact measured bundles. Arbitrary geometry and market quality
do not follow from that synthetic record.

This folder composes encoder adapters with the encoder-independent evaluation
engine. The [evaluation policy](../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_EVALUATION_POLICY.md) defines
the evidence and acceptance rules; the
[implementation review](../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EVALUATION_IMPLEMENTATION_REVIEW.md) records
which requirements are implemented and which remain pending.

New result summaries follow the
[reporting standard](../../../archive/embeddings/pre-cleanup-2026-10-18/doc/RESULTS_REPORTING_STANDARD.md) and
[version registry](../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md). Primary encoder rows use
native exports without PCA afterward; the standalone raw-data row is labelled
"PCA only — no encoder". RPB-v4 is the formal evaluated reference within those
historical protocol versions. The [historical advance plan](../encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md)
preserves the development chronology. Its validation-only
[archive protocol](cards/archive_readout_v1.md) and
[native-only curve](cards/native_curve_v1.md) provide separate historical
training/selection protocols. Historical drivers
still compute their compressed tiers, and the global driver selects using
PCA12. A native-looking table does not change that selection rule. Older cards
and results remain governed by their original declared versions.

The shared [paired comparison](cards/paired_pooling_v1.md) engine also hosts
[context-deletion-v1](cards/context_deletion_v1.md) through encoder-owned training
and checkpoint adapters. It reuses frozen reference/control assets, owns the
new TEST namespace and corruption populations, and never fits on TEST or stress.
The completed [optimization diagnostic](cards/optimization_validation_v1.md)
uses only TRAIN and known VALIDATION through the shared archive-readout engine.
The completed [context optimization diagnostic](cards/context_optimization_validation_v1.md)
adds a fixed deletion view of known VALIDATION. Each ordinary TRAIN-fitted head
scores both views without additional fitting. Its deterministic replay and live
optimizer continuation stay in the encoder adapter; shared evaluation owns the
view association, support, predictions and paired scores. It selects no checkpoint
and accesses no TEST or historical stress payload.

The completed [context replication](../encoders/raw_patch_bottleneck_mae/CONTEXT_REPLICATION_ADVANCE.md)
uses [context-replication-v1](cards/context_replication_v1.md). Shared
`native-development-v1` first prepares five fresh v4 TRAIN/VALIDATION references
without TEST or selection, then the paired engine compares fresh v6 trajectories
with those exact frozen assets before opening one new TEST namespace. Mean
linear gains are +1.71875 percentage points intact and +8.125 under additional
30% deletion, with equal 100% coverage; both mean TRAIN/VALIDATION reconstruction
guards fail. RPB-v4 remained the reference for that comparison. Independent archive audit v3 passed
103,787,020 checks, with the failed v1/v2 reader attempts and correction rationale
preserved.

The completed [RPB-v7 validation diagnostic](../encoders/raw_patch_bottleneck_mae/CONTEXT_LIGHTER_VALIDATION_DIAGNOSTIC.md)
uses the frozen [lighter-policy card](cards/context_lighter_validation_v1.md)
and separate `rpb-training-context-deletion-015-v1` encoder policy at 512 updates.
Five new trajectories are compared with ten retained v4/v6 trajectories on known
TRAIN/VALIDATION only. Native linear means improve over v6 on both views, but
both reconstruction guards against v4 fail; neural means decline and a master
loses deletion accuracy. V4 remained the reference for that experiment. Independent archive audit v3 passed
67,097,380 checks.

The integration CLI pins 313 explicit parent TRAIN/VALIDATION roles and passes
ordinary native32/raw/PCA-only exports to unchanged shared archive-readout fits.
There are 45 newly fitted pipelines per method, each containing both heads;
45 native fit/prediction parity checks pass against the named saved assets.
Each TRAIN-fitted pipeline scores intact VALIDATION and one fixed new deletion
view without refitting. Encoder policy, CUDA optimizer and checkpoint semantics
stay in the encoder adapter; shared evaluation owns maps, heads, support,
predictions and scores. No TEST/stress, checkpoint selection or promotion occurs.
Historical cards and source capsules remain unchanged in the archive.
Maintained source/reference adaptations have new source identities.

The frozen v7 runner requires the named local parent capsule and never
regenerates its historical TEST/stress. Its original root was `/embedding`;
current source commands run from `/mityane` after the frozen parent paths are
explicitly mapped:

```bash
bash code/embeddings/scripts/check-context-lighter-validation.sh
bash code/scripts/task.sh rpb-paired-pooling evaluate-context-lighter-validation
```

The admission script owns its named task calls; run it directly rather than
nesting it inside `task.sh`. The evaluation target freezes compiled source and
admission evidence and allocates a unique capsule. It accepts no recipe overrides.

The sections below document historical driver contracts and completed stages.
Use each frozen card's version and evidence for reproduction, and the current
research guide above for pause status. Do not repeat a completed encoder
run to recover already saved scores.

## Ownership

| Location | Responsibility |
| --- | --- |
| `code/shared/representation/feature_evaluation` headers/source/tests | Provider contract, versioned cards, orchestration, fitted assets, paired comparisons and reports |
| `code/shared/representation/feature_harness` headers/source/tests | Controlled tasks and legal oracles, valid-row scaling/PCA, frozen probes, scoring and diagnostics |
| `code/shared/representation/feature_stress` headers/source/tests | Testing-only missingness masks, fixed fitted-readout predictions, conditional/full-population scores and robustness sidecars |
| `code/shared/representation/reconstruction_evaluation` headers/source/tests | Held-out targets, latent interventions, training-fit metric scaling, paired reconstruction errors and source-exchange uncertainty |
| `code/shared/representation/native_curve` headers/source/tests | One continuous label-free trainer, native checkpoint selection, retained raw/PCA-only controls, exact point-zero comparison and frozen TEST/stress readouts |
| `code/embeddings/encoders/<encoder>/evaluation_adapter.*` | Exact model/checkpoint/preprocessing, optional training on permitted observations, frozen extraction, surface/support semantics and adapter assets |
| `code/embeddings/encoders/<encoder>/reconstruction_adapter.*` | Training, exact served latent decoding, independently fitted metadata control and checkpoint assets for reconstruction providers |
| `code/embeddings/evaluation/src/main.cpp` | Explicit registry and CLI composition; currently baseline and RPB-MAE |
| `cards/` | Human-readable protocol definitions; each run saves its instantiated card |

The shorthand shared paths above refer to their files under
`include/embedding/shared/`, `src/` and `tests/`. Shared sources compile without
either encoder's include directory. Encoder training loss, optimizer and teacher
logic stay inside their adapter/model workflow; probe fitting and labels stay
inside the evaluator.

The native-curve CLI registers RPB-v4 through its existing learning-curve adapter.
Its encoder-owned CUDA gate verifies actual updates and exact32 checkpoint/decoder
parity before a full run. Build and run the fixed development recipe with:

~~~powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-native-curve', '-j2', 'native-curve', 'test-native-curve', 'test-rpb-native-gate')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-native-curve', 'evaluate-native-curve')
~~~

The runner checks compiled source hashes against an exact source snapshot before
generation, saves a launch plan and CUDA gate, then creates a unique results root.
The shared engine saves its instantiated card, TRAIN/VALIDATION cohorts, immutable
checkpoints/fits and native32 selection before generating any fresh TEST cohort.
No PCA follows the encoder. Raw576/PCA-only32 maps and heads remain fixed across
milestones, TEST and the shared twelve-case stress sweep. See the
[advance record](../encoders/raw_patch_bottleneck_mae/NATIVE_CURVE_ADVANCE.md).

~~~mermaid
flowchart LR
  M[Frozen encoder] --> A[Encoder adapter]
  A -->|features, validity, semantics and provenance| E[Shared evaluation engine]
  C[Versioned evaluation card] --> E
  E --> R[Measurements and saved assets]
~~~

## Build and run

Use the existing managed container from the project root:

~~~powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'comparison', '-j4', 'evaluation', 'feature-harness', 'test-feature-harness')
~~~

The `evaluation` target builds `embedding_evaluate` with the registered baseline
and RPB adapters. `feature-harness` builds `feature_harness` with the baseline
adapter only, so minimum controls do not require the RPB model.
`test-feature-evaluation` tests the shared engine using dummy feature providers;
`test-feature-harness` includes those checks and the fitting/math tests.

Inside the container, after the build:

~~~bash
eval_bin=/opt/mityane-geometrics/embeddings/build/comparison/embedding_evaluate
mkdir -p /mityane/output/embeddings/runs
run_parent="$(mktemp -d /mityane/output/embeddings/runs/comparison-XXXXXX)"
"$eval_bin" --output "$run_parent/results" --encoders baseline,rpb \
  --seeds 101,202,303 --train-pairs 32 --validation-pairs 16 --test-pairs 32 \
  --baseline-steps 8 --rpb-steps 20
~~~

`--encoders` selects adapters. `--baseline-config`, `--rpb-config` and
`--rpb-checkpoint` configure only those adapters. Geometry, channel IDs/units,
tasks, matched dimensions and compared methods come from the shared card:

~~~bash
"$eval_bin" --output /mityane/output/embeddings/runs/new-comparison \
  --encoders baseline,rpb --channels 3 --history 32 --features 3 \
  --matched-global-width 12 --matched-channel-width 36 \
  --compare rpb_untrained_global,untrained_baseline_global,matched_global
~~~

The output directory must not exist. Repeat `--compare LEFT,RIGHT,TIER` to declare
additional pairs; the first is the primary development comparison. Without it,
the integration CLI declares named default pairs before generating data.
An unrelated additional provider cannot change an existing pair's validity
intersection. Each pair reports its own common population and effect interval.

The runners `code/embeddings/scripts/evaluate-minimum.sh` and `evaluate-rpb-mae.sh` allocate
unique output directories. The latter now selects the separate evaluator.
Run comparisons through `embedding_evaluate` directly. The RPB model executable
owns prepare/train/embed and has no evaluator dependency or evaluation command.
The original baseline `embedding evaluate` retains its historical protocol.

## Add an encoder

1. Add an encoder-owned adapter returning the shared `FeatureProviderFactory`.
   A fit callback receives `ProviderFitInput`: permitted training observations,
   shape, IDs, units, endpoint/interval, source IDs and seeds. It receives no
   labels, hidden clean signals or held-out observations.
2. Load exact frozen weights/preprocessing or fit only on that permitted input.
   Expose named CPU `[B,D]` features plus bool `[B]` validity through
   `FeatureMap`, with typed `SurfaceDescription` entries and persisted assets.
3. Register that factory in the integration CLI/build. No split, PCA, probe,
   scoring or report implementation needs to be copied into the encoder.
4. Declare meaningful tasks, feature semantics, metadata access, dimensions,
   comparisons and applicable correctness evidence in the card.

Kinds are `global`, `channel_concatenation` and `control` for the current driver.
Concatenation declares its complete semantic channel order and support rule;
local versus contextual constituents are documented by the adapter. Names do
not determine kind, dimensions or eligibility. True per-channel probes and
inferred-support tracks need explicit future protocol extensions.

The generic feature contract can wrap local, external or already exported
embeddings. A model with a different input layout maps the common observations
in its adapter and discloses that mapping. The current executable supports the
four binary controlled tasks in [controlled pairs v2](cards/controlled_pairs_v2.md);
it is not an implementation of every domain task in the policy.

The optional `rpb_mixer` registration uses the same RPB adapter with its own
surface namespace and requires an enabled mixer. Select `--encoders rpb,rpb_mixer`
to fit both variants on the same permitted training observations and score
declared paired comparisons. Its settings use `--rpb-mixer-config`,
`--rpb-mixer-steps` or `--rpb-mixer-checkpoint`. Contextual surfaces have explicit
`*_contextual_global` and `*_contextual_channel_concatenation` names; existing
local surface names retain their meaning. Splits, fitting, probes and scoring
remain in the shared engine. See the [mixer record](../encoders/raw_patch_bottleneck_mae/CHANNEL_MIXER_ADVANCE.md).

## Fixed-readout missingness

Add `--stress-sweep fixed-readout-v1` to a feature comparison to enable the
separate [fixed-readout-stress-v1 card](cards/fixed_readout_stress_v1.md)
development track. The default is `none`.
The engine saves `stress-card.json` before generating data, then fits each
provider, scaler, normalizer, PCA and readout once on the ordinary training
split. All stress predictions reuse those fitted objects. Encoders need no
stress-specific adapter or training objective.

The sweep includes intact observations, additional coordinate deletion at
10/30/60/90%, temporal blackouts covering 25/50/75% of history, each semantic
channel absent separately, and all observations absent. Deletion masks are
nested and shared within each source pair and across providers. Blackouts use
one source-specific temporal anchor with a centered, edge-clamped interval.
Corruption never adds observations; newly hidden raw cells are stored as zero.

`stress-report.json` aggregates the sidecars under each run's `stress/`
directory. Artifacts retain the corrupted batches, source identities and
requested/actual retention. Scores include coverage, conditional accuracy and
full-population correctness with abstentions counted as failure. Each declared
architecture comparison uses its own common valid population and source-group
uncertainty; stress-versus-intact comparisons also use matched populations.
Signal surfaces with no observations abstain. A complete-channel concatenation
abstains when any required channel is absent; a global surface may retain
support from the remaining channels. Mask-only controls remain explicit controls.

Stressed legal-raw oracle scores are descriptive. The ordinary fixture's oracle
gate still applies before stress. Existing v2 reports and fitted assets retain
their meaning; the optional sidecars do not revise archived measurements.
See the [fresh-seed recipe](../encoders/raw_patch_bottleneck_mae/FRESH_SEEDS_AND_STRESS.md).

## Held-out reconstruction

`embedding_evaluate reconstruct` uses a separate shared provider contract and
the [controlled-reconstruction-v1 card](cards/controlled_reconstruction_v1.md).
The initial registered provider is RPB-MAE. Fit receives the same label-free
training input as feature evaluation; decoding exposes raw-unit predictions and
the exact served latent. The shared engine owns fixed hidden targets, source
swaps, zero interventions, metric scaling, support and block bootstrap.

Build/check through the existing container task runner:

~~~bash
bash code/scripts/task.sh comparison -j2 evaluation test-reconstruction-evaluation test-rpb-reconstruction
env EVALUATION_BIN=/opt/mityane-geometrics/embeddings/build/comparison/embedding_evaluate \
  EMBEDDING_RUN_ROOT=/mityane/output/embeddings/runs/comparison \
  bash code/embeddings/scripts/reconstruct-rpb-mae.sh
~~~

The wrapper allocates a unique output directory. See
[decoder reliance](../encoders/raw_patch_bottleneck_mae/DECODER_RELIANCE.md) for
the frozen training budget and measurements. Ordinary model binaries remain
independent of evaluation. The minimum harness rejects reconstruction and does
not link its engine or RPB provider.

## Native archive controls

The separate `embedding_archive_readout` executable (`archive-readout`) links
shared preprocessing, PCA and head fitting only. It accepts explicit TRAIN and
VALIDATION archives through a manifest, so it can evaluate saved exports from
any encoder without importing that encoder or retraining it. It has no TEST
input option. See [archive readout v1](cards/archive_readout_v1.md).

~~~powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'archive-controls', '-j2', 'archive-readout', 'test-archive-readout')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'archive-controls', 'evaluate-archive', 'ARCHIVE_ARGS=--manifest /mityane/code/embeddings/output/PATH/inputs.tsv')
~~~

The runner creates a fresh output directory. The instantiated card precedes
fits; assets, row-level predictions, per-method coverage, paired common-support
effects, source-group intervals and file hashes are retained. Native rows use
the model tag alone; their descriptions sit beside result tables.

The UTF-8 TSV manifest has this exact ordered header (joined by tab characters):
`id`, `tag`, `task`, `master_seed`, `checkpoint_steps`,
`producer_source_fingerprint`, `cohort_provenance`, `training_observations`,
`validation_observations`, `training_features`, `validation_features`.
For identity-bound archives append these five fields:
`training_observations_sha256`, `validation_observations_sha256`,
`training_features_sha256`, `validation_features_sha256`,
`expected_feature_provenance`. All declared fields are nonempty; relative archive
paths resolve against the manifest directory. SHA256 strings use lowercase hex.
`cohort_provenance` records the caller-audited source order, producer checkpoint,
scaler and manifest lineage. Native archives do not themselves contain row IDs.
The C++ API additionally permits explicit archive key mappings.

Raw/PCA use any-observation support. An encoder may declare a smaller supported
population; unsupported native fits do not disable raw controls. Only standalone
raw PCA is fitted, at the declared native width. Fixed head recipes remain the
same; three declared neural initializations are paired across equal-width rows.
These are head repetitions, not extra encoder training runs. All validation
results remain development diagnosis.

## Evidence limits

The separate `embedding_projection_diagnostic` executable (`projection-diagnostic`)
compares PCA and frozen random projections using explicit archived training and
validation features. It links shared fitting code without an encoder.
`embedding_global_bottleneck` (`global-bottleneck`) registers the three RPB
trainer variants with the shared bottleneck experiment. Run its four-update
`--gpu-check true --output NEW_DIRECTORY` gate before full training. The
[experiment record](../encoders/raw_patch_bottleneck_mae/GLOBAL_BOTTLENECK_ADVANCE.md)
fixes the recipe; `test-projection-diagnostic` and `test-global-bottleneck`
validate the shared protocols independently of model implementation.

The [validation-selected learning curve](cards/learning_curve_v1.md) has a separate
`embedding_learning_curve` executable (`make learning-curve`). It preserves one
optimizer through several update budgets, measures fixed train/validation
reconstruction and selects a common budget before generating fresh test sources.
Its RPB adapter requires CUDA training. Run `--gpu-check true --output NEW_DIRECTORY`
first to verify actual GPU updates; then run with another new output directory.
The [RPB record](../encoders/raw_patch_bottleneck_mae/LEARNING_CURVE.md) freezes the
current recipe and contains its evidence.

Reports are development evidence. They save the policy/protocol/card versions,
source identities, fitted scalers/compression/probes, surface semantics,
coverage, unsupported fits and paired grouped uncertainty.
The generic feature card does not serialize a provider's full training recipe.
Save a companion experiment plan with exact configuration content, actual seed
and update overrides, source identities and command before generation/fitting.
The RPB [frozen-probe record](../encoders/raw_patch_bottleneck_mae/FROZEN_PROBE_ADVANCE.md)
contains a completed example. Adapter audits and assets retain the resolved
settings and separate weight-training from preprocessing provenance.
Insufficient valid training support or PCA rank is reported, not repaired by
inventing features or silently changing widths.

Confirmation, consumer acceptance thresholds, domain outage contracts and cost
measurement remain pending. The optional fixed-readout sweep covers the current
synthetic card. Decoder interventions have their own development
track; reconstruction evidence does not establish frozen-feature usefulness.
Architecture-specific correctness tests stay with each encoder. The historical
evaluation refactor introduced protocol v2; earlier v1 reports retain their original
measurements and interpretation.
## V7 decoder-only calibration diagnostic

The separate frozen [card](cards/v7_decoder_calibration_v1.md) asks whether the
existing decoder can recover ordinary reconstruction from unchanged retained
RPB-v7 embeddings. Encoder/scaler fitting stays in the encoder module; this
driver owns the closed85 input roles and shared fixed-query scoring. There are
zero classifier fits, new encoder updates, TEST accesses or checkpoint choices.

Inside the existing managed GPU container:

```bash
bash code/embeddings/scripts/check-v7-decoder-calibration.sh
bash code/scripts/task.sh rpb-paired-pooling evaluate-v7-decoder-calibration
```

Admission uses new synthetic correctness fixtures. Measurement starts from the
five exact saved v7@512 instances and completes 128 fresh decoder-only updates.
All model inference and optimization use CUDA; saved-arithmetic auditing uses
CPU. Typed decoder assets compose with their immutable parents and preserve
original encoder512 and additional decoder128 counters separately. They are
not ordinary checkpoints. The encoder retains the RPB-v7 tag; classification
results are reused only with exact unchanged encoder/native-output witnesses.

## Fresh equal-budget decoder replication

The frozen [fresh decoder card](cards/fresh_decoder_replication_v1.md) compares
RPB-v4 and RPB-v7 on five new TRAIN/VALIDATION source cohorts. Both receive 512
encoder updates and then 128 updates of their existing decoder with the encoder,
buffers and scaler frozen. The encoder module owns both typed calibration paths;
this driver owns the prospective recipe, paired sources, fixed queries and
composition. Its input allowlist contains no historical quality payloads.

The original recipe used `/embedding`. Current source commands run from
`/mityane` only after the frozen parent paths are explicitly mapped:

```bash
bash code/embeddings/scripts/check-fresh-decoder-replication.sh
bash code/scripts/task.sh rpb-paired-pooling evaluate-fresh-decoder-replication
```

Admission checks both parent policies on actual CUDA and the shared fixed-head
fixtures before quality generation. The runner freezes the card, exact source
inventory, binary and admission evidence, then allocates a new capsule. There
are no recipe overrides, TEST/stress inputs or automatic promotion.

`fixed_feature_readouts` accepts supplied TRAIN and two VALIDATION feature
tensors, labels and source groups without importing an encoder. It owns the
unchanged ridge/tanh-16 recipes, TRAIN-only normalization, fitted assets,
predictions, support and grouped intervals. Raw/PCA-only controls are prepared
once per cohort by the driver. PCA is never applied to native encoder features.
Its optional paired comparison preserves the historical `native_v4` and
`native_v7` default. New cards can set explicit `comparison_reference` and
`comparison_candidate` method names, with the same common-support arithmetic
and fixed recipe. Other supplied feature methods receive the same generic
fits and scores. A source fixture verifies that naming a pair differently
cannot change its fitted tensors or predictions.
Both validation views reuse one TRAIN fit. Decoder calibration also reuses those
fits and scores only after exact CUDA feature equality is established; it does
not run another classifier fit or CPU encoder forward.

Build the shared fixture independently with
`bash code/scripts/task.sh rpb-paired-pooling test-fixed-feature-readouts`.
The fresh runner records encoder training, decoder loops, feature/query inference
and the CPU readout loop separately. Decoder loop timing includes evidence
capture, so it is descriptive elapsed time rather than a pure GPU cost benchmark.

## Frozen native transfer

The [frozen amplitude card](cards/frozen_amplitude_transfer_v1.md) evaluates
new amplitude cohorts using retained timing-trained encoders and new fixed
classifiers. The encoder-owned `make_frozen_native_feature_provider` returns
only the shared `FeatureProvider`: strict checkpoint/scaler admission and CUDA
native32 extraction stay in the encoder; task generation, raw/PCA controls,
heads, views and scoring stay here. It exposes no training or reconstruction
callback. Native features receive no PCA and original timing scalers remain fixed.

The [completed diagnostic](../encoders/raw_patch_bottleneck_mae/FROZEN_AMPLITUDE_TRANSFER_DIAGNOSTIC.md)
passed independent saved-arithmetic verification. Original TRAIN exports were
checked exactly on CUDA before any new amplitude generation. Required exports
are saved once for shared CPU readout arithmetic; no CPU encoder executes.

```bash
bash code/embeddings/scripts/check-frozen-amplitude-transfer.sh
bash code/scripts/task.sh rpb-paired-pooling evaluate-frozen-amplitude-transfer
```

Each measurement uses an exclusive source/admission/card/reader-bound capsule.
The sealed reader and parent role matrix are required local evidence; this
runner does not silently regenerate or replace historical parents.

## Early channel mixing

The [early-mixer card](cards/early_mixer_reliability_v1.md) composes encoder-owned
continuous CUDA trainers and immutable snapshots with the same shared readouts.
It compares fresh RPB-v7.alt-02 and RPB-v10, both native32, at512 updates.
Two initial controls distinguish the different information paths before training.
Timing reconstruction uses each original decoder; amplitude uses separate head
data with the timing TRAIN scaler. The
[verified diagnostic](../encoders/raw_patch_bottleneck_mae/EARLY_MIXER_RELIABILITY_DIAGNOSTIC.md)
reports better timing means and worst cohorts with cohort and transfer tradeoffs.

```bash
bash code/embeddings/scripts/check-early-mixer-reliability.sh
bash code/scripts/task.sh rpb-paired-pooling evaluate-early-mixer-reliability
```

The source-bound admission and sealed independent reader must precede quality
generation. Saved features serve both views and every head; independent CPU
verification runs no encoder. The completed
[learning-curve diagnostic](../encoders/raw_patch_bottleneck_mae/EARLY_MIXER_LEARNING_CURVE_DIAGNOSTIC.md)
retains all four points through 2048. The subsequent [matched-target gain card](cards/matched_target_gain_v1.md)
uses the same early inference and fixed heads at 512, with original-scaler gained
TRAIN inputs and targets. Its [audited result](../encoders/raw_patch_bottleneck_mae/MATCHED_TARGET_GAIN_DIAGNOSTIC.md)
fails all six numeric guards; no gain-range, budget or head rescue is authorized.
