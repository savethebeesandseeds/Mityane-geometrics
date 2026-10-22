# Raw Patch Bottleneck MAE (RPB-MAE)

This folder owns the RPB model family, preprocessing, training/checkpoint
adapters and the configurable spectral/content relation APIs.

The maintained [configurable family and training surface](family/README.md)
delivers fresh v22 over the existing v20 backbone, streaming TRAIN fitting,
masked waveform updates, typed AdamW continuation and inference/embedding export.
The [producer receipt](family/evidence/portability-2026-10-22.json) records actual
bounded CUDA proof at two configurations. The [source delivery](family/migration/README.md)
carries the complete method, runnable C++ example, dependency/license closure
and byte-bound export/import evidence. The
[22 October audit](family/audit/COMPONENT_TRAINING_AUDIT_2026-10-22.md) retains its
original source snapshot, later trained-parent distinctions and engineering plan.
This delivery selects no checkpoint and adds no scientific quality result.

## Agent instructions: reusable encoder components

Follow the shared [encoder Hero instructions](../AGENTS.md). Hx16 develops
reusable RPB architecture and training tools. Keep the family definition,
supported configuration, training recipe and trained instance separate.
Historical leaders and benchmark geometries provide evidence for particular
instances; they do not define a single product shape for every consumer.

Supported changes to input geometry, latent width or capacity belong in
configuration, with derived dimensions/counts and actual architectural
constraints. Preserve the exact saved configuration when loading existing
weights. Expose fresh construction, TRAIN preprocessing fitting, training,
export and checkpoint/resume through portable interfaces where implemented;
document version-specific limits and missing paths explicitly. Use focused
alternative-configuration checks to establish engineering coverage. Cuwacunu
and other applications own substantial application training and final docking.
Preserve frozen research source, protocols and results; this guidance does not
resume an experiment or turn historical engineering checks into new evidence.

## Research references — 18 October 2026

**RPB-v24.market-01.alt-01, native36, is the explicitly designated historical
balanced leader.** It passed 8/9 market conditions and all four primary frozen
TEMPO-5 transfer comparisons with full coverage. Its exact market-trained
checkpoint retained parent32 plus learned4 and transferred to complexity5/5
timing without synthetic encoder updates. The
[saved milestone](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/MILESTONE_2026-10-14_NATIVE36_TRANSFER.md)
records that leading decision and the remaining linear-volatility failure.

**RPB-v30.market-01 subsequently passed the full 9/9 market gate and 8/8 strict
native36 floors**, with all three neural-trend fits positive and full coverage.
Its independent synthetic transfer remained pending, so the historical balanced
reference stayed native36. Read the
[saved V30 market record](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/LOCATION_SCALE_DECODER_V30.md).
RPB-v22's primary application designation belongs to the predecessor application;
it is not a new project-wide Mityane encoder selection.

These are preserved successes under their original protocols. The later
noncausal-training diagnosis changed RPB's eligibility for the clean causal
search; it did not erase the measurements. In the separate
[CPB family](../causal_patch_bottleneck_mae/README.md), V3 is the stable fresh
causal baseline and V4 is the broad clean-causal breakthrough. Neither family
has a universal winner across tasks, protocols and information horizons.

Start with the [current research guide](../../../../doc/EMBEDDINGS.md) and
[post-migration addendum](../../../../doc/EMBEDDINGS_HANDOFF_ADDENDUM_2026-10-18.md)
for aims, reference roles and the current continuation scope. The
[archived family guide](../../../../.archive/embeddings/pre-cleanup-2026-10-18/code/encoders/raw_patch_bottleneck_mae/README.md)
and [historical version registry](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md)
retain the earlier chronology and evidence. Historical next-step notes do not
resume research. No model/checkpoint selection changes here.

## Current source navigation

The [version index](versions/README.md) locates all 30 designs and existing
version-specific implementations; [shared components](shared/README.md) owns
the reusable core and adapters. Canonical files live under `versions/vN/` or
`shared/`. Older public `include/` and `config/` paths remain compatibility
surfaces. Current paths locate maintained code; historical source hashes bind
archived bytes or frozen capsules.

The additive [streaming TRAIN and resume support](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/STREAMING_TRAINING_SUPPORT.md)
provides a bounded statistics accumulator and a separate v22 AdamW continuation
archive. Its recorded engineering checks do not change historical model or
inference formats. The API and CLI sections below describe interfaces and
preserved development evidence, not a selected experiment.

## Content-only ordered relation API: RPB-v22

[RPB-v22](CONTENT_ORDERED_RELATION.md) is the separately tagged content-only
relation API. Start with the [plain encoder guide](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/RPB_V22_ENCODER_GUIDE.md)
for its input/output contract, architecture and training explanation.
It accepts values and masks `[B,C,H,F]` and serves native `[B,D]`, using the
same configurable geometry, TRAIN scaler, masks and waveform decoder as v20.
Default C3/H128/F9/P8/D32 allocates 20 learned coordinates and an unchanged
12-coordinate fixed spectral tail.

All configured channel/feature pairs contribute normalized odd/even moments
at original tick lags 1/2/4/8. Common counts normalize and validate those
comparisons; they are not additive learned inputs. A bias-free, zero-start
20x1944 projection adds their content to the learned coordinates without
consuming initialization RNG. Changing masks can still alter moment estimates
or remove information; the model does not guarantee arbitrary deletion invariance.

Construct `ContentOrderedRelationModel` from
`read_content_ordered_relation_settings` and
[content_ordered_relation.conf](config/content_ordered_relation.conf).
The separate `save_content_ordered_relation_checkpoint` and
`load_content_ordered_relation_checkpoint` functions bind its tag, layout,
geometry, scaler and update count, with immutable filenames. V20/v21/v22
typed settings and checkpoints are distinct. The sole training objective is
masked waveform Huber reconstruction; classifier heads remain in evaluation.

`rpb-v22` builds its artificial CUDA checker; `test-rpb-v22` runs it.
These targets provide engineering checks, not a general training/serving CLI.
The [completed RPB-v22.alt-01 experiment](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CONFIGURABLE_CONTENT_RELATION_V1.md)
passes the two-cohort screen and third known development check on TEMPO-5,
designed complexity 5/5, at the unchanged 512-update/B8 budget. This is synthetic
development evidence, not fresh confirmation or market quality. RPB-v20 remains
an explicit reference. All historical instances and standalone CLI defaults remain
preserved.

## Configurable spectral API: RPB-v20

[RPB-v20](CONFIGURABLE_SPECTRAL.md) is a separate reusable configurable API. It accepts values and observation masks `[B,C,H,F]` and returns
native global `[B,D]`; the model-only file defaults to C3/H128/F9/P8/D32.
C/H/F/D and semantic IDs are fixed per model/checkpoint, while batch size varies.
H>=17 and D>4×channel-pair count are required. Private padding supports histories
that do not divide into complete patches. Every feature reaches the learned
backbone and every feature pair reaches the fixed spectral mean.

Its CUDA float32 model, configurable TRAIN scaler, typed file binding and immutable
inference checkpoint API have recorded engineering checks. A separate
[completed capacity experiment](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CONFIGURABLE_CAPACITY_V1.md)
measures three profiles at C3/H128/F9/P8 on TEMPO-5, complexity 5/5.
Compact RPB-v20.alt-01 (W64/D32) remains the preserved capacity configuration; wider-backbone
alt-02 and larger-output alt-03 show no consistent final classification benefit
at the fixed budget. H256 encoder quality and market quality remain unmeasured.
`rpb-v20` builds four check executables;
`test-rpb-v20` runs them. These targets do not build a v20 training/serving CLI.
The [API guide](CONFIGURABLE_SPECTRAL.md) is the contract for this model.

## Earlier quality records and reference roles

[Content-only relations](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CONFIGURABLE_CONTENT_RELATION_V1.md) and the
[13 October continuation](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-13_AFTER_CONTENT_RELATION.md)
record the V22 stage before the later native36 and V30 work above. That stage's
proposed unchanged-v22 synthetic confirmation is a historical proposal.
The [RPB-v21.alt-01 screen](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CONFIGURABLE_LEARNED_RELATION_V1.md)
is stopped after its deletion gate fails; keep its recipe and evidence intact.
[Configurable capacity](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CONFIGURABLE_CAPACITY_V1.md) and its
[12 October continuation](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-12_AFTER_CONFIGURABLE_CAPACITY.md)
remain the preserved v20 milestone.
The preserved [RPB-v19.alt-01](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/SPECTRAL_CONFIRMATION_V1.md) record
confirms unchanged v19 on fresh TEMPO-4 slow-component timing
and AMP-2 component balance, each at designed complexity 5/5. These are synthetic
development measurements at the report's fixed geometry and budgets. Most strength
comes from the fixed prior; the report retains initial controls and all cohorts.
No new experiment is measured by this guide.

**RPB-v4** was the formal evaluated reference at this earlier CLI stage. The original **RPB-v7** instances,
all historical checkpoints/scalers/readouts, and every frozen card remain protected.
Use the [version registry](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md) for stable design
and instance tags. The harmonic decoder was a prospective idea in the
[dated continuation](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/CONTINUATION_2026-10-10_AFTER_SPECTRAL_CONFIRMATION.md).

Historical records include the [v19 screen](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/MULTIBAND_SCREEN_V1.md),
[v18 unit-relation screen](UNIT_TEMPORAL_RELATION_SCREEN.md),
[fresh v17 confirmation](FIXED_PRIOR_FRESH_CONFIRMATION.md),
[architecture screen](TEMPORAL_ARCHITECTURE_SCREEN.md),
[visible-difference diagnostic](VISIBLE_DIFFERENCE_DIAGNOSTIC.md),
[native curve](NATIVE_CURVE_ADVANCE.md), [global bottleneck](GLOBAL_BOTTLENECK_ADVANCE.md)
and [channel mixer](CHANNEL_MIXER_ADVANCE.md).
Their dates, cards, source captures and evidence stages govern those results.

## Historical CLI model and adapters

The sections below describe the preserved `embedding_raw_patch_bottleneck_mae`
CLI and its adapters. The configurable v20/v22 APIs above have their own entry points.
Shared evaluation owns datasets, splits, fixed heads, populations and reports;
model-specific training, inference and decoding stay in this folder.
Use the [current research guide](../../../../doc/EMBEDDINGS.md) for the paused work boundary and
[NEXT_ADVANCE.md](NEXT_ADVANCE.md) for the preserved development chronology.

## Preserved CLI model

The smoke configuration uses `[B,C,32,F]` histories, non-overlapping eight-step
patches, encoder width 64, three pre-normalized temporal attention blocks with
four heads, and export width 32. Values and visibility indicators preserve the
time-major `[P,F]` order. Visible patches retain their original patch indices,
so removing a target does not compress its time coordinates.

The temporal encoder shares weights across channels and processes each channel
independently. Position-aware pooling produces `z_local [B,C,D]`; an equal mean
over valid local channel vectors produces `z_global [B,D]`. Outputs include
validity, visible observation/patch counts and semantic channel IDs. An absent
channel has a finite zero vector and false validity; an entirely absent sample
also has a zero/invalid global vector.

With the default `global_bottleneck_mode=0`, a small two-layer decoder receives
only `z_local` and target position/channel metadata. It reconstructs hidden
observed cells with hierarchical masked Huber:
average within each eligible channel, then within each eligible example, then
across eligible examples. Training requires two visible observed patches and one
observed target patch per eligible channel. Inference can serve a channel with
one observed patch and reports its support.

An optional aligned channel mixer produces separately named `z_contextual` and
`z_contextual_global` exports. It attends across observed channels at original
patch positions, then uses the shared temporal pooling/export projection.
Set `channel_mixer_layers=1` through [channel_mixer.conf](config/channel_mixer.conf).
The default is zero, preserving the independent model's modules and random
initialization. Missing channels remain zero/invalid; contextual outputs do not
infer support from peers. Reconstruction trains through the exact contextual
export when enabled. The pre-mixer local branch retains independent computation
but shares the contextually trained parameters and has no separate loss.
See [the architecture ablation](CHANNEL_MIXER_ADVANCE.md) for lessons from the
first encoder, contracts and the fixed comparison.

Optional `global_bottleneck_mode=1` uses the valid-channel mean as the sole
reconstruction vector. Mode 2 learns a semantic-ordered, support-aware global
pooling MLP. Both decode only from the exact served `[B,D]` global export and
fixed target metadata. Local/contextual channel exports remain distinct
diagnostic surfaces. The mixer configurations are
[mean_global.conf](config/mean_global.conf) and
[learned_global.conf](config/learned_global.conf); defaults and mode-0 checkpoints
retain their previous behavior. See the
[global bottleneck experiments](GLOBAL_BOTTLENECK_ADVANCE.md).

The later v19/v20 designs add a fixed spectral relation bank through their own
modules; see [the configurable API](CONFIGURABLE_SPECTRAL.md). Forecasting,
JEPA/EMA, VICReg, codebooks and specialist banks remain separate proposals for
RPB. The local surface has different
semantics from the baseline's contextualized per-channel output, even when their
tensor shapes match.

## Source layout

Headers use the namespace `embedding::encoders::raw_patch_bottleneck_mae` and
include prefix `embedding/encoders/raw_patch_bottleneck_mae/`.

```text
code/embeddings/encoders/raw_patch_bottleneck_mae/
  versions/vN/                  # Existing version-specific components/configs
  shared/
    include/embedding/encoders/raw_patch_bottleneck_mae/
    src/                        # Core CLI/workflow and shared adapters
    config/                     # Existing shared configuration files
    tests/                      # Core and shared-adapter checks
  include/                      # Preserved public compatibility headers
  config/                       # Preserved configuration compatibility paths
  tests/                        # Preserved test compatibility entries
```

The model and preprocessing core are header-only. Workflow code owns raw/scaler
archives, checkpoints and CLI operations. Its evaluation adapter implements the
[shared provider interface](../../../shared/representation/include/embedding/shared/feature_evaluation.h).
Comparison orchestration and CLI dispatch live in the shared engine and
[independent evaluation integration](../../evaluation/README.md); the standalone
minimum harness does not link the RPB-MAE model.
The configurable headers have their own file binding and inference archive API;
their config-document check is owned by `code/shared/settings/tests/config_document_test.cpp`.
The reconstruction adapter supplies exact local exports, raw predictions and a
separately fitted mask/metadata control through the
[shared reconstruction interface](../../../shared/representation/include/embedding/shared/reconstruction_evaluation.h).
Shared code owns target masks, export interventions, metric scaling and reports;
the integration CLI owns the separate `reconstruct` command.
The CUDA learning-curve adapter supplies a continuous trainer and immutable
checkpoint snapshots through the
[shared learning-curve interface](../../../shared/representation/include/embedding/shared/learning_curve.h).
Its separate `embedding_learning_curve` integration owns the development card,
GPU gate and final testing. It does not change existing adapters' CPU behavior.
The shared `projection_diagnostic` fits CPU readouts on explicit archived
training/validation features without encoder linkage or retraining. The shared
`global_bottleneck_experiment` compares continuous CUDA trainers and immutable
snapshots through `embedding_global_bottleneck`. Their integration sources are
`code/embeddings/evaluation/src/projection_diagnostic_main.cpp` and
`code/embeddings/evaluation/src/global_bottleneck_main.cpp`.
Shared utilities live in [code/shared/representation](../../../shared/representation/README.md), with include
directory `code/shared/representation/include` and include prefix `embedding/shared/`.

## Container build and checks

Use the existing managed development container and the
[environment workflow](../../../../doc/ENVIRONMENT.md). From the project root on
Windows, build and run through the named `rpb-mae` session:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-mae', '-j2', 'rpb-mae', 'evaluation', 'feature-harness', 'test-rpb-mae', 'test-feature-harness')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-mae', 'smoke-rpb-mae-cuda')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'rpb-mae', 'periodic-rpb-mae')
```

The corresponding commands in a container shell at `/mityane` are:

```bash
bash code/scripts/task.sh rpb-mae -j2 rpb-mae evaluation feature-harness test-rpb-mae test-feature-harness
bash code/scripts/task.sh rpb-mae smoke-rpb-mae-cuda
bash code/scripts/task.sh rpb-mae periodic-rpb-mae
```

`test-rpb-core` runs only the preprocessing, numerical, masking and model tests.
`test-rpb-mae` also runs workflow tests. The periodic test interrupts a real
training process after an atomic checkpoint appears, then resumes and exports
embeddings. The CUDA smoke also loads the GPU-trained checkpoint on CPU.
`test-rpb-feature-adapter` checks actual training seeds/settings, separate
untrained/scaler provenance, frozen assets and held-out extraction isolation.
Model, workflow, feature and reconstruction adapter tests cover mixer-off and
mixer-on behavior.
`RPB_CONFIG` selects an explicit configuration for the CLI smoke script.
Build the separate duration diagnostic with `learning-curve`; its shared and CUDA
adapter checks are `test-learning-curve` and `test-rpb-learning-curve`. The
[learning-curve record](LEARNING_CURVE.md) retains the actual GPU-gate and run
invocations. Complete the gate before launching the full experiment.
`projection-diagnostic` and `global-bottleneck` build the separate diagnostic
executables; `test-projection-diagnostic` and `test-global-bottleneck` check their
shared protocols. `test-rpb-legacy-checkpoint` loads six archived mode-0
checkpoints and compares 24 exact training/validation feature surfaces, scaler
and settings identities, and preserved artifact bytes. CUDA continuation checks
cover mixer-off/on and all three global modes.

This session produces
`/opt/mityane-geometrics/embeddings/build/rpb-mae/embedding_raw_patch_bottleneck_mae`,
`/opt/mityane-geometrics/embeddings/build/rpb-mae/embedding_evaluate` and
`/opt/mityane-geometrics/embeddings/build/rpb-mae/feature_harness`. Objects and temporary
files stay inside the container. Smoke/evaluation runs receive unique directories
under `.temp/embeddings/runs/rpb-mae/`. Use a separate session such as `baseline` for the
existing encoder. Session locks protect runner operations; coordinate direct
binary use, shared source edits and GPU capacity separately.

## Raw data and preprocessing

The versioned raw archive stores float32 or float64 `data [B,C,H,F]`, boolean
`observed` of the same shape, unique semantic `channel_ids [C]` or `[B,C]`,
float64 `endpoints [B]`, a positive uniform sampling interval and feature units.
It carries an encoder tag, artifact kind, schema identity and dataset identity.
The shared legacy float32 archive is preserved and is not accepted implicitly.
Raw dtype, dimensions, units, channel schema and interval contribute to the
schema identity; a different source-precision schema requires an explicit adapter.

`prepare` fits a frozen scaler from its supplied training archive. Statistics
are stored per semantic channel and feature in float64, including observation
counts and applied scale floors. Centering/scaling happens before conversion to
float32 model coordinates. Already quantized float32 observations cannot recover
lost variations. Naturally absent stored values are safely removed before
arithmetic; observed NaN/Inf or overflow is rejected.

Archives contain histories already cropped for each endpoint. The core C++
`prepare_endpoint_history` helper separately accepts a longer raw stream,
timestamps and endpoints, validates uniform ordered sampling, and selects the
last H observations ending at an on-grid endpoint. It rejects off-grid endpoints.
The CLI `prepare` command fits a scaler; it does not crop a longer stream.
Bidirectional attention inside a cropped history serves endpoint embeddings.
Per-timestep causal sequence outputs are outside the current API.

## CLI training and export

After the build finishes, run the following in the container at `/mityane`.
The example allocates a new directory and keeps the original training archive:

```bash
rpb_bin=/opt/mityane-geometrics/embeddings/build/rpb-mae/embedding_raw_patch_bottleneck_mae
rpb_config=code/embeddings/encoders/raw_patch_bottleneck_mae/config/smoke.conf
mkdir -p .temp/embeddings/runs/rpb-mae
rpb_run="$(mktemp -d "$PWD/.temp/embeddings/runs/rpb-mae/manual-XXXXXX")"
"$rpb_bin" synthetic --config "$rpb_config" --output "$rpb_run/raw.pt" --samples 32
"$rpb_bin" prepare --config "$rpb_config" --input "$rpb_run/raw.pt" --output "$rpb_run/scaler.pt"
"$rpb_bin" train --config "$rpb_config" --input "$rpb_run/raw.pt" --scaler "$rpb_run/scaler.pt" \
  --checkpoint "$rpb_run/model.pt" --steps 8 --attempt-limit 1000 --checkpoint-every 4 --device cuda
"$rpb_bin" train --resume "$rpb_run/model.pt" --input "$rpb_run/raw.pt" \
  --checkpoint "$rpb_run/resumed.pt" --steps 2 --device cuda
"$rpb_bin" embed --checkpoint "$rpb_run/resumed.pt" --input "$rpb_run/raw.pt" \
  --output "$rpb_run/embeddings.pt" --batch-size 4 --device cuda
```

`synthetic` creates a deterministic float64 fixture and accepts `--seed`,
`--samples` and comma-separated `--units`. Training without `--input` uses a fixed
32-example fixture. Without `--scaler`, training fits its scaler on that training
dataset. `scaler-fit` is an alias for `prepare`.

`--steps` means additional completed optimizer updates. Every attempted batch
advances the absolute attempt counter and its deterministic row/mask/dropout seed
policy. An ineligible batch performs no optimizer update or weight decay.
`--attempt-limit` bounds additional attempts; exhaustion saves a recovery
checkpoint and returns an error. `--checkpoint-every` counts attempts, including
skips. CPU continuation targets exact parity under unchanged execution settings.
Resume requires the same dataset identity and training batch size; configuration,
seed and scaler come from the checkpoint. The historical CLI supports
`--device cpu|cuda`; use explicit CUDA for research model execution. CPU replay
descriptions above are compatibility history, not a second training instruction.

Checkpoints contain explicit encoder/format/output-semantic IDs, canonical
settings, model and AdamW state, frozen scaler, schema/dataset identities and both
counters. Scaler-fit dataset provenance is recorded separately from the model's
pretraining dataset. Compile-time source fingerprints and Git metadata accompany
the state. Baseline checkpoints are rejected before loading model weights.
Periodic saves replace the destination atomically; CLI output paths cannot alias
their input files, including hardlinks.

Exports store `z_local`, `z_global`, validity masks, visible observation/patch
counts, expanded channel IDs, endpoints and interval. They retain schema,
preprocessing and data identities, plus distinct checkpoint-producing and inference
source fingerprints. Mixer checkpoints also export separately tagged observed
contextual vectors. Global modes carry their reconstruction-path semantic tags;
missing channels retain zero/invalid local and contextual vectors.

## Shared feature evaluation

The encoder-owned `evaluation_adapter.h` and `shared/src/evaluation_adapter.cpp`
map exact RPB-MAE model/scaler
state to the [shared provider contract](../../../shared/representation/include/embedding/shared/feature_evaluation.h).
Its fit callback receives permitted training observations and declared metadata;
it receives no labels, hidden clean signals or held-out observations. The
[shared engine and integration guide](../../evaluation/README.md) owns cards,
splits, normalization/PCA/probes, paired populations, scores and reports. See
[the archived policy review](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EVALUATION_IMPLEMENTATION_REVIEW.md)
for implementation limits recorded at that historical stage. The ordinary RPB
executable owns synthetic/prepare/train/embed and does
not link the evaluator or baseline; it has no evaluation command. Run comparisons
through `embedding_evaluate`.

The `evaluation` target builds that executable with baseline and RPB adapters.
`feature-harness` builds a minimum executable with the baseline adapter only;
`test-feature-harness` also runs the shared engine's dummy-provider tests.

```bash
bash code/scripts/task.sh rpb-mae evaluation feature-harness test-feature-harness
eval_bin=/opt/mityane-geometrics/embeddings/build/rpb-mae/embedding_evaluate
mkdir -p .temp/embeddings/runs/rpb-mae
rpb_results="$(mktemp -d "$PWD/.temp/embeddings/runs/rpb-mae/feature-XXXXXX")"
"$eval_bin" --output "$rpb_results/results" --encoders baseline,rpb --seeds 101 \
  --channels 3 --history 32 --features 3 --train-pairs 32 --validation-pairs 16 --test-pairs 32 \
  --baseline-steps 4 --rpb-config code/embeddings/encoders/raw_patch_bottleneck_mae/config/evaluation.conf \
  --rpb-steps 4 --compare rpb_trained_global,rpb_untrained_global,matched_global
```

The [controlled-pairs-v2 recipe](../../evaluation/cards/controlled_pairs_v2.md)
defines development evidence. Each run saves its instantiated `evaluation-card.json`
before generation/fitting; that card fixes C/H/F, IDs/units, tasks, budgets, matched
widths, seeds and explicit compared-method pairs. Encoder configuration validates
against the card rather than choosing evaluator geometry. The four current tasks
are reversal, level, amplitude and lag sign. This is not consumer acceptance.

RPB options are `--rpb-config FILE`, `--rpb-steps N` for fresh training on permitted
observations only, and `--rpb-checkpoint FILE` for frozen model/scaler extraction
without refitting. A frozen checkpoint cannot be combined with config overrides
or fresh training. Global features require any observed-valid channel;
`rpb_*_channel_concatenation` contains the active local or contextual vectors in
declared semantic order and requires every channel to be observed-valid. Global
decoding modes retain these distinct cotrained vectors as diagnostic features.

Matched tiers always apply train-fit PCA, including when native width already
equals the requested width. Insufficient valid support or numerical rank is
reported as an unsupported fit. Every declared pair has its own fixed common-valid
population; adding unrelated providers cannot change that pair. Reports retain
coverage and grouped effect uncertainty as well as conditional scores.

Raw/mask controls, descriptors and untrained/fresh-frozen baseline controls remain
available. Fresh baseline training uses only permitted training observations and
explicit identity float64-to-float32 preparation. No verified historical
checkpoint/preprocessing association is supplied by this driver.

`bash code/scripts/task.sh rpb-mae evaluate-rpb-mae` selects the separate evaluator
and allocates a unique run directory. `evaluate-minimum` selects the minimum
harness. The [archived v1 reports](IMPLEMENTATION_STATUS.md) retain their original
scores and source identity; reproducing them requires archived v1 source/scripts.
Current v2 commands produce a different protocol and cannot recreate those scores.
Refactor validation is recorded in the status record. Development decoder reliance
and synthetic missingness have been measured. Consumer confirmation on domain
data, domain outage behavior, comparative costs and promotion gates remain open.

## Fresh seeds and fixed-readout missingness

The [fresh-seed record](FRESH_SEEDS_AND_STRESS.md) completes five further
development seeds with unchanged 128-update recipes. Compact PCA36 lag ridge
accuracy averages 79.69% for the contextual mixer versus 48.75% for independent
RPB; four of five within-run intervals are positive and seed704 crosses zero.
This result concerns concatenated channel vectors; pooled-global evidence is
weaker and heavy deletion/temporal gaps erode the benefit.

Add `--stress-sweep fixed-readout-v1` to the separate feature evaluator to reuse
each encoder and its ordinary fitted normalization/PCA/probes across testing-only
missingness cases. The shared engine owns masks, source populations, coverage,
conditional/full-population scores and sidecars. It needs no encoder-specific
stress implementation. Ordinary reports/assets remain exact with stress enabled,
and the original encoder passes the same path. Full recipe, results and
validation are linked in the record. These remain synthetic development evidence.

## Held-out decoder reliance

The separate [reconstruction protocol](../../evaluation/cards/controlled_reconstruction_v1.md)
measures the bottleneck's held-out reliance on trajectory information. See
[DECODER_RELIANCE.md](DECODER_RELIANCE.md) for the complete recipe and interpretation.
It adds no mixer, objective or checkpoint format. Every task/seed trains fresh
assets using only its permitted training observations.

Build and test through a separate named session inside the managed container:

```bash
bash code/scripts/task.sh reconstruction -j2 evaluation test-reconstruction-evaluation test-rpb-reconstruction
EVALUATION_BIN=/opt/mityane-geometrics/embeddings/build/reconstruction/embedding_evaluate \
  EMBEDDING_RUN_ROOT="$PWD/.temp/embeddings/runs/reconstruction" \
  bash code/embeddings/scripts/reconstruct-rpb-mae.sh --seeds 101 --tasks reversal \
    --train-pairs 8 --validation-pairs 4 --test-pairs 8 \
    --rpb-steps 4 --metadata-steps 4 --batch-size 8
```

This is a wiring-sized recipe. The `reconstruct-rpb-mae` Make target runs the
default development recipe through the same
unique-directory wrapper: four tasks, three seeds, 32/16/64 source pairs,
128 completed updates and minibatches of eight. Invoke it with
`bash code/scripts/task.sh reconstruction reconstruct-rpb-mae`. Direct use is
`embedding_evaluate reconstruct --output NEW_DIRECTORY`; `--help` lists card
geometry/IDs/units, patch length, source counts, tasks/seeds, bootstrap budget,
`--rpb-config`, `--rpb-steps`, `--metadata-steps` and `--batch-size`.
Omitting `--metadata-steps` matches the main update budget. This fresh-only command
accepts no checkpoint input. The ordinary RPB executable has no reconstruction
command, and the minimum harness does not register this track.

The saved `reconstruction-card.json` freezes the resolved adapter recipe before
fitting. Fully observed histories support exact cross-source swaps; all original
patch positions are enumerated as held-out targets. The shared engine compares
intact, shuffled and separately zeroed local exports, mask/metadata-only
predictions, an untrained model and the training-mean reference. It scores
training-standardized masked MAE, with Huber as secondary, and bootstraps complete
two-source exchange blocks containing paired variants and repeated patch trials.

The stronger mask/metadata control starts with identical encoder/decoder weights,
freezes the original encoder, and encodes normalized zeros with the actual
visible mask, semantic IDs and positions. It independently optimizes only
`decoder_*` parameters under the same sampled batches, masks, optimizer settings
and default update budget. Its encoder stays in evaluation mode with gradients
and dropout disabled. This control receives no signal values; it differs from
zeroing the trained decoder's input.

Each `provider-assets/` directory saves ordinary resumable `model.pt`,
versioned `training-raw.pt` and `scaler.pt`, plus distinct
`metadata-decoder.pt`, `untrained-model.pt` and `training-provenance.pt` artifacts.
The main checkpoint retains its existing format and AdamW/attempt continuation.
Its `source_fingerprint` identifies the core checkpoint writer; the companion
provenance and `provider-audit.json` separately identify the adapter training
producer with `EVALUATION_SOURCE_ID`, resolved recipe and source manifests.
That producer identity is required to identify how these weights were trained.
Metadata/untrained assets are not ordinary training checkpoints.

Callbacks encode only the supplied visible observations and decode the supplied
compact vectors into raw CPU float64 predictions. Neural decoding remains
float32; inverse scaling runs in float64. Shared evaluation owns its independent
train-fit metric scaler and retains all fixed target support. The
[measured results](DECODER_RELIANCE.md) show synthetic decoder reliance; this
development track supplies no consumer acceptance or frozen-feature quality claim.
