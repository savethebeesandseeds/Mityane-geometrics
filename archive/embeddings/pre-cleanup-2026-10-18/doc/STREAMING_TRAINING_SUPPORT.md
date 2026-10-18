<!-- Copyright (c) 2026 Waajacu. All rights reserved.
See the repository LICENSE for the copyright and permission notice. -->

# Streaming TRAIN scaling and resumable RPB-v22 training

Implemented and checked on 13 October 2026. This is reusable engineering
support, with bounded artificial tests. It changes no encoder arithmetic,
historical checkpoint, inference format, classifier recipe or model selection,
and makes no market-quality claim.

The same portable headers and tests live in `cuwacunu_dev` and
`cuwacunu_embedding`. Data loading, source selection, chronological splits,
training orchestration and evaluation remain outside these modules.

| Component | Scope |
| --- | --- |
| `StreamingScaler` | Observed raw TRAIN moments for configurable v20/v22 inputs; produces the existing `FrozenScaler` |
| RPB-v22 training-state archive | Model, decoder, scaler, AdamW state and deterministic continuation counters |
| Existing inference checkpoint | Serving model/scaler export; does not resume an optimizer |

## Streaming statistics API

Include `embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h`.
The implementation namespace is
`embedding::encoders::raw_patch_bottleneck_mae` in both repositories.

```cpp
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;

// c is the resolved logical v20/v22 model configuration.
rpb::StreamingScaler statistics(rpb::configurable_spectral_input_config(c));
for (const auto &raw_train_input : declared_train_batches)
  statistics.add_train_batch(raw_train_input);
auto scaler = statistics.finish();
auto normalized = rpb::transform_configurable_spectral_input(scaler, raw_input, c);
```

The input configuration preserves logical H while using patch length 1 for
preprocessing validation. Do not pass a nondivisible logical H directly to the
preserved backbone configuration. The encoder still uses its configured patch
length and private padding.

`add_train_batch` accepts raw `Input` values/masks `[B,C,H,F]` and semantic
channel IDs `[C]` or `[B,C]`. It gathers statistics in canonical semantic order,
ignores masked storage and rejects observed nonfinite values. CPU float64
batch moments are combined with Chan's parallel rule, using a supported-value
anchor to handle large common offsets. The retained accumulator is three
`[C,F]` arrays: int64 counts, float64 means and float64 centered second moments.
Temporary work scales with the current batch; historical tensors are not retained.

`merge(other)` combines matching input/semantic/floor schemas. The caller must
provide disjoint intended TRAIN observations or declare their intentional
duplication; merging does not deduplicate windows. Counts are exact. Different
batch/merge orders can change floating rounding, so partition equivalence is
checked with float64 tolerances, not a universal byte-equality claim.

`finish()` requires positive support in every semantic channel/feature. It
returns population standard deviations, the configured scale floor and its
floor-applied mask in the unchanged CPU `FrozenScaler` format. `count()` returns
a copy of accumulated counts. Rejected additions/merges do not partially update
the accumulator. Integer-count and nonfinite moment overflows are refused.

The scaler does not inspect role labels or open files. The caller must enforce
TRAIN-only source boundaries. Use the finished scaler unchanged for held-out
data and serving; the existing transform makes normalized CUDA float32 input.
The accumulator itself has no persistence format in this milestone.

## Training-state API and identity

Include `embedding/encoders/raw_patch_bottleneck_mae/training_state.h`.
Its distinct kind is `rpb_content_ordered_relation_training_state_v1`.
It nests an explicitly typed v22 model/scaler child and adds AdamW continuation;
existing v19/v20/v22 inference formats are unchanged.

```cpp
rpb::TrainingIdentity identity{
    input_source_manifest_identity, split_manifest_identity, recipe_identity};
rpb::TrainingProgress progress{
    completed_updates, attempted_batches, row_seed, mask_seed, context_seed};

rpb::save_content_training_state(new_path, model, scaler, optimizer,
                                 identity, progress);
auto restored = rpb::load_content_training_state(
    saved_path, expected_config, expected_identity, expected_scaler_identity);
```

`LoadedContentTrainingState` contains `config`, `model`, `scaler`,
`unique_ptr<AdamW> optimizer`, `identity` and `progress`. Loading returns a
training-mode model with cleared gradients. The format resumes the v22 family
only; an inference artifact cannot be used as training state, and a training
artifact cannot masquerade as an inference export. Use the separate existing
inference writer when a serving artifact is wanted.

The archive binds all resolved configuration vectors, semantic IDs, tag,
architecture, native layout, dtype, CUDA device, relation lags/support/energy
guards and completed update count. Resume requires the same configured CUDA
device and float32 model parameters; use an explicitly indexed device such as
`cuda:0`. It does not offer the inference loader's
CUDA-ordinal migration. The requested settings and independently expected
TRAIN scaler identity must match. Scaler content is checked as well as its
recorded identity. A mathematically similar scaler with different rounding is
not silently substituted.

Only the unchanged single-group AdamW recipe is supported: learning rate
0.001, weight decay 0.0001, betas 0.9/0.999, epsilon 1e-8 and AMSGrad disabled.
Use `content_training_adamw_options()` to construct it. The complete registered
model parameter order, trainability, archived optimizer mapping and each
parameter's step/moment state are validated. Inactive diagnostic parameters may
have no state. Frozen spectral weights cannot acquire optimizer state. Moments
must be finite CUDA float32 tensors with matching shapes, and second moments
must be nonnegative. Mismatch with the archived per-parameter step manifest,
orphan state, remapped groups, changed options and inconsistent completed
counters are refused. There is no
scheduler or old Adam step-counter clamp.

Publication uses the existing same-directory staging and atomic no-replace
publisher. Every save needs a new filename; failed replacement preserves the
existing artifact and cleans its staging file. This is immutable publication,
not a promise of power-loss durability.

## Counters and caller responsibilities

`completed_updates` counts actual optimizer updates. `attempted_batches` counts
completed attempts, including skipped ineligible batches, and identifies the
next attempt. Both are nonnegative; completed cannot exceed attempted.
Save between completed attempts, after the corresponding counter updates,
with no partial gradient accumulation pending. Gradients and partially consumed
iterator state are not continuation assets.

The three stored base seeds select row, original-patch-mask and additional
context-mask streams. Derive a stream from the absolute next attempt using
`training_counter_seed(base_seed, attempted_batches)`. The versioned
`splitmix64-counter-v1` helper returns a nonnegative int64 seed; a skipped
attempt advances the counter even when it performs no optimizer update.
The caller defines the sampler, mask draws, context repair and within-attempt
draw order. No hidden loader iterator or process-global RNG is saved.

The caller must bind `input_source_identity`, `split_identity` and
`recipe_identity` to the actual immutable manifests and settings. The archive
checks their exact expected strings; it does not hash external files or prove
that supplied role declarations are true. Include the sampler/masking policy,
base seeds, batch/loss/clipping recipe and scaler/input schema in the frozen
recipe. Before resuming, independently compare all three restored base seeds
with that recipe's declared seeds. Preserve the training artifact's external
SHA-256 record; the scaler's built-in identity is a stable content fingerprint,
not a cryptographic archive-integrity check.

Deterministic continuation assumes all future training randomness follows
these declared counter streams. V22's required dropout-zero architecture makes
this feasible. The exact witness establishes continuation in the tested
environment/device; it is not a cross-SDK or cross-hardware reproducibility claim.

## Completed focused container checks

The managed `cuwacunu` container ran these commands through its existing task
launcher, which sources `/etc/profile.d/embedding.sh`:

```bash
bash code/scripts/task.sh market-engineering -j2 test-streaming-scaler test-training-state
bash code/scripts/task.sh market-engineering -j2 test-training-state
```

The combined command passed the scaler check but initially failed the new
resume reader: one reused tensor destination aliased two serialized metadata
fields. Distinct fresh destinations fixed the reader and were mirrored. The
resume-only repair command then exited 0. The first failed log remains preserved.

| Witness | Completed result | Scope |
| --- | --- | --- |
| Streaming TRAIN scaler | PASS | Variable batches, per-row semantic reorder, masks/poison, partition/merge/reference equivalence, large offsets, population floors, incomplete/overflow guards and existing CUDA transfer at logical H129 |
| V22 training state | PASS | Artificial C3/H129/F9/D32/W16, one temporal block; exact resumed next rows/target mask/context mask/loss/gradients/model/AdamW/scaler/counters and identity/immutability refusals |

The successful resume fixture performs three optimizer actions: one pre-save
update, one uninterrupted next update and the corresponding restored next
update. It also checks one ineligible skip before model execution. The initial
failed reader attempt performed one update before stopping. These are bounded
artificial CUDA checks, with no market data, head fitting or quality experiment.
The standalone copies were hash-verified; their tests were not redundantly rerun.

Both Makefiles expose `build-streaming-scaler`, `test-streaming-scaler`,
`build-training-state` and `test-training-state` as additive focused targets.
They do not change historical standalone defaults or require market ingestion,
evaluation heads, `.jkimyei`, Hero or another orchestration runtime. A future
standalone check must run in its managed `cuwacunu_embedding` container with
the documented profile sourced; this milestone creates no new environment.

Completed evidence stays under `/opt/cuwacunu/build/market-engineering` in
the managed `cuwacunu` container:

| Evidence | SHA-256 |
| --- | --- |
| `test_streaming_scaler` binary | `816c76709343da5750b4bbc01a4fc7bc4a556cc11d9a160912d07b24129cd7ec` |
| `test_training_state` binary | `ff9b5f332fec8e9b20938ee12ac746e8b36dfd5e65c80e390499f1ab931f360f` |
| `scaler-training-state.log` | `27c0091196194a4fde7a40ab631ae45be68ff83f84f8da42cd6aaa2231213077` |
| `training-state-repair.log` | `10f53a45dcada0c3da1849c7382b8d119be859bf733ff17411e5b7367510a1d7` |

## Exact portable code/test pairs

The following pairs match byte-for-byte, including the new copyright notices.
The original portable v20/v22 headers and saved evidence were not replaced.
Make fragments adapt the target paths to each repository and are not claimed
to be byte-identical.

| File | `cuwacunu_dev` location | `cuwacunu_embedding` location | Shared SHA-256 |
| --- | --- | --- | --- |
| Scaler header | `code/encoders/rpb_spectral/include/embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h` | `code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h` | `6fbefe84aaab7d630d037846637893a71407f445e7ca58b312efa461a97ee104` |
| Scaler test | `code/encoders/rpb_spectral/tests/streaming_scaler_test.cpp` | `code/encoders/raw_patch_bottleneck_mae/tests/streaming_scaler_test.cpp` | `560af3b272ea866e60b8fcf34c337e3206175f09db347703c22aadc3053ad70a` |
| Training-state header | `code/encoders/rpb_content/include/embedding/encoders/raw_patch_bottleneck_mae/training_state.h` | `code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/training_state.h` | `cfef615a21a2c48ccce2628d93ccc3d0a070ee22ab419c9be1d3f985c66167a4` |
| Training-state test | `code/encoders/rpb_content/tests/training_state_test.cpp` | `code/encoders/raw_patch_bottleneck_mae/tests/training_state_test.cpp` | `d50a7000785446baf5f126aee0be369612f7d89c6d23d32c813cc7e4199ab22a` |

Copyright (c) 2026 Waajacu. All rights reserved. See the repository
[LICENSE](../LICENSE) for the copyright and permission notice. Existing imported
license notices remain attached to their inherited components.
