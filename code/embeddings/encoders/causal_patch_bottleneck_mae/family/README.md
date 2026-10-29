# Configurable causal patch bottleneck family

One window is one instrument. `CHANNEL_IDS` are cuwacunu_torch kline-interval
ordinals (`embedding/encoders/timeframe_channel.h`). `read_settings` refuses
the withdrawn instrument binding `{0,1,2}`. `read_withdrawn_instrument_settings`
opens only that recorded binding for the historical coordinate15 digest lock.
`read_admitted_settings` opens either a timeframe list or that binding. CPB
v17, v18, and v19 use it, because their `research.conf` files still store
`{0,1,2}` and their tests also load timeframe fixtures. A checkpoint that
already carries those ids can still be encoded.

The deliverable is a constructible, trainable encoder family. Mityane discovers
and compares architectures; Cuwacunu owns substantial application training and
final docking. The V16 seed946103 checkpoint remains a research comparison
instance, with its mixed verdict: final logical update 2048, native36, selected
on 22 October 2026. Fresh application training does not require that checkpoint
or its C3/H128/F9/native36 geometry. A later reference change uses the
[family screen](../../../evaluation/protocols/cpb_family_screen_v1/POLICY.md):
development cannot promote, and January, February and March 2024 are consumed.
The live readout scores the anchor resolution and cross-resolution agreement.
Those four names are not the published three-instrument probes, and they do
not enter the promotion rule.
[`current.json`](../versions/current.json) and [`index.json`](../versions/index.json)
are frozen catalogs. [Version roles](../versions/ROLES.md) name the one live pointer.

This maintained C++ surface separates four identities:

| Object | Binding |
| --- | --- |
| Family | `CPB-family`, the architecture and causal boundary in `family.h` |
| Configuration | Input geometry, timeframe channel IDs, latent widths and capacity in `Config` / an `ENCODER` settings file |
| Training recipe | Named Gaussian/temporal objective, optimizer options, preprocessing recipe, sampling/mask/context rules and counter seeds |
| Trained instance | Configuration, initialization seed, source/split/recipe IDs, fitted preprocessing, weights, update count and typed artifacts |

Changing a supported geometry or capacity uses configuration. A different
architecture or objective still needs its own explicit definition and evidence.
Existing weights require their saved configuration; loading does not reshape them.

## Supported construction

Include `embedding/encoders/causal_patch_bottleneck_mae/family.h`; add this
directory's `include/` and `../versions/v16/include/` to the include roots.
`read_settings()` requires `MODEL_TAG = CPB-family`. Unknown settings are refused.
The examples in [config](config/) are engineering fixtures, not recommended
training budgets or measured quality winners.

| Configuration dimension | Actual constraint |
| --- | --- |
| Channels C, features F, patch P, encoder W, level D, change U, layers L, heads, feedforward and decoder widths | Positive integers, with checked int64 products/counts and available device memory |
| History H | At least 17 real ticks for the eight non-Nyquist harmonics and fixed temporal lags |
| Encoder attention | W divisible by number of heads |
| Level width | D greater than `4*C*(C-1)/2`, leaving a positive learned part |
| Native width | D + U; learned part `D - 4*choose(C,2)`, analytic spectral part `4*choose(C,2)`, change part U |
| Partial final patch | Allowed: learned backbone pads to `ceil(H/P)*P`; harmonic/time coordinates keep original H |
| Execution | Explicit CUDA device, F32 model; genuine CPU F64 observations, Boolean masks, timeframe channel IDs and explicit aligned cutoffs |
| Architecture choices | One early channel mixer, global mode2/source0, raw patch projection and dropout0 |

Those last choices define this family variant. They are not user-facing capacity
knobs. Zero dropout also avoids an unrecorded stochastic network RNG on resume.
The focused evidence uses CUDA:0; other CUDA devices and arbitrarily large valid
configurations have not been tested. Shape validity is not a memory guarantee.

Useful training support additionally needs three observed patches per eligible
channel: two visible and at least one hidden target. Unsupported examples/batches
abstain; a syntactically valid configuration does not ensure data support.

`dimensions(config)` derives patch, pair, bank, decoder and native sizes.
`parameter_counts(config)` derives learned/fixed tensor and value counts using
checked arithmetic before allocation. `fresh()` verifies the constructed
inventory against those formulas. The historical benchmark happens to yield
75 learned tensors, 293672 learned values and 3888 fixed values; these are no
longer universal construction assertions. See [shape.h](include/embedding/encoders/causal_patch_bottleneck_mae/family/shape.h).

## Portable training path

The [runnable training example](examples/TRAINING.md) supplies a native CLI,
typed input format and fresh/resume commands. The
[generic observation-window archive](../../../../data/observations/OBSERVATION_WINDOWS.md)
keeps that example independent of a market reader. The CLI binds dataset,
settings and executable bytes, optimizer/batch policy and counter seeds;
step/attempt budgets and periodic save cadence remain explicit run options.

A numeric-series window enters through
[numeric_series_history.h](include/embedding/encoders/causal_patch_bottleneck_mae/family/numeric_series_history.h).
`history_from_edge_samples` reads one `edge_sample_t` per instrument from the
numeric-series loader. Channel order is the configuration's torch interval
ids. The anchor channel is the interval equal to `SAMPLING_INTERVAL`. Its last
`HISTORY_LENGTH` valid keys are the shared clock. Each other channel contributes
the latest valid bar at or before that key, held until its next bar. A
log-return view uses `kline_cache_t::normalize_log_returns` against the previous
valid bar of the same channel and does not write a second cache. `INPUT_WIDTH`
is the nine kline features. The same `RawHistory` then goes to the fitter and
`encode_at`. This does not select a population or replace the observation-window
example.

Include [family/training.h](include/embedding/encoders/causal_patch_bottleneck_mae/family/training.h).
The caller supplies its own declared TRAIN stream and source/split manifests.
The fitter cannot infer membership or enforce a calendar from observation values.

```cpp
namespace cp = embedding::encoders::causal_patch_bottleneck_mae::family;
namespace tr = cp::training;
auto config = cp::read_settings(settings_path);

tr::PreprocessingFitter fitter(config, {source_id, split_id, preprocessing_recipe_id});
// Only admitted TRAIN windows enter this pass; no labels or model calls.
for (const auto &batch : train_calibration_stream)
  fitter.add(batch.raw, batch.cutoffs);
auto preprocessing = fitter.freeze();
auto instance = cp::fresh(config, preprocessing.scaler, preprocessing.bank, seed);

tr::OptimizerConfig options{.0005, .0002, .7}; // lr, weight decay, global clip norm
auto optimizer = tr::make_optimizer(instance.model, options);
tr::TrainingIdentity identity{source_id, split_id, complete_training_recipe_id};
tr::TrainingProgress progress{0, 0, row_seed, mask_seed, context_seed};
// Application row/context selection must follow the bound counter recipe.
auto result = tr::train_step(instance.model, instance.scaler, *optimizer,
    progress, batch.raw, batch.cutoffs, options);
tr::save_training_state(new_state_path, instance.model, instance.scaler,
    *optimizer, identity, progress, options);
auto resumed = tr::load_training_state(new_state_path, config, identity, options);
// Continue using resumed.model/scaler/optimizer/progress; never refit on resume.
```

Preprocessing crops at each explicit cutoff before reading values or masks.
It canonicalizes timeframe channels, rounds parent values F64→F32→F64, then
accumulates window-weighted population count/mean/centered-M2 statistics using
Chan batch combination. This stable fitter is a new recipe; it is not a
replacement for historical study calibration. The relative-change bank keeps
the frozen raw-F64-to-F32 feature and bounded-feature statistics recipe.
Source/split/preprocessing-recipe IDs bind the frozen bank. Fit once, freeze
once; new training instances fit their own TRAIN statistics.

The implemented objective is Gaussian NLL plus **0.1** times a fixed mean of
hidden temporal Huber losses at lags **1, 4, 16**, scale √2 and delta1. Both
endpoints must be in the original hidden-observed target mask; deleting extra
context never adds targets. The mean path receives both gradients; variance
weights learn with their input gradient stopped. This objective is explicitly
named and typed rather than silently tuned when dimensions change.

`train_step()` owns masking, one backward pass, finite-gradient checks, one
global learned-parameter clip and one AdamW step. Learning rate, weight decay
and clip norm are configured separately and saved exactly. Other AdamW options
retain their declared defaults; this path has no scheduler or accumulation.
No-eligible batches increment attempted batches without completing an update.
Optional context deletion must be a subset of visible cells. Row selection and
context-deletion streams remain caller-owned and must use the saved counter
seeds/attempt count and the recipe bound in `TrainingIdentity`.

Training archives save the full model, frozen input scaler/bank, initialization,
configuration, progress, row/mask/context seeds, learned parameter order, AdamW
moments/steps/content identity and optimizer options. Wrong configuration,
source/split/recipe IDs or expected optimizer options are refused. No implicit
Torch CPU/CUDA RNG state is saved: exact continuation requires this dropout0
path and the explicit counter-based caller streams. Additional stochastic
augmentation must define and persist its state before claiming exact resume.

For inference, call `eval()` and `clear_gradients()`, then
`save_checkpoint(path, model, scaler, completed_updates)`. Inference artifacts
cannot resume training. Family outer artifact kinds are distinct from frozen
V16 artifacts; the nested codec explicitly identifies its preserved V16
implementation dependency. A trained family artifact needs its own Cuwacunu
registration/source closure/ModelRef; it is not an old V16 descendant by width.

## Portable now and remaining application work

| Portable here | Still owned by application docking |
| --- | --- |
| Configurable fresh factory, shape validation and derived parameter inventory | Application settings/identity registration and data semantics |
| TRAIN streaming preprocessing fit and frozen state | Dataset reader, actual TRAIN membership, admitted calibration population and manifests |
| Causal forward, masks, objective, backward, AdamW and clipping | Application-specific sampling/context policy and tuning |
| Native fresh/resume CLI, deterministic example sampler, total budgets and periodic checkpointing | Production launcher, application data adapter and durable operational recovery |
| Typed training save/load, inference export, byte-bound example run records | Streaming/sharded data, optional scheduler/accumulation/distributed execution |
| Focused synthetic shape/train/resume evidence | Substantial application training and independent application utility evaluation |

The frozen research market runner remains under
`evaluation/protocols/cpb_v16_temporal_v1/`; its BTC/ETH/SOL cache schema,
4096 calibration rows, batch8 and 2048-update budget are not the portable
application launcher. That runner placed three instruments on the channel axis.
Those ids `{0,1,2}` are the withdrawn instrument binding, not the channel
contract. No new market study, campaign capture or application
training is started by this handoff.

## Dependencies and preservation

The numerical surface needs C++20, Torch/STL and the preserved V16
`include/` closure. It requires no Python, Node, market reader, research runner,
V4 source or pretrained checkpoint. The CLI also uses the generic observation
archive header. Node orchestrates the end-to-end CLI checks and reads metadata;
all numerical work stays in C++. The original comparison fixture reads the
historical V16 settings/public entry. The separate trained-reference comparison
requires the explicit saved checkpoint and its hash. Build only inside the managed `mityane-geometrics` container,
through root `container.ps1` / `code/scripts/task.sh`. The pinned SDK is
LibTorch2.6.0+cu124, C++11 ABI1, at `/opt/cuwacunu_embedding/libtorch`.

All `versions/v16/` source, original settings, archives, protocols, study
receipts and ranking/reference decisions stay unchanged. This family API has
a new path-bound source identity. The nested legacy codec retains its original
internal names; only the new outer family contracts describe new instances.
Preserve inherited MIT notices and first-party copyright/license terms when
importing both include roots. No environment configuration or SDK is copied.

## Focused engineering evidence

Run once in a new named session/output leaf:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'cpb-family-portability-check-2026-10-22', '-j2', 'test-cpb-family')
```

The fixture generates its own TRAIN data and reads no trained checkpoint or
market dataset. It covers C1/H25/F1/P4/native10, C2/H33/F2/P6/native15 and
C4/H49/F3/P8/native37, changing capacity as well as shape. It also checks
benchmark constructor/forward parity, poisoned future values/masks, canonical
channel order, partial-patch trace, frozen state, high-offset input scaling,
typed export and refusal paths. The
[22 October receipt](evidence/portability-2026-10-22.json) records the passed
**899 assertions**, **nine physical synthetic updates** and **three exact
next-step resume cases**, with source and tested-binary pins. Learned parameter
counts are 6197, 22751 and 81228 for the three fixtures; fixed counts are 0, 64
and 864. The benchmark's constructor and forward output also match the preserved
entry at its original configuration. These are engineering checks, not evidence
of quality at the alternative geometries.

## Trained-reference compatibility and runnable example

The [compatibility receipt](evidence/compatibility-2026-10-22.json) records two
additional passed checks under the same managed environment:

- **5193 native assertions:** the selected trained V16 seed946103/final2048
  matches the configurable interface's embeddings, distributions, objective,
  support and all 75 learned gradients. Future-poison invariance and the original
  archive/state are preserved. A deep-copied trained adapter is comparison-only;
  it is neither updated nor exported with fresh ancestry.
- **Matched fresh continuation:** original and family interfaces share identical
  frozen reference preprocessing, seed, masks, batches, context deletion and
  historical optimizer options. Three logical updates plus save-after1 and two
  resumed updates match parameters, losses and AdamW state exactly. This costs
  ten physical synthetic updates and performs zero fits; the new fitter is
  deliberately outside that equivalence comparison.
- **Runnable CLI:** 36 assertions over 15 commands cover fresh2, resume-to4 and
  uninterrupted4 at native15, periodic typed saves, final export reloading and
  nine refusal cases. Resumed model/optimizer/statistic identities match the
  uninterrupted run. The example performs eight physical synthetic updates,
  two TRAIN fixture fits and zero fits on resume.

The numerical family headers and all 49 frozen V16 files remain byte-identical.
The earlier 899-check receipt retains its original bytes; its build wrapper and
this guide have since expanded, so the Hero qualifies that receipt's historical
source snapshot separately. The current receipt pins the extended tool surface.
The centered fitter's earlier high-offset regression remains separate evidence;
this comparison does not claim that new and historical fitting arithmetic are
bit-identical. No new encoder-quality ranking follows from these checks.
