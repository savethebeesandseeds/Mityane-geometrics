# Hx16 RPB component and training-method audit

Date: 22 October 2026. Scope: `encoder-rpb-training-delivery-audit`.

The reusable first delivery surface is **configurable RPB-v22 training over
the existing v20 backbone**, with the shared streaming TRAIN fitter, existing
v22 inference codec and separate typed AdamW continuation codec. Preserve v20
as an explicitly tagged construction/inference surface and comparison reference.
Reuse these implementations rather than inventing another scaler, optimizer
format or network. This selects engineering scope; it selects no trained
checkpoint, application recipe, research winner or new experiment.

The source already supplies substantial numerical machinery. A documented,
runnable fresh/fit/update/save/resume/export example, complete portable recipe
binding and independently verifiable delivery package are still missing.
Implementation, new numerical proofs and release packaging remain proposed.
No model training, scientific evaluation, historical score replay, GPU work,
artifact promotion or application change was performed for this audit.

## Evidence boundary and source identity

The accompanying [source inventory](source-inventory-2026-10-22.json) pins the
inspected working-tree bytes, Git ancestry, selected public roots, quoted-include
edges and origin/destination plan. Its `sourceFiles` distinguish the selected
numerical/fixture closure from supporting source, configuration, documentation,
build, license and historical evidence. This is a source-inspection snapshot,
not a compile, numerical execution receipt or finished export. A Git revision
alone does not identify uncommitted source. Later changes require a new inventory
and appropriate verification; they do not inherit historical path-bound proofs.

Current rules are the [encoder instructions](../../../AGENTS.md),
[component delivery contract](../../../COMPONENT_DELIVERY.md), root instructions
and project charter `ORR-DESIGN-001` v0.1.0. The
[CPB package](../../../causal_patch_bottleneck_mae/family/migration/README.md)
and its consumer review supply packaging/ownership discipline only. CPB's
architecture, fitter arithmetic, objectives, sampling and causality guarantees
are not RPB mechanisms. Frozen source, protocols, receipts and original ignored
artifacts remain unchanged.

Paths in the following source tables are relative to the RPB family root.
Public `include/embedding/encoders/raw_patch_bottleneck_mae/*.h` files are
compatibility wrappers; the canonical bodies are in `shared/` and `versions/vN/`.

## Implemented surface and remaining coverage

| Concern | Inspected implementation | Coverage and gap |
| --- | --- | --- |
| Fresh construction | `versions/v20/include/.../configurable_spectral_config.h`, `configurable_spectral_model.h`; `versions/v22/include/.../content_ordered_relation_config.h`, `content_ordered_relation_model.h` | Constructors allocate fresh learned weights and literal fixed priors. V22 appends a bias-free zero projection without consuming initialization RNG. Fresh v22 needs no checkpoint, teacher or trained parent. The caller sets and records the LibTorch initialization seed before construction. |
| Configuration | Shared `config.h`; v20/v22 `*_settings.h`; version-owned `.conf` files | C/H/F/P/D, semantic channel IDs, W, layers/heads, feedforward and decoder widths and numeric settings are validated. Native layout/architecture tags differ by version. Dimensions follow configuration; no universal parameter count or benchmark shape is imposed. A public derived parameter inventory and advertised configuration coverage matrix remain to be delivered. |
| Input and support | Shared `types.h`, `tokenization.h`, `encoder.h`; v20 `configurable_spectral_model.h` | Values/masks `[B,C,H,F]`, IDs `[C]` or `[B,C]`, interval and optional endpoints. Model input is normalized float32; numerical execution is on configured CUDA. Input tensors may arrive on CPU. Missing storage is sanitized; observed nonfinite data is refused. Semantic reorder is handled explicitly. Missing channel/global support returns finite zero vectors and false validity; support is not inferred from peer channels. |
| TRAIN preprocessing | Shared `preprocessing.h`, `streaming_scaler.h`; v20 scaler helpers | In-memory and bounded streaming fitting produce `FrozenScaler`. Canonical per-channel/feature CPU float64 statistics, counts, floors and semantic IDs travel with weights. Actual TRAIN membership and duplication policy are caller responsibilities. Streaming accumulator persistence and distributed fitting orchestration are absent. |
| Masks and targets | Shared `masking.h`, `objectives.h`; v20 padding/mask helpers | Complete original patches are hidden; logical nondivisible histories are supported by private padding. Hidden observed values are targets; missing/padded values are excluded. Eligibility requires at least three observed patches/channel, with two visible and one target. |
| Objective and gradients | v22 `ContentOrderedRelationModelImpl::forward`; shared `hierarchical_huber`; v20 `freeze_and_validate_prior` | Sole v22 loss is hierarchical masked waveform Huber through the exact served native global export. Targets are detached. Learned backbone/projection/decoder receive gradients; literal spectral prior stays frozen. No EMA teacher, forecast loss, task labels, contrastive objective or CPB variance mechanism is supplied by this surface. |
| Optimization | v22 `training_state.h`, `content_training_adamw_options`, fixture `tests/training_state_test.cpp::update` | Typed continuation supports one complete registered parameter group and the fixed AdamW recipe described below. Backward/clipping/step are actual reusable LibTorch primitives composed by the caller; there is no complete generic v22 training-loop API/CLI. |
| Randomness and progress | v22 `TrainingProgress`, `training_counter_seed`; shared `masking.h` | Separate row/mask/context base seeds plus completed updates and attempted batches. Counter seeding does not define row sampling or additional context deletion by itself. All draw algorithms/order and skip behavior need recipe bindings. Ambient RNG/iterator state and pending gradients are not saved. |
| Inference export/load | v20 `configurable_spectral_checkpoint.h`, `configurable_spectral_checkpoint_io.h`; v22 `content_ordered_relation_checkpoint.h` | Distinct typed inference archives bind resolved config, native semantics, scaler and update count; validate finite state and refuse cross-tag/config mismatches. Immutable atomic publication to a new filename. An inference archive cannot resume AdamW. |
| Training continuation | v22 `training_state.h` | Separate `rpb_content_ordered_relation_training_state_v1` binds model/scaler, AdamW, identities and progress. Resume returns train mode with cleared gradients. Exact continuation is conditional on the full caller recipe and same tested runtime/device, not a general RNG or cross-hardware promise. |
| Workflow and CLI | Shared `src/workflow.cpp`; v20/v22 Make fragments and tests | Historical `synthetic/prepare/train/embed` CLI is a different shared-model workflow and codec. Its train/resume paths reject the early-mixer configuration required by v20/v22. Its runnable example cannot be advertised as a v22 trainer. V20/v22 targets currently build check executables. |

### Genuine geometry and execution constraints

For v20/v22 let `Q=C(C-1)/2`, fixed width `S=4Q`, learned width `L=D-S`.
Require positive dimensions, `H>=17`, `D>S`, unique C semantic IDs,
`W % num_heads == 0`, positive interval/floors/Huber delta/layer-norm epsilon,
`0<mask_ratio<1`, and the integer-overflow guards in validation. C1 is supported:
there is no cross-channel bank or fixed tail. P can exceed the number of useful
training patches; constructibility does not imply training eligibility.

V20/v22 require **actual CUDA float32 model execution**, one early channel mixer,
global bottleneck mode 2/source 0, raw patches, temporal-difference input 0 and
dropout 0. CPU support in the older shared model is not CPU inference/training
support for these configurable APIs. There is no declared maximum memory size,
arbitrary precision, multi-device, distributed, mixed-precision or accumulation
contract. Capacity is limited by actual resources; geometric validity is not a
quality guarantee. B may vary; C/H/F/P/D/IDs/capacity remain fixed per artifact.

Logical H need not divide P. Only the backbone pads to `ceil(H/P)*P`; private
cells are absent, harmonics retain original H and decoder output is cropped.
Use `configurable_spectral_input_config`/scaler helpers for logical H; direct
older `FrozenScaler` validation still expects divisible H. Changing H changes
the harmonic basis and learned coordinates; equal D does not preserve a space.

## Learning semantics that must travel with the network

### Preprocessing and actual input identity

`FrozenScaler::transform` centers/scales raw data in CPU float64 before CUDA
float32 conversion; observed float32 source values cannot regain lost precision.
Masked storage is removed before arithmetic. Input validation checks rank,
dtype, schema/IDs, interval and finite observed values; scaled overflow is refused.
All configured channel/feature entries require positive fit support.

`fit_configurable_spectral_scaler` uses the existing whole-input fitter.
`StreamingScaler::add_train_batch`, `merge` and `finish` provide a different
explicit fit procedure: CPU float64 batch-centered moments with supported-value
anchors and Chan combination, retaining only count/mean/centered-M2 `[C,F]`.
Population SD is floored, with the applied-floor mask retained. Failed adds/merges
do not partially change the accumulator. Merge validates matching semantic and
floor schemas, refuses count/moment overflow and does not deduplicate windows.
Overlapping windows or intentional repeats affect weighting and must be declared.
Batch/merge order can change rounding; equivalence is tolerance-based, not a
universal byte-equality assertion. Record fit algorithm, ordering/partitioning,
source membership and resulting exact scaler bytes/identity in each recipe.

Neither fitter can infer TRAIN roles from a tensor. Fit only on admitted TRAIN;
freeze for held-out data and serving. The scaler's `rpb-scaler-v1-fnv1a64-*`
identity is a stable content fingerprint, not SHA256 archive authentication.
Source/schema/split manifests and independent cryptographic artifact pins are
additional bindings. No public source package should contain datasets or fits.

### Network, masks, loss and update

The learned backbone uses value/visibility patch projections, original positions,
early cross-channel mixing, bidirectional temporal attention and learned global
pooling. V20 appends fixed two-band complex spectral relations: q1–3 low and
q4–8 high, all configured channel pairs and F² feature pairs, fixed mean and
per-pair unit normalization with its recorded support/energy floors. Literal
prior weights are checked and frozen; the learned shape projection starts as a
truncated identity. Initialization/registration order matters for seed parity.

V22 adds content-only odd/even normalized moments at original lags 1/2/4/8 to
the learned coordinates via a literal-zero bias-free projection. For
`N=Q*F²*4`, the content bank has `2N` columns and its learned projection has
`L*2N` parameters (38,880 at C3/F9/D32). Jointly centered four-endpoint comparisons
need common observed support >=2 and positive energies above 1e-12. Bank arithmetic
is CUDA float64, exported float32. Counts/eligibility are metadata, not additive
projection inputs. Changing masks changes estimates/support; arbitrary deletion
invariance is not promised. C1's empty relation route contributes zeros.

`make_training_mask` uses CPU MT19937_64 and `std::shuffle` over observed patch
positions, hiding `clamp(floor(mask_ratio*M+0.5),1,M-2)` of M supported patches.
The partial final patch retains its choice across private cells for validation,
but unavailable padding never becomes a target. Visible input is
`observed & !hidden`; fixed target support is `observed & hidden`. The model
encodes visible support and decodes from its exact native global vector.
The actual normalized waveform target is detached.

Huber averages cells **within each eligible channel**, then equally averages
eligible channels within each eligible example, then equally averages eligible
examples. This differs from a global cell average. Nonfinite supported targets
or predictions are refused. With no eligible example the loss is a detached
zero; a trainer must skip forward/backward/optimizer as appropriate, perform no
weight decay/update, and consume the attempt. Extra context deletion is a
separate caller policy: name its cell/patch granularity, draw order and support
repair/abstention without changing original targets silently.

The existing v22 continuation fixture uses waveform Huber, `zero_grad`, backward,
clip norm 1 and AdamW step. The typed state accepts only learning rate 0.001,
weight decay 0.0001, betas 0.9/0.999, epsilon 1e-8, AMSGrad off and one group
in full registered order.
Inactive diagnostic parameters may have no optimizer slot; frozen spectral
weights must not have one. Per-parameter steps/moment shapes/device/dtype,
finite first/second moments, nonnegative second moments, mapping/order and
counter consistency are validated. There is no scheduler or historical step
clamp. Changing supported optimizer choices needs a new explicit codec/recipe
contract and proof; it is not covered by the current fixed codec.

### Stochastic state and exact continuation

`TrainingIdentity` stores source, split and recipe identity strings. Those
strings are compared, but external files are not independently authenticated
by the codec. `TrainingProgress` stores completed updates, attempted batches
and row/mask/context seeds. `training_counter_seed(base, absolute_attempt)`
implements `splitmix64-counter-v1`; each completed attempted batch, including
a skip, consumes its absolute attempt. Initialization seed and sampler semantics
must also be retained in the external recipe.

The fixture composes those counters with MT19937_64/uniform-integer row selection
with replacement, the original patch masker, and independent uniform-real
cell deletion at 0.15. It checks sufficient remaining visible patches rather
than supplying a general repair algorithm. This fixture is a concrete witness,
not a default application context policy. Library distribution/shuffle behavior,
device/backend settings and within-attempt draw order are part of reproducibility.
The older shared CLI's `splitmix64-counter-rows-masks-torch-attempt-v1` and sampler
are distinct; identical seed integers do not make these recipes interchangeable.

Save only between completed attempts with no pending accumulation. Gradients,
loader iterator, arbitrary CPU/CUDA process RNG and partially completed attempts
are absent. Independently compare restored seeds to the declared recipe; source/
split/recipe/scaler strings alone do not do that. Resume requires exact expected
config, scaler content/identity and the same configured CUDA device (use an indexed
ordinal such as `cuda:0`). Inference ordinal migration is not optimizer-resume
migration. Retain independent SHA256 pins and runtime/compiler/backend identities.
Dropout 0 makes a fully counter-driven recipe feasible; it does not prove
cross-SDK, cross-device or hardware-independent exactness.

## Later versions: trained ancestry is a separate dependency

These source constructors and useful historical recipes are different objects.
All paths below are canonical `versions/vN/include/.../` headers plus the
unchanged version registry and frozen study records.

| Design | Mechanism and source | Protected or fitted dependency in historical recipes |
| --- | --- | --- |
| v23 | `consecutive_change_model.h`, `consecutive_change_frozen_level.h`, `consecutive_change_training_state.h`: v22 plus zero-start additive change projection, waveform Huber; explicit frozen-parent helper clears protected gradients before clipping/step and rejects drift/moments | `RPB-v23.market-01` was fresh trained; `.alt-01` protects trained v22. Do not label every v23 instance inherited. |
| v24 | `separated_change_model.h`, `separated_change_frozen_level.h`: concatenated exact parent prefix plus random change projection, zero residual decoder; parent encode/decode under `NoGradGuard` | Market variants warm-start **RPB-v22.market-02**, 2048 parent updates. Native40=parent32+learned8; native36=parent32+learned4. Random constructor availability does not reproduce that trained-parent recipe. |
| v25 | `energy_change_model.h`, `energy_change_training_state.h`: v24 native path plus zero moment decoder; balanced signed-change/RMS auxiliary is training-only | `warm_start_from_separated` copies the saved v24 **initial** state whose change decoder is zero, retaining the trained v22 prefix. It does not copy the trained v24 final branch. |
| v26 | `conditioned_change_model.h`, `conditioned_change_normalizer.h`, `conditioned_change_training_state.h`: frozen standardized bank, trainable change branch and waveform Huber | Same v24 initialization ancestry; TRAIN bank mean/SD are frozen CPU-F64 count/sum/sumsq over the unchanged F32 bank. This fit is not the streaming raw scaler. |
| v27 | `parent_conditional_change_model.h`, `parent_conditional_change_map.h`, training state: residualize conditioned bank against protected parent32 before learned projection | Exact v26 normalizer plus frozen F64 ridge slopes with penalty 1/unpenalized intercept, fit on 4096 TRAIN windows/512 parent forwards, no calibration updates. Map binds parent/scaler/source/normalizer identity; it supplies no causal or orthogonality guarantee. |
| v28 | `relative_change_model.h`, `relative_change_bank.h`, `relative_change_normalizer.h`, training state: raw-F64 relative bank with frozen standardizer | Saved v24 initial branch and trained v22 parent; new relative-bank TRAIN normalizer. Parent input deliberately rounds raw CPU values to F32 before CPU-F64 scaler arithmetic and CUDA-F32 conversion. Do not replace with direct raw-F64 parent preprocessing. |
| v29 | `parent_modulated_decoder_model.h`, training state: v28 native path; detached parent32 drives zero-start residual gain modulation | Saved v24 initialization, never trained v28 branch weights; reuses exact frozen V28 standardizer. |
| v30 | `location_scale_decoder_model.h`, config and training state: same relative native export; unmodulated frozen-parent+change mean, zero variance matrix/bias, anchored softplus, CUDA-F64 Gaussian NLL with hierarchical reduction | Saved v24 initial weights, never trained v28/v29 change weights; protected trained v22 parent and frozen V28 normalizer. The 2048 new branch updates do not erase the 2048 inherited parent updates. Valid negative NLL is retained. |

V25–v30 `warm_start_from_separated` refuses a nonzero source change decoder;
v30 also checks zero new variance heads. Later typed states reuse source/split/
recipe, counter-stream and fixed AdamW discipline, adding the applicable frozen
bank/map state. V30 restricts active updates to four tensors and forbids parent
gradients/moments. These are useful version-owned mechanisms, not evidence that
every later historical model was constructed and trained independently from scratch.
No later branch is included in the first proposed portable training runtime.

Raw archives contain parent tensors but generally do not authenticate the
historical ancestry independently. External manifests/checkpoint hashes and
source-bound receipts remain necessary; v27's explicit map-parent binding is
an additional numerical restriction. A later delivery can expose these extension
APIs separately with their required initialization/fitting and artifact closure.

### Temporal semantics and historical achievements

RPB represents an **endpoint's retained window** with bidirectional attention and
original-position moment banks. End-of-window use may consume the full available
retained window. It is not a prefix-causal per-timestep sequence interface.
Training reconstruction can use observed ticks after a hidden target within
that window. Availability/split eligibility of fitted scaler, weights, parents
and any fitted bank/map must be checked for the actual operation and cutoff.
Runtime future-input invariance of a descendant does not establish clean causal
training ancestry. Changing these mechanisms would need a new identity and
proof; it must not turn RPB into CPB by relabeling.

Preserve **RPB-v24.market-01.alt-01 native36** as the historical balanced leader:
8/9 market conditions, all four primary frozen TEMPO-5 comparisons and full
coverage, with its linear-volatility failure. Preserve **RPB-v30.market-01**:
9/9 market conditions and 8/8 strict native36 floors, three positive neural-trend
fits; independent transfer remained pending. Later noncausal-training eligibility
limits exclude this ancestry from the clean causal comparison; they do not erase
the historical measurements or select a replacement universal checkpoint.

Evidence: family `README.md`, `versions/RPB-v23.json` through `RPB-v30.json`,
current `doc/EMBEDDINGS.md`, and frozen `MILESTONE_2026-10-14_NATIVE36_TRANSFER.md`,
`LOCATION_SCALE_DECODER_V30.md`, `ENCODER_SEARCH_LEDGER.md` and
`CAUSAL_TRAINING_PROVENANCE_CORRECTION_2026-10-15.md` under
`.archive/embeddings/pre-cleanup-2026-10-18/doc/`. These files are pinned in the
inventory. Historical source and artifact paths retain their original meanings.

## Four identities, recipes and two artifact layers

| Object | Required binding |
| --- | --- |
| Family/mechanism | RPB design/API revision, architecture and native layout tags, complete source closure and licenses; actual masks, fitting arithmetic, loss/reduction, gradient routing and optimizer/stochastic rules |
| Supported configuration | Resolved geometry, semantic IDs, capacity, dtype/device and numerical settings; derived dimensions/parameter inventory; explicitly tested versus only supported settings |
| Complete recipe | Source/schema and admitted TRAIN/split membership; preprocessing fit algorithm/order and scaler identity; initialization algorithm/seed; row sampler/replacement/batch ordering, target mask and any context policy; loss/detachment/reduction; optimizer/clipping, skip/progress/budget/save boundaries, all seeds and runtime determinism assumptions |
| Trained instance | Exact config/recipe/source bindings, weights/scaler/optimizer where relevant, parent and fitted auxiliary state identities, attempts/updates, immutable artifact SHA256, exposure and limitations |

Historical v20/v22 capacity/content studies retain their frozen configs, cohorts,
512-update/B8 recipes, sampling/context rules and receipts; the typed continuation
fixture is not a reproduction of those studies. The v24–v30 market recipes also
retain trained-parent and fit dependencies. A future portable reference recipe
must bind every item above, explicitly state changes from historical arithmetic
or sampling, and use synthetic data without importing a market runner as its
default. Current model-only `.conf` files are not complete training recipes.
Applications choose real data/semantics, admitted TRAIN/calibration membership,
supported hyperparameters, batch/context choices, budgets and substantial
training/scaling/recovery. A changed mechanism receives an explicit variant and
appropriate evidence; consumer ownership is not permission to inherit old claims.

Raw **inference** archives contain numerical model/config/scaler state, semantics
and completed update count. Raw **training** state adds the tightly scoped
optimizer/progress/identity contract. Neither format supplies trusted application
publication, authenticated ModelRef/sidecars, selection, capture, data admission
or compatible downstream fitted parents. Consumers own those envelopes and
independent checks. A changed encoder requires regenerated representations and
compatible fitted descendants; equal dimensions or successful raw loading cannot
establish shared coordinates or downstream usefulness.

## Dependency and license closure

The inventory computes all quoted-include dependencies from all v20/v22 public
canonical headers, the shared streaming scaler and seven existing fixture sources
(three v20, two v22, shared scaler and shared config-document tests). The inspected
closure is **73 files, zero unresolved quoted includes**. It includes:

- 32 public RPB compatibility wrappers; preserve their relative path shape.
- Nine shared canonical headers, six v20 and six v22 headers.
- Seven older canonical headers from v13/v14/v16/v17/v18/v19, compiled through
  shared optional model routes, even when those routes are not enabled.
- Four v21 headers needed by cross-tag/initialized-parity fixtures.
- Seven fixture `.cpp` files and the two settings headers
  `cuwacunu/config/document.h` and `piaabo/parse/simple_kv_block.h`.

The two required local include roots are family `include/` and
`code/shared/settings/include/`. No dataset reader, evaluator, application source,
Python numerical implementation, sibling checkout or research checkpoint is in
this selected closure. This is a quoted-include audit, not evaluation of every
preprocessor branch or an independently compiled recipient. System/Torch headers
are external dependencies, not missing project files. Existing Make fragments,
configs and fixtures are separately pinned support inputs.

| Dependency | Required delivery/runtime treatment |
| --- | --- |
| Project C++ and config parser | MIT; include root `LICENSE`, retained file notices, `THIRD_PARTY_NOTICES.md`, provenance and origin/destination hashes. Imported predecessor base is `b15702b29753188482b11ca85e997006b503d0f0`; per-file migration and current hashes identify actual bytes beyond that revision. |
| Compiler/build | Existing root Makefile supplies C++20, ABI1, Torch's two include roots, O1 and links `torch`, `torch_cpu`, `torch_cuda`, `c10_cuda`, `c10`, pthread. Consumer integrates through its own root build; do not copy the research environment or create component setup/build-environment duplicates. |
| Numerical libraries | LibTorch 2.6.0+cu124 and CUDA 12.4 in the pinned managed Linux environment. The inherited SDK remains `/opt/cuwacunu_embedding/libtorch`; do not copy, relocate or modify it. LibTorch/CUDA/cuDNN retain their upstream binary licenses; Debian compiler/runtime copyright texts remain `/usr/share/doc/<package>/copyright`. Source delivery excludes binaries; any separate binary redistribution carries its complete upstream notices. |
| Filesystem/OS | Linux/POSIX checkpoint publisher uses `mkstemp`, fsync and atomic no-replace `renameat2`, with hardlink fallback where supported. Preserve destination immutability and rejection behavior; this is not Windows runtime portability or a power-loss durability promise. |
| Packaging/metadata | A future exporter may use existing Node stdlib for copy/hash/manifest tasks; numerical implementation stays C++. No npm, Python, SDK installation or image download is required by this audit. |
| Excluded licenses/assets | Dashboard AGPL code/artwork and the Python-2.0 HTML entity table are outside the numerical package. Do not include them accidentally or relicense them under MIT. Data, weights, optimizer archives, SDKs, images, generated runs and credentials stay separate. |

Source closure is established for the selected existing surface. The finished
release still needs a reviewed inventory that includes its new runner/recipe/
tests, an exporter/verifier, retained license texts and explicit change notes.
The audit inventory is the input to that work, not a claim it has shipped.

## Bounded implementation, proof and export plan

1. **Maintained API and recipe.** Add new family-owned adapters/docs under a new
   source identity, keeping all versioned numerical bodies and codecs unchanged.
   Expose discovery/config validation and derived shape/parameter inventories;
   fresh v22 construction; streamed TRAIN fit; normalized encode/forward; explicit
   eligible/skip update; typed save/resume; separate inference export/load.
   Retain v20 as a separate tagged API. Initially advertise only the existing
   fixed AdamW/CUDA-F32/dropout0 continuation scope. Any expanded optimizer,
   fitter, format or sampling semantics needs its own version and evidence.
2. **Complete runnable reference.** Add a C++ artificial observation writer and
   fresh/fit/train/save/resume/export example, with an input schema, role/source/
   split/recipe identities, initialization and sampler/mask algorithms, attempt
   limits, update budget and immutable outputs. Reuse `StreamingScaler`, existing
   patch masking/Huber and typed state. Explicitly bind optional context deletion
   or choose and name no additional deletion. Compare restored seeds independently;
   reject source/config/scaler/recipe drift. Add model-specific launcher source
   under `code/embeddings/scripts/` only in the implementation milestone. Root
   build integration requires a later coordinated change; none is part of this audit.
3. **Bounded artificial engineering proof.** In the existing managed container,
   use at least A=C1/H65/F3/P8/D8/W16/one layer/heads4/FF32/decoder16 and
   B=C3/H129/F9/P8/D32/W32/one layer/heads4/FF64/decoder32, CUDA:0/F32,
   unique semantic IDs. These are proposed meaningful geometry/output/capacity
   alternatives, not newly tested settings. Use small artificial populations and
   only a few actual optimizer actions. Check fresh construction, parameter
   inventory, streamed fit with masks/reorders/large offsets/floors, forward,
   nonzero learned gradients, literal fixed-tail preservation, real weight
   changes, export/reload and exact uninterrupted-versus-restored next
   rows/masks/loss/gradients/model/AdamW/scaler/counters. Include an ineligible
   skip. Fit comparisons use declared float64 tolerances; continuation compares
   identical fitted bytes and the declared runtime.
4. **Failure and temporal coverage.** Refuse invalid dimensions/IDs/intervals,
   observed NaN/Inf/overflow, missing TRAIN fit support, wrong tag/native layout,
   changed logical H even with equal padded H, altered scaler/config/recipe/source,
   wrong device/dtype, optimizer reorder/options/orphan/nonfinite moments and
   existing destination. Test allmissing zero/invalid output, missing/poisoned
   storage, exact target support and absent/skip counters. Verify hidden values
   cannot enter a visible representation; document that future ticks *within*
   the retained window may affect it. No prefix-causal assertion is expected.
5. **Receipts and resource boundary.** Run through `container.ps1` and root task
   entries in named `/opt/mityane-geometrics/embeddings/build/hx16-rpb-portable-*`
   and ignored `/mityane/.temp/embeddings/runs/hx16-rpb-portable-*` directories.
   Coordinate shared GPU capacity and verify actual CUDA before numerical work.
   Retain source/config/runtime/binary/output pins, physical attempts/updates,
   seeds and refusal/coverage records. Reuse unchanged historical evidence in its
   original scope; do not replay market/synthetic quality scores. That future
   numerical work requires its actual bounded scope, bindings and protocol.
6. **Source delivery.** Extend the 73-file closure with new API/CLI/recipe/tests,
   explicit configs and research rationale/failures. Export a new leaf with
   `sources/<repository-relative-path>` and record origin path/hash to destination
   path/hash for every member. Preserve wrapper-relative layout and include roots.
   Reject unsafe paths, symlinks, missing dependencies and overwrite; verify
   offline against an independently retained manifest SHA256. Compile and run the
   bounded example from an isolated import through the recipient's managed build,
   with no sibling checkout. Copying old receipts does not certify relocated code.
   Publish API/config/numerical/artifact/resume compatibility notes and explicit
   refit/retrain/consumer requirements. Application acceptance separately binds
   data, substantial training, publication/selection/capture, exact descendants
   and operational/downstream utility. This plan makes no application changes.

## Existing evidence and acceptance disposition

The source inventory pins the focused fixtures and the frozen
`STREAMING_TRAINING_SUPPORT.md`. That historical support record preserves the
initial failed resume reader attempt, repair, exact-next-update witness and
binary/log hashes. V20 fixtures cover alternative geometry, typed checkpoints
and integrity; v22 fixtures cover independent arithmetic, CUDA gradients,
zero-start parity and cross-tag refusal. V24 native36 and v30 fixtures cover
their protected-prefix and version-specific continuation behavior. These are
inspected existing evidence, **not tests executed or renewed by this audit**.

The audit acceptance is met by the implemented/gap inventory, four-object and
artifact-layer contracts, explicit v20/v22/streaming/AdamW reuse, later ancestry
map, temporal limitations, selected source/license closure and concrete bounded
engineering/export plan above. Only `encoder-rpb-training-delivery-audit` may be
recorded done. Runtime, portability proof, complete source delivery and broader
toolset/provider milestones remain unexecuted and proposed.
