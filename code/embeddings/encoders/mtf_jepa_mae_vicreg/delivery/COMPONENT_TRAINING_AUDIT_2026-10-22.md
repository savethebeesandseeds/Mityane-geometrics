# MTF component and training-method audit

Hx15 · 22 October 2026 · source/contract audit, not a numerical result.

The retained MTF-v1 implementation already supplies configurable construction,
objectives, an explicit teacher update, CLI training and v1 checkpoint loading.
It does **not** yet supply a complete independently delivered training component:
the update loop is embedded in runners, preparation and continuation recipes
differ, raw state lacks recipe/data/source bindings, and there is no dedicated
inference model codec or verified standalone source package. These are delivery
gaps; this audit does not establish a new quality finding or select a checkpoint.

The [source inventory](component-training-audit-2026-10-22.json) pins the reviewed
working-tree source and supporting contracts by repository-relative path and
SHA256. Line locations below refer to those bytes. In this note, `include/`,
`src/`, `tests/` and `config/` are relative to the MTF family directory; shared
paths are repository-relative. The inventory is an audit receipt, not a release
bundle, numerical test receipt or claim that all listed configurations passed.

## Scope and ownership

This completes only `encoder-mtf-training-delivery-audit` under
`encoder-mtf-portable-training-delivery`. The shared
[encoder instructions](../../AGENTS.md),
[delivery contract](../../COMPONENT_DELIVERY.md) and
[project charter](../../../../../doc/operations_over_representations_design_spec_v0.1.0.md)
govern this work. Implementation, proof, source export and provider/operation
integration remain separate milestones. No build, model training, scientific
evaluation, score replay, GPU work, promotion or application change is part of
this audit. Dashboard refresh is maintenance validation only.

| Object | Required identity and ownership | Current state |
| --- | --- | --- |
| Family | MTF-v1 architecture and learning mechanisms, with actual source identity | Retained modules and public core APIs exist; current paths have their own identity |
| Configuration | C/H/F, scales/strides, widths/depths, output selector, dtype/device and mechanism switches | `Config`, `Settings`, validation and strict text serialization exist |
| Recipe | Initialization, preparation, data membership/order, sampling, masks, losses/gradients, optimizer, RNG/update order and budget | Mechanisms exist; three runner recipes must remain distinct; no complete typed recipe identity is stored in v1 |
| Instance | Exact weights including teacher, configuration, preparation, optimizer/progress when resumable, source/runtime and data/recipe bindings | v1 retains model/AdamW/settings/progress, but several bindings are absent |

Mityane owns reusable numerical mechanisms, their contracts, examples and scoped
proof. Applications such as Cuwacunu own real features/units and availability,
TRAIN/calibration membership, complete supported recipe choices and budgets,
substantial training/scaling/recovery, admission, authenticated publication,
ModelRef/selection/capture and compatible downstream parents. Raw family loading
does not authenticate an artifact or admit its population. Equal latent width
does not identify compatible coordinates. An encoder change can require fresh
representations and refitted descendants under their respective owners.

## Construction, configuration and outputs

`include/embedding/encoders/mtf_jepa_mae_vicreg/config.h:15-77,108-182`
exposes channel/history/feature dimensions; model, latent, projector and predictor
widths; encoder/predictor/decoder depths; heads; dropout; time scales/strides;
frequency bins/log magnitude; mask/loss/teacher settings; dtype/device.

- Dimensions are positive; encoder depth can be zero, predictor/decoder depths
  must be at least one; `latent_dim` must divide evenly by `num_heads`.
- Scales and strides must be positive, with equal list lengths. Empty strides
  resolve to `max(1, scale/2)`. A scale need not divide H or be bounded by H:
  `tokenization.h:11-29` clips width to H and includes a final endpoint window.
- No benchmark geometry is a universal assertion. Counts and memory depend on
  configuration. There is no complete parameter/resource inventory API or broad
  dtype/device qualification. The workflow specifically requires F32
  (`src/workflow.cpp:102-114`).
- `Config{}` defaults differ from `default_settings()` and omitted text settings
  (`config.h:15-77`; `workflow.cpp:182-212,215-290`). A delivery must name which
  defaults it uses. `config/default.conf` is a small reference instance, not the
  family interface. The synthetic CLI fixture requires F=3; explicit archives
  support other architecturally valid feature widths.

`Model`, `tokenize`, `create_masks`, `encode`, `target_encode`, `project_vicreg`,
`forward` and `update_target_network` are callable C++ surfaces
(`model.h:13-108,234-253`). Encoding serves online state, not the teacher:
token latents `[B,N,D]`, global `[B,D]`, contextual channels `[B,C,D]`, pooled
time/frequency outputs, metadata and sample/channel validity (`69-92`). CLI
`embed` exports only global/channel arrays and validity masks
(`workflow.cpp:450-481`), not a portable inference model or a SpaceRef.

## Actual preparation and temporal/dependency contract

There are **three runner preparation paths**, alongside the core tensor policy.
A scaler is not implied by a diagnostic message recommending standardization.

| Path | Implemented policy | Boundary or gap |
| --- | --- | --- |
| Core tensor API | Shared `tensor_ops.h:13-51` accepts floating BCHF (BHF only for C=1), checks geometry, casts to configured dtype/device, converts optional support to bool, treats nonfinite values after conversion as missing and zeros them | No fitter; even F64-to-F32 overflow can become missing. This permissive core behavior differs from archive/adapter refusal |
| CLI train/embed | `workflow.cpp:299-300,393-400,457-460`; observation `src/data.cpp:28-45,141-150` requires nonempty F32 BCHF, bool matching mask, observed finiteness; missing entries become zero | Caller-prepared values, no fitter/scaler or saved preparation identity. TRAIN membership is not independently checked |
| Shared evaluation adapter | `evaluation_adapter.cpp:67-79,195-218` checks CPU floating BCHF and bool support, casts to F32, rejects observed nonfinite/overflow, zeros missing | Explicit identity/no fitted scaler. Its documented F64-to-F32 control is real; do not fabricate a CPB-like scaler |
| Historical `embedding evaluate` | `evaluation.cpp:91-134,589-606` privately fits F64 per-feature observed TRAIN mean and population std pooled over B/C/H, scale floor 1e-6; saves F64 `[F]` mean/scale/counts and normalized TRAIN/test inputs | Evaluator-only fitter, separate recipe. Supplied checkpoint needs saved sidecar (`510-516`); association and training independence remain unverified (`578-581`) |

Time tokens contain observed mean/std descriptors per configured window;
arithmetic accumulates in F64, population variance receives `1e-6` before square
root and descriptors return in input dtype (`tokenization.h:32-50,101-185`).
Frequency descriptors consume their **complete configured window**, using
observed centering, Hann weights, explicit DFT matrices, magnitude divided by
observed Hann weight, DC replacement by absolute mean, optional `log1p` and
padded K bins (`302-306,341-398`). Padded bins can be supported zero-valued
targets. Full-history windows are possible. Masks or time labels do not make
these prefix-causal outputs.

Positive-depth `SharedTokenEncoder` layers mix every eligible token through a
transformed masked global mean and a residual token MLP/dropout
(`encoder.h:55-98`). Per-channel pools therefore remain contextual across
channels, scales, domains and windows. Depth zero skips that mixing loop but
does not establish a family-wide causality claim. Predictor/decoder metadata
queries attend over eligible context without a temporal triangle (`113-163,
178-239`). The honest default interpretation is an observation-window embedding
available after its required window/input support is available. Prefix
causality, independent channel representations, latent dynamics, forecasting,
intervention effects and application utility are unproved here.

## Learning method, masks and gradient routing

Fresh construction registers online/target tokenizer and encoder, predictor,
decoder and VICReg head; uses LibTorch default initialization; copies online
to teacher with tau=0; freezes both teacher modules (`model.h:13-34`). Fresh
construction requires no pretrained or frozen-parent checkpoint. Teacher
constructors still consume initialization RNG before copying. Frequency and
decoder modules are constructed even when corresponding functions/losses are
disabled (`tokenization.h:443-462`; `encoder.h:178-214`). Removing apparently
unused draws/branches can change a trajectory and needs an explicit identity.

Mask behavior is a method choice, not interchangeable terminology:

- Legacy default (`masking.h:159-170,351-433`) proposes a contiguous block in
  flattened time-token order, a random frequency subset and Bernoulli whole
  channel targets, with domain/window coupling and target caps. Time ratio zero
  still proposes at least one time token where feasible (`352-354`).
- Hard bans include same-window cross-domain coupling and optional whole-channel
  blocking (`449-464`). The temporal-overlap threshold is a soft ban, compares
  across channels and can relax to keep minimum context (`466-471,529-535`).
- Strict support first consumes legacy proposal RNG, then forbids same-channel
  temporal overlap across scales/domains without relaxation, reduces targets
  for context and may draw an extra CPU `randperm` for a singleton (`174-290`).
  It can legitimately yield no target; cross-channel context is allowed.
- The conservative support audit (`55-156`, `types.h:56-58`) uses metadata
  windows, not exact sparse raw-feature support. It proves neither causal
  ordering nor absence of channel leakage.

`model.h:108-205` always computes full online latents, masked context latents,
target path, predictor and decoder; flags gate contributions, not all execution.
The total is lambda-weighted JEPA + MAE + time/frequency alignment + VICReg.

| Term | Numerical and gradient meaning |
| --- | --- |
| JEPA | Target-mask MSE over latent elements (`132-135`; shared `tensor_ops.h:63-68`). EMA teacher tokenizer/encoder run in eval/no-grad (`294-314`). They remain frozen even if `stop_gradient_target=false`. With EMA disabled, targets fall back to full online latents; turning stop-gradient off permits gradients through that target branch |
| MAE | Default detached raw time mean/std and frequency descriptors; losses normalize separately over selected supported descriptor elements and then sum (`143-164`; shared `tensor_ops.h:71-88`). Alternate projected targets are detached current tokenizer bases |
| Alignment | Full online latents, supported same-channel/scale/start/width time/frequency pairs, weighted `1-cosine` (`316-372`). This branch uses full observations rather than only JEPA context |
| VICReg | Two augmented views, online encodes and projector; global valid sample rows and optional flattened valid sample-channel rows (`381-437`). Shared invariance MSE, biased variance-floor penalty and off-diagonal covariance squared sum divided by dimension, covariance denominator rows-1 (shared `objectives.h:12-88`). Zero rows gives zero; one row has variance penalty but no covariance; the model does not propagate `statistics_supported` |

Weak views use feature deletion `min(.20, mask_ratio_channel)`, shared-over-feature
time deletion `min(.10, mask_ratio_time*time_dropout_scale)` and observed-position
Gaussian jitter (`objectives.h:11-46`). Gaussian draws still occur at zero jitter;
some time draws occur at zero effective deletion. Online dropout consumes device
RNG too. CPU masks, augmentation, sampling and device dropout draw order are part
of a complete stochastic recipe.

Strict finite loss throws; permissive mode sanitizes nonfinite losses
(`model.h:126-129,194-205`). Diagnostic losses are sanitized on return. Zero
objective weights and insufficient support are constructible and do not
guarantee a useful gradient update. No-target, all-missing or nonfinite cases
need explicit caller policy, not manufactured learning evidence.

EMA is external to `forward`: both target tokenizer and target encoder receive
`teacher = tau*teacher + (1-tau)*online.detach()` under no-grad
(`model.h:238-265`). No schedule is embedded. An optimizer step without the
subsequent teacher update is a different learning sequence.

## Runner recipes, optimization and continuation

| Concern | CLI training | Shared evaluation adapter |
| --- | --- | --- |
| Initialization | `activate` seeds configured seed before fresh construction (`workflow.cpp:116-122,392-406`) | `stream_seed(run_seed,500000)` creates matched random/fresh models (`evaluation_adapter.cpp:219-226`) |
| Execution | Saved CPU/CUDA setting, F32, configured dropout and batch | Forced CPU/F32/dropout=0; card C/H/F and ordered unique channels must agree (`195-216`) |
| Sampling and RNG | CPU `torch::randint` with replacement over input rows, or step-specific synthetic batch; `manual_seed(seed+absolute_step)` before sampling/forward (`412-420`) | Entire permitted TRAIN partition per update; `stream_seed(run_seed,600000+update_index)` (`230-241`) |
| Optimization | Trainable-parameter AdamW, configured LR/weight decay with other LibTorch defaults; zero-grad, forward, finite loss, backward, L2 clipping with nonfinite error, optimizer, EMA (`403-431`) | AdamW receives all parameters (teacher frozen); clipping if positive; explicit finite-gradient check; optimizer, EMA (`228-241`) |
| Continuation | v1 architecture/model/AdamW/progress restoration; steps are additional updates. Config/seed overrides rejected, but batch/device/save interval can change (`370-391`) | Assets are frozen evaluation controls; no adapter resume/load API. Its saved v1 checkpoint is CLI-loadable, but CLI sampling/RNG differ (`186-189,277-288`) |

The historical evaluator is a third reference recipe: synthetic TRAIN/test
generation, its own fitted preparation and its own training loop
(`evaluation.cpp:91-134,371-421,583-606`). Its minibatch sampling with replacement,
`seed+step` reseeding, AdamW/clipping and optimizer-then-EMA order follow the CLI
schedule over its saved normalized TRAIN archive. It must not be relabelled as
the shared identity adapter or proof of application admission.

v1 `save_checkpoint/load_checkpoint/load_optimizer` (`workflow.cpp:303-343`)
store format=1, settings text, completed steps, **all registered model state**
including teacher/tokenizers, and AdamW. Optimizer loading is a separate call.
There is no raw generator snapshot: CLI continuation derives seeds from absolute
step. Under unchanged data bytes/order, batch, source/draw order, device,
threads and runtime, that can reproduce its trajectory; it is not arbitrary
interruption or cross-runtime bitwise equivalence. Allowed batch/device changes
can change it. Raw v1 omits source/config digest, recipe/sampling/preparation
identity, input/split hashes, runtime contract and authenticated provenance.
Restoring a v1 file cannot prove those missing associations.

Observation `src/data.cpp:48-59` writes a neighboring temporary archive and
renames it on Debian. This preserves an existing destination if writing fails;
it is not a durable application journal or checked-publication layer. The
adapter's random evaluation archive has a separate kind/version and no paired
public loader (`evaluation_adapter.cpp:142-161`). Neither representation-tensor
export nor that private archive is a complete portable inference codec.

## Dependency and license closure

The inventory assigns exact files to these additive tiers. Quoted includes and
root link rules were inspected; no external service, Python runtime or Node
runtime is a numerical dependency. A broad build include directory alone does
not establish a dependency (MTF uses its own settings parser).

| Tier | Repository source required beyond preceding tiers |
| --- | --- |
| Core | MTF headers `config.h`, `types.h`, `tensor_ops.h`, `tokenization.h`, `masking.h`, `encoder.h`, `objectives.h`, `model.h`; shared representation headers `types.h`, `tensor_ops.h`, `objectives.h` |
| CLI and historical evaluator | MTF `workflow.h`, `evaluation.h`, `src/main.cpp`, `src/workflow.cpp`, `src/evaluation.cpp`; observation `include/embedding/shared/data.h`, `src/data.cpp`; shared representation `evaluation.h`, `src/evaluation.cpp`. Existing workflow links historical evaluation through its command dispatch |
| Adapter library | MTF `evaluation_adapter.h`, `src/evaluation_adapter.cpp`; shared `feature_evaluation.h`, `feature_harness.h`, `src/feature_harness.cpp` (including `stream_seed`) |
| Existing baseline-only feature harness | Shared `feature_stress.h`, `src/feature_stress.cpp`, `src/feature_evaluation.cpp`; embeddings evaluation `src/main.cpp`, `src/reconstruction_main.cpp`, `include/reconstruction_cli.h`, compiled with `EMBEDDING_WITH_RPB=0` (`Makefile:269`). Full `evaluation` target has a broader RPB closure |
| Fixtures | Family tests and `test_support.h`; shared `shared_test_support.h`, `objectives_test.cpp`, `evaluation_test.cpp`; observation `data_test.cpp`; configurations and shell fixtures. The separate historical `baseline_test.cpp` additionally requires ignored `.build/reference/reference_model.h` |

Root Makefile (`63-70`) uses C++20 and the pinned LibTorch 2.6.0+cu124 C++11 ABI1,
Torch API include roots, `torch`, `torch_cpu`, `torch_cuda`, `c10_cuda`, `c10` and
pthread. Even CPU-mode root executables link this installed CUDA-capable SDK.
The inherited SDK stays at `/opt/cuwacunu_embedding/libtorch`. Compiler/make and
full Debian/CUDA environment versions are recorded in `dependencies.lock` and
`environment-image.json`; that complete environment is larger than the minimal
source closure. `container.ps1`, root task runner and root Makefile remain the
authoritative build/orchestration boundary. No component environment is needed.

Retain root MIT `LICENSE`, `THIRD_PARTY_NOTICES.md` and `doc/PROVENANCE.md` with
source delivery. The recorded upstream base is
`b15702b29753188482b11ca85e997006b503d0f0` of Waajacu's
`cuwacunu_embedding`; the import includes local bytes, so the base alone is not
the current source identity. Migration receipts and current per-file hashes
provide distinct evidence. Preserve cuwacunu_torch attribution and original
notices. LibTorch, CUDA, cuDNN and compiler/runtime binaries retain their upstream
terms. Complete binary-license texts were not inspected in this source audit;
they must accompany any separately authorized binary redistribution. Planned
source delivery excludes all SDKs/images/binaries and carries that obligation.
Dashboard AGPL code is outside the MTF source bundle and retains its own notices.

The historical reference header exists locally but is an ignored dependency,
not an independently packaged fixture. A future release must either explicitly
inventory/map it under its correct historical identity or omit parity coverage
and say so. Fresh construction/training does not depend on that header. The
audit inventory identifies this exception separately from public source files.

## Existing evidence and missing proof

The [18 October migration verification](../../../../../doc/EMBEDDINGS_MIGRATION_VERIFICATION_2026-10-18.json)
records eight retained baseline engineering executables passing at that time.
Those unchanged receipts retain their original source/runtime scope; this
source inspection does not renew them for a package or relocated consumer.

| Inspected fixture | What it asserts; not rerun here |
| --- | --- |
| `tests/model_test.cpp:107-195` | Exact initial teacher copy/freeze, online branch gradients, actual Adam update, teacher unaffected by optimizer, EMA arithmetic, repeated fixed-batch updates and module reload |
| `tests/workflow_test.cpp:90-120,176-255` | Strict settings/old defaults, archive shape refusal, all named-parameter reload, CLI embed parity, AdamW next seeded update equality and CPU/single-thread CLI 3 updates versus 1+2 continuation; fixture dropout=0.1 |
| `tests/masking_test.cpp:96-249` | Strict infeasible/no-target, cross-channel context, singleton and multiscale/domain support cases |
| `tests/numerics_test.cpp:87-193` | F64 frequency accumulation/F32 descriptors, numeric/scalar refusal cases and strict support opt-in |
| `tests/evaluation_test.cpp:20-110` | Historical saved normalization and no-refit behavior, rejection and alias protections |
| `tests/cli_smoke.sh` | CPU/CUDA train/resume/representation export; optional CUDA-checkpoint CPU serving |
| `tests/periodic_checkpoint.sh` | Interrupt after a saved checkpoint and resume/export; no uninterrupted trajectory equality. Its large requested budget is killed after first save; it was not launched |

Existing tests cover useful building blocks and one continuation setting. They
do not establish two-configuration delivery proof, parameter inventory, dedicated
inference codec, complete typed training state, shared-adapter resume, malformed
state/data/recipe identity refusal, independent export/import or application
quality. Historical scores and unknown checkpoint/preparation associations remain
historical and unknown. The current CPB V16 reference and mixed study verdict
are outside this audit's scope.

## Concrete bounded engineering and export plan

The following is an unexecuted implementation/verification plan, not execution
admission. Keep numerical work in C++ and existing root container/task entries;
use named `hx15-mtf-*` build/output paths and coordinate shared CPU/GPU capacity.
Do not change historical codecs, mask defaults, initializer/draw order or loss
reductions under the same identity.

1. **Runtime milestone:** add a family-owned typed configuration/preparation/
   recipe/session layer reusing `Model`, masks, objective helpers, AdamW and
   archive primitives. Expose fresh creation, one update, progress, save/resume
   and online inference export/load. Record dimensions and trainable/frozen
   parameter inventories from actual construction. Preserve legacy APIs and v1
   loading. Keep identity preparation first-class. If making the evaluator
   scaler reusable, give it a separately named exact-arithmetic variant rather
   than changing identity behavior.
2. **State contract:** introduce a new versioned raw training format with full
   configuration and source identities, preparation kind/state or explicit none,
   complete recipe/sampling algorithm, declared data/split hashes, optimizer
   options/state, teacher/tokenizer state, progress, stochastic algorithm and
   runtime limits. Name the CLI absolute-step and adapter full-partition stream
   recipes separately. Reject incompatible continuation or classify it as a new
   recipe; do not infer exactness from load success. Decide whether to support
   adapter continuation explicitly or keep it frozen-only. Keep application
   trust/envelopes/ModelRef outside this raw codec.
3. **Bounded proof milestone:** write a prospective engineering card for tiny
   artificial observations, no labels/market data/quality selection. Candidate
   configurations below change geometry, feature width, latent and capacity;
   they are proposed supported choices, not measured proof. Use a few declared
   updates per trajectory and retain actual physical update counts, source/
   config/binary/runtime/output hashes and seeds. Test uninterrupted versus
   interrupted/resumed state, including AdamW and both EMA modules, with nonzero
   dropout/jitter under each advertised recipe. Check forward, real parameter
   updates, online/teacher gradient routing, optimizer-then-EMA arithmetic,
   inference export/reload, identity preparation and all-missing/nonfinite/
   overflow cases. Exercise strict no-target and legacy mask behavior; reject
   changed geometry/preparation/state/recipe/data where exactness requires it.
   CPU/F32 is the first bounded target. CUDA/cross-device equivalence and large
   scale execution need separate evidence; do not silently claim them.
4. **Delivery milestone:** curate one explicit source/config/test/method/notice
   inventory with source SHA256 and origin/destination mappings. Reuse CPB's
   [packaging discipline](../../causal_patch_bottleneck_mae/family/migration/README.md)
   for overwrite refusal, safe paths, include-closure checks and an independent
   offline manifest verifier. Do not import CPB architecture, preprocessing,
   masks, causality or trained reference. Provide a runnable fresh/prepare/train/
   save/resume/export C++ example and declared source tier. Either carry the
   current CLI's historical-evaluator closure or explicitly decouple it under a
   new source identity. Verify the exported copy in isolation without a sibling
   research checkout. Emit API/config/numerical/raw-state/exact-resume
   compatibility notes and required refit/retrain/consumer checks. Keep datasets,
   weights, generated runs, credentials, SDKs and images outside the package.

| Proposed proof configuration | Geometry/capacity |
| --- | --- |
| A, retained small-core shape | C2/H16/F3, d_model12, latent8, projector12, predictor16, depths1/1/1, heads2, scales4/8, strides4/4, bins4; explicitly nonzero dropout for continuation |
| B, alternative artificial archive | C1/H23/F5, d_model20, latent12, projector18, predictor24, depths2/2/1, heads3, scales3/7/31, strides2/4/16, bins9, dropout0.1; exercises clipped scale and non-F3 input without using the F3-only generator |

Producer completion requires those scoped receipts and a byte-verifiable full
training source package. Consumer acceptance then verifies its delivery pin,
mapped/adapted source and bounded integration in its own managed build. It
separately admits populations/recipes, chooses artifacts and verifies descendants
and application utility. This audit supplies the next engineering specification;
runtime, numerical proof, packaging and provider milestones remain proposed.
