# CPB-v18: contiguous training query spans

V18 tests a different training form on the corrected configurable CPB
component: hide one physical span of observed patches rather than scatter
the same patch quota through the retained history. The encoder architecture,
native layout, initialization, TRAIN fitter, Gaussian NLL plus temporal
coefficient0.1 at lags1/4/16, gradient routing, AdamW and one global clip remain
unchanged. A harder reconstruction query is a hypothesis about learning better
representations, not a demonstrated improvement.

The [completed three-pair study](../../../../evaluation/protocols/cpb_v18_span_v1/RESULTS.md)
**failed every continuation condition** at 1,024 updates. Median paired R²
changes were trend −0.221846, activity −0.090620, volatility −0.057857 and
relation +0.015733. Trend worsened in all three pairs. All six readouts and
independent audits passed, but no new scoped record or better encoder emerged.
V16 seed946103/final2048 remains the research reference. The
[compact record](../../../../evaluation/records/CPB-v18.span-001.json) preserves
all instances, masks, work counts and limitations. The reusable API and complete
training method remain available; this negative mask recipe is not a new default.

This version has its own training-method and artifact identity. It does not
import trained weights or fitted statistics from V16/V17 when constructing a
fresh instance. The original family headers and frozen V16/V17 source remain
unchanged. The [prospective study](../../../../evaluation/protocols/cpb_v18_span_v1/PROTOCOL.md)
owns experimental scope, comparison bindings and permissions. Engineering
implementation, scientific evidence and the selected V16 research reference
remain separate.

## Exact masking rule

`Recipe{MaskMode::contiguous}` is the candidate. For every retained example and
channel independently:

1. Crop raw observation values and Boolean support at the explicit cutoff,
   retaining the configured history. Right-pad only the private partial-patch
   representation, as the preserved core requires.
2. A patch is observed when any real cell in it is observed. Let M be the
   observed-patch count and K the original patch-grid size, `ceil(H/P)`.
3. If M<3, select no query and mark the channel unsupported. Otherwise use the
   unchanged quota `q = clamp(floor(mask_ratio*M + 0.5), 1, M-2)`.
4. Enumerate original-grid starts0 throughK-q whose entire q-patch interval is
   observed. Missing patches remain holes: the candidate never compresses
   observed ranks or joins nonadjacent grid patches.
5. If no legal start exists, select no query and abstain for that channel.
   There is no scattered fallback or rescue quota. Otherwise choose one
   uniformly from the legal starts and hide its complete real patch cells.
6. Crop back to logical H and use the preserved mask validator to derive
   visible support, original hidden-observed query Q and eligibility. Keep at
   least two visible observed patches and at least one target patch.

The stream is a local `std::mt19937_64`, reset for each absolute attempt from
the family's `training_counter_seed(mask_seed, attempt)`. Rows are traversed
in physical B*C order. For a positive legal-start count n, reject draws below
`(uint64_t(0)-n)%n`, then select `draw%n`; this avoids modulo bias. A singleton
legal set still consumes its one generator draw. No legal set consumes none.
No implicit Torch RNG state is needed under the retained dropout0 scope.

`Recipe{MaskMode::scattered}` calls the unchanged family's complete mask
generator, including its original shuffle implementation and draws. It is
the exact numerical control surface for the changed method.

Uniform legal starts do **not** imply uniform per-patch exposure: boundary
patches generally appear in fewer spans. This is part of the declared training
mask law. On the full observed research grid, q=4 of16 and the query is one
32-tick span. Sparse or partial-patch configurations may have different target
cell counts despite equal patch quotas; unsupported channels remain explicit.
Contiguous masking also changes temporal Q-pair support. It does not change
the fixed1/4/16 loss coefficients, reductions or empty-lag rules. For example,
a native15 H33/P6 two-patch span can have no lag16 pairs; that remains
unsupported rather than being expanded into a larger query.

## Reusable API

Include `embedding/encoders/causal_patch_bottleneck_mae/versions/v18.h` in
namespace `embedding::encoders::causal_patch_bottleneck_mae::v18`.

The public `Config`, `Encoder`, `RawHistory`, `FrozenScaler`,
`BankStandardizer`, fresh construction, settings, derived dimensions/counts,
TRAIN preprocessing fitter and optimizer types reuse the configurable family.
Settings retain `MODEL_TAG=CPB-family`, the architecture construction identity;
the separate V18 recipe identifies the training method. The
[research configuration](config/research.conf) is one benchmark geometry,
not the family interface's complete shape contract.

`make_mask(raw, cutoffs, config, base_seed, attempt, recipe)` returns the
ordinary hidden/visible/target/eligible tensors plus CPU B,C quota, legal-span
count and selected-start tensors. Unsupported or scattered selected starts
are-1. Legal-span counts are applicable only to contiguous mode; scattered
mode leaves them0. Counts distinguish insufficient patch support from an
observed channel with no legal contiguous span.

`forward_at(..., recipe)` delegates to the unchanged family forward under an
explicit supplied mask. It does not choose a new mask or change the objective.
`train_step(..., recipe, optimizer_options, context_deleted)` selects the
versioned mask and retains the base backward/clip/AdamW sequence. Its
`StepResult` extends the family's result with actual selected starts, legal
span counts, quotas, selected-patch count, unsupported/insufficient/no-span
channel counts and original-Q pair counts in lag1/4/16 order. An ineligible
batch consumes an attempt and reports its geometry without an optimizer step.

The [portable C++ trainer](../../../../scripts/cpb_v18_train.cpp) accepts
`--mask-mode contiguous|scattered`, configured typed TRAIN observation windows,
fresh or typed resume state, explicit stochastic stream seeds and total update/
attempt ceilings. It carries fitting, training, events and inference export
together. Application data admission, budgets and production orchestration stay
application-owned. A declared TRAIN archive alone does not prove split
membership.

The numerical source closure needs this version's include root, `family/include`
and the preserved `versions/v16/include`; the optional observation archive/CLI
also needs `code/data/observations/include`. Numerical code uses the inherited
LibTorch/CUDA SDK and C++20. No Python, pretrained ancestor, sibling checkout
or standalone environment is required by the fresh training component.

## Typed artifacts and continuation

Inference kind is `cpb_v18_mask_geometry_inference_v1`; training kind is
`cpb_v18_mask_geometry_training_v1`. Both bind the mask enum, complete recipe,
fixed temporal coefficient0.1 and the meaning of their nested structural
encoder payload. The retained family/V16 codec fields identify the numerical
container; the outer V18 recipe additionally records the actual training-mask
law. The nested payload is not a historical trained ancestor or a supported
way to discard the method identity.

`save_checkpoint(..., updates, recipe)` / `load_checkpoint(...,
expected_config, expected_recipe)` return `.numerical` and `.recipe`.
`save_training_state(..., identity, progress, recipe, options)` /
`load_training_state(..., identity, recipe, options)` additionally preserve
optimizer content/order/steps, progress, stochastic counters and frozen
preprocessing. Resume requires the exact version, mask mode, full caller
source/split/recipe and optimizer options. Inference cannot resume training;
family/V17 kinds cannot masquerade as V18 kinds. Applications add their own
authenticated ModelRef, publication, selection and lineage envelopes.

## Verification scope

The new [engineering fixture](tests/engineering_test.cpp) is designed to check
scattered-control masks, losses, named gradients, updates and optimizer parity;
physical span/quota and sparse abstention rules; partial final patches;
cutoff-first future poisoning; candidate exact three-step continuation; and
typed mode/version/type/optimizer/source refusal behavior at native36/native15.
Actual build/execution receipts determine which checks passed. Implemented
source alone is not evidence of engineering success or better encoder quality.

Keep the complete construction/fitting/training/masking/resume/export source,
configuration, dependencies/licenses, tests and research outcomes together in
any later source delivery. Existing completed V17 evidence and earlier source
packages retain their original byte identities.
