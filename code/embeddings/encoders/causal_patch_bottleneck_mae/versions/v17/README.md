# CPB-v17: configurable fixed-weight temporal learning recipe

## Channel axis

cuwacunu_torch `interval_type_e` is the standard. One window is one instrument. Each channel is one time resolution of that instrument: minutes, hours, weeks, and the other kline intervals. The channel id is that interval's torch ordinal. Different instruments are different rows. A stored id set `{0,1,2}` is the withdrawn instrument binding from historical checkpoints. It is not this axis, and scores measured on it stay scores of that binding. The contract is `code/embeddings/encoders/include/embedding/encoders/timeframe_channel.h`.

This candidate builds on the corrected, configurable CPB component. Its sole
learning change is an explicit nonnegative temporal coefficient, default **0.03**,
over the same Gaussian NLL plus fixed-mean hidden temporal Huber1/4/16 objective.
The study control uses coefficient **0.1** through the same interface. The
bounded paired research study is complete and valid; its predeclared
continuation criterion **failed**. No new best encoder was established.
**V16 seed 946103/final 2048 remains the selected research reference.**

The maintained `family/` and frozen V16 implementation stay unchanged. V17 uses
their fresh construction, dimensions/capacity validation, causal cropping,
stable TRAIN-only fitter, original mask/support semantics, objective components,
gradient routing, one global clip and AdamW primitives. It needs both family
and V16 include roots in addition to this version's `include/`. The optional
observation archive/CLI also needs the generic observation-data include root.
Preserve the root MIT license and inherited notices with every source delivery.

## Engineering evidence and research status

The managed-container CUDA engineering check passed **859 assertions**, comparing
**126 named gradient tensors**, with **16 physical synthetic updates**,
**2 preprocessing fits** and **23 expected refusals** across native36 and
native15 configurations. It checked the coefficient 0.1 control against the
unchanged family, actual updates and optimizer moments, coefficient 0.03 typed
resume, inference export/load, causal cutoffs and wrong-recipe/kind refusals.
These are engineering results; they do not promote encoder quality.

The completed study compared three fresh matched coefficient 0.1/0.03 pairs at
256 successful updates per arm: **1536 scientific updates in 1536 attempts**,
using 12 planned fresh128/resume256 native processes. All six final readouts
had complete support and passed independent scalar arithmetic audits.

| Past-window task | Median candidate-minus-control R2 | Nonnegative pairs | Declared interpretation |
| --- | ---: | ---: | --- |
| Trend | -0.003029098 | 1/3 | Fails both trend conditions. |
| Activity | -0.008995412 | 0/3 | Fails both activity conditions. |
| Volatility | -0.002418553 | 1/3 | Passes the median floor of -0.01. |
| Relation | +0.003185947 | 2/3 | Descriptive; not a continuation condition. |

The TRAIN-only zero-update diagnostic did not support raw temporal-gradient
dominance at the selected frozen V16 state. Global learned-gradient cosines
were positive in all eight batches; the backbone subgroup had four negative
cosines. These local raw gradients do not measure AdamW-preconditioned update
contributions or identify the cause of earlier readout changes.

The immutable [study results](../../../../evaluation/protocols/cpb_v17_balance_v1/RESULTS.md)
and [compact record](../../../../evaluation/records/CPB-v17.balance-001.json)
retain full-precision scores, every pair, actual work, identities and limitations
and travel in the final source delivery. This known-January 256-update screen
does not establish full-budget superiority, timing retention, fresh confirmation
or application utility. No reference or application selection changed.

## Public use

Include `embedding/encoders/causal_patch_bottleneck_mae/versions/v17.h` in
namespace `embedding::encoders::causal_patch_bottleneck_mae::v17`.

`Config`, `RawHistory`, `Encoder`, `FrozenScaler`, `BankStandardizer`, dimensions,
parameter counting, settings, fresh construction, preprocessing fitting and
optimizer construction reuse the configurable family's public types. Settings
keep `MODEL_TAG=CPB-family`, which identifies the architecture. The separate
`Recipe{temporal_weight}` and typed artifact envelope identify CPB-v17 learning.
`config/research.conf` is one comparative geometry, never a universal shape.

Use the free `v17::forward_at(..., recipe)` or `v17::train_step(..., recipe,
optimizer_options, context_deleted)` to apply the versioned learning objective.
The underlying family model's own member `forward_at()` retains the family
default coefficient 0.1; it is an implementation primitive, not the V17 objective
entry. Encoding itself is independent of this loss coefficient.

The same complete training identity must bind the actual source/split,
preprocessing, batching/sampling/context policies and stochastic counter streams.
Training state records optimizer moments/content/order, progress, configuration,
initialization and frozen preprocessing. No implicit Torch CPU/CUDA RNG state is
saved; exact continuation retains the family's dropout0/counter-stream scope.
Inference state is not a resumable training state.

## Artifact contract

`save_checkpoint(..., updates, recipe)` and `load_checkpoint(..., device,
expected_config, expected_recipe)` use kind
`cpb_v17_weighted_temporal_inference_v1`. Loading returns `.numerical` (the
family configuration/model/scaler) and `.recipe`.

`save_training_state(..., identity, progress, recipe, options)` and
`load_training_state(..., expected_config, identity, recipe, options)` use kind
`cpb_v17_weighted_temporal_training_v1`. Loading requires the exact version,
coefficient, application recipe, optimizer settings and saved progress bindings.
It returns model/scaler/optimizer/progress and recipe. Old raw family loaders
refuse the new outer kind. Equal dimensions do not establish compatibility.

The outer envelope is authoritative for the actual V17 recipe. The nested
`cpb-family-structural-state-container-v1` payload is explicitly a **structural
model/configuration/fitted-state container**. It retains original family/V16
codec identifiers, including legacy 0.1 objective metadata, without claiming that
those identifiers describe the new instance's actual training. This meaning is
itself authenticated in both archive types. The payload is not a historical
V16 trained ancestor, and extracting it is not a supported application import.

Applications wrap the typed numerical artifacts in their own authenticated
publication, ModelRef and selection/lineage contracts. The complete V17 source
closure and training method travel with any export; no checkpoint, dataset,
SDK, sibling checkout or application-specific training pipeline is a required
fresh ancestor. Numerical behavior, engineering verification and scientific
quality remain separately evidenced under the study protocol.

The [delivery guide](migration/README.md) explains the complete source package,
fresh/resume CLI, method dependencies and independent recipient verification.
