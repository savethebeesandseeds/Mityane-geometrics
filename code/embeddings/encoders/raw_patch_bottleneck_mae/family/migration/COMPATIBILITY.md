# RPB release and compatibility contract

This release adds `rpb-family-v1` wrappers, a fully bound portable example and
source packaging around the existing configurable v22/v20/shared mechanisms.
It changes no frozen numerical body, historical checkpoint, protocol, result,
trained-parent ancestry or research reference. New wrapper/runner paths and
hashes are new source identities. Historical receipts remain historical.

| Layer | Compatibility and required consumer action |
| --- | --- |
| C++ API | New family headers offer fitting, fresh construction, dimension/parameter discovery, encoding, explicit update/skip and typed save/resume. Existing v20/v22 APIs remain distinct. Native namespace/include paths are retained in the bundle. Consumers integrate through their root build and recompile under their runtime. |
| Configuration | Supported C/H/F/P/D/capacity/IDs vary by configuration; saved artifacts retain exact settings. A different valid geometry is a new configuration, not a claim that old weights load. Changing H also changes harmonic coordinates even when private padded H matches. |
| Numerical mechanism | Existing fitter, patch masker, waveform Huber, gradient routes and fixed optimizer are reused. A changed fitter arithmetic, masks/targets, reduction, optimizer or random algorithm needs a named variant and fresh evidence. The example's sampling/context policy is identified separately from historical recipes. |
| Preprocessing | Fit on the consumer's actual admitted TRAIN; preserve exact fitted content and identity thereafter. Different weighting/order can change rounding. A fresh fit creates a new numerical instance; close statistics do not establish exact continuation. |
| Inference artifact | Existing typed v22 inference format is unchanged; v20/v21/v22 tags/layouts cannot substitute for one another. It stores model/config/scaler/update count, not AdamW. Weights/config/layout/fitted state must agree, and consumers retain independent archive hashes. |
| Training artifact | Existing `rpb_content_ordered_relation_training_state_v1` is unchanged. One complete fixed AdamW group, exact model order, finite matching moments and correct progress are required. Inference archives cannot resume training. |
| Exact continuation | Same supported runtime/device, config, scaler bytes, full recipe and source/split identities plus independently checked row/mask/context seeds. Saves occur between completed attempts. No ambient RNG, loader iterator, partial gradients or partial attempts are restored. Cross-device/SDK/compiler/hardware exactness is unverified. |
| Stopping budget | The portable example permits an increased total stopping budget after resume, without relabeling completed work or changing mechanism identity. An application may instead bind an immutable original budget in its own recipe. This is not an artifact-format equivalence claim across recipes. |
| Trusted application artifacts | The family raw codecs do not provide admitted data, trusted publication, ModelRef, selection, capture or downstream-parent compatibility. Consumers retain their own authenticated envelopes, actual budgets/recovery and exact descendants. |

The portable reference row sampler is counter-seeded MT19937_64/modulo with
replacement and has no extra context deletion. Historical v20/v22 studies and
the typed-state fixture use their own row/context algorithms. Identical seed
integers do not make these recipes or trajectories interchangeable. The example
does not reproduce native36 or v30, which retain protected trained-parent and
frozen-fit dependencies and different learning rules.

Any changed trained encoder or fitted preprocessing requires new representations
and compatible fitted descendants or an independently validated mapping.
Successful loading or equal D proves neither shared coordinates nor application
usefulness. End-of-window dependency semantics remain full retained context;
this release supplies no prefix-causal promise or CPB equivalence.

For an import, retain the independent manifest pin, all origin/destination
path/hash mappings, source/recipe adaptations, runtime/binary identities and
recipient checks. Source closure and byte verification are necessary source
acceptance, not data admission or a scientific finding. Original-path receipts
do not certify relocated binaries. The portability and delivery receipts state
their own measured scope and any remaining gaps.
