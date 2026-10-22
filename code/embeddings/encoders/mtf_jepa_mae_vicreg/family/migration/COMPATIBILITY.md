# MTF compatibility and acceptance

| Surface | Compatibility contract |
| --- | --- |
| Original MTF-v1 | Core parameters, registration names, historical workflow, raw v1 format and evidence retain their original identities; this delivery does not rewrite them |
| New API | `mtf_jepa_mae_vicreg::family` is a new typed portable layer over retained numerical primitives; source callers use its canonical include |
| Configuration | Supported dimensions/depths/masks are configurable; construction validates architectural constraints. Existing weights still require their own exact configuration |
| Numerical behavior | The network/mask/loss helpers are reused. New CPU/F32 preparation, one intra-op thread with bound inter-op setting, sampler, explicit optimizer options and update-seed recipe are separately identified; shared primitive reuse does not establish legacy trajectory equality |
| Raw training state | `mtf-family-training-v1` belongs to the new session/recipe contract. Original v1 or adapter-control archives are not interchangeable resume states |
| Inference state | `mtf-family-inference-v1` carries model-only registered state and bound configuration/preparation identity. All registered teacher/heads may be retained but online eval encoding is served; no optimizer or training continuation is provided |
| Exact continuation | Supported only at completed-update save boundaries within the same executable SHA256, source/configuration, prepared data/support/order, recipe, total budget and recorded CPU/F32/thread runtime; same CPU/backend/hardware and pinned SDK assumptions apply; no mid-update, cross-runtime or cross-device bitwise guarantee |
| Application artifacts | Raw state is not an authenticated application envelope, ModelRef, selected-artifact record or admitted calibration population |

Identity preparation has no fitter. Consumers requiring normalization must bind
their actual preprocessing and data membership under their own recipe. Replacing
the portable identity policy with the historical evaluator's scaler is a method
change requiring new identity/evidence. Changed fitted preprocessing generally
requires refitting/retraining and regenerated representations.

Resume checks complete declared bindings. Reusing a recipe ID with different
source/data/split/optimizer/configuration does not make it compatible. The
portable reference binds the original total budget; increasing it creates a new
recipe rather than silently extending the declared run. A successful raw load
cannot prove the truth of caller declarations or operational data admission.
The current session API supplies no warm-start/recipe-rebinding operation;
changed-budget continuation from an old training archive is unsupported.
Inference loading checks expected source/configuration/preparation/tensor pins
without demanding the training executable hash; serving in another compatible
binary still needs its own bounded integration evidence.

Session updates touch global Torch RNG/thread state and must be serialized.
An error after AdamW/EMA mutation has no rollback guarantee. Preserve that failed
state for diagnosis and recover from trusted valid saved state through the
application's recovery procedure. The raw archive helper is not a durable
checked-publication journal. CLI outputs refuse existing paths; raw API writes
use the inherited neighboring-temporary-file/rename behavior.

Changed learning semantics—including optimizer algorithms, mask eligibility,
loss reductions, teacher update order or RNG draw order—need explicit variant
identities and appropriate proof. Preserve older checkpoints, receipts and
failures as historical. An export/import relocation gives a new path-bound
source identity; copied receipts certify their original source/runtime only.

A consumer accepts the independently retained bundle manifest pin, verifies
all member hashes and origin/destination mappings offline, integrates its own
managed build, and runs bounded mechanism/load/refusal checks. It separately
admits real features/populations/TRAIN membership, budgets/scaling/recovery,
trusted artifact selection, captures, exact fitted parents and application
utility. Equal output width does not imply learned-coordinate compatibility.
New encoder instances require compatible representations and descendants.

`Settings.threads=1` means one intra-op Torch thread. The inherited inter-op
setting is recorded and bound rather than forced to one; it was 12 in the
primary proof. See the [bounded protocol](../verification/PROTOCOL_2026-10-22.md),
[actual results](../verification/RESULTS_2026-10-22.md) and
[engineering receipt](../verification/engineering-proof-2026-10-22.json).
API/configuration support, engineering proof, scientific quality and application
acceptance remain distinct claims.

The completed isolated build/run used staged bundle
`mtf-family-2026-10-22-001`, manifest SHA256
`a3cc64dc41b944db2e495a7bd27eb8540f37b4359cba575dd9cb12e410a64975`.
The final `mtf-family-2026-10-22-002` documentation/receipt successor preserves
all 33 admitted source/configuration/fixture/tool pins and reuses that evidence.
The [delivery receipt](DELIVERY_2026-10-22.json) records the tested manifest/path,
separate binary/runtime and final manifest; it asserts no rerun at the final path.
