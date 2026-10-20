# Synthetic study Requirements-and-Evidence Record

**ID:** `OR06-SYNTH-RESP-001-RER`, revision `1.0.0`.  
**Parent:** [OR06-PCR-RER 0.1.0](../../EVIDENCE.md).  
**Protocol:** [OR06-SYNTH-RESP-001 1.0.0](PROTOCOL.md).  
**Status at authoring:** prospective; results belong in a separate report.

| Required field | Actual binding |
| --- | --- |
| Hypothesis | Mixed pipeline per-seed source-unit MAE <=0.02, raw-polynomial equivalence <=1e-9, pooled error <=20% zero baseline; full in-support coverage, stress abstention and numerical audit |
| Population and provenance | Project-authored finite synthetic two-coordinate generator, LCG seeds 1103/2207/3301, dyadic denominator 4096, fixed noisy training response; exact counts/splits and unique-b reservation in INTERFACE/configuration |
| Representations and model | Fixed mixed, raw-polynomial, raw-linear and lossy feature maps; TRAIN-only population standardization and OLS heads; no inherited/pretrained provider. Actual generated model/space/pipeline/response IDs are hashes of saved bytes in the exclusive attempt |
| Geometry and information horizon | Dense vector features with explicit Euclidean geometry/no normalization before the saved TRAIN standardization; static non-temporal model-input response. The support origin and target horizon have no historical deployment meaning |
| Controls and baselines | No-change, both signs/two in-support magnitudes, +2 support stress, zero-response, same-information raw-linear, equivalent-capacity raw-polynomial and deliberately lossy representation |
| Evaluation protocol | 512/128/256 units per seed, all models sealed before assessment generation, no tuning/search, all changes retained, independent saved-model and analytic response oracles |
| Measures of success | Conjunctive prospective screen in PROTOCOL/configuration; error averaged per source unit then per seed, explicit total coverage/refusal accounting, numerical tolerance 1e-10 |
| Robustness and uncertainty | Three independently generated/fitted seeds; signs/magnitudes/no-change/support stress; within-seed source bootstrap 2000 resamples seed 9001, conditional on fitted models, all seeds reported |
| Results and limitations | No result at authoring; frozen source/config/binary identities and actual run records are saved before execution. Outcome reported separately, including negative/inconclusive/invalid cases. Static synthetic scope only; no causal or real-world interpretation |
| Reproduction | Root task entry, selected native binary, source closure, environment, dataset snapshots/state, models/scalers, actual payload hashes, native outputs, independent audit and operation-local append-only index |
| Decision | Unassessed prospectively. A valid completed positive/negative/inconclusive benchmark can close the bounded evaluation; failed/invalid execution cannot |

Admission resolves actual byte identities in `freeze.json`, `model-seal.json`
and the saved file manifests. Those run-bound records link the reusable
contract/RER rather than changing its API identities. Permissions trace the
user's instruction to continue Hx06's next goal and the bounded synthetic
implementation described in this protocol; unrelated research permissions
and pauses remain independent.

The code and data generator are authored here under the repository's MIT
license. There are no external data access, retention or model license
dependencies. The actual data and fitted artifacts remain ignored local files.
