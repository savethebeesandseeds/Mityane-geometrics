# Requirements and evidence: imported forecasting and readouts

Record revision: `0.1.0` (18 October 2026). Linked
[Operation Contracts](CONTRACT.md): OR-11 `state-direct-horizon-v1`, OR-11
`native-embedding-forecast-v1`, OR-05 `common-observable-ridge-v1`.

**Decision: exploratory imported implementation; no application-quality
promotion.** This structural cleanup introduces no experiment, capture, fit,
quality measurement or reinterpretation of earlier results. The evidence below
records what the existing implementation/protocol can support and what is still
required. Passing artificial fixtures is engineering evidence, not scientific
evidence of market forecasting, useful representations or portfolio performance.

## Hypotheses and population

| Operation | Falsifiable application question | Evidence needed for a supported result |
| --- | --- | --- |
| OR-11 state forecasts | On a declared chronological population and horizon, does a frozen state/dynamics tuple improve matched fixed-target-coordinate prediction beyond declared mean, persistence and direct-embedding controls? | Same-source/same-information, same-row/horizon comparisons in the exact target coordinates; preregistered useful-effect and dependence-aware uncertainty rules |
| OR-11 native forecasts | On that population and horizon, do declared ridge/residual-MLP forecasts improve upon frozen TRAIN mean and persistence in the native embedding coordinates? | Immutable fitted models/transforms, legal issued predictions and matched held-out errors with the registered controls and uncertainty |
| OR-05 observable readouts | Do state-only or forecast-assisted features improve a declared unchanged observable over matched embedding/history, conventional, capacity and placebo alternatives? | Same mature target intersection, correct staged fitting, registered retention/diversity safeguards, and separate tests for utility, predictive-shaping attribution and forecast increment |

These hypotheses describe the existing prospective application lane. They are
not new experimental admission. The project remains domain-agnostic; the
imported market protocol is one conditional development study, not evidence about
other providers or populations.

## Data and provenance

The source import's upstream base is `savethebeesandseeds/cuwacunu_embedding`
revision `b15702b29753188482b11ca85e997006b503d0f0`, as recorded in the
[migration record](../../../doc/EMBEDDINGS_MIGRATION_2026-10-18.md). This base
revision does not identify the complete imported working-tree bytes; the
migration and cleanup inventories bind the actual files and local adaptations.
Its original
MIT attribution, model tags, frozen evidence and source capsules remain
authoritative for historical claims. A relocated or edited maintained source
tree has its own current source fingerprint; it must not be substituted for a
historically captured source identity.

The preserved prospective market registration consists of the
[numeric card](../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/latent_state_control_v1/numeric-card.json),
[amendment 001](../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/latent_state_control_v1/numeric-card-amendment-001.json),
[amendment 002](../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/latent_state_control_v1/numeric-card-amendment-002.json),
[campaign admission](../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/latent_state_control_v1/campaign-admission.json)
and [registration review](../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/latent_state_control_v1/registration-review.json).
Read the original card and amendments in order; later amendments supersede their
listed clauses while preserving original bytes. The separate
[issuance policy](../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/forecast_issuance_v1/policy.json) governs
the joint-frozen serving stage.

The registered lane fixes CPB-v3.market-01 native width 36, H128 windows, K4
oldest-first histories, one-minute spacing and horizons 1/16/128. Exact checkpoint,
sidecar, ModelRef, source/cache bytes and semantic ordering are pinned in those
records. January 22–28, 2024 is development-exposed; it is not independent TEST.
Representation TRAIN is January 22–23, later readout TRAIN January 24–25 and
development evaluation January 26–28, with complete-envelope purges and target
maturity. January 29–31 remains unassigned. Assumed close-time receipt is a
conditional observation model, not measured feed latency. Actual future runs
must record their exact observation population, exclusions, fitted models,
geometry, seeds, environment and output hashes; this record invents none.

## Controls, evaluation and success criteria

The inherited protocol declares TRAIN mean, persistence, K1/K4 ridge and a
per-horizon residual-MLP in native coordinates; fixed frozen/blocked/full
predictive arms; common-observable readouts; conventional history, capacity and
label-independent placebo controls; fixed retention probes; and state-diversity
guards. Point forecasts must be scored in one exact frozen coordinate system.
Raw native/state latent MSE from different spaces cannot establish superiority.
Readouts compare unchanged observables and identical eligible rows.

Chronological fitting, input availability, source support and target maturity
must be checked independently. The registered recipe forbids held-out tuning,
favorable seed selection and retrospective forecasts masquerading as legal
issued features. A favorable state-only result, point forecast, shaping
attribution or forecast increment is a separate claim with its own comparison.
The exact numeric criteria and amendment precedence remain in the preserved
registration; this summary cannot replace them.

The scientific amendment requires paired circular moving blocks of whole
origin vectors, all registered seeds, explicit claim-family multiplicity and
enough complete blocks/extreme-tail resampling support. Unsupported variance,
insufficient support/coverage, or uncertainty crossing the useful-effect rule
is inconclusive. New data, budgets, representations or confirmation require a
new prospective identity rather than silently extending the old study.

## Engineering witnesses and results

The maintained source includes the following artificial test targets. Execution
belongs in the root managed Linux container through the current documented
launcher; relocation does not authorize historical quality reruns.

| Targets | Bounded engineering behavior exercised |
| --- | --- |
| `test-predictive-v1`, `test-predictive-snapshot` | Gradient routing, pooled covariance, EMA/skipped updates, fixed-checkpoint affine folding, typed exact resume and checked artifact identities |
| `test-direct-control-v1`, `test-direct-control-artifact` | Native-coordinate comparator arithmetic, frozen fitting transforms, local RNG/resume and immutable publication/parent pins |
| `test-forecast-issuance-v1` | Synthetic fit-before-origin and availability boundaries, future-target independence, identity/parent/coordinate rejection and prediction publication |
| `test-latent-dynamics-v1-evaluation` | Saved-array mean/persistence/ridge, fixed-coordinate error/coverage, diversity/retention arithmetic and archive behavior |
| `test-latent-risk-v1-readouts` | Common TRAIN populations, input-only issuance, matched mature-target scoring, units and immutable readout/prediction round trips |

This record does not assert that every target was rerun during cleanup. A current
maintenance verification receipt must name the targets actually executed, current
source identity, command, container and logs. Earlier migration validation covers
only its recorded checks; it is not a substitute for a current full-chain pass.

**Scientific result status:** the preserved registration review explicitly
records zero market captures, model fits, Encoder calls/updates, quality
measurements and confirmation outcomes for that registration task. It lists
remaining engineering, source closure, control/reporting, capture and shared-study
gates. The issuance policy is prospective engineering only. No numeric market
forecast skill, state utility, predictive-shaping gain, observable-risk gain,
independent confirmation or portfolio eligibility is established here. Historical
Encoder results retain their original scope and do not establish these operation
claims.

## Reproduction and limitations

Use the preserved records above to identify the original source/configuration,
model/data parents and protocol hashes. The
[migration verification receipt](../../../doc/EMBEDDINGS_MIGRATION_VERIFICATION_2026-10-18.json)
records the initial import boundary. Historical absolute paths and captured
source closures are evidence identities; historical reproduction needs an
explicit reviewed path mapping. Never rewrite them to make a current layout
appear historically identical.

The numerical baseline/readout packages intentionally consume saved arrays and
do not validate an Encoder or raw application loader. Full pipeline claims need
end-to-end source/chronology/identity evidence in addition to component tests.
Deterministic forecasts provide no calibrated uncertainty, and the current
affine state/dynamics/readout composition has an exact affine folding identity.
Covariance regularization, low latent loss, retention arithmetic, model-source
ancestry and inherited dependencies do not independently establish application
quality. OR-12 policy value and identified intervention effects are outside these
contracts. Preserve failed and inconclusive results alongside favorable evidence.
