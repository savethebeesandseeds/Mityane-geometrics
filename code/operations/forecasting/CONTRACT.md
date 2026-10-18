# Imported forecasting and observable-readout contracts

Contract revision: `0.1.0` (18 October 2026). Status: imported exploratory
implementations, with application evidence governed separately. This document
organizes maintained source under [ORR-DESIGN-001 v0.1.0](../../../doc/operations_over_representations_design_spec_v0.1.0.md)
and links its [Requirements-and-Evidence Record](EVIDENCE.md). It does not revise
the imported numerical model versions, frozen protocols or trained identities.

## Operation identities and ownership

| Family / instance | Primary question | Maintained responsibility |
| --- | --- | --- |
| OR-11 / `state-direct-horizon-v1` | What future target-state vectors follow from the available state history at each declared horizon? | `latent_dynamics`, its `predictive_training` objective/checkpoint/serving helpers, and checked forecast issuance |
| OR-11 / `native-embedding-forecast-v1` | What future native embedding vectors follow from the same permitted embedding history? | Saved-array mean/persistence/ridge baselines and `direct_embedding_control` residual-MLP comparator |
| OR-05 / `common-observable-ridge-v1` | What externally defined observable can each permitted representation or forecast-assisted feature arm estimate? | Saved-array `risk_readouts` fit, input-only prediction and matched scoring |

The direct embedding **control is an experimental comparator** that outputs a
forecast. It chooses no action and supplies no OR-12 decision policy. The online
and target State Representation Networks remain representation-provider
dependencies in `code/embeddings/state_representation`; their use in predictive
training does not merge representations, forecasts and readouts into one claim.
The imported `cuwacunu` namespaces and `v1` model tags retain their meaning.

## OR-11: state and native-embedding forecasts

**Input contract.** Each run supplies exact representation and preprocessing
identities, model/checkpoint and sidecar pins, frozen transforms/reference
scales, oldest-first history, distinct positive horizons, and a chronological
origin/pair ledger with source support, roles, availability and exclusion reasons.
For state dynamics, `AffineDynamics` consumes `[B,K,d]` and emits `[B,Q,d]` in
the supplied horizon order. These are independent direct forecasts; `v1` does
not recursively roll out. Predictive training consumes saved native embeddings
`[B,K,D]`, future embedding targets `[B,Q,D]`, a target-side eligibility mask
`[B,Q]`, and a separately declared distinct-origin regularizer population.
Eligibility and raw temporal support must be established outside the numerical
trainer. Future data and target masks do not enter the dynamics forward call.

The native-coordinate comparators consume their declared K1/K4 histories and
return forecasts in the exact frozen native coordinate system. The residual-MLP
control fits a separate model per horizon. A model version or equal vector width
alone never establishes coordinate compatibility.

**Output contract.** Forecast arrays preserve every origin/horizon association,
coordinate identity, validity/reasons, fit cutoff and provenance. A checked
state forecast bank belongs to one exact frozen Encoder/online-state/target-state/
Dynamics tuple. Outputs represent predicted future states, not observed encodings.
These deterministic versions provide point forecasts without calibrated
uncertainty. Invalid rows retain finite zero storage and explicit reasons; zero
storage is not a valid forecast or observation. Scoring reports fixed-reference
normalized MSE, native-unit MAE, matched persistence skill, coverage and separate
input/target/scored counts.

**Required representation structure.** History and targets must have compatible,
versioned coordinates and enough task-relevant predictive information at the
declared horizon. Any mapping from native embeddings to a target-state space
must use that exact frozen target map. Losses from independently fitted target
spaces cannot rank their usefulness. State normalization, metric/reference scale,
pooling, order, dimensions and checkpoint form part of the representation identity.

**Other dependencies.** Frozen saved embeddings; a separately admitted temporal
ledger and dataset; TRAIN-only fitted transforms; explicit source/fit/maturity
cutoffs; a selected representation provider; immutable model/source identities;
and the application protocol's controls and evaluation rules. The simple affine
forward method does not validate timestamps or source support. Checked issuance
must independently validate them. The imported issuance policy admits only a
joint-frozen single stage: every component fit cutoff strictly precedes the
origin; a later decision time does not legalize a later-fitted model. Native
embedding comparators have their own source, parent, transform and fit pins.

**Scope of claim.** These are predictive operations and engineering mechanisms.
Predictive-gradient attribution arms (`frozen_state`, `blocked_prediction`,
`full`) describe controlled changes within model training, not identified
intervention effects on a market or another external system. Affine serving
folding is an algebraic equivalence check. Latent error, diversity, covariance
regularization and exact resume do not establish observable utility or decision
value.

**Failure and abstention.** Reject mismatched archives, parents, coordinate
identities, configuration, policies or mixed independently refitted folds. Retain
unavailable, incomplete, nonfinite, role-incompatible or unsupported origins with
reasons; never substitute neighboring observations. Future target validity and
maturity affect fitting/scoring separately and cannot determine issuance. With
an all-invalid target mask, predictive training updates neither trainable models
nor EMA state. Unsupported reference variance makes relative skill unavailable;
insufficient coverage, chronology or uncertainty support prevents a scientific
decision. Cross-fitting, new representations, new populations and probabilistic
or action-conditioned models require their own contracts/prospective protocols.

## OR-05: common-observable ridge readouts

**Input contract.** `ObservableTargets` identifies formulas, units, source/split
identity, `[N,L]` CPU float64 target values, per-label validity, support and
availability. Each `FeatureArm` carries `[N,P]` CPU float64 features, origin IDs,
validity, information contract, support/availability and latest upstream artifact
fit cutoff. A fixed `UtilityRecipe` declares TRAIN boundaries, readout cutoff,
ridge penalty and scale safeguards. Representations, histories and forecast
augmentations are separate named arms with declared widths.

**Output contract.** `fit_risk_readouts` produces frozen arm-specific feature
transforms and affine ridge heads using the same common mature TRAIN population
per observable, with an unpenalized target mean. `predict_risk_readouts` takes
origin metadata/features without future target values or eligibility, returning
target-unit estimates plus validity/reasons. `score_risk_readouts` compares arms
on the same per-observable intersection of mature targets and issued predictions
outside TRAIN; it reports MAE, MSE, frozen-TRAIN-target-scale normalized MSE,
same-row TRAIN-mean skill, coverage and counts. Immutable save/load formats retain
identities and frozen heads.

**Required representation structure and dependencies.** Features must contain
accessible information about the declared observable under a controlled readout
recipe. External label definitions, correct chronology/availability, TRAIN-only
transforms, source/model identities and application controls remain required.
Using one ridge recipe across arms of different widths does not establish equal
capacity. Conventional, capacity and forecast-placebo comparators are separate
registered requirements where the application protocol calls for them.

**Scope of claim.** A properly evaluated result can support task association or
prediction for a named observable, population, horizon and representation. It
does not establish OR-11 point skill, predictive-shaping attribution, a forecast
increment over the identical unaugmented history, independent confirmation,
causality or OR-12 decision value without the corresponding separate tests.

**Failure and abstention.** Reject incompatible target/feature identities and
illegal chronology. Preserve invalid, unavailable and unfitted predictions with
reasons. Do not score TRAIN as held-out evidence or use arm-specific favorable
rows. Near-zero reference variance/loss yields unavailable normalized/relative
claims; absent common support, coverage or uncertainty makes the scientific
decision inconclusive. Retrospective representation-TRAIN forecasts cannot be
relabeled as later-stage issued risk-TRAIN features.
