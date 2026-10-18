<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE. -->
# Saved-array latent dynamics evaluation v1

This complete physical version owns its header, implementation, tests and build
fragment. Future versions copy their implementation into a new version folder.
It executes no encoder, ingests no market sources and changes no representation.

`ArrayBank` contains frozen native embeddings, or states from one frozen Target
State Representation Network. `coordinate_id` must bind the exact checkpoint,
target mapping, preprocessing and reference policy. Different coordinates cannot
share fitted baselines or scores. The application adapts its timestamp/pair ledger
to numeric row IDs; it retains the original immutable source identity separately.

`fit_baselines` accepts K1/K4 suffix histories and fits an unpenalized target mean
and weighted ridge per declared horizon. Feature mean/population standard deviation
are fitted on the same eligible TRAIN pairs only. The objective is weighted SSE
plus `ridge_penalty * squared_weight_norm`; the intercept is unpenalized. Primal
and dual solutions are equivalent. Roles, raw support, true candle spacing,
origin availability, complete support and target maturity are checked before fit.
VALIDATION/TEST rows are excluded even if supplied with TRAIN associations.
No held-out selection, normalizer fitting or horizon rescue occurs here.

`predict_baselines` emits mean, persistence and ridge, retaining every association
and its input-side reason mask. The fit must be available at the decision time.
Persistence reads the current row in the same target coordinates as the future
target. Target values, masks and eligibility are never consulted to issue a
forecast. Missing targets remain prediction associations; `score_predictions`
checks them independently when targets mature. Passing a K4 ledger whose input
eligibility already rejected an older history to K1 does not undo that ledger
decision: the runner must construct each declared history eligibility correctly.

Scoring uses fixed positive reference scales, nonnegative coordinate weights and
positive pair weights. The reported MSE is the weighted mean squared normalized
coordinate error; MAE stays in native coordinate units. Skill uses persistence
on exactly the scored rows and weights. Near-zero persistence MSE yields N/A.
Coverage is scored rows divided by target-eligible rows; total/input/target/scored
counts are separate. Reports retain each horizon and complete/partial/stale plus
actual-overlap strata. Overlap fractions come from the source ledger rather than
an assumed decomposition of a nonlinear encoder. Dependence groups survive the
archive; these descriptive scores claim no independent temporal confidence.

For a frozen neural state/dynamics forecast, construct a `PredictionSet` with the
same coordinate identity and associations and call the same scorer. `diversity`
reports native/fixed-reference sample standard deviations, entropy effective rank,
largest covariance factor fraction and the finite-bank rank bound. A constant bank
retains zero diversity. `retention` evaluates saved fixed-probe outputs against
unchanged observables, without fitting anything. Neither helper selects thresholds
or establishes future-risk usefulness. Application-specific future labels and
matching of arms belong to the application, outside this package.

Typed CPU prediction/model archives publish without replacement. Prediction files
retain invalid rows, complete histories, ledger reasons, weights, overlap and
dependence groups. A numeric zero in an invalid row never becomes an observation.

Run synthetic correctness checks inside the existing managed container:

```sh
bash code/scripts/task.sh latent-dynamics-evaluation-v1 -f Makefile \
  -f code/operations/forecasting/evaluation/versions/v1/baselines.mk \
  -j2 test-latent-dynamics-v1-evaluation
```

These checks establish engineering behavior only. No market baseline result,
encoder admission, experiment selection or advancement evidence is supplied.

## Common future observable readouts

`risk_readouts.h` accepts generic immutable `ObservableTargets`, with per-label
support/availability/validity, and saved `FeatureArm` arrays. It knows no market
label formula or encoder. The application binds formulas, units, semantic assets,
roles and input information contracts. Current embedding/state, matched history
and forecast-augmented arms can have different declared widths; every head uses
the same frozen ridge recipe. This controls the head recipe and eligible rows,
without claiming equal capacity between different input widths.

For each observable, fitting intersects eligible input rows from every arm with
mature TRAIN targets whose complete declared support remains inside TRAIN. Each
arm fits its own TRAIN-only feature transform, an unpenalized target mean and
ridge weights on exactly that common population. The target mean and reference
standard deviation are identical across arms. There is no held-out tuning.

`predict_risk_readouts` receives origin metadata and feature arrays only: it
cannot inspect target values, validity or maturity. Artifact/readout fit cutoffs
must precede issued predictions; invalid inputs and unfitted heads retain
explicit reason masks. Retrospective TRAIN features are legal supervised-fitting
inputs and must be declared as such; they are never issued out-of-sample forecasts
or permissible surprise features. `score_risk_readouts` rejects TRAIN scoring and
compares every arm on the exact same per-observable intersection of mature targets
and issued predictions. MAE/MSE retain observable units; normalized MSE uses the
frozen TRAIN target scale. Skill compares the same-row TRAIN-mean reference,
with near-zero denominators N/A. Separate total, target, input and common counts
make abstention visible. Typed immutable heads/predictions have CPU load/save APIs.

```sh
bash code/scripts/task.sh latent-risk-readouts-v1 -f Makefile \
  -f code/operations/forecasting/evaluation/versions/v1/baselines.mk \
  -j2 test-latent-risk-v1-readouts
```

This is a fixed linear readout backbone. Conventional observable-history models,
equal-capacity controls, forecast placebos, historical retention guards and
temporal-block uncertainty remain separately declared comparisons. A favorable
readout alone does not authorize an advancement gate or portfolio claim.
