# Forecast issuance v1

This additive boundary issues state/Dynamics forecasts from a complete checked
frozen Encoder, online State Representation Network, Target State Representation
Network and Dynamics Model tuple. The prospective policy is
`doc/forecast_issuance_v1/policy.json`, SHA256
`18fcaeab478e0b66dfe2fca132cc4c5d8ae3de368eb4a74d7428fdf9bfef2805`.

Only its joint-frozen single-stage policy is supported. Online state, target
state and Dynamics share one archive-proven fit cutoff; Encoder fitting cannot
be later. All four cutoffs must strictly precede each origin. A delayed decision
may admit observations available by that decision, but cannot admit a model fit
at or after its origin. Equality is unavailable.

`make_contract(timeline, pair_spec, fits, encoder_verified, predictive_verified)`
derives complete provenance and the state width from checked inference archives.
`Issuer(contract, shared_ptr<const CheckedInference>, encoder_verified,
predictive_verified, original_captured_timeline)` owns its guarded serving lifetime. It verifies raw archive
hashes, canonical sidecar pins, complete role-correct ModelRefs, their exact
parents, BundleRef, target coordinates, archived fit/source/ledger/transform
provenance and ordered model geometry. `contract_digest` binds every field;
`source_files` lists the complete local implementation dependencies for capture.
The original captured timeline's canonical bytes are authenticated once against
the immutable archive source hash. Per-origin issuance never hashes future data.

`issuer.issue(timeline, PairOrigin, roles, original_input_reasons)` exposes no
future target association. It independently reconstructs every exact past input
window and checks actual support, timestamps, roles, validity and availability.
Every requested history slot remains present with explicit reasons; invalid
origins return finite zero `[Q,S]` CPU float64 storage. Future values, masks,
availability, identities, missing targets and target eligibility cannot determine
issuance. Target maturity is independently required for fitting and scoring.

One bank belongs to one exact compatible frozen tuple and target coordinate
system. Cross-fitting and coordinate/fold pooling require a new prospectively
registered policy and are rejected here. Retrospective `fitting_forecasts`
remain separate diagnostics. Downstream risk-head chronology and market-study
execution remain separate requirements.

Run focused synthetic checks inside the existing managed development container:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'forecast-issuance-v1', '-f', 'Makefile', '-f', 'code/latent_market/forecast_issuance/versions/v1/v1.mk', 'test-forecast-issuance-v1')
```

The tests use artificial checked archives and synthetic timestamps; they do not
train or measure a market model or establish scientific skill.
