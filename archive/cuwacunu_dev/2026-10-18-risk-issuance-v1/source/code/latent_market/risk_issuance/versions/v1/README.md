# Risk issuance v1

This version adds a frozen upstream stage, later risk-head fitting and later
label-free risk prediction. The prospective engineering policy is
`doc/risk_issuance_v1/policy.json`, SHA256
`a7f52e1bc77f613041f56c8ce1ce7bfded42046dd4a9d2c31dfe3c7c2277220f`.
It changes no evaluated numerical library or scientific threshold.

`FeatureBank::build(stage,timeline,pairs,issuer,shared_serving,issued_directory,
issued_manifest_pin,native)` authenticates exact captured source, original
associations, checked state/target/Dynamics tuple, issued forecast publication
and checked direct-embedding control comparison/output pins. Seven engineering
arms use current/history native embeddings and online states, with separately
identified issued native or target-state forecasts. All arms use the same
eligible origin intersection. Stage roles are separate from original capture
roles. Every origin remains recorded with explicit support/timing exclusions.

`FittedStage::fit(bank,targets)` first selects head-TRAIN metadata. It copies only
eligible TRAIN features and mature, supported selected label values into a
TRAIN-only numerical view before calling the unchanged affine ridge library.
Evaluation masks, availability, labels and feature values do not enter fitting.
The registered ridge penalty is 1 and normalization floor is 1e-8. Selected
label identities and maturity/support witnesses remain available for audit.
`ObservableTargets` carries caller-authenticated label/formula/source authority.
An application adapter must pin the original risk archive and join its original
origin associations before calling the core; the synthetic fixture does this.

`head.predict(bank)` receives no labels. It enforces strict head-fit-before-origin
timing even with delayed decisions, preserves unavailable predictions as finite
zeros, and binds one exact admitted saved-array bank. This does not claim live
serving, cross-fitting, coordinate pooling, market utility or full campaign
completion. Frozen feature/head tensors remain private; all tensor snapshots
are deep copies. Checked publication reconstructs heads through a private friend
factory and validates the fixed recipe, population, identities and tensor state.

Run only in the existing managed container:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'risk-issuance-core-v1', '-f', 'Makefile', '-f', 'code/latent_market/risk_issuance/versions/v1/v1.mk', 'test-risk-issuance-v1')
```

The synthetic fixture uses two bounded artificial CUDA upstream updates and
two CPU direct-control updates to construct checked inputs. Head fitting and serving execute no Encoder
and perform no upstream update. The full registered market comparison panel,
additional controls and scientific gates remain separate work.
