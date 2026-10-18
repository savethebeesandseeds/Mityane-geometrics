# Continue the RPB encoder on 8 October

Today's fixed experiment is complete. **RPB-v4 remains active.**
**RPB-v9 — Global bottleneck with native view agreement** is the last measured
candidate and failed all six numeric guards. Its implementation and recorded
calculations passed actual CUDA admission and an independent 74,721,050-check
audit; the training recipe itself performed badly.

Read the [measured report](../code/encoders/raw_patch_bottleneck_mae/NATIVE_VIEW_AGREEMENT_VALIDATION_DIAGNOSTIC.md)
and [exact numerical summary](results/native_view_agreement_validation_v1.json).
Mean linear timing accuracy is 59.21875% intact and 57.03125% under extra
deletion. Mean standardized TRAIN/VALIDATION MAE is 29.194490 / 29.371655.
Master 4404 has roughly 142 / 143 MAE; the other four are roughly 0.9–1.0, also much
worse than v4's approximately 0.07. Keep all runs; the large mean is not grounds
to drop the unstable instance. No TEST/stress, selection or promotion occurred.

The audited component means already show that weighted agreement dominates the
recorded total in every run: roughly 156–1,287 versus 0.52–0.63 ordinary Huber in
the four less unstable runs, and roughly 11 million versus 15 in 4404. The nominal
0.05 coefficient did not make this a small auxiliary. Scalar magnitudes alone
do not establish gradient dominance or the cause of the failure; investigate
the calibration scales and early traces before changing the objective.

## First work tomorrow

1. Freeze one short TRAIN-only diagnostic card and exact input role list before
   opening additional payloads. Use the saved point0/512 calibration vectors,
   support and per-update component evidence from all five v9 runs. Do not
   train encoders, refit heads, reopen TEST/stress or tune this rejected recipe.
2. Inspect the range/floors of the fixed point0 loss scales, and the early
   reconstruction/agreement/variance/total/gradient-norm traces. Locate the
   first deterioration in 4404 and compare it with every other master. Test
   whether loss scaling or gradient allocation plausibly overwhelms the
   ordinary reconstruction anchor. This is a hypothesis, not an established cause.
3. Record the diagnosis and propose one new bounded mechanism only if the
   evidence supports it. Keep the native32 export, shared fixed heads and
   frozen references. Avoid another local coefficient/rate/budget sweep.

## Execution and saved state

Encoder quality training ran once per declared seed on CUDA. CPU work fitted
the regression/neural heads and raw-data PCA and audited saved tensor arithmetic.
Frozen inference extracted or checked feature archives; no encoder was trained
again on CPU. Routine scoring should reuse those archives. Keep unavoidable
correctness witnesses explicit and separate from performance measurement.

The five new runs and ten retained controls use 256 TRAIN examples / 128 source
pairs and 128 known VALIDATION examples / 64 pairs per run; batch 8, 512 updates,
RTX A2000 Laptop GPU. Heads stay Ridge1 and tanh16/Adam0.01/100 updates, three
declared repetitions. PCA remains a standalone raw baseline only.

Completed capsule:
`output/runs/rpb-native-view-agreement-validation/native-view-agreement-validation-glrBZE`.
Inventory SHA256:
`5d8ae964a2d05fd68aaf2fb5c6173e5faaea35e68fbf4a3807b7f04afa0781dc`.
Passed audit:
`output/runs/rpb-native-view-agreement-validation/audit-tools/run-glrBZE-v2/validation.json`,
SHA256 `c0a96603d9b5e38e619841a5027088a383ce76f8a376409b1dd63b71fe8946c5`.
The failed compile admission, frozen cards, readers, checkpoints and large
archives are preserved. Source files and all 326 permitted parent inputs stayed
unchanged. Large run data is intentionally ignored by Git; the report and
numerical summary are versioned.

Reuse the existing managed container; inspect it with `./container.ps1 status`.
Do not rebuild or recreate it, delete Docker data, or rerun the completed
comparison just to recover these measurements. No background training,
scheduled experiment or additional model run is needed overnight.
