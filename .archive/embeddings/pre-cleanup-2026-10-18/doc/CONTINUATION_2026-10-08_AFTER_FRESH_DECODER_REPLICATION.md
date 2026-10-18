# Continuation after fresh equal-budget decoder replication

Date: 2026-10-08. Completed milestone: `fresh-decoder-replication-v1`.
This note records the state after the fresh comparison. The two earlier dated
continuation notes, historical cards and measured reports remain unchanged.

## Where we are

Continue from RPB-v7, the learned global bottleneck with lighter context deletion.
Keep RPB-v4, the learned global bottleneck, as the active reference. RPB-v9 stays
stopped. No new embedding tag or promotion follows decoder calibration.

Five fresh paired timing cohorts used 256 TRAIN / 128 VALIDATION examples each,
or 128 / 64 independent source pairs. Both policies trained once on CUDA for
512 completed, unskipped updates, B8, then received the same 128 frozen-decoder
updates. All 214,277 non-decoder parameters, buffers, scaler and native32 exports
remain exact during calibration. Fixed ridge/tanh-16 heads use the existing
recipes and three paired repetitions. There is no PCA after an encoder.
The 90 pipelines contain 180 individual heads; they are not extra encoder runs.
Classification is newly fitted once for these fresh cohorts, then reused after
calibration without another fit or score pass. CPU work is fixed heads, raw PCA
and saved-arithmetic auditing, with no CPU encoder forward or training.

The [measured report](../code/encoders/raw_patch_bottleneck_mae/FRESH_DECODER_REPLICATION_DIAGNOSTIC.md)
and [machine summary](results/fresh_decoder_replication_v1.json) retain all
methods, both views, all masters, separate costs and provenance. Main findings:

- Intact linear means: v7 92.96875%, v4 92.34375%, with 100% coverage. The paired
  gain is +0.625 percentage points; its descriptive five-master 95% interval
  is [-6.40625, +6.09375]. The clean-data advantage is unresolved.
- Additional 30% VALIDATION deletion: v7 89.84375%, v4 82.03125%. The paired
  linear gain is +7.8125 points, interval [+1.875, +14.84375]. Worst-master
  deletion accuracy is also higher for v7.
- Intact worst-master linear accuracy: v7 80.46875%, v4 89.0625%. Master12412
  loses 13.28125 points in its paired comparison and is also weaker on TRAIN.
  Intact neural mean is lower for v7, 95.26042% versus 98.125%.
- Post-calibration TRAIN/VALIDATION standardized MAE: v7 0.056375/0.059200,
  v4 0.053609/0.056835. V7 recovers below pre-calibration v4, but loses both
  equal-decoder-budget reconstruction guards. The joint guard fails three of
  six numeric conditions; do not describe this as an overall successful advance.

Independent sealed audit passed 67,932,331 checks / 670 CPU archives in 183.9
seconds, without model or head execution. No TEST/stress, checkpoint selection
or promotion occurred. The failed first engineering admission is preserved;
only its fixture RNG boundary was corrected before quality generation. Source,
card, backend and recipe were not tuned after seeing these results.

## Evidence to preserve

Capsule: `output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b`.
Source: `6463926ae71ee5e5547aa660d654afb02832fa843c978c41609cbd814266d78c`.
Inventory: `a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90`.
Frozen card: `code/evaluation/cards/fresh_decoder_replication_v1.md`, SHA256
`2236fb794d608c8f7b8bde81814f199abcc3e27c641cfba5902cdc3f4c4a25cf`.
Passed admission: `output/runs/rpb-fresh-decoder-replication/admission/admission-evzwzh`.
Audit: `output/runs/rpb-fresh-decoder-replication/audit-tools/run-4p9U4b-v2/validation.json`,
SHA256 `241efd4705e4f6d67edf7070fa650f41e5a2a1b2d6e1e371e3d2835391c567e4`.
Sealed reader SHA256:
`ab9d2a241ee746389cd1c4b32de8db1e9e9fca2943ec3ecd6d26e16d4148ca88`.
Generated tensor archives stay in ignored `output/`; tracked reports preserve
their identities. Do not delete them, overwrite failed attempts or rerun models
just to reproduce a table.

## Next focused action

Follow the [saved-TRAIN reliability plan](../code/encoders/raw_patch_bottleneck_mae/FROZEN_NATIVE_RELIABILITY_PLAN.md).
Freeze its new card and closed TRAIN input manifest before analysis. Use all
five paired v4/v7 TRAIN exports, retained classifier fits/predictions and
training traces to compare margins, source-pair separation and feature variation.
The question is why timing separation is less reliable in one v7 run, not how
to improve the classifiers. This step requires zero model updates, zero head
refits and no held-out analysis inputs. Run saved arithmetic inside the existing
managed container; no GPU training rerun is necessary.

Only after that diagnosis should we define one supported encoder-training
hypothesis and a new prospective comparison. Keep heads and native width fixed.
Do not reopen v9, tune deletion rates or extend decoder budgets to rescue the
fresh guard. If the diagnosis is inconclusive, report that and consider one
bounded frozen amplitude-transfer check before adding another loss.
