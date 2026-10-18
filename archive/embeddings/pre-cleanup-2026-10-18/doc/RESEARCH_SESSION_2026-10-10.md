# RPB research session, 10 October 2026

The user authorized continued encoder variations on the harder structured timing
tasks. The original saved models remain protected, and no other Codex chat was
contacted. Work used the existing managed GPU container and its internal pinned
LibTorch SDK; no Docker objects, dependencies or host toolchains were changed.

## Completed saved-TRAIN diagnosis

The [TRAIN diagnostic](../code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_TRAIN_DIAGNOSTIC.md)
and [durable JSON](results/structured_hard_timing_train_diagnostic_v1.json) read
only declared saved TRAIN roles, without model execution, head/PCA refits or new
validation tensor inputs. Saved validation metadata supplies the displayed
comparison. Early-v10 neural TRAIN/intact-VAL accuracy is 82.06%/54.90%; raw is
99.69%/50.73%. The target query timing rule scores 100%, while late/early saved
reconstruction scores68.28%/61.02% on the same support. Different query banks use
different masked contexts, so these results do not prove information loss in a
single full-context native32 vector.

## Completed visible-difference variation

The separately frozen [card](../code/evaluation/cards/visible_difference_v1.md)
tests RPB-v13: early mixing with an additive projection of currently visible
adjacent backward differences and their support. It adds 3,072 zero-initialized
values, making228,877 total. Original common initialization/scaler and all 15
point0 feature surfaces match exactly. Default0 preserves historical model,
configuration and checkpoint behavior; the new route has its own typed bindings,
continuation state and consumer guards.

Five candidate-only CUDA trajectories completed 512 updates each with B8 and .15
context deletion on the exact saved TEMPO-3 cohorts. There were15 new fixed-head
pipelines/30 heads,30 native exports including15 parity witnesses, and40 necessary
masked query forwards. Older encoders and their fitted heads were reused, not
rerun. No new raw/PCA fit, generator call, CPU encoder execution, TEST/stress,
selection or promotion occurred.

The [report](../code/encoders/raw_patch_bottleneck_mae/VISIBLE_DIFFERENCE_DIAGNOSTIC.md)
and [durable JSON](results/visible_difference_v1.json) retain all eight methods,
five cohorts, three head repetitions, traces, reconstruction reductions and
conditional source-group intervals. Mean intact linear/neural accuracy is
53.28%/52.03%, versus early control 52.81%/54.90%. Deleted is52.81%/50.52%, versus
53.75%/54.32%. Coverage stays 100%. TRAIN/VAL query MAE improves to
0.622320/0.664399. Better reconstruction does not establish a better timing
embedding; the variation does not give a consistent classification gain.

Registry registration preserved all 13 historical designs and15 earlier instance
groups, active v4, original v7 and the saved harder-timing comparison. RPB-v13 is a
separate measured descriptive group, not a replacement or promotion. The strong
TEMPO-1 RPB-v10.alt-03 remains intact.

## Preserved attempts and corrections

All attempts remain under `output/runs/rpb-visible-difference`; none was deleted.

- `admission-YHtTvd` failed an engineering CUDA device equality check: unspecified
  `CUDA` versus actual `CUDA:0`. The narrow guard now handles an unspecified index
  while retaining actual-device equality for values, visibility and packed tensors.
  `admission-JxVJ39` then passed the GPU checks.
- `visible-difference-OBX2LO` failed its first controlled TRAIN load because the
  shared native loader expected `observed`, while these retained archives use
  `feature_mask`. It created no encoder, updates, forwards, heads or `.pt` results.
  Its executable is also retained under the internal build's
  `retained-binaries/failed-OBX2LO`. The private protocol loader now checks the exact
  retained schema, randomized class order and gappy original source indices. Its
  four serialized positives and19 negative cases call the production loader.
  The shared historical loader and dataset remain unchanged.
- `admission-QjYYR7` stopped at the source guard when an artificial fixture filename
  changed during source freezing. No build or GPU/model work occurred. Artificial
  input fixtures now use `.serialized-archive`, preserving their serialization
  while keeping them outside the quality reader's exact `.pt` role matrix.
- Final `admission-P7tDKh` passed the serialized input, actual CUDA model/adapter,
  early adapter, fixed-readout, frozen-role and internal SDK checks. This admitted
  the final source identity 14c99891… before the one completed quality capsule
  `visible-difference-wnUUCa` was launched.
- Saved audit `run-wnUUCa-v1` failed after 672,861 checks/27 CPU witness decodes at
  `saved query report artifact`: writer metadata uses a basename and the reader
  supplied a full path. Its failure JSON SHA is
  `173eaceca4ef7e782d03f4e8aaeca5fbea2e05dfd0387e32aaa340518f4fe2d0`.
  Original reader EA60… and seal 9a68… remain unchanged. Additive reader-v2 uses
  `p.name` with the unchanged literal query check, binds both old/new source
  identities, and proves the exact beforeimage. SOURCE fixtures passed 76,292
  checks/69 negatives without real archives. No numerical or scoring rule changed.
- Root then audited the same frozen quality capsule with reader-v2. It passed
  17,152,339 checks/280 CPU witness archives in 170.61 seconds. Passed validation
  SHA is`b4849d9aca2530523b12093706f8f52229bc512cf9947447c68870d2d31ee070`.
  There was no model rerun, fit, resampling by the publisher or capsule mutation.

The complete inventory SHA is
`384ed378bd0cd563e2f0e38f65c35a20d4e8db746dac4830c86cb8f29129f2fb`:
723 files/210,928,579 bytes. The metadata publisher passed 7,867 checks with three
explicit hashed inputs, no archives/models/fits/resampling. Published JSON SHA is
`648586a087590bb85d92256fae25cdb99f96da5b90eb93b95a464bf95a34c690`;
report SHA is`54dbf40e1989a0b4d6cbd45f7bbeeea974516f1c0a45c76fae5bb3f31f7b0721`.
Generated weights/capsules remain protected locally; source and durable results
are repository artifacts.

Independent metadata-only QA passed 140,525 comparisons with zero archive reads,
model/fitting work or audit reruns. All three eight-method panels, five cohorts,
three head repetitions, nine rendered tables, traces, intervals and bindings
agree. All 13 prior designs, 15 prior instance groups and 21 protected registry top
objects match Git 71c6e046, including active v4 and original v7. The exclusive proof is
`output/runs/rpb-visible-difference/audit-tools/run-wnUUCa-v2/metadata-independent-qa.json`,
SHA`b38782d76efe7ca41dcd6b5479cc0505e4657658704a9099ddd37d7a6470b31d`.

## Continuation

The [next direction](CONTINUATION_2026-10-10_AFTER_VISIBLE_DIFFERENCE.md) is one
fresh 2×2 comparison: early versus visible-difference architecture, each with128
versus512 independent TRAIN pairs, a common fresh validation set and common
small-TRAIN-fitted scaler. Keep the signal law, native32, .15/B8/512 and heads.
Disclose equal encoder draws but increased head/extraction cost for the larger
set. Freeze exact data/versioning/reader contracts before implementing and running.
Historical scores are context only; the new comparison requires matched fresh
controls and fitted heads. No head tuning, post-encoder PCA or stopped-recipe
rescue. The aim remains a timing embedding that generalizes on complex structured
data, rather than a selected improvement on known cohorts.
