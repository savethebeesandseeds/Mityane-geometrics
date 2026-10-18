# Cuwacunu instructions

## Latest completed task: causal encoder, 15 October 2026

The earlier continuous encoder search was stopped by the user. The separately
requested causal conversion is now complete: **CPB-v1.market-01**, family
`causal_patch_bottleneck_mae`, imports exact saved V30 tensors with zero new
encoder updates or input/bank fits. Read
[the results](doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md),
[the portable guide](doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1.md) and
[the family entry](code/encoders/causal_patch_bottleneck_mae/README.md).
The complete implementation/config/tests belong to physical `versions/v1/` in
both repositories; future versions require their own complete copies, without
shared mutable encoder implementation. Historical RPB versions have additive
metadata entries; their source/results/checkpoints remain intact.

Actual CUDA cutoff proof passed: all128 cutoffs,2350 suffix value/mask cases,
30 temporal hidden-state cases, exact imported/frozen state and typed load/save.
Native36 unchanged fixed heads pass original market9/9 and strict8/8, but only
2/3 neural trend fits are positive. TEMPO-5 remains strong but retains only3/8
exact native36 floors across both scaler policies/intact/deleted views, all at
100% coverage. This is the first verified causal reference, not an overall
replacement. Keep RPB-v22 primary, native36 balanced and every scoped leader.
Old market probes already ended at their anchor; score differences are causal
execution effects, not a measured quantity of future leakage or forecasting.

Preserve both engineering attempts (including the repaired LibTorch mask
failure), the new typed CPB checkpoint, the one quality run and all55 older
checkpoint pins. The old V30 TEMPO-5 extraction stays interrupted/unscored.
No new training, head tuning, old replay, primary promotion, commit or push is
implied. Next propose a separately declared causal-training version that
protects market gains, improves neural-trend robustness and restores hard-signal
retention. Do not start it automatically.

This is the clean integration repository. The sibling `cuwacunu_torch`
repository is a reference source. Reusable encoder/config changes must also
be backported to `cuwacunu_embedding`, as the user explicitly requested.

- The local checkout is `C:\Work\cuwacunu\cuwacunu_dev`; the project and
  repository name remain `cuwacunu`. Its authorized GitHub remote is the private
  `savethebeesandseeds/cuwacunu` repository. Keep it private and verify that
  destination before pushing. Public visibility, public releases or another
  push destination require explicit user authorization.
- New original code and documentation are all rights reserved under `LICENSE`
  until the user decides otherwise. Do not add MIT or another open-source
  grant by default. Preserve inherited notices described in `LICENSES/NOTICE`;
  keep private additions to MIT components in separately marked files or
  sections. Backporting authorizes local synchronization only; publishing
  new work to a public sibling remote requires explicit user authorization.
- RPB-v22 is the primary public encoder, selected by config/encoder.conf and
  cuwacunu::encoder::primary. RPB-v20 remains its reusable base and an explicit
  reference with config/encoder_v20.conf; keep its portable source unchanged.
  Keep the imported RPB-v19
  reference, its 17 original headers and all saved evidence unchanged.
  Declare changed geometry with the new tag; do not relabel old checkpoints.
  Keep data loading independent of the encoder and core settings independent
  of application orchestration.
- RPB-v23 is an explicit consecutive-change candidate in `rpb_changes`, locally
  mirrored to the embedding repository. Its MKT-HIST-2 experiment improves linear
  trend but misses neural-trend/volatility retention goals. Keep RPB-v22 primary;
  use `doc/MARKET_CONSECUTIVE_CHANGE_RESULTS.md` and `doc/CONTINUATION.md` for
  the next direction. Never relabel v22 checkpoints as v23.
- Use C++20 and Makefiles. Build, test and run exclusively in the managed
  `cuwacunu` Linux development container; follow `doc/ENVIRONMENT.md` and its
  authoritative launcher. Preserve existing containers and their data. Do not
  install host toolchains.
- Follow `doc/ENCODER_BACKPORT.md`: reusable encoder changes belong back in
  `cuwacunu_embedding`. Market ingestion and application adapters stay here.
- Keep evaluation separate from the encoder. Native embeddings have no PCA
  afterward. Do not infer market quality from synthetic confirmation results.
- Keep changes small; preserve existing checkpoints and historical evidence.
  Run focused import/integration checks, not repeated research comparisons.
- When reporting measurements, name the exact model version and dataset,
  identify reused versus new evidence, and explain changes and next steps in
  plain words. Model tags describe designs; seeds, scalers and checkpoint
  hashes identify trained instances.

- RPB-v24 is the explicit separate-change candidate in rpb_separate, locally
  mirrored into research. Read doc/MARKET_SEPARATE_CHANGE_RESULTS.md and the
  latest continuation. It preserves trained v22 native32 exactly and appends8;
  trend improves but balanced retention fails. Preserve v22 primary, useful
  v23 native32 trend and saved v24 native40 trend references. Never relabel
  their checkpoints or rerun stopped cards to rescue a failure.

- Latest completed variation: RPB-v24.market-01.alt-01 preserves parent32 and
  appends4/native36. It reaches trend R²0.655/0.134 and passes8/9; linear
  volatility retention fails, so keep v22 primary and all native40/native36
  checkpoints. Read doc/MARKET_SEPARATE_CHANGE_NATIVE36_RESULTS.md and the
  latest CONTINUATION.md before choosing work. Next propose saved-TRAIN-only
  level/change covariance and fitted-linear-contribution diagnosis, with no
  encoder updates, head refits, old replays or held-out diagnostic inputs.
  No width/rate/budget/head rescue is authorized. Publication authorization is
  recorded separately below.

- Frozen market-to-synthetic transfer is now measured: the exact saved
  RPB-v24.market-01.alt-01 checkpoint performs strongly on TEMPO-5 complexity 5/5
  with zero encoder updates; both declared scaler policies and all fixed-head
  repetitions are in [the transfer result](doc/MARKET_SYNTHETIC_TRANSFER_RESULTS.md). This is scoped
  to the compatible C3/H128/F9 family; TEMPO-4/AMP-2 remain unmeasured. Keep
  RPB-v22 primary, every historical artifact and the separate volatility failure.

- Save milestone, 14 October 2026: native36 is the leading research candidate,
  while RPB-v22 remains the primary encoder. Read
  [the milestone](doc/MILESTONE_2026-10-14_NATIVE36_TRANSFER.md) for the exact
  distinction and next saved-TRAIN-only volatility diagnosis. The user explicitly
  requested updating docs, committing and pushing this completed native36 and
  frozen-transfer work to both existing remotes. This authorizes this source,
  documentation and nonbinary evidence payload to the private integration and
  public embedding repositories, superseding earlier local-only wording for
  this payload. Keep licenses, visibility, primary selection and ignored local
  checkpoints unchanged; this is not standing authorization for later work.

- The saved-TRAIN volatility diagnosis is now complete. Read
  [its result](doc/MARKET_TRAIN_VOLATILITY_DIAGNOSIS_RESULTS_READABLE.md) and
  [the next-direction proposal](doc/MARKET_CHANGE_ENERGY_NEXT_DIRECTION.md).
  Parent32 and the old predictor are exact; ridge redistributes coefficients
  and improves TRAIN fit, with uneven early/late benefit. Strong pairwise
  level/change duplication and large full-block volatility cancellation are
  not supported; TRAIN alone cannot explain held-out generalization causally.
  Next propose preserving signed-change/energy moments during branch training,
  under a new fully declared instance/card and strict protection of the other
  gains. That proposal is now measured separately as v25 below. Keep v22 primary, native36 and all
  history. No encoder calls/updates, head fits or held-out tensor diagnostics
  occurred in this diagnosis; its new source/docs/evidence remain local.

- Latest experiment,15 October2026: RPB-v25.market-01 adds training-only balanced
  signed-change/RMS reconstruction while keeping exact parent32 plus4/native36.
  Read [the result](doc/MARKET_CHANGE_ENERGY_RESULTS.md) and
  [the follow-up](doc/MARKET_CHANGE_ENERGY_FOLLOWUP.md). Trend/activity/relation
  improve, but both volatility heads fail: original7/9 and strict native36
  protection6/8. Stop this exact objective, preserving all five new checkpoints;
  no coefficient/head/rate/width/seed/budget rescue or synthetic transfer.
  RPB-v22 stays primary; native36 stays the leading balanced research reference.
  V25 is a stronger three-task reference, not a replacement. All54 portable pairs
  match locally. Next propose saved-TRAIN-only analysis of centered, dimensionless
  change statistics versus the existing scaled raw-difference RMS bank before
  declaring another mechanism; no new diagnosis or architecture is implied.
  This later work remains local, uncommitted and unpublished.

- Active goal,15 October2026: search for a verified9/9 encoder while keeping
  the best balanced and per-task references. Read the live
  [search ledger](doc/ENCODER_SEARCH_LEDGER.md) and its exact JSON companion.
  Preserve every candidate entry and checkpoint; only verified evidence may
  update leader pointers. Keep native36's strict protection floors explicit,
  and retain stronger individual task references even when a balanced candidate
  supersedes them. A stopped card ends that recipe, not separately declared
  research toward this goal. The saved-TRAIN centering/price-level diagnosis
  and the separate v26/v27/v28 experiments are complete; their frozen outcomes
  remain preserved. V28's raw-F64 symmetric-relative bank passes original7/9
  and strict4/8, with all three neural trend fits positive and full coverage;
  both volatility heads fail. Read [its result](doc/MARKET_RELATIVE_CHANGE_RESULTS.md)
  and [portable guide](../cuwacunu_embedding/doc/RAW_RELATIVE_CHANGE_V28.md).
  All83 reusable pairs match and45 candidate checkpoints remain local/ignored.
  V29's separate parent-modulated residual decoder is now measured and stopped:
  original 4/9, strict 4/8, all three final trend neural fits negative and full
  coverage. It added the then-leading linear activity/relation references
  0.8255824303229565/0.4539605718451353, without changing balanced/primary selection.
  Read [its result](doc/MARKET_PARENT_MODULATED_DECODER_RESULTS.md) and
  [portable guide](../cuwacunu_embedding/doc/PARENT_MODULATED_DECODER_V29.md).
  All 91 reusable pairs match and 50 checkpoints remain local/ignored.
  Fresh saved v24 initialization gives exact full36 parity to saved v28 initial
  banks; initial scoring is reused with zero refits. Only final heads were newly
  fitted. The original v28 normalizer is reused with zero new calibration calls/fits.
  Preserve the failed reporter and its additive v2 correction; no quality rerun
  occurred. The separate [v30 result](doc/MARKET_LOCATION_SCALE_DECODER_RESULTS.md)
  now passes all original9/9, strict8/8, all3 positive neural trend fits and full
  coverage. V30 uses proper Gaussian waveform location/scale reconstruction,
  an unmodulated v28 mean and new change-conditioned variance matrix108/bias27.
  Native36,2048/B8, saved normalizer and initial scoring reuse stay fixed; zero
  initial refits or new calibration fits/forwards. Legitimate negative F64 NLL
  is retained; mean query MAE is separate and does not improve uniformly.
  All99 reusable pairs match;55 checkpoints remain available/local/ignored.
  Read the [portable guide](../cuwacunu_embedding/doc/LOCATION_SCALE_DECODER_V30.md)
  and exact registry. V30 now leads all four linear heads and neural volatility
  in the recorded matched scope; preserve v25 neural trend/relation, native40
  neural activity and all displaced v26/v29 records. RPB-v22 stays primary and
  native36 stays balanced pending the separately declared
  [frozen TEMPO-5 transfer](doc/MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md).
  That complexity5/5 transfer is unmeasured, with zero updates and both fixed
  scaler policies; do not infer synthetic quality from the market pass.
  New mechanisms require prospective cards before quality testing. The goal
  does not authorize primary promotion, publication or rerunning old encoders.

- The TRAIN scale diagnosis is now verified in
  doc/MARKET_CHANGE_SCALE_DIAGNOSIS_RESULTS.md (3,263,004 independent checks,
  zero encoder calls/fits). The next card is
  doc/MARKET_CONDITIONED_CHANGE_CARD.md: RPB-v26.market-01 conditions the existing
  bank with frozen TRAIN coordinate mean/std before projection. Admission and
  quality are now complete: original8/9 and strict5/8, with a new preserved
  neural-volatility reference0.41926036121396154. Linear volatility, strict
  linear activity and neural relation fail. See MARKET_CONDITIONED_CHANGE_RESULTS.md.
  Stop v26 without hyperparameter rescue. The separate v27 card is
  doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md for RPB-v27.market-01, a frozen
  unlabeled TRAIN parent-to-bank conditional map before the learned4 branch.
  V27 is now complete and stopped: original5/9, strict2/8, all three neural
  trend fits negative and100% coverage. Read
  [its frozen result](doc/MARKET_PARENT_CONDITIONAL_CHANGE_RESULTS.md) and
  [portable guide](../cuwacunu_embedding/doc/PARENT_CONDITIONAL_CHANGE_V27.md).
  All73 reusable pairs match;40 candidate checkpoints remain local and ignored.
  The new TRAIN map required512 frozen CUDA parent forwards, zero calibration
  updates and no bank-standardizer refit. No v27 synthetic transfer ran.
  Keep native36 balanced, v26's preserved neural-volatility reference and every partial
  reference. Preserve the original9 and native36 protection floors; do not
  change them or stopped recipes after seeing scores. V28's later result is
  recorded above; stop its exact mechanism without rescue or synthetic transfer.
  V29's later stopped outcome and milestone-local per-task leaders are recorded
  above, followed by V30's verified market pass and pending frozen transfer.
  No primary promotion, commit or push is implied.
