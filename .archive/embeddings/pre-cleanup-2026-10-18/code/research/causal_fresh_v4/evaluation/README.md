> Source implemented, pending admission: complete owned CPB-v4 copy of CPB-v3 with only variance_input_gradient_stop selected. Forward distribution values and numerical Gaussian NLL are unchanged; the encoder receives the partial gradient through live mean/residual paths. The prospective card, input manifest and execution admission are pending; no build, calibration, learning, proof or quality run has occurred.

# Fresh causal v4 fixed-quality tools

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

This independently owned evaluation closure follows
`doc/CAUSAL_FRESH_V4_CARD.json`. Root orchestration must admit frozen sources,
inputs, binaries, fresh ancestry, actual CUDA learning and typed resume, plus
causal proofs of the main saved initial0 and final2048 artifacts before quality.
These tools do not grant execution permission.

Every learned encoder tensor and both market normalizers belong to this new
fresh run. Historical encoders provide no checkpoint, initialization, teacher,
statistics, pass/fail target or score floor. The only reused quality rows are
encoder-free raw recent128/full256 controls selected into the separately
source-pinned `doc/causal_fresh_v4/raw-controls.json`. The reporter never reads a
combined historical encoder-results file.

`extract_main.cpp` opens both new typed checkpoints, requires source_updates0
and causal_updates0/2048 respectively, and exports new native36 features through
their cutoff APIs. It reads genuine raw CACHE88 doubles and saved observation
populations; it opens no old model, old scaler witness or old embeddings.
Three synthetic input scalers are fitted once from the permitted cohort TRAIN
values/masks using the owned population-statistics recipe, after the declared
F64-to-F32 parent rounding. Those new scalers serve both checkpoint stages and
are saved under `features/synthetic-scalers/`. The alternative policy uses this
new run’s frozen market TRAIN scaler; the new market bank stays fixed in both.

The extractor makes786 CUDA calls per checkpoint at batch8,1572 total, fitting
no encoder or head. Its CLI is:

```text
causal_fresh_v4_extract --initial-checkpoint INITIAL --final-checkpoint FINAL
  --settings V4_CONFIG --output NEW_QUALITY/features --data-root /cuwacunu
```

Features are written exclusively to
`NEW_QUALITY/features/{initial,final}/{market,synthetic}/`, with stage summaries
and an aggregate extraction summary. Market files are `cpb_v4-train.pt` and
`cpb_v4-validation.pt`; synthetic paths are
`synthetic/{synthetic_train,market_scaler}/master-94110{1,2,3}/features.pt`.

`evaluation.mk` provides `build-cpb-v4-quality`, producing
`causal_fresh_v4_extract`, `causal_fresh_v4_market_read` and
`causal_fresh_v4_synthetic_read` under `$(BUILD_DIR)`. Every fixed-head dependency
and CACHE88 record header is a physical owned copy, with exact source hashes in
`dependency-provenance.json`. Head arithmetic, seeds and budgets are unchanged.
No old compiled object or mutable implementation include is used.

The market reader accepts its original arguments, once per STAGE:

```text
--train-targets /cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/train-targets.pt
--validation-targets /cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/validation-targets.pt
--method cpb_v4 NEW_QUALITY/features/STAGE/market/cpb_v4-train.pt NEW_QUALITY/features/STAGE/market/cpb_v4-validation.pt
--output-directory NEW_QUALITY/STAGE/market-readouts
--card-sha256 PENDING_ROOT_FROZEN_CPB_V4_CARD_SHA256
--input-manifest-sha256 PENDING_ROOT_FROZEN_CPB_V4_INPUT_MANIFEST_SHA256
--dataset-id MKT-HIST-2
```

The synthetic reader accepts, once per STAGE:

```text
--features NEW_QUALITY/features/STAGE/synthetic
--output NEW_QUALITY/STAGE/synthetic-readouts --stage initial|final
```

Both initial and final heads are newly fitted on their permitted TRAIN features.
Each checkpoint has one ridge/four target market fits and three neural fits;
each of three synthetic cohorts has two policies with three fixed fits each.
No baseline replay, initial-score reuse, synthetic encoder training, post-encoder
PCA, head tuning, intermediate checkpoint selection or silent retry is allowed.

`report.py` accepts `--run NEW_QUALITY --training TRAIN_ROOT --card CARD_JSON
--input-manifest INPUT_MANIFEST --raw-controls RAW_CONTROLS --quality-receipt
RECEIPT --output NEW_QUALITY/report`. It validates the root receipt’s
`initial_checkpoint`, `final_checkpoint`, `initial_saved_proof`,
`post_training_proof`, `fresh_initialization`, `engineering_proof` and build
admission pins and `card_sha256`, along with
the actual saved training/calibration/query records. Its cost fields are
`stage_seconds.extraction`, `initial_market_heads`, `final_market_heads`,
`initial_synthetic_heads`, `final_synthetic_heads`.

Four market task tables and four synthetic policy/view tables report new initial
and final scores independently, with full-precision repetitions, physical
MAE/RMSE, source-group intervals, standardized fixed-query MAE, separate Gaussian
NLL, coverage and stage costs. Trust admission plus finite complete coverage can
establish a clean fresh baseline. Scores remain descriptive initial-to-final
measurements; no overall winner, historical retention gate or application
promotion follows. Failures and partial outputs must remain preserved.
