# CPB-v1 portable encoder guide

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

CPB-v1 is the first independently owned version of
`causal_patch_bottleneck_mae`. Its first declared instance is
`CPB-v1.market-01`: exact saved RPB-v30 weights and frozen normalizers, with
zero additional encoder updates. The causal conversion changes temporal
execution, so it does not promise the old native32 or native36 values.
RPB-v22 remains primary, and every historical RPB result and checkpoint stays
preserved. CUDA proof and fixed-head quality scoring are complete: the original
market 9/9 and strict 8/8 mean floors pass, but only 2/3 neural trend fits are
positive and only 3/8 TEMPO-5 retention floors pass. Neither full comparison
gate passes, so this is a preserved causal reference without an overall win.

## Version entry and complete implementation

The authoritative version directory in both repositories is:

`code/encoders/causal_patch_bottleneck_mae/versions/v1/`

Place that directory's `include/` on the include path and include only:

```cpp
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h"
namespace cp = embedding::encoders::causal_patch_bottleneck_mae::v1;
```

This public entry, the complete 42-header implementation/parser closure under
`include/`, `config/v1.conf`, `tests/v1_test.cpp`, `v1.mk`, version README and
`provenance.json` belong to this physical version. Private implementation is
under `include/embedding/encoders/causal_patch_bottleneck_mae/detail/v1/`.
There is no include, alias, inheritance or runtime implementation dependency on
RPB or another causal version. Torch, the C++ standard library and platform
checkpoint-publication facilities remain external dependencies. Inherited
notices are preserved; marked CPB adaptations and new original files remain
all rights reserved under the repository license.

Future `versions/v2/` receives its own complete implementation, configuration
and tests. It must not forward into mutable v1 or RPB implementation. Preserve
an evaluated version; declare a later behavioral change separately. Family-level
[version metadata](../code/encoders/causal_patch_bottleneck_mae/versions/index.json)
and [CPB-v1 metadata](../code/encoders/causal_patch_bottleneck_mae/versions/CPB-v1.json)
provide frozen prospective navigation. The additive
[measured instance record](../code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v1.market-01.json)
binds later proof, quality and exact local artifacts without rewriting those
admitted declaration snapshots.

The additive [historical RPB index](../code/encoders/raw_patch_bottleneck_mae/versions/index.json)
describes actual historical definitions and any known gaps. It does not move,
rewrite or freeze today's implementation under an old label. The new physical
copy policy applies to CPB versions.

## Cutoff-enforced API

`RawHistory` contains CPU float64 data `[B,C,N,F]`, a matching CPU boolean
observed mask, semantic channel IDs, a uniform float64 timestamp grid and its
sampling interval. `encode_at(raw, cutoffs, scaler)` requires explicit CPU
float64 cutoff timestamps `[B]`; the scalar overload delegates to the same
path. It returns native float32 `[B,36]`, validity `[B]` and the requested
cutoffs. There is no public unguarded full-history `encode`, decoder or training
API. `trace_at` exposes evidence from the actual guarded forward.

After structural schema checks, each example retains at most the last H grid
observations through its cutoff. Missing earlier slots are left-padded with
zero values and observed=false. This happens before observed-value finiteness
checks, float32 rounding, scaling, support counting, banks and token packing.
Future values and future masks therefore cannot enter those operations. A
cutoff outside the supplied grid rejects. The declared timestamp grid, channel
schema, immutable weights and frozen TRAIN normalizers remain dependencies.

For the measured C3/H128/F9/P8 geometry, patches end no later than the cutoff.
Valid packed tokens retain strictly increasing original patch coordinates and
form a prefix followed by padding. Temporal attention excludes later keys
with an additive 0/-infinity mask and retains the key-padding mask. Channel
attention mixes channels only at the same patch. Every content/lag, spectral,
change/quarter moment and pooling operation receives only the retained window.
Feature-width LayerNorm is unchanged. Each sequence output must use its own
cutoff; one final vector must not be broadcast to earlier timestamps.

Parent preprocessing preserves sanitized rawF64 -> rawF32 -> CPUF64 saved
input scaler -> CUDAF32. The relative-change branch consumes genuine rawF64.
An external input scaler may be passed under a declared policy; it does not
replace the checkpoint's market scaler or its frozen bank normalizer.

## Frozen import and local checkpoint

The explicit `import_v30_checkpoint` accepts the pinned source as data through
this version's own deserialization code. It verifies the file hash, all imported
parameter/buffer bytes, input-scaler tensors/metadata and bank-normalizer
statistics/metadata. No old encoder executes. Ordinary `load_checkpoint`
accepts only the new kind `cpb_v1_inference_checkpoint_v1`, tag `CPB-v1`, causal
policy and bound configuration/source/state/scaler/normalizer identities.
`save_checkpoint` publishes immutably. Its private nested payload preserves the
source data schema for exact transfer; that is not an implementation dependency.

The exact local, Git-ignored source checkpoint is in the integration repository:

`output/runs/market-location-scale-decoder/run-MAmvTJ/checkpoints/inference-2048.pt`

SHA256: `c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb`
(1,285,583 bytes). It already contains the trained v22 parent and trained V30
branch; CPB's zero new updates do not mean untrained weights. The new family
retains 76 parameter tensors and 297,560 parameters. No optimizer or causal
TRAIN artifact is admitted in this first comparison.

Frozen market input scaler:
`rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969`.
Frozen relative-bank normalizer:
`rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-a79ac4a6bfe8d476`.
No input-scaler fit, bank calibration or encoder training is repeated.

The usable new typed checkpoint is the passing attempt's local integration file:

`output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/proof/inference.pt`

SHA256: `54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895`
(1,290,592 bytes). Use this artifact with `load_checkpoint`; do not relabel the
old RPB checkpoint. `proof/refusal-1.pt` through `proof/refusal-10.pt` are
**deliberately malformed typed-loader test inputs**, preserved as engineering
evidence and unusable as model checkpoints. Large model/data/feature archives
remain local and Git-ignored; mirrored source does not imply their presence.

## Protocol and artifact map

The prospective [CPB-v1 card](../../cuwacunu_dev/doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_CARD.md)
is SHA256 `3b059b84a60a7d193ffcb034480f54fa5aba532ed6cb6b541fa620c4ab584d5f`.
Its [physical-layout addendum](../../cuwacunu_dev/doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_LAYOUT_ADDENDUM.md)
is SHA256 `ae0a8adae317a7b2daee3f58a645c9000faab130cde17ba6662a66fae06047dd`.
Both live in integration. The version-local provenance records original source
hashes and destination hashes. A new attempt captures exact source and binary
before invocation; failed attempts remain available.

Paths below are relative to the integration repository. The passing engineering
attempt is `attempt-002`; the sole completed quality run is `run-001`. Prospective
admission snapshots keep their original stage wording; separate passed proof,
audit and execution receipts establish the completed stages.

| Artifact | Location | Purpose |
| --- | --- | --- |
| Owned runtime source/config | `code/encoders/causal_patch_bottleneck_mae/versions/v1/` | Compile the version and validate configuration; mirrored in both repositories. |
| New typed checkpoint | `output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/proof/inference.pt` | Sole usable CPB model/scaler/normalizer artifact, verified typed roundtrip. |
| Source/binary admission | Same attempt's source manifest/capsule, build log, dependency bindings and admission receipt | Bind what was compiled and invoked; engineering evidence. |
| Bounded causal proof | Same attempt's `proof/` receipts plus `proof-stage.json` and `independent-audit.json` | Verify suffix-value and suffix-mask independence, temporal states, typed import/reload and frozen state. |
| Market feature banks | `output/runs/causal-patch-bottleneck-mae-v1/run-###/features/market/` | Saved native36 TRAIN/VALIDATION embeddings consumed by fixed heads. |
| Synthetic feature banks | Same run's `features/synthetic/{synthetic_train,market_scaler}/master-94110{1,2,3}/` | Saved embeddings under both declared input-scaler policies. |
| Fit/prediction banks | Same run's `market-readouts/` and `synthetic-readouts/` | Necessary saved head fits, predictions and scoring evidence; no encoder retraining. |
| Compact results | `doc/results/causal_patch_bottleneck_mae_v1/` and [result report](../../cuwacunu_dev/doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md) | Durable hashes, measured tables, arithmetic/preservation checks and exact artifact pointers; complete. |

Attempt 001 stopped before quality on the installed C++ LibTorch attention-mask
broadcast behavior. Its source, binary, log and partial artifacts are preserved.
The pre-quality repair uses the installed path's two-dimensional additive mask
with explicit packing-order checks, preserving attention projections and channel
mixing. Attempt002 passed reviewed engineering and CUDA proof before quality. Both
attempts retain exact source capsules and all three admitted binaries in their
own `binaries/` directories; the failed attempt is not overwritten or counted
as an extra quality trajectory.

Proof checks include every cutoff within H128, all positions within P8 patches,
longer histories, heterogeneous cutoffs, altered suffix values and masks,
observed future NaN/infinity/float32 overflow, missing/gapped support and semantic
channel permutations. Observed poison inside the retained window must reject;
hidden storage poison must be ignored. Separate earlier-state checks intervene
on later patches within the same retained window. Positive controls prevent a
constant-output proof. Exact frozen state, undefined gradients, actual CUDA
execution and new typed checkpoint roundtrip are required. The proof is bounded
engineering evidence, not an accuracy result.

## Fixed quality comparison and measured status

The completed MKT-HIST-2 complexity4/5 comparison uses the saved 112 TRAIN/32 known
VALIDATION historical windows and four separate retention tasks: volatility,
trend, activity and relation. TEMPO-5 complexity5/5 uses saved cohorts
941101/941102/941103, each 512 TRAIN/256 VALIDATION, intact and 30%-erased views,
and both saved synthetic-TRAIN and market input-scaler policies. The market
bank normalizer stays frozen under both. Full-window endpoints are scored;
early cutoffs receive correctness tests, not invented quality targets.

The ridge/neural recipes, budgets, seeds and coverage rules stay fixed. Only
new CPB feature heads are fitted on TRAIN; old encoder/head scores are reused,
with no post-encoder PCA, tuning or old-control replay. Market historical
retention and synthetic timing classification remain separate tables. Neither
is automatically a future-forecasting result. Zero encoder updates must be
reported separately from proof, inference, head fitting and artifact costs.

The frozen [result report](../../cuwacunu_dev/doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md)
and [completion record](../../cuwacunu_dev/doc/results/causal_patch_bottleneck_mae_v1/completion.json)
contain all measured gates and evidence bindings. Completion `status: passed`
means the declared procedure completed; it does not mean every quality gate
passed. No primary/balanced promotion, commit or publication follows.

## Measured market results

The following are fixed-head R² means on MKT-HIST-2's known chronological
VALIDATION. All neural means use repetitions 2701/2802/2903, not independent
encoder-training cohorts. D36 uses seeds 13088411117524848790,
10018485049408248344 and 9083751155097190455. Linear ridge penalty1 and neural
16-tanh/Adam0.01/100 updates remain unchanged; D36 heads have 148/660 parameters.
One new ridge pipeline/four target fits and three neural fits were performed
on TRAIN. The same heads consume saved embeddings with no PCA or tuning.

### Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute history, recent128 past-window targets · **complexity4/5** · known chronological VALIDATION32, TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728106930 | 0.380336442 | 100.0 |
| RPB-v24.market-01.alt-01 | 36 | 0.446315410 | 0.392614192 | 100.0 |
| RPB-v30.market-01 | 36 | 0.953912865 | 0.588276565 | 100.0 |
| CPB-v1.market-01 | 36 | 0.954421681 | 0.605930583 | 100.0 |

CPB neural R² repetitions: 0.624805518073028, 0.6614974721031207, 0.5314887581828607.
CPB linear MAE/RMSE: 5.4406701903979004e-05 / 7.212641896829908e-05; neural repetition-mean MAE/RMSE: 0.00016624754519196607 / 0.00021158217223371328, in logreturn.

### Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute history, recent128 past-window targets · **complexity4/5** · known chronological VALIDATION32, TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | -0.383904242 | -0.542353049 | 100.0 |
| RPB-v24.market-01.alt-01 | 36 | 0.654911255 | 0.134295520 | 100.0 |
| RPB-v25.market-01 | 36 | 0.774228435 | 0.488464661 | 100.0 |
| RPB-v30.market-01 | 36 | 0.794365502 | 0.320209792 | 100.0 |
| CPB-v1.market-01 | 36 | 0.796803646 | 0.243687749 | 100.0 |

CPB neural R² repetitions: -0.33939984380671695, 0.657262428919963, 0.4132006631059153.
CPB linear MAE/RMSE: 2.4165928465186302e-05 / 3.018542388780316e-05; neural repetition-mean MAE/RMSE: 4.434894082999679e-05 / 5.5999286171599476e-05, in logreturn/tick.

### Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute history, recent128 past-window targets · **complexity4/5** · known chronological VALIDATION32, TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.728431186 | 0.512167460 | 100.0 |
| RPB-v24.market-01.alt-01 | 36 | 0.751429682 | 0.563152901 | 100.0 |
| RPB-v30.market-01 | 36 | 0.865731877 | 0.613572485 | 100.0 |
| CPB-v1.market-01 | 36 | 0.875125214 | 0.686646622 | 100.0 |

CPB neural R² repetitions: 0.6866727610844631, 0.6900023672032143, 0.6832647365351475.
CPB linear MAE/RMSE: 0.166562319118347 / 0.20308033874250978; neural repetition-mean MAE/RMSE: 0.2619879309461553 / 0.3216944053604875, in log1p(base_volume).

### Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute history, recent128 past-window targets · **complexity4/5** · known chronological VALIDATION32, TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.market-02 | 32 | 0.388583093 | 0.257995833 | 100.0 |
| RPB-v24.market-01.alt-01 | 36 | 0.340685389 | 0.349583488 | 100.0 |
| RPB-v25.market-01 | 36 | 0.373556762 | 0.407149546 | 100.0 |
| RPB-v30.market-01 | 36 | 0.487476261 | 0.406836079 | 100.0 |
| CPB-v1.market-01 | 36 | 0.493705254 | 0.389716310 | 100.0 |

CPB neural R² repetitions: 0.3546264613165948, 0.48288169743556975, 0.3316407706162555.
CPB linear MAE/RMSE: 0.04962402354779031 / 0.061327849082980085; neural repetition-mean MAE/RMSE: 0.0546814115339698 / 0.06722787255979351, in correlation.

CPB passes original 9/9 and strict native36 8/8 mean-score floors with full
32/32 coverage for every task/head/repetition. The neural trend fit at repetition
2701 is negative (-0.33939984380671695), so only 2/3 trend fits are positive and
the full market comparison gate fails. Arithmetic tolerance is 1e-12; it does
not relax the criteria. In the matched trained-final subset the four linear
means, neural volatility and neural activity are new descriptive highs. V25
retains stronger neural trend (0.4884646605379312) and relation
(0.4071495455067296). Every displaced leader and historical record remains
preserved. Causal execution differs, so these are scoped retention comparisons,
not a universal winner or a quantitative estimate of old future leakage.

## Measured TEMPO-5 results

Both policies retain the frozen market bank normalizer. `synthetic_train` uses
the already saved synthetic TRAIN input scaler; `market_scaler` uses the
checkpoint market input scaler. Parent rawF64-to-F32 rounding is fixed and
differs from old native36 synthetic preprocessing; neither precision nor scaler
arm is selected after scoring. New fixed classification heads have 74/626
parameters: six TRAIN feature normalizers, 18 ridge and 18 neural fits, each
scoring intact and deleted views. All old scores below are saved reuse.

### synthetic_train / validation_intact

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity5/5** · known VALIDATION, saved intact/additional30%-erased masks, cohorts941101/941102/941103.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.alt-01 | 32 | 97.135417 | 95.963542 | 100.0 |
| RPB-v24.market-01.alt-01 | 36 | 98.958333 | 97.743056 | 100.0 |
| CPB-v1.market-01 | 36 | 98.828125 | 97.873264 | 100.0 |

### synthetic_train / validation_deleted

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity5/5** · known VALIDATION, saved intact/additional30%-erased masks, cohorts941101/941102/941103.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.alt-01 | 32 | 91.276042 | 91.059028 | 100.0 |
| RPB-v24.market-01.alt-01 | 36 | 93.359375 | 92.534722 | 100.0 |
| CPB-v1.market-01 | 36 | 92.708333 | 92.144097 | 100.0 |

### market_scaler / validation_intact

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity5/5** · known VALIDATION, saved intact/additional30%-erased masks, cohorts941101/941102/941103.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.alt-01 | 32 | 97.135417 | 95.963542 | 100.0 |
| RPB-v24.market-01.alt-01 | 36 | 100.000000 | 99.522569 | 100.0 |
| CPB-v1.market-01 | 36 | 100.000000 | 99.348958 | 100.0 |

### market_scaler / validation_deleted

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity5/5** · known VALIDATION, saved intact/additional30%-erased masks, cohorts941101/941102/941103.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v22.alt-01 | 32 | 91.276042 | 91.059028 | 100.0 |
| RPB-v24.market-01.alt-01 | 36 | 96.093750 | 95.052083 | 100.0 |
| CPB-v1.market-01 | 36 | 96.093750 | 94.878472 | 100.0 |

TEMPO-5 passes 3/8 frozen native36 retention floors: synthetic-scaled intact
neural, market-scaled intact linear and market-scaled deleted linear. The other
five fail. Coverage is 100% for all cohorts, roles, policies, heads and paired
source groups; full coverage does not change the failed retention outcome.
Each cohort retains 512 TRAIN rows/256 paired groups and 256 VALIDATION rows/128
disjoint paired groups. All repetitions, group counts, physical errors and
conditional 95% group-bootstrap intervals are preserved in the frozen report
and `table_rows.json`; the instance record binds those exact files. The 1,000
bootstrap draws condition on saved features/fits and do not measure encoder
training uncertainty. Saved scalar arithmetic was checked 2,562 times without
tensor deserialization or model/head calls. Full predictions and bootstrap
quantiles rely on the pinned unchanged eight-source evaluator; no new tensor
replay is claimed.

## Correctness, costs and preservation

The passing artificial CUDA proof records all128 cutoffs, 2,350 suffix cases
under nine variants/two layouts, 30 internal temporal cases and 14 typed negative
cases. There are 3,687 successful CUDA encoder calls/3,700 public attempts and
107,766 engineering checks. All76 parameter tensors/297,560 values and132 source
tensor leaves import exactly; normalizers, scaler and state remain exact and
gradients undefined. The separate stdlib proof audit passed 2,290 saved-receipt
and bounded-witness checks with zero encoder/head calls or tensor loads. These
are bounded correctness checks, not a new quality sample or causal training.

On CUDA:0 NVIDIA RTX A2000 8GB, batch8, proof import/state/I/O took
0.76141305s, proof/state checks 167.33324873s and typed artifacts 2.929673615s.
The single quality run used 786 CUDA calls: 18 market and 768 synthetic, with
6,144 synthetic rows. Guarded feature export including input I/O/state checks
took 36.319712263s. Wrapper stage times were extraction 39.33743146300185s,
market heads 2.3194627049997507s and synthetic heads 5.941477216998464s;
report/hash capture took 16.575647551999282s. These stages include preprocessing,
transfers, state checks and I/O, so they are not a matched kernel-speed benchmark.
New encoder updates/training seconds, decoder calls and input/bank fits are
all zero. The imported V30 checkpoint was trained previously; zero new CPB
updates does not mean random weights. This encode-only card measures no causal
reconstruction MAE or Gaussian NLL.

Preflight verified all55 old checkpoint pins as available, hash-matched and
Git-ignored, all99 old portable pairs and44 historical document bindings
(43 distinct files). It also preserved275 old raw source files and23 files from
the interrupted, unscored V30 synthetic run. The prior77,831-file historical
closure and its additive totals correction remain bound without a new20GB scan.
Quality before/after verification passed for4,144 protected files/799,383,919
bytes. Large tensors and witnesses stay local/ignored; compact hashes and
necessary scalar evidence are durable. Both engineering attempts, every RPB
source/checkpoint/card/result and the frozen prospective CPB metadata remain
preserved. The same guide and instance registry are mirrored in both repositories.

RPB-v22 remains primary; RPB-v24.market-01.alt-01 native36 remains the balanced
reference. CPB-v1.market-01 is a proved causal, zero-update reference with strong
market means and incomplete trend/transfer retention. Preserve this completed
recipe. Any later causal training, different encoder, fresh chronology or
forecasting task requires its own prospective protocol; do not retune these
known quality windows or relabel historical targets.

## Exact result bindings

- Report: `doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md`, SHA256 `12409f35f7f7d2dabf4837d3bde142bec5197a4f5d8dd70dc98c3e6da9ce0ad3`.
- Completion: `doc/results/causal_patch_bottleneck_mae_v1/completion.json`, SHA256 `cc4948effa5920aae6b3204eee754cbe352559401a88217074f1beee687fed0c`.
- Table rows: `doc/results/causal_patch_bottleneck_mae_v1/table_rows.json`, SHA256 `905c2fcd69e8c0d513db589213a5c1e4da67b51407d2fcea0bcfcf389f61c6f6`.
- Asset inventory: `doc/results/causal_patch_bottleneck_mae_v1/asset_hashes.json`, SHA256 `8c6134fb8e9a08f8cfa0c447a5939e05ac110779bdff1eb036598357767ce26b`.
- Durable inventory: `doc/results/causal_patch_bottleneck_mae_v1/durable_files.json`, SHA256 `db1e1bd46be727776d1b547798e69e7012ea4c070b7ef670383756bf344bcf2b`.
- Quality verification: `output/runs/causal-patch-bottleneck-mae-v1/run-001/verification.json`, SHA256 `817eaef147f523f6893bad33cf27c857601dbbfe3a53701eadfe92e4e13c0a93`.
- Proof: `output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/proof/proof.json`, SHA256 `cfa8b8a2879d3c7366c15d53f1114193339e7fb56d2a84cc80e9c5fad0a39c99`.
- Admission: `output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/admission.json`, SHA256 `68e6273bb4199bbbd1991c993237bbca1de057cb76c3a0cdde854b2439cce8f3`.
- Quality-tools admission: `output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/quality-tools-admission.json`, SHA256 `0822653c13e37c747f38ef6ce9afbcaa772b9ad92039207f258e8d8bcbd6efc0`.
