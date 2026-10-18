<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v4.market-01: isolate variance gradients

Declared before builds, calibration, training or quality. The machine card is authoritative.
CPB-v3.market-01 remains the clean baseline and selected dev encoder. V4 is a new
research candidate with a complete owned source copy; it does not initialize from
v3 weights or any noncausally trained model.

## One controlled change

Keep the existing conditional Gaussian forward model, native36 and all297560
parameters (293672 learned in75 tensors;3888 literal analytic prior). Only the
variance projection reads `z_change.detach()`. Its108 weights and27 biases still
learn from NLL. The mean residual reads live change4. The numeric NLL is unchanged,
but encoder updates use the mean-path partial gradient rather than the complete
joint Gaussian NLL gradient. This optimization distinction is intentional.

The saved TRAIN diagnosis found a B8 local variance/mean gradient norm ratio6.94,
cosine+0.101 and3/8 opposing examples; this does not prove universal conflict.
Across112 full-context TRAIN rows, scale-dependent change4 participation ratio
fell3.66 to1.20. Relation was already learnable on TRAIN. V4 tests whether removing
that particular gradient path improves retention without adding a scored-task bank.

## Fresh causal training and fixed evaluation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute-history retention ·
**complexity 4/5**. Keep4096 TRAIN anchors from January1–21, consuming recent128
raw candles, and112 TRAIN/32 known chronological VALIDATION probes (January22–28).
January29–31, new periods and forecasting remain unmeasured.

Initialize every learned value directly with seed943001, reconstruct the fixed
analytic prior, freshly fit input/bank statistics on the4096 TRAIN windows and
create a fresh AdamW optimizer. No pretrained initializer, teacher, old optimizer
or fitted statistics enter construction. Geometry stays C3/H128/F9/P8/native36,
width64/layers3/heads4/feedforward256/decoder128. Source updates stay0.

One trajectory:2048 successful CUDA:0 updates, B8, at most4096 attempts; AdamW
0.001/weight decay0.0001, clip1, dropout0, query masking25%, context deletion15%.
Keep paired row/mask/context seeds943002/943003/943004. Use immutable1/16 fixed
query plans. Save inference0/2048 and typed TRAIN512/1024/2048, resuming the actual
trajectory at512 without repeated updates. Report synchronized training compute
separately from calibration, proofs, extraction, heads and I/O.

Before quality, prove fresh constructor equality, actual learning in all75 learned
tensors, frozen prior/statistics, all128cutoffs/2350suffix/30internal causal cases
before/after engineering learning and on actual saved initial/final artifacts,
typed refusals/exact resume and exact source/compiler/checkpoint bindings. Also
run a nonvacuous CUDA variance-gradient test with nonzero engineering weights:
forward conditioning must remain, variance-only native gradients must be blocked,
head gradients must remain, mean/change4 gradients must stay live, and the full
NLL native derivative must match the manual mean-only rule.

Both saved checkpoints receive new embeddings and new unchanged fixed heads;
no intermediate checkpoint selection, head tuning, PCA, seed/budget rescue or
old-encoder replay. Market uses one joint four-output D36→16tanh→4 neural head,
Adam0.01/100 steps, ridge1 and reps2701/2802/2903. The D36 seeds remain
13088411117524848790 /10018485049408248344 /9083751155097190455; head sizes148/660.

Dataset: **TEMPO-5** · slow/fast nine-feature lag-sign timing · **complexity 5/5**.
Use the three saved cohorts941101/941102/941103, each512 TRAIN/256 VALIDATION
examples from256/128 paired source groups. Score intact and extra30%-deleted
views under fresh synthetic-TRAIN and fresh market-TRAIN input-scaler policies,
with zero synthetic encoder training. Same binary ridge1/tanh16/100-step recipes
have74/626 parameters. Fit synthetic scalers once/cohort and share across stages.

## Prospective advancement

The clean v3 final2048 scores are frozen controls, reused without encoder/head
execution. Inherited v1/v2/RPB scores impose no gates. All22 numeric conditions
below plus trust admission and100% coverage are required for overall advancement.
Every failed condition and scoped gain remains visible; v3 stays preserved and
selected regardless of outcome. These are known development comparisons, not
confirmation, uncertainty over independent market cohorts or investment evidence.

Dataset: **MKT-HIST-2** · past-window retention · **complexity 4/5** · known VALIDATION32.
Require no decrease in each of the following eight v3 fixed-head R² values:

| Task | Linear R² floor | Neural mean R² floor |
| --- | ---: | ---: |
| past_volatility | 0.964673433811061 | 0.4630519227215611 |
| past_trend | 0.2631329634696965 | -0.2539790011410479 |
| past_activity | 0.8628403110608782 | 0.541426055582653 |
| past_relation | -0.00700444683432222 | -0.08556567147059875 |

Require all8 TEMPO-5 policy/view/head means to retain their exact v3 values in the
JSON card. Require fixed-query TRAIN MAE≤0.25904599043427573 and VALIDATION
MAE≤0.37641423057615903. Require VALIDATION NLL<1.521796577918092, all3 neural
trend R² repetitions>0, and both relation R² heads>0. TRAIN NLL is reported
separately. Retention uses tolerance1e-12, lower-is-better retention allows+1e-12,
NLL strict improvement requires at least1e-12, and positivity is strictly>0.

The count is8 market +8 timing +2 MAE +1 NLL +1 trend robustness +2 relation
usefulness=22. No averaging of tasks supplies an overall accuracy. Keep physical
MAE/RMSE, every neural fit and per-cohort timing scores in the detailed report.

## Preservation and execution

Use only the existing GPU-enabled cuwacunu container and an exclusive verified
research source/build/run capsule. Keep the concurrent dev artifact-snapshot work
untouched. Preserve every v3 source/result/checkpoint, all historical models and
local ignored weights/data. No dev replacement, commit/push or automatic later
experiment is authorized by this declaration. Stop this exact recipe after its
fixed final measurement; further changes require a separate named declaration.
