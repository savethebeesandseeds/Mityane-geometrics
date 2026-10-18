# Frozen-feature historical-information readouts

Current integration uses the freshly trained causal CPB-v3.market-01. This directory's root readout library
and `market_readouts.mk` remain encoder-independent optional tools. Historical
experiment extractors/runners were retired from the active development tree;
their exact source is preserved in the
[causal-primary source archive](../../doc/CAUSAL_PRIMARY_INTEGRATION.md).
The protocol and MKT-HIST-1 explanation below describe the preserved original
readouts. Application encoder inference does not fit readout heads.

This small C++ evaluator consumes saved CPU features and all-past target values.
It never loads an encoder, reads market files, trains a representation, changes
its geometry or applies PCA. The runner owns chronological selection, the
TRAIN-only streaming encoder scaler, CUDA extraction, checkpoint identities and
source/input manifests. A prospective card fixes the pilot before fitting.

The initial pilot is **MKT-HIST-1**, designed complexity **4/5**: three aligned
1-minute assets, all nine candle features, H128 and a chronological holdout.
The level is an ordinal design label, not a measured difficulty or accuracy.
This is a bounded development diagnostic of information within the input history;
it makes no forecasting, trading, independent confirmation or acceptance claim.

## Targets and support

`past_window_targets(raw_values, observed_values)` consumes CPU raw `[N,C,H,9]`
values and matching boolean value masks in the shared kline registry order.
It returns `TargetValues{values, valid}` with float64 `[N,4]` and bool `[N,4]`.
Every calculation uses past ticks only. Masked storage is ignored, including
masked NaNs; observed nonfinite storage is rejected. Positive adjacent observed
closes give `log(close[t]) - log(close[t-1])`. Gaps do not bridge missing ticks.

| Target | Calculation | Minimum support | Units |
| --- | --- | --- | --- |
| Past volatility | Mean across channels of population std of adjacent log close returns | 8 eligible returns in every channel | log return |
| Past trend | Mean across channels of mean adjacent log close returns | 8 eligible returns in every channel | log return/tick |
| Past activity | Mean across channels of mean `log1p(base_volume)` over observed nonnegative volumes | 8 eligible volumes in every channel | log volume |
| Past relation | Mean Pearson return correlation across eligible channel pairs, on each pair's common adjacent support | At least 2 pairs; each needs 8 returns and both population variances at least 1e-12 | correlation |

An unsupported target has `valid=false`; its stored zero is not a target value.
The same target tensors and IDs must accompany raw, initialized and trained
features. Labels are generated from original legal windows before any optional
mask robustness view. The initial pilot uses intact observations only.

## Archive interface

Use the header `cuwacunu/evaluation/fixed_regression.h`. The helpers
`save_features`/`load_features` and `save_targets`/`load_targets` use LibTorch
archives, force CPU loading and validate exact row association. Archive publication
uses a staged file and atomic no-replace hard link; the readout directory is
allocated as an exclusive leaf. Feature and target populations are generic, not
fixed in code.

| Archive | Key | Type/shape |
| --- | --- | --- |
| Per-method features | `values` | CPU float32/float64 `[N,D]` |
| Per-method features | `valid` | CPU bool `[N]` |
| Per-method features | `row_ids` | CPU int64 `[N]`, unique |
| Shared targets | `targets` | CPU float64 `[N,T]` |
| Shared targets | `valid` | CPU bool `[N,T]` |
| Shared targets | `row_ids` | CPU int64 `[N]`, unique |
| Shared targets | `group_ids` | CPU int64 `[N]`, source/support grouping |

Every method must retain the exact target row IDs/order, including invalid rows.
TRAIN and VALIDATION row/group sets must be disjoint. The runner audits actual
time support and role boundaries; group IDs alone cannot prove chronological
purging. Group IDs and row IDs are persisted with predictions. There is one
chronological development split, so no independent-source confidence interval
or retraining uncertainty is claimed.

For this pilot the runner prepares 192 TRAIN and 64 VALIDATION readout windows
with disjoint H128 support inside their roles. Encoder training may use a larger
overlapping selection within TRAIN; these are distinct budgets. No TEST target
or feature is supplied to this evaluator.

## Fixed heads and metrics

The feature normalizer, ridge primal/dual arithmetic and seed mixing follow the
existing C++ helpers in the `mityane_geometrics` embeddings component:
[`code/shared/src/feature_harness.cpp`](../../../../Mityane-geometrics/code/embeddings/code/shared/src/feature_harness.cpp).
Those helpers classify binary labels, so this local regression API changes the
output to T continuous values and the neural loss to masked MSE. It imports no
synthetic generators or old protocol driver and reruns no archived encoder.

The recipe is ridge penalty **1** and one neural head with **16 tanh** units,
Adam **0.01**, **100** full-batch updates and T outputs. T=4 is fixed for this
pilot and identical across methods. Repetitions **2701, 2802, 2903** use the
historical `stream_seed(repetition, feature_width)`; equal-width initial/trained
features receive equal initialization streams. Different raw/native widths
have different parameter counts, explicitly recorded. Ridge is deterministic:
fit it once per method, then reuse it across the three neural repetition records.
These three head fits are not three independent trained encoders.

Fit feature population means/std on valid TRAIN feature rows only, with std
floor 1e-8. A method with fewer than two valid TRAIN feature rows remains named
and unsupported, retaining neutral normalizer storage; it cannot stop supported
methods or fit either head. Fit each target mean/std on its eligible shared TRAIN target rows
only, with std floor 1e-8. Targets are standardized for fitting; predictions are
inverted for scoring. The ridge intercept is unpenalized. Each target's ridge
fit requires two jointly eligible TRAIN rows. Unsupported targets remain named
and unscored. Shared target support/counts are persisted; a target with fewer
than two eligible TRAIN rows retains mean0/scale1 storage defaults and is never
fitted or scored. The batched neural loss averages only eligible supported cells.
If no target is supported for a method, no head fit or update runs; the saved
zero weights are explicitly marked unsupported and fit/update counters are zero.

For each task report raw-unit **MAE** and **RMSE** (lower is better), and **R²**
(higher is better). R² uses the mean of the scored VALIDATION targets in its
denominator, may be negative and is N/A for a constant scored target or fewer
than two scored rows. Coverage is scored valid rows / all declared VALIDATION
rows, not confidence. Target eligibility is retained separately. Do not average
the four tasks into an encoder accuracy. Do not label these values as classifier
percentages. Report each task in its own table with head columns and coverage;
reconstruction error and GPU encoder cost stay in a separate training table.

## Build and run

Build/test exclusively through the managed container, once coordinated with
the runner. Include `code/evaluation/market_readouts.mk` from the root Makefile.

```bash
bash code/scripts/task.sh market-pilot -j2 build-market-readouts test-market-readouts

/opt/cuwacunu/build/market-pilot/market_fixed_regression \
  --train-targets RUN/features/targets_train.pt \
  --validation-targets RUN/features/targets_validation.pt \
  --method raw_mask RUN/features/raw_train.pt RUN/features/raw_validation.pt \
  --method initialized_v22 RUN/features/initial_train.pt RUN/features/initial_validation.pt \
  --method trained_v22 RUN/features/trained_train.pt RUN/features/trained_validation.pt \
  --output-directory RUN/readouts \
  --card-sha256 CARD_SHA256 --input-manifest-sha256 INPUT_MANIFEST_SHA256
```

The new exclusive output leaf holds `report.json` and
`models/{method}/rep-{2701,2802,2903}/{fit.pt,predictions_train.pt,predictions_validation.pt}`.
The complete JSON is also printed to stdout. `fit.pt` includes all feature/target
normalization tensors, ridge weights/intercepts, neural weights/biases, supported
target flags and the actual seed. Predictions include raw-unit ridge/neural
outputs, joint valid masks, exact targets and IDs. The artificial engineering
test reloads these assets and replays their arithmetic without refitting.

Before actual fitting, the root runner must freeze the input/card/source identities
and complete CUDA engineering checks. This evaluator's provenance hash strings
bind its report to that root run; the root must verify the referenced bytes.
No source/card declaration is itself a measured result.
