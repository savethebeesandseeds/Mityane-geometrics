# CPB family screen v1

This screen decides whether a later causal-patch-bottleneck candidate may replace the research reference. It does not change that reference. The research default remains V16 seed 946103, final logical update 2048, native36, selected on 22 October 2026, as recorded in `research-reference.json`.

`versions/current.json` and `versions/index.json` are frozen catalogs. Version roles are named in `code/embeddings/encoders/causal_patch_bottleneck_mae/versions/ROLES.md`.

## Populations

January 2024 MKT-HIST-2 days 22–28 (`january-2024-mkt-hist-2-days-22-28`) is development. February 2024 and March 2024 are consumed confirmation. `confirmation.json` starts unbound. Development training on that January population may proceed while confirmation stays unbound. It cannot promote. Binding a later population is an owner edit of that file (`status` `sealed`, `bound.id` equal to the population, `bound.consumed` false) and must happen before any training on that population. This tool has no bind command and does not write `confirmation.json` or `research-reference.json`.

A development population cannot promote. Passing development scores return `continue_at_reference_budget` and still leave the reference unchanged. Take that survivor to 2048 updates before spending the sealed window. Do not download a later market window, train on a confirmation population, or edit the research reference until that population is sealed. No later month is invented here.

## Decision

`decision.mjs` reads paired candidate-minus-control R² values for `past_volatility`, `past_trend`, `past_activity` and `past_relation`. The pair count must be odd so the median is unique. The practical threshold is 0.01 R². Nonnegative pair counts are reported with every decision; they do not add a second gate.

| Result | When |
| --- | --- |
| `promote` | Confirmation, sealed to this population, not consumed, `recipe_identity` begins with `cpb-family-train-example-v1;`, at least 2048 completed updates, trend median ≥ 0.01, activity median ≥ 0.01, volatility median ≥ −0.01 |
| `refuse_promotion` | Those score gates pass, but seal, recipe, budget or consumption does not |
| `continue_at_reference_budget` | Development, and the same three score gates pass |
| `extend_for_volatility_or_relation` | Otherwise, when volatility or relation median ≥ 0.01. A trend or activity failure stays in the reasons. Relation can buy this extension even when trend is largely negative. There is no tradeoff score. |
| `stop` | Trend median < −0.01 and activity median < −0.01, and neither volatility nor relation buys an extension |
| `inconclusive` | Remaining screens, including every median inside [−0.01, 0.01). This neither kills nor passes. |

Historical V17, V18 and V19 paired tables are fixtures for this rule. Their old study predicates and records stay as published. V17 is inconclusive here. V18 and V19 extend. None of them promote: their recipes do not begin with `cpb-family-train-example-v1;`, and their populations are consumed.

The promotion recipe is the shipping trainer's stored `recipe_identity`. That string begins `cpb-family-train-example-v1;` and then records the executable, settings, dataset, batch, seeds and optimizer. The short token `cpb-family` is not a recipe and cannot promote. Pass the training state's `recipe_identity` unchanged. Historical recipe strings cannot be re-promoted.

The first comparable control for the published probes is the reference geometry trained by `code/embeddings/scripts/cpb_family_train.cpp` and scored with those four names. The historical V16 inference file is not a `cpb_family_inference_checkpoint_v1` checkpoint, so this readout cannot rescore seed 946103. January development windows use the withdrawn instrument binding `{0,1,2}`. This readout refuses that binding. It does not open a timeframe training run and it does not bind a confirmation population.

These four past-window probes are the whole promotion gate for paired tables that already use those names. The live readout no longer emits them. Do not place `cross_resolution` in `past_relation`, or any anchor column in the matching old name. The screen does not measure OR-09 reconstruction, forecasting, decision utility, or an application selection. `Mityane_application_selection` stays null. A `promote` result still leaves `research-reference.json` unchanged until the owner edits it.

## Measurement

The live readout scores one instrument. Channels are time resolutions of that instrument. It does not average those resolutions together, and it does not score the published three-instrument probes.

`cpb_family_screen_readout` loads a `cpb_family_inference_checkpoint_v1` checkpoint through the family loader. `--training-state` must be the `cpb_family_causal_training_state_v1` archive for that same encoder. The readout copies its `recipe_identity` and checks the nested encoder's state, scaler, normalizer and causal update count against the inference checkpoint. It does not rename that string to `cpb-family`. It takes native width from `dimensions(config)` and from the saved `native_width`, and the feature tensors must match. `--expected-updates` must equal the checkpoint `causal_updates`; any nonnegative count is legal. Row counts come from the observation-window datasets. Channel ids must match the encoder, in order, and the dataset sampling interval must match `SAMPLING_INTERVAL`.

`--feature-view` is required. `raw` computes anchor log returns from positive closes and log1p from nonnegative volume. `log-returns` means close at index 3 is already the per-bar log return and volume at index 4 is already log1p, which is the read-time view from `history_from_edge_samples`. The readout does not infer the view and does not apply a second logarithm. A negative stored return is legal only in `log-returns`.

Targets are taken from the window `boundary::retain` gives the encoder: the last `HISTORY_LENGTH` ticks ending at the cutoff. Feature count is 9. The anchor channel is the one whose fixed millisecond step equals `SAMPLING_INTERVAL`. Every other channel step is a positive multiple of that step. Ids `{0,1,2}` are refused. A calendar month, whose step is 0, is refused.

| Task | Definition |
| --- | --- |
| `anchor_volatility` | Population standard deviation of the anchor channel's returns. At least 8 returns. |
| `anchor_trend` | Mean of those same returns. |
| `anchor_activity` | Mean anchor log1p base volume. At least 8 values. |
| `cross_resolution` | Mean Pearson correlation between the anchor return and each coarser channel, using one sample per held bar. A bar is a run of identical observed feature vectors. A channel needs at least 8 paired bars and both population variances at least 1e-12. The row abstains when no coarser channel qualifies, including a one-channel window. Abstention stores an invalid column, not a correlation of zero. |

The paired anchor value is the anchor return at the bar's change tick. Held repeats are not extra samples, and the anchor ticks inside the hold are not summed. Two different bars that store the same nine values count as one run, because this window has no per-channel key.

`anchor_volatility`, `anchor_trend` and `anchor_activity` must be valid on every retained TRAIN and validation row. `cross_resolution` may abstain on some or all rows. The ridge penalty is 1, with TRAIN-only feature standardization, standard-deviation floor 1e-8, and one fit per column. An anchor column with zero TRAIN variance fails closed. An abstaining cross-resolution column is reported with null R² and does not remove the anchor scores. Feature standardization uses every TRAIN row. A cross-resolution fit uses only the TRAIN rows where that column is valid.

These names are not inputs to `decision.mjs`. They are not a ranking against V16 seed 946103. A comparison to that seed needs a control trained and scored on this same target contract. This readout does not select a population, bind confirmation, or edit the research reference. Keep native width 36 when comparing encoder width to existing native36 records. A different native width is a legal configuration experiment.

The first measured pair on this contract is recorded in the [29 October handoff](../../../encoders/causal_patch_bottleneck_mae/family/migration/HANDOFF_2026-10-29.md). Width 128, seed 949102, leads width 64, seed 949101, on anchor volatility and anchor trend after 2048 updates. Anchor activity is saturated. Cross resolution abstains for a 128-minute history. That note offers the width-128 instance to Cuwacunu and leaves this policy unchanged.

The frozen `past_window_targets` in `cpb_v16_temporal_v1/deps` is unchanged. Historical three-instrument scores stay scores of that function, including the 2026-10-29 log-return screen, which applied that raw-price target function to an already logged window. V17, V18 and V19 readouts are unchanged.

The readout checks that TRAIN and validation row ids are disjoint. Observation windows have no group ids; stored group ids repeat the row ids and are labeled `group_identity: row-id`. A confirmation population must be a disjoint later period. That separation is a protocol obligation. The tool cannot see it from row ids alone.

Running the readout requires CUDA because the family checkpoint loader does. Compiling it does not train a model. The witness kind is `cpb_family_screen_timeframe_witness_v1`. The report kind is `cpb_family_screen_timeframe_readout_v1`. It does not claim legacy target parity. `comparable_to_seed_946103` and `decision_input` are both false.

`cpb_family_screen_timeframe_targets_test` checks the target contract on synthetic CPU tensors. It does not load a checkpoint or select a population. It checks that a coarser channel cannot change the anchor statistics, that a log-return view is not logged again, that held repeats are one cross-resolution sample, and that too few pairs abstain.

`cpb_family_screen_audit` still accepts only `cpb_family_screen_ridge_witness_v1` and still names `past_volatility`, `past_trend`, `past_activity` and `past_relation`. It does not audit a timeframe witness. It recomputes that older ridge with scalar long-double centered normal equations and Cholesky. Tolerances are 1e-9 absolute and 1e-8 relative. On an audit of a real witness, Torch is used for the archive codec and for shape and type inspection. The audit repeats the witness recipe and does not require the shipping prefix. Width and row counts come from the witness tensors. The target count stays 4. `--self-test` checks a 2×2 Cholesky, a perfect-prediction metric, agreement between one float64 torch `linalg_solve` and the scalar fit on a synthetic problem, and a synthetic witness with 8 training rows, 3 features, 4 targets and 4 validation rows. The production audit path does not solve with Torch.

```text
node decision.mjs --pairs pairs.json --population POPULATION --role development|confirmation \
  --recipe RECIPE_IDENTITY --completed-updates N --confirmation confirmation.json
cpb_family_screen_readout --settings FILE --checkpoint FILE --training-state FILE --train FILE --validation FILE \
  --output NEWLEAF --expected-updates N --feature-view raw|log-returns
cpb_family_screen_audit --witness FILE --witness-sha256 HEX --output NEWLEAF
cpb_family_screen_audit --self-test --output NEWLEAF
```
