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

The first comparable control is the reference geometry trained by `code/embeddings/scripts/cpb_family_train.cpp`. The historical V16 inference file is not a `cpb_family_inference_checkpoint_v1` checkpoint, so this readout cannot rescore seed 946103. Family inference equivalence with that seed does not make the historical file a shipping-recipe training run. A full-budget confirmation score of the shipping-recipe control waits until a sealed later window is bound. Development scoring of that control on January can happen while confirmation stays unbound.

These four past-window probes are the whole promotion gate. The screen does not measure OR-09 reconstruction, forecasting, decision utility, or an application selection. `Mityane_application_selection` stays null. A `promote` result still leaves `research-reference.json` unchanged until the owner edits it.

## Measurement

`cpb_family_screen_readout` loads a `cpb_family_inference_checkpoint_v1` checkpoint through the family loader. `--training-state` must be the `cpb_family_causal_training_state_v1` archive for that same encoder. The readout copies its `recipe_identity` and checks the nested encoder's state, scaler, normalizer and causal update count against the inference checkpoint. It does not rename that string to `cpb-family`. It takes native width from `dimensions(config)` and from the saved `native_width`, and the feature tensors must match. `--expected-updates` must equal the checkpoint `causal_updates`; any nonnegative count is legal. Row counts come from the observation-window datasets.

The four targets are the existing `past_window_targets` from `cpb_v16_temporal_v1/deps`. That function assumes the market layout: feature count 9, history at least 9, close at index 3, volume at index 4. Other geometries can still be encoded by the family. This readout fails closed unless the supplied windows have F=9 and H≥9, and unless all four targets are valid. Values are cast through float32 before the target function so the target preparation matches the V17 market readout. The ridge penalty is 1, with TRAIN-only standardization and standard-deviation floor 1e-8. Keep native width 36 when comparing encoder width, so the ridge feature count matches existing native36 records. A different native width is a legal configuration experiment, not a drop-in ranking against those records.

The readout checks that TRAIN and validation row ids are disjoint. Observation windows have no group ids; stored group ids repeat the row ids and are labeled `group_identity: row-id`. A confirmation population must be a disjoint later period. That separation is a protocol obligation. The tool cannot see it from row ids alone.

Running the readout requires CUDA because the family checkpoint loader does. Compiling it does not train a model. The witness kind is `cpb_family_screen_ridge_witness_v1`. The report kind is `cpb_family_screen_linear_readout_v1`. It does not claim legacy target parity.

`cpb_family_screen_audit` recomputes the ridge with scalar long-double centered normal equations and Cholesky. Tolerances are 1e-9 absolute and 1e-8 relative. On an audit of a real witness, Torch is used for the archive codec and for shape and type inspection. The audit repeats the witness recipe and does not require the shipping prefix. Width and row counts come from the witness tensors. The target count stays 4. `--self-test` checks a 2×2 Cholesky, a perfect-prediction metric, agreement between one float64 torch `linalg_solve` and the scalar fit on a synthetic problem, and a synthetic witness with 8 training rows, 3 features, 4 targets and 4 validation rows. The production audit path does not solve with Torch.

```text
node decision.mjs --pairs pairs.json --population POPULATION --role development|confirmation \
  --recipe RECIPE_IDENTITY --completed-updates N --confirmation confirmation.json
cpb_family_screen_readout --settings FILE --checkpoint FILE --training-state FILE --train FILE --validation FILE \
  --output NEWLEAF --expected-updates N
cpb_family_screen_audit --witness FILE --witness-sha256 HEX --output NEWLEAF
cpb_family_screen_audit --self-test --output NEWLEAF
```
