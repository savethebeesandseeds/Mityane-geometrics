# Proposed human-report outline: saved TRAIN V9/V11 diagnosis

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root.

This is a planning artifact prepared from the frozen scientific source and reporting standard. It contains no diagnostic result, new numerical claim, selected mechanism or execution admission. Fill measured placeholders only after the actual main diagnosis, independent arithmetic audit and exact mirrors pass and their current bytes are reviewed. Preserve this outline and every earlier source, card, result, failure and report.

## Evidence navigation and publication boundary

Expected canonical documents, whose future existence and contents are not asserted by this outline:

- Main diagnosis: `doc/causal_v9_v11_train_diagnosis/diagnosis.json`.
- Main execution: `doc/causal_v9_v11_train_diagnosis/execution.json`; exact mirror: `doc/causal_v9_v11_train_diagnosis/mirror.json`.
- Independent audit: `doc/causal_v9_v11_train_diagnosis/independent-audit.json`; execution: `independent-audit-execution.json`; exact mirror: `independent-audit-mirror.json` in that same directory.
- Completion receipt: use the exact path returned by the separately invoked completion verifier; do not invent a successful receipt or its hash.
- Actual admitted main card: `doc/CAUSAL_V9_V11_TRAIN_DIAGNOSIS_CARD.ADMITTED.json`; preserve the distinct false `.PROPOSED.json` and both Markdown companions.
- Historical quality remains in `doc/CAUSAL_FRESH_V9_RESULTS.md` and `doc/CAUSAL_FRESH_V11_RESULTS.md`, under each original card, evaluation population and pass count. Do not replace their held-out results with TRAIN diagnosis scores.

Before final prose, bind exact paths, byte lengths and SHA256 of both actual cards, main/audit results, executions, full mirrors and completion review. Confirm actual statuses/types, zero mismatches, unchanged source/guard bytes and the actual independent check count. Source code describes the intended output schema; it cannot establish that a run passed. Do not copy a status or check count from this outline or a historical audit.

The final report should open with a factual evidence-stage sentence after review: "No new encoder experiment. This report describes newly audited scalar arithmetic on saved V11 TRAIN artifacts and four unchanged cached V9 subtrees." Add actual main/audit status, counts and timings only from their saved receipts. State that V3 remains the selected baseline and sole DEV encoder, and that this diagnosis selects no new encoder or application change.

## Population, methods and what changed

Dataset: **MKT-HIST-2** · joint BTC/ETH/SOL January2024 minute histories, C3/H128/F9, recent128 past-window retention · **complexity4/5, ordinal**. The three assets are joint input channels in one historical panel, not three independent market cohorts.

The declared diagnosis scores the same112 chronological TRAIN examples used to fit the saved heads. Initial/final mean the saved0/2048-update checkpoints. There was one historical fresh CUDA encoder trajectory per version,2048 updates at batch8; no new encoder training, calibration, extraction or head fit is part of this diagnosis. Three head repetitions2701/2802/2903 measure variation of fixed heads on one checkpoint/split, not independent encoder retraining or market-cohort uncertainty. Known chronological VALIDATION32 and all TEST/stress/TEMPO numeric inputs are excluded.

Both versions serve native40: learned content20, literal spectral12, mean4 and independent uncertainty4. V11's scientific source/objective parent is V9; this is physical code lineage only. V11 learned values, optimizer and permitted TRAIN statistics start fresh with source_updates0. No trained V9 or historical oracle weights initialize it. V10 is a historical measured control, not V11's scientific source or initializer.

The architecture difference is intermediate temporal pooling: V9 uses eight mean-export rows per original bin; V11 uses four mean and four componentwise centered-variance rows under the same normalized half-local/half-global attention. The existing two-sum mean, original-position bins, post-bias whole-slice zero for empty local support, downstream content bottleneck, native40 and registered parameter shapes remain. The intermediate variance summaries are distinct from native uncertainty4 and decoder conditional variance. Equal parameter counts do not imply equal functions, capacity allocation, units, gradients or FLOPs.

Both retain unit Huber(delta1) mean plus unit detached-error Gaussian variance calibration, shared global norm2 clip1 and one AdamW. V10's live joint-Gaussian mean route is excluded. Future residual targets, clipping and optimizer history retain coupling despite zero direct cross-path native derivatives.

Initial pairing semantics remain exact saved V9 DATA20:40 plus five whole fresh-statistic tensors. Learned content0:20 is unpaired, so whole-native40 or initial-score equality is not implied. The historical V10 whole40 pairing has different semantics and stays unchanged. This diagnosis does not replay initial pairing or causal/learning/typed-resume proofs.

Saved fixed readouts use ridge penalty1 and16-tanh/four-output neural heads, Adam0.01/100 updates, permitted TRAIN-only normalizers and D40 seeds6049515622977958931/8262941649267994858/2095371932330426401. Market heads have164/724 fitted values. Preserve all repetitions; select no best fit, head setting, checkpoint or latent transform.

## Exact JSON field navigation

In this map, `S` is `initial` or `final`; `T` is one of the four task keys; `H` is `linear` or `neural`; `R` means select the array record with `repetition`2701,2802 or2903; `L` is `early56` or `late56`. These are descriptive selectors, not executable project code.

- `V = models.v11.stages.S`: all newly computed V11 stage descriptions.
- `C = cached_v9_TRAIN.stages.S`: unchanged old audited V9 stage subtree, copied from the prior diagnosis. New V9 numerical reductions remain zero.
- Other cached references are exactly `cached_v9_TRAIN.actual_pre_step_batch_diagnosis` and `cached_v9_TRAIN.temporal_learning_observations`. Do not describe these cached old panels as newly recomputed or as matched V11 moment panels.

| Report panel | Exact main-result field or selection | Meaning and limit |
| --- | --- | --- |
| Four primary TRAIN tables | `V.per_task_mean_across_saved_repetitions.T.H.{r2,mae,rmse,coverage}` and the identical path under `C` | Saved-head TRAIN scores; neural column is the saved aggregate across all three repetitions. Coverage is stored as a fraction; percent display is formatting only. |
| Full repetitions and physical errors | `V.saved_TRAIN_head_scores[R].tasks.T.H.{r2,mae,rmse,valid,total,coverage}`; same under `C` | Copy every repetition and both heads. Physical task MAE/RMSE are distinct from standardized reconstruction errors. |
| Fixed chronological halves | `V.saved_TRAIN_head_scores[R].chronological_TRAIN_halves.L.T.H` and same under `C` | Fixed early56/late56 descriptions. No regime selection, independent holdout or causal time-shift explanation. |
| Native blocks | `V.native_features.blocks.{learned20,fixed_spectral12,mean4,uncertainty4,complete40}`; same under `C` | `width`, `valid_rows`, `covariance_trace`, `covariance_frobenius_squared`, `participation_ratio`. Participation ratio is coordinate-scale-dependent covariance concentration, not numerical rank. |
| Coordinates | `V.native_features.coordinates[]` and same under `C` | `coordinate`, `block`, `mean`, `population_std`, `minimum`, `maximum`, `pearson_to_TRAIN_targets`. No cross-version basis alignment. |
| Saved ridge block allocation | `V.saved_TRAIN_head_scores[R].saved_linear_readout_block_decomposition.T` and same under `C` | Four block contributions of an already fitted predictor, including `blocks`, `pairwise_component_covariance`, variance totals, `constant_target_units` and `chronological_TRAIN_halves`. No block ablation/refit. |
| All112 fixed queries | `V.fixed_query_TRAIN_diagnosis.all112_TRAIN` and same under `C` | `standardized_fixed_query_MAE`, `Gaussian_NLL_F64`, `mean_Huber_F64`, `variance_NLL_F64`, `total_objective_F64`, `target_cells`, saved row vectors and `chronological_TRAIN_halves`. Keep metric units and objectives separate. |
| Detailed first8 queries | `V.fixed_query_TRAIN_diagnosis.{detailed_population,detailed_row_ids,per_channel_feature,B8_hierarchical_mean_MAE,B8_hierarchical_mean_NLL,B8_target_cells,B8_fixed_Huber_delta1}` and same under `C` | Cellwise evidence covers first8, not all112. New V11 output excludes old derivative/raw-age panels even though cached V9 subtree retains its historical extra fields. |
| Mean4 residual allocation | `V.saved_mean_residual_allocation.per_channel_feature[]` and same under `C` | `channel_id`, `feature`, `target_cells`, `residual_quantiles`, `parent_remainder_quantiles`, `parent_residual_pearson`; matching stage's saved linear change decoder only. |
| Reconstruction/readout associations, V11 only | `V.reconstruction_readout_associations[]` filtered by `repetition`, `task`, `head`, `scope` | `eligible_rows`; `components.{row_mae,row_nll,row_mean_huber,row_variance_nll,row_total_objective}.{pearson_to_signed_readout_residual,pearson_to_absolute_readout_residual}`. No matched V9 panel or validation cause. |
| Cached scalar comparisons | `comparison.saved_TRAIN_readout_scalar_differences[]`, `native_block_descriptor_differences[]`, `full112_reconstruction_scalar_differences[]` | Explicit saved `cached_V9`, `new_V11`, `V11_minus_cached_V9`; no old V9 raw reduction/refit, new floor or aggregate encoder score. |
| Moment subgroup CSV, V11 only | `temporal_moment_observations.csv.{recorded_rows,CSV_columns,overall_and_fixed512[]}` | Fixed blocks contain `name`, update range and `subgroups[]`; each subgroup has `bin`, `moment`, export-row interval, `records`, `scalars.<column>.{mean,quantiles}`. |
| Actual captured moments, V11 only | `temporal_moment_observations.actual_witnesses[]` selected by `update`1 or2048 | `pre_step_completed_updates`, `attempt`, `clip_factor`, packed row/position/valid metadata, geometry and both routes. Update2048 describes pre-step2047, never final2048. |
| Actual per-bin allocation | `...actual_witnesses[update].routes[route].bins[bin]` | `present_local_rows`, `local_member_counts`, moment/mixture/export/PRECLIP-upstream quantiles and manual export values. Select routes by `route` ordinary/contextual and retain both. |
| Typed ancestry | `V.checkpoint_provenance` | Fresh source0; initial/final causal progress and exact layout/objective/pooling/moment metadata. Reuse proof context without replaying trust tests. |
| Disclosed arithmetic | `manual_arithmetic_counters`, `manual_arithmetic_counter_units` | Five manual affine families and implementation-specific scalar-summary invocation counters. These are performed arithmetic even while module/model/fit/update counters are zero. |
| Access and integrity | `card`, `owned_source_pins`, `all_owned_source_files_unchanged`, `verified_allowed_inputs`, `verified_hash_only_guards`, `verified_context_inputs`, `verified_pre_execution_preservation`, `opened_allowed_ids`, `numeric_fields_accessed` | Bind the actual result, guards and exact read scope. Opaque full-file hashing is distinct from selected numeric value access. |

## Four primary TRAIN tables — placeholders only

Use the five-column regression format below for each task. Copy reviewed saved aggregate fields; do not derive new scores for reporting. Preserve full precision in linked JSON and disclose display rounding. Use `N/A` for constant/unsupported scored targets, never zero. If support differs, report valid/total for each head and the common comparison population; do not imply one universal coverage number. Each task's units also accompany detailed physical-error tables.

### Past volatility

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns in recent128 · **complexity4/5, ordinal** · score: saved TRAIN112, in-sample. Physical error unit: logreturn.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 40 | pending reviewed cached field | pending reviewed cached field | pending support |
| CPB-v11.market-01.init | 40 | pending reviewed new field | pending reviewed new field | pending support |
| CPB-v9.market-01 | 40 | pending reviewed cached field | pending reviewed cached field | pending support |
| CPB-v11.market-01 | 40 | pending reviewed new field | pending reviewed new field | pending support |

Task selector: `past_volatility`.

### Past trend

Dataset: **MKT-HIST-2** · channel-mean supported adjacent close log-return trend in recent128 · **complexity4/5, ordinal** · score: saved TRAIN112, in-sample. Physical error unit: logreturn/tick.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 40 | pending reviewed cached field | pending reviewed cached field | pending support |
| CPB-v11.market-01.init | 40 | pending reviewed new field | pending reviewed new field | pending support |
| CPB-v9.market-01 | 40 | pending reviewed cached field | pending reviewed cached field | pending support |
| CPB-v11.market-01 | 40 | pending reviewed new field | pending reviewed new field | pending support |

Task selector: `past_trend`.

### Past activity

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume) in recent128 · **complexity4/5, ordinal** · score: saved TRAIN112, in-sample. Physical error unit: log1p(base_volume).

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 40 | pending reviewed cached field | pending reviewed cached field | pending support |
| CPB-v11.market-01.init | 40 | pending reviewed new field | pending reviewed new field | pending support |
| CPB-v9.market-01 | 40 | pending reviewed cached field | pending reviewed cached field | pending support |
| CPB-v11.market-01 | 40 | pending reviewed new field | pending reviewed new field | pending support |

Task selector: `past_activity`.

### Past relation

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns in recent128 · **complexity4/5, ordinal** · score: saved TRAIN112, in-sample. Physical error unit: correlation.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v9.market-01.init | 40 | pending reviewed cached field | pending reviewed cached field | pending support |
| CPB-v11.market-01.init | 40 | pending reviewed new field | pending reviewed new field | pending support |
| CPB-v9.market-01 | 40 | pending reviewed cached field | pending reviewed cached field | pending support |
| CPB-v11.market-01 | 40 | pending reviewed new field | pending reviewed new field | pending support |

Task selector: `past_relation`.

## Detailed report panels to populate after audit

Repeat the MKT-HIST-2 legend beside every final diagnostic/cost table. Maintain task separation; no overall R² or timing/reconstruction/readout composite score.

1. For each task, show both stages, both versions, all three saved repetitions, both head scores and physical MAE/RMSE/support. Show fixed early56/late56 descriptions without selecting a regime or fit.
2. Describe native block covariance concentration and saved ridge contributions, all four tasks/repetitions, including pairwise contribution covariance and fixed halves. Correlated readout components are not latent-gradient conflict, destructive training, or evidence that deleting a branch helps.
3. Keep all112 fixed-query MAE, reporting Gaussian NLL, applied Huber mean, detached-error variance term and total objective separate. Gaussian NLL omits its constant and may be negative; TRAIN NLL adds no retention gate. First8 cellwise findings may differ from all112 and do not represent its whole distribution.
4. Describe the first8 same-stage saved linear mean4 residual and stored-total-minus-residual parent remainder. The F64 contraction approximates the original F32 parent-plus-residual addition. No parent nonlinear decoder, content/spectral decoder split, intervention or new predictor is constructed.
5. Include every V11 task/head/repetition for all112, early56 and late56 reconstruction/residual associations. Name eligible support and signed/absolute physical readout residual. Query errors use standardized reconstruction units. Keep null/undefined correlations; choose no attractive subset and attach no diagnostic success gate.
6. Describe all eight V11 bin/moment subgroups and the prescribed overall/four512-update blocks. Six recorded numeric columns are weight/bias gradient L2, weight/bias movement L2 and two element counts. These are postclip observations in parameter coordinates; movement includes AdamW decay. Do not recompute event/learning/gauge/Hadamard trust gates or infer importance, task contribution or gradient dominance from magnitudes.
7. Retain ordinary/contextual routes at captured updates1/2048. Report original support, present-local row counts, means/centered variance, full-mixture concentration and exported mean4/variance4 allocation. Quantiles select present-local rows; mixture concentration uses valid whole-mixture tokens. Manual affine exports compute all packed rows before post-bias empty-slice zeroing. Preserve the upstream-defined flag; an unused route's saved zero fallback is not evidence of a dead learned branch.

There are no matched V9 intermediate moment captures and no all112 V11 query-row intermediate moments. The two actual batches cannot establish per-task moment contributions. The update2048 witness is pre-step2047 and uses its saved before-step export coefficients; it must never receive the final2048 decoder or be described as final2048 activation evidence. The initial/final first8 query residual panel separately uses its own stage-matched decoder.

## Arithmetic, timing, access and preservation disclosure

The final report must say that manual scalar algebra was performed. It must not summarize this as "no inference/arithmetic" merely because the module-forward counters are zero.

- Copy the actual five affine-family counters: `ridge_total`, `ridge_block`, `mean4_decoder`, `temporal_mean4_export`, `temporal_variance4_export`. `dot_outputs` counts scalar dot outputs; `multiply_add_terms` counts products in those dot sums, excluding bias additions. Empty-local export rows still count because they are computed then masked.
- Scalar mean/quantile/variance/Pearson/score invocation counts are implementation-specific summary calls, not primitive FLOPs. Independent audit counters need not equal producer summary-call counts; its affine output/product counters must follow their declared exact contract.
- Separately state actual zero model/encoder/decoder/head/module/neural forward, fit, training-update, held-out numeric-read, old V9 raw-reduction and old trust-replay counters. Saved neural predictions are read; neural weights or forwards are excluded. Do not claim the manual saved linear decoder contraction was absent.
- Bind25 numeric roles,34 hash-only guards,12 metadata contexts and the declared alias/count relationships to actual before/after receipts. Hash-only old V9/PT bytes are never new numeric evidence. Query-summary/cached JSON can be decoded whole while numerical selection stays TRAIN/four closed V9 selectors; describe both facts.
- Report main result `elapsed_seconds`, independent result `elapsed_seconds`, and each execution recorder's actual `elapsed_seconds` separately. These are diagnosis/audit body and wrapper costs, not new GPU training or matched speed benchmarks. Historical encoder training cost is not added to these values. Do not fill a new encoder-training table with diagnostic runtime.
- Reuse old V9's687825-check independent audit as its historical cached-reference authority. Fill the new independent check count from the actual current audit. This new diagnosis replays no local-adjoint/finite-difference/learning-trust proof; never present the old256 finite differences as new V9/V11 checks.

## Interpretation and next-direction decision boundary

After the actual arithmetic/audit passes, write what the four TRAIN tasks, query populations, readout decompositions and new V11 moment observations show, using only their declared populations and units. Preserve every null, regression and contrary panel. State unresolved tradeoffs before any subsequent source proposal.

Use these boundaries when forming the later discussion; none selects a mechanism now:

- A saved TRAIN readout gain establishes retained in-sample fitted information under the fixed heads. A regression establishes its observed reduction on this saved population. Neither identifies a cause of the existing VALIDATION differences or supports forecasting/generalization.
- Reconstruction improvement and readout improvement are separate observations. Their rowwise association remains descriptive and cannot justify a loss coefficient, task-supervised objective, pooled-bin change or checkpoint rescue.
- Native covariance concentration and block contribution covariance describe saved feature/readout geometry. They establish neither missing dimension, necessary width increase, branch competition nor a decorrelation remedy.
- Captured moment/export and upstream magnitudes describe two recorded training forwards in their own coordinates. They do not establish branch importance, actual matched V9 effects, complete2048-step derivatives, full optimizer norms, missing intermediate traces or a task-level causal mechanism.
- Existing live subgroup evidence comes from preserved proof/audit context. Reuse it without repeating the old scientific proof or presenting its scope as every-scalar learning.

The exact V11 recipe remains closed; this report supplies no coefficient/seed/rate/budget/head/checkpoint rescue. Any future distinct mechanism requires a separate prospective source/card/criteria declaration after the diagnosis and its independent review. Use the already applied V11-retention protocol for future candidates without rescoring historical pass counts. Preserve V3 as baseline/DEV and every V3–V11 historical scoped leader, checkpoint, result, failure and card. No next encoder, overall winner, promotion or publication is selected by this outline.

## Sources used to prepare this outline

- `doc/RESULTS_REPORTING_STANDARD.md`, version1.2.
- `doc/CAUSAL_V8_V9_TRAIN_DIAGNOSIS.md`, established report format and historical cached-reference scope.
- `doc/CAUSAL_FRESH_V11_RESULTS.md`, established scientific change, ancestry, initial-pairing and closure semantics.
- `code/research/causal_v9_v11_train_diagnosis/diagnose.py` and `README.md`, exact intended new result schema/access/arithmetic contracts.
- `code/research/causal_v9_v11_train_diagnosis_audit/audit.py`, intended independent result schema and limits.
- `doc/causal_v9_v11_train_diagnosis/run-diagnosis.ps1` and `verify-completion.ps1`, canonical navigation only; neither invoked or changed for this outline.

Preparation performed host text inspection and this additive Markdown write only: zero helper, AST, schema, Docker, PT-value, numerical reduction, affine-contraction, model/head, fit or update calls. Actual diagnosis/audit result arrays were not inspected to prepare these placeholders.
