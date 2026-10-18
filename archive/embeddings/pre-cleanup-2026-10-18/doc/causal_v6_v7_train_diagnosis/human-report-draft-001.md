<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE. -->
# Provisional saved TRAIN V6/V7 diagnosis

**Independent diagnosis audit pending. This is an interpretation draft, not a
new encoder declaration or a final scientific report.** No new experiment,
head fit or quality measurement. The once-executed saved-data diagnosis passes
its producer checks. It uses zero model, encoder, decoder or neural-head calls,
zero fits or training updates, zero raw-cache numeric reads and zero held-out
numeric fields. The subprocess took2.231799121s; the report's internal elapsed
field is1.733104871s. These are scalar-diagnosis costs, not training costs.

The bounded finding is that **both final encoders represent the TRAIN tasks
well, and the saved evidence does not identify a temporal-pooling failure**.
V7's improved reconstruction over112 TRAIN queries reverses on the first-eight
subset. On that subset the oldest32-tick bin has the lowest error, while middle
bins have higher error and different support. V7's uncertainty branch clips
frequently, but its mean branch never clips; these observations cannot reconstruct
V6's missing branch norms or explain the known held-out regressions. A different
temporal summary is worth a separately declared structural comparison after the
audit, rather than another adjustment of the stopped clipping recipe.

The [prospective card](../CAUSAL_V6_V7_TRAIN_DIAGNOSIS_CARD.md) binds49 permitted
inputs,31 preservation pins and the final V7 preservation receipt. The machine
card is67180 bytes, SHA256
`6067c3f5128991619e72197dbc5b8ea3f44f6e7e4186a56f4962f70c43b03304`.
The [saved diagnosis](diagnosis.json) is4561712 bytes, SHA256
`235aaecbd19be26292944f4e539192286332713f189f681898bf915c2764cf58`.
All81 admitted pins and the frozen owned source remain exact. The independent
audit must verify the saved arithmetic before this draft is finalized.

## Population and evidence boundaries

V6 and V7 each completed one fresh2048-update, batch8 CUDA trajectory,
source_updates=0. Native40 is learned content20, fixed literal spectral12,
mean4 and independent uncertainty4. The same constructor seed943001, draw
order, TRAIN-only calibration policy, dataset, sampling/mask streams, forward
capacity and D40 head recipes were used. All294752 learned values began fresh;
the3888 literal prior values are fixed. No learned weights or statistics were
imported from either reference.

Both versions apply unweighted Huber(delta1) mean supervision plus Gaussian
variance calibration using detached mean error, coefficients1/1. Reporting
Gaussian NLL is detached and never a training mean gradient. V6 applies one
global norm2 cap1. V7 separately clips mean73 tensors/293537 values and
uncertainty3 tensors/1215 values at norm2 cap1 before one AdamW. Its combined
gradient norm may reach sqrt2. Future residual targets and optimizer history
continue to couple the trajectories; separate clipping is not full optimizer
independence.

The112 chronological TRAIN probe windows come from the same January2024
BTC/ETH/SOL history, with TRAIN cutoff1705881599999. They are not112 independent
market cohorts. The fixed heads are ridge penalty1 and tanh16/Adam0.01/100
updates, using the three saved fits2701/2802/2903 and paired D40 seeds
6049515622977958931,8262941649267994858,2095371932330426401. Market heads have
164/724 fitted parameters. Three fits on one encoder/split are head variation,
not three independently trained encoders. No post-encoder PCA or new head
forward is used here.

The evidence has five distinct scopes:

- **112 natural-mask TRAIN rows per initial/final stage:** saved four-task
  predictions, saved ridge coefficients and original TRAIN normalizers, native
  covariance/participation summaries and fixed-readout block contributions.
- **112 fixed-query TRAIN scalar rows per stage:** MAE, Gaussian NLL, Huber mean,
  variance NLL, total objective and support. Both versions save these components.
  No VALIDATION subtree is used from mixed aggregate JSON files.
- **First8 fixed-query TRAIN distributions:**1728 supported target cells per
  stage, with the matching saved initial/final decoder coefficients. Local
  mean4/uncertainty4 adjoints and four32-tick raw/reconstruction bins cover only
  these rows, not the full112 population automatically.
- **All2048 successful TRAIN update scalars per version:** observed objective
  components, norm/factor records and four prospectively fixed512-step blocks.
  V6 supplies global norm/factor; V7 supplies its two branch norms/factors.
- **Two actual TRAIN batches per version:** updates1 and2048, whose saved
  forwards are pre-step states0 and2047. They support component, mask, support,
  CSV-association and observed clipping checks. Final decoder weights are
  post-step2048, so no actual-batch decoder adjoints are reconstructed.

Complexity4/5 below is an ordinal signal-recipe label, not accuracy or observed
task difficulty. The tables reuse saved TRAIN predictions; they are in-sample
readout diagnostics and do not measure a new held-out score.

## Saved TRAIN task readouts

Both independently saved initial native40 representations and their scores
agree exactly. Final values below are means of the three saved head fits; all
112 declared rows are valid. R²=0 means the scoring population's constant mean,
and negative R² would be worse. Physical errors use the original task units.

### past_volatility

> Dataset: **MKT-HIST-2** · continuous past volatility · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, nine raw features, history128 · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v6.market-01 · initial | 40 | 0.856612 | 0.921619 | 100 |
| CPB-v7.market-01 · initial | 40 | 0.856612 | 0.921619 | 100 |
| CPB-v6.market-01 · final2048 | 40 | 0.991300 | 0.961221 | 100 |
| CPB-v7.market-01 · final2048 | 40 | 0.990833 | 0.960846 | 100 |

Final linear/neural MAE, in log-return units: V6
`3.783827e-5 / 7.310413e-5`; V7 `3.949502e-5 / 7.298620e-5`.
RMSE: V6 `4.694765e-5 / 9.884953e-5`; V7 `4.819025e-5 / 9.904618e-5`.
Both final representations support a strong TRAIN readout. The known V7 neural
volatility quality gain is not accompanied by a higher TRAIN R² here; these
different populations/metrics do not provide its cause.

### past_trend

> Dataset: **MKT-HIST-2** · continuous past trend · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, nine raw features, history128 · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v6.market-01 · initial | 40 | 0.675497 | 0.954260 | 100 |
| CPB-v7.market-01 · initial | 40 | 0.675497 | 0.954260 | 100 |
| CPB-v6.market-01 · final2048 | 40 | 0.919068 | 0.950138 | 100 |
| CPB-v7.market-01 · final2048 | 40 | 0.923769 | 0.955946 | 100 |

Final linear/neural MAE, in log-return/tick: V6
`1.829895e-5 / 1.374199e-5`; V7 `1.763741e-5 / 1.293630e-5`.
RMSE: V6 `2.375324e-5 / 1.862766e-5`; V7 `2.305306e-5 / 1.749918e-5`.
V7 fits this TRAIN population at least as well by these saved metrics. Its
known neural-trend robustness failure therefore cannot be explained by simple
inability to encode or fit TRAIN trend alone. It does not identify the source
of generalization failure.

### past_activity

> Dataset: **MKT-HIST-2** · continuous past activity · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, nine raw features, history128 · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v6.market-01 · initial | 40 | 0.952488 | 0.950540 | 100 |
| CPB-v7.market-01 · initial | 40 | 0.952488 | 0.950540 | 100 |
| CPB-v6.market-01 · final2048 | 40 | 0.973945 | 0.971945 | 100 |
| CPB-v7.market-01 · final2048 | 40 | 0.971770 | 0.969487 | 100 |

Final linear/neural MAE, in log1p(base-volume) units: V6
`0.078173 / 0.082004`; V7 `0.081501 / 0.084705`.
RMSE: V6 `0.100230 / 0.103908`; V7 `0.104330 / 0.108448`.
Both improve over initialization on this TRAIN population. V7's modest TRAIN
regression does not establish a mechanism for the separate retained-floor
failure.

### past_relation

> Dataset: **MKT-HIST-2** · continuous past relation · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, nine raw features, history128 · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v6.market-01 · initial | 40 | 0.660301 | 0.897445 | 100 |
| CPB-v7.market-01 · initial | 40 | 0.660301 | 0.897445 | 100 |
| CPB-v6.market-01 · final2048 | 40 | 0.735295 | 0.878565 | 100 |
| CPB-v7.market-01 · final2048 | 40 | 0.734780 | 0.881941 | 100 |

Final linear/neural MAE, in correlation units: V6
`0.041841 / 0.027108`; V7 `0.041867 / 0.026921`.
RMSE: V6 `0.052259 / 0.034974`; V7 `0.052310 / 0.034472`.
The final linear heads improve over initialization, while the final neural
heads remain below the initial neural R². These are descriptive results from
the same fitted TRAIN population, not independent temporal validation.

## Native blocks and saved readout contributions

The fixed spectral12 covariance is exactly unchanged. The learned blocks change
without a dramatic change in these aggregate participation summaries. The
uncertainty4 block has participation ratio close to1 in both final encoders;
that is a covariance description, not proof that it contains only one useful
task signal or that a wider bottleneck is required. Covariance traces depend on
coordinate scaling, and this diagnosis fits no PCA or replacement projection.

> Dataset: **MKT-HIST-2** · four past-statistic targets · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, history128 · saved natural-mask TRAIN112,
> final2048 coordinate statistics.

| Native block | V6 covariance trace | V7 covariance trace | V6 participation ratio | V7 participation ratio |
| --- | ---: | ---: | ---: | ---: |
| complete40 | 281.782392 | 285.779864 | 4.049737 | 4.044091 |
| learned content20 | 180.984223 | 185.722236 | 3.465338 | 3.450906 |
| fixed spectral12 | 1.529628 | 1.529628 | 6.855212 | 6.855212 |
| mean4 | 23.149092 | 22.014854 | 3.241882 | 3.160773 |
| uncertainty4 | 76.119448 | 76.513145 | 1.048589 | 1.052861 |

The original ridge weights and TRAIN normalizers reconstruct the saved linear
predictions by adding the four block contributions and intercept. They are
**contributions to one already fitted predictor**, not separately fitted block
heads, feature ablations or causal importance estimates. The following examples
reuse saved repetition2701; they are not averages of all three fits:

- **Volatility:** contribution-to-target Pearson correlations for
  content20/mean4/uncertainty4 are V6 `-0.630734 / 0.899628 / 0.980718` and
  V7 `-0.642515 / 0.881918 / 0.980270`. The uncertainty branch is exported and
  used by the existing readout even though its training objective is variance
  calibration. A negative content contribution within this multiblock fit is
  not evidence that the content block is harmful on its own.
- **Trend:** the mean4 contribution has correlation `0.927072` in V6 and
  `0.933359` in V7; content20 has `0.278440 / 0.280902`, and uncertainty4
  `0.066353 / 0.180112`. These values describe the saved joint predictor;
  they do not isolate a unique representation source.
- **Activity:** content20 and mean4 contributions both correlate strongly
  with targets: V6 `0.961189 / 0.930190`, V7 `0.958034 / 0.935124`.
  Uncertainty4 contributes with correlations `-0.446683 / -0.518094`.
  Its contribution standard deviation is `0.066236 / 0.075668` in target
  units, while mean4 is `0.283285 / 0.313260`. Signs and covariance matter;
  coordinate size or variance alone cannot rank the blocks.
- **Relation:** all four blocks contribute. Spectral12 contribution correlation
  is `0.571311 / 0.591784`; content20 `0.460497 / 0.457736`, mean4
  `0.442424 / 0.438402`, uncertainty4 `0.462230 / 0.455703`.
  No one block's removal or retraining is evaluated.

The report also retains early56/late56 descriptive readouts under the same
saved heads. They are not separate fits or independent cohorts. No block
correlation, temporal half or seed is selected to rescue a candidate.

## Fixed-query reconstruction:112 rows and first8 differ

The all112 query covers24192 target cells; its first-eight distribution archive
covers1728. The hierarchy averages eligible targets within channels and examples,
so a pooled cell mean is not substituted for the declared objective. Values
below are standardized-target metrics. Gaussian variance NLL can be negative;
the total is Huber mean plus variance NLL, not a generic positive error.

> Dataset: **MKT-HIST-2** · masked reconstruction · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, history128 · saved fixed-query TRAIN112 or
> its explicitly named first8 subset; no VALIDATION reconstruction diagnosis.

| Saved stage and population | MAE ↓ | Gaussian NLL ↓ | Huber(delta1) mean ↓ | Total objective ↓ |
| --- | ---: | ---: | ---: | ---: |
| V6/V7 initial · all112 | 0.664168 | 0.423815 | 0.323719 | 0.747534 |
| V6 final · all112 | 0.250173 | -0.400207 | 0.084376 | -0.315832 |
| V7 final · all112 | 0.241805 | -0.408827 | 0.082658 | -0.326169 |
| V6/V7 initial · first8 | 0.793746 | 0.594353 | 0.431625 | — |
| V6 final · first8 | 0.273179 | -0.361278 | 0.102840 | — |
| V7 final · first8 | 0.280297 | -0.337574 | 0.103262 | — |

V7 improves the full112 TRAIN MAE/NLL and Huber mean, yet worsens all three on
the first-eight subset. This limits any explanation built only from the detailed
eight-row tensors. Reconstruction quality also does not establish a better
embedding for a fixed task head or for TEMPO-5.

## Four32-tick bins:nonuniform error, no diagnosed age failure

The same first-eight fixed-query examples are divided into positions0–31,
32–63,64–95 and96–127, oldest to newest. Each bin recomputes the existing
eligible-example/channel/target hierarchy within that bin. Eligible example
counts and pooled target-cell counts differ substantially between bins. They
also cover different channels, examples, features and raw target distributions.
These bin means cannot be added to obtain the full-window loss.

> Dataset: **MKT-HIST-2** · masked temporal reconstruction · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, history128 · first8 saved TRAIN queries only,
> four fixed32-tick bins; identical query support across the two versions.

| Positions, oldest→newest | Eligible examples /8 | Pooled target cells | V6 final MAE ↓ | V7 final MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| 0–31 | 4 | 288 | 0.124791 | 0.125710 |
| 32–63 | 6 | 720 | 0.390876 | 0.394192 |
| 64–95 | 2 | 216 | 0.345439 | 0.354977 |
| 96–127 | 5 | 504 | 0.241876 | 0.251483 |

The oldest bin has the **lowest**, not the highest, error in both versions.
Middle bins are larger on this subset. V7's MAE is slightly worse in every bin,
consistent with its first-eight whole-window regression, while the full112
population improves. Initial bin MAEs are also uneven:
`0.688512 / 0.821522 / 1.022429 / 0.745583`.
This is not an age-controlled comparison of equally difficult targets.

The saved report includes observed rawF64 bin summaries, adjacent differences,
supports and per-channel/feature errors. They provide context without fitting
a temporal predictor. Neither a fitted probe nor an intervention shows that
the existing pool discards an older or middle segment. The bins do not establish
temporal capacity as the cause of any quality loss.

## Observed clipping across2048 TRAIN steps

The recorded factor formulas pass the producer checks. V6's global clip engages
on1326/2048 steps. V7's mean branch does not engage its cap on any step; its
uncertainty branch clips on1313/2048. The fixed512-step blocks show that observed
clipping is more frequent later in both trajectories.

> Dataset: **MKT-HIST-2** · encoder-training objective · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL TRAIN sampling · saved successful CUDA
> updates1–2048, B8; recorded norm/factor scalars, not reconstructed gradients.

| Successful updates | V6 global clips | V7 mean clips | V7 uncertainty clips |
| --- | ---: | ---: | ---: |
| 1–512 | 92 /512 | 0 /512 | 85 /512 |
| 513–1024 | 355 /512 | 0 /512 | 344 /512 |
| 1025–1536 | 445 /512 | 0 /512 | 438 /512 |
| 1537–2048 | 434 /512 | 0 /512 | 446 /512 |
| All1–2048 | 1326 /2048 | 0 /2048 | 1313 /2048 |

V6's recorded global norm has median1.123231 and maximum29.191668; its factor
has mean0.804397, median0.890288 and minimum0.034256. V7's mean-group norm has
median0.130581 and maximum0.461003, with factor1 throughout. Its uncertainty
norm has median1.120992 and maximum22.480474; factor mean0.798717, median0.892066
and minimum0.044483. These are measurements in the respective parameterizations
and trajectories, not a matched-state attribution of which branch caused V6's
global clipping. V6 does not save the branch split. No full gradient norm or
counterfactual V7 global factor is reconstructed, and loss-scalar ratios are
not interpreted as gradient dominance.

The actual update1 forward values match exactly across versions. Saved inputs
and mask/support match at updates1 and2048. Later model outputs, derivatives and
AdamW states can differ despite matched data. The two actual pre-step witnesses
verify the supported Huber/variance/total reductions, reporting NLL and recorded
clip association. They are not paired with the post-step final decoder for a
gradient calculation. V7's larger possible joint norm and future residual
coupling remain confounds of the controlled clipping comparison.

## Local decoder adjoints on the matching first8 fixed queries

Both versions apply Huber mean gradients to mean4 and detached-error variance
gradients to uncertainty4. The applied variance-to-mean4 and mean-to-uncertainty4
routes are zero. At initialization the relevant local native adjoints are zero
because the mean-residual and variance projections start at zero; that does not
say the entire encoder has no training gradient.

> Dataset: **MKT-HIST-2** · masked reconstruction derivatives · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL · first8 saved TRAIN fixed queries, each paired
> only with its own final2048 decoder; local mean4 basis within each stage.

| Final stage | Applied Huber mean4 L2 | Hypothetical mean-NLL L2 | Hypothetical MAE L2 | Mean-NLL / Huber cosine |
| --- | ---: | ---: | ---: | ---: |
| V6 | 0.00228239 | 0.01170520 | 0.00571558 | 0.734429 |
| V7 | 0.00225946 | 0.01448160 | 0.00608659 | 0.643296 |

The MAE/Huber cosines are0.767932 for V6 and0.741782 for V7. These compare
different local objectives **inside one stage's own mean4 basis**. Mean-NLL and
MAE are unapplied alternatives; reporting NLL is detached. These values neither
select a loss coefficient nor prove that an objective caused generalization
failure. Separately, the uncertainty4 adjoints are live in their own basis;
their magnitude is scale/decoder dependent. No cosine, orthogonality, conflict
or dominance is inferred across mean4/uncertainty4, versions or stages.

The scalar derivation promotes sanitized mean and target operands separately to
F64 before subtraction and uses the declared support/reduction and variance
link. It approximates the F32 intermediate chain rule, not autograd bit parity.
It omits parent32/upstream parameter Jacobians, full optimizer gradients and
trajectory replay. Finite-difference verification belongs to the pending
independent audit.

## Provisional next direction and preserved scopes

**A separately declared temporal-summary comparison is worth considering after
the audit; this diagnosis does not select its exact geometry or prove a pooling
cause.** Static source inspection shows causal contextual patch states followed
by one learned temporal softmax pool per channel and projection toward the
global export. These states already include causal context and positions; the
current summary is not claimed to be order-invariant.

A fixed four-bin summary could explicitly preserve coarse location before
compression and is a bounded structural hypothesis. Its reason is the current
single-summary architecture and the opportunity to test another representation,
not a demonstrated oldest-history deficit. Before any candidate quality, declare
how bins handle masks, empty support and cutoffs; the native coordinate budget,
learned parameter count, decoder route and head sizes; the retained objective
and clipping policy; and exact causal/CUDA/resume tests. Holding native40 would
help retain the existing readout budgets. A width or capacity increase would
add a separate confound that must be disclosed.

A learned multipool alternative is another possible structural comparison, but
adds queries/capacity and more choices. The saved evidence does not yet justify
choosing it over a fixed coarse summary, nor widening the uncertainty branch
from its participation ratio. Neither option licenses cap/rate/seed/coefficient,
budget, checkpoint or head rescue of the stopped V7 recipe. No V8 is selected,
declared or running by this report.

Preserve V7's measured neural-volatility record and its two TRAIN reconstruction
records, plus V6's timing/relation strengths and every V3/V4/V5 scope. The
[frozen V7 results](../CAUSAL_FRESH_V7_RESULTS.md) and
[review](../CAUSAL_FRESH_V7_REVIEW.md) retain the known quality outcome; no new
held-out component analysis is performed here. The [research ledger](../clean_causal_encoder_research_ledger.json)
and [final V7 evidence join](../causal_fresh_v7/final-establishment.json) remain
unchanged. CPB-v3.market-01 stays the clean baseline and sole dev encoder.

Any later quality declaration must bind the independently reviewed strongest
V3–V7 [future retention protocol](../clean_causal_encoder_search_goal_v7_retention.json),
SHA256 `c0a46527388032ea4f43dd9780ceedef486e5db7b4502ce37165d6e8616368b7`,
and its [application receipt](../causal_v7_retention_review/applied-future-protocol.json).
It protects the new V7 scopes in the same22-condition structure without
rewriting historical cards/pass counts. TRAIN NLL remains a preserved
descriptive record and adds no advancement gate. The clean encoder goal remains
active, with audit and protocol review before any next scientific execution.
