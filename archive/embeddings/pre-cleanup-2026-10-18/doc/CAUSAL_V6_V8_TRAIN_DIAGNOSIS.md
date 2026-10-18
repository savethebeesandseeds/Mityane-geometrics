<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Saved TRAIN V6/V8 diagnosis and bounded next design questions

**Independent arithmetic audit passed642801 checks with zero mismatches.** Its
256 scalar finite differences have maximum absolute error
1.3737544435343807e-10, and the V8 temporal-observation module checks30788
conditions. The [audit receipt](causal_v6_v8_train_diagnosis/independent-audit.json)
is80835 bytes, SHA256
`3d085bc6d7f6403266752f8663a270eea9f00a5a7fde02a98b65e5c936c92f23`.
These checks verify saved arithmetic and local derivatives, not full encoder
or optimizer gradients. No new encoder experiment, head fit or quality measurement was run.
The saved evidence shows that V8 improves TRAIN reconstruction MAE while both
encoders still support strong TRAIN volatility, trend and activity readouts.
V8's relation readout regresses modestly, and its final neural relation readout
is below its own initialization. Neither a missing TRAIN trend signal, a dead
pooling branch nor a conspicuous change in global clipping frequency explains
the known held-out instability by itself. The diagnosis establishes no cause
and selects no next encoder.

The [saved diagnosis](causal_v6_v8_train_diagnosis/diagnosis.json) is4617944 bytes,
SHA256 `e48e0dc05a72cd2e514ad8a5480106ba2eba5c49e7903f9565cf1b9d1c552a43`.
It binds the [prospective machine card](CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.json),
81588 bytes/SHA256
`49b22f1360593f2f0dd6d8ebb7fd8b0d4baa4749fe01ad8875fa79e5fe529829`,
its [human declaration](CAUSAL_V6_V8_TRAIN_DIAGNOSIS_CARD.md),52 permitted inputs,
33 preservation-only roles and the final94-checkpoint preservation receipt.
The [execution receipt](causal_v6_v8_train_diagnosis/execution-stage.json) records
one successful body invocation,2.315156206s; the internal arithmetic elapsed
field is1.793494563s. The [35-file exact mirror](causal_v6_v8_train_diagnosis/complete-mirror.json)
preserves the admitted inputs, source, logs and result. These are diagnosis
costs, not encoder training times.

The [independent audit execution](causal_v6_v8_train_diagnosis/independent-audit-execution.json)
ran once, wrapper1.847108343s/internal1.791172363s. Its
[19-file exact mirror](causal_v6_v8_train_diagnosis/independent-audit-mirror.json)
preserves the source, admitted inputs, logs and result. One root invocation
mistyped a mirror parameter and was rejected before that helper's body. The
[invocation failure](causal_v6_v8_train_diagnosis/audit-mirror-invocation-failure-001.json)
is preserved; correcting the parameter with the unchanged helper produced the
exact mirror without replaying the audit, producer, model or head. The
[provisional report](../output/engineering/causal-v6-v8-train-diagnosis-human-reports/draft-001/CAUSAL_V6_V8_TRAIN_DIAGNOSIS.md)
is preserved byte exact,27375 bytes/SHA256
`78abfc621777c1fa44eaf866860c7f1f8f5530ba16bf76c9c0d4c82f8eae43fd`,
with its [snapshot manifest](../output/engineering/causal-v6-v8-train-diagnosis-human-reports/draft-001/manifest.json).

All model, encoder, executable decoder, neural-head, fitting, training-update,
raw-cache numeric-read and held-out numeric-read counters are zero. Bounded
manual saved mean4-by-change-decoder contractions, variance affine/softplus-link
reconstruction and scalar derivative arithmetic are allowed. Zero executable
decoder calls does not mean zero manual scalar decoder calculations. No full
encoder, nonlinear parent decoder, library model or head was executed.

## What changed and what is being compared

V6 and V8 each completed one fresh2048-update, batch8 CUDA trajectory,
source_updates=0. Native40 is learned content20, fixed literal spectral12,
mean4 and independent uncertainty4. All294752 learned values started fresh;
3888 literal-prior values remain fixed. Constructor seed943001, draw order,
fresh TRAIN-only calibration, dataset, targets, sampling/mask streams, optimizer,
update budget and fixed D40 head recipes match. No historical learned weights
or fitted statistics were imported.

Both apply unit unweighted Huber(delta1) mean supervision plus unit Gaussian
variance calibration with the error detached. Reporting Gaussian NLL is detached
and supplies no mean-path training gradient. Both use one global norm2 cap1
and one AdamW. Future residual targets, the shared clip and optimizer history
continue to couple their trajectories.

V8 changes the temporal summary: four bins at original patch positions each
pool valid contextual states and supply eight disjoint rows of the existing
32-row export projection in both temporal paths. It preserves the forward
width and parameter shapes. The initial native coordinates20:40 pair exactly;
learned content0:20, initial means/variances and each version's own initial
fixed-head scores can differ. Whole-native40 initialization equality is not
claimed.

V8's76 registered learned tensors include the retained scalar pool-score bias,
a softmax gauge. The75 identifiable tensors are subject to the declared
data-gradient/movement requirement;294751 is their scalar capacity, not a claim
that every scalar received a nonzero gradient. The gauge remains registered in
AdamW and is observed separately. Its movement or weight decay is not learning
proof. Four export-row groups and four centered within-bin position contrasts
have their own finer saved evidence.

## Four evidence populations and their limits

1. **Natural-mask TRAIN112 per initial/final stage:** native40 values, four
   targets, all three saved linear/neural predictions, existing TRAIN normalizers
   and ridge coefficients. Covariance/participation, fixed-readout block
   contributions and early56/late56 descriptions use these rows. They are
   in-sample readout diagnostics, not newly fitted or held-out scores.
2. **Fixed-query TRAIN112 scalars, with first8 detailed distributions:** all112
   rows have MAE, Gaussian NLL, Huber mean, variance NLL, total objective and
   support. Only the first8 have detailed cell distributions, rawF64 observations,
   four original32-tick bins and stage-matched initial/final decoder data for
   local mean4/uncertainty4 adjoints. There are1728 supported cells in this
   first8 subset and24192 in all112. Four mixed aggregate JSON files are parsed
   whole but only their `train` subtree is numerically accessed.
3. **Both2048-update TRAIN scalar traces and two actual batches:** seven admitted
   columns per version describe recorded objective components, global norm and
   clip factor, overall and in four fixed512-update blocks. Actual updates1/2048
   retain pre-step states0/2047 for component, mask, support, timestamp, artifact
   and CSV association checks. No matching state2047 decoder was saved, so no
   actual-batch decoder adjoint or full-gradient replay is performed.
4. **V8-only temporal observations:**8192 update-major group rows,2048 gauge rows
   and twelve original F32 parameter/gradient arrays at each actual update1/2048.
   These support observed post-globalclip gradient/movement norms and counts.
   Each stored operand becomes Python F64 before subtraction/centering. There
   is no V6 subgroup/gauge witness, inferred branch split or counterfactual clip.

The112 chronological windows come from one genuine January2024 BTC/ETH/SOL
history, cutoff1705881599999. They are not112 independent market cohorts.
The unchanged heads are ridge penalty1 and tanh16/Adam0.01/100 updates, with
saved repetitions2701/2802/2903 and paired D40 seeds6049515622977958931,
8262941649267994858,2095371932330426401. Market heads have164/724 fitted
parameters. Three fits on one encoder/split measure head variation, not three
encoder retrainings. No PCA or replacement projection follows the encoder.

Complexity4/5 below is an ordinal signal-recipe label, not accuracy or observed
task difficulty. R²=0 is the scoring population's constant mean; negative R² is
worse. All task tables below reuse the saved TRAIN112 predictions and show
means of the three saved fits, with100% declared-row coverage.

## Saved TRAIN task readouts

### past_volatility

> Dataset: **MKT-HIST-2** · continuous past volatility · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, nine raw features, history128 · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v6.market-01 · initial | 40 | 0.856612 | 0.921619 | 100 |
| CPB-v8.market-01 · initial | 40 | 0.880334 | 0.941478 | 100 |
| CPB-v6.market-01 · final2048 | 40 | 0.991300 | 0.961221 | 100 |
| CPB-v8.market-01 · final2048 | 40 | 0.991422 | 0.963332 | 100 |

Final linear/neural MAE, in log-return units: V6
`3.783827e-5 / 7.310413e-5`; V8 `3.701699e-5 / 7.412630e-5`.
RMSE: V6 `4.694765e-5 / 9.884953e-5`; V8 `4.661844e-5 / 9.618382e-5`.
V8's slightly higher TRAIN neural R² coexists with slightly worse neural MAE;
the metrics weight errors differently. These small in-sample differences do
not establish the cause of its preserved neural-volatility quality record.

### past_trend

> Dataset: **MKT-HIST-2** · continuous past trend · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, nine raw features, history128 · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v6.market-01 · initial | 40 | 0.675497 | 0.954260 | 100 |
| CPB-v8.market-01 · initial | 40 | 0.725499 | 0.960282 | 100 |
| CPB-v6.market-01 · final2048 | 40 | 0.919068 | 0.950138 | 100 |
| CPB-v8.market-01 · final2048 | 40 | 0.913545 | 0.952652 | 100 |

Final linear/neural MAE, in log-return/tick: V6
`1.829895e-5 / 1.374199e-5`; V8 `1.885914e-5 / 1.313960e-5`.
RMSE: V6 `2.375324e-5 / 1.862766e-5`; V8 `2.455025e-5 / 1.809745e-5`.
Both final encoders support strong TRAIN trend readouts. V8's known head
robustness failure cannot be reduced to an inability to represent or fit TRAIN
trend. Its final neural TRAIN R² is below its own initial value, but a good
in-sample initial readout is not a guarantee of held-out retention either.

### past_activity

> Dataset: **MKT-HIST-2** · continuous past activity · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, nine raw features, history128 · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v6.market-01 · initial | 40 | 0.952488 | 0.950540 | 100 |
| CPB-v8.market-01 · initial | 40 | 0.920141 | 0.960265 | 100 |
| CPB-v6.market-01 · final2048 | 40 | 0.973945 | 0.971945 | 100 |
| CPB-v8.market-01 · final2048 | 40 | 0.975460 | 0.973636 | 100 |

Final linear/neural MAE, in log1p(base-volume) units: V6
`0.078173 / 0.082004`; V8 `0.075710 / 0.078055`.
RMSE: V6 `0.100230 / 0.103908`; V8 `0.097272 / 0.100727`.
Both improve over their own initial linear and neural scores. The V8 gains on
this TRAIN population remain separate from the previously measured retention
conditions.

### past_relation

> Dataset: **MKT-HIST-2** · continuous past relation · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, nine raw features, history128 · saved TRAIN112.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| CPB-v6.market-01 · initial | 40 | 0.660301 | 0.897445 | 100 |
| CPB-v8.market-01 · initial | 40 | 0.658908 | 0.919836 | 100 |
| CPB-v6.market-01 · final2048 | 40 | 0.735295 | 0.878565 | 100 |
| CPB-v8.market-01 · final2048 | 40 | 0.713898 | 0.875959 | 100 |

Final linear/neural MAE, in correlation units: V6
`0.041841 / 0.027108`; V8 `0.043995 / 0.027893`.
RMSE: V6 `0.052259 / 0.034974`; V8 `0.054330 / 0.035527`.
The final linear scores exceed initialization, but both neural scores fall
below their own initial readouts. V8 is modestly worse than V6 at final2048.
This is compatible with an objective/readout tradeoff; it does not isolate a
specific architecture defect or explain the much less stable held-out fits.

## Native blocks and existing readout contributions

The fixed spectral12 values and covariance stay exact. Learned content20 changes
at initialization because temporal pooling changes, and changes again during
training. Final covariance participation remains a descriptive population
quantity. It is sensitive to coordinate scaling and is neither a useful-signal
count nor evidence that another width or PCA is required.

> Dataset: **MKT-HIST-2** · four past-statistic targets · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, history128 · final natural-mask TRAIN112.

| Native block | V6 covariance trace | V8 covariance trace | V6 participation ratio | V8 participation ratio |
| --- | ---: | ---: | ---: | ---: |
| complete40 | 281.782392 | 291.621812 | 4.049737 | 3.961920 |
| learned content20 | 180.984223 | 190.674623 | 3.465338 | 3.234147 |
| fixed spectral12 | 1.529628 | 1.529628 | 6.855212 | 6.855212 |
| mean4 | 23.149092 | 22.278162 | 3.241882 | 3.213111 |
| uncertainty4 | 76.119448 | 77.139399 | 1.048589 | 1.053278 |

The saved ridge coefficients and their original TRAIN normalizers reconstruct
the existing predictions from four block contributions and an intercept. These
are contributions within one already fitted multiblock predictor, not separately
fitted block heads, feature-removal tests or causal importance estimates. The
next table uses repetition2701 only; it is not a three-fit average.

> Dataset: **MKT-HIST-2** · saved four-task ridge decomposition · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, history128 · final TRAIN112, repetition2701;
> entries are contribution-to-target Pearson correlations, not new head scores.

| Encoder/task | Content20 | Fixed spectral12 | Mean4 | Uncertainty4 |
| --- | ---: | ---: | ---: | ---: |
| V6 volatility | -0.630734 | 0.187147 | 0.899628 | 0.980718 |
| V8 volatility | -0.650608 | 0.137253 | 0.883371 | 0.980854 |
| V6 trend | 0.278440 | 0.030229 | 0.927072 | 0.066353 |
| V8 trend | 0.262472 | 0.035531 | 0.926676 | 0.124257 |
| V6 activity | 0.961189 | 0.131444 | 0.930190 | -0.446683 |
| V8 activity | 0.955806 | 0.164911 | 0.931026 | -0.476292 |
| V6 relation | 0.460497 | 0.571311 | 0.442424 | 0.462230 |
| V8 relation | 0.499094 | 0.588420 | 0.445609 | 0.406453 |

The existing volatility readout uses the exported uncertainty branch strongly,
while the mean4 contribution remains closely aligned with TRAIN trend in both
versions. Relation uses multiple correlated blocks; its V8 uncertainty
contribution correlation is lower even though content/spectral correlations are
higher. A negative block contribution correlation can be part of a useful joint
predictor and does not establish that removing that block helps. The report also
preserves all three repetitions, pairwise contribution covariances and fixed
chronological halves; no example here was chosen to refit or alter a head.

For context within the saved TRAIN population, repetition2701's early/late
relation neural R² is V6 `0.880595 / 0.808655` and V8 `0.883959 / 0.831460`.
These halves share the original whole-TRAIN head and normalizer; they are not
new chronological validation splits. They show why one aggregate TRAIN score
cannot locate or diagnose a later distribution shift.

## Reconstruction, calibration and the four fixed bins

V8's MAE improves on both all112 queries and the first8 subset. Its Gaussian NLL
becomes less negative in both populations. The mean/variance calibration tradeoff
therefore remains visible even with unit Huber mean supervision; a lower MAE
does not establish improved likelihood or a better downstream embedding.

> Dataset: **MKT-HIST-2** · standardized masked reconstruction · **complexity4/5** ·
> genuine January2024 BTC/ETH/SOL, history128 · fixed TRAIN112 query,24192 targets.

| Encoder/stage | MAE ↓ | Gaussian NLL ↓ | Mean Huber1 ↓ | Composite objective ↓ |
| --- | ---: | ---: | ---: | ---: |
| V6 initial | 0.664168 | 0.423815 | 0.323719 | 0.747534 |
| V8 initial | 0.664361 | 0.423883 | 0.323808 | 0.747692 |
| V6 final2048 | 0.250173 | -0.400207 | 0.084376 | -0.315832 |
| V8 final2048 | 0.247146 | -0.381215 | 0.083864 | -0.297351 |

NLL excludes the constant Gaussian term, and variance NLL or the total composite
may be negative. Variance NLL and the detached reporting NLL have equal numeric
values here; their gradient roles differ. The full112 early/late MAE is V6
`0.266752 / 0.233593` and V8 `0.256481 / 0.237811`. V8's improvement is not
uniform across those fixed TRAIN halves. No held-out query value is used here.

The first8 final MAE is V6 `0.273179` and V8 `0.266092`; NLL is
`-0.361278 / -0.322525`, and Huber1 is `0.102840 / 0.103102`.
The four bins below partition original history positions, with unequal target
support and different eligible example/channel populations. Their independently
normalized losses are not additive pieces of the full objective and are not an
age-controlled ablation.

> Dataset: **MKT-HIST-2** · fixed-query reconstruction by original time bin ·
> **complexity4/5** · genuine January2024 BTC/ETH/SOL, history128 · first8 TRAIN
> queries, final2048; eligible-example/channel/cell hierarchy within each bin.

| Original ticks | Supported cells | Eligible examples | V6 MAE ↓ | V8 MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| 0:32 · oldest | 288 | 4 | 0.124791 | 0.093975 |
| 32:64 | 720 | 6 | 0.390876 | 0.402718 |
| 64:96 | 216 | 2 | 0.345439 | 0.357328 |
| 96:128 · most recent | 504 | 5 | 0.241876 | 0.223518 |

V8 improves the two end bins and worsens the two middle bins on this subset.
The oldest bin remains the easiest by these observed errors. This does not
prove that history age caused the differences, that four pools solved temporal
capacity, or that middle history should receive a new weight or more rows.
RawF64 summaries, adjacent-difference observations and all27 channel-feature
panels remain in the saved JSON, without new labels or fitted temporal tests.

The manual mean4 residual allocation contracts the matching saved linear
change-decoder matrix with mean4. Subtracting that F64 reconstruction from the
stored total mean approximates F32 parent-plus-residual addition; it does not
execute the nonlinear parent decoder or separate content20 from spectral12.
Its quantiles/correlations describe this saved addition, not a counterfactual
parent-only reconstruction predictor.

## Recorded clipping and two actual batches

The producer checks all2048 recorded global clip-factor formulas per version,
`min(1,1/(observed_norm+1e-6))`, within the declared tolerance. It does not
independently reconstruct the full norm or a missing branch split.

> Dataset: **MKT-HIST-2** · observed TRAIN optimization scalars · **complexity4/5** ·
> genuine TRAIN anchors, B8 CUDA,2048 successful updates; fixed512-update blocks.

| Update scope | V6 clipped updates | V8 clipped updates | V6 mean factor | V8 mean factor |
| --- | ---: | ---: | ---: | ---: |
| all1:2048 | 1326 | 1333 | 0.804397 | 0.799545 |
| 1:512 | 92 | 94 | 0.945214 | 0.941138 |
| 513:1024 | 355 | 365 | 0.820481 | 0.811489 |
| 1025:1536 | 445 | 437 | 0.733855 | 0.724570 |
| 1537:2048 | 434 | 437 | 0.718038 | 0.720982 |

Recorded overall clipping frequency is `0.647461 / 0.650879`. Mean pre-clip
norms are `1.495155 / 1.534416`; medians are `1.123231 / 1.116533`.
Both trajectories clip much more often after the first512 updates. These
similar aggregate observations do not identify clipping as the cause of the
quality difference or justify a cap, rate, coefficient or seed rescue. Scalar
loss ratios do not identify parameter-gradient dominance.

> Dataset: **MKT-HIST-2** · actual pre-step TRAIN component witnesses ·
> **complexity4/5** · genuine TRAIN anchors, B8; only updates1/2048, no extra
> forwards or reconstructed pre-step decoder gradients.

| Version/update | Pre-step state | Mean Huber1 | Variance NLL | Observed clip factor |
| --- | ---: | ---: | ---: | ---: |
| V6 update1 | 0 | 0.470009 | 0.802452 | 1.000000 |
| V8 update1 | 0 | 0.470059 | 0.802425 | 1.000000 |
| V6 update2048 | 2047 | 0.057822 | -0.603823 | 0.723349 |
| V8 update2048 | 2047 | 0.057316 | -0.609135 | 0.715249 |

Input rows, timestamps, targets and support pair at both saved updates, and
the first fresh native20:40 coordinates pair exactly. Initial mean/variance
equality is deliberately not required. Update2048 used state2047, whereas the
final checkpoint is state2048. This report never substitutes final decoder
coefficients for those missing pre-step weights or extrapolates from these
two batches to unsaved tensor gradients at other updates.

## Stage-local mean4 and uncertainty4 adjoints

The next quantities are first8 fixed-query, stage-matched saved-decoder local
F64 derivatives. They omit parent32 and all encoder-parameter Jacobians,
optimizer moments and clipping. Huber1 is the applied mean route in both
versions; Gaussian/MAE mean derivatives are diagnostic alternatives, and the
reporting NLL is detached. The uncertainty route is separate and live. No
cross-model/stage/basis cosine or magnitude-dominance claim is made.

> Dataset: **MKT-HIST-2** · local saved-decoder adjoints · **complexity4/5** ·
> genuine history128, first8 fixed TRAIN queries, final2048; norms in each
> model's own mean4 coordinate basis, not full parameter-gradient norms.

| Encoder | Applied mean4 Huber L2 | Diagnostic MAE L2 | Diagnostic Gaussian mean L2 | Same-basis Huber/MAE cosine |
| --- | ---: | ---: | ---: | ---: |
| V6 | 0.002282 | 0.005716 | 0.011705 | 0.767932 |
| V8 | 0.002337 | 0.005700 | 0.014764 | 0.796275 |

Within V8's mean4 basis the Huber/diagnostic-Gaussian cosine is0.619112,
compared with0.734429 within V6's own basis. Those separately computed
descriptions do not identify a better objective or prove gradient conflict.
Initial local mean4/uncertainty4 derivatives are zero because their decoder
heads start at zero, not because the native projections are absent.

The applied variance-to-mean4 and mean-to-uncertainty4 direct adjoints remain
zero. V6's uncertainty4 local L2 is0.030983 and V8's0.050237, but their independent
bases/scales prevent using those magnitudes to declare variance dominance.
Likewise the saved variance-head parameter adjoints are local scalar checks,
not a replay of the full training gradient. The independent audit verifies all
256 declared scalar finite-difference checks within the frozen tolerances.

## V8-only temporal learning observations

All four original-position groups have recorded nonzero export weight/bias
data-gradient events, centered position-gradient events and movement events
on all2048 updates. The main prospective requirement was at least one event
per group plus final change, not all2048 events or every scalar coordinate.
The diagnosis reuses those observations; it adds no model calls.

> Dataset: **MKT-HIST-2** · observed temporal-parameter TRAIN updates ·
> **complexity4/5** · genuine TRAIN anchors, one2048/B8 CUDA trajectory;
> post-globalclip group records, no V6 group counterfactual.

| Original patch positions | Export rows | Weight/bias nonzero-gradient steps | Centered position gradient/movement steps | Saved final centered position change L2 |
| --- | --- | ---: | ---: | ---: |
| 0:4 | 0:8 | 2048 / 2048 | 2048 / 2048 | 0.164445 |
| 4:8 | 8:16 | 2048 / 2048 | 2048 / 2048 | 0.194650 |
| 8:12 | 16:24 | 2048 / 2048 | 2048 / 2048 | 0.177038 |
| 12:16 | 24:32 | 2048 / 2048 | 2048 / 2048 | 0.285409 |

Export weight/bias movement events are also2048 per group, and all saved final
weight/bias-change flags are true. Movement includes AdamW weight decay; the
separate nonzero data-gradient observations are necessary and do not imply
that every scalar learned. The larger final centered movement of one group
does not establish its task importance or justify reallocating row capacity.

The two twelve-array witnesses permit manual F64 recomputation of the recorded
post-clip norms, nonzero/movement counts and gauge associations at updates1/2048
only. They contain no nonlinear encoder replay or missing V6 subgroup evidence.
The pool-score scalar gauge has2048 defined-gradient records,2037 numerically
nonzero recorded gradients, maximum absolute gradient1.5993464330676943e-9 and
final movement0.0005295611917972565. These tiny finite-precision observations
and movement remain explicitly excluded from data-learning/identifiability
proof. No gauge threshold, bin choice or cap is selected from them.

## Bounded interpretation and next decision

The independently verified saved TRAIN arithmetic supports four limited
conclusions; the audit does not turn these descriptions into causal findings:

- V8 did not lose the ability to fit TRAIN trend. Relation shows a modest
  regression and a stronger initial neural readout than its final readout, but
  this does not explain the separate held-out head instability.
- Reconstruction MAE, Huber and likelihood do not move together on every
  population. All112 and first8 have different Huber behavior, and the four
  bins have unequal support. No reconstruction scalar is an overall encoder
  score or a causal account of downstream task changes.
- The four pooling groups receive actual recorded gradients and move. This
  argues against a disconnected or completely unused group explanation; it
  does not prove good temporal representations, age invariance or retention.
- Both trajectories retain frequent, broadly similar global clipping. No
  branch norm is available, and neither objective scalar ratios nor local
  latent adjoint norms can reconstruct optimizer coupling or dominance.

Two different hypotheses now merit a prospective design discussion:

- **A shared global plus four-bin temporal summary.** A possible controlled
  operator would combine a global summary and V8's four-bin summary50/50 using
  the same existing export weights, registration order, parameter count and
  native width. The fixed combination would need declaration before any test;
  it is not a sweep over mixing weights or bins. This would examine a tradeoff
  between globally shared aggregation and a hard allocation of eight export
  rows per bin. The present evidence does not prove that V8 lacks global
  context: each bin pools contextual states that already contain causal
  history, V8 retains the downstream99→64→32 global mixing network, and TRAIN
  trend remains strong. To isolate the aggregation change, a prospective hybrid
  should retain V8's exact zero slices for empty bins instead of adding a second
  change that fills those slices from global states. A global softmax denominator
  can distinguish three relative per-bin constant score offsets that cancel
  within separate bin denominators; the common pool-score bias remains a
  softmax gauge. Thus the identifiable parameter directions and gradient routing
  may change even with identical registration and parameter count. These are
  design implications, not measured benefits or evidence that a gauge caused
  the observed task instability.
- **A representation/readout-stability objective.** The stronger initial
  neural relation readout than final readout, changed learned-block covariance
  and differences across saved fits leave open whether reconstruction training
  preserves the geometry needed by the unchanged small heads. A separately
  declared representation-consistency objective could test that idea without
  importing task labels or changing the heads. This diagnosis does not specify
  such an objective or demonstrate that it would help: existing TRAIN head
  normalization already handles marginal scales, covariance participation is
  not a robustness measure, and any additional views/loss would introduce
  objective and compute confounds that must be prospectively controlled.

The global-plus-bin operator is a more narrowly specified architecture question;
the objective alternative currently requires more design decisions. Neither is
selected by this report. There is no ablation, full Jacobian, task-specific
encoder training or held-out explanation that identifies a cause. Another
cap/rate/seed/head adjustment to the stopped V8 recipe is not justified here.
No V9 source, card, candidate, projection, new label or objective is declared.

Preserve V8's neural-volatility and intact market-scaler TEMPO-5 neural records,
V6's timing/linear-relation strengths, V7's TRAIN reconstruction records and
every V3–V5 scoped leader/checkpoint/history. Their measured quality remains in
the [V8 results](CAUSAL_FRESH_V8_RESULTS.md), [V8 detail](CAUSAL_FRESH_V8_DETAIL.md),
[V8 review](CAUSAL_FRESH_V8_REVIEW.md), [clean ledger](clean_causal_encoder_research_ledger.json)
and separate [V8 trust join](causal_fresh_v8/final-establishment.json). This
TRAIN-only diagnosis does not revise any frozen quality result or pass count.
Any later named candidate must use the reviewed strongest-V3–V8
[future22-condition protocol](clean_causal_encoder_search_goal_v8_retention.json),
SHA256 `e7920442fbc5528cbe7e67f71ada65101974a6a28ad4d39bc57dea461c477699`,
with its [review](causal_v8_retention_review/independent-review.json) and
[application admission](causal_v8_retention_review/applied-future-protocol.json).
It has no TRAIN-NLL gate. V3 remains the sole dev encoder and clean baseline;
the search goal remains active.
