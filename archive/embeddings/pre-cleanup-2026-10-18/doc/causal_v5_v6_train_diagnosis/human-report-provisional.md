<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE. -->
# Provisional saved TRAIN V5/V6 interpretation

**Independent audit pending. Do not use this draft to declare a next encoder.**
No new experiment or quality measurement. The once-executed saved-data diagnosis
passes its producer checks, with zero model/decoder/neural forwards, fits,
training updates, raw-cache numeric reads or held-out numeric fields accessed.
Its subprocess took1.756203923s; the report's internal elapsed field is
1.543878274s. These are scalar-diagnosis costs, not encoder training costs.

The bounded finding is that **V6 does encode TRAIN volatility and activity well**:
both fixed heads improve over V5 in-sample. TRAIN trend weakens; neural relation
improves over V5 but remains below the common initialization. This does not
explain the known quality-retention failures or establish generalization.
The saved V6 trajectory also shows frequent shared clipping, which makes
separating the mean/uncertainty clipping geometry a reasonable controlled
question. It does not establish that uncertainty caused the clipping or that
group clipping would improve quality. The ranking below remains provisional.

The [prospective card](../CAUSAL_V5_V6_TRAIN_DIAGNOSIS_CARD.md) binds45 inputs,
31 preservation pins and the final V6 preservation receipt; all77 pins match
before/after. The [saved diagnosis](diagnosis.json) is2466550 bytes, SHA256
`ca992a2a39dbcac5dd9c4a1f401ba4d2a779175a5cea91e9b33508f50e8e08dd`.
The complete eight-file source/card/preservation capsule stays exact. Its
preceding seven-file draft is preserved before the prospective extension from
two CSV rows to all2048 rows; no diagnosis ran under the superseded scope.

## Population and evidence boundaries

Both saved final encoders had one fresh2048-update, batch8 CUDA:0 trajectory,
source_updates=0, native40 and the same fresh constructor draw order, TRAIN-only
calibration policy and fixed D40 heads. Native40 is learned content20, literal
spectral12, mean4 and independent uncertainty4. V5 applies joint Gaussian NLL;
V6 applies unweighted Huber(delta1) mean plus detached-error Gaussian variance
calibration, unit coefficients, with unchanged forward/reporting distributions.
Shared global clip1/AdamW and residual-dependent future variance targets remain
couplings. V6's detached reporting NLL is never an applied mean gradient.

The head recipe stays ridge penalty1 and tanh16/Adam0.01/100 updates, with three
saved repetitions2701/2802/2903 and the frozen D40 seeds. Predictions are reused,
not refitted. There is one encoder trajectory per version, not three. The112
chronological probe windows come from the same January2024 BTC/ETH/SOL history;
neither window count nor head repetitions establishes independent market cohorts.
Early56/late56 are fixed descriptive halves, never independently fitted heads.

Four scopes must remain separate:

- **112 TRAIN natural-mask rows per stage:** four-task saved head predictions,
  original ridge coefficients, coordinate/block statistics and contributions.
- **112 TRAIN fixed-query scalar rows:** MAE/NLL/support in both versions;
  only V6 supplies full112 Huber/variance/total rows. V5 Huber is unavailable
  for this population and is never inferred from aggregate MAE/NLL.
- **First8 TRAIN fixed-query distributions:**1728 supported target cells per
  stage, with stage-matched decoder coefficients for bounded local adjoints.
  V5 Huber is hypothetical here; V6 mean-NLL/MAE adjoints are unapplied
  diagnostic alternatives. These cells do not represent all112 automatically.
- **V6 actual batches and trajectory scalars:** tensor witnesses exist only
  for pre-step updates1 and2048. The last is state2047, whereas the final
  decoder is state2048; no actual-batch decoder adjoints are reconstructed.
  Seven declared CSV scalar columns cover all2048 successful updates. No
  V5 actual-batch replay or branch/full-gradient reconstruction occurs.

Four aggregate query JSON files contain both splits; only their train subtrees
are accessed. No held-out tensor diagnosis, encoder/head replay, raw-cache fit,
new predictor or quality test enters this report. The original initial/final
evidence, failure histories and all V3/V4/V5/V6 scoped strengths remain preserved.

## Saved TRAIN task readouts

The initial V5/V6 native40 values and saved scores agree exactly. The common
initial row below represents both independently saved initial stages. Neural
columns average the three saved fits, not three independently trained encoders.

### past_volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-history retention · **complexity4/5** · saved TRAIN112 natural-mask rows, complete coverage.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v5.init / CPB-v6.init |40|0.8566119371|0.9216185622|100|
| CPB-v5.market-01 |40|0.9871272299|0.9596348584|100|
| CPB-v6.market-01 |40|0.9913001038|0.9612206596|100|

Physical errors, MAE/RMSE in the target's return-RMS units:

- Common initial: linear0.0001307951/0.0001905958; neural0.0000971948/0.0001408038.
- V5 final: linear0.0000450707/0.0000571075; neural0.0000735651/0.0001009875.
- V6 final: linear0.0000378383/0.0000469476; neural0.0000731041/0.0000988495.

Final neural R² repetitions2701/2802/2903: V5
0.9570026473/0.9564337810/0.9654681471; V6
0.9530359845/0.9643915039/0.9662344905. V6's fixed ridge is strong in both
early56/late56:0.9909471414/0.9901373815. This is positive TRAIN retention,
not evidence that the older held-out volatility condition has been solved.

### past_trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-history retention · **complexity4/5** · saved TRAIN112 natural-mask rows, complete coverage.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v5.init / CPB-v6.init |40|0.6754968042|0.9542596202|100|
| CPB-v5.market-01 |40|0.9410406055|0.9595791016|100|
| CPB-v6.market-01 |40|0.9190675173|0.9501375572|100|

Physical errors, MAE/RMSE in log-price slope per bar:

- Common initial: linear3.81709369e-5/4.75631925e-5; neural1.35041836e-5/1.78355672e-5.
- V5 final: linear1.56934265e-5/2.02739334e-5; neural1.23072004e-5/1.67039605e-5.
- V6 final: linear1.82989463e-5/2.37532445e-5; neural1.37419905e-5/1.86276627e-5.

Final neural repetitions: V5 0.9498370816/0.9595862808/0.9693139426;
V6 0.9452737931/0.9555250573/0.9496138213. V6 ridge early56/late56 is
0.9357328617/0.8876880649, versus V5 0.9514553814/0.9206689667.
Trend therefore has a TRAIN-visible loss too, particularly in the later half;
the unchanged readout does not identify its upstream cause.

### past_activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-history retention · **complexity4/5** · saved TRAIN112 natural-mask rows, complete coverage.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v5.init / CPB-v6.init |40|0.9524883089|0.9505397794|100|
| CPB-v5.market-01 |40|0.9708782664|0.9657628812|100|
| CPB-v6.market-01 |40|0.9739446571|0.9719447181|100|

Physical errors, MAE/RMSE in the declared log-volume activity units:

- Common initial: linear0.1027095913/0.1353472107; neural0.1043431315/0.1376103211.
- V5 final: linear0.0813469913/0.1059638375; neural0.0882871943/0.1143124922.
- V6 final: linear0.0781732695/0.1002299380; neural0.0820039873/0.1039077002.

Final neural repetitions: V5 0.9682656044/0.9562124155/0.9728106237;
V6 0.9739830491/0.9684746438/0.9733764615. Ridge early56/late56 is
0.9655173195/0.9768769820 for V6 and0.9670081068/0.9701973244 for V5.
This does not support a blanket lack of TRAIN activity information in V6.

### past_relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute-history retention · **complexity4/5** · saved TRAIN112 natural-mask rows, complete coverage.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v5.init / CPB-v6.init |40|0.6603006899|0.8974450571|100|
| CPB-v5.market-01 |40|0.7360101104|0.8617764110|100|
| CPB-v6.market-01 |40|0.7352945962|0.8785649076|100|

Physical errors, MAE/RMSE in correlation units:

- Common initial: linear0.0477047028/0.0592010926; neural0.0246040414/0.0323663854.
- V5 final: linear0.0426297040/0.0521886469; neural0.0279002424/0.0377019682.
- V6 final: linear0.0418414545/0.0522593247; neural0.0271081125/0.0349740768.

Final neural repetitions: V5 0.8480157364/0.8833947329/0.8539187637;
V6 0.8507162320/0.9278494819/0.8571290090. The common initial neural mean
remains higher than either trained mean. V6 ridge early56/late56 is
0.7820395451/0.6666036488. Different head fits and chronological halves show
variation within this same TRAIN cohort, not independent uncertainty estimates.

## Existing ridge block contributions

Every saved ridge prediction is reconstructed from its original TRAIN feature
and target normalization, intercept and coefficients. The following examples
use the predeclared first repetition2701; all three are retained in the JSON.
These are scalar contributions to an already fitted predictor in target units,
not latent gradients, block-ablation scores or newly fitted submodels.

For volatility, uncertainty4's contribution has target Pearson0.9853 in V5
and0.9807 in V6, with population standard deviations0.00047075/0.00040829.
The mean4 contribution grows from0.00007105 to0.00015027 and its target
Pearson increases0.7653→0.8996. TRAIN volatility remains readable across a
redistributed combination of blocks; neither the four uncertainty dimensions
nor the smaller content20 raw scale establishes lost volatility information.

For activity, content20 contributes standard deviation0.45723/0.37318 with
target Pearson0.9677/0.9612; mean4 contributes0.17934/0.28328 with
Pearson0.9013/0.9302. Uncertainty4 contributes negatively correlated components
in both versions,−0.4156/−0.4467 to target. The combined ridge improves.
Negative component correlations do not prove destructive branch competition
or that deleting uncertainty coordinates would help.

For trend, mean4 remains the largest ridge component by target-unit spread,
7.7579e-5/7.5647e-5, with target Pearson0.9475/0.9271. This is consistent
with a weaker TRAIN trend readout, without identifying whether the projection,
mean supervision or upstream representation is responsible. For relation,
multiple blocks contribute; mean4/uncertainty4 scalar-contribution correlation
changes0.2581→0.8595 while total linear R² barely changes. Such scalar
contribution correlation is defined; a cosine between their independent
native-coordinate gradient bases would not be.

## Coordinate geometry and bounded query evidence

Dataset: **MKT-HIST-2** · **complexity4/5** ·112 saved TRAIN natural-mask rows; descriptive population covariance, no PCA/export or fit.

| Native block | V5 final covariance trace | V5 participation ratio | V6 final covariance trace | V6 participation ratio |
| --- | --- | --- | --- | --- |
| learned20 |313.340083|3.593199|180.984223|3.465338|
| fixed spectral12 |1.529628|6.855212|1.529628|6.855212|
| mean4 |9.608697|3.078266|23.149092|3.241882|
| uncertainty4 |87.547523|1.033541|76.119448|1.048589|
| complete40 |412.025931|4.142048|281.782392|4.049737|

The fixed spectral block is unchanged. Raw covariance traces change with
coordinate scale; the fixed heads separately use their original TRAIN
normalization. Low participation ratio is not a test of useless coordinates or
a causal explanation of quality. Uncertainty4 is concentrated near one
covariance direction in both versions, so this is not a V6-specific discovery
that alone justifies adding uncertainty capacity or a decorrelation penalty.

Dataset: **MKT-HIST-2** · **complexity4/5** ·112 TRAIN fixed-query scalar rows,24192 target cells per stage; separate from natural-mask head features.

| Saved stage | TRAIN query MAE | Gaussian NLL | Huber1 component | Composite total |
| --- | --- | --- | --- | --- |
| V5/V6 initial |0.6641678566|0.4238153535|V6:0.3237186646|V6:0.7475340181|
| V5 final |0.2544639357|−0.3243773732|Unavailable at112|Not V5's objective|
| V6 final |0.2501725785|−0.4002073617|0.0843757275|−0.3158316342|

V6's variance component equals the numerical reporting NLL; their gradient
contracts differ. Negative Gaussian/composite values are permitted. V6 TRAIN
MAE early56/late56 is0.2667522268/0.2335929302 versus V5
0.2793207650/0.2296071064; improvement is not uniform across these two halves.
There is no new TRAIN NLL gate.

The first8 query population must not stand in for all112: V6's first8 MAE is
0.2731787120 versus V5's0.2763843911, whereas first8 NLL is−0.3612780895
versus−0.3937406086. That NLL direction differs from the full112 aggregate.
V5's first8 hypothetical Huber is0.1090573411; V6's applied mean component
is0.1028396944. This is the only population permitting that Huber comparison.

On the saved final first8 mean4 basis, Gaussian mean-NLL versus Huber1 local
cosine is0.7080088179 in V5 and0.7344293484 in V6. Norms are respectively
0.0121094947/0.0026042801 and0.0117051971/0.0022823888. They show different
weighting/direction within each single basis, not wholesale opposition.
V5 applies mean-NLL; V6 applies Huber, and its reporting-NLL derivative is
hypothetical. Initial mean4 and uncertainty4 adjoints are zero because the
corresponding saved decoder heads start at zero; this is not a claim that
all initial encoder parameters lack gradients. The live uncertainty4 adjoints
are retained separately, with scale-dependent norms. No comparison between
mean4 and uncertainty4 norms establishes gradient dominance or conflict.

Saved linear residual allocation also changes. As an explicitly local example,
channel0/feature4's supported residual quartiles are−0.08360/−0.03283/0.02433
in V5 and−0.25793/−0.09371/0.03797 in V6. The parent remainder is approximate
because total-minus-rebuilt-residual uses F64 algebra over a rounded F32 sum.
All27 channel-feature summaries remain in JSON. This demonstrates different
allocation, not that the residual should be removed or separately supervised.
No nonlinear parent decoder or content20/spectral12 decoder Jacobian is run.

## Actual V6 batches and all2048 scalar clipping observations

Dataset: **MKT-HIST-2** · **complexity4/5** · V6 actual two TRAIN batch8 pre-step witnesses; different sampled batches, not fixed-query stages.

| Successful update | Mean Huber1 | Variance/report NLL | Total objective | Observed norm / clip factor |
| --- | --- | --- | --- | --- |
|1, pre-step state0|0.4700091386|0.8024520877|1.2724612263|0.2344329357 /1.0|
|2048, pre-step state2047|0.0578223636|−0.6038227385|−0.5460003749|1.3824579716 /0.7233487724|

Supported reductions, TRAIN rows/timestamps, context/target/visible separation,
component/CSV association and observed factor formula pass. V5 lacks comparable
actual batch tensor evidence. Neither available checkpoint supplies the matched
pre-step2047 decoder, so no final-checkpoint adjoint is labelled this batch's
actual gradient. There is no full parameter-gradient or optimizer replay.

Dataset: **MKT-HIST-2** · **complexity4/5** · V6's2048 successful TRAIN updates; seven frozen CSV scalar columns only, four prospective fixed512 blocks.

| Updates | Recorded clipping | Frequency | Clip factor min / mean / median / max | Observed norm mean / maximum |
| --- | --- | --- | --- | --- |
|1–512|92/512|0.1796875000|0.061721 /0.945214 /1.000000 /1|0.849428 /16.202047|
|513–1024|355/512|0.6933593750|0.085093 /0.820481 /0.904841 /1|1.511894 /11.751784|
|1025–1536|445/512|0.8691406250|0.034256 /0.733855 /0.795482 /1|1.792828 /29.191668|
|1537–2048|434/512|0.8476562500|0.065453 /0.718038 /0.786699 /1|1.826469 /15.278050|
|All2048|1326/2048|0.6474609375|0.034256 /0.804397 /0.890288 /1|1.495155 /29.191668|

Clipping means recorded factor below1. Every saved factor independently matches
min(1,1/(observed_norm+1e-6)); all unit-component sums and numerical
variance/reporting-NLL identities match. Full min/mean/median/max for all six
loss/norm/factor scalars is in the JSON. Those component values are loss
scalars, not component gradient norms. Negative NLL does not make a gradient
negative in any globally comparable sense. Frequent clipping proves that the
shared scaling mechanism was active; it does not identify which branch drove
it, quantify a branch-specific counterfactual update or establish causation.

## Provisional mechanism ranking for review

1. **Disjoint mean/uncertainty parameter-group clipping.** This is the smallest
   next question directly motivated by a recorded active coupling: shared
   clipping affects1326 updates and is frequent in later fixed blocks. A
   controlled comparison could retain V6's forward architecture, fresh draw
   order, objective, unit coefficients, optimizer and fixed heads, changing
   only the clipping geometry for the genuinely disjoint trainable paths.
   A future declaration must specify every parameter exactly once and the
   group caps prospectively, plus group norms/scales as engineering evidence.
   The product of group norm balls differs from one joint ball and can permit
   a larger total norm; it must be disclosed as a mechanism, not independent
   optimization or an equivalent clipping budget. We do not know whether mean
   or uncertainty caused observed clipping. No cap/seed/rate/coefficient sweep,
   rescue or new candidate is selected by this draft.
2. **Separately supervised parent/residual mean allocation.** The saved ridge
   and residual decomposition documents redistribution, and trend weakens
   in-sample. However, volatility/activity TRAIN readouts and query MAE improve;
   these data do not identify residual domination as a bottleneck. Extra
   supervision would introduce another objective constraint and therefore a
   broader explanation/confound. It ranks below testing the observed clipping
   coupling.
3. **Representation diversity or added branch capacity.** Low uncertainty4
   participation ratio exists in both versions while volatility remains
   readable. Raw feature scales and fitted contribution correlations cannot
   establish a diversity defect. Extra capacity changes width-derived readout
   settings and diversity losses add objective assumptions. Neither is the
   minimal next step justified by this diagnosis.

These are bounded hypotheses, not conclusions about held-out causation. A
single same-cohort encoder pair, three saved head fits and eight detailed
queries cannot establish seed robustness or forecasting. Independent audit and
root review must precede a final recommendation or any candidate declaration.

Preserve V3's linear volatility/activity and application baseline; V4's trend,
neural activity, deleted-market neural timing and validation reconstruction;
V5's neural volatility/relation and every displaced record; V6's seven timing
scopes, linear relation and two TRAIN reconstruction records. The separately
reviewed [future strongest-final protocol](../clean_causal_encoder_search_goal_v6_retention.json)
protects the V3–V6 scope maxima/minima under the same22 trust/coverage/positivity
rules, with no TRAIN NLL gate. Earlier cards/counts remain unchanged. No dev
promotion, commit/push or later encoder is declared here. Read the
[V6 review](../CAUSAL_FRESH_V6_REVIEW.md),
[ledger](../clean_causal_encoder_research_ledger.json) and
[final establishment](../causal_fresh_v6/final-establishment.json).
