<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root. -->
# CPB-v13.market-01: five scoped records, 8/22 overall

The [separate final evidence join](causal_fresh_v13/final-establishment.json) passes. **V13 is a verified fresh causal measured candidate; it is not an overall winner.** The unchanged strongest V3–V11 protocol records **8/22 conditions at 100% coverage**: five retention conditions and three positivity conditions pass; fourteen retention/reconstruction conditions fail. **CPB-v3.market-01 remains the clean baseline and sole DEV encoder.** All earlier sources, scores, checkpoints and leaders remain preserved.

V13 sets five scoped records on these saved evaluations:

- MKT-HIST-2 activity neural mean R²: **0.762543630810775**, previously V11 0.7324899092829259.
- MKT-HIST-2 relation neural mean R²: **0.45719709780016277**, previously V11 0.4487745118258597.
- TEMPO-5 synthetic-TRAIN scaler, intact linear accuracy: **0.9895833333333334**, previously V11 0.98828125.
- TEMPO-5 synthetic-TRAIN scaler, deleted linear accuracy: **0.9583333333333334**, previously V9 0.9557291666666666.
- TEMPO-5 synthetic-TRAIN scaler, deleted neural accuracy: **0.9457465277777778**, previously V11 0.9427083333333334.

Timing accuracies above are fractions; tables below show percentages. Each record belongs to its named task, scaler, scoring view and head. None replaces a leader on another task. The activity/relation gains do not establish statistical superiority beyond this known split and three fixed readout fits.

Volatility remains below V3's linear and V8's neural records: V13 reaches 0.9063119722482395 / 0.5557671328731888 linear/neural mean R². Trend reaches 0.6896330438834029 / 0.33696510837131644, below V4's protected pair. Activity and relation linear heads still miss their protected floors. All four market-scaler TEMPO-5 retention conditions fail. All three neural trend repetitions remain positive at 0.37015831436094715, 0.35449716203470905 and 0.2862398487182931.

Fixed-query TRAIN/known-VALIDATION standardized MAE is 0.2546742454449093 / 0.34156313834978447. Reporting Gaussian NLL is −0.3716349321665015 / 0.670673489478906. No reconstruction record is set; all three reconstruction/variance conditions fail. TRAIN NLL remains descriptive, with no additional gate. These measure the encoder/decoder training route separately from fixed-head usefulness.

V13 changes the V11 temporal summary's dispersion scale. Each original bin retains four learned-attention mean rows; its four centered-variance rows become **sqrt(v + 2^-10) − 1/32**, evaluated as actual F32 add, square root and subtraction. The same learned half-local/half-global attention, actual two-sum center, finite-F32 residual correction and live hidden/score/position routes remain. The analytic variance-to-dispersion slope is bounded by 16. Empty local bins still zero the whole eight-row served slice after bias; raw roots remain 1/32 where raw variance is zero. This intermediate summary feeds the existing global32-to-content20 map. It adds no native coordinates or parameters and is separate from native uncertainty4 and decoder conditional variance.

V11 supplies SOURCE only. All 294752 learned values in 76 tensors, optimizer and TRAIN statistics start fresh with source_updates=0; no trained ancestor or future-looking weights are imported. Native40 is learned20 + literal spectral12 + mean4 + uncertainty4, with 3888 fixed prior values and 298640 total values. The unit Huber1 mean plus detached-error Gaussian variance objective, single AdamW/global clip1, constructor shapes/draw order, seeds, 2048 updates and batch8 remain. Equal shapes/counts do not imply equal activations, gradients, functional capacity or FLOPs. No tau sweep, head tuning, post-encoder PCA, checkpoint selection or old encoder/head replay occurs.

One fresh CUDA:0 encoder completes 2048 successful updates in 2048 attempts, zero skips. The recorded synchronized encoder-step time is 163.96149416799994 seconds; trajectory wall time is 226.093972196 seconds and the outer training wrapper 233.08510959600244 seconds. These overlap and must not be added. Synchronized steps exclude loading, masking, checkpoint/evidence/logging and heads, while retaining in-step contract checks; this is a descriptive run, not a matched repeated speed benchmark.

MKT-HIST-2 is BTC/ETH/SOL minute-history past-window retention, complexity4/5, with 4096 TRAIN calibration windows and 112 TRAIN/32 known chronological VALIDATION readout examples per run. TEMPO-5 is slow lead/lag under fast rhythms and grouped distractors, complexity5/5. Each of three cohorts 941101/941102/941103 has 512 TRAIN examples from 256 paired source groups and 256 VALIDATION examples from 128 disjoint source groups. Extra30% deletion transforms those same validation groups. No synthetic encoder training occurs. Complexity is ordinal designed recipe difficulty. There is no TEST or fresh-period forecasting confirmation; architecture selection has repeatedly observed this validation split.

Both saved initial0/final2048 exports receive their own new fixed heads; 1572 CUDA exports are recorded. Ridge penalty1 and neural tanh16/Adam0.01/100 updates retain repetitions2701/2802/2903 and D40 seeds6049515622977958931,8262941649267994858,2095371932330426401. Market heads have164/724 fitted parameters; timing heads82/690. Input, feature and target scaling use the permitted TRAIN data. Synthetic scalers fit once per cohort and are shared across initial/final; market-scaler panels use fresh market TRAIN statistics. Three readout repetitions on one encoder/split are not three encoder retrainings. V3/V4 native36 versus later native40 has width/head-seed confounds. Raw references use larger inputs/heads, disclosed in DETAIL.

Engineering passes 402848 serialized checks (402851 terminal checks, including three post-serialization exclusive-write checks), with636 typed negative cases and7372 CUDA/0 CPU encoder calls. Each saved initial/final checkpoint passes200090 serialized causal checks (200093 terminal), 3685 CUDA calls,128 cutoffs,2350 future-suffix cases and30 internal cases. Changing values or masks after a cutoff cannot change that cutoff's embedding in these tested cases; saved states remain unchanged. Typed resume at512 preserves the exact next loss, parameters and optimizer state. The temporal fixture passes200 checks,51 direct kernel calls,18 F64/18 actual-F32 finite-difference coordinates,4 isolated gradient calls and all nine new predicates.

The independent TRAIN audit passes **566082** disjoint checks with zero mismatches:25634 CORE +61153 statistics +1242 saved tensor-state +117864 saved Gaussian +69301 actual-step objective +290888 temporal learning. Its239469 nested moment checks are already included. All75 identifiable tensors and eight mean/dispersion subgroups show data-gradient/movement evidence;294751 identifiable values describe tensor capacity, not every-scalar learning. Actual1/2048 same-forward witnesses contain60 fields, including25 temporal fields and four raw-root/transformed-dispersion traces. Local saved export/position adjoints use actual F32 roots, the residual correction, PRECLIP route sums and one recorded clip factor. No full hidden gradient, score-to-hidden route, full encoder/global norm or other2046 raw-gradient reconstruction is claimed.

The independent quality audit passes **13624** checks with zero mismatches. All13 saved V11 DATA/statistic pairings pass: eight feature archives compare native20:40 plus declared support/IDs/labels; five fresh-statistic archives compare whole. Learned0:20 is unpaired; no whole40 initial equivalence is claimed. The [Complete mirror](causal_fresh_v13/complete-mirror-attempt-001.json) preserves964 files/1809712300 bytes/866 ignored PT artifacts, including the five canonical checkpoints. The866 artifacts include refusal fixtures, features, readouts and witnesses, not866 trained encoders. The [build mirror](causal_fresh_v13/build-binaries-mirror.json) separately preserves seven binaries and eleven dependency files.

Close this exact2048 recipe and preserve its five scoped records together with every earlier leader. Next: separately declare saved TRAIN-only V11/V13 diagnosis of dispersion scales, gradients, native blocks, each encoder's own saved readout contributions and reconstruction evidence. Protect the new V13 records in any later prospective criterion set. No V14 is selected here, and no old model/head replay, refit, held-out diagnostic tensor selection or validation-cause claim is admitted by this report.

No new experiment was run for this report finalization. Tables below copy the completed saved initial/final evaluation and ten historical final controls unchanged. Raw producer pending-trust records remain literal; the separate final join supplies full trust, while overall advancement, winner selection, application change and publication remain false. The [frozen card](CAUSAL_FRESH_V13_CARD.json), [TRAIN audit](causal_fresh_v13/independent-training-audit.json), [quality audit](causal_fresh_v13/quality-independent-audit.json), [curated byte-copy receipt](causal_fresh_v13/durable-evidence-mirror.json), [REVIEW](CAUSAL_FRESH_V13_REVIEW.md) and [DETAIL](CAUSAL_FRESH_V13_DETAIL.md) retain provenance under [reporting standard1.2](RESULTS_REPORTING_STANDARD.md).

Final local ignored inference checkpoint:1306929 bytes, SHA256 `ba972212bfa4eadda714509fb537b84dc96c55da503f9670550068f1fea537c0`. Initial and all training512/1024/2048 checkpoints remain preserved.

---
## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_volatility; V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 final rows reused, V13 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.964673434 | 0.463051923 | 100 |
| CPB-v4.market-01 | 36 | 0.84956221 | 0.555176229 | 100 |
| CPB-v5.market-01 | 40 | 0.95221109 | 0.648190685 | 100 |
| CPB-v6.market-01 | 40 | 0.877120189 | 0.58463377 | 100 |
| CPB-v7.market-01 | 40 | 0.880509577 | 0.673304976 | 100 |
| CPB-v8.market-01 | 40 | 0.907581398 | 0.735969246 | 100 |
| CPB-v9.market-01 | 40 | 0.90992222 | 0.618522958 | 100 |
| CPB-v10.market-01 | 40 | 0.935674903 | 0.617667046 | 100 |
| CPB-v11.market-01 | 40 | 0.91455411 | 0.537853116 | 100 |
| CPB-v12.market-01 | 40 | 0.908006596 | 0.516889349 | 100 |
| CPB-v13.market-01.init | 40 | -0.0504456065 | -0.0381457267 | 100 |
| CPB-v13.market-01 | 40 | 0.906311972 | 0.555767133 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in logreturn.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 5.14382445e-05 | 6.34988601e-05 | 0.000215581406 | 0.000245836333 |
| CPB-v4.market-01 | 0.000103097727 | 0.000131036924 | 0.000175939865 | 0.000220232799 |
| CPB-v5.market-01 | 5.54373122e-05 | 7.38548092e-05 | 0.00016620698 | 0.000198930285 |
| CPB-v6.market-01 | 9.71580786e-05 | 0.000118428297 | 0.00017129703 | 0.000212519589 |
| CPB-v7.market-01 | 9.79104568e-05 | 0.000116783575 | 0.000149546807 | 0.000188788891 |
| CPB-v8.market-01 | 8.37153482e-05 | 0.000102705787 | 0.000136567527 | 0.00016947626 |
| CPB-v9.market-01 | 8.37456004e-05 | 0.000101396754 | 0.000165643587 | 0.000199147828 |
| CPB-v10.market-01 | 6.52879978e-05 | 8.56851263e-05 | 0.000165531139 | 0.000206288288 |
| CPB-v11.market-01 | 8.03830788e-05 | 9.8755389e-05 | 0.000182269727 | 0.000224867701 |
| CPB-v12.market-01 | 8.31887127e-05 | 0.000102469251 | 0.000185168421 | 0.000230100019 |
| CPB-v13.market-01.init | 0.0002802249 | 0.000346259742 | 0.000273398972 | 0.00034279046 |
| CPB-v13.market-01 | 8.5201116e-05 | 0.000103408744 | 0.000178091321 | 0.000219387568 |

CPB-v3.market-01 neural R²2701/2802/2903: 0.344525499, 0.414672854, 0.629957415.
CPB-v4.market-01 neural R²2701/2802/2903: 0.275507022, 0.68909438, 0.700927285.
CPB-v5.market-01 neural R²2701/2802/2903: 0.539831415, 0.658428065, 0.746312576.
CPB-v6.market-01 neural R²2701/2802/2903: 0.634143112, 0.778464525, 0.341293674.
CPB-v7.market-01 neural R²2701/2802/2903: 0.701381536, 0.825073085, 0.493460306.
CPB-v8.market-01 neural R²2701/2802/2903: 0.78830625, 0.847167508, 0.57243398.
CPB-v9.market-01 neural R²2701/2802/2903: 0.756459839, 0.817281148, 0.281827888.
CPB-v10.market-01 neural R²2701/2802/2903: 0.720076151, 0.691819812, 0.441105176.
CPB-v11.market-01 neural R²2701/2802/2903: 0.267427319, 0.717651399, 0.628480632.
CPB-v12.market-01 neural R²2701/2802/2903: 0.238112203, 0.614622651, 0.697933193.
CPB-v13.market-01.init neural R²2701/2802/2903: 0.0420792479, 0.147964216, -0.304480644.
CPB-v13.market-01 neural R²2701/2802/2903: 0.2774651, 0.760991695, 0.628844604.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_trend; V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 final rows reused, V13 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.263132963 | -0.253979001 | 100 |
| CPB-v4.market-01 | 36 | 0.90721343 | 0.569875231 | 100 |
| CPB-v5.market-01 | 40 | 0.868645445 | 0.523928784 | 100 |
| CPB-v6.market-01 | 40 | 0.675860749 | 0.270702751 | 100 |
| CPB-v7.market-01 | 40 | 0.685042973 | 0.10961324 | 100 |
| CPB-v8.market-01 | 40 | 0.69630659 | -0.0374078453 | 100 |
| CPB-v9.market-01 | 40 | 0.660819198 | -0.133673874 | 100 |
| CPB-v10.market-01 | 40 | 0.897085281 | 0.548230367 | 100 |
| CPB-v11.market-01 | 40 | 0.66199268 | 0.463768775 | 100 |
| CPB-v12.market-01 | 40 | 0.693216439 | 0.378337967 | 100 |
| CPB-v13.market-01.init | 40 | -0.400923531 | -0.405221991 | 100 |
| CPB-v13.market-01 | 40 | 0.689633044 | 0.336965108 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in logreturn/tick.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 4.37234916e-05 | 5.7482267e-05 | 5.85732789e-05 | 7.49730861e-05 |
| CPB-v4.market-01 | 1.59795965e-05 | 2.03977321e-05 | 3.49918123e-05 | 4.32661974e-05 |
| CPB-v5.market-01 | 2.00877722e-05 | 2.42695594e-05 | 3.60577464e-05 | 4.60696486e-05 |
| CPB-v6.market-01 | 3.03383766e-05 | 3.81245859e-05 | 4.64219318e-05 | 5.71545039e-05 |
| CPB-v7.market-01 | 3.07774069e-05 | 3.75807094e-05 | 4.92686076e-05 | 6.27498501e-05 |
| CPB-v8.market-01 | 2.94136081e-05 | 3.69026035e-05 | 5.64403121e-05 | 6.56585445e-05 |
| CPB-v9.market-01 | 3.02697375e-05 | 3.89991332e-05 | 5.81803966e-05 | 6.9751324e-05 |
| CPB-v10.market-01 | 1.81716577e-05 | 2.14821661e-05 | 3.52016131e-05 | 4.49001138e-05 |
| CPB-v11.market-01 | 3.1835506e-05 | 3.8931611e-05 | 3.83098944e-05 | 4.89292963e-05 |
| CPB-v12.market-01 | 2.98776048e-05 | 3.70898746e-05 | 4.01882356e-05 | 5.27403339e-05 |
| CPB-v13.market-01.init | 6.31024592e-05 | 7.92586258e-05 | 5.80784269e-05 | 7.9262557e-05 |
| CPB-v13.market-01 | 3.09169268e-05 | 3.73058605e-05 | 4.22202787e-05 | 5.45062235e-05 |

CPB-v3.market-01 neural R²2701/2802/2903: -0.319245159, -0.237747895, -0.204943949.
CPB-v4.market-01 neural R²2701/2802/2903: 0.640552095, 0.358116367, 0.71095723.
CPB-v5.market-01 neural R²2701/2802/2903: 0.512587238, 0.442624323, 0.61657479.
CPB-v6.market-01 neural R²2701/2802/2903: 0.204803498, 0.321815119, 0.285489636.
CPB-v7.market-01 neural R²2701/2802/2903: 0.239513425, -0.194818323, 0.284144619.
CPB-v8.market-01 neural R²2701/2802/2903: 0.34873419, -0.875923336, 0.41496561.
CPB-v9.market-01 neural R²2701/2802/2903: 0.120397496, -0.818075045, 0.296655928.
CPB-v10.market-01 neural R²2701/2802/2903: 0.579763226, 0.459067372, 0.605860505.
CPB-v11.market-01 neural R²2701/2802/2903: 0.36674785, 0.537284886, 0.487273589.
CPB-v12.market-01 neural R²2701/2802/2903: 0.456439928, 0.320628818, 0.357945156.
CPB-v13.market-01.init neural R²2701/2802/2903: -0.611146037, -0.240275213, -0.364244723.
CPB-v13.market-01 neural R²2701/2802/2903: 0.370158314, 0.354497162, 0.286239849.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_activity; V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 final rows reused, V13 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.862840311 | 0.541426056 | 100 |
| CPB-v4.market-01 | 36 | 0.835143075 | 0.687370545 | 100 |
| CPB-v5.market-01 | 40 | 0.844256819 | 0.617983776 | 100 |
| CPB-v6.market-01 | 40 | 0.672769992 | 0.323179998 | 100 |
| CPB-v7.market-01 | 40 | 0.712390048 | 0.352638034 | 100 |
| CPB-v8.market-01 | 40 | 0.819083477 | 0.654256366 | 100 |
| CPB-v9.market-01 | 40 | 0.83053287 | 0.603294851 | 100 |
| CPB-v10.market-01 | 40 | 0.797199602 | 0.489360129 | 100 |
| CPB-v11.market-01 | 40 | 0.909438776 | 0.732489909 | 100 |
| CPB-v12.market-01 | 40 | 0.88029496 | 0.694805907 | 100 |
| CPB-v13.market-01.init | 40 | 0.552886986 | 0.495099631 | 100 |
| CPB-v13.market-01 | 40 | 0.884358791 | 0.762543631 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in log1p(base_volume).

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.178143553 | 0.212835342 | 0.303552885 | 0.387150693 |
| CPB-v4.market-01 | 0.186144676 | 0.233337263 | 0.251684365 | 0.312491365 |
| CPB-v5.market-01 | 0.196732151 | 0.226795807 | 0.283586415 | 0.346423083 |
| CPB-v6.market-01 | 0.274086555 | 0.328743415 | 0.381790574 | 0.471796096 |
| CPB-v7.market-01 | 0.259703709 | 0.308199873 | 0.382186321 | 0.461718526 |
| CPB-v8.market-01 | 0.219784475 | 0.244438504 | 0.273331807 | 0.337677229 |
| CPB-v9.market-01 | 0.210665121 | 0.236577391 | 0.289818227 | 0.361463983 |
| CPB-v10.market-01 | 0.227485314 | 0.258800376 | 0.343489494 | 0.407410496 |
| CPB-v11.market-01 | 0.147632525 | 0.172942464 | 0.235173927 | 0.295206958 |
| CPB-v12.market-01 | 0.175147275 | 0.1988322 | 0.264633054 | 0.316578048 |
| CPB-v13.market-01.init | 0.334000635 | 0.384272367 | 0.32915858 | 0.406845837 |
| CPB-v13.market-01 | 0.16999593 | 0.195428011 | 0.22303562 | 0.279135153 |

CPB-v3.market-01 neural R²2701/2802/2903: 0.551959766, 0.64999403, 0.42232437.
CPB-v4.market-01 neural R²2701/2802/2903: 0.868267037, 0.635226168, 0.558618431.
CPB-v5.market-01 neural R²2701/2802/2903: 0.777561455, 0.374143234, 0.702246639.
CPB-v6.market-01 neural R²2701/2802/2903: 0.19661957, 0.386097575, 0.386822848.
CPB-v7.market-01 neural R²2701/2802/2903: 0.253826868, 0.414419927, 0.389667307.
CPB-v8.market-01 neural R²2701/2802/2903: 0.617280912, 0.675654682, 0.669833502.
CPB-v9.market-01 neural R²2701/2802/2903: 0.567901982, 0.660584005, 0.581398567.
CPB-v10.market-01 neural R²2701/2802/2903: 0.540950734, 0.617987364, 0.309142288.
CPB-v11.market-01 neural R²2701/2802/2903: 0.641732517, 0.784916728, 0.770820483.
CPB-v12.market-01 neural R²2701/2802/2903: 0.632617188, 0.707396224, 0.744404307.
CPB-v13.market-01.init neural R²2701/2802/2903: 0.404009574, 0.473132145, 0.608157175.
CPB-v13.market-01 neural R²2701/2802/2903: 0.713490985, 0.767283883, 0.806856024.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_relation; V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 final rows reused, V13 initial/final newly measured.

| Method | Size | Linear R² | Neural mean R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.00700444683 | -0.0855656715 | 100 |
| CPB-v4.market-01 | 36 | 0.326405067 | 0.209510464 | 100 |
| CPB-v5.market-01 | 40 | 0.252328556 | 0.370505348 | 100 |
| CPB-v6.market-01 | 40 | 0.622100687 | 0.322683609 | 100 |
| CPB-v7.market-01 | 40 | 0.600800572 | 0.324861744 | 100 |
| CPB-v8.market-01 | 40 | 0.504487017 | 0.0389512195 | 100 |
| CPB-v9.market-01 | 40 | 0.531691195 | 0.326550049 | 100 |
| CPB-v10.market-01 | 40 | 0.0816643994 | 0.368204964 | 100 |
| CPB-v11.market-01 | 40 | 0.552942248 | 0.448774512 | 100 |
| CPB-v12.market-01 | 40 | 0.54557653 | 0.397732658 | 100 |
| CPB-v13.market-01.init | 40 | -0.0867635753 | -0.234424318 | 100 |
| CPB-v13.market-01 | 40 | 0.536356573 | 0.457197098 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Physical errors in correlation.

| Method | Linear MAE | Linear RMSE | Neural mean MAE | Neural mean RMSE |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 0.0723982929 | 0.086491159 | 0.0702199215 | 0.0894458833 |
| CPB-v4.market-01 | 0.0562974528 | 0.0707384319 | 0.0615928121 | 0.0766223466 |
| CPB-v5.market-01 | 0.0586723303 | 0.074526619 | 0.0547690214 | 0.0682288649 |
| CPB-v6.market-01 | 0.0456908024 | 0.0529839185 | 0.055655748 | 0.0702095263 |
| CPB-v7.market-01 | 0.0474929235 | 0.0544566571 | 0.056510372 | 0.0703589445 |
| CPB-v8.market-01 | 0.0514085353 | 0.0606713337 | 0.067009939 | 0.0816960994 |
| CPB-v9.market-01 | 0.0507565183 | 0.0589823653 | 0.0556393026 | 0.0700030395 |
| CPB-v10.market-01 | 0.0678380386 | 0.0825955659 | 0.0545920982 | 0.0684830795 |
| CPB-v11.market-01 | 0.0473268932 | 0.0576285693 | 0.0511515364 | 0.0639328596 |
| CPB-v12.market-01 | 0.0477718477 | 0.0581013736 | 0.0533119269 | 0.0667596203 |
| CPB-v13.market-01.init | 0.0710592507 | 0.0898511334 | 0.0760069219 | 0.0949204572 |
| CPB-v13.market-01 | 0.0487867236 | 0.0586878334 | 0.052747273 | 0.0634132012 |

CPB-v3.market-01 neural R²2701/2802/2903: -0.300654281, 0.161014634, -0.117057368.
CPB-v4.market-01 neural R²2701/2802/2903: 0.182210538, 0.23983439, 0.206486466.
CPB-v5.market-01 neural R²2701/2802/2903: 0.317728582, 0.48696529, 0.306822173.
CPB-v6.market-01 neural R²2701/2802/2903: 0.0570793218, 0.525771962, 0.385199543.
CPB-v7.market-01 neural R²2701/2802/2903: 0.115499431, 0.489241068, 0.369844732.
CPB-v8.market-01 neural R²2701/2802/2903: 0.356560851, -0.689937124, 0.450229932.
CPB-v9.market-01 neural R²2701/2802/2903: 0.466017712, 0.0430395951, 0.470592839.
CPB-v10.market-01 neural R²2701/2802/2903: 0.331177144, 0.359772869, 0.413664879.
CPB-v11.market-01 neural R²2701/2802/2903: 0.389406727, 0.504697432, 0.452219377.
CPB-v12.market-01 neural R²2701/2802/2903: 0.290003339, 0.449589084, 0.453605552.
CPB-v13.market-01.init neural R²2701/2802/2903: -0.398155006, -0.503859164, 0.198741217.
CPB-v13.market-01 neural R²2701/2802/2903: 0.394060777, 0.530924934, 0.446605582.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact; V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.484375 | 95.703125 | 100 |
| CPB-v4.market-01 | 36 | 98.3072917 | 96.9618056 | 100 |
| CPB-v5.market-01 | 40 | 97.1354167 | 96.5711806 | 100 |
| CPB-v6.market-01 | 40 | 98.6979167 | 98.4809028 | 100 |
| CPB-v7.market-01 | 40 | 98.5677083 | 97.8732639 | 100 |
| CPB-v8.market-01 | 40 | 97.65625 | 97.2222222 | 100 |
| CPB-v9.market-01 | 40 | 98.6979167 | 97.8732639 | 100 |
| CPB-v10.market-01 | 40 | 97.5260417 | 97.0920139 | 100 |
| CPB-v11.market-01 | 40 | 98.828125 | 98.2638889 | 100 |
| CPB-v12.market-01 | 40 | 98.4375 | 97.7430556 | 100 |
| CPB-v13.market-01.init | 40 | 93.8802083 | 93.4895833 | 100 |
| CPB-v13.market-01 | 40 | 98.9583333 | 98.2204861 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted; V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.1458333 | 91.7534722 | 100 |
| CPB-v4.market-01 | 36 | 94.6614583 | 93.359375 | 100 |
| CPB-v5.market-01 | 40 | 92.96875 | 92.7951389 | 100 |
| CPB-v6.market-01 | 40 | 95.3125 | 93.4027778 | 100 |
| CPB-v7.market-01 | 40 | 93.359375 | 92.2309028 | 100 |
| CPB-v8.market-01 | 40 | 93.2291667 | 93.0121528 | 100 |
| CPB-v9.market-01 | 40 | 95.5729167 | 93.6197917 | 100 |
| CPB-v10.market-01 | 40 | 91.9270833 | 91.1458333 | 100 |
| CPB-v11.market-01 | 40 | 94.53125 | 94.2708333 | 100 |
| CPB-v12.market-01 | 40 | 94.140625 | 93.5329861 | 100 |
| CPB-v13.market-01.init | 40 | 88.5416667 | 86.8489583 | 100 |
| CPB-v13.market-01 | 40 | 95.8333333 | 94.5746528 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact; V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.3489583 | 98.1770833 | 100 |
| CPB-v4.market-01 | 36 | 99.3489583 | 98.828125 | 100 |
| CPB-v5.market-01 | 40 | 99.21875 | 98.8715278 | 100 |
| CPB-v6.market-01 | 40 | 99.7395833 | 99.21875 | 100 |
| CPB-v7.market-01 | 40 | 99.21875 | 98.4809028 | 100 |
| CPB-v8.market-01 | 40 | 99.609375 | 99.2621528 | 100 |
| CPB-v9.market-01 | 40 | 99.8697917 | 98.4809028 | 100 |
| CPB-v10.market-01 | 40 | 98.9583333 | 98.3940972 | 100 |
| CPB-v11.market-01 | 40 | 99.7395833 | 98.6111111 | 100 |
| CPB-v12.market-01 | 40 | 99.7395833 | 97.7430556 | 100 |
| CPB-v13.market-01.init | 40 | 95.1822917 | 94.8350694 | 100 |
| CPB-v13.market-01 | 40 | 99.609375 | 98.046875 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted; V3/V4/V5/V6/V7/V8/V9/V10/V11/V12 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.921875 | 94.2708333 | 100 |
| CPB-v4.market-01 | 36 | 95.8333333 | 95.6163194 | 100 |
| CPB-v5.market-01 | 40 | 94.140625 | 94.0972222 | 100 |
| CPB-v6.market-01 | 40 | 97.265625 | 95.3125 | 100 |
| CPB-v7.market-01 | 40 | 94.53125 | 91.8836806 | 100 |
| CPB-v8.market-01 | 40 | 96.2239583 | 95.0520833 | 100 |
| CPB-v9.market-01 | 40 | 93.359375 | 92.8385417 | 100 |
| CPB-v10.market-01 | 40 | 93.359375 | 91.9270833 | 100 |
| CPB-v11.market-01 | 40 | 95.3125 | 93.7065972 | 100 |
| CPB-v12.market-01 | 40 | 93.75 | 93.0121528 | 100 |
| CPB-v13.market-01.init | 40 | 87.7604167 | 85.1128472 | 100 |
| CPB-v13.market-01 | 40 | 95.3125 | 94.3142361 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.

| Method | Updates | TRAIN MAE | VALIDATION MAE | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | 0.25904599 | 0.376414231 | 144.341116 (reused) |
| CPB-v4.market-01 | 2048 | 0.272508032 | 0.308316924 | 172.860055 (reused) |
| CPB-v5.market-01 | 2048 | 0.254463936 | 0.375196294 | 134.983334 (reused) |
| CPB-v6.market-01 | 2048 | 0.250172579 | 0.354519735 | 175.575555 (reused) |
| CPB-v7.market-01 | 2048 | 0.241805338 | 0.370420511 | 219.730088 (reused) |
| CPB-v8.market-01 | 2048 | 0.247146261 | 0.324973012 | 178.284957 (reused) |
| CPB-v9.market-01 | 2048 | 0.244540347 | 0.326107515 | 147.328444 (reused) |
| CPB-v10.market-01 | 2048 | 0.251425894 | 0.386122321 | 146.063882 (reused) |
| CPB-v11.market-01 | 2048 | 0.251797512 | 0.320134118 | 146.497975 (reused) |
| CPB-v12.market-01 | 2048 | 0.252382241 | 0.315910852 | 184.812345 (reused) |
| CPB-v13.market-01.init | 0 | 0.664068256 | 0.958038458 | 0 |
| CPB-v13.market-01 | 2048 | 0.254674245 | 0.341563138 | 163.961494 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Same query distributions; NLL omits constant and can be negative.

| Method | TRAIN Gaussian NLL | VALIDATION Gaussian NLL |
| --- | --- | --- |
| CPB-v3.market-01 | -0.38017808 | 1.52179658 |
| CPB-v4.market-01 | -0.356692772 | -0.0453814881 |
| CPB-v5.market-01 | -0.324377373 | 2.68745022 |
| CPB-v6.market-01 | -0.400207362 | 0.648448045 |
| CPB-v7.market-01 | -0.408827278 | 0.868789874 |
| CPB-v8.market-01 | -0.381215145 | 0.424167868 |
| CPB-v9.market-01 | -0.392451626 | 0.475387925 |
| CPB-v10.market-01 | -0.377460728 | 1.61986075 |
| CPB-v11.market-01 | -0.369659381 | 0.312348846 |
| CPB-v12.market-01 | -0.374763669 | 0.297063679 |
| CPB-v13.market-01.init | 0.42373283 | 0.795646599 |
| CPB-v13.market-01 | -0.371634932 | 0.670673489 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. All22 frozen numeric conditions; trust and coverage are separate.

| Dataset | Condition | Candidate | Frozen floor/ceiling | Source | Rule | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| MKT-HIST-2 | past_volatility:linear_r2 | 0.906311972 | 0.964673434 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_volatility:neural_mean_r2 | 0.555767133 | 0.735969246 | CPB-v8.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:linear_r2 | 0.689633044 | 0.90721343 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:neural_mean_r2 | 0.336965108 | 0.569875231 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:linear_r2 | 0.884358791 | 0.909438776 | CPB-v11.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:neural_mean_r2 | 0.762543631 | 0.732489909 | CPB-v11.market-01 | candidate + 1e-12 >= floor | PASS |
| MKT-HIST-2 | past_relation:linear_r2 | 0.536356573 | 0.622100687 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:neural_mean_r2 | 0.457197098 | 0.448774512 | CPB-v11.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_intact:linear_head_accuracy | 0.989583333 | 0.98828125 | CPB-v11.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_intact:neural_head_accuracy | 0.982204861 | 0.984809028 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:linear_head_accuracy | 0.958333333 | 0.955729167 | CPB-v9.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_deleted:neural_head_accuracy | 0.945746528 | 0.942708333 | CPB-v11.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_intact:linear_head_accuracy | 0.99609375 | 0.998697917 | CPB-v9.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_intact:neural_head_accuracy | 0.98046875 | 0.992621528 | CPB-v8.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:linear_head_accuracy | 0.953125 | 0.97265625 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:neural_head_accuracy | 0.943142361 | 0.956163194 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | train:standardized_MAE | 0.254674245 | 0.241805338 | CPB-v7.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | validation:standardized_MAE | 0.341563138 | 0.308316924 | CPB-v4.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | known_validation:Gaussian_NLL_F64 | 0.670673489 | -0.0453814881 | CPB-v4.market-01 | candidate < floor - 1e-12 | FAIL |
| MKT-HIST-2 | past_trend:all3_neural_R2_positive | 0.370158314, 0.354497162, 0.286239849 | 0 | strict positivity | all three final neural trend R2 values > 0 | PASS |
| MKT-HIST-2 | past_relation:linear_r2:positive | 0.536356573 | 0 | strict positivity | candidate > 0 | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2:positive | 0.457197098 | 0 | strict positivity | candidate > 0 | PASS |

The exact declared trajectory is stopped at2048. Retain every failed condition and all ten references and their scoped strengths. Further mechanisms require separate prospective declarations; no coefficient/seed/rate/budget/head rescue. V3's reported144s training cost is reused historical evidence,not a new matched benchmark.

