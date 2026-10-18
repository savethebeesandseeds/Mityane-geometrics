<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root. -->
# DRAFT — CPB-v11.market-01: 8/22, useful scoped gains with retention failures

**Draft only: the separate final trust join is pending.** The independent quality audit has passed 13287 checks with zero mismatches. This is a report of saved producer measurements, not final measured-candidate or leader establishment. Runtime proofs, the independent TRAIN audit and independent quality arithmetic have passed. All 13 declared partial initial DATA/statistic pairings are exact; separate final trust admission remains pending. No overall winner, advancement, application promotion or publication is selected. **CPB-v3.market-01 remains the clean baseline and sole DEV encoder.** Producer pending-trust fields are preserved byte-exactly.

The raw fixed-head comparison reports **8/22 frozen conditions and 100% coverage**. Five passing retention scopes exceed their existing protected floors: market activity linear R² 0.9094387755129657 and neural mean R² 0.7324899092829259; market relation neural mean R² 0.4487745118258597; TEMPO-5 synthetic-TRAIN-scaler intact linear accuracy 0.98828125 and deleted neural accuracy 0.9427083333333334. These are preliminary scoped record directions awaiting separate final trust admission, not newly established leaders. The other three passes are all-three neural trend positivity and positive relation linear/neural readouts. Preserve every earlier scoped leader and all failed conditions.

Both volatility floors, both trend floors, linear relation, six timing floors, both query-MAE ceilings and the strict known-VALIDATION Gaussian-NLL ceiling fail. Final neural trend repetitions 2701/2802/2903 are all positive:0.36674785041014712,0.53728488582285727,0.48727358908838192, mean R² 0.4637687751071288. Their positivity does not satisfy the stronger V4 trend-retention floor. TRAIN query MAE 0.25179751241660819 and known-VALIDATION MAE 0.32013411834078387 miss the strongest protected ceilings; reporting NLL −0.36965938103736928/0.31234884646836464 sets no reconstruction record. TRAIN NLL is descriptive and adds no gate.

V11 changes one architectural summary relative to V9 source: each original temporal bin's eight intermediate export rows now use four weighted-mean rows and four componentwise centered-variance rows under the same half-local/half-global attention. The actual V9 two-sum mean is preserved; variance remains live through attention, and an empty local bin zeros the whole eight-row slice after bias. The 16 intermediate variance rows feed the unchanged 32-to-content 20 path; they are distinct from native uncertainty 4 and decoder conditional variance. No native coordinates or learned parameters are added. Equal tensor counts do not imply equal functional capacity, scales, gradients or FLOPs, and these results do not identify a validation-error cause.

Native 40 is learned 20 + literal spectral 12 + mean 4 + uncertainty 4, with 298640 total values,294752 learned values across 76 registered tensors and 3888 fixed prior values. All learned values, optimizer and TRAIN statistics start fresh, source_updates=0. The source recipe restores V9's unit Huber(delta 1) mean plus detached-error Gaussian variance calibration, coefficients 1/1, shared global clip 1 and one AdamW; V10's live joint mean-NLL route is excluded. No old weights, teachers, fitted statistics, initial quality scores or old model/head execution are imported. V10 is the latest established historical control, not the scientific source parent or a winner.

MKT-HIST-2 uses BTC/ETH/SOL minute histories to test retention of past-window information, not forecasting:4096 TRAIN calibration anchors and 112 TRAIN/32 known chronological VALIDATION readout windows per run. TEMPO-5 uses independent generator cohorts 941101/941102/941103, each 512 TRAIN examples from 256 paired source groups and 256 VALIDATION examples from 128 separate paired groups; deletion transforms those same validation groups. No TEST or fresh future-period confirmation is claimed. Complexity 4/5 and 5/5 are ordinal designed recipes, not accuracy or observed intrinsic difficulty.

There is one fresh encoder trajectory: CUDA:0,2048 successful updates/2048 attempts, zero skips, batch 8. Both initial 0 and final 2048 embeddings receive new fixed heads, with 1572 recorded CUDA exports. Ridge penalty 1 and 16 tanh/Adam 0.01/100-update neural heads keep repetitions 2701/2802/2903 and D40 seeds 6049515622977958931,8262941649267994858,2095371932330426401. Market heads have 164/724 fitted parameters; TEMPO heads 82/690. Feature/target normalization uses permitted TRAIN data. Synthetic input scales are fresh per cohort and shared across initial/final; the other timing policy uses fresh market TRAIN scaling. Three neural-head repetitions are head variation on one encoder/split, not independent encoder runs or market-cohort uncertainty. V3/V4 native 36 and V5–V11 native 40 comparisons retain capacity and width-derived head-seed confounds. No PCA, tuning, old replay or synthetic encoder training occurs.

Engineering passes 360571 checks and 491 typed negative cases; each saved initial/final causal proof passes 179097 checks with 3685 CUDA calls. All 128 cutoffs,2350 future-suffix and 30 internal cases cover future values and masks. Typed resume 512 is exact. The independent TRAIN audit passes 516865 total checks with zero mismatches, covering 75 identifiable tensors, eight mean/variance subgroups, four export/position groups and three between-bin directions. The 294751 values in identifiable tensors are capacity, not every-scalar learning proof; the registered common score-bias gauge is separately observed. Initial pairing remains 13 saved V9 DATA/statistic references: eight feature archives compare native 20:40 only, five statistics compare whole declared tensors; content 0:20 is unpaired. Historical V10 full 40 pairing retains its own semantics.

The [independent quality audit](../quality-independent-audit.json) confirms all 13 current pairings, five scientific plus two aggregation stages and the unchanged 22-condition arithmetic. It does not replace the still-pending separate final trust join. The tables below are literal sections of the saved comparison report: eight final historical controls plus newly measured own initial/final rows, with four market tasks and four timing panels. Historical scores and costs are reused unchanged; historical pass counts retain their own frozen criteria. [DETAIL](DETAIL.md) preserves the producer report tail with all full-precision current neural repetitions, physical errors, cohort/view repetitions, reconstruction components and separate costs. [REVIEW](REVIEW.md) explains the decision boundary. Source evidence is the [copied table](../../results/causal_fresh_v11/table_rows.json), [comparison](../../results/causal_fresh_v11/comparison.json) and [byte-copy receipt](../durable-evidence-mirror.json), under [reporting standard 1.2](../../RESULTS_REPORTING_STANDARD.md). No metrics were recomputed for these drafts.

After the pending separate trust join, close this exact 2048 recipe without coefficient/seed/rate/budget/head/checkpoint rescue. Preserve all historical leaders and the five preliminary scoped directions if established. A future strengthened retention declaration should precede a separately declared saved TRAIN-only diagnosis before selecting another mechanism; no next encoder is admitted here. The broader search goal remains active.

---
## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_volatility; V3/V4/V5/V6/V7/V8/V9/V10 final rows reused, V11 initial/final newly measured.

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
| CPB-v11.market-01.init | 40 | -0.0121365493 | -0.166095383 | 100 |
| CPB-v11.market-01 | 40 | 0.91455411 | 0.537853116 | 100 |

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
| CPB-v11.market-01.init | 0.000271180275 | 0.000339887169 | 0.00029486327 | 0.000364195847 |
| CPB-v11.market-01 | 8.03830788e-05 | 9.8755389e-05 | 0.000182269727 | 0.000224867701 |

CPB-v3.market-01 neural R²2701/2802/2903: 0.344525499, 0.414672854, 0.629957415.
CPB-v4.market-01 neural R²2701/2802/2903: 0.275507022, 0.68909438, 0.700927285.
CPB-v5.market-01 neural R²2701/2802/2903: 0.539831415, 0.658428065, 0.746312576.
CPB-v6.market-01 neural R²2701/2802/2903: 0.634143112, 0.778464525, 0.341293674.
CPB-v7.market-01 neural R²2701/2802/2903: 0.701381536, 0.825073085, 0.493460306.
CPB-v8.market-01 neural R²2701/2802/2903: 0.78830625, 0.847167508, 0.57243398.
CPB-v9.market-01 neural R²2701/2802/2903: 0.756459839, 0.817281148, 0.281827888.
CPB-v10.market-01 neural R²2701/2802/2903: 0.720076151, 0.691819812, 0.441105176.
CPB-v11.market-01.init neural R²2701/2802/2903: -0.320157911, 0.00940701251, -0.187535251.
CPB-v11.market-01 neural R²2701/2802/2903: 0.267427319, 0.717651399, 0.628480632.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_trend; V3/V4/V5/V6/V7/V8/V9/V10 final rows reused, V11 initial/final newly measured.

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
| CPB-v11.market-01.init | 40 | -0.46505239 | -0.439238836 | 100 |
| CPB-v11.market-01 | 40 | 0.66199268 | 0.463768775 | 100 |

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
| CPB-v11.market-01.init | 6.19010459e-05 | 8.1052404e-05 | 5.97012658e-05 | 8.01797704e-05 |
| CPB-v11.market-01 | 3.1835506e-05 | 3.8931611e-05 | 3.83098944e-05 | 4.89292963e-05 |

CPB-v3.market-01 neural R²2701/2802/2903: -0.319245159, -0.237747895, -0.204943949.
CPB-v4.market-01 neural R²2701/2802/2903: 0.640552095, 0.358116367, 0.71095723.
CPB-v5.market-01 neural R²2701/2802/2903: 0.512587238, 0.442624323, 0.61657479.
CPB-v6.market-01 neural R²2701/2802/2903: 0.204803498, 0.321815119, 0.285489636.
CPB-v7.market-01 neural R²2701/2802/2903: 0.239513425, -0.194818323, 0.284144619.
CPB-v8.market-01 neural R²2701/2802/2903: 0.34873419, -0.875923336, 0.41496561.
CPB-v9.market-01 neural R²2701/2802/2903: 0.120397496, -0.818075045, 0.296655928.
CPB-v10.market-01 neural R²2701/2802/2903: 0.579763226, 0.459067372, 0.605860505.
CPB-v11.market-01.init neural R²2701/2802/2903: -0.470057514, -0.208693189, -0.638965804.
CPB-v11.market-01 neural R²2701/2802/2903: 0.36674785, 0.537284886, 0.487273589.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_activity; V3/V4/V5/V6/V7/V8/V9/V10 final rows reused, V11 initial/final newly measured.

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
| CPB-v11.market-01.init | 40 | 0.51387467 | 0.424806722 | 100 |
| CPB-v11.market-01 | 40 | 0.909438776 | 0.732489909 | 100 |

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
| CPB-v11.market-01.init | 0.343889946 | 0.400686422 | 0.3456807 | 0.434734659 |
| CPB-v11.market-01 | 0.147632525 | 0.172942464 | 0.235173927 | 0.295206958 |

CPB-v3.market-01 neural R²2701/2802/2903: 0.551959766, 0.64999403, 0.42232437.
CPB-v4.market-01 neural R²2701/2802/2903: 0.868267037, 0.635226168, 0.558618431.
CPB-v5.market-01 neural R²2701/2802/2903: 0.777561455, 0.374143234, 0.702246639.
CPB-v6.market-01 neural R²2701/2802/2903: 0.19661957, 0.386097575, 0.386822848.
CPB-v7.market-01 neural R²2701/2802/2903: 0.253826868, 0.414419927, 0.389667307.
CPB-v8.market-01 neural R²2701/2802/2903: 0.617280912, 0.675654682, 0.669833502.
CPB-v9.market-01 neural R²2701/2802/2903: 0.567901982, 0.660584005, 0.581398567.
CPB-v10.market-01 neural R²2701/2802/2903: 0.540950734, 0.617987364, 0.309142288.
CPB-v11.market-01.init neural R²2701/2802/2903: 0.424036663, 0.324916921, 0.525466583.
CPB-v11.market-01 neural R²2701/2802/2903: 0.641732517, 0.784916728, 0.770820483.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_relation; V3/V4/V5/V6/V7/V8/V9/V10 final rows reused, V11 initial/final newly measured.

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
| CPB-v11.market-01.init | 40 | -0.0597245578 | -0.0898973968 | 100 |
| CPB-v11.market-01 | 40 | 0.552942248 | 0.448774512 | 100 |

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
| CPB-v11.market-01.init | 0.0700664851 | 0.0887263308 | 0.0724569869 | 0.0895625398 |
| CPB-v11.market-01 | 0.0473268932 | 0.0576285693 | 0.0511515364 | 0.0639328596 |

CPB-v3.market-01 neural R²2701/2802/2903: -0.300654281, 0.161014634, -0.117057368.
CPB-v4.market-01 neural R²2701/2802/2903: 0.182210538, 0.23983439, 0.206486466.
CPB-v5.market-01 neural R²2701/2802/2903: 0.317728582, 0.48696529, 0.306822173.
CPB-v6.market-01 neural R²2701/2802/2903: 0.0570793218, 0.525771962, 0.385199543.
CPB-v7.market-01 neural R²2701/2802/2903: 0.115499431, 0.489241068, 0.369844732.
CPB-v8.market-01 neural R²2701/2802/2903: 0.356560851, -0.689937124, 0.450229932.
CPB-v9.market-01 neural R²2701/2802/2903: 0.466017712, 0.0430395951, 0.470592839.
CPB-v10.market-01 neural R²2701/2802/2903: 0.331177144, 0.359772869, 0.413664879.
CPB-v11.market-01.init neural R²2701/2802/2903: -0.292307292, -0.165861408, 0.18847651.
CPB-v11.market-01 neural R²2701/2802/2903: 0.389406727, 0.504697432, 0.452219377.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact; V3/V4/V5/V6/V7/V8/V9/V10 reused.

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
| CPB-v11.market-01.init | 40 | 93.75 | 93.0989583 | 100 |
| CPB-v11.market-01 | 40 | 98.828125 | 98.2638889 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted; V3/V4/V5/V6/V7/V8/V9/V10 reused.

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
| CPB-v11.market-01.init | 40 | 88.0208333 | 85.6770833 | 100 |
| CPB-v11.market-01 | 40 | 94.53125 | 94.2708333 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact; V3/V4/V5/V6/V7/V8/V9/V10 reused.

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
| CPB-v11.market-01.init | 40 | 95.8333333 | 94.4878472 | 100 |
| CPB-v11.market-01 | 40 | 99.7395833 | 98.6111111 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted; V3/V4/V5/V6/V7/V8/V9/V10 reused.

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
| CPB-v11.market-01.init | 40 | 86.9791667 | 84.2013889 | 100 |
| CPB-v11.market-01 | 40 | 95.3125 | 93.7065972 | 100 |

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
| CPB-v11.market-01.init | 0 | 0.664073361 | 0.958012412 | 0 |
| CPB-v11.market-01 | 2048 | 0.251797512 | 0.320134118 | 146.497975 |

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
| CPB-v11.market-01.init | 0.42373305 | 0.795590073 |
| CPB-v11.market-01 | -0.369659381 | 0.312348846 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. All22 frozen numeric conditions; trust and coverage are separate.

| Dataset | Condition | Candidate | Frozen floor/ceiling | Source | Rule | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| MKT-HIST-2 | past_volatility:linear_r2 | 0.91455411 | 0.964673434 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_volatility:neural_mean_r2 | 0.537853116 | 0.735969246 | CPB-v8.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:linear_r2 | 0.66199268 | 0.90721343 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:neural_mean_r2 | 0.463768775 | 0.569875231 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:linear_r2 | 0.909438776 | 0.862840311 | CPB-v3.market-01 | candidate + 1e-12 >= floor | PASS |
| MKT-HIST-2 | past_activity:neural_mean_r2 | 0.732489909 | 0.687370545 | CPB-v4.market-01 | candidate + 1e-12 >= floor | PASS |
| MKT-HIST-2 | past_relation:linear_r2 | 0.552942248 | 0.622100687 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:neural_mean_r2 | 0.448774512 | 0.370505348 | CPB-v5.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_intact:linear_head_accuracy | 0.98828125 | 0.986979167 | CPB-v6.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | synthetic_train:validation_intact:neural_head_accuracy | 0.982638889 | 0.984809028 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:linear_head_accuracy | 0.9453125 | 0.955729167 | CPB-v9.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:neural_head_accuracy | 0.942708333 | 0.936197917 | CPB-v9.market-01 | candidate + 1e-12 >= floor | PASS |
| TEMPO-5 | market_scaler:validation_intact:linear_head_accuracy | 0.997395833 | 0.998697917 | CPB-v9.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_intact:neural_head_accuracy | 0.986111111 | 0.992621528 | CPB-v8.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:linear_head_accuracy | 0.953125 | 0.97265625 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:neural_head_accuracy | 0.937065972 | 0.956163194 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | train:standardized_MAE | 0.251797512 | 0.241805338 | CPB-v7.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | validation:standardized_MAE | 0.320134118 | 0.308316924 | CPB-v4.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | known_validation:Gaussian_NLL_F64 | 0.312348846 | -0.0453814881 | CPB-v4.market-01 | candidate < floor - 1e-12 | FAIL |
| MKT-HIST-2 | past_trend:all3_neural_R2_positive | 0.36674785, 0.537284886, 0.487273589 | 0 | strict positivity | all three final neural trend R2 values > 0 | PASS |
| MKT-HIST-2 | past_relation:linear_r2:positive | 0.552942248 | 0 | strict positivity | candidate > 0 | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2:positive | 0.448774512 | 0 | strict positivity | candidate > 0 | PASS |

The exact declared trajectory is stopped at2048. Retain every failed condition and all eight references and their scoped strengths. Further mechanisms require separate prospective declarations; no coefficient/seed/rate/budget/head rescue. V3's reported144s training cost is reused historical evidence,not a new matched benchmark.

