<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root. -->
# CPB-v12.market-01: 3/22, uniform dispersion misses the retained gains

**The separate final trust join passes: CPB-v12.market-01 is a fully trusted measured candidate.** Both independent audits pass with zero mismatches: TRAIN 570648 total checks and quality 13441. Runtime causal/CUDA/typed-resume proofs and all 13 declared partial DATA/statistic pairings pass. Overall advancement, winner selection, application change and publication remain false. **CPB-v3.market-01 remains the clean baseline and sole DEV encoder.** The fixed-head comparison records **3/22 conditions at 100% coverage** under the [strongest V3–V11 retention protocol](causal_fresh_v12/goal-protocol.json). All 19 quality-retention and reconstruction conditions fail; only all-three neural trend positivity and positive relation linear/neural readouts pass. No new quality-scoped or reconstruction record is set. Preserve every earlier leader, including V11's five scoped gains, and this complete measured failure.

Relative to V11, linear trend improves from 0.6619926800951619 to 0.6932164386458095, but remains below V4's protected 0.9072134297036245. All four market neural means fall: volatility 0.5168893488615781, trend 0.37833796722165275, activity 0.6948059065418781 and relation 0.3977326581782557. The three final neural trend repetitions remain positive at 0.45643992775295772,0.3206288177944161,0.35794515611758448. All eight TEMPO-5 retention floors fail; market-scaler intact linear accuracy ties V11 at 0.9973958333333334 but does not reach V9's 0.9986979166666666 leader. This is no overall improvement.

Known-VALIDATION fixed-query standardized MAE 0.31591085236433497 and reporting Gaussian NLL 0.29706367854148069 improve relative to V11's 0.32013411834078387/0.31234884646836464, but miss V4's protected ceilings. TRAIN MAE 0.25238224097237538 is worse than V11's 0.25179751241660819 and V7's 0.24180533789857825 leader. TRAIN NLL −0.37476366899994823 remains descriptive and does not beat V7's −0.40882727825899934; it adds no gate. Reconstruction describes the encoder/decoder route, separately from task readouts.

V12 changes one V11 architectural choice: each original bin keeps its four attention-weighted mean rows, while four componentwise dispersion rows now use fixed half-local/half-global uniform weights over retained ORIGINAL patch support and their OWN live two-sum uniform center. Direct dispersion score/position derivatives are zero; hidden/center derivatives, attention-mean score derivatives and Transformer attention remain live. Empty local bins still zero the whole eight-row export after bias. The 16 intermediate dispersion rows feed the unchanged 32-to-content 20 map. They are separate from native uncertainty 4 and decoder conditional variance; no native coordinates or learned parameters are added. Equal parameter shapes/counts/draw order do not imply equal functional capacity, scales, gradients or FLOPs.

Native40 remains learned 20 + literal spectral 12 + mean 4 + uncertainty 4: 298640 total values,294752 learned values across 76 registered tensors and 3888 fixed prior values. V11 is a SOURCE-only architecture/objective parent and latest measured reference; all learned values, optimizer and TRAIN statistics start fresh with source_updates=0. The V11 unit Huber1 mean plus detached-error Gaussian variance objective, coefficients 1/1, one AdamW and shared global clip 1 remain. No old trained weights, teacher, fitted statistics, model/head execution or initial quality scores are imported.

MKT-HIST-2 uses BTC/ETH/SOL minute histories for past-window retention, with 4096 TRAIN calibration anchors and 112 TRAIN/32 known chronological VALIDATION readout windows per run. TEMPO-5 cohorts 941101/941102/941103 each contain512 TRAIN examples from 256 paired source groups and 256 VALIDATION examples from 128 separate paired groups. Extra 30% deletion transforms those same validation groups. Complexity4/5 and5/5 are ordinal designed recipes. There is no TEST or fresh future-period/forecasting confirmation.

One fresh CUDA:0 encoder trajectory completes 2048 successful updates/2048 attempts, zero skips, batch 8. Both initial 0 and final 2048 embeddings receive new fixed heads, with 1572 recorded CUDA exports. Ridge 1 and tanh 16/Adam 0.01/100-step heads retain repetitions 2701/2802/2903 and D40 seeds 6049515622977958931,8262941649267994858,2095371932330426401. Market heads have 164/724 fitted parameters; timing heads 82/690. Feature/target scales use permitted TRAIN data. Synthetic input scalers are fresh once per cohort and shared across initial/final; market-scaler panels use fresh market TRAIN statistics. Three head fits are readout variation on one encoder/split. V3/V4 native 36 versus later native 40 retains width/capacity/head-seed confounds. No PCA, head tuning, old replay or synthetic encoder training occurs.

Engineering passes 422697 checks/516 typed negative cases. Each saved initial/final causal proof passes 210135 checks with3685 CUDA calls over 128 cutoffs,2350 future-suffix and 30 internal cases, covering future values and masks. Typed resume 512 is exact. The independent TRAIN audit passes 570648 total checks and quality audit 13441, each with zero mismatches. All 13 current saved V11 DATA/statistic references are exact: eight feature archives compare native 20:40 with whole support/IDs/labels; five fresh-statistic archives compare whole. Learned0:20 remains unpaired and no whole40 initial equality is claimed.

The TRAIN evidence verifies all 75 identifiable tensors, eight mean/dispersion subgroups, four export/position groups and three between-bin directions. Actual1/2048 same-forward witnesses have 60 fields, including 25 temporal traces and four uniform weights/own-center arrays. Local adjoints cover export weight/bias/pool_positions using retained PRECLIP upstreams and the clip factor once. Common score bias and fixed-score hidden references are descriptive; no observed full-hidden gradient, score-to-hidden route, full encoder/global-norm or other 2046 raw-gradient reconstruction is claimed. Tensor-value capacity 294751 is not every-scalar learning proof.

The tables below are literal saved comparison sections: nine historical final controls and newly measured own initial/final rows, four separate market tasks, four timing panels, physical errors and readout repetitions. Historical scores/costs are reused unchanged; historical pass counts retain their own earlier cards. [DETAIL](CAUSAL_FRESH_V12_DETAIL.md) preserves the literal producer report tail with all current full-precision repetitions, physical errors, query/objective components and separate costs. [REVIEW](CAUSAL_FRESH_V12_REVIEW.md) explains the decision. Source data are the [curated table](results/causal_fresh_v12/table_rows.json) and [comparison](results/causal_fresh_v12/comparison.json), reported under [standard1.2](RESULTS_REPORTING_STANDARD.md).

Close the exact 2048 V12 recipe without coefficient/seed/rate/budget/head/checkpoint rescue. Preserve all earlier scoped leaders; the broad search remains active. The next proposal is a separately declared saved TRAIN-only V11/V12 diagnosis of moments, scales, native blocks, own saved readout contributions and reconstruction evidence before choosing V13. It must use its own bounded saved-artifact scope, with no old model/head replay, refit or held-out diagnostic tensor selection, and cannot establish a causal validation explanation. No next encoder or application replacement is admitted here.

No new experiment was run for this report finalization. All displayed measurements reuse the completed saved V12 initial/final evaluation and historical final controls. Documentary finalization executes no model, training update, head fit, extraction replay or scientific metric recomputation. The [final trust join](causal_fresh_v12/final-establishment.json) establishes measured-candidate/runtime/full-trust admission with 3/22 conditions and full coverage; advancement, overall winner, winner selection and application change remain false. Raw producer pending-trust fields and literal report tails remain byte-exact.

The [frozen V12 card](CAUSAL_FRESH_V12_CARD.json) binds the prospective source and fixed-head protocol. The [saved final inference checkpoint](../output/runs/causal-fresh-v12/run-001/checkpoints/inference-2048.pt) is 1305617 bytes, SHA256 `22ddbb83aab50c53552658abdb0d2de91781a2860329182c1811f546ac373543`. It remains a local ignored artifact. The [independent TRAIN audit](causal_fresh_v12/independent-training-audit.json), [independent quality audit](causal_fresh_v12/quality-independent-audit.json), [Complete mirror](causal_fresh_v12/complete-mirror-attempt-001.json) and [curated byte-copy receipt](causal_fresh_v12/durable-evidence-mirror.json) bind the exact saved evidence. This is one known chronological market split; no TEST or fresh future-period confirmation was measured.

---

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_volatility; V3/V4/V5/V6/V7/V8/V9/V10/V11 final rows reused, V12 initial/final newly measured.

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
| CPB-v12.market-01.init | 40 | -0.0905366176 | -0.190702149 | 100 |
| CPB-v12.market-01 | 40 | 0.908006596 | 0.516889349 | 100 |

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
| CPB-v12.market-01.init | 0.000279142026 | 0.000352805497 | 0.000307932803 | 0.000368533524 |
| CPB-v12.market-01 | 8.31887127e-05 | 0.000102469251 | 0.000185168421 | 0.000230100019 |

CPB-v3.market-01 neural R²2701/2802/2903: 0.344525499, 0.414672854, 0.629957415.
CPB-v4.market-01 neural R²2701/2802/2903: 0.275507022, 0.68909438, 0.700927285.
CPB-v5.market-01 neural R²2701/2802/2903: 0.539831415, 0.658428065, 0.746312576.
CPB-v6.market-01 neural R²2701/2802/2903: 0.634143112, 0.778464525, 0.341293674.
CPB-v7.market-01 neural R²2701/2802/2903: 0.701381536, 0.825073085, 0.493460306.
CPB-v8.market-01 neural R²2701/2802/2903: 0.78830625, 0.847167508, 0.57243398.
CPB-v9.market-01 neural R²2701/2802/2903: 0.756459839, 0.817281148, 0.281827888.
CPB-v10.market-01 neural R²2701/2802/2903: 0.720076151, 0.691819812, 0.441105176.
CPB-v11.market-01 neural R²2701/2802/2903: 0.267427319, 0.717651399, 0.628480632.
CPB-v12.market-01.init neural R²2701/2802/2903: -0.198180488, -0.260455718, -0.113470242.
CPB-v12.market-01 neural R²2701/2802/2903: 0.238112203, 0.614622651, 0.697933193.

## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_trend; V3/V4/V5/V6/V7/V8/V9/V10/V11 final rows reused, V12 initial/final newly measured.

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
| CPB-v12.market-01.init | 40 | -0.239031414 | -0.410515861 | 100 |
| CPB-v12.market-01 | 40 | 0.693216439 | 0.378337967 | 100 |

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
| CPB-v12.market-01.init | 5.40435626e-05 | 7.45384713e-05 | 5.86988095e-05 | 7.90543124e-05 |
| CPB-v12.market-01 | 2.98776048e-05 | 3.70898746e-05 | 4.01882356e-05 | 5.27403339e-05 |

CPB-v3.market-01 neural R²2701/2802/2903: -0.319245159, -0.237747895, -0.204943949.
CPB-v4.market-01 neural R²2701/2802/2903: 0.640552095, 0.358116367, 0.71095723.
CPB-v5.market-01 neural R²2701/2802/2903: 0.512587238, 0.442624323, 0.61657479.
CPB-v6.market-01 neural R²2701/2802/2903: 0.204803498, 0.321815119, 0.285489636.
CPB-v7.market-01 neural R²2701/2802/2903: 0.239513425, -0.194818323, 0.284144619.
CPB-v8.market-01 neural R²2701/2802/2903: 0.34873419, -0.875923336, 0.41496561.
CPB-v9.market-01 neural R²2701/2802/2903: 0.120397496, -0.818075045, 0.296655928.
CPB-v10.market-01 neural R²2701/2802/2903: 0.579763226, 0.459067372, 0.605860505.
CPB-v11.market-01 neural R²2701/2802/2903: 0.36674785, 0.537284886, 0.487273589.
CPB-v12.market-01.init neural R²2701/2802/2903: -0.148862928, -0.857110893, -0.225573762.
CPB-v12.market-01 neural R²2701/2802/2903: 0.456439928, 0.320628818, 0.357945156.

## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_activity; V3/V4/V5/V6/V7/V8/V9/V10/V11 final rows reused, V12 initial/final newly measured.

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
| CPB-v12.market-01.init | 40 | 0.570363128 | 0.530293021 | 100 |
| CPB-v12.market-01 | 40 | 0.88029496 | 0.694805907 | 100 |

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
| CPB-v12.market-01.init | 0.322314126 | 0.376687556 | 0.316510505 | 0.393357294 |
| CPB-v12.market-01 | 0.175147275 | 0.1988322 | 0.264633054 | 0.316578048 |

CPB-v3.market-01 neural R²2701/2802/2903: 0.551959766, 0.64999403, 0.42232437.
CPB-v4.market-01 neural R²2701/2802/2903: 0.868267037, 0.635226168, 0.558618431.
CPB-v5.market-01 neural R²2701/2802/2903: 0.777561455, 0.374143234, 0.702246639.
CPB-v6.market-01 neural R²2701/2802/2903: 0.19661957, 0.386097575, 0.386822848.
CPB-v7.market-01 neural R²2701/2802/2903: 0.253826868, 0.414419927, 0.389667307.
CPB-v8.market-01 neural R²2701/2802/2903: 0.617280912, 0.675654682, 0.669833502.
CPB-v9.market-01 neural R²2701/2802/2903: 0.567901982, 0.660584005, 0.581398567.
CPB-v10.market-01 neural R²2701/2802/2903: 0.540950734, 0.617987364, 0.309142288.
CPB-v11.market-01 neural R²2701/2802/2903: 0.641732517, 0.784916728, 0.770820483.
CPB-v12.market-01.init neural R²2701/2802/2903: 0.509294906, 0.486480011, 0.595104145.
CPB-v12.market-01 neural R²2701/2802/2903: 0.632617188, 0.707396224, 0.744404307.

## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_relation; V3/V4/V5/V6/V7/V8/V9/V10/V11 final rows reused, V12 initial/final newly measured.

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
| CPB-v12.market-01.init | 40 | -0.0162591027 | -0.133124988 | 100 |
| CPB-v12.market-01 | 40 | 0.54557653 | 0.397732658 | 100 |

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
| CPB-v12.market-01.init | 0.0720963673 | 0.0868876892 | 0.0742813854 | 0.091725589 |
| CPB-v12.market-01 | 0.0477718477 | 0.0581013736 | 0.0533119269 | 0.0667596203 |

CPB-v3.market-01 neural R²2701/2802/2903: -0.300654281, 0.161014634, -0.117057368.
CPB-v4.market-01 neural R²2701/2802/2903: 0.182210538, 0.23983439, 0.206486466.
CPB-v5.market-01 neural R²2701/2802/2903: 0.317728582, 0.48696529, 0.306822173.
CPB-v6.market-01 neural R²2701/2802/2903: 0.0570793218, 0.525771962, 0.385199543.
CPB-v7.market-01 neural R²2701/2802/2903: 0.115499431, 0.489241068, 0.369844732.
CPB-v8.market-01 neural R²2701/2802/2903: 0.356560851, -0.689937124, 0.450229932.
CPB-v9.market-01 neural R²2701/2802/2903: 0.466017712, 0.0430395951, 0.470592839.
CPB-v10.market-01 neural R²2701/2802/2903: 0.331177144, 0.359772869, 0.413664879.
CPB-v11.market-01 neural R²2701/2802/2903: 0.389406727, 0.504697432, 0.452219377.
CPB-v12.market-01.init neural R²2701/2802/2903: -0.117337205, -0.20060374, -0.0814340189.
CPB-v12.market-01 neural R²2701/2802/2903: 0.290003339, 0.449589084, 0.453605552.

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact; V3/V4/V5/V6/V7/V8/V9/V10/V11 reused.

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
| CPB-v12.market-01.init | 40 | 93.6197917 | 92.2309028 | 100 |
| CPB-v12.market-01 | 40 | 98.4375 | 97.7430556 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted; V3/V4/V5/V6/V7/V8/V9/V10/V11 reused.

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
| CPB-v12.market-01.init | 40 | 87.890625 | 86.4149306 | 100 |
| CPB-v12.market-01 | 40 | 94.140625 | 93.5329861 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact; V3/V4/V5/V6/V7/V8/V9/V10/V11 reused.

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
| CPB-v12.market-01.init | 40 | 95.4427083 | 94.9652778 | 100 |
| CPB-v12.market-01 | 40 | 99.7395833 | 97.7430556 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted; V3/V4/V5/V6/V7/V8/V9/V10/V11 reused.

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
| CPB-v12.market-01.init | 40 | 86.5885417 | 84.7222222 | 100 |
| CPB-v12.market-01 | 40 | 93.75 | 93.0121528 | 100 |

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
| CPB-v12.market-01.init | 0 | 0.664078026 | 0.957987333 | 0 |
| CPB-v12.market-01 | 2048 | 0.252382241 | 0.315910852 | 184.812345 |

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
| CPB-v12.market-01.init | 0.423736884 | 0.795548535 |
| CPB-v12.market-01 | -0.374763669 | 0.297063679 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. All22 frozen numeric conditions; trust and coverage are separate.

| Dataset | Condition | Candidate | Frozen floor/ceiling | Source | Rule | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| MKT-HIST-2 | past_volatility:linear_r2 | 0.908006596 | 0.964673434 | CPB-v3.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_volatility:neural_mean_r2 | 0.516889349 | 0.735969246 | CPB-v8.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:linear_r2 | 0.693216439 | 0.90721343 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_trend:neural_mean_r2 | 0.378337967 | 0.569875231 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:linear_r2 | 0.88029496 | 0.909438776 | CPB-v11.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_activity:neural_mean_r2 | 0.694805907 | 0.732489909 | CPB-v11.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:linear_r2 | 0.54557653 | 0.622100687 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | past_relation:neural_mean_r2 | 0.397732658 | 0.448774512 | CPB-v11.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_intact:linear_head_accuracy | 0.984375 | 0.98828125 | CPB-v11.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_intact:neural_head_accuracy | 0.977430556 | 0.984809028 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:linear_head_accuracy | 0.94140625 | 0.955729167 | CPB-v9.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | synthetic_train:validation_deleted:neural_head_accuracy | 0.935329861 | 0.942708333 | CPB-v11.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_intact:linear_head_accuracy | 0.997395833 | 0.998697917 | CPB-v9.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_intact:neural_head_accuracy | 0.977430556 | 0.992621528 | CPB-v8.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:linear_head_accuracy | 0.9375 | 0.97265625 | CPB-v6.market-01 | candidate + 1e-12 >= floor | FAIL |
| TEMPO-5 | market_scaler:validation_deleted:neural_head_accuracy | 0.930121528 | 0.956163194 | CPB-v4.market-01 | candidate + 1e-12 >= floor | FAIL |
| MKT-HIST-2 | train:standardized_MAE | 0.252382241 | 0.241805338 | CPB-v7.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | validation:standardized_MAE | 0.315910852 | 0.308316924 | CPB-v4.market-01 | candidate <= floor + 1e-12 | FAIL |
| MKT-HIST-2 | known_validation:Gaussian_NLL_F64 | 0.297063679 | -0.0453814881 | CPB-v4.market-01 | candidate < floor - 1e-12 | FAIL |
| MKT-HIST-2 | past_trend:all3_neural_R2_positive | 0.456439928, 0.320628818, 0.357945156 | 0 | strict positivity | all three final neural trend R2 values > 0 | PASS |
| MKT-HIST-2 | past_relation:linear_r2:positive | 0.54557653 | 0 | strict positivity | candidate > 0 | PASS |
| MKT-HIST-2 | past_relation:neural_mean_r2:positive | 0.397732658 | 0 | strict positivity | candidate > 0 | PASS |

The exact declared trajectory is stopped at2048. Retain every failed condition and all nine references and their scoped strengths. Further mechanisms require separate prospective declarations; no coefficient/seed/rate/budget/head rescue. V3's reported144s training cost is reused historical evidence,not a new matched benchmark.

