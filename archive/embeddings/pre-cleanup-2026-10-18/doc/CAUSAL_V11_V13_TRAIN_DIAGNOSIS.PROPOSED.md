<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root. -->
# PROPOSED saved TRAIN diagnosis: CPB-v11 and CPB-v13

**Independent numerical audit: 73,778 checks, zero mismatches. Documentary completion pending. No new encoder experiment.** One new V13 saved-arithmetic producer ran once; four existing audited V11 objects were reused opaquely. No model/encoder/decoder/head/module forward, fit or optimizer update occurred. This proposed report is not an application change or a candidate admission.

Final TRAIN results are mixed. V13 has slightly lower R² than V11 for all four linear tasks and neural volatility/activity/relation; neural trend R² is slightly higher. Neural activity MAE improves despite its lower R². Fixed-query TRAIN NLL improves while MAE worsens in both chronological halves. These different measures do not identify why known-VALIDATION performance changed. V3 remains the clean baseline and sole DEV encoder. V13's completed quality study and five scoped records remain preserved; this diagnosis does not rescore them or declare an overall winner. After the audited arithmetic, root separately selected CPB-v14.market-01 for DESIGN ONLY; no source preparation, training or quality execution is admitted by this report.

MKT-HIST-2 is one joined BTC/ETH/SOL market panel, C3/H128/F9: 112 chronological TRAIN windows and32 known chronological VALIDATION windows per historical run. These tasks retain past-window properties; they are not forecasts or three independent market cohorts. Numerical diagnosis uses only saved TRAIN112, its fixed first-eight reconstruction queries, one predeclared moment CSV and two actual training captures. No VALIDATION diagnostic tensors, TEST, stress or TEMPO numeric artifacts are opened. Complexity4/5 is an ordinal designed challenge, not measured difficulty.

Initial/final are saved updates0/2048. Each encoder has one historical fresh CUDA trajectory, batch8: all294752 learned values, optimizer and TRAIN statistics start fresh, source_updates0, no trained ancestor. V11 provides V13's scientific SOURCE, not weights. Native40 is learned20 + literal spectral12 + mean4 + independent uncertainty4. Intermediate temporal dispersion, native uncertainty4 and decoder conditional variance are distinct.

Heads were fitted previously and are unchanged: ridge penalty1 (164 fitted values); neural16 tanh hidden units/four outputs, Adam0.01/100 updates (724 values), permitted TRAIN-only normalization. Repetitions2701/2802/2903 use D40 seeds6049515622977958931,8262941649267994858,2095371932330426401. They are readout variation on one encoder/split, not encoder retraining. No best seed/checkpoint, PCA, post-encoder transform, head tuning or old head replay is selected.

V11 projects mean4 and raw centered-variance4 under half-local/half-global attention. V13 keeps that attention and mean4, but projects sqrt(v + 2^-10) - 1/32 in actual F32 add/sqrt/subtract order. Full hidden/center/attention-score/position routes and finite-F32 residual correction remain live. Raw root at zero variance is1/32; dispersion is0; absent local bins zero the whole served eight rows after bias. Shapes/counts, objective, globalclip1/singleAdamW, streams,2048/B8 and heads stay fixed. Existing causal/GPU/typed-resume/local-adjoint trust is reused, not replayed or extended here.

Every table copies saved scalars, formatted to ten significant digits. No metric, half-repetition mean, tensor reduction or CSV reduction is recomputed for this report. Full precision, all row IDs/coordinate descriptors, physical errors, signed contributions, quantiles and associations remain in the [saved JSON](causal_v11_v13_train_diagnosis/diagnosis.json). Its dataset/recipe/view metadata applies throughout; exact record paths below identify each panel. Large row arrays are not duplicated.

## Own saved TRAIN task readouts

Each checkpoint uses its own fitted head/normalizer. Neural cells show the saved mean and all three repetitions, ordered2701/2802/2903. R² uses the scored TRAIN population mean; it is neither accuracy nor generalization. Initial/final suffixes describe saved states of registered instances.

## Past volatility

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns · **complexity4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R²: mean; [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial | 40 | 0.8704379066 | 0.9304516397; [0.9480124223, 0.9248865428, 0.918455954] | 100 (112/112) |
| CPB-v13.market-01 · initial | 40 | 0.869168419 | 0.9256083333; [0.9467354945, 0.9129487678, 0.9171407377] | 100 (112/112) |
| CPB-v11.market-01 · final | 40 | 0.9915892992 | 0.9535589501; [0.9397402825, 0.9648279068, 0.956108661] | 100 (112/112) |
| CPB-v13.market-01 · final | 40 | 0.9914206895 | 0.9518350959; [0.9373938953, 0.9673894965, 0.9507218959] | 100 (112/112) |

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns; physical errors in **logreturn** · **complexity4/5, ordinal** · same saved TRAIN112 heads.

| Method | Linear MAE | Linear RMSE | Neural MAE: mean; [three reps] | Neural RMSE: mean; [three reps] |
| --- | ---: | ---: | --- | --- |
| CPB-v11.market-01 · initial | 0.0001387027591 | 0.0001811739557 | 9.516026118E-05; [8.606292762E-05, 9.910184037E-05, 0.0001003160156] | 0.0001321480457; [0.0001147642386, 0.0001379480812, 0.0001437318172] |
| CPB-v13.market-01 · initial | 0.0001393650378 | 0.00018205939 | 9.468150727E-05; [8.961859454E-05, 9.892833938E-05, 9.549758789E-05] | 0.0001365191675; [0.0001161651181, 0.0001485060853, 0.0001448862993] |
| CPB-v11.market-01 · final | 3.717972113E-05 | 4.616075248E-05 | 7.910417677E-05; [8.997031077E-05, 6.831640848E-05, 7.902581107E-05] | 0.0001078014187; [0.0001235578476, 9.43965009E-05, 0.0001054499075] |
| CPB-v13.market-01 · final | 3.767095469E-05 | 4.662114992E-05 | 8.063603731E-05; [9.181370388E-05, 6.7143436E-05, 8.295097205E-05] | 0.0001095226946; [0.0001259404178, 9.089406797E-05, 0.0001117335979] |

## Past trend

Dataset: **MKT-HIST-2** · channel-mean mean supported adjacent close log return · **complexity4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R²: mean; [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial | 40 | 0.6937123027 | 0.9444130642; [0.9472332648, 0.93154477, 0.9544611578] | 100 (112/112) |
| CPB-v13.market-01 · initial | 40 | 0.6937168549 | 0.9465431465; [0.9535646333, 0.9296175861, 0.9564472199] | 100 (112/112) |
| CPB-v11.market-01 · final | 40 | 0.9110286797 | 0.9382186199; [0.9409886975, 0.9277573259, 0.9459098363] | 100 (112/112) |
| CPB-v13.market-01 · final | 40 | 0.9081461461 | 0.9393749861; [0.9453281054, 0.9247757207, 0.9480211321] | 100 (112/112) |

Dataset: **MKT-HIST-2** · channel-mean mean supported adjacent close log return; physical errors in **logreturn/tick** · **complexity4/5, ordinal** · same saved TRAIN112 heads.

| Method | Linear MAE | Linear RMSE | Neural MAE: mean; [three reps] | Neural RMSE: mean; [three reps] |
| --- | ---: | ---: | --- | --- |
| CPB-v11.market-01 · initial | 3.566489438E-05 | 4.620896957E-05 | 1.439798006E-05; [1.313720415E-05, 1.619516159E-05, 1.386157443E-05] | 1.961435648E-05; [1.917969161E-05, 2.184564262E-05, 1.781773519E-05] |
| CPB-v13.market-01 · initial | 3.56654687E-05 | 4.620862618E-05 | 1.464838567E-05; [1.347858393E-05, 1.714881893E-05, 1.331775416E-05] | 1.918938276E-05; [1.799226949E-05, 2.215101272E-05, 1.742486608E-05] |
| CPB-v11.market-01 · final | 1.887022258E-05 | 2.490499881E-05 | 1.483424212E-05; [1.457169087E-05, 1.535159796E-05, 1.457943752E-05] | 2.071447203E-05; [2.028285603E-05, 2.244183756E-05, 1.94187225E-05] |
| CPB-v13.market-01 · final | 1.921171072E-05 | 2.530522476E-05 | 1.479970476E-05; [1.451336126E-05, 1.547009749E-05, 1.441565552E-05] | 2.048636585E-05; [1.952286584E-05, 2.290026592E-05, 1.90359658E-05] |

## Past activity

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume) · **complexity4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R²: mean; [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial | 40 | 0.9264675744 | 0.964113309; [0.9604736197, 0.9565085167, 0.9753577905] | 100 (112/112) |
| CPB-v13.market-01 · initial | 40 | 0.9306970294 | 0.9607143618; [0.9650512284, 0.9495764117, 0.9675154454] | 100 (112/112) |
| CPB-v11.market-01 · final | 40 | 0.9676749422 | 0.9699659773; [0.9666138639, 0.9702249784, 0.9730590895] | 100 (112/112) |
| CPB-v13.market-01 · final | 40 | 0.9675496527 | 0.9697152305; [0.9654435501, 0.9677484956, 0.9759536457] | 100 (112/112) |

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume); physical errors in **log1p(base_volume)** · **complexity4/5, ordinal** · same saved TRAIN112 heads.

| Method | Linear MAE | Linear RMSE | Neural MAE: mean; [three reps] | Neural RMSE: mean; [three reps] |
| --- | ---: | ---: | --- | --- |
| CPB-v11.market-01 · initial | 0.132608976 | 0.1683792272 | 0.09089358311; [0.09674711797, 0.1037722261, 0.07216140524] | 0.116806301; [0.1234504247, 0.129494456, 0.09747402243] |
| CPB-v13.market-01 · initial | 0.126961136 | 0.1634650799 | 0.09253486418; [0.08735922466, 0.1085317766, 0.08171359129] | 0.1224766521; [0.1160820377, 0.1394331182, 0.1119148003] |
| CPB-v11.market-01 · final | 0.08901114748 | 0.1116397154 | 0.08690566221; [0.09054964755, 0.08762217514, 0.08254516395] | 0.1075073407; [0.1134572219, 0.1071457892, 0.101919011] |
| CPB-v13.market-01 · final | 0.08902993795 | 0.1118558598 | 0.08557162548; [0.09049170094, 0.09046235327, 0.07576082222] | 0.1077432061; [0.1154286524, 0.1115126289, 0.09628833705] |

## Past relation

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns · **complexity4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R²: mean; [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial | 40 | 0.6710415617 | 0.8933798703; [0.8962817312, 0.9124124261, 0.8714454537] | 100 (112/112) |
| CPB-v13.market-01 · initial | 40 | 0.6807749653 | 0.896028301; [0.9013159269, 0.9299680008, 0.8568009753] | 100 (112/112) |
| CPB-v11.market-01 · final | 40 | 0.7345649307 | 0.8704845323; [0.8593115988, 0.8898075441, 0.862334454] | 100 (112/112) |
| CPB-v13.market-01 · final | 40 | 0.7304930438 | 0.864862085; [0.8579062991, 0.8835814476, 0.8530985083] | 100 (112/112) |

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns; physical errors in **correlation** · **complexity4/5, ordinal** · same saved TRAIN112 heads.

| Method | Linear MAE | Linear RMSE | Neural MAE: mean; [three reps] | Neural RMSE: mean; [three reps] |
| --- | ---: | ---: | --- | --- |
| CPB-v11.market-01 · initial | 0.04625253306 | 0.05825764237 | 0.02526688974; [0.02456989577, 0.02334426332, 0.02788651013] | 0.03306402499; [0.03271222006, 0.03006101484, 0.03641884007] |
| CPB-v13.market-01 · initial | 0.04550265122 | 0.05738929121 | 0.02488890878; [0.02352616809, 0.02144470897, 0.02969584927] | 0.03240860382; [0.03190846576, 0.02688008587, 0.03843725984] |
| CPB-v11.market-01 · final | 0.04309224556 | 0.05233130202 | 0.0284738675; [0.02879099296, 0.02699234473, 0.0296382648] | 0.03650128335; [0.03809882103, 0.03371772939, 0.03768729962] |
| CPB-v13.market-01 · final | 0.04364231971 | 0.05273116655 | 0.02898336391; [0.02959278471, 0.0277922184, 0.02956508862] | 0.03729227433; [0.03828862783, 0.03465720115, 0.03893099401] |

Exact paths: cached_v11_TRAIN.stages.{initial,final} and models.v13.stages.{initial,final}, then per_task_mean_across_saved_repetitions and saved_TRAIN_head_scores[*].tasks. Cached V11 was not re-reduced.

## Chronological TRAIN halves

Early56/late56 are fixed slices scored with the existing full-TRAIN fitted heads, not half-specific fits or held-out tests. Each half has its own R² denominator. Neural cells show all three saved R² values; no unrecorded half mean is manufactured. Half physical MAE/RMSE and own ridge decomposition records remain beside these scores at saved_TRAIN_head_scores[*].chronological_TRAIN_halves.

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns · **complexity4/5, ordinal** · early56/late56 TRAIN, full-TRAIN fit.

| Method · half | Size | Linear head R² | Neural head R²: [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial · early56 | 40 | 0.8397487093 | [0.941384225, 0.9082501405, 0.9119655267] | 100 (56/56) |
| CPB-v11.market-01 · initial · late56 | 40 | 0.9016060089 | [0.9496792335, 0.9407531898, 0.9136240004] | 100 (56/56) |
| CPB-v13.market-01 · initial · early56 | 40 | 0.8374243876 | [0.9441791999, 0.8904654331, 0.9025606801] | 100 (56/56) |
| CPB-v13.market-01 · initial · late56 | 40 | 0.902079558 | [0.9403704026, 0.9374429773, 0.9274529764] | 100 (56/56) |
| CPB-v11.market-01 · final · early56 | 40 | 0.9914083517 | [0.9279915956, 0.960409621, 0.9563455124] | 100 (56/56) |
| CPB-v11.market-01 · final · late56 | 40 | 0.9901596732 | [0.9494232132, 0.96582975, 0.9463964942] | 100 (56/56) |
| CPB-v13.market-01 · final · early56 | 40 | 0.9912165401 | [0.9301735892, 0.9627008217, 0.9490795969] | 100 (56/56) |
| CPB-v13.market-01 · final · late56 | 40 | 0.9899997195 | [0.9379490695, 0.969447267, 0.9434554979] | 100 (56/56) |

Dataset: **MKT-HIST-2** · channel-mean mean supported adjacent close log return · **complexity4/5, ordinal** · early56/late56 TRAIN, full-TRAIN fit.

| Method · half | Size | Linear head R² | Neural head R²: [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial · early56 | 40 | 0.6893365576 | [0.9385002326, 0.9366415947, 0.9573315882] | 100 (56/56) |
| CPB-v11.market-01 · initial · late56 | 40 | 0.6721945724 | [0.9550299564, 0.9179774143, 0.9461855929] | 100 (56/56) |
| CPB-v13.market-01 · initial · early56 | 40 | 0.6974597162 | [0.9533381597, 0.9329752586, 0.9552180582] | 100 (56/56) |
| CPB-v13.market-01 · initial · late56 | 40 | 0.660494803 | [0.949672479, 0.9183825109, 0.9542625421] | 100 (56/56) |
| CPB-v11.market-01 · final · early56 | 40 | 0.9258787108 | [0.9429869459, 0.9129259862, 0.9500132081] | 100 (56/56) |
| CPB-v11.market-01 · final · late56 | 40 | 0.8815360655 | [0.932746596, 0.9425766586, 0.9350798287] | 100 (56/56) |
| CPB-v13.market-01 · final · early56 | 40 | 0.9200332461 | [0.9478497677, 0.910366809, 0.9539685355] | 100 (56/56) |
| CPB-v13.market-01 · final · late56 | 40 | 0.882663362 | [0.9367256253, 0.9387151521, 0.9347243641] | 100 (56/56) |

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume) · **complexity4/5, ordinal** · early56/late56 TRAIN, full-TRAIN fit.

| Method · half | Size | Linear head R² | Neural head R²: [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial · early56 | 40 | 0.912024115 | [0.9347167787, 0.9383068735, 0.965591328] | 100 (56/56) |
| CPB-v11.market-01 · initial · late56 | 40 | 0.9280804719 | [0.9741759043, 0.96435268, 0.9794123916] | 100 (56/56) |
| CPB-v13.market-01 · initial · early56 | 40 | 0.9162420037 | [0.9423285453, 0.9262930144, 0.9465451018] | 100 (56/56) |
| CPB-v13.market-01 · initial · late56 | 40 | 0.932818032 | [0.9771300935, 0.960226386, 0.9786355081] | 100 (56/56) |
| CPB-v11.market-01 · final · early56 | 40 | 0.9562909474 | [0.9476537109, 0.9581707739, 0.9660828515] | 100 (56/56) |
| CPB-v11.market-01 · final · late56 | 40 | 0.9719754216 | [0.9761934022, 0.9753049264, 0.9748516107] | 100 (56/56) |
| CPB-v13.market-01 · final · early56 | 40 | 0.9560823174 | [0.9513853174, 0.9547581243, 0.9689179629] | 100 (56/56) |
| CPB-v13.market-01 · final · late56 | 40 | 0.9718947758 | [0.9713879853, 0.9732035784, 0.9781306624] | 100 (56/56) |

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns · **complexity4/5, ordinal** · early56/late56 TRAIN, full-TRAIN fit.

| Method · half | Size | Linear head R² | Neural head R²: [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial · early56 | 40 | 0.6814440561 | [0.896761562, 0.9148278746, 0.9078366312] | 100 (56/56) |
| CPB-v11.market-01 · initial · late56 | 40 | 0.6307338162 | [0.8862184394, 0.902014932, 0.8251517632] | 100 (56/56) |
| CPB-v13.market-01 · initial · early56 | 40 | 0.7003319813 | [0.9019014893, 0.9324253428, 0.8845069179] | 100 (56/56) |
| CPB-v13.market-01 · initial · late56 | 40 | 0.6327198263 | [0.8916191794, 0.9211575253, 0.8173562617] | 100 (56/56) |
| CPB-v11.market-01 · final · early56 | 40 | 0.7490639695 | [0.8875837342, 0.8911551919, 0.887649928] | 100 (56/56) |
| CPB-v11.market-01 · final · late56 | 40 | 0.6962724651 | [0.8195645638, 0.8783244734, 0.8256609672] | 100 (56/56) |
| CPB-v13.market-01 · final · early56 | 40 | 0.7441479718 | [0.8761953529, 0.8801338301, 0.8755643775] | 100 (56/56) |
| CPB-v13.market-01 · final · late56 | 40 | 0.6926208202 | [0.8274611196, 0.8760520914, 0.8182615913] | 100 (56/56) |

## Fixed-query TRAIN reconstruction

Huber(delta1) mean reconstruction and detached-error Gaussian variance calibration are the optimization routes; reporting Gaussian NLL is a distinct measure. Lower MAE and lower NLL need not agree. These are fixed-query scores, not final randomly masked minibatch losses. No VALIDATION reconstruction is added by this TRAIN-only diagnosis.

Dataset: **MKT-HIST-2** · fixed hidden-target query, standardized feature units · **complexity4/5, ordinal** · all112 TRAIN rows.

| Method/state | Standardized MAE ↓ | Gaussian NLL ↓ | Mean Huber1 | Total objective |
| --- | ---: | ---: | ---: | ---: |
| CPB-v11.market-01 · initial | 0.6640733605 | 0.4237330498 | 0.3236635132 | 0.747396563 |
| CPB-v13.market-01 · initial | 0.6640682557 | 0.4237328304 | 0.323665619 | 0.7473984494 |
| CPB-v11.market-01 · final | 0.2517975124 | -0.369659381 | 0.08466702545 | -0.2849923556 |
| CPB-v13.market-01 · final | 0.2546742454 | -0.3716349322 | 0.08498626473 | -0.2866486674 |

Dataset: **MKT-HIST-2** · same fixed query, standardized units · **complexity4/5, ordinal** · early56/late56 TRAIN.

| Method/state · half | Rows | Standardized MAE ↓ | Gaussian NLL ↓ | Mean Huber1 |
| --- | ---: | ---: | ---: | ---: |
| CPB-v11.market-01 · initial · early56 | 56 | 0.68598546 | 0.4564872877 | 0.3535453735 |
| CPB-v11.market-01 · initial · late56 | 56 | 0.642161261 | 0.3909788118 | 0.293781653 |
| CPB-v13.market-01 · initial · early56 | 56 | 0.6859780815 | 0.4564814286 | 0.3535461198 |
| CPB-v13.market-01 · initial · late56 | 56 | 0.6421584299 | 0.3909842321 | 0.2937851183 |
| CPB-v11.market-01 · final · early56 | 56 | 0.2716146277 | -0.159891757 | 0.09432302765 |
| CPB-v11.market-01 · final · late56 | 56 | 0.2319803972 | -0.579427005 | 0.07501102325 |
| CPB-v13.market-01 · final · early56 | 56 | 0.275712625 | -0.1632356276 | 0.09490040223 |
| CPB-v13.market-01 · final · late56 | 56 | 0.2336358658 | -0.5800342367 | 0.07507212723 |

Dataset: **MKT-HIST-2** · fixed first-eight TRAIN queries, standardized units · **complexity4/5, ordinal** · B8 detail, not the112-row score.

| Method/state | Rows / target cells | Standardized MAE ↓ | Gaussian NLL ↓ | Mean Huber1 |
| --- | ---: | ---: | ---: | ---: |
| CPB-v11.market-01 · initial | 8 / 1728 | 0.7936131665 | 0.5939645186 | 0.4315045196 |
| CPB-v13.market-01 · initial | 8 / 1728 | 0.7936023193 | 0.5939488985 | 0.4315014742 |
| CPB-v11.market-01 · final | 8 / 1728 | 0.2630482911 | -0.3107464577 | 0.1043249677 |
| CPB-v13.market-01 · final | 8 / 1728 | 0.2664559977 | -0.3254297291 | 0.1034630219 |

Exact paths: each stage's fixed_query_TRAIN_diagnosis.{all112_TRAIN,per_channel_feature} and saved_mean_residual_allocation. The latter contracts only the saved same-stage mean4 decoder, without a decoder call or weight swapping. Per-channel/feature residual/inverse-variance/Huber/NLL summaries retain their declared B8 or112 population. Reconstruction_readout_associations preserves existing signed/absolute residual correlations for all112/halves; association is not gradient conflict or a validation cause.

## Native blocks

Raw population covariance trace describes coordinate scale. Participation ratio is trace(C)^2/sum(C_ij^2); it is not rank, importance or information content. No PCA/eigensolver or fitted transform occurs. Each own head's TRAIN normalizer can cancel simple multiplicative scale; nonlinear transforms, feature mixing and correlation need not cancel.

Dataset: **MKT-HIST-2** · four native blocks and complete40 · **complexity4/5, ordinal** · all112 TRAIN.

| Method/state | Block | Width | Covariance trace | Participation ratio |
| --- | --- | ---: | ---: | ---: |
| CPB-v11.market-01 · initial | learned20 | 20 | 0.006245551576 | 3.24406852 |
| CPB-v11.market-01 · initial | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v11.market-01 · initial | mean4 | 4 | 1.122570926 | 3.659153602 |
| CPB-v11.market-01 · initial | uncertainty4 | 4 | 1.265117761 | 3.399078554 |
| CPB-v11.market-01 · initial | complete40 | 40 | 3.923562637 | 11.73760173 |
| CPB-v13.market-01 · initial | learned20 | 20 | 0.006110571507 | 3.128690877 |
| CPB-v13.market-01 · initial | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v13.market-01 · initial | mean4 | 4 | 1.122570926 | 3.659153602 |
| CPB-v13.market-01 · initial | uncertainty4 | 4 | 1.265117761 | 3.399078554 |
| CPB-v13.market-01 · initial | complete40 | 40 | 3.923427657 | 11.73686196 |
| CPB-v11.market-01 · final | learned20 | 20 | 184.8944722 | 3.276767195 |
| CPB-v11.market-01 · final | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v11.market-01 · final | mean4 | 4 | 23.0285827 | 3.241013044 |
| CPB-v11.market-01 · final | uncertainty4 | 4 | 78.78198941 | 1.048875014 |
| CPB-v11.market-01 · final | complete40 | 40 | 288.2346727 | 3.941626973 |
| CPB-v13.market-01 · final | learned20 | 20 | 188.6700513 | 3.323343105 |
| CPB-v13.market-01 · final | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v13.market-01 · final | mean4 | 4 | 22.77782796 | 3.241542867 |
| CPB-v13.market-01 · final | uncertainty4 | 4 | 77.84594671 | 1.048878752 |
| CPB-v13.market-01 · final | complete40 | 40 | 290.8234544 | 3.999668805 |

Dataset: **MKT-HIST-2** · four native blocks and complete40 · **complexity4/5, ordinal** · V13 early56/late56 TRAIN. V11 has no cached native-half counterpart.

| V13 state · half | Block | Width | Covariance trace | Participation ratio |
| --- | --- | ---: | ---: | ---: |
| initial · early56 | learned20 | 20 | 0.004578678529 | 3.189163417 |
| initial · early56 | fixed_spectral12 | 12 | 1.520660418 | 5.938095454 |
| initial · early56 | mean4 | 4 | 1.287138195 | 3.53768253 |
| initial · early56 | uncertainty4 | 4 | 1.427203853 | 3.27316786 |
| initial · early56 | complete40 | 40 | 4.239581144 | 9.737852862 |
| initial · late56 | learned20 | 20 | 0.002505290253 | 3.156533384 |
| initial · late56 | fixed_spectral12 | 12 | 1.520371612 | 7.140454679 |
| initial · late56 | mean4 | 4 | 0.9427055605 | 3.556506007 |
| initial · late56 | uncertainty4 | 4 | 1.033023726 | 3.510710201 |
| initial · late56 | complete40 | 40 | 3.498606189 | 11.76664694 |
| final · early56 | learned20 | 20 | 162.3990239 | 2.677198917 |
| final · early56 | fixed_spectral12 | 12 | 1.520660418 | 5.938095454 |
| final · early56 | mean4 | 4 | 28.28704326 | 2.843357638 |
| final · early56 | uncertainty4 | 4 | 93.98442048 | 1.041481476 |
| final · early56 | complete40 | 40 | 286.1911481 | 3.66159234 |
| final · late56 | learned20 | 20 | 114.622809 | 2.233966068 |
| final · late56 | fixed_spectral12 | 12 | 1.520371612 | 7.140454679 |
| final · late56 | mean4 | 4 | 16.19202934 | 3.020689701 |
| final · late56 | uncertainty4 | 4 | 54.36767385 | 1.065288218 |
| final · late56 | complete40 | 40 | 186.7028838 | 2.559713511 |

All40 coordinate means/population scales/minima/maxima and covariance trace/Frobenius/participation descriptors remain at native_features.{coordinates,blocks}; V13 half records add chronological_TRAIN_halves.{early56,late56}. No cached V11 halves are fabricated. V13 final learned20 trace188.6700513/participation3.323343105 and low uncertainty4 participation describe concentrated covariance; they do not show a dead branch or dispensability.

## Own saved ridge contributions

This is an exact decomposition using the same version/stage's saved ridge weights, normalizer and intercept, not an ablation predictor, refit or neural attribution. Components can cancel or correlate; their magnitudes do not define unique importance. Final-state tables show physical-unit contribution STD and correlation to the same saved residual, all three repetitions. Initial/full112/early56/late56 signed contributions, norms, intercepts and pairwise covariance remain at saved_TRAIN_head_scores[*].saved_linear_readout_block_decomposition.<task>. No model's head weights are swapped into another representation.

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns; contribution STD in **logreturn** · **complexity4/5, ordinal** · final TRAIN112, own ridge; reps2701/2802/2903.

| Method | Block | Size | Contribution STD: [three reps] | Correlation to own residual: [three reps] |
| --- | --- | ---: | --- | --- |
| CPB-v11.market-01 | learned20 | 20 | [6.953642371E-05, 6.953642371E-05, 6.953642371E-05] | [-0.04490087667, -0.04490087667, -0.04490087667] |
| CPB-v11.market-01 | fixed_spectral12 | 12 | [2.457638087E-05, 2.457638087E-05, 2.457638087E-05] | [-0.004832993463, -0.004832993463, -0.004832993463] |
| CPB-v11.market-01 | mean4 | 4 | [0.0001629035127, 0.0001629035127, 0.0001629035127] | [-0.02362524006, -0.02362524006, -0.02362524006] |
| CPB-v11.market-01 | uncertainty4 | 4 | [0.0004022723643, 0.0004022723643, 0.0004022723643] | [-0.02181488879, -0.02181488879, -0.02181488879] |
| CPB-v13.market-01 | learned20 | 20 | [6.645585177E-05, 6.645585177E-05, 6.645585177E-05] | [-0.04829906638, -0.04829906638, -0.04829906638] |
| CPB-v13.market-01 | fixed_spectral12 | 12 | [2.57991168E-05, 2.57991168E-05, 2.57991168E-05] | [-0.005231105921, -0.005231105921, -0.005231105921] |
| CPB-v13.market-01 | mean4 | 4 | [0.0001510482093, 0.0001510482093, 0.0001510482093] | [-0.0221795775, -0.0221795775, -0.0221795775] |
| CPB-v13.market-01 | uncertainty4 | 4 | [0.0004110337114, 0.0004110337114, 0.0004110337114] | [-0.02177561675, -0.02177561675, -0.02177561675] |

Dataset: **MKT-HIST-2** · channel-mean mean supported adjacent close log return; contribution STD in **logreturn/tick** · **complexity4/5, ordinal** · final TRAIN112, own ridge; reps2701/2802/2903.

| Method | Block | Size | Contribution STD: [three reps] | Correlation to own residual: [three reps] |
| --- | --- | ---: | --- | --- |
| CPB-v11.market-01 | learned20 | 20 | [1.000414076E-05, 1.000414076E-05, 1.000414076E-05] | [-0.2312025144, -0.2312025144, -0.2312025144] |
| CPB-v11.market-01 | fixed_spectral12 | 12 | [1.187122916E-05, 1.187122916E-05, 1.187122916E-05] | [-0.007340830887, -0.007340830887, -0.007340830887] |
| CPB-v11.market-01 | mean4 | 4 | [7.573073952E-05, 7.573073952E-05, 7.573073952E-05] | [-0.05812313017, -0.05812313017, -0.05812313017] |
| CPB-v11.market-01 | uncertainty4 | 4 | [6.607119845E-06, 6.607119845E-06, 6.607119845E-06] | [-0.05982904835, -0.05982904835, -0.05982904835] |
| CPB-v13.market-01 | learned20 | 20 | [8.997906176E-06, 8.997906176E-06, 8.997906176E-06] | [-0.1965488451, -0.1965488451, -0.1965488451] |
| CPB-v13.market-01 | fixed_spectral12 | 12 | [1.187324467E-05, 1.187324467E-05, 1.187324467E-05] | [-0.00699819164, -0.00699819164, -0.00699819164] |
| CPB-v13.market-01 | mean4 | 4 | [7.739399915E-05, 7.739399915E-05, 7.739399915E-05] | [-0.05764803218, -0.05764803218, -0.05764803218] |
| CPB-v13.market-01 | uncertainty4 | 4 | [5.777991927E-06, 5.777991927E-06, 5.777991927E-06] | [-0.0673834965, -0.0673834965, -0.0673834965] |

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume); contribution STD in **log1p(base_volume)** · **complexity4/5, ordinal** · final TRAIN112, own ridge; reps2701/2802/2903.

| Method | Block | Size | Contribution STD: [three reps] | Correlation to own residual: [three reps] |
| --- | --- | ---: | --- | --- |
| CPB-v11.market-01 | learned20 | 20 | [0.342309534, 0.342309534, 0.342309534] | [-0.02824333504, -0.02824333504, -0.02824333504] |
| CPB-v11.market-01 | fixed_spectral12 | 12 | [0.03854826198, 0.03854826198, 0.03854826198] | [-0.006547193776, -0.006547193776, -0.006547193776] |
| CPB-v11.market-01 | mean4 | 4 | [0.3185033448, 0.3185033448, 0.3185033448] | [-0.02462481799, -0.02462481799, -0.02462481799] |
| CPB-v11.market-01 | uncertainty4 | 4 | [0.06962693766, 0.06962693766, 0.06962693766] | [-0.04684463955, -0.04684463955, -0.04684463955] |
| CPB-v13.market-01 | learned20 | 20 | [0.3401359429, 0.3401359429, 0.3401359429] | [-0.02441516249, -0.02441516249, -0.02441516249] |
| CPB-v13.market-01 | fixed_spectral12 | 12 | [0.04462892695, 0.04462892695, 0.04462892695] | [-0.006113251108, -0.006113251108, -0.006113251108] |
| CPB-v13.market-01 | mean4 | 4 | [0.3270058792, 0.3270058792, 0.3270058792] | [-0.02569311994, -0.02569311994, -0.02569311994] |
| CPB-v13.market-01 | uncertainty4 | 4 | [0.07640926297, 0.07640926297, 0.07640926297] | [-0.04311938072, -0.04311938072, -0.04311938072] |

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns; contribution STD in **correlation** · **complexity4/5, ordinal** · final TRAIN112, own ridge; reps2701/2802/2903.

| Method | Block | Size | Contribution STD: [three reps] | Correlation to own residual: [three reps] |
| --- | --- | ---: | --- | --- |
| CPB-v11.market-01 | learned20 | 20 | [0.0643768039, 0.0643768039, 0.0643768039] | [-0.04723236946, -0.04723236946, -0.04723236946] |
| CPB-v11.market-01 | fixed_spectral12 | 12 | [0.02439904319, 0.02439904319, 0.02439904319] | [-0.003794108496, -0.003794108496, -0.003794108496] |
| CPB-v11.market-01 | mean4 | 4 | [0.05546361618, 0.05546361618, 0.05546361618] | [-0.009924604203, -0.009924604203, -0.009924604203] |
| CPB-v11.market-01 | uncertainty4 | 4 | [0.008381747381, 0.008381747381, 0.008381747381] | [-0.02667728257, -0.02667728257, -0.02667728257] |
| CPB-v13.market-01 | learned20 | 20 | [0.06502059652, 0.06502059652, 0.06502059652] | [-0.04748185375, -0.04748185375, -0.04748185375] |
| CPB-v13.market-01 | fixed_spectral12 | 12 | [0.02604187162, 0.02604187162, 0.02604187162] | [-0.003633142551, -0.003633142551, -0.003633142551] |
| CPB-v13.market-01 | mean4 | 4 | [0.05689058216, 0.05689058216, 0.05689058216] | [-0.01075454415, -0.01075454415, -0.01075454415] |
| CPB-v13.market-01 | uncertainty4 | 4 | [0.008353228193, 0.008353228193, 0.008353228193] | [-0.02227320751, -0.02227320751, -0.02227320751] |

## Saved temporal learning descriptors

The new V13 CSV has16384 rows: eight groups across2048 successful updates,2048 records/group. Cached V11 summaries retain its own historical mean/variance groups; V13 groups are mean/regularized_stddev. These are recorded postclip gradient/movement magnitudes, and movement includes AdamW decay. They do not measure raw data-gradient importance, clipping damage or task contribution. PRECLIP upstreams belong to the separate actual captures.

Dataset: **MKT-HIST-2** · four temporal bins, mean4 and second-moment4 groups · **complexity4/5, ordinal** · saved TRAIN trajectory,2048 updates.

| Model | Bin / moment | Mean weight-gradient L2 | Mean weight-movement L2 | Records |
| --- | --- | ---: | ---: | ---: |
| CPB-v11.market-01 | 0 / mean | 0.008813839777 | 0.002559086183 | 2048 |
| CPB-v11.market-01 | 0 / centered_variance | 0.001298440467 | 0.002709726305 | 2048 |
| CPB-v11.market-01 | 1 / mean | 0.009470987799 | 0.002690457412 | 2048 |
| CPB-v11.market-01 | 1 / centered_variance | 0.001227150446 | 0.002428820784 | 2048 |
| CPB-v11.market-01 | 2 / mean | 0.01223881131 | 0.002565176811 | 2048 |
| CPB-v11.market-01 | 2 / centered_variance | 0.001461729294 | 0.00266621598 | 2048 |
| CPB-v11.market-01 | 3 / mean | 0.01137127991 | 0.002777608034 | 2048 |
| CPB-v11.market-01 | 3 / centered_variance | 0.001415239652 | 0.002515110956 | 2048 |
| CPB-v13.market-01 | 0 / mean | 0.008844830076 | 0.002556662098 | 2048 |
| CPB-v13.market-01 | 0 / regularized_stddev | 0.001668848261 | 0.002254833246 | 2048 |
| CPB-v13.market-01 | 1 / mean | 0.009478959209 | 0.002672021022 | 2048 |
| CPB-v13.market-01 | 1 / regularized_stddev | 0.001521521908 | 0.002122115472 | 2048 |
| CPB-v13.market-01 | 2 / mean | 0.01285775915 | 0.002517899654 | 2048 |
| CPB-v13.market-01 | 2 / regularized_stddev | 0.001824294059 | 0.00240411371 | 2048 |
| CPB-v13.market-01 | 3 / mean | 0.01207095935 | 0.002757847453 | 2048 |
| CPB-v13.market-01 | 3 / regularized_stddev | 0.001885202678 | 0.00223860812 | 2048 |

Dataset: **MKT-HIST-2** · V13 eight mean/regularized_stddev groups · **complexity4/5, ordinal** · fixed TRAIN blocks1–512,513–1024,1025–1536,1537–2048, in this order.

| Bin / moment | Mean weight-gradient L2: [four blocks] | Mean weight-movement L2: [four blocks] | Nonzero-gradient count mean: [four blocks] | Moved-value count mean: [four blocks] |
| --- | --- | --- | --- | --- |
| 0 / mean | [0.01217590983, 0.008946142052, 0.007538432902, 0.006718835523] | [0.003450630602, 0.002323252142, 0.002233737787, 0.00221902786] | [260, 260, 260, 260] | [259.9941406, 259.9980469, 260, 259.9960938] |
| 0 / regularized_stddev | [0.002764204869, 0.001637577645, 0.001249750654, 0.001023859877] | [0.003065229053, 0.002023288381, 0.001949090558, 0.001981724991] | [260, 260, 260, 260] | [259.9980469, 259.9980469, 259.9882812, 259.9902344] |
| 1 / mean | [0.01237794839, 0.00985033612, 0.00828633185, 0.00740122048] | [0.00365298904, 0.002418204835, 0.002325348079, 0.002291542135] | [260, 260, 260, 260] | [260, 260, 259.9980469, 259.9921875] |
| 1 / regularized_stddev | [0.002555803, 0.001537614022, 0.001116018022, 0.0008766525896] | [0.002993747917, 0.001871115904, 0.001860790105, 0.001762807964] | [260, 260, 260, 260] | [259.9960938, 259.9980469, 259.9941406, 259.9960938] |
| 2 / mean | [0.018132052, 0.01286372074, 0.01064396565, 0.009791298208] | [0.003627243773, 0.002117464421, 0.002142555105, 0.002184335318] | [260, 260, 260, 260] | [260, 259.984375, 259.9960938, 259.9863281] |
| 2 / regularized_stddev | [0.002855394303, 0.001792834144, 0.001436957656, 0.001211990133] | [0.003157448139, 0.002189387017, 0.002140881328, 0.002128738358] | [260, 260, 260, 260] | [259.9980469, 259.9921875, 259.9960938, 259.9941406] |
| 3 / mean | [0.01672775951, 0.01215273115, 0.01018318462, 0.009220162093] | [0.00369524293, 0.002527938967, 0.002442521621, 0.002365686293] | [260, 260, 260, 260] | [259.9941406, 259.9980469, 260, 259.9960938] |
| 3 / regularized_stddev | [0.002911802022, 0.001918240785, 0.001466508484, 0.001244259419] | [0.003163978068, 0.001987143469, 0.001893382428, 0.001909928513] | [260, 260, 260, 260] | [259.9980469, 259.9941406, 259.9882812, 259.9960938] |

All five period objects and six scalar distributions—including bias fields/full quantiles—remain at temporal_moment_observations.csv.overall_and_fixed512. V11's corresponding objects are opaque cached DATA. Differently defined variance/stddev gradient units do not rank importance.

## Actual saved captures

Update1 and2048 capture both ordinary/contextual routes and all four bins from each existing successful update's original forward. Update2048 uses actual pre-step2047 coefficients. Selected24of60 fields include rawvariance, F32 root/dispersion, attention mixture/mean, saved projections and PRECLIP upstreams. No new forward or beta contraction occurs. A counted division gives the descriptive 1/(2*captured_root) for every present coordinate; it is not a recomputed root or full hidden/loss adjoint.

Dataset: **MKT-HIST-2** · captured temporal hidden moments · **complexity4/5, ordinal** · V13 TRAIN update1/2048, two routes/four bins. Medians/maxima are saved per-capture quantiles; hidden units are not task units.

| Update / route / bin | Rawvariance p50 | Raw root p50 | Dispersion p50 | Slope p50 | Slope max | Exact zeros: variance / dispersion | Present scalar coordinates |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 / ordinary / 0 | 0.2594244778 | 0.5102950633 | 0.4790450633 | 0.9798252742 | 3.493525494 | 0 / 0 | 1536 |
| 1 / ordinary / 1 | 0.3123014271 | 0.5597123504 | 0.5284623504 | 0.8933161642 | 3.197216169 | 0 / 0 | 1536 |
| 1 / ordinary / 2 | 0.2916825861 | 0.5409796834 | 0.5097296834 | 0.9242496003 | 3.487130861 | 0 / 0 | 1536 |
| 1 / ordinary / 3 | 0.272650674 | 0.523093909 | 0.491843909 | 0.9558513188 | 3.065198343 | 0 / 0 | 1536 |
| 1 / contextual / 0 | 0.27465868 | 0.5250097215 | 0.4937597215 | 0.9523634125 | 3.655084068 | 0 / 0 | 1536 |
| 1 / contextual / 1 | 0.3095651716 | 0.5572627187 | 0.5260127187 | 0.8972428694 | 2.423547151 | 0 / 0 | 1536 |
| 1 / contextual / 2 | 0.3028485179 | 0.5512032807 | 0.5199532807 | 0.9071064098 | 2.716919778 | 0 / 0 | 1536 |
| 1 / contextual / 3 | 0.2911931872 | 0.5405272841 | 0.5092772841 | 0.9250226897 | 3.687588743 | 0 / 0 | 1536 |
| 2048 / ordinary / 0 | 0.08452340215 | 0.2924035639 | 0.2611535639 | 1.709967839 | 5.671201959 | 0 / 0 | 1536 |
| 2048 / ordinary / 1 | 0.08447014913 | 0.2923125029 | 0.2610625029 | 1.710500314 | 6.569729842 | 0 / 0 | 1536 |
| 2048 / ordinary / 2 | 0.08963179961 | 0.301012218 | 0.269762218 | 1.661062184 | 5.514843653 | 0 / 0 | 1536 |
| 2048 / ordinary / 3 | 0.0851289928 | 0.293437466 | 0.262187466 | 1.70394073 | 5.085992624 | 0 / 0 | 1536 |
| 2048 / contextual / 0 | 0.08327325806 | 0.290258199 | 0.259008199 | 1.722604248 | 5.848439793 | 0 / 0 | 1536 |
| 2048 / contextual / 1 | 0.0784933567 | 0.2819038332 | 0.2506538332 | 1.773657769 | 6.022494231 | 0 / 0 | 1536 |
| 2048 / contextual / 2 | 0.08482414111 | 0.2929175794 | 0.2616675794 | 1.706964833 | 5.258397614 | 0 / 0 | 1536 |
| 2048 / contextual / 3 | 0.08088424429 | 0.2861132175 | 0.2548632175 | 1.747560634 | 6.071063677 | 0 / 0 | 1536 |

No captured present coordinate has exact-zero rawvariance/dispersion. Saved slopes are below the analytic bound16; no observed zero-variance/slope-limit issue supports tuning tau. Raw-root absent-bin offset remains a source/trust exception; absent coordinates are excluded from these present-bin summaries.

Full quantiles, attention concentration, local support, manual projected mean4/stddev4 values and PRECLIP upstreams remain at temporal_moment_observations.actual_witnesses[*].routes[*].bins[*]. V11's16 cached bins remain at cached_v11_TRAIN.temporal_moment_observations.actual_witnesses; no V11 regularized-root/dispersion/slope panel exists. New V13 has16 bins,32 only when the cached V11 bins are also counted.

Missing packed global anchor IDs prevent matched-batch V11/V13 or update1/2048 claims. Magnitudes therefore do not establish training-caused variance decay. Intermediate moments for every one of the112 diagnostic rows were not captured and cannot be recreated here.

## Execution and provenance

Dataset: **MKT-HIST-2** · saved TRAIN diagnosis/cached V11 authority · **complexity4/5, ordinal** · runtime scope, no new encoder/head training.

| Operation | Evidence stage | Count / elapsed scopes | New model/head fits | New encoder updates |
| --- | --- | --- | ---: | ---: |
| V13 saved-arithmetic producer | Passed once | Body0.699064302s; outer recorder1.091231744s | 0 | 0 |
| Independent saved-arithmetic audit | Passed once, zero mismatches | 73778 checks; numerical body0.715012255s; child-process recorder0.769492269s | 0 | 0 |
| V11 cached reference | Four opaque objects | Latest audit69952/completion2202; upstream68264 | 0 | 0 |
| Documentary completion | Pending | — | — | — |

Times describe saved-array arithmetic in the existing cuwacunu container, not synchronized GPU training or inference benchmarks. Historical GPU/extraction/head costs remain in the [V13 results](CAUSAL_FRESH_V13_RESULTS.md), [detail](CAUSAL_FRESH_V13_DETAIL.md) and [V11 results](CAUSAL_FRESH_V11_RESULTS.md); none is replayed here.

Dataset: **MKT-HIST-2** · manual saved affine descriptors · **complexity4/5, ordinal** · TRAIN scalar operation counts, not module calls.

| Family | Dot outputs | Multiply-add terms | Module forwards | Head refits |
| --- | ---: | ---: | ---: | ---: |
| mean4_decoder | 55296 | 221184 | 0 | 0 |
| ridge_block | 10752 | 107520 | 0 | 0 |
| ridge_total | 2688 | 107520 | 0 | 0 |
| temporal_mean4_export | 1536 | 98304 | 0 | 0 |
| temporal_regularized_stddev4_export | 1536 | 98304 | 0 | 0 |

Captured-root scalar divisions: **24576**, one per present coordinate. Model/head/module calls, fits, updates, held-out tensor reads, old V11 raw/V12/V9/V8 numeric reads, trust rechecks and new quality scores are zero. Five positive affine families and scalar divisions remain explicit; zero module calls does not mean zero arithmetic.

Scope is25 numerical roles (23 V13 artifacts + shared TRAIN targets + four-selector cache),35 hash guards,14 metadata contexts and12 typed assertions. Checkpoint49 selects one saved mean4 decoder parameter plus48 typed metadata fields; no other model parameter. Transform is UTF8 ByteStorage rank1 and tau Float64[1], exact2^-10, outer/payload. Actual24of60 witnesses are selected; five-node trust is reused opaquely.

Provenance and full-precision evidence:

- [Admitted main card](CAUSAL_V11_V13_TRAIN_DIAGNOSIS_CARD.ADMITTED.json): 74190 B; SHA256 6b2d4c3b995bc4904cbeaddf97a73a930866838a2338e15744b472da42bf232f.
- [Main diagnosis](causal_v11_v13_train_diagnosis/diagnosis.json): 4183840 B; SHA256 716959d4428b3b65d71d4e0fa84fe5d96a84bdd9ab40266785b837d84f54ab27.
- [Execution](causal_v11_v13_train_diagnosis/execution.json): 2993 B; SHA256 75f7b8d3a60c2820cd8d2feea106c872a11d8ee3a8441eff6c455ef897d37a07.
- [Main whole mirror](causal_v11_v13_train_diagnosis/mirror.json): 7568 B; SHA256 86a5966991322d00cdc9f5402ece9f592471dc5157490cf93538038ad0b31386.
- [Independent audit](causal_v11_v13_train_diagnosis/independent-audit.json): 52464 B; SHA256 0d47cc5306b6fbfae2a886c0cf1d1a90dc90a9b18c767cffef8357fdb20d3d11.
- [Audit execution](causal_v11_v13_train_diagnosis/independent-audit-execution.json): 6909 B; SHA256 2953de46c724e70518bb9ae66bd37cb63835390c2d77c0e8607ddfafd4cfd0ef.
- [Audit whole mirror](causal_v11_v13_train_diagnosis/independent-audit-mirror.json): 9702 B; SHA256 9d1bbeda8ff047232e3a72fc547a9ac2dda8836d7cccfa008ce7f79cf0b89ce8;19 files including inventory self.
- [Closed design002](causal_v11_v13_train_diagnosis/design-inventory-002.PROPOSED.json), [whole20 parents](causal_v11_v13_train_diagnosis/parents-preservation.json), [owned producer](../code/research/causal_v11_v13_train_diagnosis/README.md), [source provenance](../code/research/causal_v11_v13_train_diagnosis/provenance.json), [producer peer review](causal_v11_v13_train_diagnosis/independent-producer-source-review-001.json), [workflow peer review](causal_v11_v13_train_diagnosis/workflow-independent-source-review-001.PROPOSED.json).
- [Cached V11 carrier](causal_v11_v12_train_diagnosis/diagnosis.json), [latest cache audit](causal_v11_v12_train_diagnosis/independent-audit.json), [completion](causal_v11_v12_train_diagnosis/completion.json), [upstream audit](causal_v9_v11_train_diagnosis/independent-audit.json). Prior checks are reused authority, not added to73778.
- [V13 final trust](causal_fresh_v13/final-establishment.json), [final preservation](causal_fresh_v13/final-preservation.json), [standard1.2](RESULTS_REPORTING_STANDARD.md). Current119/main143, predecessor V12/114 and frozen strongest866-floor V11/109 ancestry stay distinct.

Owned source bytes remain unchanged. Whole parent copies/native-half source before-files/workflow protocol-alias and truthful four-declared-checkpoint hash-disclosure corrections are preserved. These are preparation history, not invented numeric failures or scientific reruns. Documentary completion is still pending.

## What this supports and the next question

The narrow result is that regularized stddev changes scales and learned representations with mixed TRAIN readout/reconstruction effects. It neither establishes an overall better encoder nor explains held-out gains/losses causally. Existing V13 quality remains8/22/full coverage/five scoped records, not rescored here. Preserve all earlier leaders, sources, checkpoints, results, failures and reviews.

Raw covariance concentration, contribution magnitude, postclip movement and residual correlation do not prove branch importance, dead uncertainty, gradient conflict, global-clipping damage or missing task information. Full112 temporal intermediate captures are unavailable; bounded captures cannot provide per-task hidden-route attribution. Readout variation does not supply encoder-retraining/cohort uncertainty. Previously seen VALIDATION remains adaptively selected evidence.

After the independently audited arithmetic, root separately selected **CPB-v14.market-01 for DESIGN ONLY**: four fixed nested trailing supports of lengths4/8/12/16 instead of the four disjoint anchored bins, while retaining V13 regularized dispersion and the other shapes/budget/head settings. This is a new prospective design question, not a diagnosis conclusion or promised winner. Source preparation, proofs, training and quality still require separate admission. Fixed log1p(v) remains an **unselected hypothesis**, not part of that V14 direction. These captures do not support tau rescue. The saved diagnosis field diagnostic_gate_or_next_encoder_selected=false remains unchanged as a body-time record; this later root decision does not rewrite it.

V3 remains the sole DEV encoder. This report authorizes no new training/head fit/quality rescore, old model replay, DEV replacement, Git publication or deletion.
