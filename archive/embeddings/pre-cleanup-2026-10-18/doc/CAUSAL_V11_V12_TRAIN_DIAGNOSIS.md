<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root. -->
# Audited saved TRAIN diagnosis: CPB-v11 and CPB-v12

**Independent audit: 69,952 checks, zero mismatches. No new encoder experiment.** One new V12 saved-arithmetic diagnosis ran once; four existing audited V11 objects were reused without old raw reductions. Audit attempt001 stopped before tensor values; repaired attempt002 supplies the first independent numerical audit. No model, encoder, decoder, neural-head or module forward, fit, calibration or optimizer update occurred.

V12 does not show an overall TRAIN gain over V11. Final linear trend R² rises slightly, while linear volatility/activity/relation and all four neural R² means are lower. Measures are mixed: neural volatility/activity MAE improves slightly despite worse RMSE/R². TRAIN reconstruction NLL improves while MAE worsens. These observations do not explain the separately recorded known-VALIDATION failures causally. V3 remains the clean baseline and sole DEV encoder.

MKT-HIST-2 joins BTC/ETH/SOL as three channels in one market panel, C3/H128/F9; these are past-window retention tasks, not forecasts or three independent cohorts. Each saved readout was fitted on and scored here on the same112 chronological TRAIN windows. The32 known chronological VALIDATION windows and all TEST/stress/TEMPO/raw caches are excluded from numerical diagnosis. Complexity4/5 is a designed ordinal recipe level, not measured difficulty.

Initial/final are saved0/2048. Each version has one historical fresh2048-update/B8 CUDA trajectory; three readout repetitions2701/2802/2903 are head variation on one checkpoint/split. Both native40 exports are learned20 + literal spectral12 + mean4 + independent uncertainty4. Fixed heads: ridge penalty1 (164 fitted values);16 tanh hidden units/four outputs, Adam0.01/100 updates (724 values), permitted TRAIN-only scaling. D40 seeds are6049515622977958931,8262941649267994858,2095371932330426401. No best seed, checkpoint, head tuning, PCA or other post-encoder transform is selected.

V12 uses V11 SOURCE only: all learned values, optimizer and TRAIN statistics start fresh, source_updates=0. It retains each bin's attention mean4, while dispersion4 uses mask-only half-local/half-global uniform weights and its OWN two-sum center. Shapes/counts, objective, globalclip1/singleAdamW, update budget and fixed heads remain. Native uncertainty4, intermediate centered-dispersion export16 and decoder conditional variance are different quantities. Initial pairing covers native20:40 plus five whole statistics; learned0:20 is unpaired.

All tables copy saved scalars. Neural entries show the saved mean followed by all three individual values in repetition order2701/2802/2903. Values display ten significant digits; [documentary tables](causal_v11_v12_train_diagnosis/report-tables.json) retain full saved precision and exact full-record links. R² is relative to the scored TRAIN population mean, not accuracy or generalization. Initial/final state suffixes below describe saved states of the registered instance, not new instance tags.

## Past volatility

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R²: mean; [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial | 40 | 0.8704379066 | 0.9304516397; [0.9480124223, 0.9248865428, 0.918455954] | 100 (112/112) |
| CPB-v12.market-01 · initial | 40 | 0.8496729 | 0.9286436646; [0.946278875, 0.9234167795, 0.9162353392] | 100 (112/112) |
| CPB-v11.market-01 · final | 40 | 0.9915892992 | 0.9535589501; [0.9397402825, 0.9648279068, 0.956108661] | 100 (112/112) |
| CPB-v12.market-01 · final | 40 | 0.9912823724 | 0.9528875352; [0.9423509536, 0.9647051713, 0.9516064808] | 100 (112/112) |

Dataset: **MKT-HIST-2** · channel-mean population standard deviation of supported adjacent close log returns; physical errors in **logreturn** · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Linear MAE | Linear RMSE | Neural MAE: mean; [three reps] | Neural RMSE: mean; [three reps] |
| --- | ---: | ---: | --- | --- |
| CPB-v11.market-01 · initial | 0.0001387027591 | 0.0001811739557 | 9.516026118E-05; [8.606292762E-05, 9.910184037E-05, 0.0001003160156] | 0.0001321480457; [0.0001147642386, 0.0001379480812, 0.0001437318172] |
| CPB-v12.market-01 · initial | 0.0001487773579 | 0.0001951530882 | 9.062494976E-05; [8.656292712E-05, 9.522623452E-05, 9.008568764E-05] | 0.0001338762947; [0.0001166619785, 0.0001392911748, 0.0001456757308] |
| CPB-v11.market-01 · final | 3.717972113E-05 | 4.616075248E-05 | 7.910417677E-05; [8.997031077E-05, 6.831640848E-05, 7.902581107E-05] | 0.0001078014187; [0.0001235578476, 9.43965009E-05, 0.0001054499075] |
| CPB-v12.market-01 · final | 3.820181077E-05 | 4.699546436E-05 | 7.85978734E-05; [8.468070638E-05, 6.880643002E-05, 8.230648381E-05] | 0.0001087129942; [0.0001208517246, 9.456105903E-05, 0.0001107261988] |

## Past trend

Dataset: **MKT-HIST-2** · channel-mean supported adjacent close log-return trend · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R²: mean; [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial | 40 | 0.6937123027 | 0.9444130642; [0.9472332648, 0.93154477, 0.9544611578] | 100 (112/112) |
| CPB-v12.market-01 · initial | 40 | 0.71148221 | 0.9557782638; [0.9582860018, 0.9548443422, 0.9542044473] | 100 (112/112) |
| CPB-v11.market-01 · final | 40 | 0.9110286797 | 0.9382186199; [0.9409886975, 0.9277573259, 0.9459098363] | 100 (112/112) |
| CPB-v12.market-01 · final | 40 | 0.9114453922 | 0.9378110442; [0.9399173599, 0.926986095, 0.9465296779] | 100 (112/112) |

Dataset: **MKT-HIST-2** · channel-mean supported adjacent close log-return trend; physical errors in **logreturn/tick** · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Linear MAE | Linear RMSE | Neural MAE: mean; [three reps] | Neural RMSE: mean; [three reps] |
| --- | ---: | ---: | --- | --- |
| CPB-v11.market-01 · initial | 3.566489438E-05 | 4.620896957E-05 | 1.439798006E-05; [1.313720415E-05, 1.619516159E-05, 1.386157443E-05] | 1.961435648E-05; [1.917969161E-05, 2.184564262E-05, 1.781773519E-05] |
| CPB-v12.market-01 · initial | 3.552924244E-05 | 4.484848802E-05 | 1.327182358E-05; [1.305871266E-05, 1.293821237E-05, 1.381854571E-05] | 1.755452095E-05; [1.705306379E-05, 1.774261359E-05, 1.786788547E-05] |
| CPB-v11.market-01 · final | 1.887022258E-05 | 2.490499881E-05 | 1.483424212E-05; [1.457169087E-05, 1.535159796E-05, 1.457943752E-05] | 2.071447203E-05; [2.028285603E-05, 2.244183756E-05, 1.94187225E-05] |
| CPB-v12.market-01 · final | 1.901025657E-05 | 2.484660694E-05 | 1.492033611E-05; [1.474124194E-05, 1.551187219E-05, 1.450789421E-05] | 2.077819693E-05; [2.046614334E-05, 2.256130913E-05, 1.930713832E-05] |

## Past activity

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume) · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R²: mean; [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial | 40 | 0.9264675744 | 0.964113309; [0.9604736197, 0.9565085167, 0.9753577905] | 100 (112/112) |
| CPB-v12.market-01 · initial | 40 | 0.9236156409 | 0.9567070542; [0.9604951674, 0.9508049021, 0.9588210932] | 100 (112/112) |
| CPB-v11.market-01 · final | 40 | 0.9676749422 | 0.9699659773; [0.9666138639, 0.9702249784, 0.9730590895] | 100 (112/112) |
| CPB-v12.market-01 · final | 40 | 0.9667600749 | 0.9698500267; [0.9693579331, 0.9673663799, 0.972825767] | 100 (112/112) |

Dataset: **MKT-HIST-2** · channel-mean mean log1p(nonnegative observed base volume); physical errors in **log1p(base_volume)** · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Linear MAE | Linear RMSE | Neural MAE: mean; [three reps] | Neural RMSE: mean; [three reps] |
| --- | ---: | ---: | --- | --- |
| CPB-v11.market-01 · initial | 0.132608976 | 0.1683792272 | 0.09089358311; [0.09674711797, 0.1037722261, 0.07216140524] | 0.116806301; [0.1234504247, 0.129494456, 0.09747402243] |
| CPB-v12.market-01 · initial | 0.1355338379 | 0.1716134354 | 0.1020239705; [0.09818842274, 0.1102509141, 0.09763257473] | 0.1290485011; [0.1234167707, 0.1377241118, 0.1260046209] |
| CPB-v11.market-01 · final | 0.08901114748 | 0.1116397154 | 0.08690566221; [0.09054964755, 0.08762217514, 0.08254516395] | 0.1075073407; [0.1134572219, 0.1071457892, 0.101919011] |
| CPB-v12.market-01 · final | 0.09077403452 | 0.1132085125 | 0.08682266912; [0.0869561388, 0.09233072841, 0.08118114015] | 0.1077417704; [0.1086946313, 0.1121712844, 0.1023593956] |

## Past relation

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Size | Linear head R² | Neural head R²: mean; [three reps] | Coverage % |
| --- | ---: | ---: | --- | ---: |
| CPB-v11.market-01 · initial | 40 | 0.6710415617 | 0.8933798703; [0.8962817312, 0.9124124261, 0.8714454537] | 100 (112/112) |
| CPB-v12.market-01 · initial | 40 | 0.6902570383 | 0.8967708148; [0.8996968913, 0.9121944877, 0.8784210654] | 100 (112/112) |
| CPB-v11.market-01 · final | 40 | 0.7345649307 | 0.8704845323; [0.8593115988, 0.8898075441, 0.862334454] | 100 (112/112) |
| CPB-v12.market-01 · final | 40 | 0.7290370441 | 0.851814337; [0.844582596, 0.8647150635, 0.8461453514] | 100 (112/112) |

Dataset: **MKT-HIST-2** · eligible-pair mean Pearson correlation of supported adjacent close returns; physical errors in **correlation** · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method | Linear MAE | Linear RMSE | Neural MAE: mean; [three reps] | Neural RMSE: mean; [three reps] |
| --- | ---: | ---: | --- | --- |
| CPB-v11.market-01 · initial | 0.04625253306 | 0.05825764237 | 0.02526688974; [0.02456989577, 0.02334426332, 0.02788651013] | 0.03306402499; [0.03271222006, 0.03006101484, 0.03641884007] |
| CPB-v12.market-01 · initial | 0.04501269002 | 0.05653053725 | 0.0252683858; [0.02467427427, 0.02312158485, 0.02800929829] | 0.03256150784; [0.03216915004, 0.03009839104, 0.03541698244] |
| CPB-v11.market-01 · final | 0.04309224556 | 0.05233130202 | 0.0284738675; [0.02879099296, 0.02699234473, 0.0296382648] | 0.03650128335; [0.03809882103, 0.03371772939, 0.03768729962] |
| CPB-v12.market-01 · final | 0.04342270222 | 0.05287341361 | 0.0298499684; [0.02952521192, 0.02953181159, 0.03049288169] | 0.03908174109; [0.04004351704, 0.03736002091, 0.03984168531] |

## Native blocks and reconstruction

Dataset: **MKT-HIST-2** · native covariance/participation, four blocks and complete40 · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method/state | Block | Width | Covariance trace | Participation ratio |
| --- | --- | ---: | ---: | ---: |
| CPB-v11.market-01 · initial | learned20 | 20 | 0.006245551576 | 3.24406852 |
| CPB-v11.market-01 · initial | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v11.market-01 · initial | mean4 | 4 | 1.122570926 | 3.659153602 |
| CPB-v11.market-01 · initial | uncertainty4 | 4 | 1.265117761 | 3.399078554 |
| CPB-v11.market-01 · initial | complete40 | 40 | 3.923562637 | 11.73760173 |
| CPB-v12.market-01 · initial | learned20 | 20 | 0.006364810261 | 3.205083672 |
| CPB-v12.market-01 · initial | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v12.market-01 · initial | mean4 | 4 | 1.122570926 | 3.659153602 |
| CPB-v12.market-01 · initial | uncertainty4 | 4 | 1.265117761 | 3.399078554 |
| CPB-v12.market-01 · initial | complete40 | 40 | 3.923681896 | 11.73840515 |
| CPB-v11.market-01 · final | learned20 | 20 | 184.8944722 | 3.276767195 |
| CPB-v11.market-01 · final | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v11.market-01 · final | mean4 | 4 | 23.0285827 | 3.241013044 |
| CPB-v11.market-01 · final | uncertainty4 | 4 | 78.78198941 | 1.048875014 |
| CPB-v11.market-01 · final | complete40 | 40 | 288.2346727 | 3.941626973 |
| CPB-v12.market-01 · final | learned20 | 20 | 186.3848292 | 3.260927953 |
| CPB-v12.market-01 · final | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v12.market-01 · final | mean4 | 4 | 23.03999631 | 3.225616857 |
| CPB-v12.market-01 · final | uncertainty4 | 4 | 78.61843251 | 1.049061677 |
| CPB-v12.market-01 · final | complete40 | 40 | 289.5728865 | 3.925672747 |

Final native uncertainty4 has participation ratio1.048875014 (V11) and1.049061677 (V12), near one in these coordinate units. This describes covariance concentration, not mathematical rank, unused coordinates, task information, a dead branch or intermediate dispersion16. Learned bases are unaligned. TRAIN head normalization removes per-coordinate scale; raw covariance trace is not information importance.

Dataset: **MKT-HIST-2** · fixed hidden-target reconstruction; standardized query units · **complexity 4/5, ordinal** · saved TRAIN112, in-sample.

| Method/state | TRAIN MAE ↓ | Gaussian NLL ↓ | Mean Huber1 ↓ | Total objective ↓ |
| --- | ---: | ---: | ---: | ---: |
| CPB-v11.market-01 · initial | 0.6640733605 | 0.4237330498 | 0.3236635132 | 0.747396563 |
| CPB-v12.market-01 · initial | 0.6640780263 | 0.4237368835 | 0.323667908 | 0.7474047915 |
| CPB-v11.market-01 · final | 0.2517975124 | -0.369659381 | 0.08466702545 | -0.2849923556 |
| CPB-v12.market-01 · final | 0.252382241 | -0.374763669 | 0.08486778332 | -0.2898958857 |

Final V12 TRAIN MAE0.2523822409723754 is worse than V11's0.2517975124166082; NLL−0.37476366899994823 is better than−0.3696593810373693. No TRAIN-NLL gate is added. Full112 saved row components/halves remain in the [audited result](causal_v11_v12_train_diagnosis/diagnosis.json); first8 queries permit only the same-stage saved mean4 affine residual and approximate stored-total-minus-residual parent remainder. No nonlinear decoder is run.

## Own saved readout contributions and associations

The JSON links full records for every task, both stages, all three repetitions, both readouts and fixed halves. Its ridge_contributions_* panels copy the six block-pair covariances and full/marginal contribution variances, with exact links to all112 row contributions. Final sum-to-marginal ridge-contribution variance ratios are1.288701292→1.276602625 (volatility),0.9954969325→0.9896987405 (trend),1.637701257→1.662712264 (activity), and0.9091577071→0.9329902351 (relation), V11→V12. All three saved repetitions retain their records; these ridge ratios are identical across them. The ratios do not support a broad new cancellation collapse. They describe the existing fitted predictor, not a block-removal/refit counterfactual or latent-gradient conflict.

Reconstruction/residual associations depend on task, head, repetition and scope. V12 final all112 neural absolute-residual associations with query MAE are positive for volatility/trend/activity and negative for relation in all three repetitions. reconstruction_associations_* panels link every full signed/absolute coefficient record for MAE/NLL/Huber/variance/objective across all declared tasks, heads, repetitions and scopes. These are in-sample associations, not evidence that reconstruction causes a validation result.

## Captured moment evidence

Dataset: **MKT-HIST-2** · CPB-v11.market-01 recorded moment subgroups; coordinate-dependent scalar means · **complexity 4/5, ordinal** · all2048 recorded TRAIN updates.

| Bin | Export half | Weight gradient L2 | Bias gradient L2 | Weight movement L2 | Bias movement L2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | mean | 0.008813839777 | 0.001410660074 | 0.002559086183 | 0.0002601005402 |
| 0 | centered_variance | 0.001298440467 | 0.0008583272502 | 0.002709726305 | 0.0002844791214 |
| 1 | mean | 0.009470987799 | 0.001481106932 | 0.002690457412 | 0.0002749760933 |
| 1 | centered_variance | 0.001227150446 | 0.0008170369549 | 0.002428820784 | 0.0002716271614 |
| 2 | mean | 0.01223881131 | 0.001973526408 | 0.002565176811 | 0.0002600609508 |
| 2 | centered_variance | 0.001461729294 | 0.000918990993 | 0.00266621598 | 0.000294728354 |
| 3 | mean | 0.01137127991 | 0.002098249717 | 0.002777608034 | 0.0002902464901 |
| 3 | centered_variance | 0.001415239652 | 0.0008702144324 | 0.002515110956 | 0.0002779033825 |

Dataset: **MKT-HIST-2** · CPB-v12.market-01 recorded moment subgroups; coordinate-dependent scalar means · **complexity 4/5, ordinal** · all2048 recorded TRAIN updates.

| Bin | Export half | Weight gradient L2 | Bias gradient L2 | Weight movement L2 | Bias movement L2 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | mean | 0.008893722611 | 0.001439830072 | 0.002552412 | 0.0002585571938 |
| 0 | centered_variance | 0.001377098695 | 0.0007808783055 | 0.002675491466 | 0.0002860357165 |
| 1 | mean | 0.009310714315 | 0.001542757316 | 0.002682483308 | 0.0002723376479 |
| 1 | centered_variance | 0.001315509744 | 0.0008342508464 | 0.002421446923 | 0.0002745204838 |
| 2 | mean | 0.01194032377 | 0.001970196621 | 0.002615781196 | 0.0002635407514 |
| 2 | centered_variance | 0.001508766022 | 0.0008228391943 | 0.002668281611 | 0.0003009785882 |
| 3 | mean | 0.01132166179 | 0.002140805267 | 0.00269498805 | 0.0002796983475 |
| 3 | centered_variance | 0.001343012254 | 0.00080792054 | 0.002484958186 | 0.0002818230248 |

The JSON copies all eight groups' six norm/count means for overall and fixed updates1–512/513–1024/1025–1536/1537–2048, with exact links to every full quantile record. It also covers all actual update1/2048 ordinary/contextual routes and all four bins, linking every saved support/mean/variance/concentration/export/PRECLIP-upstream descriptor. V12 additionally has captured uniform concentration/OWN center/attention-minus-own-center gap. V11 lacks these captures; no missing comparison was fabricated.

Within the late captured batches, centered-variance medians are around0.08 in both versions. Many dispersion4 export medians have smaller absolute magnitude than mean4 medians; for example V12 ordinary bin1 is0.08002890262 versus−0.9003196363. This is a within-capture, coordinate-dependent magnitude observation only. It proves neither information importance, starvation nor harmful global clipping. Export values include bias, saved heads normalize coordinates, and movement includes AdamW decay. Attention mean and uniform OWN center remain separate captured arrays; their gap reconstructs no hidden state or alternative encoder.

Actual1 and2048 use different batches/masks and lack global anchor IDs: similar packed shapes/positions do not establish same-input paired batches, nor do their differences establish training-caused variance decay. Actual2048 is BEFORE-step completed state2047, with its own saved export coefficients, never final2048 decoder. No intermediate moments exist for all112 task rows, preventing moment-to-native-to-task path reconstruction. No full hidden adjoint, score-to-hidden route, whole encoder/global norm or remaining2046 raw-gradient reconstruction is claimed.

## Cost, provenance and metadata refusal

Dataset: **MKT-HIST-2** · saved scalar audit work; no new encoder training · **complexity 4/5, ordinal** · diagnostic body only.

| Saved affine family | Dot outputs | Product terms |
| --- | ---: | ---: |
| mean4_decoder | 55296 | 221184 |
| ridge_block | 10752 | 107520 |
| ridge_total | 2688 | 107520 |
| temporal_mean4_export | 1536 | 98304 |
| temporal_variance4_export | 1536 | 98304 |

The result additionally discloses24,576 captured-center scalar subtractions and implementation-specific summary-call counts. These are performed scalar work, separate from zero module/fit/update counts. Saved main body elapsed0.850063549998595 s and independent body0.7052637600008893 s include reading/hashing/scalar work; neither is a new GPU-training time or a repeated cost benchmark. Prior encoder training, extraction and head fitting are reused and were not timed again.

Audit attempt001 refused in metadata before tensor values: the old completed V11 mirror lists18 files and separately pins its inventory, for19 total; completion records19. The original check compared18 directly to19. Failed attempt001 itself preserves18 files (17 listed plus inventory), stderr and execution, with no audit.json. A complete copied repair changes exactly ONE auditor metadata expression to len(files)+1 == mirror.file_count == completion count. Reader, component arithmetic, tolerances, selectors and counters remain unchanged. Attempt002 supplies the first independent numerical audit. Original20 sources/cards/reviews and all failed leaves remain preserved; the successful main diagnosis was not replayed.

Exact source chain: [main result](causal_v11_v12_train_diagnosis/diagnosis.json),4,069,556 B/SHA256 44d33fab98e8f39d09d2b829f659e7122353e5775149b776080bbbbfbc813e3d; [independent audit](causal_v11_v12_train_diagnosis/independent-audit.json),50,560 B/SHA256 f9ce383f1262eb4f6fa95badcf286f8a59f726d83e9bcda506d31b3816146ba5; [passed audit mirror](causal_v11_v12_train_diagnosis/independent-audit-mirror.json); [repair source review](causal_v11_v12_train_diagnosis/audit-repair-001/repair-independent-source-review-001.json); [documentary tables](causal_v11_v12_train_diagnosis/report-tables.json). The main mirror has36 files; passed audit mirror has19 total (18 listed plus inventory). Current114-checkpoint preservation and strongest866 protocol ancestry remain intact. Documentary completion/root review follows this report; no completion pass is claimed here.

## Proposed next direction

Keep V11's scoped leaders and every earlier result/checkpoint; V3 remains sole DEV. Stop the exact V12 recipe without coefficient/seed/rate/budget/head/checkpoint rescue. Strongest866 retention22 remains unchanged, protecting all established gains and adding no TRAIN-NLL gate. No new overall winner or encoder is selected by this diagnosis.

A proposed hypothesis for separate design review is amplitude scaling of centered dispersion, for example sqrt(variance) with a prospectively fixed floor, from stronger V11 SOURCE with entirely fresh weights/statistics and the same budget/heads. Weigh this against retaining attention-coupled dispersion or changing temporal information allocation: current magnitude evidence does not establish which mechanism would help. Declare exact support/empty-bin/F32 arithmetic, gradients, causal masking, typed policy/resume and strongest-floor criteria before any quality test. No V13 design/card, implementation, training, head fit, new measurement or DEV change is admitted here.
