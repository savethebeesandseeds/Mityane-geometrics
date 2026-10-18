# TEMPO-3 saved training diagnosis

This is a new diagnosis of saved TRAIN tensors. There was no model training, model inference, head fitting or PCA fitting. Validation percentages below reuse the audited TEMPO-3 comparison metadata. All five cohorts and all three fixed head repetitions are retained.

RPB-v7.alt-05 uses late channel mixing. RPB-v10.alt-05 mixes channels before temporal encoding. Both were trained from scratch on TEMPO-3 for 512 CUDA updates, batch 8; their native embedding has 32 values. These are separate from the successful TEMPO-1 trained copies.

Dataset: **TEMPO-3**, variable period and short lead/lag, independent gains and offsets, unrelated channel, noise and missing observations; **designed complexity 4/5**. View: TRAIN, 256 rows / 128 source pairs per cohort. Fixed heads: Ridge penalty 1; tanh16, Adam .01, 100 updates; repetitions 2701/2802/2903.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 86.41 | 99.69 | 100.00 |
| RPB-v7.alt-05 | 32 | 58.36 | 81.59 | 100.00 |
| RPB-v10.alt-05 | 32 | 55.70 | 82.06 | 100.00 |

Dataset: **TEMPO-3**, variable period and short lead/lag, independent gains and offsets, unrelated channel, noise and missing observations; **designed complexity 4/5**. View: VALIDATION, original observations, 128 rows / 64 pairs per cohort. Fixed heads: Ridge penalty 1; tanh16, Adam .01, 100 updates; repetitions 2701/2802/2903.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 51.25 | 50.73 | 100.00 |
| RPB-v7.alt-05 | 32 | 55.47 | 54.64 | 100.00 |
| RPB-v10.alt-05 | 32 | 52.81 | 54.90 | 100.00 |

Dataset: **TEMPO-3**, variable period and short lead/lag, independent gains and offsets, unrelated channel, noise and missing observations; **designed complexity 4/5**. View: VALIDATION, additional 30% coordinate deletion, 128 rows / 64 pairs per cohort. Fixed heads: Ridge penalty 1; tanh16, Adam .01, 100 updates; repetitions 2701/2802/2903.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.53 | 50.78 | 100.00 |
| RPB-v7.alt-05 | 32 | 54.22 | 54.01 | 100.00 |
| RPB-v10.alt-05 | 32 | 53.75 | 54.32 | 100.00 |

The raw neural head fits almost every training example but performs near chance on unseen source pairs. It can memorize this small training set. The trained native embeddings fit TRAIN less well and also transfer weakly. These scores do not establish that the encoder has destroyed timing information.

Dataset: **TEMPO-3**, same structured timing recipe, **designed complexity 4/5**. The following is a separate TRAIN timing rule applied within each saved eight-tick query patch. It uses four different masked contexts; it is not a classifier head attached to one full-context embedding.

| Saved surface | Timing accuracy % | Coverage % | Correct / all rows % |
| --- | ---: | ---: | ---: |
| Observed query targets | 100.00 | 100.00 | 100.00 |
| RPB-v7.alt-05 reconstruction | 68.28 | 100.00 | 68.28 |
| RPB-v10.alt-05 reconstruction | 61.02 | 100.00 | 61.02 |

The target row checks that the same supported short patches contain the answer. Weak timing in the reconstructed waveforms locates a problem somewhere in the learned reconstruction path; it does not separate encoder and decoder causes or prove what information the full-context native32 contains.

The next bounded candidate will add visible adjacent-time differences to the raw patch input. This supplies a local shape prior that cancels constant offsets in that added branch. Raw values remain available; the original reconstruction target, native32 bottleneck, context masks and fixed heads remain unchanged. The branch adds capacity and changes the optimization parameterization, so an improvement would not prove information recovery or general affine invariance.

Evidence: 125 unique TRAIN payload roles, decoded 130 times across the two tools because five controlled TRAIN roles are shared. Both tools checked frozen input/source bytes before and after arithmetic. No VAL archive bodies, TEST or stress data were read. No promotion.

Head result: `/embedding/output/runs/rpb-structured-hard-timing-train/head-fit-xrZMAS-v1/head-fit-diagnostic.json`; SHA `03624d5c349255d90a0851ead61914095b9ce179fb98a787fc647345b53ac59f`.
Query result: `/embedding/output/runs/rpb-structured-hard-timing-train/query-timing-xrZMAS-v1/query-diagnostic.json`; SHA `d9b545e18febe5fed9ab1a800db29a6f3b914f339af3790261cdc9098b7e228c`.
