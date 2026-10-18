# Continue from v7 after decoder calibration

Date: 2026-10-08. This new record follows the completed diagnostics; the original
[dated continuation](CONTINUATION_2026-10-08.md) is preserved.

## Saved milestone

We continued from RPB-v7 and kept RPB-v4 as the active reference. The existing
v7 decoder received 128 CUDA updates while all 214,277 non-decoder parameters, buffers,
configuration and TRAIN scaler stayed frozen. Native32 outputs are exact before
and after. The original 512 encoder updates remain unchanged.

Mean TRAIN/known-VALIDATION fixed-query standardized MAE fell from
0.08117869543536711/0.08362986890671145 to
0.054038532332366426/0.05698931712331416. All five masters beat their paired
v4@512 reconstruction errors. Mean new decoder loop time is 2.0918182938 seconds,
including CPU evidence copies, separate from retained encoder 15.6919237332 seconds.
No CPU encoder training/inference, head fits or new accuracy measurements occurred.
The cached v7 timing quality remains 97.96875% linear/98.9583333% neural, 100%
coverage on the same known VALIDATION; this is no fresh held-out confirmation.

This shows that unchanged v7 features can support better reconstruction through
the existing decoder. It does not change the embedding, introduce a version tag,
promote v7 or establish generalization. Read the
[measured record](../code/encoders/raw_patch_bottleneck_mae/V7_DECODER_CALIBRATION_DIAGNOSTIC.md)
and [durable numerical record](results/v7_decoder_calibration_v1.json).

The separate [v9 saved-TRAIN lesson](../code/encoders/raw_patch_bottleneck_mae/NATIVE_VIEW_LOSS_SCALE_TRAIN_DIAGNOSTIC.md)
shows zero scale-floor activations and later agreement growth. Keep v9 stopped.
Scalar ratios and combined gradient norms do not establish objective-specific
gradient allocation or causation. Do not rescue its frozen coefficients/budget.

## Next bounded experiment

1. Freeze one fresh TRAIN/VALIDATION replication card before generating data:
   five new disjoint masters/source groups, timing task, the same 256/128 examples
   and 128/64 source pairs, natural missingness 0.10, native32 and original queries.
   No seed, rate, capacity or budget search; no TEST/stress access.
2. Pair fresh v4/v7 initial weights/scalers. Train each encoder once on CUDA for
   exactly 512 unskipped updates. Keep the original v4 ordinary and v7 lighter
   deletion 0.15 policies. Use the same fixed batch 8/AdamW recipe.
3. Give both existing decoders exactly 128 additional updates with frozen encoders,
   original context/targets and fresh decoder-only AdamW. The current adapter
   admits v7 only. Add an explicit v4 admission path and a separate prospective
   driver/card; preserve today's captured source/assets/readers. Do not relax
   the old v7 card or convert typed assets into ordinary checkpoints.
4. Fit the declared linear/neural heads and standalone raw/PCA controls once
   from saved native TRAIN features. Keep ridge 1 and tanh16 Adam 0.01/100 with
   three paired repetitions. Reuse all fits/predictions after calibration once
   exact unchanged native/encoder invariants pass. No post-encoder PCA.
5. Report intact and declared 0.30-deletion VALIDATION quality, standardized
   reconstruction at decoder 0/128, every master and separate encoder/decoder
   time. Predeclare recovery against v4@512 and the equally calibrated v4
   comparison separately; extra decoder compute must not become an encoder-gain
   claim. Check uncertainty/coverage without treating head repetitions as new
   encoder runs. Save and audit before discussing any default/promotion decision.

This experiment checks whether today's result transfers to fresh sources and
whether its advantage survives an equal decoder budget. Keep broader architecture
changes and additional objectives outside this next step.

## Replay and execution

Use the existing managed container `cuwacunu_embedding`, immutable ID
`1e8de9cfec33591fb06b2f182bff04f6e9fecf2ddeb0aa865f34eeac3639c2ca`,
RTX A2000 8GB CUDA. All development/tests/model work run inside it. CPU may verify
saved arithmetic or fit fixed heads/raw PCA; it must not repeat encoder or decoder training.

Today's source identity is
`fb5aef0d015b9ca373bc7a52a24ada6cb86bde3f9ac0e451d7a419c684d8de4d`.
The frozen [card](../code/evaluation/cards/v7_decoder_calibration_v1.md) SHA is
`5ae05a3b5ec253d1743842f82fc28b45e0f20c8c84da1b58bf4ab9c514d292e9`.
The measured capsule is
`output/runs/rpb-v7-decoder-calibration/decoder-calibration-jXsW20`, inventory
`51400954bc2e9cae483de2e2a784e83f93482b02a888a3bdf95581d207967442`.
Actual CUDA admission 0xwMZJ passed; first compile-only admission rZA1Ma is retained.
The two-reference compile fix preceded quality fitting and changed no recipe.

Independent saved-arithmetic audit passed 25,450,287 checks:
`output/runs/rpb-decoder-calibration/audit-tools/run-jXsW20-v2/validation.json`, SHA
`dfc8d96b0f1657d22c7e2f8dbd6ae53637f62f3d627d3763d4f5f47ef2808cea`.
Reader SHA
`bf85b364ad9b41d4d6abc23756dd2d0b216ba877d71377b99a786ce37a0a6628`.
The [small source-only replay bundle](../output/runs/rpb-decoder-calibration/audit-tools/replay-bundle-jXsW20-20261008T094909Z-f32057cd/README.md)
copies no tensors and reruns no models; inventory
`20279aa73d6c6bad15a24457ba61100c06a36f66f6123bb67ff8150aa4919fd4`.
Keep all historical cards/checkpoints/readers/results immutable.
