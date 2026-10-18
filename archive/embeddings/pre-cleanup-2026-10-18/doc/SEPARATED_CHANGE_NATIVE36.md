# RPB-v24.market-01.alt-01: optional native36 export

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

This completed variation uses unchanged portable RPB-v24 arithmetic with four
change coordinates instead of eight. RPB-v22 remains primary. Preserve the
original native40 instance, all earlier versions and this native36 initial/final
bundle. Use the explicit optional settings and typed loaders; never relabel
old checkpoints. The [instance registry](separated_change_native36_embedding_versions.json)
pins exact geometry, exposure, checkpoint hashes and portable pairs.
The later [frozen TEMPO-5 transfer](MARKET_ENCODER_SYNTHETIC_TRANSFER.md) makes
native36 the leading research candidate. The
[save milestone](MILESTONE_2026-10-14_NATIVE36_TRANSFER.md) records that scoped
decision and the user's publication authorization; v22 stays primary.

## Input, export and reconstruction

Supply TRAIN-scaled CUDA float32 values `[B,C,H,F]`, Boolean observed masks
with the same shape, semantic channel IDs, endpoints and sampling interval.
The market instance uses `[B,3,128,9]`, P8/W64 and interval60000 milliseconds.
The export is one rank2 `[B,36]` vector per whole window: the entire frozen
trained v22 prefix32 followed by four learned change coordinates. The prefix
contains 20 learned and 12 fixed coordinates, giving 24 learned and 12 fixed overall.
No post-encoder PCA or other projection is applied.

The generic change bank uses mean/RMS over adjacent visible endpoints and
four original-time quarters. Both endpoints must be visible; gaps are never
bridged and observation counts are not projection inputs. A bias-free270-to4
projection feeds a separate bias-free4-to3456 waveform residual decoder.
The decoder starts at zero, so initial reconstruction equals the frozen parent.
Its first update opens the decoder; projection learning is checked at the next
update. The parent encoder and decoder execute under NoGrad. Exactly 14904 new
parameters update while 282521 parent parameters remain protected.

`SeparatedChangeConfig` contains the v22 `level` configuration and
`change_width`. File settings specify `LEVEL_EMBEDDING_SIZE=32` and
`EMBEDDING_SIZE=36`; their positive difference is four. C/H/F/P/W remain
configurable under inherited constraints, including H>=17 and CUDA float32.
The coordinate-erasure market recipe fixes complete P8 patches at H128;
partial-history engineering checks do not establish market quality at another
history length. The explicit [native36 settings](../code/encoders/raw_patch_bottleneck_mae/config/separated_changes_native36.conf)
leave the original native40 and primary settings intact.

## Checkpoints, safety and engineering evidence

Warm-start from the hash-pinned typed RPB-v22.market-02 checkpoint, matching
level config, TRAIN scaler and 2048 parent updates. Copy values into the
registered child with `separated_change_training::warm_start_level`; retain
`FrozenLevelGuard` throughout training. Protected gradients, weight decay and
optimizer moments are forbidden. Snapshot checks protect exact parent values
and buffers. Both initial/final native banks match saved parent prefix values,
validity and row IDs exactly, without old encoder quality replay.

Distinct v24 inference and model/scaler/AdamW/counter training archives bind
the geometry and source/split/recipe/scaler identities. The counter records new
branch updates; inherited exposure is separate. D36/D40 mismatches and wrong
archive types are rejected. The integration fixture proves immutable saves,
typed reload and exact next resumed loss/weights, plus masks/poison/gaps,
semantic order, partial history, actual decoder/projection learning and
protected values/no moments. Only one integration fixture execution is claimed.

The five ignored checkpoint files were present and their SHA256/size matched
the completion on 14 October 2026; cloning source does not supply these weights.
The saved [initial inference](../../cuwacunu_dev/output/runs/market-separate-changes-native36/run-4SauBf/checkpoints/inference-0.pt)
SHA256 is `2b614ca9866048a850cdeb73dc719c5816fd04bb39e1746b626d1abf1ef68108`.
The [final inference](../../cuwacunu_dev/output/runs/market-separate-changes-native36/run-4SauBf/checkpoints/inference-2048.pt)
SHA256 is `a3939b13df8b1704fc3621876cf3adc6efaddd98951147ff43ec6408978fa98c`;
the [final training state](../../cuwacunu_dev/output/runs/market-separate-changes-native36/run-4SauBf/checkpoints/training-2048.pt)
SHA256 is `81f6f700676d3afccdcf6e07923741dfc89fdd24b549443612b7dc5af330f4d2`.
The registry preserves the 512/1024 states too. The additive
[verification receipt](../../cuwacunu_dev/doc/results/separate_change_native36_v1/verification.json)
checks all local artifacts, durable copies, portable pairs and Git-ignore state.

The fixture and settings are locally mirrored byte-for-byte; all 45 portable
pairs match, including 43 preserved pairs. The additive standalone targets are
`rpb-v24-native36` and `test-rpb-v24-native36`; their presence is not a standalone
test run. This work executed only in the existing GPU-enabled `cuwacunu`
container. Follow the [integration environment](../../cuwacunu_dev/doc/ENVIRONMENT.md)
and [sync receipt](../../cuwacunu_dev/doc/results/separate_change_native36_v1/portable-sync.json).
Market loading, training producer and evaluation remain integration-owned.

## Completed result and next direction

The [full report](../../cuwacunu_dev/doc/MARKET_SEPARATE_CHANGE_NATIVE36_RESULTS.md)
contains separate per-task tables, physical-unit errors and all head fits.
MKT-HIST-2 is BTC/ETH/SOL January 2024 minute history, recent128 past-only
targets, designed complexity 4/5. TRAIN is Jan 1–21; known VALIDATION Jan 22–28;
Jan 29–31 is unused. The same 4096 TRAIN anchors, 112 TRAIN/32 VALIDATION readout
windows, scaler/targets and 2048-update/B8 budget are reused. Initial/final new
banks alone receive the unchanged ridge1 and neural16-tanh/Adam.01/100-step
recipes, three neural fits per view. D36 head counts are 148/660; exact
width-derived seeds are in the registry. Cross-width comparisons also change
head counts/seeds and are development evidence, not equal-budget causal proof.

Trend R² reaches 0.655 linear/0.134 mean neural, with all three neural fits
positive: `0.17724993789041088`, `0.09238170986535554`,
`0.13325491256021194`, copied from the saved readout. Eight of nine unchanged
preferences pass; linear volatility falls
to 0.446 versus the parent's 0.728 and fails its maximum 0.05-loss condition.
Activity, both relation heads and neural volatility retain their declared
scores. Coverage is 100%. GPU training took 93.377385897 seconds including
checkpoint I/O, separately from captures and saved-feature readouts. This is
a stronger trend reference, not an overall winner or promotion. No forecast,
TEST, fresh chronology, synthetic confirmation or serving-checkpoint choice
follows. Original native40 checkpoints remain available.

Next propose a separately declared saved-TRAIN-only diagnosis of level/change
covariance and the existing fitted linear contributions/predictions to
investigate linear volatility loss. Use saved banks and head archives only:
no encoder updates, head refits, old replays or VALIDATION/TEST diagnostic
inputs. Correlated inputs and altered fitting are possible contributors,
not an established cause. Declare a separate new objective or branch design
only after that diagnosis; no width, rate, budget or head rescue. Fresh
chronological confirmation follows a useful balanced candidate. This diagnosis
is proposed, not measured or scheduled. The save milestone governs publication
of this completed source/docs/evidence payload; checkpoints stay local.
