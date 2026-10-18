# RPB-v24 separate change block

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../LICENSE.

This explicit candidate preserves a trained v22 encoder's complete native
embedding and appends a small learned change block. Primary v22, the original
v23 joint-training attempt and its frozen additive variation remain intact.
Use the separate API/settings deliberately; do not relabel old checkpoints.

The caller supplies TRAIN-normalized rank4 values `[B,C,H,F]`, a rank4 Boolean
observed mask, semantic channel IDs, endpoints and sampling interval. The
encoder returns one rank2 window vector `[B,D_level+D_change]`, not a sequence
or separate per-channel embedding. No market labels or extra input features
enter the encoder, and no PCA follows it. For this experiment the input is
`[B,3,128,9]`, the preserved prefix32 is20 learned+12 fixed, and the new suffix
has8 learned coordinates: native40.

The unchanged v23 generic change bank computes ten moments per channel/feature
from actually adjacent visible values: global mean/RMS and four original-time
quarters' mean/RMS relative to the global RMS. Two visible endpoints and minimum
support2 are required; missing gaps are never bridged. Counts are not projected.
The bias-free270-to8 projection starts with fan-in uniform random weights.
The original32 coordinates are concatenated verbatim, never mixed into new
coordinates. Old level encoding runs under NoGrad and remains independent.

The unchanged v22 decoder receives only its32-coordinate prefix under NoGrad.
A separate bias-free8-to3456 linear decoder reconstructs a rank4 residual in
canonical channel/time/feature order, then follows the caller's semantic IDs.
It starts at zero, so initial waveform reconstruction equals the parent exactly.
Training minimizes the same masked waveform Huber loss on past observations.
The first update opens the new decoder; projection gradients become active on
the second update. First-step projection decay is not credited as learning.
Only2160 projection+27648 decoder parameters update in the market geometry.

`SeparatedChangeConfig` contains an unchanged v22 `level` config and
`change_width`. File settings use `LEVEL_EMBEDDING_SIZE` and total
`EMBEDDING_SIZE`; their positive difference defines the change width. C/H/F/P/W
remain configurable under inherited constraints, including H>=17. Increasing
H also increases this simple decoder's parameter count. Partial model history
is checked at33; the coordinate-erasure market recipe requires complete P8
patches and fixesH128. This is not a new universal input-size limit.

Warm-start from a hash-pinned typed v22 checkpoint with matching level config,
TRAIN scaler and declared exposure. Copy values into the registered child with
`separated_change_training::warm_start_level`; replacing its public holder
would break registration and is rejected. Use standard AdamW over the complete
registered parameter list, retain validated flags and call `FrozenLevelGuard`
before/throughout training. The new training format rejects protected gradients
or optimizer states. Snapshot checks prove exact child values/buffers; saved
initial/final prefix banks prove preservation without replaying the old encoder.

Distinct typed v24 inference and model/scaler/AdamW/counter training archives
bind the new dimensions, layout, bank rules and recipe. Resume with the same
parent/source/split/recipe/scaler identities and guard; no generic v22/v23
reader can substitute for this typed loader. The new update counter describes
additional branch training, while parent exposure is declared in the recipe.

Build/check in the managed container:

```bash
bash code/scripts/task.sh rpb-v24 -j2 test-rpb-v24
```

The [recorded experiment](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md) fixes one
initial/final MKT-HIST-2 experiment and unchanged evaluator. Wider native40
means164/724 head parameters and different width-derived seeds compared with
native32's132/596; both new views use the same D40 seeds. An initial control
distinguishes architecture information from learned improvement. Keep data,
targets and fixed-head evaluation outside the portable model. The six headers, settings and focused fixture are available here;
the check target is `test-rpb-v24`. A copy is not another test run.
