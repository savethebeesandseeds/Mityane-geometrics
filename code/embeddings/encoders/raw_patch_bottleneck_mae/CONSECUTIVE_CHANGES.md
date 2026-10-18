# RPB-v23 consecutive-change candidate

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../LICENSE.

The separately tagged RPB-v23 wraps unchanged RPB-v22 level/content-relation
paths and adds zero-start consecutive-change moments to the learned native
coordinates. Native32 remains20 learned+12 literal fixed spectral coordinates.
The public input/output contract remains `[B,C,H,F] -> [B,D]`, with explicit
observed masks, semantic IDs, endpoints and sampling interval. No PCA follows
the encoder. Historical defaults, v22 checkpoints and synthetic results remain
preserved.

The six `consecutive_change_*.h` portable headers, artificial CUDA fixture and
`config/consecutive_changes.conf` have recorded source-parity checks. They depend
only on reusable v22/v20/model/settings code, without a market loader. Explicit
build/check targets are `rpb-v23` and `test-rpb-v23`; run them in the managed
embedding container. The recorded GPU fixture is the initial engineering execution; copied files alone do not claim another standalone test run.

Changes require two adjacent visible values and retain original-time quarters.
Unsupported/constant moments are zero; minimum support2 and RMS floor1e-6
guard arithmetic. Counts are not projected. At C3/F9, ten moments per channel/
feature form270 values and a20-by270 zero-start projection adds5400 parameters.
The fixed spectral tail and decoder are inherited. H>=17 and the base geometry
rules still apply. Distinct typed immutable inference/training formats bind
architecture, configuration, scaler and standard AdamW/counter state.

The [version registry](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md) records the matched
market development recipe and measured instance. Fresh chronological confirmation
and synthetic quality remain separate work.

The additive [frozen-level recipe](FROZEN_LEVEL_TRAINING.md) imports a trained
v22 child, preserves its complete state and updates only the change projection.
It uses the same v23 inference arithmetic and a distinct instance/recipe.
Its test-rpb-v23-frozen target is separate; the recorded CUDA gate passed.
The [market result](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/EMBEDDING_VERSIONS.md)
misses the full goal; keep the original fresh v23 instance and the v22 reference.
