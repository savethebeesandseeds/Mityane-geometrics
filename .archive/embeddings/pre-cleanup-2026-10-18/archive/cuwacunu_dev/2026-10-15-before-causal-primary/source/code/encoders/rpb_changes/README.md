# RPB-v23 consecutive-change candidate

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

This separate encoder retains an unchanged RPB-v22 child and adds a zero-start
path for consecutive changes. Primary RPB-v22 settings/checkpoints remain intact.
Use `cuwacunu::encoder::changes` and `config/market_studies/encoder_changes.conf`
explicitly; this candidate is not selected by `config/encoder.conf`.

The caller supplies TRAIN-scaled values and a Boolean observed mask of rank4
`[B,C,H,F]`, semantic channel IDs, endpoints and sampling interval. The candidate
exports one native rank2 `[B,D]` embedding for the whole multichannel window.
It does not export a separate change embedding or require additional input
features. The inherited scaler, data contract and loader remain independent.

For each channel/feature, valid changes require both adjacent coordinates to be
visible. Ten generic moments describe their global mean/RMS and four time
quarters' mean/RMS relative to global RMS. The quarters retain original-time
positions; gaps are not compressed or bridged. Minimum support2 and RMS floor
1e-6 guard arithmetic. Counts are never projected. Hidden storage is sanitized
before subtraction, so training masks cannot leak target values.

At C3/F9, the bank has270 values; its bias-free20-by270 projection adds5400
trainable values to the learned20. The fixed12 tail and existing level/relation
paths and decoder remain unchanged. The added projection starts at literal
zero and consumes no random draws: an equal-seed initialized v23 export equals
v22 exactly. All existing v22 child weights can learn during the fresh trajectory.
No PCA or head transformation follows the native32 export.

Geometry uses the inherited configurable C/H/F/D/P/W contract. The spectral
base requires H>=17, D greater than four times the channel-pair count, CUDA
float32 and its existing mixer settings. Partial-patch model inference/masked
forward is checked at H33. The inherited coordinate-erasure training helper
requires complete patches; the bounded market runner fixes H128/P8 accordingly.

Distinct immutable typed inference and training archives reject v22 files as
v23 (and vice versa) rather than guessing architecture from parameter shapes.
Standard AdamW states, parameter order, counters, identities and scaler content
are validated for recovery. The focused CUDA fixture verifies exact next-update
resume as well as masks, poisoning, semantics, constants and initialization.

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'change-engineering', '-j2', 'test-v23')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/run-market-changes.sh')
```

The [prospective market card](../../../doc/MARKET_CONSECUTIVE_CHANGE_CARD.md)
declares one fresh2048-update MKT-HIST-2 experiment with the exact existing
prepared bank and unchanged fixed heads. Market loading/orchestration stays
in integration. Six portable headers, the fixture and settings are mirrored
into sibling `cuwacunu_embedding`; its explicit target is `test-rpb-v23`.

An additive [frozen-level training recipe](FROZEN_LEVEL_TRAINING.md) retains a
typed trained v22 parent and updates only the change projection. Its separate
card/runner/instance preserve this original fresh-training record. Use
`test-v23-frozen` for its protection/resume fixture and
`code/scripts/run-market-frozen-changes.sh` for the separately declared market
run. The recipe changes no primary selection or v23 inference arithmetic.
