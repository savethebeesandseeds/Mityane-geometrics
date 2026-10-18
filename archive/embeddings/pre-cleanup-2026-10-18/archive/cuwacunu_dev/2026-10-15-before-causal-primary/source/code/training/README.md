# Bounded market-history training runner

Copyright (c) 2026 Waajacu. All rights reserved. See the root LICENSE.

## Training duration and dimension studies

The additive [study runner](src/market_study_main.cpp) preserves the completed
pilot. Its [continuation card](../../doc/MARKET_TRAINING_CONTINUATION_CARD.md)
extends the original 512-update AdamW state to 2048 updates under the separate
tag `RPB-v22.market-01.alt-01`. Original raw, initialized and 512-update controls
are reused; only the new 2048-update native features receive new fixed heads.

The [capacity card](../../doc/MARKET_CAPACITY_STUDY_CARD.md) defines four fresh
2048-update profiles. H128/W64/D32 is the baseline; each alternative changes
only W128, D64 or H256. Every profile uses common H256-legal endpoints, one
TRAIN scaler and labels/query masks for the final 128 candles. The shared panel
contains 112 TRAIN and 32 VALIDATION rows, separate from the pilot's 192/64.

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/run-market-studies.sh', 'continuation')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/run-market-studies.sh', 'capacity')
```

The launcher builds in its own container cache and checks that source files
did not change during compilation. It trains and captures each encoder only on
CUDA. CPU readouts consume saved feature arrays without replaying the encoder.
The first optimizer update must demonstrate actual CUDA gradients and changed
weights. Logs distinguish updates, window presentations, query MAE and GPU
allocator peaks. Reconstruction losses and head scores are separate measures.

All output leaves are exclusive. Capacity preparation intentionally refuses an
existing shared-bank directory, including an incomplete one, preserving it for
inspection. A further study needs a new declared bank/configuration; removing
or overwriting evidence is not a normal retry. The initial pilot command below
remains available for its original contract.

This application layer joins the independent mmap loader, core file settings,
primary v22 encoder and independent evaluation archives. It does not import
Hero, Runtime, Lattice, the legacy optimizer counter clamp or a forecasting head.
The [prospective card](../../doc/MARKET_HISTORY_PILOT_CARD.md) fixes the first
experiment; [settings](../../config/market_history_pilot.conf) describe its
source/split/budget. This bounded command refuses a changed experiment recipe.
The portable encoder remains configurable through its own settings/API.

From PowerShell in this repository, use the existing managed container:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/fetch-market-pilot.sh')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'market-engineering', '-j2', 'test-streaming-scaler', 'test-training-state')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'market-readout-engineering', '-j2', 'test-market-readouts')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/run-market-pilot.sh')
```

The fetch command reuses checksum-verified existing files. The runner emits an
exclusive `output/runs/market-history-pilot/run-*` leaf before work begins, and
retains failed leaves. Data, model artifacts and outputs are ignored by Git.
Every sample has a complete past window inside its own chronological role;
future length is zero. Streaming TRAIN population moments retain only `[C,F]`
state and the current batch. Actual encoder encode/decode/training is CUDA;
CPU handles source loading, statistics, past-history labels and fixed readouts.
Inference captures are saved once per checkpoint/view and reused by readouts.

Resume from an existing run's `checkpoints/training-N.pt`:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/run-market-pilot.sh', '--resume', '/cuwacunu/output/runs/market-history-pilot/run-EXAMPLE/checkpoints/training-128.pt')
```

Resume creates another exclusive leaf, verifies actual CSV/mmap-cache bytes,
source closure and executable identity, exact anchor selections, TRAIN scaler,
geometry, recipe and row/mask/context seed bases, and restores standard AdamW
moments plus completed/attempted counters. It verifies and copies the exact
initialized assets through their checksum ledger bound to the training archive.
Inference checkpoints cannot be used as training states. A completed512 state
can finish interrupted artifact extraction with zero extra optimizer updates.
Timing and new CUDA-gradient evidence describe only the executed segment;
restored optimizer state is explicitly identified separately. The artificial
CUDA fixture proves exact next-update parity, including an ineligible skip.

Artifacts include separate inference/training archives, source/selection/input
receipts, initialized and trained native features, raw+mask features, all-past
targets, fixed-query reconstruction, the optimizer-loss trace, RAM/CUDA allocator
peaks and fixed readout fits/predictions/report. CUDA memory is allocator usage,
not a universal maximum input size. The synchronized segment training time
includes checkpoint writing and excludes scaler fitting, feature extraction
and CPU readouts. Head width-dependent parameter counts remain explicit.

Market loading/targets/run orchestration stay here. Reusable streaming scaler
and training-state headers/tests are mirrored to `cuwacunu_embedding`; see
[portable training support](../../doc/STREAMING_TRAINING_SUPPORT.md).

## Separate frozen-level development recipe

RPB-v23.market-01.alt-01 has its own runner/card/run directory. It warm-starts
the typed RPB-v22.market-02 at2048 updates, uses the exact prepared MKT-HIST-2
bank/scaler and performs2048 additional updates on only its change projection.
The protected child includes the decoder. Its
[results](../../doc/MARKET_FROZEN_CHANGE_RESULTS.md) miss the development goal;
this is a preserved stopped development run, not a serving/default selection.

Focused gate: bash code/scripts/task.sh market-study -j2 test-v23-frozen.
Dedicated runner: bash code/scripts/run-market-frozen-changes.sh.
Both run only in the existing managed container. The runner creates a new
exclusive leaf and fits only the new final native features; previous heads
and encoder captures are reused. It does not implement a resume CLI.
The [portable guide](../encoders/rpb_changes/FROZEN_LEVEL_TRAINING.md)
documents the typed resume/guard sequence and inherited exposure.

## Separate change-block RPB-v24 experiment

The [v24 report](../../doc/MARKET_SEPARATE_CHANGE_RESULTS.md) records one
completed2048-update native40 CUDA run after a protected2048-update v22 parent.
The separate runner is bash code/scripts/run-market-separate-changes.sh;
its engineering gate is bash code/scripts/task.sh market-study -j2 test-v24.
Run both only inside the documented managed container. This stopped card has
no resume CLI or automatic rerun schedule. Its typed portable state supports
exact resume with the declared recipe and protection guard.

Initial/final40-coordinate banks are saved once and fitted by the unchanged
CPU evaluator; all older controls are reused. Prefix32 is compared exactly
against saved trained v22 features at both captures, without replaying that
old encoder. The new block's dimensions/decoder count and width-derived head
seeds are declared in its own settings/card. Primary v22 stays unchanged.
