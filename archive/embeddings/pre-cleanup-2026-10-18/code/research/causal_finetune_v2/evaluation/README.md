# Fixed quality tools for CPB-v2

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

These tools are prospective source for the separately declared
`doc/CAUSAL_FINETUNE_V2_CARD.json`. They do not authorize execution. Root
orchestration must admit exact source/configuration/binaries, verify initial
baseline parity, actual CUDA learning and typed resume, then pass the final
causality proof before the single final quality pass.

`extract_main.cpp` includes only the owned CPB-v2 public header plus physical
local CACHE88 record headers. It requires the final typed checkpoint at causal
update2048 and copies no baseline tensors. `--data-root /cuwacunu` reads original
raw cache doubles, the saved market row IDs, raw-parent parity witness, saved
TEMPO-5 values/masks/labels and frozen scalers. Output is exclusive. It performs
18 market and768 synthetic CUDA calls at batch8, with immutable model/scaler/bank
guards, producing final native36 banks only. It fits nothing.

`market_read_main.cpp` and the physically owned `deps/` regression files retain
the fixed market ridge/neural arithmetic exactly. `synthetic_read_main.cpp` and
the eight owned classification dependency files retain the CPB-v1 fixed recipe,
seeds, source populations and intervals. These readers include no encoder and
open no checkpoint. Every dependency's original path, SHA256 and copied SHA256
are in `dependency-provenance.json`; provenance does not create an execution
dependency on those earlier paths. Inherited license notices remain intact.

The parent Makefile includes `evaluation.mk`. The single build target
`build-cpb-v2-quality` produces:

- `$(BUILD_DIR)/causal_finetune_v2_extract`
- `$(BUILD_DIR)/causal_finetune_v2_market_read`
- `$(BUILD_DIR)/causal_finetune_v2_synthetic_read`

After admission, orchestration invokes the final extractor with:

```text
--checkpoint FINAL_INFERENCE_2048 --settings V2_SETTINGS
--output NEW_RUN/features --data-root /cuwacunu
```

The market reader retains its frozen arguments:

```text
--train-targets /cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/train-targets.pt
--validation-targets /cuwacunu/output/runs/market-capacity-study/shared-MKT-HIST-2-v1/features/validation-targets.pt
--method cpb_v2 NEW_RUN/features/market/cpb_v2-train.pt NEW_RUN/features/market/cpb_v2-validation.pt
--output-directory NEW_RUN/market-readouts
--card-sha256 30aa70ce6f3999d9f6a991c1b298516db41caf3db276cacb75d91c9d3dcba222
--input-manifest-sha256 828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8
--dataset-id MKT-HIST-2
```

The synthetic reader accepts `--features NEW_RUN/features/synthetic --output
NEW_RUN/synthetic-readouts` and validates all six cohort/policy banks before
fitting. It fits each permitted final head once; three fixed neural seeds are
repetitions, not independent encoder retrainings.

`report.py` uses the JSON machine card, immutable CPB-v1 result hash, passed
initial-parity/quality receipts, final saved readouts and training/query records.
Its arguments are `--run`, `--training`, `--data-root`, `--card` (JSON),
`--initial-parity`, `--quality-receipt` and exclusive `--output`.
It writes four market task panels, four TEMPO-5 policy/view panels, physical
MAE/RMSE, full-precision repetitions/conditional intervals, separate query MAE/NLL,
training/cost tables and exact prospective gate decisions. It does not execute
an encoder, fit a head, choose a checkpoint or modify a baseline/leader pointer.

Initial CPB-v2 quality is reused from CPB-v1 only after exact initial model,
scaler/bank and saved-feature parity. No initial feature export, initial head
refit, baseline encoder replay, synthetic training, PCA, head tuning or silent
retry belongs to this workflow. Missing or mismatched inputs stop the attempt;
all failures, partial outputs and claims must remain preserved.
