# CPB-v4 saved-training independent audit

This owned standard-library auditor reads saved JSON/CSV and inert C++ archive
storage. It never imports Torch, instantiates a model or repeats training. The
prospective audit card declares checks and limits. V3 audit bytes and the frozen
88-file V4 training capsule remain unchanged.

Run only in the existing managed cuwacunu container after the main run exists:

```sh
python3 audit_training.py --run RUN --attempt ATTEMPT --source FROZEN_SOURCE \
  --data-root /cuwacunu --audit-card training-audit-card.json \
  --engineering-proof ENGINEERING_JSON --gradient-proof GRADIENT_JSON \
  --output EXCLUSIVE_JSON
```

The output is exclusive. Unsupported archive formats fail admission rather than
falling back to model execution. Full112/32 fixed-query scalar reductions are
checked; detailed per-cell Gaussian arithmetic covers only the saved first8 rows
per split/stage. Live CUDA gradients and actual typed continuation are explicitly
source-bound producer evidence, augmented by independent saved tensor movement,
optimizer-state/counter checks and pinned engineering evidence.
