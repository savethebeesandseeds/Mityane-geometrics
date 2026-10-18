# Controlled pairs: protocol v2

Policy: `EMBEDDING_EVALUATION_POLICY.md` version 1.1. Evidence stage:
`development`. This protocol does not issue acceptance decisions.

Each run saves `evaluation-card.json` with its actual input schema, dimensions,
source budgets, tasks, seeds and declared comparisons before generation/fitting.
This document describes the recipe; the instantiated card governs that run.

| Item | Definition |
| --- | --- |
| Population | Independently generated source trajectories; both transformed variants stay in one split |
| Defaults | 3 channels, 32 uniform steps, 3 features; raw CPU float64, interval one, endpoint H-1 |
| Tasks | Reversal/order, level, amplitude, signed cross-channel lag; tasks selected before scores |
| Splits | 32 training, 16 validation and 32 testing source pairs per task/run by default |
| Randomness | Separate source assignment, signal, mask, pair order, adapter initialization/training and probe streams |
| Metadata | IDs, units, grid and endpoint declared; mask coordinates and adapter-added metadata count in dimensions |
| Controls | Raw histories plus support, masks/metadata, registered adapter controls including matched untrained exports and descriptors where available |
| Linear tier | Frozen ridge, penalty one; fitted only on valid training rows |
| Interaction tier | Separate tanh-16 probe, Adam 0.01, 100 updates; no selection using final scores |
| Compression | Train-fit normalization/PCA with centered-row and numerical-rank guards; default matched global 12 and joint concatenation 36 |
| Support | Declared per-surface validity; all required channels for complete concatenation; invalid placeholders excluded from fits |
| Population comparisons | Explicit left/right/tier pairs saved before data; each pair has its own fixed common-valid intersection |
| Reporting | Conditional accuracy, correct/total counts, coverage/abstentions, unsupported fits, validation/test scores and grouped intervals |
| Uncertainty | Paired source-group percentile bootstrap within each run; three development seeds by default |
| Decision | No promotion; confirmation/cost/coverage thresholds and across-training-run uncertainty are incomplete |

Native dimensions remain separately reported. PCA is an explicit matched-tier
transform, including when requested width equals native width. This and typed
surface names/paired populations differ from the initial v1 driver; do not
reinterpret archived v1 scores as v2.

The legal oracle receives observed values and support only. It cannot access
clean hidden values or labels. The initial clean/light-missing synthetic recipe
requires a supported raw solvability reference; this is a task-construction
check, not an assumption of perfect recoverability on arbitrary real data.

Labels are used only by the shared readout/scoring path. Adapter fit callbacks
receive training observations and permitted metadata without labels or held-out
data. Fresh adapter budgets are disclosed separately and need not match compute
or exposure in this wiring/development protocol.

Applicability matters: one-channel cards can omit lag; configurations must
validate against card geometry and metadata. Future chronology, forecasting,
external-pretraining, robustness or domain tracks require their own versioned
cards and evidence, rather than changing this recipe after inspecting scores.
