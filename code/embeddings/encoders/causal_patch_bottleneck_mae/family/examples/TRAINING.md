# Fresh training and resume example

This example constructs `CPB-family` from configuration, fits preprocessing on
the caller-declared TRAIN archive, trains, saves a typed training state and
resumes it. It needs no original weights or market reader. The selected V16
seed946103/final2048 checkpoint remains a frozen research comparison reference;
it does not supply weights for this fresh instance.

The numerical CLI is [cpb_family_train.cpp](../../../../scripts/cpb_family_train.cpp).
Its typed data boundary is documented in
[OBSERVATION_WINDOWS.md](../../../../../data/observations/OBSERVATION_WINDOWS.md).
The [compatibility receipt](../evidence/compatibility-2026-10-22.json) records
the actual passing engineering execution and its scope. This guide describes
the runnable example and its contract; it does not itself establish a test pass,
representation quality or application readiness.

## Build and make a small dataset

Run from the repository root in PowerShell. Choose an unused session and output
paths when repeating the example. Compilation and native execution stay inside
the existing managed container; PowerShell only orchestrates them.

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/task.sh', 'cpb-family-demo-2026-10-22', '-j2',
  'build-cpb-family-trainer', 'build-observation-window-fixture'
)

.\container.ps1 -Action exec -Command @(
  '/opt/mityane-geometrics/embeddings/build/cpb-family-demo-2026-10-22/observation_windows_fixture',
  '--output', '/mityane/.temp/embeddings/runs/cpb-family-demo-2026-10-22/synthetic-TRAIN.pt',
  '--channels', '2', '--history', '40', '--features', '2', '--rows', '8',
  '--sampling-interval', '1000'
)
```

The fixture creates eight explicitly synthetic rows, each with two semantic
channels `[11,18]`, 40 raw ticks, two features and a 1000-unit timestamp interval.
The supplied [paired.conf](../config/paired.conf) uses the most recent 33 ticks,
patch6, encoder width32, level12/change3 and native15. Raw supplied history N and
configured retained history H are separate quantities. This cohort is a tooling
fixture, not a proposed real TRAIN population or a useful trained model.

## Train fresh to total update2

```powershell
.\container.ps1 -Action exec -Command @(
  '/opt/mityane-geometrics/embeddings/build/cpb-family-demo-2026-10-22/cpb_family_train',
  '--settings', '/mityane/code/embeddings/encoders/causal_patch_bottleneck_mae/family/config/paired.conf',
  '--dataset', '/mityane/.temp/embeddings/runs/cpb-family-demo-2026-10-22/synthetic-TRAIN.pt',
  '--output', '/mityane/.temp/embeddings/runs/cpb-family-demo-2026-10-22/fresh-2',
  '--seed', '221101', '--steps', '2', '--max-attempts', '10', '--batch-size', '2',
  '--row-seed', '22001', '--mask-seed', '22002', '--context-seed', '22003',
  '--learning-rate', '0.001', '--weight-decay', '0.0001', '--clip-norm', '1',
  '--save-every', '1'
)
```

Fresh fitting visits every declared TRAIN row once, in source order and
batch-size chunks. It crops at each cutoff before fitting the parent input
scaler and relative-change bank, freezes them, then initializes fresh weights
from seed221101. The fit is window-weighted; overlapping source observations
are not deduplicated. Actual split membership and an appropriate calibration
population remain the caller's responsibility.

Training samples rows with replacement using
`mt19937_64(training_counter_seed(row_seed, attempt))` modulo row count. Masking
uses its saved counter seed and attempt. This example performs no additional
context deletion; it still binds the supplied context seed. Each supported
attempt runs one backward pass, one global learned-parameter clip and one
AdamW update. Unsupported masks consume an attempt without an update.

## Resume to total update4

Use the same executable, dataset, settings, batch size, stream seeds and
optimizer options. Replace `--seed` with `--resume` and use a new output leaf.

```powershell
.\container.ps1 -Action exec -Command @(
  '/opt/mityane-geometrics/embeddings/build/cpb-family-demo-2026-10-22/cpb_family_train',
  '--settings', '/mityane/code/embeddings/encoders/causal_patch_bottleneck_mae/family/config/paired.conf',
  '--dataset', '/mityane/.temp/embeddings/runs/cpb-family-demo-2026-10-22/synthetic-TRAIN.pt',
  '--output', '/mityane/.temp/embeddings/runs/cpb-family-demo-2026-10-22/resumed-4',
  '--resume', '/mityane/.temp/embeddings/runs/cpb-family-demo-2026-10-22/fresh-2/training-final.pt',
  '--steps', '4', '--max-attempts', '20', '--batch-size', '2',
  '--row-seed', '22001', '--mask-seed', '22002', '--context-seed', '22003',
  '--learning-rate', '0.001', '--weight-decay', '0.0001', '--clip-norm', '1',
  '--save-every', '1'
)
```

`--steps` is the **total completed-update target**, so resuming update2 with
`--steps 4` requests two further updates. `--max-attempts` is also a total ceiling,
not an additional-attempt budget. Stop budgets and save cadence may change;
the target cannot precede the saved completed count. Resume loads the original
fitted statistics, model, AdamW moments/steps, update/attempt counts and stream
seeds. It never refits preprocessing. An inference checkpoint is not a training
resume state.

The recipe binds SHA256 identities of the exact dataset, settings and executable
bytes, together with batching, row policy, stream seeds and optimizer options.
Moving byte-identical inputs is allowed; changing even settings comments or
recompiling to different executable bytes changes the recipe. Use a new run or
an explicitly specified state migration rather than weakening resume checks.
The dataset's declared source/split identities and archive hash are bound too.

## Outputs and failures

| Output | Purpose |
| --- | --- |
| `inference-0.pt` | Fresh instance and frozen preprocessing before any update; fresh runs only |
| `preprocessing.pt` | Fresh-fit scaler/bank, fit-window count and identities; fresh runs only |
| `steps.jsonl` | Attempt/update counts, selected source row IDs, objective components and gradient norm |
| `training-N.pt` | Periodic typed state when `--save-every N` cadence is reached; filename uses completed count |
| `training-final.pt` | Final typed model/preprocessing/optimizer/progress state, including a valid partial state if attempts run out |
| `inference-final.pt` | Final inference export, reloaded and checked against the live model |
| `run.json` | Actual status, identities, budgets, counters, fitting scope and artifact SHA256/size records |

The output directory must not already exist. Existing files, datasets and
resume states are never overwritten. `--save-every 0` disables periodic saves;
the final state is still written after the bounded loop completes.

If the attempt ceiling is reached before the update target, the CLI writes the
valid partial final state and a `run.json` with `attempt_budget_exhausted`, then
returns exit2. That is not a successful completion. Resume that typed partial
state with suitable total budgets if its scope remains appropriate. Other
exceptions return exit1; an interrupted or failed leaf may contain only earlier
periodic states and event fragments. Preserve it, identify a valid saved state,
and use a new output leaf. Events alone cannot reconstruct missing optimizer
moments or prove updates beyond the last saved state.

This is a single-device, in-memory dataset example with dropout0 and explicit
counter streams. It has no scheduler, gradient accumulation, distributed/sharded
training, production run manager or automatic recovery service. Cuwacunu can
dock its reader, split verification, budgets, durable ledger and operational
recovery around these portable APIs. Actual application utility needs its own
evaluation.
