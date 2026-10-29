# Fresh RPB training and resume example

This C++ example delivers a configurable **RPB-v22** training method over the
existing v20 backbone. It constructs fresh learned weights, fits the shared
streaming TRAIN scaler, performs masked waveform updates, saves/resumes the
existing typed AdamW state, and exports endpoint embeddings. It needs no market
reader, teacher or trained parent. The historical native36 balanced leader and
V30 instances retain their own trained ancestry and results; neither supplies
weights for this example.

The CLI source is [rpb_family_train.cpp](../../../../scripts/rpb_family_train.cpp).
The input boundary is the existing
[observation-window dataset](../../../../../data/observations/OBSERVATION_WINDOWS.md).
The [engineering protocol](../evidence/PROTOCOL_2026-10-22.md) defines verification
scope. A runnable example or a passing engineering check does not establish
representation quality or application admission.

## Build and create an artificial TRAIN archive

Run from the repository root in PowerShell. Choose unused session/output names
when repeating. Project compilation and execution stay in the existing managed
Linux container; PowerShell performs orchestration and receipt inspection.

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/task.sh', 'hx16-rpb-portable-demo-2026-10-22', '-j2',
  'build-rpb-family-trainer', 'build-rpb-family-fixtures'
)

.\container.ps1 -Action exec -Command @(
  '/opt/mityane-geometrics/embeddings/build/hx16-rpb-portable-demo-2026-10-22/rpb_observation_windows_fixture',
  '--output', '/mityane/.temp/embeddings/runs/hx16-rpb-portable-demo-2026-10-22/synthetic-TRAIN.pt',
  '--channels', '1', '--history', '72', '--features', '3', '--rows', '6',
  '--sampling-interval', '1'
)
```

The fixture creates six artificial rows. Each row is one instrument. The single
configuration uses timeframe channel `11` (`4h`) and 72 raw ticks.
[single.conf](../config/single.conf) retains the last H65 ticks, P8,
native D8 and W16. [multi.conf](../config/multi.conf) instead uses
C3/H129/F9/P8/D32/W32, torch ordinals `11`/`18`/`4` (`4h`, `1M`, `1m`), and a
learned20+fixed12 native layout.
Generate an appropriately shaped archive for each configuration. These are
small engineering cohorts, not proposed useful trained instances.

## Train fresh to total update2

```powershell
.\container.ps1 -Action exec -Command @(
  '/opt/mityane-geometrics/embeddings/build/hx16-rpb-portable-demo-2026-10-22/rpb_family_train',
  '--settings', '/mityane/code/embeddings/encoders/raw_patch_bottleneck_mae/family/config/single.conf',
  '--dataset', '/mityane/.temp/embeddings/runs/hx16-rpb-portable-demo-2026-10-22/synthetic-TRAIN.pt',
  '--output', '/mityane/.temp/embeddings/runs/hx16-rpb-portable-demo-2026-10-22/fresh-2',
  '--initialization-seed', '221601', '--steps', '2', '--max-attempts', '10', '--batch-size', '2',
  '--row-seed', '221611', '--mask-seed', '221612', '--context-seed', '221613', '--save-every', '1'
)
```

The archive must declare role TRAIN. Each row is cropped at its aligned endpoint
to its configured retained H before observed values enter the scaler or model.
The grid must be ordered and uniform with the configured sampling interval.
Every declared TRAIN row contributes once in source order, in batch-size chunks,
to `StreamingScaler`'s supported-value-anchor/Chan CPU-F64 statistics. Overlapping
windows count repeatedly; the example does not deduplicate source observations.
The scaler freezes before weights are initialized from the explicit seed.
Actual data/split membership remains the caller's responsibility.

Rows are selected with replacement using
`mt19937_64(training_counter_seed(row_seed, absolute_attempt)) % row_count`.
The original whole-patch masker uses the mask counter seed and original patch
positions. There is **no extra context deletion**; the context seed is retained
and checked as a recipe field. Targets are hidden observed normalized waveform
cells. The hierarchical Huber loss averages cells within channels, channels
within eligible examples, and eligible examples. Targets are detached. The
literal spectral prior remains frozen; learned backbone, relation projection
and decoder receive their actual gradients.

Each eligible attempt runs `zero_grad`, forward/backward, global clip norm1 and
one standard AdamW step: lr0.001, decay0.0001, betas0.9/0.999, eps1e-8,
AMSGrad off, one registered parameter group. Unsupported masks consume an
attempt with no backward, optimizer state, weight decay or completed update.
These optimizer choices are fixed by the existing codec. There is no scheduler,
gradient accumulation or distributed training surface in this example.

RPB supplies a bidirectional representation of the endpoint's retained window.
Reconstruction of a hidden earlier tick may use observed later ticks within
that window. Cropping excludes ticks beyond the endpoint, but does not make
this a prefix-causal per-timestep encoder.

## Resume to total update4

Retain the independently recorded scaler identity from the fresh fit receipt.
Supply it to resume; do not discover it by trusting the state being loaded.
The receipt and archive still require the consumer's own integrity/admission
checks. The scaler's content identity is not a cryptographic authenticity claim.

```powershell
$rpbFitReceipt = Get-Content '.temp/embeddings/runs/hx16-rpb-portable-demo-2026-10-22/fresh-2/preprocessing-fit.json' -Raw | ConvertFrom-Json

.\container.ps1 -Action exec -Command @(
  '/opt/mityane-geometrics/embeddings/build/hx16-rpb-portable-demo-2026-10-22/rpb_family_train',
  '--settings', '/mityane/code/embeddings/encoders/raw_patch_bottleneck_mae/family/config/single.conf',
  '--dataset', '/mityane/.temp/embeddings/runs/hx16-rpb-portable-demo-2026-10-22/synthetic-TRAIN.pt',
  '--output', '/mityane/.temp/embeddings/runs/hx16-rpb-portable-demo-2026-10-22/resumed-4',
  '--resume', '/mityane/.temp/embeddings/runs/hx16-rpb-portable-demo-2026-10-22/fresh-2/training-final.pt',
  '--expected-scaler', $rpbFitReceipt.scaler_identity,
  '--initialization-seed', '221601', '--steps', '4', '--max-attempts', '20', '--batch-size', '2',
  '--row-seed', '221611', '--mask-seed', '221612', '--context-seed', '221613', '--save-every', '1'
)
```

`--steps` is a **total completed-update target**, so resuming update2 with
`--steps 4` requests two further updates. `--max-attempts` is a total attempt
ceiling. Budgets/save cadence may change; budgets cannot precede saved progress.
Initialization seed remains explicit on resume, even though weights are loaded.
Resume restores original scaler/model/AdamW/counters without fitting again.

The recipe binds the exact dataset, settings and executable SHA256 bytes,
resolved configuration identity, initialization seed, batch/chunk size,
source/split identities, sampler/mask/context rules, stream seeds and fixed
optimizer. Byte-identical paths may move. Changing settings comments or building
a different executable changes the recipe. A deliberate change needs a new
instance or explicit migration, not a weakened resume check.

Save only between completed attempts. The archive retains counters and their
seed rules, not arbitrary ambient CPU/CUDA RNG, loader iterator, pending
accumulation or partially completed attempts. Exact continuation is scoped to
the named same runtime/device and draw algorithms; it is not a cross-SDK or
cross-hardware guarantee. The example requires actual CUDA float32 execution
with dropout0. Historical study samplers/context policies remain separate.

## Outputs and failure handling

| File | Meaning |
| --- | --- |
| `inference-0.pt` | Fresh weights/config/frozen scaler, before updates; fresh runs only |
| `preprocessing.pt`, `preprocessing-fit.json` | Fresh scaler, source/split/recipe, fit-row count and independent scaler identity receipt |
| `steps.jsonl` | Actual attempt/update counters, source row IDs, hidden-mask and preclip-gradient SHA256, Huber loss, gradient norm and target support |
| `training-N.pt` | Optional periodic typed state, named by completed update count |
| `training-final.pt` | Final resumable state, including a valid partial state on attempt exhaustion |
| `inference-final.pt` | Separate typed inference export, reloaded and checked against live numerical state |
| `embeddings.pt` | Source-order native `[B,D]`, sample/channel validity, observed/patch support counts, semantic IDs, row IDs, endpoints, interval and schema/config/scaler/source/split/recipe/inference bindings |
| `run.json` | Actual status, fit/actions/budgets/counters, numerical state identities, artifact sizes/SHA256 and the explicit absence of quality evaluation |

`embeddings.pt` has kind `rpb_family_endpoint_embeddings_v1`, explicit architecture
and native-layout tags, and output selector `z_contextual_global`. The actual
published archive is reopened: its metadata, tensor shapes/dtypes and values
must agree exactly with the verified inference output. Its schema identity
binds the F64/bool retained-window geometry, semantic IDs and interval. Native
coordinates have the same numerical meaning as the final inference instance;
equal width does not establish compatibility with another instance. This raw
export is separate from application ModelRef, selected-artifact/publication,
capture, descendant fitting and trusted admission.

The output leaf must be new; existing inputs and checkpoints are preserved.
`--save-every 0` disables periodic saves. If the total attempt ceiling is reached
first, the CLI writes a valid partial `training-final.pt`, inference/embedding
exports and `run.json` with `attempt_budget_exhausted`, then returns exit2.
That receipt does not report success. Resume the partial state with suitable
total budgets and a new leaf when its recipe remains appropriate. Other
exceptions return exit1; preserve failed leaves and use valid periodic states
for recovery. Events cannot recreate missing optimizer moments.

Allmissing TRAIN data is refused during fresh fitting because required scaler
coordinates lack support. A positive-fit dataset with fewer than three observed
patches per channel can fit but remains training-ineligible; it produces explicit
skips and bounded recovery. Numerical failures during a step do not promise
transactional rollback; recover from a previously published valid state.

## Bounded example check

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/task.sh', 'hx16-rpb-portable-example-check-2026-10-22', '-j2',
  'test-rpb-family-training-example'
)
```

The Node orchestration runs native C++ at both named configurations, compares
split/resumed and continuous rows/masks/losses/preclip gradients/numerical state,
optimizer/scaler/embeddings, and verifies artifact hashes independently. It also
checks valid TEST-role refusal, wrong data/settings/seeds/scaler, inference-as-
resume, immutable outputs, nonfinite/unknown-ID/allmissing-fit refusal, and
insufficient-patch-support recovery without optimizer actions. Its test-only
[dataset variant writer](../tests/dataset_variant.cpp) produces new artificial
archives while preserving the shared observation writer.

This in-memory, single-device example supplies portable numerical mechanics.
Applications own real feature semantics, TRAIN/calibration admission, budgets,
substantial training, durable recovery, selected artifacts and downstream utility.
