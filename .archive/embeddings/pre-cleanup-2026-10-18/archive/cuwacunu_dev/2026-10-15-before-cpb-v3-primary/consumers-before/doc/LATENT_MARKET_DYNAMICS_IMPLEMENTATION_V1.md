<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Latent market dynamics backbone v1

This implements the deterministic frozen-Encoder backbone in the
[roadmap](LATENT_MARKET_DYNAMICS_ROADMAP.md). It is an engineering implementation,
not evidence that a learned state forecasts markets or improves portfolio results.
The existing Encoder and its checkpoints are reused without changing their training.

Each network has a physical version directory and its own public namespace:

| Component | Source | Responsibility |
| --- | --- | --- |
| State Representation Network | `code/networks/state_representation/versions/v1` | Affine G, frozen TRAIN input transform, independent state width |
| Dynamics Model | `code/networks/latent_dynamics/versions/v1` | Horizon-specific affine A on complete K-step state histories; optional last-state residual |
| Predictive training | `code/networks/predictive_training/versions/v1` | Frozen/blocked/full arms, detached EMA target, pooled variance/covariance, AdamW and typed resume |
| Temporal integration | `code/latent_market/versions/v1` | Exact time/support/availability/role ledgers, native capture, future labels and application runners |
| Saved-array evaluation | `code/evaluation/latent_dynamics/versions/v1` | Mean/persistence/K1/K4 ridge, within-space skill, diversity/retention arithmetic and matched future-risk readouts |

There is no shared mutable network implementation or implicit “latest” network
selection. Predictive training v1 explicitly depends on State Representation v1
and Dynamics v1. New architecture versions get separate folders, configuration,
tests and artifact kinds. Training and inference archives are different types;
neither accepts the other's checkpoint kind. The Encoder remains independently
versioned in its existing family. Reusable network sources are mirrored locally
to `cuwacunu_embedding`; market ingestion and orchestration stay here.

The served flow is `raw history -> Encoder -> native e -> G -> z`. Forecasting
adds `A(G(e history))`; its outputs are in the checkpoint's target-G coordinates.
An affine G/A combination adds no nonlinear expressivity. `serving.h` folds
state maps, forecasts and fitted affine readouts into history-only coefficients
without an inverse or refit, including rectangular/singular maps and residual A.

## Build and engineering checks

Use the existing managed container and a named session:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/latent-market-v1.sh', 'check')
```

The focused target is `test-latent-market-v1`; `build-latent-market-v1` builds the
application executables. All binaries, dependency files and caches stay under
`/opt/cuwacunu/build/latent-market-v1`. No Windows toolchain is used. CPU is used
for ledger arithmetic and ridge/readout fitting; Encoder capture and neural
state learning require CUDA and verify actual tensor/parameter/gradient devices.

Create a synthetic end-to-end input fixture in a **new** container directory:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/latent-market-v1.sh', 'fixture', '/opt/cuwacunu/build/latent-market-v1/tmp/example-001')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/latent-market-v1.sh', 'study', '/opt/cuwacunu/build/latent-market-v1/tmp/example-001/study.conf', '/opt/cuwacunu/build/latent-market-v1/tmp/example-study-001')
```

The fixture defines H4/D4/K4, horizons1/4, strict disjoint chronological roles,
simulated availability, two artificial CUDA updates, B4 and eight distinct
regularizer origins. It uses deterministic synthetic embeddings and no Encoder
execution or market observations. Its card is explicitly an engineering card.
Failed and successful output leaves are preserved; choose a new leaf for a retry.

Verified on 15 October 2026 in the existing managed container: 54 temporal,
47 baseline and 31 future-risk checks pass; the actual CUDA attribution/EMA/
typed-resume/folding fixture passes for all three arms. The full application
workflow completes on synthetic inputs, and resumed inference/state-bank
archives are byte-identical to the uninterrupted two-update run. A separate
synthetic raw-cache witness captures18 native windows through the selected
frozen Encoder, with zero Encoder updates or scaler fits. Seven future-risk
arms score the same50 held-out synthetic origins. The source-only
[engineering receipt](results/latent_market_dynamics_v1/engineering.json)
records scope, actual identities and the preserved initial fixture correction.
None of these checks measures market forecasting or utility.

## Prepare an admitted timeline

`latent-market-v1.sh prepare PLAN.conf NEW_LEAF` uses the versioned
`LATENT_MARKET_V1` configuration. The template is
`code/latent_market/versions/v1/config/preparation.conf.example`.
Paths are relative to the configuration file. It requires:

- An exact prospective card, explicit origin ledger, pinned CPB-v1 checkpoint
  and settings, source/cache manifest and source-closure manifest, each with SHA256.
- Three explicitly ordered distinct raw source caches, or CSV paths with existing
  fresh raw caches. Both CSV and actual mmap bytes must be pinned. The runner
  performs no cache rebuild, downloads or data selection.
- Explicit TRAIN/VALIDATION close-time boundaries, assumed receipt latency,
  history/spacing, latent horizons and future-label horizons. Include h>=H.

Origins are whitespace-separated, one per line:

```text
"pair-001" 1800000000000 1800000000005 train "chronological-block-001"
```

Fields are pair ID, endpoint milliseconds, decision milliseconds, role
(`train` or `validation`) and a dependence-group ID. Quoting permits spaces in
IDs. There is no implicit random selection, TEST opening, interpolation or
row-number shift. SHA256 manifests contain `hash`, two spaces and a path; relative
paths resolve against the manifest directory.

Generate the current compiled application dependency manifest with:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/latent-market-v1.sh', 'source-pins', '/cuwacunu/output/inputs/latent-market-v1/source-001.sha256')
```

The manifest uses compiler dependency files plus build/orchestration sources;
the external LibTorch/runtime remains covered by the managed environment's SDK
verification. Freeze sources before generating it. Admission verifies manifest
bytes and every listed file before and after work. The checkpoint's claimed
historical fit cutoff must be justified by the card/provenance; CPB-v1's archive
does not itself encode its source dataset's timestamp cutoff.

Preparation opens the existing mmap loader with future length zero, captures
each unique required native window once in CUDA eval/NoGrad mode, and verifies
frozen model/scaler state. Each cutoff uses its own actual uniform candle grid.
It uses `cuwacunu::encoder::primary` and admits only the currently selected
CPB-v1.market-01 instance and geometry; the plan does not add a model-selection
toggle or permit an alternate checkpoint as an inference fallback.
Uncapturable windows are recorded in `uncaptured-windows.tsv`; associations
retain missing/invalid reasons rather than compressing the timeline.

Outputs include `timeline.bin`, `pairs.bin`, `future-risk.bin` and a hash receipt.
The timeline contains native e only, exact IDs, timestamps, assumed availability,
semantic channels/features and time-major observed masks. Pair ledgers contain
input and target associations, conservative raw support envelopes, role purges,
mask-aware overlap and invalid associations. Loading independently reconstructs
pair eligibility from the timeline. Binary ledgers are versioned little-endian,
bounded and published without replacing existing files.

Risk labels use positive observed adjacent closes without bridging gaps.
Dispersion is the centered **population standard deviation of one-minute log
returns**, without sqrt(R) or annualization. Downside is the worst cumulative
log loss relative to the origin close. Full-window eligibility, actual support,
target maturity, counts, formula identity and units are saved separately.

## Learn and evaluate a state

`latent-market-v1.sh study STUDY.conf NEW_LEAF [TRAINING_CHECKPOINT]` consumes
saved arrays, never replays Encoder. The template
`code/latent_market/versions/v1/config/study.conf.example` deliberately leaves
experiment-specific budgets and cutoffs unfilled. It requires a pinned card,
timeline/pair hashes and source closure. Every fitted target must mature by the
declared fit cutoff and be fully contained in TRAIN; held-out roles begin later.
TEST rows require the separately registered future confirmation protocol.

Select `frozen_state`, `blocked_prediction` or `full` explicitly. Each is a separate
run with the same prospective architecture, observations, seed and objective
settings when testing attribution. The blocked arm detaches the entire
prediction history while variance/covariance still train G. Target G starts as
an exact detached copy, stays outside AdamW, and follows EMA only on successful
updates. The initial implementation fixes lambda_ret=0 and requires external
retention probes before advancement.

Embedding transforms and initial state reference scales are fitted once on
complete, mature, unique TRAIN windows. Their exact values and endpoints are
saved. The regularizer evaluates one graph-preserving centered sample covariance
over `N_COV` distinct TRAIN endpoints, independently of optimizer `MICROBATCH`.
It rejects repeated origins and reports the n_cov-1 rank bound. Frozen scales
never follow shrinking outputs; diagnostic diversity uses the declared TRAIN
population rather than held-out observations. All-invalid updates skip the
optimizer, regularizers and EMA together under the stored policy.

Artifacts include periodic immutable `training-N.pt`, final `training.pt` and
`inference.pt`, online/target `state-bank.pt`, state predictions, K1/K4 e-space
and target-state baselines, fixed-space scores, update receipts and provenance.
Resume restores G/G_bar/A, standard AdamW moments and named parameter order,
all counters, EMA progress, frozen transforms/reference scales, sampler and
CPU/CUDA RNG states. The complete recipe and identity must match, including
source/executable hashes; a resume writes to a new leaf.

State banks pin the inference snapshot, timeline and pair hashes. Target-space
persistence and ridge use that same snapshot's target G. Issued predictions
require the model fit cutoff to precede the origin decision. Separately named
`fitting_forecasts` in the state bank apply the final frozen predictor to TRAIN
histories solely to fit downstream readouts; they are in-sample features, not
issued historical forecasts or cross-fitted surprise residuals.

`scores.csv` preserves per-horizon role, support and overlap counts. Near-zero
persistence denominators yield N/A. e-space and z-space errors are not directly
comparable; the common future-risk task supplies that comparison.

## Future-risk comparison and remaining gates

The saved-array future-risk module fits the same ridge recipe to matched,
mature TRAIN rows across feature arms and evaluates identical held-out
intersection populations. Prediction receives origins/features but no future
labels or target masks. CPU archives retain the fitted heads, predictions,
target identities and units. The utility CLI and public `risk_readouts.h` are
separate from state learning.

Use `latent-market-v1.sh utility UTILITY.conf NEW_LEAF`, with the
`config/utility.conf.example` template in the application version directory.
It pins all timelines, labels, model/bank and issued-prediction files, fits seven
explicit current/history/forecast feature arms, and saves typed readouts,
predictions and a common-population score table. The synthetic helper
`tests/make_utility_fixture.py INPUT_LEAF STUDY_LEAF NEW_SETTINGS` declares a
separate engineering card for exercising this path. It uses no market inputs.

This backbone does not implement optional nonlinear/MLP arms, auxiliary
objectives, Encoder adaptation, probabilistic dynamics, a realized-surprise
service, independent chronological confirmation, portfolio accounting or live
trading. The supplied utility comparison does not replace the prospective
conventional-risk, capacity/placebo, historical-retention and uncertainty
controls required for a research claim. Those choices, numerical gates and
untouched data selections remain experiment-specific. No research quality run,
automatic promotion, commit, push or publication follows from engineering tests.
