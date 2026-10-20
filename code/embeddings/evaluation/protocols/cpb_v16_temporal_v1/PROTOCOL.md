# V16 temporal reconstruction: prospective protocol

Declared 19 October 2026, Asia/Dubai, before source build, training or access to
March confirmation values. The owner explicitly chose the proposed second
direction, reconstructing temporal changes at several scales alongside values,
with “go for your second choice, let's see”. This authorizes this new controlled
study, not the alternative V4/V15 objective blend. Actual admission, source,
environment and execution identities belong to the local bound records.

## Hypothesis and one controlled change

Test whether additional reconstruction of hidden temporal changes improves
past-window trend and activity information while retaining volatility and
synthetic timing. Control is fresh V4. Candidate is fresh V16, with V4's entire
native36 forward, value Gaussian NLL and detached variance-head input unchanged.
Add no parameters, output coordinates, supervised readout targets or fitted
preprocessing. Native36 remains learned20 + fixed spectral12 + mean4: 293672
learned values and 3888 fixed values, 297560 total across 76 parameter tensors
(75 trainable). No trained ancestor, teacher or historical fit is imported.

For lags **1, 4, 16** on the original one-minute grid, each pair uses the same
channel and feature at t and t-lag, with stride1 and no wrap, compressed-observed
indexing, cross-channel pairs or values outside the retained 128 positions.
Both endpoints must belong to Q, the **original observed-and-hidden target
mask**, disjoint from final visible input. Additional context deletion never
creates targets. Use the existing TRAIN-standardized waveform targets. Sanitize
each endpoint under pair support and promote to CUDA F64 before subtraction:

`e_lag = ((mean[t]-mean[t-lag]) - (stop(target[t])-stop(target[t-lag]))) / sqrt(2)`.

Each L_lag is Huber(delta1): average supported time/feature pairs within each
example/channel, then equally average eligible channels per example, then
eligible examples. Its own pair counts determine eligibility. The auxiliary is
**(L1+L4+L16)/3**, with fixed equal weights even when a lag has no pairs. An empty
lag contributes graph-connected zero and is separately marked unsupported.
Do not renormalize over active lags or skip a base-eligible update because an
auxiliary lag is empty. The full objective is **V4 Gaussian NLL + 0.1 auxiliary**.
The coefficient, delta, fixed sqrt2 endpoint scale and lags are one predeclared
recipe, not fitted or claimed optimal. Do not divide by lag or fit a new scaler.

Auxiliary direct gradients reach the mean decoder/residual/backbone and neither
targets nor variance. Gaussian variance supervision is unchanged and occurs
once. The single shared global clip1 and one AdamW still couple update scaling;
this is not optimization independence. Differencing ignores constant offsets,
which remain supervised by the retained value objective. The published Gaussian
NLL score is separate from V16's composite training objective.

Related earlier RPB-v25 moment reconstruction used visible-bank targets, an
added decoder and a protected trained parent. It is not this hidden waveform
loss on fresh V4 native36. No exact equivalent was found in the inspected
CPB/RPB source/cards; that bounded search does not claim literature novelty.

## Execution, masks and frozen evaluation

Run six fresh trajectories: encoder seeds **946101, 946102, 946103**, each with
V4 control and V16 candidate. Row/mask/context seeds are encoder+1000/+2000/+3000
and matched within each pair. Run order is A/B, B/A, A/B. Each trajectory has
2048 successful updates, B8, at most4096 attempts, AdamW lr0.001,
weight_decay0.0001, global norm2 clip1, mask25%, context deletion15%, dropout0.
Save initial0/final2048 and typed training states; actually resume at512 with
exact parameters, moments, counters and streams. Final2048 is the sole outcome.
No coefficient, mask, checkpoint, seed, head, learning-rate or budget rescue.
New controls are matched replication blocks for this new comparison, not
replays of completed 945101–945103 trajectories or attempts to reproduce scores.

Use the unchanged January2024 BTC/ETH/SOL CACHE88 data and TRAIN-only fits:
days1–21 TRAIN, days22–28 known chronological validation, 4096 calibration
anchors, 112/32 fixed probes, recent128 from H256 legal support. Fixed market
heads remain ridge1 and joint four-output tanh16/Adam0.01/100 steps, reps2701,
2802,2903 and D36 uint64 seeds13088411117524848790,10018485049408248344,
9083751155097190455. Head repetitions are not encoder replications. TEMPO-5
retains masters941101/2/3, 512 TRAIN/256 validation, both TRAIN-fitted input
scalers and intact/30%-deleted views, with the frozen head recipe. No PCA or
native geometry fit is introduced. Calibration and all learned statistics are
fresh and tied to January TRAIN. Preserve genuine F64 encoder input and legacy
F32-loader then F64 market targets; reproduce every original112/32 target and
support bit before new confirmation preparation.

Complete all six training trajectories before development extraction/fixed-head
scoring. Training's fixed-query initial/final receipts are retained without
performance-based decisions. All initial/final checkpoints must pass the
original cutoff/suffix/internal causal proof. Independently audit saved fits,
predictions, support, normalizers, ridge equations and metrics. Actual update1
and2048 temporal witnesses additionally receive independent scalar loss/support
and output-adjoint checks, without producer model arithmetic or optimizer replay.
These bounded witnesses do not reimplement encoder Jacobians, every training
step, global clipping from full parameter gradients or bootstrap intervals.
January fixed-query waveform witnesses cover first8 of112/32; full row means
are checked from saved metrics. March query witnesses cover all32.

The fixed reconstruction query still hides one eight-step patch per eligible
channel. Its lag16 has no both-hidden support: report unsupported/null for that
diagnostic, never zero error or an expanded mask. Auxiliary support is reported
separately from unchanged readout/query coverage. Log all training lag counts
and component losses; preserve the actual first/last witness support.

## Confirmation chosen before values

February is consumed V15 evidence and is development context for this choice.
Declare **March2024 Spot1m BTCUSDT/ETHUSDT/SOLUSDT** as confirmation-only. A bounded
metadata/filename inventory found no recorded March access in inspected local
research; this does not certify unrecorded external access. Choose March by
calendar, not by values. Review and bind fresh official publisher metadata,
license/terms, exact ZIP members and checksums before import. Preserve exact
timestamp units, finite/schema/lattice checks and all rows; no filtering or
gap filling. Raw data/checkpoints remain ignored local artifacts.

March has44640 rows per channel, first open1709251200000, first
close1709251259999, last close1711929599999. Choose32 probe close-row indices
`floor((i+0.5)*44640/32)`, i0..31, with H256 disjoint support inside each calendar
bin. Retain H128 past-window targets and original32 Boolean query geometries;
discard old IDs/labels/support and recompute all new support without mask draws.
Seal all six final checkpoints, original January TRAIN-fitted heads and
development audits before opening March features, predictions or outcomes.
No March calibration, training, head fitting, row replacement, tuning or
selection. Confirmation query MAE/NLL is descriptive and has no gate.

## Prospective outcomes and protection

Primary outcomes are March final **linear R2 for past trend and past activity**,
separately, candidate minus same-seed control. Report all three differences,
their median/range and every neural repetition. A three-pair screen is
**promising** only if BOTH targets have positive median and at least2/3 positive
pairs, median March linear volatility change is at least-0.01, and every
candidate retains all eight unchanged strongest TEMPO protections in January.
Undefined metrics, incomplete support, missing pairs or invalid execution
prevent promising. Otherwise record mixed/negative/inconclusive with exact
failed conditions. This new trend/activity question does not rewrite V15's
volatility/activity question or its result.

Keep the exact22 strongest V3–V13 January predicates from the frozen V14 goal
protocol as a separate complete protection report for every seed/arm. Earlier
leaders and failures remain. Formal advancement requires all applicable frozen
protections, trust and coverage. Historical V4's19/22 has different floors and
cannot rank against these counts. Secondary reports retain all four market
tasks, both heads, all TEMPO views/masters/repetitions, TRAIN/known-validation/
March query metrics, initial/final changes, parameter movement and measured
compute scopes. No invented overall score, automatic promotion, catalog write,
best-seed selection or Mityane application selection. Once viewed, March is
consumed evidence for later recipe choices.

## Operation contracts and Requirements-and-Evidence Record

**OR-05 / CPB-past-readout v1:** input is the exact observation embedding,
checkpoint/native36/mask/support/TRAIN-fit/head identity. Output is four frozen
past-window estimates and validity/source IDs. Required properties are
accessible past volatility/trend/activity/relation at fixed head capacity.
Dependencies are source data, target definition and frozen TRAIN-fitted heads;
baseline is paired fresh V4 with within-source initial checkpoints. Invalid
support, zero target variance, missing artifacts or failed audit abstains.
Claims are observed associations on the stated population/horizon.

**OR-09 / CPB-fixed-query v1:** add fixed query geometry and model-owned decoder;
output standardized masked MAE/Gaussian NLL with eligible counts. Keep masked
reconstruction and readout quality independent. The auxiliary is training
supervision, not an additional operation-quality score or a forecasting target.
TEMPO is a separate OR-05 synthetic-transfer population. No OR-11 forecasting,
identified intervention, decision or composed application utility follows.

The linked RER is this declaration plus the actual resumption, source-copy and
source/protocol reviews, input manifest, publisher/import records, build/CUDA
engineering, per-phase receipts, arithmetic audits, pre-confirmation seal and
complete final report/independent review. Exact hashes/configuration/environment
are in those bound records. Failed and partial attempts remain preserved.
All execution uses C++, Node and Bash in the existing managed container;
no Python fallback, dependency/image/container changes, cleanup, commit, push
or publication. Missing executable details block their dependent execution
until resolved; documented illustrative plans are not additional prerequisites.
