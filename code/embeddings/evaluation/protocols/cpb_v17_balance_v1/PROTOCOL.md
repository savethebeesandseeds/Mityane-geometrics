# V17 temporal-strength comparison: prospective bounded protocol

Declared **22 October 2026, Asia/Dubai**, before this study's data preparation,
model execution, diagnostic or training. The human owner accepted the proposed
TRAIN-only gradient diagnostic followed by one controlled weaker-temporal-loss
comparison, while explicitly requiring that configurable construction, complete
training code, resume and export remain the development foundation.

This declaration fixes the first bounded comparison. Exact source, executable,
configuration, dataset and artifact hashes belong in an execution manifest
written before their dependent execution. A changed protocol is an explicit
prospective amendment with preserved earlier bytes, never a silent rescue.

## Question and controlled change

Test whether reducing temporal supervision from **0.1** to **0.03** improves the
balance of past-window trend/activity and volatility at an early fixed budget.
V16's measured trend/activity gains motivate retaining temporal supervision;
its volatility/reconstruction regressions motivate this strength ablation.
Objective dominance or gradient conflict is a hypothesis, not an established
cause. The coefficient 0.03 is one prospective choice, not a fitted optimum.

Both fresh arms use the corrected configurable CPB component and the same
native36 architecture. The complete objective is

`L = GaussianNLL + lambda * (HuberLag1 + HuberLag4 + HuberLag16) / 3`.

Control has `lambda=0.1`; V17 candidate has `lambda=0.03`. Preserve lags1/4/16,
Huber delta1, fixed sqrt2 endpoint scale, within-channel/eligible-channel/
eligible-example reductions, original hidden-observed target support Q at both
endpoints, and graph-connected zero with fixed one-third weight for an empty
lag. Targets are detached. Variance-head input gradient remains stopped;
Gaussian supervision occurs once. One AdamW and one shared global clip1 remain.
No parameters, output coordinates, labels or inference features are added.

The selected historical V16 seed946103/final2048 remains the research reference.
Fresh controls here are **V16-equivalent objective controls under this new
bounded recipe**, not reproductions of the historical V16 trajectory or claims
that its source, calibration, sampler or budget is unchanged.

## Reusable component and instance configuration

The experiment builds on the maintained family APIs for supported geometry,
fresh construction, TRAIN fitting, masking, state and inference loading/export.
Keep learning recipe and state identities with every instance. The coefficient
must be bound to the training recipe and checked on resume; inference archives
remain distinct from resumable training state. No frozen V16 or earlier family
source is rewritten. A new learning recipe and runner have new source identities.

The study configuration is **C3/H128/F9/P8/native36**, semantic channel IDs0/1/2
for ordered BTC/ETH/SOL. Level width32 contains learned20 plus literal spectral12;
change width4 gives native36. Encoder W64, three layers, four heads, feedforward
width256 and decoder hidden width128 retain the selected architecture. Mask
ratio0.25, dropout0, scale floor1e-6, LayerNorm epsilon1e-5 and minute interval
60000 are explicit instance settings, not universal factory assertions.

The new entry must retain the family's supported alternative geometry/capacity
surface. Focused artificial construction/update/export/resume/refusal evidence
establishes any changed recipe interface before market execution. Numerical
implementation uses C++; orchestration/record keeping uses Node/Bash. No Python,
new SDK, dependency, environment image, container or host toolchain is needed.

## Data identities and temporal boundary

Reuse the admitted January2024 Spot1m CACHE88 population already local to Mityane.
No download or new period is required. Original source paths below retain their
evidence identities; resolve them from the repository root and verify exact
bytes before use. The common directory prefix is

`.temp/embeddings/closeout/cpb-v14/attempt-001/portable-external-evidence/data/mkt-hist-1/csv/`.

| Ordered cache | Bytes | SHA256 |
| --- | ---: | --- |
| `BTCUSDT-1m-2024-01.cache.bin` | 3928320 | `67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58` |
| `ETHUSDT-1m-2024-01.cache.bin` | 3928320 | `0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa` |
| `SOLUSDT-1m-2024-01.cache.bin` | 3928320 | `364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7` |

CACHE88 is the original packed little-endian88-byte record with close time at
offset48. Validate complete records and the exact minute lattice. The nine
ordered features are open/high/low/close price, base volume, quote volume,
number of trades, taker-buy base volume and taker-buy quote volume. Preserve
genuine observation F64 bits, support and semantic ordering. Unsupported values
do not become observed zeros.

TRAIN close-time bounds are **1704067259999 through1705881599999** (January1–21).
Known chronological development validation is **1705881659999 through
1706486399999** (January22–28). No January29–31 observations are selected.
Each observation window ends at its explicit cutoff, retains128 ticks and stays
within its role. The generic archive carries original cutoff/row IDs, times,
CPU F64 values, Boolean support, channel IDs, source/split identity and TRAIN or
VALIDATION role. A source declaration does not independently prove membership;
the preparation receipt must verify actual selected bounds and cache association.

For a legal range `[begin,end]`, grid step60000 and history `h`, define
`first=begin+(h-1)*60000`, `available=(end-first)/stride+1`, and anchor i of n as
`first + floor(i*(available-1)/(n-1))*stride`. Use stride60000 for overlapping
calibration/training pools and stride`h*60000` for nonoverlapping readout probes.
All selections are deterministic from timestamps before observation values.

## Stage A: TRAIN-only, zero-update gradient diagnostic

Create exactly **64** evenly spaced legal TRAIN H128 anchors with the formula
above and overlapping stride60000. Archive them in increasing source order.
The diagnostic processes these first64 typed TRAIN rows as eight consecutive
**B8** batches, in fixed order, with masking seed **949103** and absolute mask
attempts0–7. Additional context deletion is **none**. These are new diagnostic
masks, not a replay of the historical trajectory's masks or update events.

Load the selected immutable V16 seed946103/final2048 inference state and its
embedded original frozen preprocessing. Its checkpoint path is

`.temp/embeddings/runs/cpb-v16-temporal-001/attempt-001/subruns/seed-946103/candidate/training/checkpoints/inference-2048.pt`,

1295473 bytes, SHA256
`b6f3cbe711235307f362bf2f999571d65d00c85825f8a4a5c2fbc4fa4242e51e`.

An evaluation-only in-memory copy may expose its parameters through the
maintained family interface. It is explicitly a trained historical state, not
fresh ancestry for either Stage B arm. Perform **zero optimizer updates, zero
preprocessing fits and zero saved-weight changes**. Verify the checkpoint hash,
model parameters/frozen-state identities before and after. Clear diagnostic
gradients before any inference/export check; no diagnostic-trained checkpoint
or adapted serving artifact is published.

For each batch report NLL and unweighted temporal loss, per-lag support/loss,
actual learned-parameter gradient norms and dot/cosine between NLL and temporal
gradients. Unused-gradient parameters contribute explicit zeros in the same
registered learned-parameter order. Report full learned vector and available
backbone/mean/variance groups with the exact group mapping. A zero-norm cosine
is null, not zero alignment. Report combined gradient norms and clip1 factors
for both coefficient0.1 and0.03 from those same component gradients. Unsupported
batches/lags remain explicit; do not expand targets to obtain support.

These parameter-gradient measurements are stronger than saved output-adjoint
witnesses for this local question, but describe only eight TRAIN batches at one
selected trained state. They do not prove global interference, identify a cause
of readout regressions, or establish an optimal coefficient. The diagnostic is
**descriptive**: it cannot tune the coefficient, seeds, masks, population or
Stage B budget. Numerical/identity/support defects stop dependent execution for
repair with an explicit record; diagnostic values do not choose a rescue arm.

## Stage B: fixed-budget paired development screen

Use fresh encoder seeds **947101,947102,947103**, each with control0.1 and
candidate0.03. Row/mask/context seeds are encoder+1000/+2000/+3000, matched within
each pair. Context deletion is none, with the unused context seed still bound.
Use the portable example's counter-seeded MT19937_64 modulo-row sampling with
replacement; record exact row IDs and attempted/completed update counters.
Run order is control/candidate, candidate/control, control/candidate.

Each arm starts from fresh learned weights, optimizer state and preprocessing.
Both use exactly **1024** evenly spaced legal TRAIN H128 calibration/training
windows. Fit the maintained rounded-parent **Chan count/mean/centered-M2**
population statistics and unchanged relative-change bank, in increasing source
order and B8 chunks, before construction. Both arms use the same fitter, data,
ordering and fit identity policy. Verify matching fitted tensors and fresh
parameter initialization within a seed pair; do not import historical weights,
statistics, heads or optimizer moments. The stable fitter,1024 calibration
windows, sampling and context policy differ from historical V16 and are shared
by both arms, preventing those engineering changes from entering only one arm.

Fix **256 successful B8 updates**, with **at most512 total attempts** per arm:
six trajectories and1536 logical successful updates if all complete. AdamW
lr0.001, weight_decay0.0001, default betas0.9/0.999, epsilon1e-8, AMSGrad false,
global learned-gradient norm2 clip1, mask25% and dropout0. No scheduler,
accumulation, distributed training, early stopping, rescue seed, alternative
coefficient, selected intermediate checkpoint or outcome-dependent extension.
An ineligible batch consumes an attempt but no update. Exhaustion is an honest
partial failure, not a completed256 trajectory.

Save initial0, periodic resumable state at64/128/192 and final256 training and
inference artifacts. Carry full source/configuration/recipe, preprocessing,
initialization, optimizer, progress and counter identities. Verify declared
resume and coefficient mismatch refusal in bounded engineering fixtures before
the scientific screen; a resume of an actual interrupted run retains its saved
state and exact continuation record. Count all physical work honestly. Exact
state continuation does not establish zero recomputation after external failure.

Prospective execution detail, fixed before admission: run each trajectory in
two native processes. The first starts fresh and stops at total update128;
the second loads that exact typed training-final state and stops at total256.
Both use the same total-attempt ceiling512, with saved attempts and counter
streams carried across the boundary. The second process does no preprocessing
fit. This is a planned resume witness within the256-update budget, not an
additional trajectory or a comparison of interrupted and uninterrupted quality.

## Fixed final256 January development readouts

Finish and seal all six final256 instances before readout extraction, fitting
or comparison. Use exactly **112 TRAIN and32 January known-VALIDATION** probe
anchors from the original MKT-HIST-2 deterministic **H256 nonoverlapping**
selection. The encoder retains recent128 from each legal H256 support window;
the four target definitions and observed support retain original recent128
MKT-HIST-2 meanings. Copy the declared original **legacy F32-loader-to-F64**
target arithmetic, not a new all-F64 label definition. Keep genuine F64 encoder
input. Bind the exact target source implementation and selected IDs before
execution and verify row associations/support; equivalent source relocation
has its own identity.

Fit only the unchanged **ridge1 linear** readout recipe on the112 TRAIN probes,
using their TRAIN-only feature/target preprocessing and fixed original target
definitions for past trend, activity, volatility and relation. Save fitted
normalizers/coefficients, predictions, targets, support and metrics with their
associations. Produce candidate-minus-same-seed-control differences for all
four tasks, medians/ranges, every arm and initial/final interpretation where
actually measured. No neural heads, head repetitions, bootstrap, TEMPO timing
suite or fresh-month scoring is performed in this bounded screen.

At final256 also report fixed-query masked reconstruction MAE/Gaussian NLL under
the unchanged saved TRAIN/known-validation query geometry if available with
verified IDs and bindings. These are descriptive and supply no rescue or
promotion gate. An unsupported diagnostic remains null; in particular do not
expand a single hidden patch to manufacture lag16 query support.

### Predeclared screen decision

The screen is a **development continuation signal** only if all six trajectories
and arithmetic/identity/support checks are valid, and:

- Paired median linear trend R2 change is at least0, with at least2/3 pairs
  nonnegative.
- Paired median linear activity R2 change is at least0, with at least2/3 pairs
  nonnegative.
- Paired median linear volatility R2 change is at least-0.01.

Report exactly which conditions pass/fail. Equal scores meet the nonnegative
rule but are not an improvement. Report whether any gain was observed; no
strict improvement can be inferred from an all-tie pass. Missing/undefined
metrics, incomplete support, exhausted budgets, failed arithmetic or absent
pairs prevent a positive continuation signal. Relation and reconstruction are
descriptive. This early screen cannot automatically promote V17, replace the
reference, select the best seed, update application selection or establish
full-budget superiority.

The historical strongest eight TEMPO protections and22 January conditions stay
unchanged in their original records; they are **not rerun here**. Both V16
candidates and fresh V4 controls previously missed the complete eight-condition
timing guard. That alone does not identify the temporal objective as the cause.
No current timing-retention claim follows without actually measuring it.

## Confirmation and evidence boundaries

January, February and March2024 have informed earlier choices and are consumed
evidence. This January comparison is explicitly development, not fresh
confirmation. No new month, outcome population or synthetic cohort is opened
by this protocol. A later full study requires a separately declared fresh
confirmation population, checked exposure/access, actual data rights/bindings,
appropriate full-budget comparisons and timing evaluation. It is not silently
authorized by a positive256-update screen.

The operation scope is OR-05 **past-window linear readout** with exact
representation, TRAIN fit, data/target/support/normalization and ridge1 identity;
baseline is the paired fresh0.1 control under the common bounded recipe. OR-09
fixed-query reconstruction is separate descriptive evidence where measured.
The Requirements-and-Evidence Record consists of this prospective declaration
plus the source/configuration/input manifests, implementation review, actual
CUDA/build checks, fitting/gradient/training/resume/export receipts, saved
head equations/predictions and independent arithmetic/source review. Invalid
support, zero target variance, nonfinite data or failed provenance abstains.
There is no forecasting, intervention, decision or composed application claim.

Keep the network and method exportable: complete owned source/dependency/license
inventory, configurable examples, learning recipe, tests, typed artifacts and
resume behavior, rationale/failures and consumer mapping travel together.
Historical snapshots, receipts, datasets, checkpoints, failed attempts and
original version files remain preserved. Outputs remain under ignored named
`.temp/embeddings/runs/` leaves and builds/temp under `/opt/mityane-geometrics`.
No commit, push, public checkpoint, application docking or external chat message
is part of this bounded study. Shared GPU capacity and actual CUDA execution
must be checked before numerical work.
