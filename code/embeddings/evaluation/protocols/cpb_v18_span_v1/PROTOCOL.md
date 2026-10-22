# V18 contiguous-target training: prospective fixed-budget comparison

Declared **22 October 2026, Asia/Dubai**, before this study's engineering/model
execution, resumed control updates, candidate training or new readout fitting.
The owner asked to keep searching for a better encoder and clarified that the
encoder or training form should change while the evaluation can stay the same.
This declaration fixes one training-mask intervention and a **1024-update
primary endpoint**. There is no256-update gate or adaptive budget extension.
Exact source, executable, configuration, input and artifact pins are admitted
before dependent execution. A repair preserves earlier bytes and failed output;
it cannot silently change the scientific question, population or budget.

## Question and one controlled change

Does hiding one contiguous span of original patch positions during training
improve the information accessible to the unchanged linear readouts, compared
with the existing scattered-patch training recipe at the same1024-update total?

Control uses the existing uniformly shuffled observed-patch subset. Candidate
V18 instead selects one legal contiguous **original-grid** span independently
for each example/channel, with the same rounded target-patch quota. Keep the
architecture, all learned/fixed tensor shapes and constructor draws, genuine
supported configuration surface, preprocessing, loss, optimizer, row sampler,
mask rate and evaluation unchanged. At study geometry H128/P8 with all16
patches observed, each arm hides four patches per channel. The candidate chooses
one of13 four-patch starts and retains12 visible patches. No cross-channel
synchronization, mask-rate sweep, curriculum or additional context deletion is
introduced. Contiguity is a training-form change, not a new evaluation mask.

The unchanged loss is

`L = GaussianNLL + 0.1 * (HuberLag1 + HuberLag4 + HuberLag16) / 3`.

Retain original observed-and-hidden Q support at both temporal endpoints,
Huber delta1, fixed sqrt2 endpoint scaling, the existing pair/channel/example
reduction hierarchy and fixed equal lag weights. Empty lags contribute their
graph-connected zero; do not renormalize over active lags or manufacture targets.
Targets remain detached and variance-head input gradients remain stopped.
Gaussian supervision occurs once, followed by one shared global clip1 and one
AdamW update. The contiguous mask changes eligible lag-pair support as an
intended consequence; log the candidate's actual lag counts/support rather than
claiming they match the scattered arm. The preserved control executable did not
log per-lag pair counts; do not present missing fields as measured counts. Any
optional later mask-only reconstruction from recorded rows/counters must be
labeled reconstructed and is not an added execution prerequisite. With complete market support, B8 has6912 observed
hidden target cells in either arm. No readout label enters encoder training.

The previous V17 coefficient0.03 candidate failed its early continuation rule.
It is **not extended** here. V18 retains coefficient0.1. The historical V16
seed946103/final2048 remains the selected research reference, separately from
this fresh-initialization comparison. No old V16 checkpoint or fitted state
initializes a candidate.

## History and limits of the hypothesis

Inspected CPB versions already changed variance routes, uncertainty coordinates,
mean/variance objectives, clipping, temporal pooling/moments and temporal loss
strength. Current and archived RPB/CPB masking implementations select shuffled
observed patch positions. The original RPB design specification mentions
span/patch masking as general guidance; the inspected implemented recipes and
cards did not establish this contiguous-target comparison. This is a bounded
repository-history statement, not a claim of literature novelty. V12's mask-only
dispersion weighting changed pooling support, not target-mask geometry. Plain
native-view consistency was already present in historical RPB-v9.

The hypothesis is that longer missing intervals may encourage temporal
structure rather than scattered reconstruction shortcuts. That mechanism is
not established. V17's local trained-state gradient diagnosis did not support
broad raw temporal-gradient dominance; its small global ratios do not measure
AdamW-preconditioned update contributions or every fresh-training point.

## Portable mask contract

Use the same cutoff-first retained logical history, patch length P, original
patch positions and supported partial-final-patch convention as the family.
Let K=ceil(H/P), and let an original patch be observed if any of its real
retained cells is observed. Private padded cells never create observations or
targets. For M observed patches:

1. If M<3, select no targets and retain explicit ineligibility, as before.
2. Otherwise use `q=clamp(floor(mask_ratio*M+0.5),1,M-2)`, exactly the existing
   quota. Enumerate original-grid starts s in increasing order,0 through K-q,
   whose q consecutive patches are all observed. At least two observed patches
   remain visible by the quota. Do not join observed patches across natural gaps.
3. If no start is legal, select no target patches for that channel and record
   `no_legal_contiguous_span`. Do not silently fall back to scattering, shorten
   q, wrap around, bridge gaps or alter the row. Other eligible channels may
   still support the batch under the unchanged loss hierarchy.
4. For N legal starts, use MT19937_64 and unbiased bounded-integer rejection:
   in uint64 arithmetic let `threshold=(uint64_t(0)-N)%N`; draw until
   `draw>=threshold`, then select index `draw%N`. N=1 still consumes one draw.
   No draw is needed for an ineligible/no-legal-start channel. This specifies
   bounded integer selection, not a promised fixed number of rejection draws.
5. Use the existing counter-derived mask seed for the absolute attempted batch,
   one MT19937_64 generator per batch, and the original example/channel traversal
   order. Channels are sampled separately; no shared start is imposed. Expand
   selected real patches to the logical hidden mask and validate through the
   existing complete-patch/partial-padding helper. Q is observed AND hidden;
   naturally absent cells are never zero-valued targets.

Uniformity is over legal starts. Edge cells consequently have different
inclusion probabilities from interior cells; do not claim uniform cell or
patch marginal inclusion. Support changes can affect later generator draws
within a batch, as prescribed by this explicit stream contract. Candidate mask
starts, legal-start counts, target/visible counts and abstention reasons remain
observable engineering/training evidence. Original control logs retain their
actual recorded schema; no absent mask/support field becomes a measurement.

## Shared architecture and TRAIN fitting

Study geometry remains **C3/H128/F9/P8/native36**, ordered BTC/ETH/SOL semantic
IDs0/1/2, minute interval60000. Native coordinates are learned20 + fixed
spectral12 + mean-change4. Level width32, encoder W64, three layers, four heads,
feedforward256, decoder hidden128, mask ratio0.25, dropout0, scale floor1e-6 and
LayerNorm epsilon1e-5 match the completed V17 common recipe. These are instance
settings, not the family's only supported input or latent shape.

Candidate construction uses the maintained rounded-parent Chan count/mean/
centered-M2 TRAIN fitter and unchanged relative-change bank. Fit exactly1024
existing TRAIN windows once, in increasing source order and B8 chunks. Bind the
same source/split/fitting recipe as the old controls, before fresh initialization.
The candidate's initial parameter-state, scaler and bank numerical identities
must match its same-seed control's recorded initial identities. They are byte-
derived FNV engineering witnesses, not cryptographic equality proofs. The
engineering fixture additionally compares actual parameter/preprocessing tensors.
No candidate receives learned weights, optimizer moments or fitted statistics
from a control, historical V16 or coefficient0.03 model.

## Reuse existing data and immutable control prefixes

Use the already admitted V17 generic observation archives; no new data
preparation, download, period or fitting population is introduced. Root prefix:

`.temp/embeddings/runs/cpb-v17-balance-001/attempt-001/`.

| Archive | Role/count | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `data-train/calibration-TRAIN.pt` | TRAIN1024 | 32920080 | `7020ebf1d6c2b793893a9c0ccb8ae524e76c478c6389e31204c3e92dd88efcaa` |
| `data-probes/probe-TRAIN.pt` | TRAIN112 | 3604752 | `a9b6531ee0457c0411386300565279a558651df47a0b231b3f118088b8bc8778` |
| `data-probes/probe-VALIDATION.pt` | VALIDATION32 | 1033232 | `4dc66210cc76793b6223bb8af9faad259679d45a1e2faf316a23e57642fa664c` |

The V17 preparation receipts and study admission bind genuine CPU F64 values,
Boolean support, exact minute times/cutoffs, row/channel/source identities,
CACHE88 sources and actual role membership. TRAIN close times are
1704067259999–1705881599999 (January1–21); known validation is
1705881659999–1706486399999 (January22–28). Readout selection uses original
nonoverlapping H256 support, retaining recent128 encoder/target observations.
The original ordered caches retain their byte identities:

- BTC: `67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58`.
- ETH: `0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa`.
- SOL: `364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7`.

Reuse **all three** completed coefficient0.1/scattered control TRAIN states at
logical256. They were freshly initialized and fitted in the completed V17
study, with256 physical updates and256 attempts each. Their known January256
scores are exposed development evidence, not a new blind control cohort. No
best control, seed or intermediate checkpoint is selected. Preserve the entire
original V17 study and its negative0.03 verdict unchanged.

For seed S, prefix paths are
`trajectories/seed-S-control/resume-256/{run.json,training-final.pt}` below the
V17 root above. Exact original pins are:

| Seed | Report bytes / SHA256 | TRAIN-state bytes / SHA256 | Initial numerical identity suffix |
| --- | --- | --- | --- |
| 947101 | 3776 / `ec264b4d9b8ef726ba3facd87199c51e993347cdcdf9e6b4d64f88901ec2fe48` | 3767751 / `460f8b4fb4e76c8c68b8d83a261841b2ede20595609e8dd3341237c6af8242f9` | `17426e39fea5d2fd` |
| 947102 | 3776 / `50faf02804c0c86cf9270cdb2b7d7366e6ca6561a247558405e52eefb4dc6f4f` | 3767879 / `46ca742f1fcfd3a438e9f1c47e04e4bae2ac2c2d18bd1a136944a991a2a10f88` | `09bd37d81f312f52` |
| 947103 | 3776 / `eacae8e3c66224d102888f1fbd8086f0daf09ff5d219c285227526153ed20e43` | 3767879 / `1cbcdc87a57ed8e55939f071ec97159d2051ddafea8e3d85aee487c8f435260f` | `dd3614ec72f6bfa5` |

Full initial identity prefix is `cpb-family-state-fnv1a64-`. Each control has
scaler identity `rpb-scaler-v1-fnv1a64-d6fb768d6fa7ac09` and bank identity
`rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-687aaa9aed358ed5`.
Its original settings are the622-byte
`code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v17/config/research.conf`,
SHA256 `f48767f0a49261033fc0e291e51327bbcde07c5d9e0769e4cb03a973085c5a4b`.
Continue controls with the **unchanged original**848288-byte executable
`/opt/mityane-geometrics/embeddings/build/cpb-v17-balance-001/cpb_v17_train`,
SHA256 `767c94e29f0687f881522a2045cd23ebcd1290db7e74a7d97fcc59e55d8c71a4`.
The old TRAIN identity binds those exact executable/settings/data bytes;
do not relabel a newly compiled binary as the old executable.

## Fixed1024 schedule and honest work accounting

Encoder seeds are **947101,947102,947103**. Row/mask/context streams remain
seed+1000/+2000/+3000, with context deletion **none**. The row sampler is the
unchanged counter-seeded MT19937_64 modulo-row sampling with replacement. Match
actual row IDs and absolute mask-counter attempts within each pair; target masks
are intentionally different. Use B8, AdamW lr0.001, weight decay0.0001,
betas0.9/0.999, epsilon1e-8, AMSGrad false and global norm2 clip1.

The fixed primary outcome is **1024 successful updates per arm**, with at most
**2048 total attempted batches** from each trajectory's own original attempt0.
Save resumable state every128 successful updates and final1024 training and
inference artifacts. Cadence and total stop budgets are excluded from the old
trainer's numerical recipe identity, permitting this declared continuation;
sampling, optimizer, data, source/settings and counter identities remain exact.

For every seed:

- **Scattered control:** load its original typed256 state, continue256→640,
  then actually resume that new typed640 state and continue640→1024. Each new
  invocation performs384 physical successful updates; it fits preprocessing
  zero times. Preserve original initialization and optimizer moments/counters.
- **Span candidate:** start fresh at0, fit the1024 TRAIN windows, train0→512,
  then actually resume its own typed512 state and train512→1024. Each invocation
  performs512 physical successful updates. Resume never refits preprocessing.

Run sequentially on the shared GPU in order control/candidate for947101,
candidate/control for947102, control/candidate for947103; complete both planned
invocations for each arm before proceeding to the next scheduled arm. Preserve
initial candidate0, the inherited control0 evidence and all intermediate states.
Intermediate saves are engineering/continuation evidence, not alternative
outcome checkpoints. No intermediate head fitting/scoring,256 gate, optional
stopping, replacement seed, coefficient tuning, rescue mask or outcome-dependent
extension occurs. Attempt exhaustion publishes honest partial state and fails
the completed screen; it is not silently reported as1024 successful updates.

If all six trajectories complete, logical training totals **6144 updates**.
Of those, **768 control-prefix updates already occurred in V17** and are reused,
not rerun. New physical scientific work is **5376 updates**:
three candidates×1024 plus three control continuations×768. Report separately
the768 inherited updates,5376 new updates,6144 logical total, actual attempts,
fits, twelve planned new training invocations, any failed/repeated physical work
and artificial engineering updates. Do not count historical work as newly
executed or imply a complete old-plus-new cost that unavailable records cannot
establish. V17's coefficient0.03 trajectories are preserved and never extended.

## Engineering, artifacts and execution admission

Before market updates, verify the unchanged scattered path against its original
mask/loss/gradient/update behavior, and the new span policy's real original-grid
membership, quota, missing/partial support, no-legal-span abstention, no wrap and
uniform-start rejection arithmetic. Cover supported native36 and a meaningful
alternative geometry, fresh construction, actual updates, typed inference,
future-value/mask causality, exact resumable state and wrong-kind/recipe refusal.
Save observed support and actual work; an engineering pass does not establish
quality. A failure stops dependent execution for explicit reviewed repair.

V18 state must bind the span policy, quota, legal-start/sampling/abstention
semantics, coefficient0.1, source/configuration, input/split/fitting recipe,
initialization, model/preprocessing state, optimizer moments/order and progress/
counter streams. Existing control archives retain their original V17 typed
identity and exact continuation requirements; they are not rewritten as V18.
Equal native width does not establish artifact compatibility. Keep inference
state distinct from TRAIN state. No implicit Torch RNG state is promised beyond
the declared dropout0/counter-driven scope.

Use only the existing managed Linux container and pinned SDK, C++ numerical
code with Node/Bash orchestration, root build/task entries and named ignored
run/build directories. Verify actual CUDA and shared GPU availability. No
Python, host toolchain, dependency/image substitution or application deployment
is introduced. Preserve frozen V16, the existing family numerical headers,
all V17 source/receipts/bundles and original source/checkpoint/statistic bytes.
Source/config/executable/input/capsule manifests precede their dependent phase.

## Unchanged final1024 evaluation

Seal **all six final1024 inference/TRAIN states before any new probe extraction,
readout fitting or comparison**. Reuse the exact existing112/32 generic probe
archives and original target archives. Do not regenerate selections, targets,
head hyperparameters, validation masks or populations. The target archives are
under the original portable MKT-HIST-2 `features/` root recorded by V17:

- TRAIN8782 bytes, SHA256
  `521d8662f498bce27ad26a41c4723bbebb9d2ab6b924cfbcce7132c25b7e43e6`.
- VALIDATION4677 bytes, SHA256
  `cbd08bd01c728cfbb7a500922852573c45f87dd98821b42e20ee202a7a743181`.

Keep genuine F64 encoder input and exact legacy F32-loader→F64 target arithmetic,
target definitions/units/order, row/group/cutoff associations and support. Each
frozen instance receives its own unchanged TRAIN-standardized **ridge1** fit:
TRAIN population means/scales with1e-8 floor, centered normal equations,
unpenalized intercept and penalty1. Only checkpoint-kind loading, the expected
update binding1024 and evidence names may be adapted in the producer; preserve
the mathematical target/head/scoring code and independent scalar audit. Bind
the actual adapted source/binary before first use and document any source-path
mapping. This is not a change in the test mechanism.

Save features, fitted means/scales/weights/intercepts, predictions/targets,
validity and row/group/cutoff IDs in witnesses. Report all four tasks' R2, MAE,
RMSE and coverage for every arm and all candidate-minus-control differences,
medians and ranges. Independently audit each saved ridge using scalar long-
double normalizers, centered normal equations and Cholesky, without shared
Torch matrix/solve/scoring arithmetic. Retain V17's absolute1e-9+relative1e-8
comparison tolerance and exact legacy target/association checks. All six audits
and complete finite support are required for a positive screen.

No neural heads, fixed-query reconstruction panel, TEMPO timing, bootstrap,
confidence interval, new month or initial/intermediate readout is run here.
Training loss/support logs are not substituted for those absent evaluations.

### Predeclared continuation rule

At the fixed1024 endpoint, call the development continuation criterion passed
only if all six trajectories/identity/support/arithmetic checks are valid and:

- Trend median paired linear R2 change is≥0, with at least2/3 pairs nonnegative.
- Activity median paired linear R2 change is≥0, with at least2/3 nonnegative.
- Volatility median paired linear R2 change is≥-0.01.

These are the unchanged V17 early-screen predicates, now evaluated solely at
the predeclared1024 endpoint. Relation remains descriptive. All ties may satisfy
the nonnegative predicates but establish no gain; report actual positive gains
separately. Missing/nonfinite/unsupported results or absent pairs cannot pass.
Preserve each failed condition and every scoped improvement, without pooling
tasks or selecting the most favorable seed.

## Evidence and claim boundary

The [Operation Contract and Requirements-and-Evidence map](CONTRACT.md), this
prospective protocol, phase admissions, frozen-prefix provenance, engineering/
training/resume/export receipts, all-six seal, saved heads/predictions and six
independent arithmetic audits form the evidence record. Later results/compact
records state actual counts and the scientific decision; declaration alone is
not a result. Keep complete portable construction, TRAIN fitting/training,
mask/target/gradient rules, typed continuation/inference, configuration,
dependencies/licenses, method rationale and outcomes with the V18 component.
Consumers retain ownership of substantial application training and docking.

January is known development evidence; January/February/March2024 informed
previous choices and are consumed for later selection. Reusing already exposed
controls and data is explicit. This1024 comparison is longer than V17's early
screen but is **not the historical2048 budget, fresh confirmation, an ultimate
encoder claim or automatic promotion**. The selected V16 reference remains
unchanged unless a separate justified decision changes it. A positive result
supports considering further work; it does not silently authorize another
budget, population, application selection, external message, commit/push or
publication. A valid negative result completes this bounded comparison.
