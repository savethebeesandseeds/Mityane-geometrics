# V19 learned-state EMA teacher: prospective fixed-budget comparison

Declared **22 October 2026, Asia/Dubai**, before V19 engineering/model
execution, candidate training or new target-head fitting. The owner authorized
continued search for a better encoder while retaining the established test
mechanism. This declaration fixes one learning-method intervention and a
**1,024 successful-update endpoint**. The three completed scattered controls
and their exact audited heads are reused evidence. They are already observed
development results, not a new blind control cohort.

## Question and hypothesis

Does a masked student learning to predict a slowly updated, fully observed
teacher's **learned shape coordinates** improve the existing past-window linear
readouts over the original scattered-mask Gaussian/temporal recipe at the same
1,024-update endpoint?

V17 weakened temporal supervision and failed its 256-update continuation rule.
V18 changed query patches to contiguous spans and failed every condition at
1,024 updates. V19 restores the unchanged scattered training mask and adds a
direct learned-representation target. Its hypothesis is that the added target
may encourage a learned bottleneck useful beyond waveform reconstruction.
This is a hypothesis, not an established explanation of earlier failures.

Plain native-view consistency already existed in historical RPB-v9. V19's
specific method uses an EMA teacher, a training-only predictor and only learned
shape coordinates. It does not claim literature novelty. No external model,
weights or implementation is imported by this declaration.

## One learning-method change

Keep the configurable CPB student architecture, inference/native layout,
constructor, fitted preprocessing, scattered query mask, original losses,
sampling and optimizer settings. Add a teacher/predictor auxiliary loss:

`L = GaussianNLL + 0.1 * (HuberLag1 + HuberLag4 + HuberLag16) / 3 + 0.1 * A`.

The original temporal term retains hidden-and-observed Q at both endpoints,
fixed sqrt2 scaling, Huber delta1, the existing pair/channel/example reduction,
equal lag weights and graph-connected zeros for empty lags. Targets remain
detached and variance-head input gradients remain stopped. Gaussian supervision
occurs once. No readout label enters encoder training.

Let `D = dimensions(config).shape_width`; at the study configuration D=20.
Only the first D learned shape coordinates of the native representation enter
the auxiliary target. Literal spectral and mean-change coordinates do not enter
the predictor, target or agreement loss. Inference remains the student's
unmodified native36: learned20 + literal spectral12 + mean-change4.

For each attempted batch:

1. Use the unchanged original scattered-patch training mask and the existing
   masked student forward trace. Retain all original missing/support and
   cutoff-first rules. Take its learned-D vector `s`; do not detach this path.
2. Encode the same raw history with the teacher's full **actually observed**
   view at the **same explicit cutoff**, under NoGrad. Full view means no
   artificial query/context deletion; naturally missing cells remain missing.
   Teacher input is retained at the cutoff before encoding, exactly as for the
   student. Take and detach the learned-D target `t`.
3. Apply the training-only predictor
   `p = Linear(D,2D) -> GELU -> Linear(2D,D)` to s. Both linear layers have
   biases; there is no batch normalization, dropout, other projection or
   predictor at inference. Dimensions derive from D, not a hard-coded20.
   Its parameter count is `4*D*D+3*D` (1,660 at D=20).
4. Promote the predictor output and detached teacher target from F32 to F64
   before normalization. For each vector v use
   `normalize(v) = v / sqrt(clamp_min(sum(v*v), (1e-6)^2))`.
   Compute the sum, clamp and square root in that order in F64. This has the
   same normalization values as dividing by `max(L2norm(v),1e-6)` while
   retaining finite derivatives at an all-zero vector; do not take an
   unguarded square root first and rely on a later denominator clamp.
   Do not center, whiten, add a variance/covariance loss, replace the target
   with fixed-bank coordinates or normalize the entire native vector.
5. Auxiliary row eligibility is the Boolean intersection of the original
   Gaussian NLL's eligible-example mask, masked student encoding validity and
   full-view teacher encoding validity. Compute each eligible row's
   `a_i = 2 - 2 * sum(normalize(p_i) * normalize(t_i))` and A as their
   unweighted arithmetic mean. Do not clip the dot product/loss or substitute
   unsupported rows. A near-zero vector remains covered by the fixed norm
   floor; it is not an extra eligibility exclusion.
6. If no auxiliary row is eligible, A is a finite graph-connected zero. A
   supported Gaussian batch still performs its usual update and teacher EMA.
   If the base NLL has no eligible support, retain the existing whole-attempt
   skip: consume one attempt, no optimizer step, no EMA update. Do not train
   an auxiliary-only batch or manufacture base support.
7. Backpropagate the combined loss once. One global norm2 clip at1 includes
   the student's learned parameters and predictor parameters. One AdamW step
   updates that joint set. Teacher parameters never enter this optimizer or
   clipping set. Use lr0.001, weight decay0.0001, betas0.9/0.999, epsilon1e-8
   and AMSGrad false. Bind and verify exact parameter names/order and moments.
8. After each successful AdamW step, under NoGrad, update every corresponding
   learned teacher parameter in place in this exact order:
   `teacher.mul_(0.99).add_(student, 0.01)`.
   The student values are the post-optimizer values. Fixed parameters,
   buffers, fitted input scaler and bank remain exact and are not EMA averaged.
   Teacher update count equals completed optimizer updates; skipped attempts
   do not advance it. Verify fixed state and count invariants after the step.

The teacher has the same base family/configuration, including its full learned
parameter set. Preserve its original parameter flags; NoGrad forward/update
and exclusion from optimizer provide the teacher's gradient boundary. Do not
silently change requires-grad flags to bypass the family's state checks.
No teacher gradient, predictor output or projection is needed for inference.

## Initialization, fitting and randomness

Study geometry remains **C3/H128/F9/P8/native36**, ordered BTC/ETH/SOL semantic
IDs0/1/2, interval60000. Other settings remain level width32, encoder W64,
three layers, four heads, feedforward256, decoder hidden128, mask ratio0.25,
dropout0, scale floor1e-6 and LayerNorm epsilon1e-5. These are benchmark
bindings, not the family's only supported geometry or capacity.

For each seed947101/947102/947103:

- Fit the unchanged rounded-parent Chan TRAIN input statistics and relative
  bank exactly once on all1,024 existing TRAIN windows, increasing source order
  in B8 chunks. Retain the exact original source/split/fitting identity.
- Construct the base student freshly with that encoder seed. Its initial base
  parameter/scaler/bank numerical identities must match its existing same-seed
  scattered control. No old learned weights, optimizer state or fitted state
  initializes the candidate. The candidate performs its own matching TRAIN fit.
- Independently construct the teacher with the exact same configuration,
  scaler/bank and encoder seed. Check exact tensor-bit equality so its initial
  base values are identical to the student, without a redundant copy. Record
  their actual identities.
- Initialize the predictor explicitly from seed `encoder_seed+4000` after base
  construction. Additional modules must not shift the base student's draws.
  This offset is the fixed study initialization law, not an architectural
  restriction: the portable fresh API accepts any explicit nonnegative
  predictor seed and stores/binds its actual value for continuation.
- Keep row/mask/context streams at encoder seed+1000/+2000/+3000, context
  deletion none, B8, the original counter-seeded MT19937_64 modulo-row sampler
  with replacement and unchanged scattered-mask generator/draws. There is no
  new data augmentation, span mask, masking curriculum or extra random view.

FNV tensor-byte state identities are engineering witnesses, not cryptographic
tensor equality proofs. The engineering fixture compares actual tensors and
verifies base parity. Full teacher/predictor/optimizer identities are separate
from the student's inference-state identity. Dropout0 and explicit counter
streams define the supported stochastic continuation scope; no implicit RNG
coverage outside that scope is claimed.

## Fixed collapse diagnostics

The teacher may collapse or learn an unhelpful target. This study adds no
variance/covariance regularizer, adaptive correction or collapse rescue.
Persist per-attempt auxiliary support count, auxiliary loss, base losses,
combined loss, update/skip state and teacher update count.

For raw student learned-D, raw teacher learned-D and predictor-output vectors,
compute detached diagnostics on the auxiliary-eligible rows using **F64 on the
model device**. For each coordinate, compute population standard deviation
with denominator N; log its mean, minimum and maximum over D, actual N and
the count of vectors whose pre-normalization L2 norm is <=1e-6. N=1 has
zero population dispersion. N=0 reports supported=false and finite zero
statistics/counts. Do not recompute the diagnostics on CPU with different
arithmetic or use full-native/fixed-bank dispersion as learned-state evidence.

These are descriptive TRAIN diagnostics, not new readout metrics or success
predicates. They do not shorten/extend the budget, change coefficients, choose
seeds, tune EMA, add a regularizer or rescue a collapsed run. Nonfinite loss or
invalid saved state is a correctness failure and preserves partial evidence;
zero/low finite dispersion is an observed learning result.

## Reused observations, targets and fitted-state bindings

Reuse the already admitted V17 generic archives and original legacy targets.
No new preparation, download, period, role selection or target generation is
performed. Prefix `.temp/embeddings/runs/cpb-v17-balance-001/attempt-001/`:

| Archive | Role/count | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `data-train/calibration-TRAIN.pt` | TRAIN1024 | 32920080 | `7020ebf1d6c2b793893a9c0ccb8ae524e76c478c6389e31204c3e92dd88efcaa` |
| `data-probes/probe-TRAIN.pt` | TRAIN112 | 3604752 | `a9b6531ee0457c0411386300565279a558651df47a0b231b3f118088b8bc8778` |
| `data-probes/probe-VALIDATION.pt` | VALIDATION32 | 1033232 | `4dc66210cc76793b6223bb8af9faad259679d45a1e2faf316a23e57642fa664c` |

The preparation/admission receipts bind genuine CPU F64 values, Boolean masks,
minute timestamps/cutoffs, exact channel/source/row identities, CACHE88 sources
and actual role membership. TRAIN close times are1704067259999–1705881599999
(January1–21); known validation is1705881659999–1706486399999 (January22–28).
Original nonoverlapping H256 probe support retains recent128 encoder/target
observations. Role text by itself does not prove actual membership.

Original ordered CACHE88 SHA256 values remain:

- BTC `67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58`.
- ETH `0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa`.
- SOL `364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7`.

The original target archives are under the portable MKT-HIST-2 features root
recorded by the V17/V18 receipts:

- `train-targets.pt`:8782 bytes, SHA256
  `521d8662f498bce27ad26a41c4723bbebb9d2ab6b924cfbcce7132c25b7e43e6`.
- `validation-targets.pt`:4677 bytes, SHA256
  `cbd08bd01c728cfbb7a500922852573c45f87dd98821b42e20ee202a7a743181`.

Retain original settings
`code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v17/config/research.conf`,
622 bytes, SHA256
`f48767f0a49261033fc0e291e51327bbcde07c5d9e0769e4cb03a973085c5a4b`.
Use matching numerical geometry for the candidate while binding its own teacher
recipe and executable. All old controls share scaler identity
`rpb-scaler-v1-fnv1a64-d6fb768d6fa7ac09` and bank identity
`rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-687aaa9aed358ed5`.
The full initial base-state prefix is `cpb-family-state-fnv1a64-`; suffixes
for947101/2/3 are17426e39fea5d2fd,09bd37d81f312f52,dd3614ec72f6bfa5.

## Reuse all three completed scattered controls and heads

The authority for inherited identities is the completed
[`CPB-v18.span-001.json`](../../records/CPB-v18.span-001.json),103351 bytes,
SHA256 `b522f1bf16f57c19cab28ee7fdb8fd3ab848523005320063b1c839a0fbdf2719`.
Its study admission is151803 bytes/SHA256
`e5c66b924d44d8599a4a2e110f7403aa5dbceb24edf9f8598e2ee5b1eda6b5db`;
its six-state training seal is17313 bytes/SHA256
`27e8375013479770d632faf2ac00ed335fd69c727dde381801508f84202cd18a`.

Below the V18 root
`.temp/embeddings/runs/cpb-v18-span-001/attempt-001/`, use each seed's
`trajectories/seed-S-control/resume-1024/inference-final.pt` and
`training-final.pt`. They retain the original V17 coefficient0.1/scattered
artifact kind and recipe. Do not convert them to V19 or continue their training.

| Seed | Inference bytes / SHA256 | TRAIN-state bytes / SHA256 |
| --- | --- | --- |
| 947101 | 1300204 / `e762a08e4f1419bbcda802fc7db359a2601ca296c6d0b79846c320463adb941c` | 3767751 / `7a1fce5c185167197c07623734573c170b1f7f8d0e483c49011b88f374e9430b` |
| 947102 | 1300204 / `c95b901c4e86beb34907d5d96ec771d0b983c7c49d1c8465cf6330272ecfb95d` | 3767751 / `b062921a72a78b44455c84ec2bb83edd6f746c200aa9f99c2947a7925aa90fc7` |
| 947103 | 1300204 / `4e637264da7b3a4c364c75b0ee618f95241a8de72ac4ad8defc64658d34eeb5c` | 3767879 / `28eb022d6bddef19458a3ac9566bd873d391e5ebc795a6c4d1c351f72e8c41e3` |

Reuse exact `readouts/seed-S-control/summary.json`, its ridge witness and all
bound inputs, plus `audits/seed-S-control/summary.json` and their successful
native receipts. Bind all hashes from the original compact record:

| Seed | Readout bytes / SHA256 | Audit bytes / SHA256 |
| --- | --- | --- |
| 947101 | 3323 / `120b069a561bafb745f9f52612b955d3e1e711552dc07120d8d5812c6669bc65` | 3530 / `d614ea2fa9a5b8088e859736302a4c0a4e23f409bc423bdaf33289a38a7c8f9f` |
| 947102 | 3321 / `90d77ed734fed71f0350956d13fe7a1a0f1ea84da8fda6d86692594e55c1414a` | 3529 / `044c8e99203375ddfc099d1f703f742951f0e7d17f049b07efa9e7a3dcd22028` |
| 947103 | 3321 / `7a20ae4cf08f5097a1c7b27950b1f2604dceb118527019104eccc73d0705a48d` | 3530 / `51f237a5eaa6eb978a3d47f1b4c0a96a70698b3d41d2ccbc60ec074aa401597d` |

No selected seed or favorable control subset replaces these three. Their
original execution used the848288-byte V17 trainer, SHA256
`767c94e29f0687f881522a2045cd23ebcd1290db7e74a7d97fcc59e55d8c71a4`.
Their readouts used the793664-byte V18 adapter, SHA256
`536d2ad23f89666e84193adeb3834ca1faf84f2f0bdb9cbb3450997c8fad2cea`.
The original104176-byte independent scalar audit remains SHA256
`defbb1b714c1900192b84483206a17f4f35301dc8887c34bc348d67d3b5fd662`.
These historical executables/receipts are retained, not rebuilt under an old
identity. Original control detailed mask fields remain unavailable where its
trainer did not log them; do not invent them during reuse.

## Fixed schedule and honest work accounting

Train new candidates sequentially in seed order947101,947102,947103. Each starts
fresh, trains0→512, saves typed state, then actually resumes512→1024 without
refitting. The endpoint is1,024 successful updates with at most2,048 total
attempts from original attempt0. Save every128 successful updates and retain
initial0, intermediate and final student/teacher/predictor/TRAIN states.
Intermediate saves are engineering/recovery evidence, not alternate outcomes.

There is no intermediate head fit,256 gate, rescue seed, EMA/weight sweep,
teacher restart, changed collapse penalty, adaptive stopping or outcome-dependent
extension. Failure preserves source, command, logs and partial state; it is not
silently rerun or reported as a completed scientific negative.

| Scientific work | New V19 execution | Reused control evidence |
| --- | ---: | ---: |
| Optimizer updates | 3072 | 3072 |
| TRAIN preprocessing fits | 3 | 3 |
| Training processes | 6 | Previously completed histories |
| Target-head fits | 12 | 12 |
| Independent scalar audits | 3 | 3 |

The six logical1,024 histories contain6,144 updates; only3,072 are new physical
scientific updates. Inherited controls contain768 V17 prefix updates and2,304
V18 continuation updates. Reused head/audit work is not newly executed. Record
actual attempts, skips, failed/repeated physical work and artificial engineering
updates/fits separately. No old data preparation or gradient diagnostic runs.

## Prospective source preservation and engineering admission

Before adding V19 root build/attribute entries, preserve the V18 manifest's107
current-source members plus its separately pinned closeout generator in a new
V18 capsule. Bind original→preserved paths and actual copied bytes, retaining
the107-source digest and generator pin separately. The original closeout-input
manifest is142170 bytes/SHA256
`ed825deee26147489d27beb1eb2afaaf8e82a343766f05fb1a84feb4a2f52425`.
Preserve the earlier108-file V17 capsule, all original artifacts and prior
source bundles. Do not weaken or invoke closed V18 live-path admission guards
after the root Makefile/attributes change; V19 verifies inherited evidence
through the explicit preserved-source mapping and unchanged numerical subset.

Before market execution, admit exact new source, configuration, protocol,
contract, executable, dataset, inherited control/head and engineering pins.
Preserve the49 frozen V16 files and three maintained family numerical headers.
Use the existing managed Linux container/pinned SDK and root task/build entries,
named build/run directories, C++ numerical code and Node/Bash orchestration.
Verify actual CUDA and shared GPU availability. No Python, host toolchain,
dependency/image substitution, application deployment or old experiment replay.

Bounded artificial fixtures must establish base/scattered loss/gradient/update
parity with the auxiliary term disabled, matching fresh base/teacher tensors,
the actual learned-D slice and predictor dimensions, target detach and student/
predictor gradients, same-cutoff future poisoning, eligibility/norm-floor/empty
support behavior, exact EMA timing/counts/fixed-state preservation and teacher
exclusion from optimizer. Exercise supported native36 and a meaningful
alternative shape. Verify real updates, inference equivalence without teacher/
predictor, typed TRAIN save/load, actual multi-step resume and refusals for
wrong source/config/recipe/kind/optimizer/order/teacher state or progress.
Engineering work remains separate from research quality.

TRAIN artifacts bind student, teacher, predictor, all fitted state, optimizer
moments/order/options, completed/attempted/EMA counts, source/config/data/split/
fitting identities, encoder/predictor seeds and stream recipes. Inference
artifacts carry only the student numerical state plus actual recipe/provenance;
neither teacher nor predictor is required to encode. New typed kinds distinguish
V19 from family/V17/V18 states. Equal native width does not establish state
compatibility. Save/load does not refit or silently reinitialize auxiliary state.

## Unchanged evaluation and new-candidate seal

Before any **new candidate** extraction or head fitting, seal all three final
V19 student states and the identities of all three admitted existing controls,
their exact heads/witnesses/audits and inputs. The control heads were fitted
before this new seal; do not claim otherwise. All new candidates finish before
any new candidate head, regardless of individual training order.

Reuse exact112 TRAIN/32 known January validation probe anchors and legacy
F32-loader→F64 targets, original row/group/cutoff/support associations, units
and target order. Keep genuine F64 encoder input. Each candidate gets the same
TRAIN population mean/scale fitting with1e-8 floor, centered normal equations,
ridge penalty1 and unpenalized intercept. Preserve mathematical target/head/
prediction/scoring regions; only typed loading and evidence bindings adapt.
The original generic ridge-witness codec may be reused with separately bound
actual encoder kind, recipe, checkpoint SHA and update count.

Persist candidate features, head means/scales/weights/intercepts, predictions,
targets, validity and row/group/cutoff IDs. Run the unchanged independent scalar
long-double normalization/Cholesky/scoring audit on each new candidate witness,
with absolute1e-9 plus relative1e-8 tolerance and exact original target/association
checks. Verify inherited controls' witnesses/audits and all bound byte identities
without refitting or replaying them. Report every arm's R², MAE, RMSE, coverage,
paired candidate-minus-control deltas, medians, minima, maxima and positive/
nonnegative pair counts.

### Unchanged continuation rule

Require valid complete six-instance evidence, finite complete32/32 support,
all three new and three reused passing arithmetic audits, and all five predicates:

- Trend median paired linear R² change >=0.
- At least2/3 trend pairs nonnegative.
- Activity median paired linear R² change >=0.
- At least2/3 activity pairs nonnegative.
- Volatility median paired linear R² change >=-0.01.

Relation is descriptive. Ties can satisfy nonnegative conditions but are not
gains; record positive counts separately. Missing/nonfinite/unsupported pairs
cannot pass. Collapse diagnostics do not add or replace a condition. Report
each failed protection and scoped improvement without selecting favorable
tasks/seeds or tuning these thresholds after execution.

## Claim boundary and portable deliverable

January/February/March2024 are consumed evidence for later selection. This is
known January development at1,024 updates using already observed controls,
not fresh confirmation or historical2,048-budget reproduction. No neural head,
TEMPO timing, fixed-query reconstruction, new population, forecasting or
application utility is measured. No new broader result is inferred from TRAIN
agreement/collapse diagnostics. **V16 seed946103/final2048 remains the research
reference.** A positive screen is a bounded continuation signal, not automatic
promotion, checkpoint selection, another budget/population or deployment.

The [Operation Contract and Requirements-and-Evidence map](CONTRACT.md), this
protocol, phase admissions, source capsules, engineering/training/resume/seal
receipts, inherited controls/heads and new witnesses/audits form the evidence.
The completed result later records actual work and every gain/regression.
Preserve valid negatives and failures. Carry configurable construction, TRAIN
fitting, masking/objective/teacher/predictor/EMA/update rules, complete typed
resume/inference code, dependencies/licenses, configuration, tests, rationale
and measured outcomes in the portable component. Applications retain their
own substantial training, data binding and final docking. This declaration
does not authorize external messages, publication or commit/push.
