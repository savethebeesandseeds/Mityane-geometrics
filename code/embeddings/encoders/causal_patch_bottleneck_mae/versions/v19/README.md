# CPB-v19: training-only EMA teacher on learned coordinates

## Channel axis

cuwacunu_torch `interval_type_e` is the standard. One window is one instrument. Each channel is one time resolution of that instrument: minutes, hours, weeks, and the other kline intervals. The channel id is that interval's torch ordinal. Different instruments are different rows. A stored id set `{0,1,2}` is the withdrawn instrument binding from historical checkpoints. It is not this axis, and scores measured on it stay scores of that binding. The contract is `code/embeddings/encoders/include/embedding/encoders/timeframe_channel.h`.

V19 keeps the configurable family encoder and its inference shape unchanged.
It adds a training-only predictor and exponential-moving-average teacher to
give the learned bottleneck a direct representation objective. Its completed
three-pair, 1,024-update study **failed the continuation criterion**. V16
seed946103/final2048 remains the selected research reference. Preserve all
earlier source, checkpoints, recipes and negative results.

The [23 October results](../../../../evaluation/protocols/cpb_v19_teacher_v1/RESULTS.md)
retain real volatility gains in 3/3 pairs (median ΔR² +0.053184) and relation
gains in 2/3 (+0.109583), alongside trend/activity median regressions
(−0.029321/−0.053271, each improving in 1/3). All four trend/activity conditions
failed; volatility protection passed. No scoped linear record or top-three
group changed. Three new candidates, their 12 heads and three audits passed;
the three control 1,024-update states and audited heads were reused without replay.
This is known January development evidence, not fresh confirmation or a
full 2,048-update or application result. The [compact record](../../../../evaluation/records/CPB-v19.teacher-001.json)
and [independent review](../../../../evaluation/protocols/cpb_v19_teacher_v1/REVIEW.md)
retain exact identities, actual counts and limitations.

## Learning method

The student uses the unchanged scattered-patch mask, Gaussian NLL and coefficient
0.1 times the fixed mean of hidden temporal Huber losses at lags 1, 4 and 16.
All original target/support, variance-input gradient stop, TRAIN fitting,
sampling and one-global-clip semantics remain. The additional loss is

`L_aux = mean_eligible_rows(2 - 2 * dot(normalize(p(z_student)), normalize(z_teacher)))`.

Use only the first `family::dimensions(config).shape_width` learned coordinates;
the literal spectral suffix and relative-change suffix are not auxiliary
targets. At the research geometry this is learned20 from native36. Neither
these widths nor the benchmark input shape define the supported family surface.
The predictor is `Linear(D,2D,bias) -> GELU -> Linear(2D,D,bias)`, with derived
dimensions and no batch normalization or dropout. Its parameters are training
tools and do not enter the served embedding.

The teacher starts with the student's exact base initialization, fitted input
scaler and bank standardizer. It sees the full observed retained history at the
same explicit cutoff, under `NoGradGuard`, and remains in eval mode. It never
receives future observations or readout labels. Parameter flags stay unchanged
because the retained core validates those flags; excluding teacher parameters
from AdamW and guarding its forward/update prevents gradients. Every teacher
gradient must remain undefined.

Predictor/teacher outputs remain F32 and promote to F64 before L2 operations.
The norm floor is 1e-6. Implementation uses
`sqrt(clamp_min(sum(x*x), 1e-12))`, which implements that norm floor while
retaining finite derivatives at an exactly zero eligible vector. No cosine or
loss rescue clamp is added. Eligible rows are the intersection of the base
NLL eligible examples, student sample validity and teacher sample validity.
An empty intersection contributes a graph-connected zero; the supported base
update still proceeds. Unsupported reconstruction batches skip the optimizer,
teacher forward and EMA, incrementing only the attempt counter.

Total loss is `L_base + 0.1 * L_aux`. Student plus predictor parameters share one
AdamW group and one global clip. After each successful AdamW update, update each
learned teacher parameter by two F32 in-place operations:
`teacher.mul_(0.99).add_(student_after_step, 0.01)`.
Fixed literal parameters, buffers and fitted statistics remain exact, never EMA
averaged. Teacher update count equals successful student update count.

Per-batch diagnostics use auxiliary-eligible rows only and detached F64 values:
population coordinate standard deviations, their mean/min/max, actual row count,
L2 norm range and number of pre-normalization row norms <=1e-6 for student,
teacher and predictor. N=1 gives standard deviation zero; N=0 gives unsupported
finite zero diagnostics. These diagnostics are not added quality gates.

RPB-v9 already had same-encoder native-view agreement with a variance safeguard.
The EMA teacher and predictor are a different related training form, not the
first consistency experiment. Current CPB already has attention pooling and a
learned global mixer; this version adds neither another pooling method nor new
served coordinates.

## Portable API and state

Include `embedding/encoders/causal_patch_bottleneck_mae/versions/v19.h`; use
namespace `embedding::encoders::causal_patch_bottleneck_mae::v19`. The numerical
closure uses the new include root plus unchanged family and frozen V16 roots.
Raw observations, supported configuration, TRAIN fitting and fitted statistics
reuse the family's public contracts.

`fresh(config,scaler,bank,encoder_seed,predictor_seed)` returns `TrainingBundle`
with student `model`, `scaler`, `teacher`, `predictor`, predictor seed, original
teacher/predictor identities and teacher update count. Both seeds are explicit
nonnegative construction choices. The matched research recipe chooses predictor
seed `encoder_seed + 4000`; that relation is not an architectural requirement.

`make_optimizer(bundle,options)` returns student-plus-predictor AdamW. Call
`train_step(bundle,optimizer,progress,raw,cutoffs,Recipe{},options)`. It reports
the base and auxiliary losses/support, teacher forward/update counts and the
three diagnostic populations. `forward_at` exposes the complete differentiable
base/auxiliary trace. `predictor_identity`, `optimizer_identity` and
`validate_optimizer` bind named order and actual moment content, not just shapes.

`save_training_state` stores student, teacher, predictor, both initialization
identities, explicit predictor seed, source/split/recipe, fitted preprocessing,
all stochastic counters, exact update counts and combined AdamW moments/order.
`load_training_state` returns `.bundle`, `.optimizer`, `.identity`, `.progress`,
`.optimizer_config` and `.recipe`. Predictor initialization is reconstructed from
its seed to verify its initial identity before trained predictor state loads.
The supported dropout0/counter-based sampler needs no implicit Torch RNG state
for continuation. No schedule, distributed execution or application recovery
service is supplied by this component.

Inference `save_checkpoint(path,model,scaler,updates,Recipe{})` exports only the
student. `load_checkpoint(path,device,&config,Recipe{})` returns `.numerical`
as the ordinary family `Loaded` plus the fixed teacher recipe. The new outer
kind binds actual V19 learning; nested family objects are structural codecs.
Inference cannot resume training, and family/V17/V18 kinds cannot masquerade
as V19 state. Existing weights remain bound to their exact saved configuration.

## Evidence scope

The artificial [engineering fixture](tests/engineering_test.cpp) covers manual
loss and diagnostic arithmetic, actual gradient destinations, untouched base
initialization/masks/forward, exact teacher EMA and timing, cutoff poisoning,
student-only inference, complete next-step/three-step resume and refusal cases
at native36 and native15. Only passed source-bound execution receipts establish
which checks ran. Source presence and a portable API establish no encoder
quality improvement or application utility.

The source-bound managed-container CUDA execution passed **1,734 assertions**,
116 named auxiliary-gradient tensor checks, 12 synthetic updates, two TRAIN
fits, 22 teacher forwards and 42 refusal cases. Native36 and native15 fixtures
passed. The admission, command, execution and summary receipts are retained
under `.temp/embeddings/runs/cpb-v19-teacher-001/`; engineering work is separate
from the market comparison and supplies no quality score.

Keep the complete teacher/predictor/masking/objective/optimizer/resume code,
configuration, tests, dependencies, licenses, method rationale and measured
outcomes together in any source delivery. Applications own data docking,
substantial training, artifact trust and end-to-end application evaluation.
