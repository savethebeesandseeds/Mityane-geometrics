# CPB-v9: fixed half-global, half-local original-position pooling

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
Inherited notices remain in every copied file and LICENSES/NOTICE.

Source preparation only: no AST, build, fixture, CUDA, training or quality
execution. The separate prospective card remains pending source review/freeze.
CPB-v3.market-01 stays the clean baseline and sole dev encoder. Preserve every
V3–V8 scoped leader, checkpoint, card, result, source and failed attempt.

This family owns fifty complete physical V8 folder copies, then version-local
adaptations, plus the byte-identical external versions/v9.h public-header copy.
No old implementation include, namespace forwarding or learned checkpoint/stat
initializer is introduced. All learned tensors and permitted TRAIN statistics
start fresh; source_updates=0 and pretrained_ancestor=none. Source ancestry
does not mean learned-weight ancestry.

The ONE architecture change is fixed pooling. For each NONEMPTY original local
bin j, compute its masked within-bin softmax pooled 64-wide hidden vector and
one shared whole-history-valid-state softmax pooled hidden vector using the SAME
scores. Average those hidden vectors 1/2 each, then apply that bin's existing
eight export rows and bias ONCE. For an EMPTY LOCAL bin, return its entire zero
eight-coordinate slice AFTER bias, even when another bin supplies global states.
All-padding uses finite softmax fallbacks before normalization and exact zero
supported weights/slices. Both actual ordinary/early-contextual routes call the
same owned kernel. No parameter, module, constructor registration or RNG draw
changes. No extra decoder route, objective, clip, rate, seed, head or budget
change is part of this operator.

H128/P8 uses original patch bins0:4,4:8,8:12,12:16, not packed rank or renumbered
visible positions. Export rows are0:8,8:16,16:24,24:32 of the same32×64 weight
and32 bias. Later contextual states already contain causal earlier history.
The downstream99→64→32 global mixing network, shape projection20, relation
contribution, literal spectral12, mean4, uncertainty4 and decoder are unchanged.
The hypothesis is complementary global/local aggregation, not a diagnosis that
V8 lacked global context or that oldest history failed. The fixed half mixture
is prospectively declared, not a coefficient/bin search.

Every export slice belongs to the CURRENT endpoint embedding at requested
cutoff t, not to the earlier endpoint of that slice's anchored bin. The shared
whole-history pool sees only valid retained states at times≤t. Information from
a later local bin is therefore still past information for the endpoint embedding.
It never bypasses crop/mask processing, changes causal Transformer attention or
turns the backward-only relation bank into a future-looking bank. Planned
engineering admission preserves exact future-value AND future-mask invariance
at all128 cutoffs,2350 suffix interventions and30 internal temporal cases,
before and after actual fixture updates, and again for saved main checkpoints.

Public cp::kTemporalPoolingPolicy =
original-position-four-bin-shared-global-half-mixture-v1.
cp::temporal_pooling_partition(config) returns [4,K,D,K/4,D/4], default
[4,16,32,4,8]. cp::temporal_pooling_mix() returns [0.5,0.5] in the order
[anchored_local,whole_history]. cp::kEmptyLocalBinPolicy =
zero-whole-export-slice-after-bias-v1.

Inference root/payload and TRAIN root/encoder/encoder.payload strictly bind:
- temporal_pooling_policy UTF-8 Byte vector;
- temporal_pooling_partition_v1 Long[5];
- temporal_pooling_mix_f64 Double[2], exact [0.5,0.5];
- empty_local_bin_policy UTF-8 Byte vector.

Missing/wrong-value/dtype/shape fields and old V8/family kinds are refused.
All existing objective/layout/provenance, scaler/bank identity, exact model/
optimizer content, counters, parameter-order and stream-resume guards remain.
The inference/training kinds are owned cpb_v9_* kinds; no old-model loader is
available as an initializer.

Both temporal traces expose actual local weights[R,T], one shared global
probability[R,T], per-export mixture weights[R,4,T] and bool bin_valid[R,4].
The exact names are pool_local_weights, pool_global_weights,
pool_mixture_weights and pool_bin_valid, with contextual_ counterparts.
Each present bin's mixture sums1, each absent bin's mixture is entirely0.
The global probability sums1 for an active row or0 for all-padding; local
weights have mass1 per present original bin. Existing pool_weights is explicitly
the SUM of all per-export mixtures, with mass=#nonempty bins. It is neither
one pooling probability nor the weights for every export row. Invalid/padded
positions are zero in all weight diagnostics.

Capacity is unchanged:76 registered learned tensors/294752 fresh values and
one fixed literal tensor/3888 values;77 tensors/298640 total. Native40 =
learned content20 + literal spectral12 + mean4 + uncertainty4. Content0:20
can change at initialization. The partial initial DATA pairing versus V8 is
20:40 plus five whole fresh-statistic archives; whole40/initial distribution
equality is not required. Own initial/final fixed heads must be measured later.

Training remains unit F64 Huber(delta1) mean plus unit Gaussian variance NLL
with detached mean error. Same supported-cell/eligible-channel/example hierarchy,
single AdamW(.001,.0001), shared global norm2 cap1,2048 successful updates/B8,
constructor943001 and sampling/mask/context streams943002/3/4. Reporting NLL is
detached; only objective.total_loss is backpropagated. Shared clipping/AdamW
history and future residual targets still couple the trajectories.

Exactly the one pool_score.bias scalar is exempt from nonzero data-gradient
admission as a common score-shift gauge. It remains registered in the same
optimizer and separately observed. The other75 identifiable tensors, covering
294751 scalar capacity, require nonzero CUDA data-gradient and movement events,
not proof that every scalar received gradient. Four export row weight/bias
groups and four within-bin-centered position contrasts retain their requirements.
The global denominator restores sensitivity to THREE relative bin-constant
position offsets. Additional learning evidence must use the fixed unit R16
Hadamard directions [++--],[+-+-],[+--+], each bin sign repeated over its four
positions and divided by4. Coefficients are signed sums of the four bin means;
their squared sum is4×sum(centered bin means²), the full-position between-bin
projection norm². Require each direction's actual data-gradient/movement EVENTS
and a positive OVERALL final between-bin projection change. Record all three
signed final contrast changes descriptively; an individual coefficient may
return toward its initial value and need not finish nonzero. Empty-bin steps
remain legitimate. Movement includes
AdamW decay and must never stand alone as data-learning proof. The common
pool-score bias remains a gauge; tiny roundoff gradients do not remove it.

The owned dedicated temporal fixture uses nonzero engineering-only projection/
score parameters. It independently computes F64 scalar forward and analytic
adjoints, checks both actual backbone paths, original gaps/padding, all-empty/
single/full bins, unsupported NaNs, and exactly-once nonzero bias application.
It checks common-shift invariance but expects sensitivity in all three relative
Hadamard directions; the old per-bin shift-invariance claim is intentionally gone.
It verifies four export groups, four within-bin contrasts, three between-bin
gradient/movement directions and one actual synthetic AdamW step. Scope:
one backbone Model::encode (both routes), five direct owned-kernel calls
(base/common/three relative), zero public Encoder/head calls, one synthetic step.
Manual F32 abs/rel tolerance2e-5, common-gauge2e-6, per-mixture/local/global mass
2e-6, aggregate mass8e-6; support, padding and empty slices exact. This is not
a whole-model75-tensor or quality proof. The other two owned fixtures preserve
causal cutoff/suffix/internal paths, typed-resume/negative contracts and the
unchanged composite-objective gradient witness. Runtime remains unexecuted.
The constructed single-step synthetic fixture also checks that all three
movement coefficients are nonzero in that fixture. This does not add a
per-coefficient final-state gate to the main2048-update trajectory.

No winner, dev replacement, future floor derivation, publication, cleanup or
historical rewrite follows. The strongest-V3–V8 e792… future22 retention
protocol remains authoritative for a separately frozen prospective card; there
is no TRAIN-NLL gate. This source is a new controlled hypothesis, not a rescue
of V8 by changing its fixed measured recipe.
