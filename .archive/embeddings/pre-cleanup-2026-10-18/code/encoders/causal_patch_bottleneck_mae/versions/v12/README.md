# CPB-v12: attention means and mask-only uniform dispersion

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
All inherited notices remain in the complete physical source copies.

SOURCE PREPARATION ONLY. The admitted design allows source copies and static
edits, not AST, imports, compilation, CUDA, proof, training or quality execution.
CPB-v3.market-01 remains the clean baseline and sole DEV encoder. Keep every
V3–V11 source, checkpoint, result, scoped leader, failure and review.

This version physically owns the complete FINAL V11 closure: 50 files beneath
this directory and a full identical public-header body at versions/v12.h.
The V11 canonical descriptor and three active V11 research tests were also
preserved before copies or edits. All55 parent files match the frozen V11 source
manifest. Their complete exact bytes and hashes live under
output/engineering/causal-fresh-v12-source-preparation/attempt-001/family/parent-copies/.
New research tests own full bodies under code/research/causal_fresh_v12/tests/.
There are no old learned-source includes, parent runtime calls or forwarders.
Torch and the standard library are the only external implementation dependencies.

Only the intermediate temporal dispersion operator changes. For a nonempty
original-position bin L_j and all currently valid retained patch positions V:

- The first four existing export rows keep the exact V11 attention expression:
  mu_a=.5*sum(local_softmax*safe_hidden)+.5*sum(global_softmax*safe_hidden).
- The last four existing rows use u=.5*1[L_j]/|L_j|+.5*1[V]/|V|.
  Their own center is mu_u=.5*sum(local_uniform*safe_hidden)+
  .5*sum(global_uniform*safe_hidden), with the actual two-sum arithmetic.
- Dispersion is sum(u*(safe_hidden-mu_u)^2), componentwise. It never uses the
  attention mean as its center. There is no sqrt, Bessel correction, epsilon,
  detached hidden/center, new parameter or mixture-coefficient choice.
- Count denominators clamp only empty counts to1. Unsupported hidden operands
  are sanitized before subtraction, square or sums. Supported nonfinite values
  and scores are rejected. Absent LOCAL bins have the WHOLE eight-row slice
  exactly zero AFTER projection bias, even when global history is present.

Uniform mass is per valid patch, not raw tick or feature cell. All slices are
coordinates of the current cutoff endpoint; the global part cannot use a time
after that cutoff. Cropping values, masks and target sources, original-position
causal Transformer attention and backward-only banks remain unchanged.

Hidden and its own uniform center stay live. At fixed hidden/support, dispersion
has zero direct derivative to temporal pool scores/position offsets. The attention
mean still trains those scores. Transformer attention inside hidden states remains
live through both statistics. With dispersion upstream b, delta=h-mu_u and
r=sum(u*delta), the hidden derivative retains both 2*u*b*delta and the own-center
chain term -2*d(mu_u)/dh*b*r. Finite F32 two-sum arithmetic need not make r exactly
zero. The attention mean has its separate complete local/global softmax adjoints.
Shared global clipping/AdamW can still couple later trajectories; no optimizer
independence or validation-cause claim follows.

No native coordinate is added: content20 + literal spectral12 + native mean4 +
native uncertainty4 remains native40. Intermediate dispersion is distinct from
native uncertainty4 and the conditional variance decoder. The unit Huber1 mean
plus detached-error Gaussian variance objective, one global clip1/AdamW, declared
data/targets/seeds/streams/2048/B8 and fixed D40 heads remain. All learned numeric
values, optimizer and TRAIN statistics start fresh, source_updates=0; source
copying imports no trained weights or fitted statistic. Intended registration/
constructor order and shapes remain76 learned tensors/294752 values and3888
fixed prior values; runtime parity/learning is pending and not inferred from
these intended counts. Initial DATA20:40/five statistic pairings versus saved V11
are prospective; content0:20 and whole-native40 equality are not promised.

Strict typed identities are CPB-v12-specific at inference root/payload and TRAIN
root/encoder/encoder.payload. New policies:
- temporal_pooling_policy:
  original-position-four-bin-attention-mean4-uniform-centered-variance4-v1.
- temporal_moment_policy:
  componentwise-centered-variance-half-local-global-uniform-own-two-sum-mean-v1.
Existing export layout, Long[2]=[4,4] moment partition, Long[5]=[4,16,32,4,8]
pool partition, Double[2]=[.5,.5] mix and post-bias empty policy remain strict.
Wrong/missing/value/dtype/shape and old V11 policies/kinds have refusal sources
at all five required nodes. Typed resume additionally binds optimizer moments/
presence/order/steps, counters, dataset/recipe and deterministic streams.

TemporalPoolOutput adds dispersion_weights[R,4,T] and dispersion_mean[R,4,64].
TemporalExecutionTrace adds pool_dispersion_weights/pool_dispersion_mean and
contextual_pool_dispersion_weights/contextual_pool_dispersion_mean. Existing
attention local/global/mixture weights, attention bin means, uniform variances
and projected exports remain separate. Both actual ordinary/contextual paths
populate those fields from the SAME owned kernel and existing encode call.
Actual step1/2048 writer fields are coordinated with the independently owned
training source; no extra encoder forward is introduced. Actual2048 is pre-step
2047, never a final2048 decoder substitution.

The dedicated temporal fixture sources declare:
- Independent scalar F64 forward/chain rule and F64 finite differences; a separate
  captured-F32-support/OWN-center adjoint calculation retains the nonzero
  sum(u*(h-mu_u)) correction. No producer helper supplies reference arithmetic.
- Both real model paths and new dual-support trace parity; masked/sparse original
  positions, five-state/nonbinary-count support, constant hidden tolerance,
  equal-mean/different-spread discrimination, absent-local/global-present,
  all-padding zeros, unsupported NaNs and supported nonfinite refusals.
- Three score perturbations leave uniform support, OWN centers and variance
  bitwise unchanged at fixed hidden, while learned attention means change.
  One isolated kernel graph checks variance hidden gradients live/direct score
  gradients zero and attention-mean score gradients live via two grad calls.
- Manual F32 abs/rel tolerance3e-5, common-shift2e-6, masses2e-6/8e-6, constant
  F32 variance bound1e-10*(1+max_abs_hidden_squared), independent F64 FD step1e-5
  abs/rel2e-8, CUDA F32 FD step2e-3 abs/rel2e-3. Each FD reference covers18
  coordinates; at least three actual reference derivatives must exceed1e-2.
- Derived47 DIRECT kernel calls: nine non-FD successes (initial1, common-shift1,
  Hadamard3, constant1, spread2, isolated1),36 F32 finite-difference evaluations
  and two supported-NaN refusal calls. One separate backbone encode uses both
  actual paths; those two internal kernel invocations are not direct calls.
  Public Encoder/head calls0; one synthetic optimizer step. These are source
  declarations, not observed test or learning results.
- Existing full-cutoff causal, actual CUDA learning, eight moment-group events,
  within-/between-bin contrasts, common score-bias gauge and exact typed-resume
  fixtures remain owned and prospective. Tensor capacity is not proof that every
  scalar learned; the common score-bias gauge remains observed separately.

Expected successful temporal log flags are:
attention_mean_and_uniform_dispersion_full_expression_adjoints,
uniform_dispersion_own_center, uniform_dispersion_score_independent,
uniform_dispersion_hidden_gradient_live, attention_mean_score_gradient_live and
dual_support_actual_trace. They are emitted only after the fixture checks pass.
No old full_attention_dependent_moment_adjoints claim is made.

The sole descriptor for now is the unmeasured DRAFT at
doc/causal_fresh_v12/CPB-v12.descriptor-draft.json with card=null.
The canonical versions/CPB-v12.json and an instance/catalog selection are absent
until separately reviewed prospective freeze. Preserve the applied strongest
V3–V11 future22 criteria and every historical original pass count. Uniform
dispersion is a hypothesis: it may lose the five V11 gains. No winner, result,
trained parent-prefix equality or application promotion is claimed.
