# Native view agreement: the next bounded encoder experiment

Date: 2026-10-07. Status: prospective proposal, not implemented or measured.
Proposed tag: **RPB-v9 — Global bottleneck with native view agreement**.
RPB-v4 remains active. This plan follows the completed TRAIN objective diagnosis;
it does not change that diagnosis, its gate or its saved evidence.

## Why this mechanism

The weight1 all-channel residual-difference auxiliary failed its fixed rule:
it improved the local fixed-Ridge gradient direction on two masters and worsened
it on three. Do not train it, search its weights or select channel pairs.
The reconstruction direction opposed the saved timing margin on four of five
masters, but a fixed head can also react to a changing feature basis. This does
not establish irrecoverable timing loss or a future classification improvement.

RPB already mixes aligned W64 channel states before channel compression. Adding
another early mixer has weaker justification than separating two jobs that the
deleted-context decoder currently combines: reconstructing masked values and
stabilizing the representation when context disappears.

Keep ordinary reconstruction as the anchor. Add agreement directly on the native
32-number embedding of the same row under extra missingness. MTF's independent
view alignment and variance safeguards motivate the mechanism; its EMA teacher,
projector and predictor are not required for this first test. The ordinary
online embedding is a detached target for this one loss, not another network.

## Exact proposed computation

Use the existing mode2/mixer1 architecture and all 225,805 parameters: C3/H32/F3,
patch8, width64, temporal layers3, heads4, feed-forward256, decoder128 and native32.
No architecture, decoder capacity, serving transform or classifier change.

For each original legal TRAIN minibatch, keep O, A, Q=O&A, original eligibility,
scaler, sampled rows and absolute mask/Torch streams unchanged. Let
V_o=O&~A. Use the existing repaired coordinate15_v1 primitive at the same
absolute attempted counter to obtain E and V_s=V_o&~E. Zero all nonvisible
storage before either encode. Do not reveal Q to either branch.

Compute z_o=encode(V_o) and z_s=encode(V_s), both through the exact served global
32 surface. Decode only z_o for the unchanged hierarchical Huber delta1 on Q.
The student z_s is used for native agreement and variance; it does not replace
the ordinary decoder input.

Before any update, extract candidate point0 native features with the CUDA model
in evaluation mode on legal TRAIN observations O. Compute their per-coordinate
population standard deviation in float64, floor at1e-6, then store float32 s0.
Use only valid TRAIN rows, record their source/order/support, and
save the vector and its identity. Fit it once without labels. Do not recompute
it as training advances. This vector scales the loss; it never scales serving
exports or replaces the heads' existing TRAIN-fitted normalizers.

Let u_s=z_s/s0 and u_o=detach(z_o)/s0. The one fixed recipe is:

```text
L = original_hierarchical_Huber(decode(z_o), Q)
  + 0.05 * mean((u_s - detach(u_o))^2)
  + 0.01 * mean_d relu(0.5 - sqrt(population_variance(u_s)_d + 1e-4))
```

Agreement averages all32 coordinates and supported same-row examples. Its target
path contributes no gradient; the ordinary branch still receives reconstruction
gradients. Shared encoder parameters receive both branch contributions.
No learned projection or prediction head intervenes between the loss and native32.

Support requires original reconstruction eligibility, matching row identity,
channel IDs and endpoints, joint sample validity and unchanged channel-valid
masks. Preserve ordinary reconstruction
for unsupported auxiliary rows and log their counts. Variance uses the first
supported sampled occurrence of each distinct source ID, without labels; with
fewer than two distinct groups it contributes differentiable zero. Use population
variance, not an unbiased estimate. Opposite-label source siblings are never
agreement positives: the positive is always the exact same row in two views.

These coefficients, scale recipe, floor and variance threshold are newly chosen
prospectively. They are not inferred optimal from the diagnostic. Keep them fixed;
do not search loss weights, rates, budgets or pair subsets after measuring them.

## Implementation and admission boundary

Add the objective in the RPB encoder folder with a distinct training-policy
identity, proposed `rpb-training-native-view-agreement-015-v1`. Keep all v4/v6/v7/v8
recipes, default behavior, initialization order and archived artifacts intact.
Add no state or transformation to the served encoder. Save s0, coefficients,
eligible/auxiliary/group counts, component traces and ordinary/deleted branch
metadata in new companions. Ordinary reload/resume must reject the new tagged
training policy until a complete explicit resume contract exists.

Tests must establish real CUDA parameters, both inputs, all losses and gradients;
same-row support; original Q/eligibility/scaler preservation; hidden-value
poisoning invariance of exports; and zero target-path agreement gradients.
Verify the ordinary reconstruction scalar/gradient against the unchanged core
forward when the new auxiliary coefficients are disabled. Check finite variance
gradients at constant inputs, duplicate-source handling and unsupported branches.
Admission must include actual weight updates and deterministic live continuation
across 0/1/2/4 snapshots, with no ordinary tagged resume and no new RNG streams
that perturb the original sampled rows, masks or initialization.

Freeze a separate human quality card, exact input role list and source identities
before implementation admission or data access. Do not borrow the diagnostic's
195-role permission: the quality experiment needs its own explicit TRAIN and
known-VALIDATION boundary. Shared evaluation consumes native feature archives;
it must not own this objective or instantiate a teacher.

## One prospective comparison and decision

Use five known masters4404/5505/6606/7707/8808, each with256 TRAIN rows/128 source
pairs and128 known VALIDATION rows/64 pairs, at original10% natural missingness.
Train five fresh candidates for512 completed, unskipped updates, batch8,
AdamW0.001, weight decay0.0001, clipping1 and dropout0. Save point0/512 on one
live trajectory. Retain relevant v4/v7 checkpoints; do not retrain older designs.
Require matched common parameter initialization and original row/query streams.

Use intact known VALIDATION and the exact saved v7 additional30% deletion view.
Keep the native32 Ridge penalty1 and tanh16/Adam0.01/100-update heads, with all
three declared repetitions2701/2802/2903 and their existing width-derived seed
law. Fit head weights only on TRAIN, separately per checkpoint. The shared
driver transparently refits retained controls with the same recipe and requires
exact native fit and TRAIN/intact prediction parity against saved control fits.
No PCA follows an encoder. Include raw/PCA baselines
only from an explicitly matched retained comparison, labelled as reused.

Advance only if every condition holds:

- Mean and worst-master linear accuracy in both validation views are at least
  the corresponding retained v7 values.
- Coverage is equal; retain all five masters and every head repetition.
- Mean fixed-query standardized TRAIN and VALIDATION MAE are no worse than
  the paired retained v4 values.
- Actual CUDA admission, source/input preservation and independent audit pass.

Report the fixed neural head as a secondary result and disclose every per-master
tradeoff. Equal512 updates are not equal compute: two encoder passes increase
training work; report synchronized GPU seconds and serving cost separately.
The small-batch variance safeguard may emphasize nuisance variation, and view
agreement may suppress useful differences. Neither guarantees timing accuracy.

A failed joint guard stops this mechanism without coefficient/rate/budget rescue.
Keep historical TEST/stress closed. Known-VALIDATION development evidence does
not supply fresh held-out confirmation or consumer acceptance.
