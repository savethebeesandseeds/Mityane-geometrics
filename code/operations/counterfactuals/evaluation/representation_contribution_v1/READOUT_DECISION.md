# Readout binding decision

**Decision:** `OR06-READOUT-DECISION-001`, version `1.0.0`, 20 October 2026.
**Outcome:** **revise the binding policy; keep TRAIN-only OLS as a scoped
reference method.** The accepted binding has the new identity
`OR06-OLS-REFERENCE 1.0.0`; its machine-readable
[binding record](readout-binding.json) pins actual preprocessing/model artifacts.
This decision completes the assigned keep/revise/replace question.

## Evidence and rationale

The independently audited [follow-up](RESULTS.md) changes genuinely different
nonlinear feature families at equal input information and equal five-parameter
head/fitting budgets. On 768 fresh assessment sources, task-aligned mean errors
were 0.001686013 (raw polynomial on polynomial), 0.001590098 (Fourier on periodic),
and 0.001675375 (hinge on hinge). The maximum aligned per-seed MAE was 0.001981757,
below the predeclared 0.02 screen in every seed.

On the periodic axis, Fourier reduced MAE relative to raw polynomial by
0.232820267, conditional paired 95% interval [0.222092321, 0.244795882].
On the hinge axis, hinge reduced it by 0.092969197 [0.090301902, 0.095871467].
On the original polynomial axis, Fourier and hinge instead increased error by
about 0.112 and 0.110. All full-information maps retain the same a,b, so the
differences concern what the frozen affine readout can access through each
feature family, not new information acquired by an encoder. Matching five head
parameters does not equalize the nonlinear spans. These hand-authored task-
aligned controls establish no learned encoder superiority.

The old mixed/raw-polynomial tie remains valid evidence of redundancy for the
old polynomial benchmark. The follow-up's raw-polynomial descriptor already
recovers that response well; there is no reason to install a more complicated
readout as the default. Conversely, a universal synthetic OLS/model binding
would hide misspecification on the other response axes. Small apparent gains
can be inconclusive: periodic hinge versus raw polynomial has a pooled paired
interval spanning zero. Complete axes, populations, seeds, changes, intervals
and negative controls remain in the report; no favorable subgroup was selected.

The source-block intervals describe conditional sensitivity of these frozen
models on the retained synthetic populations. They do not establish training-
process uncertainty, real-world generalization, causal identification or
multiplicity-controlled superiority. The lossy family ties zero because it
erases the changed coordinate, even though it has the same vector width.

## Accepted semantics and fitting rules

The reference input is one of the protocol's explicitly identified four-
coordinate representations, with its TRAIN-fitted population mean and standard
deviation. Output is one scalar named response axis in native units, under a
static non-temporal horizon. Contrasts subtract the same frozen head's baseline
response from its alternative response; spaces are never subtracted across
families. Both a and b must remain within [-1,1]. Unsupported alternatives have
no numerical contrast. Formula revisions and coordinate order are versioned
separately from fitted coefficients.

All 45 fitted heads and their controls are retained in three pinned model files.
The nine predeclared task-aligned seed/axis/family heads are accepted as response-
recovery references within this finite study. The remaining heads are diagnostic
controls, including valid negative results. They are not substituted into new
profiles. Mean/scale, coefficient and exact TRAIN split identities are bound in
the model files, source capsule, configuration and seal linked in the binding
record. This is a new policy identity, not a silent revision of historical OLS
bytes or a claim that an earlier global binding policy existed.

Fit means, scales and readout coefficients on TRAIN only. This study has one fit
per predeclared axis/family/seed, no regularization, hyperparameter search,
development selection or best-seed choice. Future bindings must freeze their
own TRAIN-only fitting and selection rule before protected assessment. Replacing
the solver, preprocessing, target, input shape, population, support or horizon
requires a new binding identity and prospective protocol. Nonfinite inputs,
unsupported alternatives and designs failing the pivot floor are failure/refusal
cases. Solvable ill-conditioned designs can remain unreliable: the implementation
does not detect all ill-conditioning, and they require their own validation.
Normal equations have no numerical-robustness guarantee outside the tested designs.

## Consequences for profiles and instances

No newly authored profile or instance may rely on the old or current OLS
reference until its contract/RER and instance dependencies link this decision,
state its limits, and bind an independently validated representation, target,
population, preprocessing, model and assessment protocol. Missing compatibility
or evidence withholds that dependent execution. Merely sharing four dimensions
or an OLS method name cannot supply the binding. This is a dependency policy,
not a newly implemented general fitting API or automatic scientific validator.

The existing [observational](../../profiles/instances/nsw_reference_v1/OBSERVATIONAL.md)
and [intervention](../../profiles/instances/nsw_reference_v1/INTERVENTION.md)
NSW references already declare `synthetic_ols_dependency: false`; their raw
one-dimensional earnings descriptor has no fitted readout. Their pinned
contracts, instance protocols and dependency records stay intact. This decision
introduces no OLS dependency, matching run or causal estimate into them.

The original `OR06-SYNTH-RESP-001 1.0.0` scalers/models, admission, verdict,
results and frozen source capsule are preserved byte for byte. The new binding
neither invalidates that scoped result nor promotes it to another population.
