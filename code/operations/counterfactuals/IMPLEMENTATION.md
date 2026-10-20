# Paired-response C++ reference

**Implementation:** `OR06-PCR-CPP`, revision `0.1.0`.  
**Interface:** [paired_response.h](include/mityane/operations/counterfactuals/paired_response.h).  
**Implementation source:** [paired_response.cpp](src/paired_response.cpp).  
**Contract:** [OR06-PCR 0.1.0](CONTRACT.md).

This standard-library C++20 library validates supplied records and compares
recorded deterministic responses. It performs no encoding, fitting, provider
calls, alternative generation or causal estimation. Its supported slice uses
finite numerical responses, a complete named scalar control manifest and one
exact input-role/SpaceRef identity.

## API and result ownership

The namespace is `mityane::operations::counterfactuals::v1`. A caller constructs
an explicitly bound `Request` and calls `compare(request)`. No request defaults
silently grant a claim level, temporal mode, tolerance or output selector.

```cpp
namespace cf = mityane::operations::counterfactuals::v1;
cf::RunResult result = cf::compare(bound_request);
if (!result.admitted) {
  // Inspect admission_diagnostics; no pair execution was recorded.
} else {
  // Inspect each pair's state before accessing its optional contrast.
}
```

`RunResult.input` retains the complete supplied request, including contract/RER,
run, source/parent, representation kind, SpaceRef, fitted dependencies,
configuration, checks and evidence references. Each `PairResult` refers back by
ordered `pair_index` and `pair_id`. Keep the run and its pair results together;
an extracted numerical delta alone is not a provenance-bearing result.

`compare(request, stop_after)` processes at most that many roster entries and
labels every remaining entry `unresolved`. An admitted run always reconciles
success, abstention, failure and unresolved counts against its full roster.
Admission rejection retains the supplied request and diagnostics without
inventing executed pairs or a success rate.

## Concrete bindings and supported policies

- Artifact IDs resolve to one SHA256 pin. Conflicting pins or contradictory
  repeated input/representation/response metadata reject admission. The same
  baseline may participate in distinct alternatives; duplicate input-pair tuples
  are rejected. Reuse preserves entity/group IDs and creates no independent
  sampling or aggregate-effect claim.
- The expected pipeline binds six required dependency roles: `provider`,
  `preprocessing`, `fitted_transform`, `response_model`, `support_policy` and
  `constructor`. The first four match the selected space/model pins; each pair
  matches the constructor pin. `support_policy` identifies the full declared
  validity, support and interpretability checking procedure. Include all further
  transitive dependencies as additional unique roles. Fit/selection populations
  and pretraining exposure are mandatory; unfitted artifacts use explicit
  `not_fitted` population declarations.
- Both branches must match the full expected space, pipeline, response axes,
  definitions and units. Changing a model alias, preprocessing, normalization,
  geometry or input role cannot pass on dimensions alone. This slice does not
  implement cross-space mappings. An upstream mapped input must already have
  the required canonical role/space, with the mapping and its transitive fitted
  dependencies retained by the adapter.
- The provider revision must be known and the stochastic policy explicitly
  `deterministic`. Unknown hosted revisions and stochastic repetitions are
  refused. No averaging, confidence interval, relative contrast or derivative
  is produced.
- `selected_output_position` is an explicit nonnegative adapter-defined
  selector. Both branches must equal it and carry an attested valid mask and
  mask pin. The adapter verifies how the selector maps into the actual tensor
  layout and whether it is in range; this library does not decode mask/tensor
  bytes.
- Every scalar manifest covers exactly the named attribute rules, with finite
  values and matching units. Rules distinguish changed, held-fixed and coupled
  values. Bounds apply to both branches. The single finite nonnegative absolute
  `control_tolerance` applies to each numeric control/identity comparison in its
  declared units; no automatic rescaling occurs. A held-fixed change beyond it
  is a failure. Bounds/support rejection is an abstention for otherwise valid
  records. Identical input-content pins with inconsistent deterministic
  responses fail the identity control.
- Baseline inputs retain observed origin; task/predicted states may refer to
  that input while remaining distinct representation kinds. Candidate/generated
  representations cannot be relabeled as an observed baseline. Alternatives
  retain their constructor, derived origin, exact baseline parents and distinct
  record identities. Latent candidates support only the explicit
  `latent_coordinate_response` interpretation in this slice.
- Arithmetic uses IEEE 754 binary64 `double`: alternative minus baseline,
  without normalizing or reordering axes. Non-finite inputs and subtraction
  overflow fail. The compiler uses no fast-math flags. Failure takes precedence
  over a supplied support/validity rejection; refusals contain no contrast.

## Temporal semantics

All timestamp integers use the same UTC Unix-second time domain. Feature support
intervals describe information dependencies, separately from the response target
horizon. Reversed intervals or support later than declared availability fail.
As-of mode requires a cutoff, complete eligible branch/dependency availability,
and known fit/selection exposure; unknown exposure cannot be invented as an
eligible date. Non-temporal mode requires a justification and has no cutoff.

The output boundary is the maximum availability across both branches and all
declared dependencies. If any is unknown in retrospective/non-temporal mode,
the result boundary stays unknown. This slice provides no tighter dependency
proof or automatic temporal-mode downgrade.

## Trust and evidence boundary

All references, scalar manifests, masks, control evidence and checker outcomes
are supplied by the caller. The library checks their structure and internal
consistency. It does **not** read/hash referenced bytes, authenticate an encoder,
reconstruct a source input, verify an upstream execution or prove an attestation
true. It also cannot discover omitted dependencies or assessment exposure.
The admitting loader/adapter must verify those facts, the completeness of the
dependency roster, actual resource/fit permissions and the validity of numerical
response definitions. Category IDs are not valid numerical response scores.

Typed admission here means API conformance, not human authorization or scientific
study admission. Artifact hashes establish declared identity relationships;
they do not establish semantic validity, overlap or causal identification.
The supplied checks remain labeled supplied checks in `RunResult.input`.

## Build and verification

From the repository root, use the existing managed container:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/task.sh', 'hx06-reference-001', 'test-counterfactuals'
)
```

The root Makefile includes operation-owned [build rules](counterfactuals.mk).
They use only the C++ standard library and write products below
`/opt/mityane-geometrics/embeddings/build/hx06-reference-001/counterfactuals`.
The [47 engineering conformance cases](tests/paired_response_test.cpp) exercise
the public interface without real data, encoders or GPU execution. The
[verification record](reviews/reference-v0.1.0.md) identifies the tested sources
and current outcome. The separately frozen [synthetic evaluation](evaluation/synthetic_response_v1/RESULTS.md)
now supplies bounded end-to-end evidence; the reference fixtures retain their
engineering-only interpretation.

The [callable tool](TOOL.md) adds provider-neutral file loading, actual artifact
hash and response-payload checks, a shared CLI/Node API and saved-result inspection
around this unchanged numerical API. Its [engineering verification](reviews/tool-v1.md)
is separate from the original reference and synthetic study records.
