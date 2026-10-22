# NSW matched observed-contrast instance

**Instance:** `OR06-MOC-NSW-REF-001`, revision `1.0.0`.  
**Instance RER:** `OR06-MOC-NSW-REF-RER`, revision `1.0.0`.  
**Profile:** [OR06-MOC / OR06-MOC-RER 0.1.0](../../observational_contrast/v0.1.0/CONTRACT.md).  
**Disposition:** actual-source binding complete; no matching execution or scientific verdict.

## Question, population and hypothesis

For this finite archival cohort, how does observed 1978 earnings differ between
the 185 assignment-group participants and controls matched on 1974 earnings?
The claim is `observational_contrast`, even though the source originated in a
randomized study. This instance deliberately reports the weaker observed
association; no causal interpretation is inherited from the source or matches.

The source roster has 445 study rows: 185 focal (`train=1`), 260 candidate
controls (`train=0`). Row IDs are immutable snapshot/row references, not names
or purported original participant IDs. No external population, individual
counterfactual or policy value is targeted. Requested focal denominator is 185;
candidate denominator is 260. Missing or rejected units remain reported.

The diagnostic hypothesis is that the fixed raw-descriptor matcher meets the
predeclared support/coverage/balance gates below. Its eventual result may refute
that hypothesis. A numeric contrast or nonzero difference does not prove
representation utility, adequate confounding control or identification.

## Actual bindings and covariate audit

[bindings.json](bindings.json) pins the dataset, source/license documentation,
provider, SpaceRef, roster and baseline descriptor bytes. Outcome `Y` is `re78`
in thousands of real dollars; the original CSV binds its measured values.
The source-described outcome is calendar-year 1978 earnings after the job-training
assignment/program period. Exact per-person assignment and availability dates
are unavailable, so mode is retrospective only.

Permitted matching information is only `[re74]`, the source-described
pre-intervention 1974 earnings. Geometry is absolute distance in the same
$1,000 units. There is no fitting/scaling/imputation and no OLS. Prohibit
`train`, `re78`, `unem78`, `lre78`, `mostrn`, later/outcome-derived or
post-assignment information, identifiers encoding the group and any hidden
version of those inputs. Other source variables are not matching coordinates.
Source-described demographic/baseline variables may be used for diagnostic
balance tables only, with timing ambiguity labeled; unsupported baseline status
cannot become an adjusted-confounder claim.

## Frozen reference protocol

- Match each focal unit to one control with replacement, using nearest absolute
  `re74` distance and caliper **2.0** ($2,000). Seal matches before attaching
  outcome values. No outcome-guided exclusions, rematching or tuning.
- Order focal rows by SHA256 of the fixed prefix `HX06-MOC-NSW-v1:focal:` plus
  decimal source row ordinal. Among equally near controls, choose least reused,
  then SHA256 of `HX06-MOC-NSW-v1:control:` plus ordinal. Neither ordering consumes
  outcome values or dataset-content hashes. IDs are not distance features.
- Use unit weight one per selected control match. Each supported focal contrast
  is `re78[focal] - re78[control]`; the declared aggregate is their uniform mean
  over the explicitly retained focal subset. It is not the full-185 mean if
  any focal unit is unsupported. Report that changed descriptive population.
- Primary gates: supported focal fraction at least 0.90, matched control-weight
  effective sample size at least 30, largest normalized control weight at most
  0.20, and absolute standardized mean difference for `re74` at most 0.10.
  SMD uses fixed pre-match pooled SD
  `sqrt(((185-1)*s_focal^2 + (260-1)*s_control^2)/(185+260-2))`;
  zero denominator yields zero only when means agree, otherwise an unsupported
  diagnostic. With control reuse counts `r_j` and `m` supported focal units,
  `p_j = r_j/m`, control ESS is `1/sum_j(p_j^2)` and the largest normalized
  control weight is `max_j(p_j)`.
- Report pre/post raw-variable balance, distributional differences, caliper
  exclusions, distances, reuse/weights, ESS, full success/refusal counts and
  omitted-source-variable inventory. Known demographic differences and unmeasured
  selection remain limitations; balanced `re74` alone addresses neither.
- Controls are unmatched group means and the identical raw-scalar matcher
  (the representation is that raw descriptor). Learned/random alternatives and
  propensity fitting are not bound; no representation-improvement claim is made.
  Negative-control outcomes are not supplied; do not relabel observed pre-period
  earnings as proof against unmeasured confounding.
- The target is a deterministic finite-cohort descriptive contrast, with no
  sampling CI or external-population inference. Report this uncertainty limitation
  explicitly; reused controls are not independent pairs. A request requiring
  unsupported sampling/generalization uncertainty receives
  `UNCERTAINTY_UNSUPPORTED`, rather than a naive pair-bootstrap interval.

No fit/development/calibration partition is used: all learned fitting is
`not_fitted`; all 445 rows belong to the exposed archival reference cohort.
This is not a protected assessment split. A future learned representation or
tuned design needs a new protocol, actual fit partitions and exposure record.
The design above was specified without computing matched or outcome summaries.

## Refusal, permissions and evidence

Apply every [profile refusal](../../observational_contrast/v0.1.0/CONTRACT.md),
including integrity/prohibited-input failure, `NO_ELIGIBLE_MATCH`, support,
balance and coverage abstention. Failure takes precedence over abstention;
never replace missing/refused contrasts with zero or silently change a gate.
Admission rejection records no executed units; interrupted execution retains
unresolved IDs. Final counts must reconcile the full 185 requested focal units
and the separately reported candidate roster.

Only acquisition, schema/artifact validation, baseline-descriptor materialization
and binding/document checks are presently authorized. Matching, effect/contrast
estimation, training, model/API calls and scientific execution are not admitted.
Future admission must bind implementation/source hashes, CPU budget, output
destination and pre-execution attempt under the shared managed environment.
Retain every disposition and the actual result/limitations before any claim.

**Current decision:** bound archival reference; no empirical result. Reproduction
starts with the exact data receipt and descriptor identities in bindings.json,
this protocol and the versioned profile/RER. Keep original bytes and create new
binding versions for changed inputs or methods.
