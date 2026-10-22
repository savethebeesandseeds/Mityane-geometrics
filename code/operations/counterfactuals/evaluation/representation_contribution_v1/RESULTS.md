# Representation contribution result

**Study:** `OR06-REP-CONTRIB-001`, version `1.0.0`, 20 October 2026.
**Disposition:** completed, valid, `supported_in_scope`.
**Bindings:** [frozen protocol](PROTOCOL.md), [requirements and evidence](RER.md),
[configuration](configuration.json), [readout decision](READOUT_DECISION.md).

Different fixed representation families change which responses an equal-capacity
affine readout can recover. The raw-polynomial descriptor remains sufficient for
the original polynomial response; Fourier and hinge features substantially reduce
error on their corresponding synthetic axes and increase it on the original axis.
The old mixed/raw-polynomial tie therefore remains evidence of redundancy within
that old benchmark. It does not establish general representation redundancy.

No learned encoder was evaluated. These are project-authored nonlinear feature
maps with known task compatibility, four coordinates and five fitted parameters
per scalar head. The first four retain identical raw a,b information; their
different affine-accessible function classes explain the comparison. The lossy
control removes a. Neither equal dimensions nor equal parameter counts equates
information content or total function classes.

## Overall protected assessment

MAE averages the five supported absolute contrast errors within each source,
then the 768 sources. Every seed and family is retained. All values below are
from the frozen independent audit, displayed to nine decimal places.

| Response axis | Raw polynomial | Raw additive | Fourier | Hinge | Lossy | Zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| polynomial | 0.001686013 | 0.111784274 | 0.113872507 | 0.112139486 | 0.294711614 | 0.294711614 |
| periodic | 0.234410365 | 0.234558541 | 0.001590098 | 0.230720280 | 0.476831820 | 0.476831820 |
| hinge | 0.094644572 | 0.093482062 | 0.048296060 | 0.001675375 | 0.291139221 | 0.291139221 |

The predeclared aligned heads passed the MAE <=0.02 screen in all three seeds.
No family, head, threshold, group or seed was selected using development or
assessment. The following conditional 95% intervals keep all alternatives of a
source together and resample sources within seed using 1,000 shared bootstrap
draws. Positive paired reductions favor the named family.

| Axis / aligned family | MAE, 95% interval | Raw polynomial minus family, 95% interval |
| --- | --- | --- |
| polynomial / raw polynomial | 0.001686013 [0.001622378, 0.001754771] | 0.000000000 [0.000000000, 0.000000000] |
| periodic / Fourier | 0.001590098 [0.001544086, 0.001637882] | 0.232820267 [0.222092321, 0.244795882] |
| hinge / hinge | 0.001675375 [0.001650110, 0.001703002] | 0.092969197 [0.090301902, 0.095871467] |

On the original polynomial axis, Fourier instead increases MAE by 0.112186494
(paired raw-polynomial-minus-Fourier interval [-0.117500703, -0.107108329]);
hinge increases it by 0.110453473 [-0.115654987, -0.106005990].
On the periodic axis, the small raw-additive difference is inconclusive
(-0.000148176 [-0.000510346, 0.000210424]); hinge versus raw polynomial is also
inconclusive (0.003690085 [-0.002372117, 0.008922309]). These results are retained,
alongside the larger task-aligned gains. Lossy produces exactly zero contrasts
and exactly ties zero on every axis, seed and supported group.

## Seed, population and perturbation breakdowns

Each seed has 384 TRAIN sources, 96 development sources and 256 assessment
sources. All 45 heads and their scalers were sealed before any scientific
assessment population was generated. Task-aligned assessment MAEs were:

| Seed | Polynomial / raw polynomial | Periodic / Fourier | Hinge / hinge |
| --- | ---: | ---: | ---: |
| 12037 | 0.001919639 | 0.001652931 | 0.001609289 |
| 23039 | 0.001851015 | 0.001800753 | 0.001981757 |
| 34057 | 0.001287386 | 0.001316610 | 0.001435080 |

The population signs contain 398 a-negative / 370 a-nonnegative sources and
389 b-negative / 379 b-nonnegative sources. Fourier improves on raw polynomial
on the periodic axis in both a groups, with paired reductions 0.274584593
[0.256762196, 0.294832850] and 0.187895398 [0.178825541, 0.196671632]. Its
aligned errors are 0.001790572 and 0.001374453 respectively. Hinge's corresponding
hinge-axis reductions are 0.092296731 [0.088522049, 0.096452859] and 0.093692551
[0.089877603, 0.097628909]. The task-aligned gains also persist in both b groups.
These predefined strata do not establish effects in other populations.

Small aggregate differences can conceal reversals. On the periodic axis, hinge
versus raw polynomial has a paired reduction of 0.069541 for a<0 but -0.067144
for a>=0, and 0.033363 for negative changes but -0.024138 for positive supported
changes. Its pooled interval spans zero. On the hinge axis, raw additive's small
pooled improvement reverses in seed 12037; the all-source intervals for seeds
12037 and 23039 span zero. These cases remain visible in the full report.

Larger supported changes have larger absolute errors. For changes of magnitude
1/8 versus 1/4, aligned MAEs are 0.001385408 versus 0.002829625 on polynomial,
0.001353415 versus 0.002621830 on periodic and 0.001434227 versus 0.002754211 on
hinge. The periodic Fourier advantage over raw polynomial is 0.190407683 versus
0.391642985 for those magnitudes. It is also asymmetric by direction:
0.332104603 [0.311452859, 0.354117741] for negative changes versus 0.249946065
[0.239266559, 0.261825391] for positive supported changes. Both directions are
retained. No-change contrasts are exactly zero for all heads. The +2 stress
alternative always abstains; its MAE and intervals are null.

The [complete breakdown report](../../../../../.temp/operations/counterfactuals/representation-contribution-v1/report-001/BREAKDOWNS.md)
contains **864 assessment rows and 864 development rows**: every combination of
three axes, five fitted families plus zero, three seeds plus pooled populations,
and all twelve predefined groups. It includes every MAE, assessment interval,
paired raw-polynomial/zero difference and interval, source count, full coverage,
finite bootstrap count and null stress result. All non-stress assessment groups
have 1,000 finite resamples and zero empty draws. Development has all point
metrics, no bootstrap intervals and no selection role. Full-precision numbers
are in the [local metrics record](../../../../../.temp/operations/counterfactuals/representation-contribution-v1/report-001/metrics.json).

## Coverage, execution and numerical audit

| Actual native split | Requested | Succeeded | Abstained | Failed | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| development | 25,920 | 21,600 | 4,320 | 0 | 0 |
| assessment | 69,120 | 57,600 | 11,520 | 0 | 0 |

These execution counts include five fitted families, not an extra execution
for zero. Each source/axis/family has five successful supported changes and one
`out_of_support` abstention. Unsupported rows contain no numerical result.
The independent [audit](../../../../../.temp/operations/counterfactuals/representation-contribution-v1/attempt-001/audit.json)
regenerated every input, exclusion, feature, TRAIN-only OLS fit, saved prediction,
contrast, truth, identity and coverage roster before computing the predeclared
breakdowns. It recorded **zero errors**. Maximum independent fit discrepancy was
4.440892098500626e-16; maximum prediction discrepancy was 2.6645352591003757e-15,
below the 1e-9 numerical threshold. No-change and lossy contrast maxima were zero.
The maximum task-aligned per-seed MAE was 0.0019817567820548374.

The single admitted scientific attempt took 69.110591882 seconds in managed
container `mityane-geometrics` (ID prefix `381c6b78f4fd`), using sequential CPU
C++20 evaluation and the independent Python 3.11.2 audit. Compiler was GCC
12.2.0; orchestration used Node v22.23.3 and the pinned existing image. No GPU,
external data, service or model was used. Native timeout was 60 seconds per
stage and audit timeout 300 seconds; expected peak memory was below 1 GiB,
but peak memory was not measured. No dependencies were installed.

The scientific task entry was:

```powershell
.\container.ps1 -Action exec -Command @('bash','code/scripts/task.sh','hx06-contribution-v1','evaluate-counterfactual-contribution','EVALUATION_ATTEMPT=/mityane/.temp/operations/counterfactuals/representation-contribution-v1/attempt-001')
```

This records the completed invocation, not permission to replay the exposed
assessment. The exclusive attempt, permanent admission and append-only ledger
prevent another execution from being called fresh confirmation.

## Attempts, prior exposure and engineering corrections

There was one scientific admission and one completed scientific attempt, with
no scientific retry or score-based change. The earlier original study was not
replayed; all eight original source files and its source capsule remain unchanged.
Its familiar task/formula and old 0.000762435 tie were already exposed. The new
0.001686013 raw-polynomial error uses different fitted heads and fresh sources;
the two studies' score difference is not a measured improvement or regression.

Before the scientific freeze, engineering work exposed candidate raw coordinates
for seeds 12011, 23003 and 34019 with counts 512/128/384. No candidate heads were
fitted and no scores were observed. These candidates were discarded from
scientific admission and every candidate b numerator was reserved. Final seed
and count choices occurred before admission to fit the remaining finite
coordinate capacity, rather than to improve scores.

The first engineering attempt at
`/opt/mityane-geometrics/operations/counterfactuals/contribution-engineering-yNC3Eb/attempt-001`
failed audit because its reduced 100-draw engineering bootstrap conflicted with
the then-fixed 1,000-draw audit expectation. Its `failure.json` is retained; it is
an invalid engineering audit, not a negative scientific result. Corrected
engineering attempt `contribution-engineering-DC1uyu/attempt-001` passed, as did
the final preflight `contribution-engineering-529VGs/attempt-001` after complete
development reporting was added. Both are under the same `/opt/mityane-geometrics/operations/counterfactuals/`
parent. The final four-test preflight covered generator separation, full
engineering execution/audit, exclusive-destination preservation and deliberate
contrast-corruption rejection; the native selftest covered 180 engineering pairs.

All original-source, exposed-candidate, engineering-recipe and native-selftest
b values were excluded before scientific admission: **5,640 unique exclusions**.
The saved [prior-exposure record](../../../../../.temp/operations/counterfactuals/representation-contribution-v1/attempt-001/prior_exposure.json)
and frozen independent regeneration verify global old/new and new/new separation
across all seeds and splits. Reserving b also separates every a-only alternative.
The saved seal precedes the assessment-open receipt; all sources are now exposed
and must not be relabeled protected in a later study.

Presentation work happened after completion. Its first invocation stopped on a
JavaScript template-string syntax error before publishing any report. Syntax was
corrected, the container syntax check passed and the final presentation completed.
It verified all 85 pinned attempt files and copied the existing audited metrics;
it did not refit, resample or reopen assessment. The report source is deliberately
outside the frozen scientific execution identity. Core conformance (47 cases),
profile/data tests (nine cases), and the actual two-instance/445-row/27-file
binding check also passed at closeout in the
[local log](../../../../../.temp/operations/counterfactuals/representation-contribution-v1/closeout-001/core-and-profiles.log).

## Bound evidence identities

Retained attempt:
`.temp/operations/counterfactuals/representation-contribution-v1/attempt-001/`.
All saved models, inputs, pair rows, source capsule, process statuses and logs
are covered by its manifest. Generated models/data/reports remain ignored by Git.

| Record | SHA-256 |
| --- | --- |
| Source closure ID | f0a174ff2dd0abff78cdd390c010495779ff523ba17294e625f2cd1734a5eb74 |
| Frozen binary | 80afabcdb8f67155060a4c47da30e08affc19df5494d957c27d0e4e30e2183bb |
| freeze.json | c5cadff0e6ab687a2857daddd7c03786d743a6adba7650c94a506b5f86e6af5a |
| seal.json | cecac2731ab2574c61319059bf45b6a661fc1757155f4c66b1a4e96e3e84c105 |
| audit.json | 7197bca45a746b68136ca57f979e17c10dfe2c3fff81d6abd66e643682be0b81 |
| file-manifest.json | d7817315cd2465a0c05ba8c84e891923b62ba7c140588d612d4f563d705a9b37 |
| final.json | 9730ed8355839ccfa537d527cb763604145b8e3a7217dcbb75272108fc609642 |
| Complete BREAKDOWNS.md | 30b6f5ce3e82dfdbf654ade21a6c4753238266f598e4d7d6599121d4a902f9fe |
| Complete metrics.json | 0a61b302c0c5fb0a638f2e949ba211d8a4c4b5dc741e1137a27b196bf79df4da |

The post-execution [report receipt](../../../../../.temp/operations/counterfactuals/representation-contribution-v1/report-001/receipt.json)
pins the presentation script and outputs separately from the scientific freeze.

## Limits and readout consequence

This resolves the finite synthetic contribution question: usefulness changes with
the target and feature/readout compatibility, and a simple descriptor already
solves the original response. It neither establishes added signal from a learned
encoder nor certifies a representation for an unseen task. Conditional source-
block intervals exclude uncertainty from retraining, task choice and external
populations, and are not corrected for multiple comparisons. Their coverage
cannot be interpreted as simultaneous superiority across all reported groups.
No geometry/distance quality, temporal forecast, observational contrast or
identified intervention effect was measured. Component evidence alone does not
establish a composed pipeline's scientific utility.

The [binding decision](READOUT_DECISION.md) keeps TRAIN-only OLS as a scoped
reference method and revises its binding policy, with a new identity and explicit
task/population validation. The old fitted artifacts remain preserved. No new
profile receives an implicit OLS dependency; the separately bound NSW profiles
continue to declare none.
