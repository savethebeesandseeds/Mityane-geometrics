# Prospective synthetic controlled-response study

**ID:** `OR06-SYNTH-RESP-001`, version `1.0.0`.  
**Operation:** [OR06-PCR 0.1.0](../../CONTRACT.md), controlled model response.  
**Instance RER:** [OR06-SYNTH-RESP-001-RER](RER.md).  
**Exact recipe:** [configuration](configuration.json) and [implementation interface](INTERFACE.md).

## Question and hypotheses

On the declared synthetic population, how accurately does the complete
representation/readout/comparison pipeline recover the specified reference
response differences, and does it refuse unsupported alternatives?

The primary hypothesis is that the mixed informative representation with its
TRAIN-fitted readout achieves source-unit MAE <=0.02 for each of the three
prespecified seeds, matches the equally expressive raw-polynomial baseline
within 1e-9 MAE, and has pooled MAE <=20% of the zero-response baseline.
In-support coverage and stress abstention must both be 100%, and the independent
numerical audit must pass. All criteria are conjunctive; all seeds are reported.
A valid run failing any scientific criterion yields a scoped negative result.

There are two distinct oracles. Saved-model contrast reconstructed independently
from its coefficients checks numerical execution. The closed-form reference
`h*(2+1.5*b+a)+0.5*h*h` checks synthetic response fidelity. Disagreement with
the latter includes readout/representation approximation error; it does not
by itself identify a defect in subtraction or a real-system causal effect.

## Population, splits, representations and controls

Use exactly the generator, rational grid, split order and unique-b reservation
in INTERFACE. Each seed supplies 512 training, 128 development and 256 protected
assessment source units. The four fitted arms have the fixed features and
parameter counts declared there. Labels have the fixed training noise; the
reference response contrast is noiseless. No external dataset, inherited
encoder, API, GPU, pretrained weights or model selection is part of this study.

Fit every mean, scale and coefficient only on training rows. Development is
reported for transparency; it changes no model, threshold or population. Seal
all three models before opening any assessment population. No hyperparameter
search, assessment-derived support calibration or best-seed choice is allowed.
Alternative inputs preserve their source-unit parents and held-fixed b.
The reserved unique b values prevent exact source/input overlap across splits,
including alternatives. Record generator state, generated rows and their hashes.

Every assessment unit has changes -1/4, -1/8, 0, +1/8, +1/4 and +2 in a.
The first five stay within the declared [-1,1] support; +2 is an explicit
out-of-support stress case. Include all six in coverage. Report zero-response,
raw-linear, raw-polynomial, mixed and lossy results. Raw-linear has lower
capacity; raw-polynomial is the capacity-matched equivalent-function control.
The lossy arm's equal dimensions do not imply retained information about a.

## Metrics and uncertainty

Per-unit primary error is mean absolute reference-contrast error over the five
in-support alternatives, including no change. Per-seed MAE averages its 256
units; the pooled metric macro-averages the three seed MAEs. Report every arm,
the zero baseline, no-change maximum error, in-support success coverage,
stress abstention, all requested/success/abstained/failed/unresolved counts,
and numerical audit maxima. No refused alternative enters an ordinary contrast.

Use the 2,000-resample within-seed source-unit bootstrap and percentile-index
rule in INTERFACE, preserving all alternatives from each source. Its interval
is conditional sensitivity for the frozen fitted models, not uncertainty for
all models or a real-world population. Report per-seed variability separately.
The retained complete finite assessment roster is the exact population for
the main descriptive metrics.

## Admission, execution and evidence

Before fitting, register an exclusive attempt, save this protocol/configuration
and RER's source identities, snapshot the complete execution/audit source
closure and binary, and record the user-authorized Hx06 continuation scope.
This task's bounded local CPU study is the selected implementation of that
request; it does not expand authorization to another Hero or external service.

Use the managed container and root task entry, one CPU process at a time,
with a 60-second native-process timeout and expected peak below 1 GiB.
Keep build products under `/opt/mityane-geometrics` and actual data, models,
records, logs and JSON receipts under the exclusive ignored
`.temp/operations/counterfactuals/synthetic-response-v1/<attempt>/` directory.
Do not overwrite an attempt. The append-only operation-local index records
registration, seal/open transitions and every final disposition, including
rejected admissions, interrupted/failed executions and invalid audits.
SIGINT/SIGTERM stop the active native child and finalize an interruption record.
After an uncatchable termination, `run.mjs --recover --attempt <exact-path>`
can finalize the preserved attempt only after its original process/start token
is absent. Recovery does not replay work or infer a scientific verdict.

The source capsule stays with the attempt. An exclusive executable snapshot is
retained and executed under `/opt/mityane-geometrics/operations/counterfactuals/sealed/`,
keeping executable build artifacts inside the container. A permanent exclusive
study admission and prior-index assessment-exposure guard prevent a second
same-protocol run from being called fresh confirmation.

The admitting wrapper hashes actual files, rechecks the frozen source and
model seal across execution, and only then trusts the native typed adapter's
pins. The independent auditor verifies saved input/model/representation/response
bytes, TRAIN-only fitting, oracles, roster and metrics from retained artifacts.
The core library's supplied-check trust boundary remains explicit.

## Changes, failures and interpretation

Separate small engineering self-check populations use seed 4001 before study
freezing and establish no scientific result. Fixing a failure preserves its
attempt. A changed scientific protocol/source needs a new sealed attempt and
explicit exposure record; an exposed assessment population cannot silently
become fresh confirmation. No retries or selection based on scientific scores.

Audit invalidity or execution failure leaves the evaluation milestone incomplete.
A valid positive, negative or inconclusive synthetic result may complete this
bounded evaluation. Its evidence concerns this static toy response map and
fitted readouts only. It supplies no intervention identification, individual
counterfactual truth, inherited encoder quality or real-world usefulness claim.
