# Synthetic controlled-response evaluation results

**Study:** `OR06-SYNTH-RESP-001`, version `1.0.0`.  
**Date:** 19 October 2026.  
**Disposition:** completed; independent audit valid, zero errors.  
**Verdict:** `supported_in_scope` for the frozen synthetic benchmark.  
**Prospective records:** [protocol](PROTOCOL.md), [instance RER](RER.md),
[configuration](configuration.json) and [exact interface](INTERFACE.md).

## Result and interpretation

The informative mixed representation with a TRAIN-fitted response readout
recovered the specified synthetic response differences with pooled source-unit
MAE **0.000762435**. The equivalent-capacity raw-polynomial readout matched it
within numerical rounding. Raw-linear error was 0.123479740. The lossy arm,
which removes the changed coordinate, matched the zero-response baseline at
0.302927979. All three seeds met every prospective screen criterion.

This is end-to-end evidence for the declared static synthetic population and
frozen readouts. Fidelity to the reference formula includes representation and
response-model fit error. The separate numerical oracle checks the comparison
operation's arithmetic. The mixed/raw-polynomial rounding difference supports
no superiority claim. A structurally valid contrast can be uninformative about
the reference response when its representation omits the changed information.

## Assessment population and uncertainty

Three seeds (1103, 2207, 3301) each used 512 training, 128 development and
256 assessment source units. All 12 fitted readouts and TRAIN-only scalers
were sealed before any assessment input was generated. No tuning, alternative
selection, best-seed selection or model changes followed development results.
All five supported changes per source unit, including no change, enter its
MAE; the sixth, unsupported change enters the refusal denominator separately.

| Arm | Pooled source-unit MAE | Conditional bootstrap 95% interval |
| --- | ---: | --- |
| Informative mixed | 0.000762435 | [0.000722313, 0.000801516] |
| Equivalent raw polynomial | 0.000762435 | [0.000722313, 0.000801516] |
| Raw linear | 0.123479740 | [0.117712027, 0.129234590] |
| Lossy | 0.302927979 | [0.292339969, 0.313371865] |
| Zero response | 0.302927979 | [0.292339969, 0.313371865] |

| Seed | Mixed MAE | Raw-polynomial MAE | Raw-linear MAE | Lossy / zero MAE |
| --- | ---: | ---: | ---: | ---: |
| 1103 | 0.000672988 | 0.000672988 | 0.121129160 | 0.294630027 |
| 2207 | 0.001194118 | 0.001194118 | 0.119770207 | 0.309329033 |
| 3301 | 0.000420199 | 0.000420199 | 0.129539854 | 0.304824877 |

The 2,000-resample source-unit bootstrap preserves each unit's alternatives
and uses common within-seed draws across arms. Its intervals are conditional
sensitivity for these three frozen fitted models; they provide no population
or training-process confidence guarantee. Raw-linear has fewer parameters;
only raw-polynomial is the equivalent-capacity control. Full development
metrics are retained in the audit rather than used to tune the study.

## Coverage, controls and independent checks

- Assessment: **18,432 requested pairs**, comprising 15,360 supported successes
  and 3,072 out-of-support abstentions; zero failures or unresolved pairs.
  Thus supported-input coverage and stress abstention were both 100%.
- Development: 9,216 requested pairs, 7,680 successes and 1,536 stress
  abstentions; zero failures or unresolved pairs. No row was silently excluded.
- No-change contrasts were exactly zero. The maximum independent saved-model
  contrast discrepancy was `1.7763568394002505e-15`, below the fixed `1e-10`
  tolerance. TRAIN-only scaler/fit reconstruction discrepancies were zero;
  maximum normal-equation residual was `1.1368683772161603e-13`.
- The independent auditor verified generator reproduction, disjoint source
  populations, actual artifact hashes, descriptors and typed request lineage,
  input changes/masks, representations, model responses, both contrast oracles,
  all state/roster counts, metrics and bootstrap. It reported zero audit errors.
- A separate seed-4001 engineering preflight passed. An intentionally corrupted
  copy, with one output contrast increased by one, was rejected by the auditor.
  Those engineering populations are excluded from scientific results.
- Independent read-only review found no remaining false-positive defect in the
  final aggregation, numeric checks, artifact references or interpretation.

## Actual run identity and resources

Run command, through the existing managed container and authoritative task:

```text
bash code/scripts/task.sh hx06-evaluation-001 evaluate-counterfactuals EVALUATION_ATTEMPT=/mityane/.temp/operations/counterfactuals/synthetic-response-v1/attempt-001
```

GNU C++ 12.2.0 and pinned Linux Node v22.23.3 were used. The wrapper ran one
native CPU process at a time, without inherited encoders, external datasets,
network providers, dependency changes or GPU execution. Total recorded wall
time was **182.103 seconds**, including source/model checks, artifact I/O and
independent audit. Peak memory and CPU time were not instrumented; no performance
claim follows. The retained file manifest covers 113 files / 585,334,681 bytes
(about 558 MiB), dominated by explicit provenance records. Evidence is preserved;
no earlier artifacts were removed.

| Identity | SHA256 |
| --- | --- |
| Frozen source closure | `245bd991d90eac8bcc0ab3aac769fa1b583b40559e4dd5f16d62161d2b4417da` |
| Executed sealed native binary | `8593233df930cb4db220e5ffb688244976727edac7e93a1781071b9f9293fd69` |
| Prospective freeze receipt | `1ba6b2340e8742021c104fb27f9f1c3f64cb88bf367c6b68d75ae29a9691334b` |
| All-model seal | `89954ba9173fb1cb3959ebe3f726e6c7f3d1432909f6571966b592fcdfeb04bb` |
| Independent audit | `72f47432427df38aef248a3bd9400a83b4f753cd97d313c6276e23083dc3eb7c` |
| Retained file manifest | `87bb0560c9ac02a1101b7ae8d7aa5215d91e42e3af330368cbda9853d183e459` |

Actual generated data, scalers/models, source capsule, run records and audit are
in the ignored [attempt directory](../../../../../.temp/operations/counterfactuals/synthetic-response-v1/attempt-001).
The append-only [operation-local index](../../../../../.temp/operations/counterfactuals/synthetic-response-v1/index.jsonl)
records registration, model seal, assessment opening and completion. The exact
executable is retained inside the container under the `/opt/mityane-geometrics/operations/counterfactuals/sealed/`
path pinned in `freeze.json`. The source capsule preserves execution-time bytes;
later report/status edits are not substitutes for that frozen source identity.

## Limits and milestone decision

Hx06's bounded operation-specific evaluation milestone is complete. Its initial
three-milestone plan covered contract, implementation and this scoped evidence.
The result establishes neither real-world counterfactual validity nor causal
identification, individual effects, market utility or inherited encoder quality.
It does not implement the observational or intervention-effect profiles.
Those require their own future scope, contracts and admitted evidence.
