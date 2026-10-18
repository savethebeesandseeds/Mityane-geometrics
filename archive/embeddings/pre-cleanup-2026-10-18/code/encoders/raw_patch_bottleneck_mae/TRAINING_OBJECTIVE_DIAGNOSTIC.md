# TRAIN objective diagnostic

Date: 2026-10-07. Measured and independently audited; zero updates and zero refits.
Protocol: training-objective-diagnostic-v1; policy 1.2; known TRAIN only.

The predeclared all-pair auxiliary hypothesis returns **null**. It improves local correct-margin alignment on two of five masters and worsens it on three. Original reconstruction descent opposes the saved Ridge margin on four masters, but adding this auxiliary does not consistently resolve that tension. RPB-v4 remains active; no encoder is introduced or promoted.

Fifteen retained RPB-v4/v7/v8 instances at 512 updates use masters 4404/5505/6606/7707/8808, timing (lag_sign), 128 TRAIN source pairs / 256 examples per master, C3/H32/F3, patch 8, batch 8 and 225,805 parameters. Actual CUDA supplies frozen forwards and gradients. The diagnosis creates no optimizer, training update or readout fit and opens no VALIDATION, TEST, stress or point 0 payload. Four full-TRAIN common-patch queries and four actual-policy banks at absolute attempts 512..515 are retained per instance. The latter are 32 sampled row exposures, with replacement, not 32 independent sources.

RPB-v4 uses ordinary context. RPB-v7 adds 0.15 TRAIN context deletion every attempt. RPB-v8 alternates ordinary even attempts and 0.30 deletion odd attempts. All share the learned global architecture, original query/target/scaler rules and exact native 32 decoder input; no PCA follows an encoder.

## Retained quality and cost — no new scoring

These known VALIDATION scores and synchronized training-loop times are copied from the checked-in [v8 durable summary](../../../doc/results/context_balanced_validation_v1.json). They are context, not results of this TRAIN diagnosis. Each run has 128 known VALIDATION examples / 64 source pairs. Three declared head repetitions use Ridge penalty 1 and tanh16 / Adam 0.01 / 100 updates. Size 32 heads have 66 / 562 parameters; raw 576 heads have 1,154 / 9,266. Repetitions and saved checkpoints are not additional encoder runs.

**Intact known VALIDATION, retained**

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 49.69 | 50.83 | 100.00 |
| PCA only — no encoder | 32 | 49.06 | 70.99 | 100.00 |
| RPB-v4 | 32 | 93.59 | 95.99 | 100.00 |
| RPB-v7 | 32 | 97.97 | 98.96 | 100.00 |
| RPB-v8 | 32 | 95.94 | 97.76 | 100.00 |

**Fixed additional 30% known VALIDATION deletion, retained**

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| Raw data — no encoder | 576 | 52.03 | 48.65 | 100.00 |
| PCA only — no encoder | 32 | 49.84 | 54.79 | 100.00 |
| RPB-v4 | 32 | 84.38 | 91.46 | 100.00 |
| RPB-v7 | 32 | 95.16 | 96.25 | 100.00 |
| RPB-v8 | 32 | 91.72 | 94.22 | 100.00 |

Raw 576 contains observed TRAIN-scaled values and observation flags; standalone PCA32 is fitted only on raw TRAIN. The retained deletion view uses the cached mask from its original protocol. This report draws no corruption and performs no quality scoring.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 512 | 0.067593 | 0.071309 | 13.85 |
| RPB-v7 | 512 | 0.081179 | 0.083630 | 15.69 |
| RPB-v8 | 512 | 0.085022 | 0.087620 | 14.79 |

Errors are retained standardized fixed-query MAE, not the Huber quantities below. All timings are retained measurements and exclude the present diagnosis; its CUDA inspection cost is not benchmarked. Both v7 and v8 retain failed reconstruction guards against v4.

## Actual-policy decision

Positive descent cosine means that infinitesimal reconstruction descent helps the current saved correct Ridge margin. Negative means it opposes that margin. The combined direction is base plus weight 1 auxiliary; norms at or below 1e-20 are undefined. All required directions here are defined.

| Master | Auxiliary support / 32 | Base cosine | Combined cosine | Gain | Original-Huber direction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4404 | 22/32 | -0.020841 | -0.008867 | +0.011974 | -0.088818 |
| 5505 | 20/32 | -0.003637 | +0.001116 | +0.004752 | -0.111707 |
| 6606 | 19/32 | -0.004243 | -0.028553 | -0.024310 | -0.119050 |
| 7707 | 21/32 | -0.036588 | -0.056528 | -0.019940 | -0.098586 |
| 8808 | 17/32 | +0.031556 | +0.020611 | -0.010945 | -0.098122 |

All five masters exceed the 16/32 auxiliary-support requirement; four have negative base cosine, and every combined unit direction decreases original Huber to first order. The required improvement on every master fails on 6606, 7707 and 8808. The fixed trigger therefore stops this loss hypothesis. These equal-four-bank means are local plain-gradient geometry, not observed AdamW updates, uncertainty estimates or held-out accuracy.

The auxiliary is Huber of normalized prediction differences minus normalized target differences, divided by sqrt(2), only on intersecting original Q cells. It averages cells, supported pairs and supported examples. It adds no target information and is not phase-invariant lag supervision. Under complete equal C3 support in the quadratic region, weight 1 raises centered residual weighting by 2.5. Two of three pairs include unrelated-phase channel 2; actual per-channel queries have far less overlap than common-patch queries.

The saved head and both TRAIN affine maps remain fixed. Latent perturbations multiply outer_scale × ridge_scale; the real encoder gradient uses a separate graph. A change that rotates the latent basis can reduce this frozen head margin without proving loss of recoverable timing information. The diagnosis establishes local tension against one identified readout, not a causal explanation of generalization.

## Full-TRAIN geometry

Each surface uses its own TRAIN coordinate standardizer. Total trace variance is approximately its dimension count; compare normalized fractions, not raw 96 versus 32 energies. Means below weight the five masters equally. Contrast fraction is mean squared paired half-difference / total variance; coherence is squared mean half-difference / half-difference energy.

| Encoder | Surface / size | Contrast / total % | Coherent / contrast % | Saved Ridge pair margin |
| --- | --- | ---: | ---: | ---: |
| RPB-v4 | contextual / 96 | 26.73 | 1.25 | N/A |
| RPB-v4 | global / 32 | 27.12 | 1.29 | 0.670969 |
| RPB-v7 | contextual / 96 | 27.85 | 1.47 | N/A |
| RPB-v7 | global / 32 | 28.44 | 1.72 | 0.729945 |
| RPB-v8 | contextual / 96 | 24.00 | 1.79 | N/A |
| RPB-v8 | global / 32 | 24.91 | 1.78 | 0.700838 |

The served vector retains substantial standardized pair contrast; these fractions do not identify a universal pooling loss. Low coherent contrast can reflect random signal phase and does not establish information collapse. Contextual 96 is a diagnostic co-trained export, not the decoder bottleneck. Per-master fractions, variance identities and margins remain in the durable summary.

## Decoder reliance and channel allocation

On the four common queries, the receiver metadata, targets, query cells and hierarchical Huber denominators remain fixed. Sibling swaps exchange opposite-label variants of the same source; cross-source donors follow a lexical cyclic derangement preserving label. The TRAIN mean latent is an out-of-distribution counterfactual. No decoder is refitted.

| Encoder | Factual Huber | TRAIN-mean latent | Sibling latent | Other source, same label |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4 | 0.003676 | 0.473713 | 0.275926 | 0.734975 |
| RPB-v7 | 0.005296 | 0.474266 | 0.278881 | 0.738307 |
| RPB-v8 | 0.005785 | 0.474283 | 0.278506 | 0.737942 |

Swaps sharply increase reconstruction error: the decoder uses source-specific latent content. A label-preserving swap still changes phase/noise, so its error does not isolate timing. These tests establish reliance, not insufficient decoder capacity or an adequate global bottleneck for every consumer.

| Encoder | Channel 0 loss share % | Channel 1 loss share % | Channel 2 loss share % |
| --- | ---: | ---: | ---: |
| RPB-v4 | 33.94 | 32.82 | 33.24 |
| RPB-v7 | 30.93 | 31.69 | 37.38 |
| RPB-v8 | 30.13 | 34.68 | 35.19 |

Shares are realized original-Huber contributions, normalized per instance across the four common queries, then averaged over masters. The objective still weights eligible channels equally. Channel 2 receives substantial error allocation, but this alone does not prove that nuisance reconstruction caused the quality tradeoff.

## Verification and limits

The [independent v3 audit](../../../output/runs/rpb-training-objective-diagnostic/audit-tools/run-0vc70T-v3/validation.json) passed 16,119,003 checks: 195 protected TRAIN roles, 15 instances, 45 saved native fits, 45 TRAIN prediction archives, 30 geometry surfaces, 120 banks and 240 decoder-intervention predictions. It decoded 315 permitted CPU archives. All original bytes and 39 compiled source snapshots were preserved. The [measured capsule](../../../output/runs/rpb-training-objective-diagnostic/training-objective-0vc70T/) retains exact vectors, donors, queries, per-channel reductions and all parameter-group comparisons.

The failed v2 reader and its failure record remain preserved. It incorrectly required CUDA full-observation geometry equal to cached CPU exports and reused CPU-feature margins for CUDA geometry. V3 instead replays geometry/margins from saved CUDA values and reports all cross-device differences descriptively; maximum is 1.1920928955078125e-6, with no new cross-device acceptance threshold. Exact same-CUDA query checks, CPU head equations and the original 1e-6 graph-versus-frozen CUDA bound are unchanged.

The CPU reader does not rerun encoders, decoder functions, CUDA autodiff, classifier fitting or AdamW, and does not decode GPU checkpoint parameter/optimizer storage. Exact CPU checkpoint-to-cached-export equality and actual CUDA gradients/state preservation are producer/admission-source-bound. Original std::shuffle queries are checked structurally and paired across versions, with generation source-bound; row sampling and context erasure/repair are independently replayed. No confidence intervals across masters are produced.

| Identity | SHA256 |
| --- | --- |
| Compiled source | `c9dbd427d998dd35c98522554000832e9ee1d30ba8ff1014ef46a089487c8412` |
| Passed CUDA admission | `d7df02c49f2ac41a3d74d910d83b6a87c9468ea0510aa05f2e865ff619b58968` |
| Frozen card | `37212638b08afc54ce4a0daf803cf3c512c697e21c1631f4a5a49acf7505cf18` |
| Capsule inventory | `e51a6ce1f2056f9f8742dc583e3440e122e53f3c94687976e8444f3b20e12947` |
| Independent reader | `ab5433bb97ec9ee80150d4518fecdd6807b3024d68c4932c61af3906728901c0` |
| Passed validation | `29325651ed56630cd984b2853ad9355178bc5378cfd4c94cb5554816c1e2cd07` |

The [durable measured summary](../../../doc/results/training_objective_diagnostic_v1.json) retains exact per-master diagnostics, normalized fractions, original producer identities, retained-quality provenance and reader corrections. The [source-only audit replay bundle](../../../output/runs/rpb-training-objective-diagnostic/audit-tools/replay-bundle-0vc70T-20261007T062810Z-487b37d6/README.md) preserves the passed audit, source dependencies and frozen compiled sources; it contains no model/data payloads.

Stop the all-pair auxiliary hypothesis under this card. A native-view agreement mechanism can be planned separately to retain ordinary reconstruction while studying deletion robustness. It is unimplemented and unmeasured here; this report creates no new model tag or promotion.
