# CPB-v7 owned causal research

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
Inherited dependency notices and LICENSES/NOTICE remain intact.

Complete physical copies of the frozen V6 research closure, with separate V7
paths, namespaces and typed artifacts. No execution has occurred during source
preparation. doc/CAUSAL_FRESH_V7_CARD.json declares the prospective experiment.

V7 preserves V6 forward arithmetic, native40 (parent32+mean4+uncertainty4),
constructor draw order/seed943001,294752 fresh learned values and3888 fixed
prior values. Huber(delta1) mean supervision plus detached-error Gaussian
variance calibration keep coefficients1 each. Only objective.total_loss is
backpropagated; Gaussian NLL and standardized MAE remain unchanged diagnostics.

The sole mechanism changes one joint norm2 clip1 into separate norm2 clip1
operations on disjoint parameter groups, followed by the same single AdamW
step. Mean has73 tensors/293537 values; uncertainty has exactly three tensors
(core.uncertainty_projection, core.variance_projection, core.variance_bias),
1215 values; the fixed prior1/3888 is excluded. Combined clipped norm may
approach sqrt2. Shared optimizer/history and future residual-dependent variance
targets remain: no complete optimizer-independence claim.

One fresh CUDA trajectory uses2048 successful updates/B8, unchanged row/mask/
context streams and fresh4096TRAIN-window statistics. The portable clipping
API validates exact names/coverage at every step. gradient-clipping-partition.json
saves canonical typed membership. Every CSV step records both preclip norms,
both factors, and defined/nonzero gradient counts by branch. Update1/2048
component witnesses come from the actual pre-step forward with no replay;
they retain distribution/support/objective rows plus clipping metadata.
Typed resume512 continues the actual trajectory with exact moments/order.

Six binaries/10 translation units use only the owned V7 implementation.
Engineering checks initialized/trained128 cutoffs,2350 suffix cases,30 internal
interventions, CUDA learning and objective/clipping typed-negative cases.
The artificial CUDA fixture independently checks full named-group scaling,
manual clip parity, both directions of clipping isolation, both caps and
combined sqrt2, while retaining objective/native-gradient parity and one actual
AdamW step. Saved initial0/final2048 receive complete frozen causal proofs.

Both saved stages export once and fit their own unchanged D40 heads. All13
initial DATA/statistic archives are independently compared numerically with
saved V6 references; no old model or head execution. Repetitions2701/2802/2903,
D40 seeds6049515622977958931/8262941649267994858/2095371932330426401,
market heads164/724 and TEMPO-5 heads82/690 remain unchanged. No PCA or tuning.
Four V3/V4/V5/V6 final controls and the reviewed096584 V6-retention protocol
protect22 prospective conditions. No TRAIN NLL gate. Producer runtime admission
remains distinct from pending initial pairing and independent audits; a later
separate trust join must establish complete trust before advancement.

Build/run only in the existing managed GPU cuwacunu container after root
admission of the frozen card/source capsule. No DEV changes or publication.
