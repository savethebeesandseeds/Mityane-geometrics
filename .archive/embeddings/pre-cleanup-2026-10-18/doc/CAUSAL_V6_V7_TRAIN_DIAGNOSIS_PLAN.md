<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Prospective saved TRAIN V6/V7 diagnosis plan

This is preparation for a saved-data diagnosis, not a candidate or a numerical
result. V3 remains the clean baseline and sole dev encoder. V6/V7 sources,
trajectories, heads, checkpoints, failed preparations and all earlier strengths
remain preserved. No next encoder, coefficient, clipping cap, seed, rate, head,
budget or checkpoint rescue is selected here.

Dataset: **MKT-HIST-2**, past-window market retention, **complexity4/5**. These
ordinal task labels are not inferred from performance. Numeric access is TRAIN
only through1705881599999:112 saved natural-mask probes, first8 detailed fixed
queries and two actual pre-step TRAIN minibatches per version. Metadata pins of
broader reports do not admit held-out numeric selectors.

Both models have the same fresh native40 geometry and constructor, with20
learned content,12 fixed spectral,4 mean and4 independent uncertainty coordinates.
They apply the same unit-weight Huber(delta1) mean objective plus Gaussian NLL
variance calibration on detached error. Reporting Gaussian NLL never supervises
the mean. V6 clips one global norm2 group at1. V7 clips73 mean tensors/293537
values and3 uncertainty tensors/1215 values separately at1 before one AdamW;
fixed prior1/3888 is excluded. The V7 combined clipped norm can reach sqrt2.
Future residual targets, data/statistics and optimizer histories retain coupling.

The bounded diagnosis has three panels:

1. Read both complete2048-row TRAIN CSV traces using only update, four objective
   scalars and their actual observed clipping fields. V6 has one global norm/
   factor; V7 has separate mean and uncertainty norms/factors. Check the recorded
   formulas and report overall/four fixed512-update blocks. Recompute supported
   objective components, masks, timestamps and CSV associations from saved actual
   update1/2048 witnesses for both models. The first fresh forward must match;
   later trajectories differ. Do not infer V6 branch norms or gradient dominance
   from V7, construct counterfactual global clipping, or replay an optimizer.
2. Use all112 saved TRAIN native40 rows and all three already-fitted readout
   repetitions. Describe block variance/participation, four-task saved prediction
   errors and exact existing-ridge block contributions/covariances, including the
   fixed early56/late56 halves. Neural predictions are data; neural weights or
   forwards, new fits, block-ablation predictors and task-supervised encoder
   construction are forbidden.
3. Use first8 fixed TRAIN query witnesses at initial0/final2048 only. Describe
   support, MAE/Huber/variance/reporting NLL and saved-decoder local adjoints. Four
   prespecified32-step bins[0,32),[32,64),[64,96),[96,128) describe raw observed
   F64 means/std/within-bin absolute first differences and supported reconstruction
   by channel/feature. Raw summaries receive no task labels or fitted transform.
   Bin-specific support differs; their hierarchical means cannot be added as loss.

Only query initial0/final2048 receives matching saved decoder coefficients.
Actual witness2048 is a forward from state2047, so it must never be paired with
the final2048 decoder. Local adjoints omit parent32/upstream Jacobians and full
parameter gradients. Mean-NLL/MAE alternatives are unapplied descriptive adjoints
within each same mean4 basis. No cross-basis/model/stage cosine or latent-norm
dominance claim is made. F64 scalar algebra promotes saved mean and target
independently before subtraction; link derivatives approximate F32 intermediates.

The prospective generator is
[prepare-card.ps1](causal_v6_v7_train_diagnosis/prepare-card.ps1). It has no project
arithmetic and is not invoked during preparation. After root review, final V7
metadata/preservation and the independently reviewed V7-retention protocol exist,
it can hash and declare49 exact inputs, their typed field allowlists,31
preservation-only pins, the broad final preservation receipt and all five owned
source files. Existing Complete V6/V7 receipts provide canonical main paths;
host-only context metadata is copied into a declared future source leaf. It
creates new machine/Markdown cards exclusively, refusing any existing card.

The49 inputs are one shared TRAIN target archive plus20 common saved roles and
four trajectory roles per version. Both versions supply full112 saved component
rows. Four mixed-split query JSON files are decoded but only their train subtree
is selected. V6 CSV access is restricted to seven columns and V7 to nine. Inert
checkpoint reads include typed routing/clipping metadata and only the saved
linear change decoder/variance head/link constants. No other model or optimizer
weights are admitted. All source/card/allowed-input/preservation hashes are
checked before and after analysis. The broad final V7 preservation receipt must
verify before the first tensor storage value is read.

An independent source/card preflight and explicit root execution admission are
required even after card generation. Schema inspection and the single numeric
pass run only inside the existing managed cuwacunu container, with exclusive
source/output leaves and preserved refusal evidence. There is no Torch/NumPy,
model/encoder/decoder/neural forward, new head/scaler/statistic fit, training,
raw-cache access, PCA, held-out tensor diagnosis, dev mutation or publication.
An independent saved-evidence audit precedes interpretation.

The actual post-V7 ledger, separate V7 establishment and final preservation are
bound before execution. The future strongest V3-V7 retention protocol is context
only: this diagnosis neither derives floors nor retests a candidate against them.
Historical protocols/counts stay unchanged; no TRAIN NLL gate is added. First8
TRAIN temporal associations cannot establish broad representation quality,
forecasting or held-out improvement. The diagnosis informs a later architectural
decision without declaring one.
