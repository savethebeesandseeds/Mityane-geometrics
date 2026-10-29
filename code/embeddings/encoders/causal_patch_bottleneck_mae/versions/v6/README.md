# CPB-v6: Huber mean and detached-error variance supervision

## Channel axis

cuwacunu_torch `interval_type_e` is the standard. One window is one instrument. Each channel is one time resolution of that instrument: minutes, hours, weeks, and the other kline intervals. The channel id is that interval's torch ordinal. Different instruments are different rows. A stored id set `{0,1,2}` is the withdrawn instrument binding from historical checkpoints. It is not this axis, and scores measured on it stay scores of that binding. The contract is `code/embeddings/encoders/include/embedding/encoders/timeframe_channel.h`.

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
Inherited notices remain in all complete copied files and LICENSES/NOTICE.

Implemented source, pending independent preflight, frozen capsule admission,
build and runtime proof. No passing execution or measured quality is claimed.
The prospective protocol is .archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V6_CARD.json. CPB-v3 remains the
selected clean baseline/application; V4/V5 scoped strengths remain protected.

This version owns a full physical copy of the V5 architecture/parser closure,
under its own namespace, include paths, config and typed checkpoint kinds.
The sibling versions/v6.h is a complete byte-identical public-header copy,
not a forwarding shim. The provenance receipt binds actual direct V5 source
bytes and owned destination bytes. Source copying imports no weights or fitted
statistics. Torch and the standard library are the only external code closure.

The native40 architecture, parameter registration and constructor draw order
are identical to V5: parent32 [0:32], mean4 [32:36], uncertainty4 [36:40].
Independent learned [4,270] projections read one shared frozen raw-relative bank.
The mean residual reads mean4; the positive variance head [27,4]+27 reads
uncertainty4. The shared raw bank is fixed computation, not shared learned
upstream parameters. There are76 learned tensors/294752 values and one fixed
analytic prior/3888 values:77 tensors/298640 total. No capacity is added.

The sole numerical training change is the explicit two-component objective:

    mean_loss = R[Huber_delta1(mean - target)]
    variance_loss = R[0.5 * (stopgrad(mean-target)^2 / variance + log(variance))]
    total_loss = mean_loss + variance_loss

R is the unchanged supported-cell, eligible-channel, eligible-example mean.
Unsupported operands are sanitized before independently promoting mean/target
to F64 and subtracting. Both coefficients and delta are fixed at1. Mean Huber
has no inverse-variance weight and no direct uncertainty gradient; variance NLL
has no direct parent32/mean4 gradient. The variance projection and positive link
remain live. The ordinary numeric Gaussian NLL is reported from detached
distribution tensors; it is never the backward tensor. Total and NLL may be
negative; Huber is nonnegative. No numeric-value gradient surrogate is used.

The trace exposes objective.mean_huber, objective.variance_nll and
objective.total_loss. output.loss equals that total. nll retains the ordinary
Gaussian value and eligibility trace as a detached diagnostic.

Training retains one AdamW group (.001 learning rate/.0001 decay) and one
global norm clip1. Thus variance gradients can alter the common clipping
factor and scale mean updates. Direct graph separation is not complete
optimizer independence; no separate optimizer or branch clip was introduced.

fresh(Config,FrozenScaler,BankStandardizer,seed) constructs all learned values
directly under the declared seed with source_updates0 and no pretrained
ancestor. Newly fitted TRAIN-only statistics bind their actual identities.
No old checkpoint/scaler/normalizer importer exists. Actual CPU rawF64 values,
boolean masks, timeframe channel IDs, timestamps and explicit cutoffs retain
the crop-before-validation boundary, causal attention and backward-only banks.

Own inference/payload/training kinds bind objective_policy, deltaF64, unit
coefficient vector, detached-error flag, direct routing and reporting-backward
flag. Missing/altered fields and old version kinds are rejected. Typed resume
also binds dataset/split/recipe, optimizer moment contents/order/presence/steps,
counters and deterministic streams. Saves never overwrite artifacts.

The active research Makefile compiles its physically owned engineering and
objective fixtures. Family fixtures are independent full copies of those files.
The nonvacuous CUDA objective fixture covers heterogeneous variance, both Huber
branches, unequal support, semantic reorder, unsupported NaNs, direct gradient
routing, manual F64 component/native40 parity, actual head/projection learning
and a shared-clipping coupling witness. All76 learned tensors are checked by
the separate whole-model engineering/main trajectory. Both actual saved initial
and final checkpoints require complete frozen causal proofs before quality.
