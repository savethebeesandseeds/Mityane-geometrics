# CPB-v6 fixed saved-feature evaluation

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

This directory owns full physical copies of frozen V5 readout/CACHE88
dependencies and exporter/readers. Fixed arithmetic is unchanged. V3/V4/V5
saved reports and feature/statistic references are read only; their models and
heads are never executed. Each V6 initial0/final2048 native40 bank and fixed-head
evaluation is new, even when all initial tensors pair exactly with saved V5.

Exports use genuine rawF64, semantic IDs, real timestamps and explicit causal
cutoffs, CUDA-F32 encoder execution and frozen parameters/gradients/statistics.
The market112TRAIN/32VALIDATION windows and three saved TEMPO cohorts retain
512TRAIN/256VALIDATION, intact/deleted views and both scaler policies. Three
fresh synthetic TRAIN scalers fit once; no validation statistics are fitted.

Readouts keep TRAIN-only normalization, ridge1 and16-tanh Adam.01/100steps.
D40 seeds6049515622977958931,8262941649267994858,2095371932330426401 remain fixed.
Market four-output heads164/724 and timing two-output heads82/690 are unchanged.
There is no head tuning, post-encoder PCA, initial-score reuse or old model replay.

The reporter admits explicit Huber-mean/detached-error-variance supervision,
typed objective/delta/unit-weights/routing, all76 learned tensors, actual
resume512, six binaries, nonvacuous CUDA mechanism proof and both saved-state
causality proofs. Training component/total traces remain separate from ordinary
evaluation Gaussian NLL and standardized mean MAE. Shared clip1 coupling is
disclosed; direct loss-path separation does not imply optimizer independence.

The comparator rehashes all three clean saved final controls and the card-bound
stronger retention protocol, then evaluates its22 conditions verbatim. It keeps
trust/coverage separate, records every failure and scoped gain, and preserves
V3 as selected clean baseline/application. No automatic winner or promotion.
