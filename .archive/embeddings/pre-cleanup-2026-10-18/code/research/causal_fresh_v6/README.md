# CPB-v6 owned fresh causal research

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
Inherited notices remain in copied dependencies and LICENSES/NOTICE.

Source implemented, pending independent admission, build and execution.
doc/CAUSAL_FRESH_V6_CARD.json owns the prospective recipe and hash bindings.
This directory owns complete physical copies of the frozen V5 research closure;
the two provenance receipts identify actual direct V5 parents.

Native40, constructor draws, all294752 learned values/76 tensors,3888 literal
prior values, mean/uncertainty routing and positive link remain identical to V5.
The sole objective change is unweighted Huber delta1 mean reconstruction plus
unit-weight Gaussian variance NLL using detached current residual. Both use
sanitized F64 operands and the same eligible-example/channel/cell reduction.
Only objective.total_loss is backpropagated. Ordinary Gaussian NLL and
standardized mean MAE remain separate evaluation metrics. Total/NLL can be
negative; Huber is nonnegative.

One fresh CUDA2048-update/B8 trajectory uses construction seed943001 and paired
row/mask/context seeds943002/3/4; fresh4096TRAIN-window input/bank calibration,
AdamW.001/.0001 and shared global clip1 remain fixed. Actual typed resume512
continues the same trajectory. Joint clipping still couples branch scaling;
there is no complete optimizer-independence claim or separate clipping rule.

All2048 CSV rows record mean_Huber_F64, variance_NLL_F64, total_objective_F64,
reporting Gaussian_NLL_F64, pre-clip norm and common clip factor. Summary retains
first/last values. Two component witnesses bind actual pre-step minibatches at
updates1/2048, with support/eligibility/distribution/target/per-row/scalar
components and no extra model forward. Initial/final fixed-query witnesses and
all112TRAIN/32VALIDATION scalar rows retain MAE/NLL and add component traces.

The active Makefile builds six binaries/10 translation units, using its own
tests/v6_test.cpp and v6_separation_test.cpp. Engineering verifies all128
cutoffs/2350 suffix cases/30 internal interventions, actual CUDA learning of
all76 tensors, complete typed objective metadata refusals and exact resume.
The dedicated artificial objective fixture proves direct separation, F64
component/manual native40 parity, outlier/heteroscedastic/NaN handling and
joint clipping coupling. Both actual initial0/final2048 artifacts receive
frozen saved-state proofs before quality.

Quality separately evaluates this candidate's own initial/final native40 banks
and unchanged fixed heads. D40 repetition seeds remain6049515622977958931,
8262941649267994858,2095371932330426401. Market heads have164/724 parameters;
TEMPO-5 heads82/690. No post-encoder PCA or head tuning.

All40 initial coordinates and all13 feature/statistic archive pairings must
match saved V5 numerically: two market banks, six TEMPO banks, market scaler/
bank statistics and three synthetic TRAIN scalers. They are inert pairing
references, not initial-score reuse; no old encoder/head is executed.

The card-bound retention protocol protects the strongest saved V3/V4/V5 final
scope scores under its22 unchanged declared conditions. Reports reuse saved
clean controls, preserve all failed conditions and distinguish trust/coverage
from advancement. CPB-v3 remains the selected clean baseline/application;
all earlier candidates, results and checkpoints remain intact. No automatic
promotion, publication, rescue or forecasting claim follows.
