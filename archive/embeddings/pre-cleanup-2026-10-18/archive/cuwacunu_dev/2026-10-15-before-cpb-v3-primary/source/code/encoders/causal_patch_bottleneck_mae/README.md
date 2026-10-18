Current qualification (15 October 2026): CPB-v1 and CPB-v2 inherit noncausally trained weights. V1 remains a historical reference/application selection; neither supplies oracle targets for clean causal training. See [the correction](../../../doc/CAUSAL_TRAINING_PROVENANCE_CORRECTION_2026-10-15.md). Fresh causal research starts with CPB-v3 in the research repo.

# Causal patch bottleneck MAE

Copyright (c) 2026 Waajacu. All rights reserved. Preserve the inherited notices
inside the copied implementation.

CPB-v1 is a separately owned causal encoder derived from the saved V30 design.
Its first instance, CPB-v1.market-01, imports the pinned V30 weights and frozen
input/bank normalization as data, with zero new encoder training updates or
input/bank normalization fits. CUDA causal proof and fixed-head scoring are
complete. CPB passes the original market 9/9 and strict 8/8 mean-score floors,
but only 2/3 neural trend fits are positive and only 3/8 TEMPO-5 retention floors
pass. These remain measured quality limitations; the application selection below
does not declare an overall research winner.

## Current application selection — 15 October 2026

The user selected **CPB-v1.market-01** as the sole active application encoder in
cuwacunu_dev. Its [integration record](../../../../cuwacunu_dev/doc/CAUSAL_PRIMARY_INTEGRATION.md)
carries current runtime verification status. Application config/encoder.conf
fixes the evaluated geometry; separate config/encoder_runtime.conf selects
the local ignored typed checkpoint and exact SHA256. The application facade
uses frozen inference with raw CPU F64 values, actual timestamps, semantic IDs,
masks and explicit cutoffs. It does not fit scalers or start causal training.

Previous RPB implementations/drivers have left the active dev source tree.
The exact former dev paths/bytes are in the sibling
[archive](../../../../cuwacunu_embedding/archive/cuwacunu_dev/2026-10-15-before-causal-primary/README.md).
Historical docs/results/checkpoints and portable research implementations remain
preserved. This operational choice supersedes the old dev RPB-v22-primary
selection, while retaining balanced/task research leaders and stopped research
state. No new quality measurement, commit or push follows.

## Physical version entry

The complete v1 implementation lives in [`versions/v1/`](versions/v1/).
That directory owns its `include/`, `config/`, `tests/`, `v1.mk` and provenance.
The sole public C++ entry is
[`versions/v1/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h`](versions/v1/include/embedding/encoders/causal_patch_bottleneck_mae/versions/v1.h),
in namespace `embedding::encoders::causal_patch_bottleneck_mae::v1`.
Its settings are in [`versions/v1/config/v1.conf`](versions/v1/config/v1.conf).

This is a complete owned copy, including its reachable private headers and
parser utilities. It does not forward encoder execution into the historical
raw family, integration research modules, another causal version or a sibling
checkout. Torch and the C++ standard library are external dependencies. Future
versions must own their own complete implementation copies; there is no mutable
family-level encoder implementation shared between evaluated versions.

## Cutoff contract

The public `Encoder::encode_at` and `trace_at` paths take raw CPU float64 history,
boolean observation masks, semantic channel IDs, a timestamp grid and explicit
cutoffs. They discard later values **and masks** before observed-finiteness
checks, rounding, normalization, token packing, support counts or bank statistics.
Each output `z_t` uses the last at most H observations ending at its own cutoff.
Unavailable earlier slots are left-padded with hidden zeros. Temporal attention
uses original patch endpoints and blocks later keys while retaining padding
masks; channel mixing stays at the same patch time.

Every returned timestamp must use its own cutoff path. A final global vector
must not be broadcast to earlier times. The public API has no unguarded
full-history `encode` bypass. Old decoder helpers are not advertised as an
autoregressive forecasting API. The mandatory CUDA proof checks suffix values
and masks, future poison, short and missing prefixes, semantic permutations,
internal temporal states and legal past positive controls before quality work.

## Version metadata and latest instance

The [version index](versions/index.json) and
[prospective CPB-v1 manifest](versions/CPB-v1.json) are admitted declaration
snapshots. Their pending wording records the evidence stage when they were
frozen; they are preserved rather than rewritten as a live status page.

The [verified instance record](versions/instances/CPB-v1.market-01.json) binds
actual source/configuration, the usable typed checkpoint, both engineering
attempts, proof/admission/quality evidence, report and full-precision task
results. The [portable guide](../../../doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1.md)
contains measured task tables and reuse instructions. The frozen [result report](../../../../cuwacunu_dev/doc/CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md)
lives in the integration repository; the source mirror does not copy ignored
checkpoints or data. These additive records carry later status outside the
admitted physical implementation and prospective manifest.

The passing CUDA proof covers all 128 cutoff indices, 2,350 suffix cases,
30 internal temporal cases and 14 typed negative cases. All 76 imported
parameter tensors (297,560 values), frozen statistics and saved/reloaded state
match. The proof is bounded correctness evidence. Legal past positive controls
change outputs/support; future values and masks do not.

MKT-HIST-2 complexity4/5 retains 112 TRAIN/32 known chronological VALIDATION
windows and four separate past-window targets. CPB establishes new matched
trained-final comparison highs in all four linear tasks and neural volatility
and activity. V25 retains the strongest neural trend and relation means.
The neural trend repetitions are -0.33939984380671695,
0.657262428919963 and 0.4132006631059153, so the full market gate fails despite
passing every mean-score floor. TEMPO-5 complexity5/5 uses all three saved
cohorts, both scaler policies and intact/30%-erased views. Coverage is 100%
throughout; five of eight frozen transfer floors fail. Strong absolute scores
do not satisfy the full retention rule.

The usable local model is only
`output/engineering/causal-patch-bottleneck-mae-v1/attempt-002/proof/inference.pt`
in integration: SHA256
`54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895`,
1,290,592 bytes. `proof/refusal-1.pt` through `proof/refusal-10.pt` are deliberately
malformed typed-loader test inputs, preserved as engineering evidence and
unusable as models. Attempt001's failed attention-mask proof, its exact source
and all three binaries remain preserved; attempt002's reviewed repair passed
before the single quality run. There were no new encoder updates, input/bank
fits, decoder calls, old encoder replays or head tuning. New fixed heads were
fitted once on TRAIN. Historical RPB targets retain their original window
semantics and are not relabeled causal or forecasting results.

CPB-v1.market-01 is the selected dev application encoder; native36 remains the
balanced research reference and all scoped leaders remain preserved.
Preserve all historical source, checkpoints, cards and results, including the
interrupted, unscored V30 synthetic extraction. This work is local and
unpublished. The later user-requested application selection above starts no
new research, training, commit or push.
