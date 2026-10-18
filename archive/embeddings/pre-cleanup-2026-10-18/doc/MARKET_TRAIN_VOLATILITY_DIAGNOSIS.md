<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# Saved-TRAIN volatility diagnosis

Completed 14 October 2026. **New saved-TRAIN arithmetic; no new encoder quality
experiment.** All execution occurred in the existing GPU-enabled integration
container. CPU calculations loaded22 hash-pinned TRAIN archives: the shared
target bank, parent/native36 initialized/native36 trained feature banks, and
their existing three-repetition fit/prediction archives. No encoder calls,
updates, new head/scaler fits, held-out tensor diagnostics or PCA occurred.

Read the [prospective card](../../cuwacunu_dev/doc/MARKET_TRAIN_VOLATILITY_DIAGNOSIS_CARD.md),
[full results](../../cuwacunu_dev/doc/MARKET_TRAIN_VOLATILITY_DIAGNOSIS_RESULTS_READABLE.md)
and [completion receipt](../../cuwacunu_dev/doc/results/market_train_volatility_diagnosis_v1/completion.json).
The report contains the four separate standard TRAIN task tables, physical
errors, every coordinate's correlations, saved coefficients, ridge objectives,
prediction decomposition and fixed early/late context. MKT-HIST-2 is BTC/ETH/SOL
January2024 minute history, recent128 past-only properties, designed complexity4/5.
All112 TRAIN rows are from one chronological source period; both halves share
the same existing fitted head. The original CUDA encoder batch/budget and fixed
head recipes remain unchanged.

The parent32 coordinates and normalizers are exact. Appending zero coefficients
to the old linear head reproduces its predictions exactly. Nevertheless, joint
ridge fitting on native36 shifts volatility prefix weights by L2 0.602036,
against a parent coefficient norm1.275868. The saved ridge redistribution
identity passes; TRAIN volatility linear R² improves0.767220→0.832289 and its
objective falls27.699222→20.181915. This is a changed fitted predictor, with
the original information still available.

Strong pairwise prefix/change duplication is not observed: maximum|r|0.320435,
mean|r|0.093580, with0/128 pairs at least0.9. This does not rule out multivariate
dependencies. Full volatility prefix/change contributions correlate+0.276161,
with cancellation ratio0.835096, so they reinforce overall. The previously
suggested large-cancellation explanation is unsupported.

The benefit varies within TRAIN. Linear volatility R² improves0.654690→0.772730
on the declared first56 rows but falls0.932627→0.910444 on the last56. During
branch learning, TRAIN trend improves0.531002 initialized→0.808109 trained while
volatility decreases0.857450→0.832289. These quantities motivate retaining
change-energy information; they do not prove why the published validation
predictor generalizes worse.

Independent standard-library arithmetic passed15,428 checks, with maximum
normalized difference2.53e−15. Source-audited fixed paths and the actual loader
ledger match the22-input allowlist; no OS isolation/tracing claim is made.
Historical checks preserve1,249 files, including four original handoff and
five native36 checkpoint files. All raw native arrays, numerical witnesses,
tensor archives and binaries remain local and ignored; nonbinary diagnostic
receipts and source snapshots are saved in integration. No portable encoder
arithmetic changed or second standalone run occurred.

The [next-direction proposal](../../cuwacunu_dev/doc/MARKET_CHANGE_ENERGY_NEXT_DIRECTION.md)
keeps parent32, native36,2048/B8 and fixed heads, and proposes a training-only
decoder that reconstructs the existing signed-change/RMS moment bank from the
same four change coordinates alongside waveform reconstruction. It uses
observed input structure rather than market task labels. Its complete objective,
scaling/coefficient, new instance/card and typed state still need declaration
and implementation before execution. It is not a tested remedy.

Future acceptance must fix linear volatility against the unchanged0.6781069298
floor and retain the native36 market/compatible TEMPO-5 gains under strict
prospective no-decline gates. Preserve all original cards/results/checkpoints;
RPB-v22 remains primary and native36 remains the leading research candidate,
with its original8/9 disposition. New diagnosis work is local, uncommitted and
unpublished at this milestone; the earlier commit/push authorization covered
only the preceding completed payload.
