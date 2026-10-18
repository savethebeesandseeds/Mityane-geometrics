# Saved native TRAIN reliability diagnosis

Measured 9 October 2026 (Asia/Dubai) under
[saved-native-reliability-v1](../../evaluation/cards/saved_native_reliability_v1.md).
This is a new descriptive analysis of reused TRAIN evidence, with zero encoder
or decoder updates, zero head refits and zero held-out analysis roles.
The separate independent arithmetic audit passed.

Task: timing/lag sign. Each of five masters has 256 TRAIN rows from 128 paired
sources, 640 independent source groups across masters. There are five trained
instances per method; the three saved head repetitions 2701/2802/2903 do not
create more encoder runs. Historical encoder budget 512, batch 8, CUDA RTX A2000;
this analysis uses CPU saved arithmetic inside the managed container.
Heads remain Ridge penalty 1 and tanh16/Adam0.01/100 updates. Native32 has no PCA.

RPB-v4.alt-01 is the fresh learned-global control group. RPB-v7.alt-01 is the
same architecture with lighter TRAIN context deletion. Original RPB-v7 remains
frozen and is not an analysis input.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4.alt-01 | 32 | 94.92188 | 99.89583 | 100 |
| RPB-v7.alt-01 | 32 | 94.45313 | 98.35938 | 100 |

Equal-master means; neural scores average all three retained heads per master.
Raw/PCA/mask controls are outside this deliberately closed TRAIN diagnosis.
These fitted TRAIN scores are not new validation accuracy.

| Encoder | Updates | Train error MAE ↓ | Validation error MAE ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4.alt-01 | 0 new | — | — | 0 |
| RPB-v7.alt-01 | 0 new | — | — | 0 |

No fixed-query reconstruction archive is admitted, so reconstruction MAE is not
remeasured. Saved arithmetic took 4.99951 seconds including its file I/O and
post-analysis input verification. Independent archive arithmetic took21.91721
seconds; source freeze, initial hash admission and final inventory are separate.
No CPU encoder/decoder construction, forward, optimizer or fitting is executed.

## All instances

| Master | RPB-v4.alt-01 linear % | RPB-v7.alt-01 linear % | RPB-v4.alt-01 neural % | RPB-v7.alt-01 neural % |
| --- | ---: | ---: | ---: | ---: |
| 9109 | 92.57813 | 93.35938 | 99.47917 | 98.82813 |
| 10210 | 99.60938 | 95.70313 | 100 | 100 |
| 11311 | 92.96875 | 98.04688 | 100 | 100 |
| 12412 | 94.14063 | 85.15625 | 100 | 92.96875 |
| 13513 | 95.31250 | 100 | 100 | 100 |

All instances have 100% coverage, 32 finite varying coordinates and zero
constant coordinates under the declared population-SD threshold 1e-12.
Linear TRAIN scores range 92.57813–99.60938% for v4 and 85.15625–100% for v7.

## Geometry and optimization traces

Pair distance is the L2 distance between the two timing variants of one source,
using the existing outer TRAIN-normalized features. It depends on representation
coordinates and scaling. Correct pair order means the saved Ridge score is
strictly larger for label 1 than label 0 within a source; it is distinct from
row accuracy. Median linear margin uses each saved row's true-class logit gap.

| Method | Master | Mean normalized pair distance | Correct pair order % | Median linear margin |
| --- | ---: | ---: | ---: | ---: |
| RPB-v4.alt-01 | 9109 | 4.78542 | 100 | 0.69559 |
| RPB-v7.alt-01 | 9109 | 5.00951 | 100 | 0.69776 |
| RPB-v4.alt-01 | 10210 | 6.35984 | 100 | 0.68031 |
| RPB-v7.alt-01 | 10210 | 5.76253 | 100 | 0.73088 |
| RPB-v4.alt-01 | 11311 | 6.01117 | 96.09375 | 0.59254 |
| RPB-v7.alt-01 | 11311 | 4.92998 | 100 | 0.57513 |
| RPB-v4.alt-01 | 12412 | 5.75155 | 100 | 0.56467 |
| RPB-v7.alt-01 | 12412 | 4.48032 | 94.53125 | 0.42415 |
| RPB-v4.alt-01 | 13513 | 6.15626 | 97.65625 | 0.53869 |
| RPB-v7.alt-01 | 13513 | 6.37898 | 100 | 0.81008 |

No Ridge pair ties occur. Full served/outer/ridge geometry, all row margins,
all source-paired differences and all four fixed trace blocks are retained in
the [machine summary](../../../doc/results/saved_native_reliability_v1.json)
and bound tensor capsule.

The following losses describe sampled optimization queries over updates 385–512;
they are Huber losses, not fixed-query MAE. V7's training context is harder, so
these losses alone cannot identify the cause of its classifier differences.

| Master | RPB-v4.alt-01 sampled Huber | RPB-v7.alt-01 sampled Huber | RPB-v4.alt-01 gradient norm | RPB-v7.alt-01 gradient norm |
| --- | ---: | ---: | ---: | ---: |
| 9109 | 0.004097 | 0.005733 | 0.12494 | 0.14325 |
| 10210 | 0.004160 | 0.006885 | 0.12356 | 0.15861 |
| 11311 | 0.003728 | 0.006537 | 0.11668 | 0.15694 |
| 12412 | 0.004136 | 0.006052 | 0.12305 | 0.14622 |
| 13513 | 0.004479 | 0.007138 | 0.12125 | 0.15467 |

## Decision

Master 12412's weakness is already visible in both fitted TRAIN heads. Its three
v7 neural scores are 91.40625/92.96875/94.53125%, against 100% for each v4 head.
The other masters show mixed classifier, margin and distance differences.
Master 11311 improves linear accuracy despite a smaller normalized pair distance;
master 13513 has the highest final v7 sampled loss and perfect TRAIN scores.
The diagnosis does not identify a consistent information-path defect, collapse,
rank loss, memorization or causal pooling failure.

Proceed with one separately frozen amplitude-transfer check on new numeric RNG
masters and source IDs, using frozen saved encoders and their original timing
scalers. Fit only the fixed new-task TRAIN readouts; run required encoder
inference on CUDA. Keep raw/PCA32/mask/untrained controls and all five pairs.
This checks bounded breadth before selecting an architecture change. Do not
tune deletion rates, head recipes or the failed v9 objective from this result.

## Integrity and preserved failures

The closed 85 TRAIN roles bind parent
`fresh-decoder-replication-4p9U4b`; its inventory SHA256 is
`a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90`.
Completed capsule: `output/runs/rpb-saved-native-reliability/saved-native-reliability-vCYkSj`.
Source fingerprint: `c28dc727bb2b4ac8c0ac9a106a7a744173ed7eed5d607cc48fb720495675c74c`.
Inventory SHA256: `fcac1b874c5b2925b8e65e749d5a467a45bfe515bbae583c6e001efb2b13fe34`.
Card SHA256: `f93898f6dd92869e319b251eabdb466fc9ac57da13e46c53685dcd25f7b38a91`.
CPU admission: `admission-6Hl0OI`, log SHA256
`4a7eed29e2749daf1087a7d2ed20e0abf85194cd7192f8e748c9dfe07a23ee81`.
Independent reader v3 SHA256:
`42b21b9a73a3d50f3046fbe3ba04fc14a6556b58e76c7303bb1e2765041592c0`.
Audit: 1,693,154 checks, 130 CPU archives; validation SHA256
`0e757b9a07ad9a1c1636a63cc041c24dea5b29a3a10a66a550a7816947262c79`.
Machine summary SHA256: `4e9680f6531e6127b396cc3231815e8698871f701c71834953662a3e0be6ae39`.

Failed compile admission `bji68q` preserves a LibTorch ADL helper-name collision;
the local helper was renamed without changing its formula. The first measured
attempt `2qYQ4Q`, source `1efb6650...`, stopped at the first controlled archive:
its loader asked for `observed` instead of the writer's `observations`. No scores
were computed. That capsule, its passed fixture admission `z9PQHk` and sealed
reader v2 remain unchanged. The corrected admission binds the actual producer
SOURCE field schemas and rejects the old wrong-key fixture before payload access.
Neither the metric definitions, tolerance nor quality recipe was tuned.
No TEST/stress, selection or promotion occurred.
