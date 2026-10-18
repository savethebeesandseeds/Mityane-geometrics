# Saved native-view loss-scale TRAIN diagnosis

Date: 2026-10-08. Protocol: `native-view-loss-scale-train-v1`. Completed descriptive extraction.

RPB-v9 retained its failed disposition. Its fixed point 0 scale floor was never activated: all 160 coordinates across the five runs have zero floor flags. Agreement grew later in training, rather than beginning with a floor hit. This is a saved-TRAIN lesson with zero new encoder updates, head fits, gradient evaluations, interventions or quality measurements.

RPB-v9 is the native-view agreement training recipe with native32, weight 0.05 for agreement and weight 0.01 for variance. RPB-v7 remains the starting point for separately declared work; RPB-v4 remains the reference. The historical quality report and frozen card are unchanged.

Exactly 30 files from the completed glrBZE capsule were admitted: five controlled-TRAIN archives, five trainer JSONs, and point 0/512 point JSONs plus CPU audit companions for each master. There were 15 tensor archives. Point JSON validation summaries were not used; no VALIDATION, TEST or stress arrays, model checkpoints, native exports or head assets were opened. All 30 source files were checked before and after extraction and remained byte-identical.

Each run retained 256 TRAIN rows / 128 source groups and 512 unskipped updates at batch 8: 4,096 sampled row exposures, not 4,096 independent examples. The fixed s0 values come from saved full-O point 0 TRAIN calibration. The comparison arrays come from ordinary and deleted masked training views.

| Master | s0 min | s0 median | s0 max | Floor axes | Mean Huber | Mean 0.05 agreement |
|---|---:|---:|---:|---:|---:|---:|
| 4404 | 0.005515 | 0.007861 | 0.010124 | 0 | 15.010216 | 11088821.360378 |
| 5505 | 0.006994 | 0.014918 | 0.024029 | 0 | 0.517389 | 205.790310 |
| 6606 | 0.006651 | 0.010857 | 0.014986 | 0 | 0.627142 | 456.424500 |
| 7707 | 0.009793 | 0.018066 | 0.031481 | 0 | 0.520022 | 155.782707 |
| 8808 | 0.009489 | 0.015145 | 0.024065 | 0 | 0.576378 | 1287.075912 |

For each coordinate's mean squared difference over supported rows, the fixed coefficient is `0.05/(32*s0²)`. Across all runs it ranges from 1.5766 to 51.3734. This is coordinate weighting in the scalar objective, not a measured per-objective gradient allocation. All 32 axes, quantiles, coefficients and every saved update remain in the numerical evidence.

| Master | First 0.05 agreement/Huber >1 | First >10 | First Huber >2× early median | First >10× early median | Preclip norm >1 updates |
|---|---:|---:|---:|---:|---:|
| 4404 | 192 | 259 | 294 | 427 | 512 |
| 5505 | 119 | 142 | — | — | 511 |
| 6606 | 125 | 129 | 455 | — | 512 |
| 7707 | 89 | 98 | 451 | — | 511 |
| 8808 | 63 | 70 | 261 | — | 512 |

The early baseline is the median Huber of updates 1–8. Threshold crossings are noisy minibatch markers; they do not establish deterioration, causal order or acceptance gates. In master 4404, Huber block means were 0.4852/0.4790/0.5023/19.8480 for updates 1–8/9–32/33–128/129–512, while weighted agreement block means were 0.1061/0.0390/0.0070/14,785,095.14. Other masters also show later agreement growth; all trajectories are retained.

The saved gradient norm is the combined pre-clipping norm and exceeded the unchanged bound 1 on 511 or 512 updates per run. No objective-specific gradients were saved, so scalar ratios cannot prove gradient dominance or identify the cause of v9 failure. The extraction does not justify a replacement scale, coefficient or rescue experiment.

The source-only suite covered population standard deviation/floors, valid-row exclusion, quantile interpolation, strict one-based crossings, zero ratios, supported same-row F32 reductions, concentration and corrupted-component rejection. Its late-invalid-last-path fixture proves the entire closed input matrix is validated before the first payload hash. Peer source review and the one-flag release proof preceded extraction. Saved agreement arithmetic passed the previously used absolute 2e-6 plus relative 2e-5 component tolerance; this is a small consistency replay, not a rerun of the full independent audit.

The [compact numerical record](../../../doc/results/native_view_loss_scale_train_v1.json) binds the full roles and identities. The [frozen TRAIN-only card](../../evaluation/cards/native_view_loss_scale_train_v1.md), [all saved statistics](../../../output/runs/rpb-v9-scale-train-diagnostic/saved-train-glrBZE-v2-6b1d590a/summary.json), [source/input inventory](../../../output/runs/rpb-v9-scale-train-diagnostic/saved-train-glrBZE-v2-6b1d590a/artifact-integrity.json) and [input after-check](../../../output/runs/rpb-v9-scale-train-diagnostic/saved-train-glrBZE-v2-6b1d590a/input-integrity-after.json) are preserved. The [historical v9 quality report](NATIVE_VIEW_AGREEMENT_VALIDATION_DIAGNOSTIC.md) is separate.

Identities: source capsule inventory `5d8ae964a2d05fd68aaf2fb5c6173e5faaea35e68fbf4a3807b7f04afa0781dc`; orchestration/training producer `3439df37b420c0b4f7ade3787f85befa77624b66a27904d446ed824315abda0e`; core writer `e464f7d71bbf511457a3385dd796099a923808240d80a5e061950825f97f8d4d`.

Extraction card `dbf62bc154786064113abfc984bbf42e5bf2264ac0fc9eb449b262d3b2e78860`; reader `3363a550467cf23f5c1a39791a791e42301f67b25278f885281661b10c0a3ade`; pinned CPU source `4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d`; summary `e8009714304da47c04faca79a181ae4f99b1db5e76296f36c547940fe651cbae`; extraction inventory `be7b29a3c5b1a9c84278304b78ada8dfbedac6c3161ee0e02105a083200fc94f`. The original passed audit `c0a96603d9b5e38e619841a5027088a383ce76f8a376409b1dd63b71fe8946c5` is a metadata reference and was not rerun.
