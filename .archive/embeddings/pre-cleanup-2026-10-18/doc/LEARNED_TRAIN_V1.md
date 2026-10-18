# Learned-coordinate TRAIN diagnosis

Completed on13 October2026 under the prospective
[learned-train-v1 card](../code/evaluation/cards/learned_train_v1.md).
This reads saved RPB-v20.alt-01 arrays and parameters. It trains no encoder or
classifier and decodes no numeric VALIDATION field.

Dataset: **TEMPO-5**, slow timing beneath independent fast rhythms, nine grouped
features and structured distractors; designed complexity **5/5**. Full legal
TRAIN context only. Three original cohorts941101/941102/941103,512 examples
and256 source pairs each; initialized and512-update points are reused.

The learned20 coordinates are active and change with training. None is constant
or reaches the head normalizer's variance floor. Every one of the30,720 learned
TRAIN cells changes; the18,432 fixed TRAIN cells remain exact. Saved TRAIN
normalization and encoder scalers agree with their original fitting population.

Dataset: **TEMPO-5**, complexity **5/5**, full-context TRAIN. Means below are equal
means over the three cohorts; these are descriptive diagnostics, not accuracy.

| Native subset | Point | Mean coordinate std | Mean within-source pair L2 distance | Decoder linear potential RMS |
| --- | ---: | ---: | ---: | ---: |
| Learned20 | 0 | 0.01602 | 0.00707 | 0.05894 |
| Learned20 | 512 | 1.16759 | 0.08434 | 5.76633 |
| Fixed12 | 0 | 0.42172 | 0.78567 | 1.25105 |
| Fixed12 | 512 | 0.42172 | 0.78567 | 1.24475 |

Pair distance compares adjacent same-source variants without reading or using
their labels. Its branches have different dimensions and scales. The decoder
column/covariance calculation measures potential centered first-layer linear
contribution under these full-context arrays; it is not activation, masked
training use, timing information or causal reconstruction reliance. The learned
covariance participation rank remains6.34–8.13 at512; this is variance
concentration, not information capacity.

The diagnosis supports investigating a learned route that keeps ordered timing
comparisons available before compressing them. It does not support describing
the learned route as disconnected or constant. The separate RPB-v21 candidate
adds a zero-start projection of mask-aware ordered comparisons, retaining the
working fixed12 and complete32 output. Its prospective quality screen must
demonstrate learned-coordinate usefulness, complete32 protection and acceptable
original-query reconstruction before allocating its third development cohort.

Evidence: [compact results](results/learned_train_v1/completed/compact.json),
[full results and read inventory](results/learned_train_v1/completed/report.json),
[completion](results/learned_train_v1/completed/complete.json) and
[prospective capture](results/learned_train_v1/completed/freeze/receipt.json).
The full per-coordinate, source-pair and decoder-weight metrics are preserved.

The capsule is `output/runs/rpb-v20-learned-train/diagnostic-ryYYYp`.
Source fingerprint:
`854a71d890c9310edd14d9d75afce1993fcc7d173be9b7a4f1bd1e67ba4371f0`.
Card SHA256:
`ef16d4f4358d556cf4e38fde7a0be5014a0b9b1251d952f4d3c8654af05780d3`.
Artifact manifest SHA256:
`46cf765548b6bda4f4ac3398411974669b570f9552bd431388ada3522d1efa7d`.
Report SHA256:
`09dcfd1c10f6f9745adcf4dae9358957f8349ab3a283d856cea1eda93c73bc67`.

All815 original artifacts passed hashes before and after;23,386 arithmetic and
binding checks passed. Twelve meaningful engineering tests passed under normal
Python and `-O`. Lazy extraction unpacks only allowed logical elements and
rejects unselected-storage aliases. There were zero encoder calls, updates,
classifier fits or numeric validation reads. All21 capsule files were copied
byte-identically to the durable evidence folder. No historical checkpoint or
quality result is changed.
