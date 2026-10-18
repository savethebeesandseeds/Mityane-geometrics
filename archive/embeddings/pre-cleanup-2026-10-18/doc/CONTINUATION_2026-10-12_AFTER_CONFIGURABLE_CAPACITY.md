# Continue after the configurable capacity experiment

The requested runner, TEMPO-5 challenge and nine GPU trajectories are complete.
Use [the measured report](CONFIGURABLE_CAPACITY_V1.md),
[machine record](results/configurable_capacity_v1.json) and
[instance registry](configurable_capacity_embedding_versions.json).
Preserve every initial/final checkpoint and fitted head under
`output/runs/rpb-v20-capacity/admission-0gN4uD`. Read saved evidence rather than
retraining or running a CPU encoder to obtain the same scores.

RPB-v20.alt-01, C3/H128/F9/P8/W64/D32, is the compact continuation configuration.
It gives94.66%/94.31% intact and90.10%/88.72% deleted Linear/Neural means with
100% coverage. Raw/PCA remain near50%. Wider alt-02 and larger-output alt-03
show no consistent final classification benefit at the equal512-update budget.
This does not promote a new default or overwrite any historical model.

Initial alt-01 already gives94.01%/93.23% intact. Training moves the learned
coordinates and reduces fixed VALIDATION query MAE from0.8250 to0.7986, but
classification improves modestly. The next work should identify useful learned
information and improve missing-value robustness, rather than increase width
or tune classifier heads based on these validation results.

1. Specify a separate saved-feature diagnostic of fixed12, learned20 and complete32
   coordinates at0/512. Keep the existing ridge/tanh16 recipes and declare the
   different input widths and resulting head sizes. This needs no encoder forward
   or update; new diagnostic head fits must be identified explicitly, not counted
   as the original fixed evaluations. It can locate which branch needs attention
   but is a development diagnostic on known validation sources.
2. Before market use, predeclare a matched H128/H256 geometry comparison with
   W64/D32, unchanged physical periods/delays, explicit source pairing and the
   existing observed-information gate. The spectral bank's q/H frequencies change
   when H changes; engineering shape checks do not establish quality robustness.
   Fit scales and fixed heads on each permitted TRAIN population, keep GPU model
   operations separate, preserve both0/512 points and use new declared quality
   cohorts. No seeds or outcome threshold are allocated by this note.
3. If the fixed aggregate prior supplies nearly all timing usefulness, define a
   new challenge whose recoverable relation depends on a specific feature group.
   One precise candidate gives two matched groups opposite slow shifts and swaps
   their roles across labels, preserving the feature multiset, corresponding
   gains/noise/masks and thus the aggregate prior. A declared group's timing must
   remain recoverable from its legal observations. Admit that recipe
   through observed-only solvability before model experiments. Then compare a
   feature/band selection architecture or a learned relation route against compact
   v20 at the same export/head budget. Do not silently make TEMPO-5 harder or
   label a known task by its measured accuracy.

The earlier harmonic linear decoder remains a separate, unimplemented proposal.
Reconstruction is still weak enough to justify it later, but improving its loss
alone would not show better embedding quality. First establish what learned
information matters to these fixed heads and to a feature-specific challenge.

Run only in the existing managed `cuwacunu_embedding` container, following its
README/launcher/setup. No container lifecycle change is needed. New tests should
use separately named build sessions and immutable capsules. Historical generators,
cards, source captures, reports and checkpoints stay frozen. No market dataloader
is imported into this repository; reusable core changes belong in both projects,
while task-specific evaluation remains here. This experiment changed no reusable
encoder/config/loader implementation, so there is no new core backport to apply.
