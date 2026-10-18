# After the multiband screen, 10 October 2026

The [completed report](MULTIBAND_SCREEN_V1.md) and
[durable summary](results/multiband_screen_v1.json) retain the successful
two-task screen. **RPB-v19** is the new mixed spectral-relation design;
**RPB-v18.alt-01** names new unchanged v18 instances on these harder data.
The original [TEMPO-3 v18 milestone](MILESTONE_2026-10-10_RPB_V18.md), formal
RPB-v4, original RPB-v7 and every historical group remain preserved.

**TEMPO-4**, designed complexity **5/5**, tests slow-component lead/lag in a
coherent two-rhythm mixture. **AMP-2**, separately designed complexity **5/5**,
tests relative slow/fast strength. Both use gains/offsets, natural missingness
and channel gaps; an extra30% deletion view retains the same validation rows.
The unchanged `two-component-v1` generator first passed its separate observed-only
information admission. That calculation is never an encoder feature or target.

Two fresh quality masters920903/920904 per task. TRAIN256/128 source pairs and
VALIDATION128/64 pairs per trajectory. Both designs use native32, B8,512 CUDA
updates, original waveform Huber1, context deletion.15, Ridge1 and tanh16/Adam
.01/100 with repetitions2701/2802/2903. No post-encoder PCA or head changes.

V19 replaces v18's four-spacing temporal relations with mask-aware generic
harmonic low/high complex relations. It retains20 learned shape coordinates,
12 fixed relations,226,877 total parameters and226,445 trainable parameters.
The sole served32 is also the sole decoder input. Both actual CUDA primitive
and runner admissions passed before quality; every quality trajectory ran once.

The frozen joint gate passes all10 conditions. V19's intact Linear/Neural means
are98.83%/98.83% on TEMPO-4 and98.44%/97.92% on AMP-2, all coverage100%.
Extra deletion gives92.97%/94.27% timing and94.53%/88.41% balance. Matched v18
deleted means were78.12%/75.26% and60.16%/56.64%. Every candidate cohort/view
exceeds75% with both heads; the weakest is87.50% Neural on AMP-2 deleted920903.
Deleted Linear gains are14.84375pp timing and34.375pp balance.

The initial prior already supplies nearly all Linear performance. Initial v19
deleted means are92.97%/92.32% timing and94.14%/86.07% balance. Training supplies
modest Neural gains, not the main architectural advance. Original masked-query
MAE remains around.75 timing/.77 balance and is slightly worse than v18.
Do not claim improved reconstruction or established general learned-shape utility.

Quality ledger: eight trajectories,4,096 updates,32,768 sampled rows,48 native
exports,48 fixed head pipelines/96 individual heads,16 query writers/64 necessary
masked CUDA forwards. Four data-only quality generator calls created12 archives
shared by both designs. Saved-only verification passes9,646,440 checks/288 CPU
archives and49,152 byte-exact fixed coordinate values at0/512. It executes no
encoder, optimizer, new head fit, bootstrap or frequency search. Full traces,
checkpoints/optimizer/RNG, scalers, heads, predictions and queries persist.

Evidence: `output/runs/rpb-multiband-screen/admission-0Yzi6p`.
The72-file SOURCE fingerprint is
`f966efd3745727447e1a601b08e354761b1ad554d380d34a8ee610192848a2ee`;
card `8fdbdf57363e5cbeaf2e55d267770ffb59a3dccb253109204d2f77b7577079a5`;
summary `e2c151a212e73dc157bce36c7b2dea15c92ebeb9fc0a14ec1a288b00d405f870`;
artifact inventory `e1ddeb7044b4846cb4be4d6eaad079d42d7e06e9a006ba01035942f1b79f0545`.
The additive registry preserves all19 earlier designs and17 earlier instance
bundles, adding v19 and v18.alt-01. Source/data/CUDA admission was committed and
pushed as `41df6e0` before quality began. No TEST, stress, promotion or default change.

Next, confirm the unchanged v19 recipe on fresh independent sources through a
separate prospective card and **RPB-v19.alt-01** instance bundle. Fix all seeds,
counts, absolute quality/coverage guards and the small continuation budget before
generation. Keep both datasets separate, retain initial0/trained512 and every
failure, and use the same native32/heads. Reuse evaluation helpers and saved
historical metadata; avoid another large old-encoder comparison. The completed
screen allocates no further quality seeds.

After confirmation, target the learned shape path's missing useful rhythm
information through one separate architecture hypothesis: replace the nonlinear
patch-conditioned decoder with a linear generic harmonic-synthesis decoder.
Keep the exact encoder/shape20 and fixed spectral12. Predict DC plus sine/cosine
coefficients at unchanged q1..8 for all nine channel/feature streams from the sole
native32:153 coefficients through a32-to153 affine map (5,049 parameters), then
fixed CUDA synthesis on original time ticks and semantic channel gathering.
Keep waveform Huber and all masks/budgets/heads unchanged; no coefficient target,
hidden-frequency input or auxiliary. Preserve exact initial native32 parity
with v19, so trained gains test learning rather than a stronger initial prior.

This decoder would require globally coherent, linearly accessible shape, while
off-grid truncation and the20-coordinate learned allocation may hurt waveform
reconstruction. Freeze native trained-versus-initial gains and original-query MAE
guards prospectively; lower MAE alone cannot establish useful classification
geometry. This is proposed and unmeasured, with no new version tag or quality
allocation yet. Protect the fixed12 relations and original Q metrics.
The next candidate must improve useful learned
representation without tuning heads, masking, frequency cutoffs or stopped recipes.
Keep the two-rhythm data fixed while testing that question; do not add chaotic
noise to manufacture difficulty or call this small family universal evidence.
