# After fresh RPB-v17 verification, 10 October 2026

The [milestone note](MILESTONE_2026-10-10_RPB_V17.md) preserves the successful
five-known-cohort result and its commit. The [new fresh record](../code/encoders/raw_patch_bottleneck_mae/FIXED_PRIOR_FRESH_CONFIRMATION.md)
retains all five **RPB-v17.alt-01** instances from0/512 on fresh TEMPO-3 sources.
Design, native32, .15/B8/512, frozen432, waveform Huber and fixed heads are unchanged.

TEMPO-3, complexity4/5: variable small signed lag/period, gains/offsets, natural
missingness and channel gaps; TRAIN256/VALIDATION128 per cohort. Intact means
are100% with both heads; extra .30 deletion gives82.1875% Linear/99.895833% Neural,
coverage100%. Worst deleted Linear/Neural is70.3125%/99.479167%. Master84090
fails the prospective75%-both-heads continuity rule; its Neural remains100%.
Both predeclared joint rules fail that linear condition. Keep this distinction:
neural timing robustness replicates, all-cohort linear robustness does not.

Initial controls remain100%/100% intact and82.03125%/99.84375% deleted. Credit
the generic fixed prior, not waveform learning. Mean standardized query MAE is
0.705814 TRAIN/0.731698 VALIDATION; mean synchronized training-loop wall is21.82s.
Five quality trajectories ran once on CUDA,2,560 updates. CPU fits the fixed
heads and checks saved tensors; it never runs an encoder or optimizer.

Evidence: capsule `output/runs/rpb-fixed-prior-fresh-confirmation/admission-3CGpxM`,
compiled62-file source9cc2b7dd4f69c54ab0c74360a2c1e01b9ecdcb584724e3d88fc2ea0f1520057c,
card533afbcd14f03ad373f0dcfb928747d105b008e2e201a11f89f4a85f3d501c02,
quality reporta7db8524b6c51d40dde1ab132f12d8fcfea58f2f614b4df401ff28647b972f28,
inventory3e50676e863324c135cb53a920b3e436b73e54d687f9a9b8483e5b45526ae9af.
Saved check passed5,833,011 checks/180 CPU witness archives and30,720 exact
odd-coordinate values. Full losses/checkpoints/scalers/RNG/heads/queries persist.
The compact [JSON](results/fixed_prior_fresh_confirmation_v1.json) and additive
registry entry preserve every original version and instance group semantically.

The focused development hypothesis was **RPB-v18** under the separate
[unit-relation card](../code/evaluation/cards/unit_relation_screen_v1.md): normalize
each generic four-spacing odd vector to unit length inside native32. Same learned
20 coordinates and fixed432, no new weights, labels or head changes. This removes
per-example strength variation while preserving signs/relative spacing; it can
amplify weak noisy pairs and loses magnitude/confidence. It is a hypothesis,
not a learned-timing claim. The completed screen below retains its initial control.

Initial known screen80787/84090 explicitly included the weak linear case.
The prospective gate required75% in both heads/views and100% coverage per
cohort, Neural no worse than matched saved v17 per view/cohort, and mean deleted
Linear improvement of at least5pp. All conditions passed: deleted Linear rose
from75.00% to100%; Neural from99.74% to99.87%. That saved gate authorized
the unchanged recipe's three remaining known-cohort runs, which are now complete.
All v17 comparisons reuse saved metadata; no old-model/head repetition.

## Completed RPB-v18 development

The [full report](../code/encoders/raw_patch_bottleneck_mae/UNIT_TEMPORAL_RELATION_SCREEN.md),
[durable JSON](results/unit_relation_screen_v1.json) and
[milestone note](MILESTONE_2026-10-10_RPB_V18.md) retain all five cohorts.
Intact Linear/Neural means are100%/100%; extra-deleted means100%/99.947917%,
coverage100%. The worst cohort's deleted Linear/Neural is100%/99.739583%.
The same v17.alt-01 group was82.1875%/99.895833% deleted. Native32, fixed
heads, original waveform Huber1, .15/B8/512,226,877 total parameters and
226,445 trainable parameters remain unchanged. Per-pair unit normalization
adds no weights and keeps the432 grouped relation weights fixed.

Initial0 v18 is already100%/100% intact and100%/99.895833% deleted. The
timing advance therefore comes from the generic architecture prior. Training
continues to learn waveform shape and reconstruction; that ability needs a
separate harder task. Standardized fixed-query MAE is0.692694 TRAIN and
0.719535 VALIDATION, with21.89s mean synchronized CUDA training-loop wall.
These are known-data development results, not a new unseen confirmation.
No TEST, stress suite, default change or reference promotion occurred.

Evidence: `output/runs/rpb-unit-relation-screen/admission-DiR2yH`;
59-file compiled source fingerprint
`d82d92864c87abb985fd0b1153927bcded70904b54e9c7ba74b0101cbe850993`;
card `906b5ab75da3c12d243ebe915316c1f9cb4a46727f32e1896219ff66305997b9`;
final summary `48d8beaa9fa0bdfd9440f291d0c8a5046f0dc9c747c7f03475a92ccdf15cc5ba`;
inventory `30a4ae4a463c947c3ea49dcfd53fa1b11e167fef955a5577942e89518b445b4b`.
The prospective gate receipt remains in `gate-summary.json`/`gate-report.md`.
Both quality subsets passed saved CPU arithmetic checks:5,833,018 checks,
180 archives and30,720 exact fixed coordinate values, with zero model execution.

This turn's complete quality ledger is ten CUDA trajectories,5,120 updates,
40,960 sampled rows,60 native exports,60 fixed head pipelines/120 individual
heads,20 query writers/80 necessary masked CUDA forwards. The saved checks
total11,666,029 checks/360 CPU witness archives/61,440 exact prior values.
Only five new quality generator calls were needed for fresh v17; v18 reused
all15 legal dataset files. Tiny engineering admissions are separate. No old
quality model/head refits, CPU encoder execution, post-encoder PCA or skipped
quality updates occurred. Complete checkpoints, optimizer/RNG, scalers,
initial/trained features, fitted heads, queries and512-entry loss traces persist.
The additive registry preserves all18 prior designs and17 prior instance bundles.
Independent final v18 metadata review passed3,947 comparisons with no substantive
mismatch. Final guards confirmed both62-file fresh source freezes and both59-file
unit source freezes, their captured metadata tools and Python/shell syntax.

The [TEMPO-4 proposal](PROPOSED_TEMPO4_CHALLENGE.md) stays unmeasured. It uses
two coherent separated frequency components: **TEMPO-4** slow-band timing and
separate **AMP-2** relative-balance shape, each proposed complexity5/5. Before quality,
freeze its final generator/card/solvability admission. Do not create chaotic
labels or tune the shared classifier to reach an arbitrary number. This richer
challenge should test what the learned shape block adds beyond the fixed prior.
Use v18's preserved prior and initial control for a small screen of a separately
specified frequency-sensitive candidate; do not repeat a large old-version
comparison. Decide from both tasks separately whether to expand it.
Original v7, formal v4, TEMPO-1 v10.alt-03 and all defaults remain protected.
