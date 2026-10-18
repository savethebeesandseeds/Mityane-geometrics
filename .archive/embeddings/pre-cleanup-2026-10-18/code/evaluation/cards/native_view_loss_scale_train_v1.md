# Saved native-view loss-scale TRAIN diagnosis v1

Date: 2026-10-08. Protocol: `native-view-loss-scale-train-v1`.
Stage: retrospective saved-TRAIN diagnosis; not a new encoder experiment.
Status: frozen prospective extraction recipe; payload access awaits root review.

This records the lesson from failed RPB-v9 without rescuing its frozen recipe.
RPB-v4 remains the reference; RPB-v7 is the starting point for a separately
declared future advance. No encoder execution, gradients, optimizer updates,
head fitting, recalibration forward pass, quality scoring or audit rerun occurs.
No VALIDATION, TEST or stress arrays are opened. No coefficient, rate, budget,
dimension or source subset is chosen from this diagnosis.

## Exact source and thirty roles

Use only `output/runs/rpb-native-view-agreement-validation/native-view-agreement-validation-glrBZE`.
Pin its inventory SHA256
`5d8ae964a2d05fd68aaf2fb5c6173e5faaea35e68fbf4a3807b7f04afa0781dc`,
orchestration/training producer
`3439df37b420c0b4f7ade3787f85befa77624b66a27904d446ed824315abda0e`
and core writer
`e464f7d71bbf511457a3385dd796099a923808240d80a5e061950825f97f8d4d`.
The passed independent audit is a metadata reference only, SHA256
`c0a96603d9b5e38e619841a5027088a383ce76f8a376409b1dd63b71fe8946c5`.
Its audit is not repeated.

For each master 4404, 5505, 6606, 7707 and 8808, let cohort be
`results/candidate-development/seed-<master>-lag_sign`. Admit exactly:

1. `<cohort>/controlled-training.pt` — legal observed TRAIN and source metadata.
2. `<cohort>/trainer-audit.json` — configured recipe/source metadata.
3. `<cohort>/milestone-0/point.json` — point identity and TRAIN progress only.
4. `<cohort>/milestone-0/checkpoint.pt.audit.pt` — fixed calibration evidence.
5. `<cohort>/milestone-512/point.json` — point identity and TRAIN progress only.
6. `<cohort>/milestone-512/checkpoint.pt.audit.pt` — same calibration and saved
   TRAIN per-update component/native/support evidence.

There are 30 distinct files: 15 tensor archives and 15 JSON metadata files.
Point JSON also contains known-VALIDATION score summaries; only identity and
TRAIN progress fields are used here. TRAIN labels may be physically present in
the controlled archive but are not used. The audit companion's component/native
arrays contain sampled TRAIN rows only, as bound by the captured writer source.
Do not admit a whole cohort directory or reuse an older broader role permission.
Exclude model checkpoints, scaler archives, native exports, head fits and
predictions, validation observations/manifests/queries, all test/stress payloads
and recursive payload discovery.

Before any role payload hash/decode, save the card and a closed role JSON whose
expected bytes/SHA256 come solely from the pinned inventory metadata. Check the
whole exact path/role matrix, then hash only these 30 files. Save the actual
manifest before decoding, and verify the same bytes/hashes after extraction.
The new directory is exclusive under `output/runs/rpb-v9-scale-train-diagnostic/`.
Original inputs and evidence remain unchanged.

## Fixed descriptive outputs

Use all five runs, each with 256 TRAIN examples / 128 source groups, batch 8,
512 unskipped updates and 4,096 sampled row exposures. These are saved actual-CUDA
values; the new stdlib reader runs only in the existing managed container.
Classifier scores and new GPU timings are not measured.

For all 32 coordinates, preserve stored float32 s0, floor flags and identity.
Describe the float64 population standard deviation from the saved point0
calibration values and valid TRAIN rows, without replacing s0 or applying a new
transform. Verify calibration/s0 identity across saved points. Report the floor
count and quantiles at 0%, 10%, 25%, 50%, 75%, 90% and 100%, with linear
interpolation at index `(n-1)*p`, for std, s0, `1/s0^2` and `0.05/(32*s0^2)`.

Retain all 512 component rows per run. Describe ordinary hierarchical Huber,
agreement, variance, saved total and recorded pre-clipping gradient norm.
Calculate `0.05*agreement`, `0.01*variance`, weighted-agreement/Huber ratio
(null when Huber is zero), and their saved-total shares. Show updates
1/2/4/8/16/32/64/128/256/512 and fixed blocks 1–8, 9–32, 33–128 and 129–512.
Report the first weighted-agreement ratio exceeding 1 and 10. For a descriptive
early reconstruction change marker, use the first Huber value exceeding 2 and
10 times the median of updates 1–8; report null for a zero baseline. These are
noisy minibatch threshold crossings, not proof of deterioration or model gates.

Using saved ordinary/student native32 values and auxiliary support, describe
same-row unscaled squared differences and their fixed-s0 scaled contributions.
Retain all coordinate contributions and the declared blocks; do not choose
opposite-label siblings or fit a different scale. Agreement replay is a small
saved-array consistency check with the existing audited float32 component
tolerance (absolute 2e-6 plus relative 2e-5); this does not repeat the full audit.
Report concentration in the largest coordinate contribution and largest five,
including which coordinates they are, rather than dropping small-scale axes.

The recorded gradient norm is a total pre-clipping norm; report how often it
exceeds the unchanged clipping bound 1. No per-objective gradients were saved,
so scalar loss ratios cannot establish gradient allocation, dominance or cause.
Calibration uses full observed point0 TRAIN support, whereas component latents
use masked training views; that difference is descriptive, not a new comparison.

## Reader and preservation

Use only the pinned stdlib CPU archive-reader SOURCE
`output/runs/archive-controls/input-audit-20261006T191016Z-d7b99529/validate_phase1.py`,
SHA256 `4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d`.
No PyTorch import, package install, host Python, model/decoder execution, head
fit, optimizer or old evaluator entry point is permitted. Source-only fixtures
cover scale quantiles/floors, ratio zero cases, threshold indexing and same-row
supported contribution arithmetic before payload access.

Save exact source/card/input identities, all masters, an append-only numerical
summary and human diagnostic report. State explicitly: no new experiment or
accuracy measurement; TRAIN arithmetic only, no gradients/generalization or
causal conclusion. Any next mechanism needs its own prospective card.
