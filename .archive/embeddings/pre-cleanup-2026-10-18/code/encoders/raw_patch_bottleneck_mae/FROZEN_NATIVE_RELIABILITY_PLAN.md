# Saved TRAIN native-feature reliability plan

Status: prospective; no diagnostic run or new quality scores under this plan.
Date: 2026-10-08.

## Question and scope

The [fresh replication](FRESH_DECODER_REPLICATION_DIAGNOSTIC.md) preserves v7's
missing-observation advantage but fails intact reliability. On master12412,
the retained linear head scores 85.15625% on TRAIN for v7 versus 94.140625% for
v4. Its intact VALIDATION scores are 80.46875% versus 93.75%. The weakness is
already present on TRAIN; these observations do not identify its cause.

Before changing training, ask whether this weakness accompanies reduced
within-source timing separation, unusual native-feature variation, or an
exceptional reconstruction-training trajectory. Compare every one of the five
paired masters, including successful runs. Do not select the best seed or treat
the fragile run as an excluded outlier.

RPB-v4 is the learned global bottleneck. RPB-v7 uses the same architecture with
lighter training-context deletion. V7 remains the working direction, v4 the
active reference. Both export native32, with no PCA afterward. This diagnosis
does not train or promote an encoder and cannot establish a causal mechanism.

## Frozen evidence boundary

Use only the completed capsule
`output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b`.
Its source fingerprint is
`6463926ae71ee5e5547aa660d654afb02832fa843c978c41609cbd814266d78c`,
inventory SHA256 is
`a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90`,
and passed independent audit SHA256 is
`241efd4705e4f6d67edf7070fa650f41e5a2a1b2d6e1e371e3d2835391c567e4`.
Masters are 9109, 10210, 11311, 12412 and 13513. Exact instances are identified
by the capsule, checkpoint/scaler provenance and source bindings, not tag alone.

Prepare and freeze a new TRAIN-only card, reader source and complete input-role
manifest before any diagnostic payload reads. Admit the whole role/path/alias
matrix before hashing. For each master, permit only:

- `controlled-training.pt`, containing legal observations, masks, TRAIN labels
  and grouped source identities;
- `readouts/native_v4/training-features.pt` and the corresponding native_v7 file;
- all three repetitions' `fit.pt` and `training-predictions.pt` for both native
  methods, retaining the existing TRAIN normalizers and classifier weights;
- both encoder `encoder-progress.json` files, containing the retained TRAIN
  optimization traces;
- explicit source/admission/inventory metadata required to bind these roles.

Do not open VALIDATION features/predictions, query archives, whole readout
reports, the capsule's whole quality report, decoder assets with extra payloads,
TEST/stress, or historical quality capsules as analysis inputs. The known
measured report may identify this plan's motivation, but its held-out values
must not become diagnostic observations. Use a new output directory and retain
all failures. Do not rewrite the completed capsule, cards, readers or reports.

## Declared calculations

Use saved native features and fits on CPU inside the managed container. No
encoder forward, model construction, gradient, optimizer, PCA fit or head fit
is necessary. Keep the geometry C3/H32/F3, source order and legal observation
support explicit. Preserve all 256 TRAIN rows / 128 source groups per master.

For both methods on every master:

1. Replay the retained TRAIN predictions using the exact fitted normalizers,
   weights and saved-logit witnesses. Report accuracy and signed true-class
   margins for ridge and all three fixed tanh-16 heads. This reuses existing
   measurements; it does not create new fitted classifiers.
2. Report native coordinate variation, row norms and finite/constant-coordinate
   counts, with definitions fixed in the card. Report both served feature space
   and the existing TRAIN-normalized head input space when their meaning differs.
3. Group the two label variants of each source. Measure their feature separation
   and the ridge projection of that difference. Report whole-source distributions
   and the fraction with the expected signed ordering. Pair v7-minus-v4 values
   on the same source; never treat its two examples as independent sources.
4. Compare reconstruction loss/gradient-norm traces in fixed absolute-update
   blocks 1–128, 129–256, 257–384 and 385–512. These are sampled training losses,
   not fixed-query reconstruction MAE or objective-specific gradients. Keep
   attempts, completed counts and finite/skip evidence explicit.

Define quantiles, tolerances and any descriptive whole-source bootstrap before
reading tensors. Do not search cutoffs, discover favorable source subsets,
invert label conventions, choose a head repetition or invent a combined score.
Rank/collapse or memorization claims require applicable evidence; low margins,
small distances or scalar training traces alone do not establish them.

## Decision and next boundary

The output is one standard all-master table of reused TRAIN scores, a compact
geometry/trace table and a statement of what remains unresolved. It is not a
pass/fail promotion gate. Preserve zero encoder updates, zero decoder updates,
zero head refits and zero held-out analysis payloads in the machine summary.
Verify calculations with a separate saved-arithmetic reader and source fixtures;
do not repeat encoder training to audit them.

If a consistent difference supports a concrete encoder-training hypothesis,
write a separate prospective comparison with new sources and fixed heads before
testing it. If the evidence is inconclusive, say so and consider one bounded
amplitude-transfer check of these frozen encoders, with genuinely new RNG seeds
as well as source IDs. Such transfer must use the original timing-fit scaler
and checkpoint identity; its new TRAIN data fits only the readout. Do not expand
the task suite or alter an objective simply to obtain a favorable result.

No deletion-rate grid, decoder-budget search, head tuning or v9 coefficient
rescue is authorized by this plan. The objective remains a reliable compact
encoder, with evaluation reusable across encoder families.
