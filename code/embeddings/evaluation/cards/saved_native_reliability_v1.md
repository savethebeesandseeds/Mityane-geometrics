# Saved native reliability v1

Protocol: `saved-native-reliability-v1`. Frozen prospectively before diagnostic
payload access. Date: 2026-10-09 (Asia/Dubai).
Companion motivation: the original
[TRAIN-only plan](../../encoders/raw_patch_bottleneck_mae/FROZEN_NATIVE_RELIABILITY_PLAN.md).

## Question and evidence stage

Describe how reliably existing native32 TRAIN features separate the two timing
variants of each source, and whether weak fitted separation accompanies unusual
feature variation or reconstruction-training traces. Use all five paired v4/v7
instances. This is saved TRAIN development diagnosis, not new training, a
promotion gate, causal identification or consumer confirmation. No fitted
classifier recipe, weight, normalizer or model may be changed.

Display labels: `RPB-v4.alt-01` is the matched fresh v4 control group;
`RPB-v7.alt-01` is the separately trained v7 group. The original saved `RPB-v7`
on masters4404/5505/6606/7707/8808 remains frozen and is not an analysis input.
Both analysis groups use masters9109/10210/11311/12412/13513. The display labels
do not change physical paths, stored provenance or historical model tags.

## Closed inputs and identity

The sole parent is
`output/runs/rpb-fresh-decoder-replication/fresh-decoder-replication-4p9U4b`.
Parent source fingerprint:
`6463926ae71ee5e5547aa660d654afb02832fa843c978c41609cbd814266d78c`.
Parent inventory SHA256:
`a65901eb7f14d57152188a86297ed44a966b9858d84848e3ae7e5b047d083a90`.
The previously published independent audit SHA256 is
`241efd4705e4f6d67edf7070fa650f41e5a2a1b2d6e1e371e3d2835391c567e4`;
its mixed quality payload is not an analysis input and must not be decoded.

Admit all roles, resolved paths, sizes, symlinks and inode aliases before hashing
or opening any payload. Admit source and metadata matrices separately. Bind
exact source/card/binary/fixture-admission fingerprints before measurement.
Each master contributes exactly 17 unique TRAIN roles, for 85 total:

- `results/seed-{master}-lag_sign/controlled-training.pt` once, shared by the
  two declared methods on that master;
- `readouts/native_v4/training-features.pt` and native_v7 equivalent;
- each native method's `rep-2701`, `rep-2802`, `rep-2903` `fit.pt` and
  `training-predictions.pt` (12 roles per master);
- each `v4` and `v7` `encoder-progress.json` (two roles per master).

Paths in the last three bullets are beneath the same declared master directory.
The fixed instances TSV header has 12 fields: `id`, `display_tag`, `master_seed`,
`controlled_training`, `native_training`, `encoder_progress`, `fit_2701`,
`pred_2701`, `fit_2802`, `pred_2802`, `fit_2903`, `pred_2903`. Ten instances use
IDs `v4-{master}`/`v7-{master}` and the labels above. The canonical checksum
manifest contains exactly the 85 unique absolute paths and SHA256 values copied
from the pinned parent inventory. Shared controlled TRAIN is the sole permitted
role sharing; unrelated hardlinks, symlinks, missing or extra roles reject.

Permitted metadata is the pinned parent inventory, explicit captured/admitted
source lists and admission files, plus this new protocol's manifests/card.
Exclude VALIDATION/TEST/stress, decoder/query assets, whole readout/quality
reports, older quality capsules and newly generated quality cohorts. Do not
discover input candidates with a recursive payload scan.

Each instance has 256 rows from 128 sources, two rows per source with one label0
and one label1, native width32 and complete support. Controlled observations
are CPU float64 C3/H32/F3 with bool feature masks and zero hidden storage. Labels
are diagnostic/readout labels; they were never encoder-fitting inputs. Source
identity/order/labels and native provenance must agree across supplied assets.

## Exact retained prediction replay

Run on CPU inside the existing managed container. Load saved tensors and apply
the existing outer TRAIN affine normalizer, followed by each existing probe's
affine normalizer. Recompute ridge logits, tanh hidden preactivation and neural
logits from saved weights. No fit, model forward, encoder/decoder optimizer,
autodiff, PCA or quality generation occurs. Retain all three repetitions as head
repetitions of one instance, not independent encoder runs.

Compare float64 arithmetic using absolute2e-9 + relative2e-9. This tolerance is
declared before access; it is not selected from measured differences. Saved
prediction classes must exactly equal argmax of their own saved logits. Reuse
those classes for reported TRAIN accuracy; a scalar replay near a tie cannot
change a retained class decision. Stored valid masks, source groups and labels
must be exact. Explicitly report any unsupported/failed asset; do not repair it
or omit its instance.

For each head, true-class margin is `(2*y-1)*(logit1-logit0)`. Report all 256
margins, accuracy, correct/valid/total counts, coverage and margin distribution.
Zero margin is an ambiguous margin even when argmax's tie convention is correct.

## Descriptive feature and source geometry

Compute separately in served native space, saved outer-normalized space and
final ridge-input space. Do not refit a map or deploy a geometric transformation.
Report per-coordinate population mean and population standard deviation, row
L2 norms, centered row L2 norms, finite/constant coordinate counts and pooled
label-mean distance. A constant coordinate has population SD <=1e-12;
near-zero row/pair norm means <=1e-12. These are descriptive definitions, not
failure/selection cutoffs. Never rescale or repair small norms.

Group by complete source ID and form `delta = feature(label1)-feature(label0)`.
Report each of the 128 source-pair L2 distances, pair midpoint norms and ridge
score differences `(logit1-logit0)[label1]-(logit1-logit0)[label0]`. Report the
fraction with strictly positive ridge ordering, along with zero/negative counts.
Pair v7-minus-v4 values by the same source, not independent row samples. Any
cosine or norm-normalized quantity is null for a zero denominator, retaining its
zero count and population; do not insert epsilon or discard that source.

Distribution summaries use arithmetic mean, population SD, minimum, maximum
and quantiles at 0, .05, .25, .5, .75, .95, 1 with sorted linear interpolation
at `(n-1)*p`. No bootstrap, significance gate, combined score, cutoff search,
source subset selection or head-seed selection is declared.

## Reconstruction-training trace description

Retain all 512 attempted/completed/unskipped updates and 4,096 row presentations
per instance. Check finite CUDA-source input/loss/gradient/update evidence and
parameter count225805 from recorded provenance. Summarize fixed absolute-update
blocks1–128,129–256,257–384,385–512. For each block retain all optimization
losses and gradient norms with the same distribution definitions above.
These sampled Huber losses are not original-query MAE and do not identify
objective-specific gradients. Do not use them as causal evidence alone.

## Evidence, reporting and next boundary

Use exclusive new output directories. Capture sources/card/role manifests and
synthetic fixture admission before measurement; verify all inputs and sources
unchanged afterward. A separate reader verifies retained maps/logits/classes,
margins/geometry/trace arithmetic and source grouping without model/head fits.
Freeze its source and fixtures before measured payload reads. Keep failed
attempts additive. Report independent audit limits and elapsed time separately.

Use standard quality tables for reused TRAIN scores and a compact separate
geometry/trace table, all masters and both methods. State zero encoder updates,
zero decoder updates, zero head refits and zero held-out analysis roles. No new
accuracy is attributed to a changed encoder. Original v7 stays frozen.
This descriptive result may motivate one separately planned encoder variation;
it does not justify a rate/budget/head grid or rescue of the stopped v9 loss.
