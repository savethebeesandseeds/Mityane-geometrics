# TEMPO-3 saved-TRAIN diagnosis, version 1

Protocol: `structured-hard-timing-train-diagnostic-v1`. Prospective saved arithmetic
only. Freeze this card, both tools, unchanged codec/math and artificial fixtures
before ROOT authorizes the one TRAIN analysis. No encoder, decoder, head or
optimizer is instantiated or fitted; no PCA/SVD, CUDA inference, VAL/TEST archive bodies,
hidden waveform, checkpoint bodies or new observations. No promotion gate.

Parent: `structured-hard-timing-xrZMAS`; dataset TEMPO-3, generator
`structured-hard-timing-v1`, designed ordinal complexity 4/5. Parent comparison
is `structured-hard-timing-comparison-v2`, fixed original .15/512/B8/native32,
225,805 parameters and fixed heads. Preserve all five masters
75272/76373/77474/78575/79676, late RPB-v7.alt-05 and early RPB-v10.alt-05.

Parent inventory SHA:
`9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa`.
Passed audit SHA:
`cf79abdca511f22778622d2b08e62b74b592c10618d1090ea2bd53eefd84539c`.
Durable summary SHA:
`1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4`.
Inventory is role-binding metadata. The prior audit is hash-bound provenance,
including exact paired scaler/target/support verification; it is not a new
quality input. Head analysis may use the durable summary's already audited
TRAIN/intact/deleted summaries as comparison metadata, without opening VAL bodies.

## Closed TRAIN input roles

Under `results/seed-{M}-lag_sign/`, the query tool reads exactly four CPU roles
per master (20): `controlled-training.pt`,
`late/point-512/training-reconstruction.pt`,
`early/point-512/training-reconstruction.pt`, and
`late/point-512/checkpoint.pt.scaler.pt`. The single frozen TRAIN scaler is
already proven byte-equal between designs by the immutable passed audit. Verify
its observed TRAIN counts, population mean/scale, floor and saved target affine
arithmetic again from the selected controlled TRAIN rows. No second scaler fit.

The head tool reads 110 CPU TRAIN roles: the five shared controlled-training
files, plus `lag_sign/readouts/{raw,native_late,native_early}/training-features.pt`
(15), and `rep-{2701,2802,2903}/{fit.pt,training-predictions.pt}` under those
method directories (90). Raw width is 576; native widths are 32. Raw is prepared
by the saved TRAIN map; the saved raw head's outer map is identity. Native saved
outer/head maps remain separate. Query and head tools share five controlled
files, making the union exactly 125 unique TRAIN payloads. Admit the complete
selected path/inode/size matrix before hashing or decoding; reject aliases and
anything outside these roles. Recheck selected bytes after the arithmetic.

## Fixed-head TRAIN diagnosis

Retain every method, master and all three fixed repetitions. Replay saved outer
and head normalizers, Ridge affine and tanh16 equations against saved witnesses
with F64 absolute 2e-9 plus relative 2e-9 tolerance. This verifies arithmetic;
it does not reproduce fitting. Original saved logits define exact argmax
classes, first-class ties, accuracy and sign/margin statistics. Independent
replay rounding does not rewrite decisions near ties.

Report conditional TRAIN accuracy, coverage and full-population correctness;
true-class signed logit gap; lexical source label1-minus-label0 gap contrast;
strict positive score order, strict opposite signs, both-correct and same-class
pair counts. Keep unsupported rows/pairs in original denominators (256 rows,
128 source pairs), with explicit supported counts and null supported ratios
when support is zero. Compare to existing audited held-out metadata descriptively.
Test opposite signs with exact sign comparisons, avoiding multiplication
underflow; zero is neither positive nor negative.
No classifier or normalizer is fitted; no PCA or bootstrap is executed.

## Reconstruction timing diagnosis

Each saved prediction and standardized target has shape [4,256,3,32,3]. Four
banks came from four different original-Q masked-context forwards. They are not
one full-original-context latent. For bank b, use only its patch [8b,8b+8).
Eligibility is the saved original-observed `target_mask`, verified against the
controlled original mask and saved trial/channel eligibility. Never stitch
values from different banks, cross a patch seam, use visible-context values or
replace unsupported coordinates. Saved prediction/target values outside that
mask are exactly zero; do not invert the scaler there.

Invert the fixed TRAIN affine map on legal cells: raw = standardized*scale+mean,
using the channel/feature map. Verify saved target values against F32 casts of
original controlled TRAIN standardization with absolute 2e-6 plus relative
2e-5 tolerance. No rescaling or new target normalization.

For centre c and spacing k in 1..4, require c-k,c,c+k all inside the same bank's
eight-tick patch (so k=4 cannot qualify). Independently for channel 0 and 1,
take features whose three cells are legal. Set left=x(c)-x(c-k) and
right=x(c+k)-x(c). Average left and right differences
over those eligible features, then calculate
`mean(d1_left)*mean(d0_right)-mean(d0_left)*mean(d1_right)`.
Average eligible spacings within each centre, then equally average the supported
centres across the four banks. Offsets cancel; positive feature gains preserve
the ideal sign. Apply exactly the same cells and reductions to saved targets
and each design's predictions. The saved target is the information baseline,
not the full-window observed-only analytic result.

A row is supported when at least four distinct centres qualify, all legal
arithmetic is finite and its final margin is nonzero. Prediction is margin>0
for label1, margin<0 for label0; unsupported rows have prediction zero but remain
invalid in scores. Zero margin abstains; no epsilon sign repair. Preserve each
row's centre count, per-bank counts, margin, class, validity and source ID.

Report baseline/late/early TRAIN accuracy, coverage, full-population correctness,
per-source paired order, both-correct and same-class counts, and common-support
accuracy differences. Source labels are scoring-only. Retain all five masters
and explicit total/support denominators; average master statistics equally,
leaving undefined statistics null. Descriptive distributions use mean,
population SD, minimum/maximum and linear-interpolation (n-1)p quantiles at
0,.05,.25,.5,.75,.95,1. No across-master CI, bootstrap, fitted threshold or gate.

Poor reconstructed timing against a strong same-support target baseline would
locate a reconstruction-path limitation, not prove encoder information loss.
Strong reconstructed timing with weak served-feature fixed-head scores would
show multi-context encoder/decoder accessibility, not sufficiency of native32.
Targets can lose support under the bank restriction, so first report that
baseline's actual support/accuracy. This diagnosis chooses no architecture;
ROOT will decide the next bounded variation after the complete result.

## SOURCE fixtures and execution boundary

Use unchanged stdlib CPU codec SHA
`4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d`
and pure saved arithmetic SHA
`1993af23915ba8200389727e2810caad175613fae7ead00d5ea0cb5250e303b9`.
SOURCE-only artificial tests cover affine offset/gain cancellation, both signs,
independent eligible feature sets, bank seams, four-versus-three centres,
zero-margin abstention, unsupported/nonfinite handling, tensor geometry,
source pairs and exact closed role names. ROOT authorizes actual TRAIN reads
only after card/source/fixtures and peer review. Each tool executes once into
a new exclusive diagnostic directory, preserves failures and reports its
SOURCE/card/input hashes. Original parent artifacts and earlier reports remain
unchanged; no automatic correction or rerun follows a measured failure.
