# Configurable RPB-v20 saved-coordinate diagnostic v1

Prospective development diagnostic. Freeze this card, its code/binary closure
and the original capsule binding before any new quality classifier fit. This
does not train or forward an encoder, allocate new quality sources, alter a
checkpoint, or change its native export. Retain every declared outcome.

## Original evidence and question

Use only **RPB-v20.alt-01**, profile A, C3/H128/F9/P8/W64/D32, from
`output/runs/rpb-v20-capacity/admission-0gN4uD`. Masters are exactly
941101, 941102 and 941103; retain points 0 and 512. The original experiment
source fingerprint is
`8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1`;
its artifact manifest SHA-256 is
`62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83`.

Dataset: **TEMPO-5**, slow-component timing beneath faster rhythms, nine grouped
features and structured distractors, designed complexity **5/5**. Each master
has TRAIN 512 examples / 256 paired sources and VALIDATION 256 examples /
128 paired sources. Extra-deleted VALIDATION removes 30% of remaining
coordinates and retains the same source population. These are known development
sources, not new confirmation or TEST. The original three encoder trajectories
used 512 CUDA updates each, batch 8. No encoder operation occurs in this study.

Question: how well can the fixed classifier recipes use the initialized and
trained learned coordinates, the fixed coordinates, and their existing complete
native32 export? Coordinate subsets are diagnostic readout inputs, not encoder
versions. All encoder result rows retain the tag **RPB-v20.alt-01**.

## Fixed diagnostic matrix

- Complete32: reuse original heads, predictions and metrics at 0 and 512.
  Do not fit them again.
- Learned20: exact first 20 saved native coordinates, separately at 0 and 512.
  Fit each permitted TRAIN population once per declared head repetition.
- Fixed12: exact last 12 coordinates. Before fitting, require exact equality of
  values and validity between 0 and 512 for TRAIN, intact and deleted VALIDATION.
  Fit once per master/repetition and reuse this one result at both points.
- Raw and raw-PCA baselines, if displayed, reuse the original saved metrics and
  decisions. No raw preprocessing, PCA fitting or baseline classifier fitting.

Exactly 27 new quality fitted head pairs / 54 classifiers: 9 fixed12 pairs and 18
learned20 pairs. Exactly 18 original complete32 fitted pairs are reused. Reused
raw/PCA fits, if displayed, remain separately counted. No data redraw, extra
repetition, checkpoint selection, adaptive training budget or unsupported-row
discard. Fit failure is retained explicitly and stops completion if the fixed
artifact/schema contract cannot be audited.

The separate engineering self-test fits 9 head pairs / 18 classifiers to
artificial arrays, with no quality dataset access. These are additional
engineering fits, excluded from the quality counters and result tables.

Use unchanged shared heads: ridge penalty 1; tanh neural head with 16 hidden
units, Adam learning rate .01, 100 updates; repetitions 2701, 2802, 2903 and
the existing `stream_seed(repetition,width)` initialization policy. Fit all
scales/normalizers only on that branch's legal TRAIN population. Reuse the
TRAIN-fitted head unchanged on both validation views. No PCA follows an encoder.
CPU work operates only on saved arrays and classifier parameters; CPU encoder
calls, GPU encoder calls and new encoder updates are all zero.

Widths 12/20/32 have respectively 26/42/66 linear parameters (including
intercepts) and 242/370/562 neural parameters. Width also changes neural
initialization draws and input scaling. Comparisons across branches therefore
measure usefulness to these fixed recipes, not intrinsic information or a
perfectly controlled causal contribution. Learned0 versus learned512 retains
the same width and initialization seed and is the cleaner training comparison.
Initialized learned coordinates are random features, not learned improvements.

## Audit and reporting

Bind every read input to the original artifact manifest and source/card/settings
receipts. Check exact source IDs and original labels, exact coordinate slices,
dtype/shape/finiteness/support, TRAIN-only normalization, saved head arithmetic,
predictions, populations and source-group intervals. Independently audit all
new fits and every reused decision that appears in the report. Check original
artifact hashes before and after execution and current source/binary hashes
afterward. Save artifacts in one new immutable capsule; preserve failed attempts.

Report all three masters and all three head repetitions; average repetitions
within master, then masters equally. Ridge repetitions are deterministic,
not independent uncertainty estimates. Conditional 95% source-group intervals
and paired differences remain descriptive on known validation. Do not treat
source intervals as uncertainty across independently trained encoders.

Use the agreed five-column quality tables. Place coordinate-part descriptions
beside the tables: size32 is the complete embedding, size20 is learned-only,
size12 is fixed-only. Separate initialized/trained and intact/deleted panels,
each with a TEMPO-5 / 5-of-5 legend. State that fixed12 fits are reused between
points. Preserve full numerical reports, head parameters, seeds, fit counters,
CPU fitting/audit wall time and provenance. Reconstruction and GPU training values
may only be referenced as original measurements; this study measures none.

There is no promotion gate or automatic encoder change. Use the outcome to
decide which subsequent matched experiment is informative, keeping history,
feature grouping and physical frequency coverage distinct. Any reusable encoder
change requires its own tag, focused CUDA verification and synchronization with
the clean `cuwacunu` integration repository before handoff.
