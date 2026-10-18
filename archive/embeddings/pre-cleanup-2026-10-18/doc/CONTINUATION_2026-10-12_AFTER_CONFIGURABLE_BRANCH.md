# Continue after the saved-coordinate diagnostic

The [coordinate diagnostic](CONFIGURABLE_BRANCH_V1.md) is complete. Preserve
RPB-v20.alt-01, its native32 export and all original capacity checkpoints.
Complete32 scores remain94.66%/94.31% intact and90.10%/88.72% deleted, with
100% coverage. The new fixed12-only results are95.18%/94.66% intact and
89.97%/89.54% deleted. Learned20-only results remain near50%, including
TRAIN after training. Branch width changes head capacity and initialization;
these results establish usefulness under the declared recipes, not information
loss or a fully controlled causal attribution.

The useful next question is why the learned route does not expose timing to
these heads. More width, a larger validation sweep or a harder challenge would
not answer that directly. The learned route pools patch states and compresses
them globally; reconstruction uses complete32 and position/channel metadata.
Its waveform Huber objective does not explicitly require learned20 alone to
retain slow timing. This is an architectural hypothesis, not a measured cause.

1. Predeclare one small saved-TRAIN-only diagnostic. Inspect initialized/trained
   learned-coordinate variation, within-pair changes, TRAIN normalizer scales
   and decoder-column weights. Read saved tensors/parameters without constructing
   or forwarding an encoder. Declare the exact metrics and interpretation limits
   before calculation. Do not fit heads, use validation for tuning or claim that
   weight norms alone measure causal decoder use. No new sources or thresholds
   are allocated by this continuation note.
2. Use that evidence to choose one learned-relation or pooling change. Give it
   a separate design tag, outputs and focused CUDA engineering checks. Retain
   the working fixed prior and the native export/head budget. Freeze its quality
   card before any fresh experiment; preserve all declared outcomes. Avoid a
   collection of capacity or classifier tweaks.
3. Keep matched H128/H256 quality and a recoverable feature-specific relation
   challenge as later steps before market use. Physical frequency coverage and
   feature-group identity require separate controls. Do not alter TEMPO-5 or
   infer market quality from its existing scores.

Completed capsule: `output/runs/rpb-v20-branches/diagnostic-YA3Nws`.
The initial source-role validation failure is separately preserved at
`diagnostic-FaORAo`; it performed zero quality fits. Both attempts retain their
source/card captures and9 artificial engineering head pairs each. The successful
quality diagnostic fitted27 new pairs/54 classifiers and executed no encoder.
All815 original artifacts remain byte-identical. Reuse these saved scores.

All15 reusable encoder/config/test file pairs match between `cuwacunu_embedding`
and `cuwacunu`, including the pinned backport hashes. This diagnostic changes no
reusable encoder code. Future model changes must be synchronized in both
repositories before handoff. Keep evaluation here and market ingestion there.
Build and run only inside the existing managed containers with named sessions;
no container lifecycle or dependency change is needed.
