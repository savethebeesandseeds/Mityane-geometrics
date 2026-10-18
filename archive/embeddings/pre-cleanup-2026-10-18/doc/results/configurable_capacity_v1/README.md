# Exact evidence subset

These files are byte-for-byte copies from the completed local capsule
`output/runs/rpb-v20-capacity/admission-0gN4uD`, except this explanatory README.
Git attributes disable text conversion in this directory to retain hashes.

`freeze/source/` contains all92 captured build inputs; the receipt binds their
source identity, prospective card, settings and binaries. The original source
revision predates the new experiment source commit, so captured bytes and the
source manifest identify the implementation that actually ran.

The copy includes the published report/summary, all producer JSON reports and
512-update traces, fixed-reader reports and arithmetic audits, observation-only
information reports, environment/engineering receipts and the full artifact
manifest. These reports remain unchanged; interpretation and panel metadata live
in the parent `../configurable_capacity_v1.json` and `../../CONFIGURABLE_CAPACITY_V1.md`.

The815-entry `artifacts.sha256` addresses the original full capsule. Its relative
paths refer to that capsule, not this evidence subset. Checkpoints, fitted tensor
archives, observed inputs, decision witnesses and reconstruction query arrays are
retained locally in the ignored capsule and are not in Git. To verify every entry,
run `sha256sum -c artifacts.sha256` from the original capsule inside the managed
container. The post-completion verification passed all815 entries.

`summary.json`'s zero encoder-call count refers to saved-only publication. Training,
checkpoint restoration, native extraction and original queries ran on CUDA in
the producer; no encoder operation was replayed on CPU.
