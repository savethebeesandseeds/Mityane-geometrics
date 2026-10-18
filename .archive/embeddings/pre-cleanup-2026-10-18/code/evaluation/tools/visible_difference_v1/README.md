# Visible-difference evidence publishing

These tools publish and register a completed RPB-v13 experiment from saved
metadata. They do not train, run an encoder, fit a head or fit PCA.

The training driver, prospective card, admission scripts and independent saved
arithmetic reader live in
[`../../protocols/visible_difference_v1/`](../../protocols/visible_difference_v1/).
Run project commands inside the existing managed development container through
the root `container.ps1` launcher. Build and evaluate with its additive Makefile
and the named `rpb-visible-difference` task session.

The experiment needs the protected local TEMPO-3 capsule `structured-hard-timing-xrZMAS`.
It captures the declared 130 retained roles and their original hashes before
candidate training. It reuses saved baseline/control metadata and fitted heads;
it does not repeat their encoder training or inference. A missing parent capsule
requires restoring the recorded artifacts, rather than silently regenerating
the comparison.

After the independent audit passes, run `publish_results.py` with three explicit
metadata roles: the capsule's `results/report.json`, the passed `validation.json`,
and the capsule's `parent-metadata/parent-summary.json`. Supply each role's SHA256,
the reviewed publisher source SHA256, and exclusive summary/report/proof outputs.
Then run `register_results.py --capsule <capsule> --validation <validation.json>`.
Publishing writes the standard tables and a durable JSON record.
Registration adds RPB-v13 and the saved-TRAIN diagnosis while preserving every
historical version and instance-group object, the original v7 weights and the
formal v4 reference. Neither operation promotes an encoder.

The completed `visible-difference-wnUUCa` capsule retains its original sealed
reader and failed audit. Its passed audit uses the additive `reader-v2/`:
the query report's artifact field is a filename, while the original call passed
a full path. The repaired reader proves exact reversal to the original source,
keeps all numerical functions and tolerances unchanged, and records both source
identities. See the dated continuation for its exact pins. Preserve these measured
sources and failures; use the corrected artifact convention prospectively in a
new protocol rather than changing the captured reader or inventory.
