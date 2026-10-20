# Python working-copy removal, 18 October 2026

The owner authorized deleting the replaced Python helpers after the full
migration. All 40 obsolete working files are removed. The 40 scientific Python
files remain because they provide independent saved-tensor decoding, numerical
audits, TRAIN analysis, scientific contracts and their adversarial fixtures.

The [closed removal manifest](PYTHON_MIGRATION_REMOVAL_2026-10-18.json) records
each deleted path, baseline/source hash, byte count, Node replacement and exact
preserved source identity. Four exact copies already existed in the frozen
embedding snapshot; 36 distinct source versions are archived once under
`.archive/python-migration/retired-2026-10-18/`. No replaced helper remains in
maintained `code/` or `web/` source. Scientific APIs in mixed modules remain
captured where actually required.

Current preparation, protocol and operation build/source closures refer to the
Node implementations and their actual dependencies. The layout verifier checks
declared retirement separately from the immutable original relocation map.
Regression checks require deleted paths to remain absent, exact archived bytes
to remain available, and the complete maintained Python inventory to equal the
40 explicitly retained scientific files.

Validation uses the existing managed `mityane-geometrics` container, immutable ID
`381c6b78f4fdb301cd132050ffd8095587c0794829a7a73a547fdbc0484630e1`,
with the existing pinned image and Node runtime. Actual Git metadata is HEAD
`6c0a4353c8b7a86eb9ae575ef4a3b8f67bb8c5ab`, dirty worktree. No image,
package, SDK or container lifecycle change is needed.

| Check after deletion | Result |
| --- | --- |
| Bash and Node syntax | 77 Bash and 78 Node files passed |
| Shared artifacts, preparation, protocol and removal fixtures | 109 Node tests passed |
| Retained scientific adversarial fixtures | 105 tests passed in each of ordinary and optimized Python modes; 210 executions |
| Strict visible-reader SOURCE fixtures | 76,295 checks and 69 negative fixtures passed in each mode; 14 self-tests per mode |
| Strict structured-reader SOURCE fixtures | 12,475 checks and 27 negative fixtures passed in each mode |
| Current Make source closures | All 24 passed; no retired Python dependency, missing source or omitted production Node import |
| Maintenance regression | 69 tests passed, including retirement rejection cases and the actual Make closures |
| Artwork preparation | All 5 tests passed |
| Offline dependency setup | 321 exact installed packages, zero installs/downloads; all 8,821 SDK files verified with zero mutations |
| Frozen source preservation and layout | 6,424 frozen files unchanged, exact inventory, 1,626 relocations and all 40 retirements verified; 790 maintained links and 70 shell roots passed |
| C++ producer/reader compile checks | All 30 translation units compiled and ELF objects verified; nine frozen shared objects unchanged |
| Final dashboard refresh and freshness | 34 tests passed on both refresh and freshness check |

The artificial strict-reader receipts are new post-cleanup SOURCE fixtures,
not a historical research seal. Raw logs, Make invocation/output, reader fixture
sources and receipts, and the 1,771-file current source snapshot are preserved
under ignored `.temp/maintenance/language-migration-removal-2026-10-18/`.
The JSON manifest binds the final logs, compile receipt, source snapshot and integrity check. The compiler log retains all inherited nonfatal warnings; compilation had no errors.

The [earlier verification](PYTHON_MIGRATION_VERIFICATION_2026-10-18.md) remains
dated evidence from before deletion; its source snapshot is not the current
post-cleanup source identity.

This cleanup does not change frozen archive bytes, saved research evidence,
datasets, checkpoints, SDK contents or image packages. It performs engineering
fixtures and compile checks only; paused research and V14 metadata work remain
paused. No historical numerical or serialized-byte equivalence is inferred.
