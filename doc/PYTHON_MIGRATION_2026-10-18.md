# Python maintenance migration — 18 October 2026

The owner requested the incremental C++/Node migration after reviewing the
repository's Python footprint. This first change replaces active maintenance
entry points. It does not resume encoder research or change model, experiment,
evaluation, checkpoint or historical source identities.

The later [complete implementation record](PYTHON_MIGRATION_IMPLEMENTATION_2026-10-18.md)
documents the owner's revised language policy, all remaining source cutovers and
the scientific Python responsibilities. Its unexecuted changes do not extend
the tested evidence recorded for this maintenance tranche.

## Maintained execution

| Responsibility | Current entry | Preserved Python reference |
| --- | --- | --- |
| Read-only installed SDK verification | `code/scripts/verify-libtorch.mjs` via root `setup.sh` | `code/scripts/install-libtorch.py` |
| Snapshot, relocation, ignored-tree, link and shell-root checks | `code/scripts/check-layout.mjs` | `code/scripts/check-layout.py` |
| Ordered length-prefixed direct-control source fingerprint | `code/operations/forecasting/direct_embedding_control/versions/v1/source_identity.mjs` via `v1.mk` | Adjacent `source_identity.py` |
| Dashboard image preparation | `web/dashboard/scripts/prepare-branding.mjs` and `prepare-branding-native.cpp` via refresh | Adjacent `prepare-branding.py` |

Original Python files are retained for historical commands, source closures and
migration comparison. In particular, historical environment provenance and
protocols name `install-libtorch.py`, and the frozen relocation record names
`source_identity.py`. New entry points do not silently substitute new source
bytes into those old identities. The complete `.archive/` snapshot stays
byte-preserved; dated `archive/...` paths use the documented root mapping.

## Runtime and behavior

Shared `code/scripts/node.sh` executes the already available pinned Linux Node
22.23.3. Root dependency setup verifies the existing exact cached binary/archive
and full upstream license. It performs no runtime download, SDK copy, compiler
build, test or service startup. The image and Debian package lock retain their
existing contents. No npm packages are introduced.

The C++ artwork helper dynamically opens the same GdkPixbuf, GObject and libwebp
libraries used by Python's former ctypes entry. Its declarations cover only the
used C ABI; there is no new SDK or dependency installation. The root Makefile
owns its build under `/opt/mityane-geometrics/web/build/`; Node owns assignment
validation, hashing, report serialization and freshness checks.

The replacements preserve saved-result roles and read-only verification. They
do not call the encoder/model implementation to establish research agreement.
The source fingerprint keeps the ordered UTF-8 fields and little-endian 64-bit
length prefixes; an equivalent-looking JSON hash would be a different identity.

## Verification

All execution is through `container.ps1` in the existing managed Linux container.
Focused fixtures use container-side temporary directories. The persistent test
suite uses Node/C++ and does not invoke Python; Python is used only for the
one-time migration comparisons against the retained implementations.

Verified outcomes:

- All 42 maintenance fixtures and 5 artwork fixtures pass with `python` and
  `python3` commands blocked in `PATH`. Persistent tests invoke neither runtime.
- Root setup passes with 321 exact installed Debian packages, zero cached APT
  installs and the reused pinned Node runtime. It verifies the inherited SDK,
  runtime library resolution, CUDA toolkit and visible GPU without SDK mutation.
- Python and Node SDK verification return the same JSON values for 8,821 files
  and 4,633,956,070 bytes. The proof SHA256 is
  `e370b91a50e264739c284c0fc5b461436b60a50735b36a2777750a6603b3a0bb`;
  the inventory SHA256 is
  `c76492a34792d1449ac2a71903e8610d444de801b78064321c9ad89d6ca6a04d`.
- Original and replacement source fingerprints agree on
  `53fea1016df88727454ea823b6bfdbb20cb40d6d2b2a6f133b271d54b906b0ba`.
  Permanent fixtures include UTF-8, CRLF and NUL bytes and an independently
  recorded expected digest, plus missing/directory/escaping-source rejection.
  The ordinary root `test-direct-control-v1` build dry run also passes and
  embeds this same digest in its C++ compile command; no model build or research
  execution is needed for this check.
- Node verifies all 6,424 original archived files against their recorded hashes
  and exact inventory, 1,626 relocations, 787 maintained links and 70 literal
  shell roots. Original/replacement links-only checks return the same counts.
- All 19 WebPs and both artwork JSON files are byte-identical to the Python
  outputs, totaling 329,360 image bytes within the unchanged 524,288-byte budget.
- HTML link entity decoding matches all 2,231 CPython HTML5 names and the
  tested numeric/malformed boundaries. The static table's explicit source,
  revision, license and hashes are stored with `code/scripts/html5-entities.json`.
- The dashboard refresh passes all 34 focused checks, validates 18 Heroes,
  50 Goals and 62 Milestones, and regenerates the standalone HTML. Existing
  artwork and browser CSP behavior are preserved.

The root Makefile isolates maintenance-only targets from research source/Git
scans. Mixed/model builds retain their existing provenance rules. The global
artwork helper uses a shared build lock and atomic executable replacement.
Local one-time comparison receipts are under
`/opt/mityane-geometrics/setup/python-migration/`; integrated logs are under
`/opt/mityane-geometrics/maintenance/python-migration-20261018/`.

```powershell
.\container.ps1 -Action exec -Command @('bash', 'setup.sh', '--no-download')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'maintenance', 'test-maintenance', 'test-dashboard-branding', 'check-layout')
.\web\dashboard\scripts\refresh.ps1 -Check
```

## Remaining Python

The embedding preparation/evaluation protocols and Python fragments in research
shell scripts remain. Their saved archive decoding, independent CPU arithmetic,
admission conditions and protocol-specific outputs require separate equivalence
checks before replacement. Many retain historical `/embedding` bindings. Their
presence grants no authority to execute paused experiments or rewrite records.
This tranche changes active maintenance execution; it does not complete the
whole-repository migration.

The [remaining migration design and source inventory](PYTHON_MIGRATION_REMAINDER_2026-10-18.md)
support the owner-requested follow-on plan. Current work packages, dependencies,
states and acceptance criteria live in the foundations Hero's
`foundations-python-migration` Goal.
