# Remaining Python migration: implementation boundaries

The owner requested a plan for the remaining migration after the
[maintenance tranche](PYTHON_MIGRATION_2026-10-18.md). The current work packages,
priorities, dependencies, states and acceptance criteria are in the
[`foundations` Hero](../web/dashboard/heroes/14-foundations.json), under
`foundations-python-migration`. This document records the inspected surface,
ownership decisions and compatibility rules supporting that plan. It adds no
experimental prerequisites or permission to resume paused research.

## Validation of this planning change

The inventory was parsed and its 80 Python-source hashes and 59 caller-source
hashes were cross-checked against the inspected files. The six planned
milestones received an independent source review. These are planning checks,
not executed migration or protocol parity tests.

Docker was unavailable when dashboard refresh was attempted. At the owner's
direction, subsequent work is limited to file edits while Docker is recovered;
no further Docker commands, builds, tests or runtime checks are to run until
execution is permitted again. Dashboard refresh/freshness and container-side
validation of this change remain pending. The Hero JSON holds the new plan;
the generated standalone HTML does not yet include it.

## Target and inventory

This is the original remainder design. The owner's later instruction permits
Python where it is better suited. The
[complete implementation record](PYTHON_MIGRATION_IMPLEMENTATION_2026-10-18.md)
and [per-file disposition](PYTHON_MIGRATION_DISPOSITION_2026-10-18.json) supersede
the unconditional Python-free target below. Runtime verification is still deferred.

The original target was Python-free **maintained execution** using the existing Node/C++
stack. Historical sources, seals, failed readers, receipts and original commands
remain identifiable. Removing Python packages from the pinned image, erasing
historical `.py` files and reimplementing retired publication are outside this
target.

The [source inventory](PYTHON_MIGRATION_INVENTORY_2026-10-18.json) records every
inspected Python file, source SHA256, proposed responsibility/language, caller
location, inline-block boundary and the four already replaced references.
Counts are physical source lines, including comments and blanks; static call
sites do not establish that a workflow currently executes successfully.

The subsequent [authored-source record](PYTHON_MIGRATION_SOURCE_2026-10-18.md)
lists the new preparation, artifact, capsule/gate and C++ foundation files. Those
sources have not been executed or cut over; this inventory remains the dated
before-port baseline.

| Inspected surface | Files / blocks | Physical lines |
| --- | ---: | ---: |
| Functional migration candidates | 70 Python files | 19,562 |
| Already retired publication entry points | 6 Python files | 795 |
| Retained originals from the maintenance tranche | 4 Python files | 1,002 |
| Python embedded in shell runners | 34 blocks in 18 files | 658 |

The 76 not-yet-replaced Python files comprise 66 files across 15 evaluation
families and 10 experiment-input helpers. There are 161 static Python command
lines across 59 shell/Make files, including calls to retired commands. Sixty-eight
Python files have Python shebangs; no additional Python-shebang file with another
extension was found. `.archive/`, `.temp/` and historical ignored output/build/
dependency trees are excluded.

The six retired entry points remain refusals: unit `record_screen`, fresh
confirmation `record_confirmation`, partitioned `record_versions`, and the
two-component, multiband and spectral `publish` commands. Their Python launch
dependency can be replaced by a Bash/Node refusal. Their historical registry or
global-summary writes must not be restored. Capsule-local summaries in other
commands remain part of their owning protocol's migration.

## Ownership and implementation choices

| Responsibility | Destination | Boundary |
| --- | --- | --- |
| Generic admitted-file reads, hashes, exclusive writes and evidence manifests | New versioned Node utilities in `code/shared/artifacts/` | Preserve existing sidecar/identity APIs; protocol adapters supply policy |
| Model-specific input/capsule preparation | `code/embeddings/scripts/` Node adapters | Keep exact roles, splits, parent/source bindings and admission semantics |
| Generic restricted saved-tensor decoding | New versioned C++ module in `code/shared/artifacts/` | Closed supported format, no arbitrary pickle execution or model loading |
| Independent saved CPU arithmetic and embedding protocol validation | `code/embeddings/evaluation/` C++ | No linkage to producer/model arithmetic as an independent oracle |
| Per-protocol gates, evidence association and report rendering | Owning evaluation protocol adapters in Node/C++ | Numerical checks in C++; explicit policy and output behavior retained per family |
| Launchers and root build/tests | Existing Bash/PowerShell, root Makefile and task runner | Container-only execution, shared pinned Node, no additional environment definitions |

Existing generic utilities should be reused only when they preserve the
required behavior. Shared mechanisms must not absorb seeds, population matrices,
protection floors, split rules or source/card pins into an implicit global
policy. Helpers currently imported by mutating Python module globals should
become adapters with explicit immutable configuration.

## Reference identities and saved-artifact availability

The inventory records an existing identity divergence: the maintained
`visible_difference_v1/reader-v2/archive_codec.py` is SHA256
`55fd43b7b2a869647c5788add733fd2a7f8f5c11e9a30ba08c813b3bbd4471d8`,
whereas the unchanged archived copy is
`4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d`.
Historical consumer pins match the archived bytes. Many current scripts also
retain `/embedding` and `output/runs` bindings. A relocated Python file is
therefore not automatically a runnable reference under its old seal.

The baseline work must associate current paths, exact historical source bytes,
protocol cards, seals, fixtures and permitted saved witnesses. New readers and
wrappers receive new implementation/source identities and separate equivalence
receipts; old pins are never replaced by a new hash to make a check pass.
The reader-v2 repair record is explicitly SOURCE-only, with zero actual archive
or capsule reads. Preserve that distinction from an executed saved-data audit.

Saved-artifact availability and permission have **not been inspected** by this
planning audit. Record them per family before reading real payloads. Missing or
restricted witnesses defer that family's saved-parity certification and cutover;
they do not prevent artificial fixture work or unrelated families. Historical
inputs must not be regenerated by repeating completed training or head fitting.

## Protocol dependency structure

| Group | Families and inspected files | Dependency to preserve |
| --- | --- | --- |
| Foundational contracts | `visible_difference_v1` (11), `structured_hard_timing_v1` (5) | Restricted decoder, saved arithmetic, legacy contracts; retain both visible-difference reader generations and repair provenance |
| Screens and confirmations | `architecture_screen_v1` (4), `fixed_prior_relation_screen_v1` (3), `partitioned_relation_screen_v1` (4), `unit_relation_screen_v1` (5), `fixed_prior_fresh_confirmation_v1` (5), `two_component_information_v1` (3), `multiband_screen_v1` (3), `spectral_confirmation_v1` (3) | Foundational saved reader; some publishers only render metadata, others also verify numeric witnesses; retirement rules above still apply |
| Branch diagnostic | `configurable_branch_v1` (3) | Saved normalization, fit/prediction witnesses and population arithmetic; `publish` is substantially numerical |
| Capacity/scaler evidence | `configurable_capacity_v1` (5) | Branch arithmetic plus independent query/scaler/checkpoint contracts and capsule archive admission |
| Relation families | `configurable_learned_relation_v1` (5), `configurable_content_relation_v1` (5) | Branch and capacity/scaler helpers plus each family's own gate and typed metadata |
| Restricted TRAIN diagnostic | `learned_train_v1` (2) | Lazy selected-field decoding and saved arithmetic; selection/alias rejection before storage reads |

Three maintained directories contain identical copies of `archive_codec.py`,
`saved_cpu_math.py` and `legacy_implementation.py`: 3,834 lines are three copies
of a 1,278-line foundation. Consolidate the new implementation where semantics
are equal; preserve historical copies and keep differing fixtures/validators
separate. Screen migration need not hold up independent branch/capacity work
once their actual shared dependencies are available.

The input-helper surface has two standalone binders and a shared fresh-decoder
base with seven adapters. The standalone binders declare exactly 195 TRAIN
files/15 instances and 85 TRAIN/VALIDATION roles respectively. Source/capsule
bookkeeping can be ported independently; adapters that capture a sealed reader
also need that replacement reader's accepted identity before cutover.

## Compatibility principles

- Preserve the exact role matrix, complete admission-before-payload-read rules,
  symlink/hardlink/alias handling, input change detection, exclusive completion,
  durable evidence and output-read inventory. New destinations use `.temp/` and
  new path-bound identities; historical records keep their recorded paths.
- Match storage dtype, little-endian bytes, signed int64 IDs, offsets/strides,
  masks, UTF-8 metadata, empty/scalar cases and permitted device tags by artifact
  role. A checkpoint role and a saved CPU feature role need different admission.
- Keep uint64 wraparound, FNV/stream seeds, MT19937-64 draws, rejection-sampling
  indices and bootstrap/quantile order exact. Do not substitute a standard
  distribution implementation or producer code merely because outputs look close.
- Preserve float32 rounding and cancellation-sensitive reductions, frozen
  tolerances, ties, support/coverage, undefined values, TRAIN-only maps, scale
  floors and gate boundary slack. Boolean admission/gate verdicts, discrete
  predictions, tie decisions and failure outcomes must match exactly even when
  associated numeric values are within tolerance. Never widen tolerances or
  permit a changed verdict to pass the port.
- Specify JSON/text compatibility per protocol, including large integers,
  duplicate keys, finite-number rules, Unicode, key order and newline/number
  formatting wherever bytes are hashed. Python's plain `json.loads` accepts
  duplicate keys; stricter rejection is a separately identified behavior change.
- Preserve per-cohort/per-repetition policy, negative/unsupported outcomes,
  protection floors, no automatic promotion and exclusive report publication.
  Consolidating implementations does not merge protocol verdicts or reopen
  retired publication destinations.

Artificial fixture corpora should include malformed/truncated storage, forbidden
pickle globals, disallowed device tags, poisoned unselected data, storage aliases,
missing/extra roles, stale hashes, source changes, changed masks/cohorts and exact
threshold boundaries. Record exact goldens before comparing replacement outputs.
Available permitted saved witnesses then establish the narrower declared parity
scope; synthetic success alone does not certify all historical protocols.

Final integration must inspect direct, absolute, versioned and shebang Python
launches and transitive subprocesses, in addition to running with Python commands
blocked. Use fixture-only launch paths, inert producers and already permitted
saved checks. Migration verification does not authorize CUDA training, fitting,
new TEST/stress access, V14 metadata application or experimental reproduction.
Retained Python references must be hash-bound and excluded from maintained
execution explicitly; an unported active family remains incomplete.
