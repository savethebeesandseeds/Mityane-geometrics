# Sweep, diff and retained-record verification

20 October 2026. Engineering delivery `HX06-TOOLSET 0.4.0` adds CLI and public
Node API capabilities above the unchanged OR06-PCR 0.1.0 comparison. No Hero
JSON, operation contract or C++ implementation was changed.

## Delivered surface and limits

- `sweep` / `operations.sweep`: bounded Cartesian plans with explicit scalar
  axes and hold declarations; optional complete precomputed catalogs package
  actual records through existing `prepare` and `check`. Planning support labels
  stay separate from native pair states. Missing records are not fabricated.
- `diff` / `operations.diff`: independently verified retained source bundles,
  canonical experimental-roster identity, separate pipeline provenance, explicit
  tolerances, all states/reasons and success/success numerical differences only.
- `verify-study` / `operations.verifyStudy`: reviewed final-hash anchors for two
  exposed retained studies, complete source/seal/manifest integrity and current
  checks of recorded executable pairs. It executes no fitting, model, generator,
  old study runner, auditor or bootstrap. Archived and recomputed metrics have
  separate origins; missing upstream records stay explicit and non-executable.

The [guide](../TOOL.md#sweep-a-scenario-space), schemas in `describe`, static engineering
space/catalog examples and [surface map](../TOOL.md#public-calling-surface) document CLI/Node parity,
outputs, exit codes, support boundaries and scientific limits. The admitted
claim remains solely `controlled_model_response`; the other two authored
profiles appear only in a separate non-callable descriptor note.

## Regression and focused verification

Actual managed-container entry:

```text
bash code/scripts/task.sh hx06-tool-ergonomics-v1 test-operation-tools test-counterfactuals
```

**49/49 tool tests passed** (28 existing plus 21 new), followed by **47/47 C++
conformance cases**. The retained [full log](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/full-tests.log)
records the command and outcomes. Tool tests took 548.407 seconds, primarily
including artifact I/O on the Windows bind mount; this is no performance claim.

New coverage includes complete planning and recorded grids, deterministic decimal
ranges, overflow/size bounds, hold/mask violations, incomplete catalogs, actual
support abstentions, upstream failure, public Node and CLI sweep/check/run/diff,
distinct provider widths on common raw inputs, ordered-roster/support/mask/axis
mismatches, reason/state alignment, unresolved outcomes, finite tolerances,
tampering, unsupported claims and original-admission integrity. Existing commands
and the generic non-pair JavaScript adapter continue to pass their tests.

The first focused run passed 17/18 checks; its sole failure was missing `$schema`
metadata in the verification descriptor. The [initial log](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/focused-tests.log)
is preserved. Complete schema metadata was added before the final passing run.
Read-only review also identified later unbound reads in retained verification.
The fix keeps original admissions and hashes for every consumed buffer, final
and manifest parsing, bridge-source bytes and scratch request/artifact records.
Regression tests reject changed-and-restored records even when their content
hash is restored, and archives are bound to the exact records checked. Output
manifest pins derive from the expected published buffers rather than adopting
later file contents.

## Actual retained-study verification

Original study `OR06-SYNTH-RESP-001 1.0.0` completed verification through the
Linux CLI, with all 24 seed/split/arm partitions retained:

| Source/current-check roster, both splits | Requested | Success | Abstained | Failed | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original retained study | 27,648 | 23,040 | 4,608 | 0 | 0 |

Every source row had complete upstream records and was checked. Original MAE
remains a verified archived audit statistic because no per-pair oracle truth
was retained. It is not a recomputed scientific score. The actual
[verification report](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/verify-original-001/verification.json)
has SHA256 `88dbd112fb8a17cbbc25f07b1fafb4e360e34c6f94b50e03b37b4449b7ee2e76`.

Contribution study `OR06-REP-CONTRIB-001 1.0.0` also completed through the Windows
launcher, with all 90 seed/split/axis/arm partitions retained:

| Contribution roster, both splits | Requested | Success | Abstained | Failed | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source records | 95,040 | 79,200 | 15,840 | 0 | 0 |
| Current executable checks | 79,200 | 79,200 | 0 | 0 | 0 |

All 15,840 support-stress rows lack retained upstream values and remain explicit
retained-only abstentions with null contrasts. They are not represented as
currently executed refusal checks. The actual
[contribution verification](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/verify-contribution-001/verification.json)
has SHA256 `88a34ba013b56980cdf99c529e256adfa0f2297f8f6c131a33eacf9ab45083aa`.

An independent [output check](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/retained-output-check.json)
rehashed both complete output manifests (26 original and 92 contribution
members), confirmed exact inventories and compared all 30 recomputed saved-truth
MAEs across both contribution splits against the pinned audit. Maximum absolute
difference was `1.0547118733938987e-15`, below `1e-12`. No fit, bootstrap,
generation or fresh assessment occurred. Original MAE and all historical
uncertainty remain explicitly archived statistics.

## Windows orchestration and protected files

Actual Windows PowerShell invocations passed for recorded `sweep` and `diff`.
The sweep packaged four cells and produced three successes, one actual support
abstention, zero failures and zero unresolved pairs. Diff retained all four rows,
compared the three successes and kept the abstention's comparison null. Its
experimental fingerprint was
`f819fb806e379a41eb5bfd5bce6cb0345a8fee48ef245c5517953ddc07501e9c`.
The [sweep receipt](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/windows-sweep-command.json)
and [diff receipt](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/windows-diff-command.json)
also exercise new request/space/catalog/left/right path forwarding.

The before/after [protected-file check](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/protected-after.json)
confirms all 22 hashes unchanged: contract, numerical header/source, native tool
adapter and all 18 Hero JSON files. Outputs remain under ignored `.temp/`;
scratch/build files remain inside `/opt/mityane-geometrics`. The existing managed
container, pinned Node/compiler and standard libraries were reused. No dependency,
environment, image, service or SDK change was needed.

The managed [layout/link check](../../../../.temp/operations/counterfactuals/tool-ergonomics-v1/layout-check.log)
passed: 6,424 preserved files, 1,626 relocations, 40 later source retirements,
1,280 maintained Markdown links and 71 literal shell roots. Historical artifact
trees remain ignored. The delivery is complete within the requested scope;
there is no remaining blocker.
