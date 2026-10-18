# Code review, 12 October 2026

The review corrected validation, immutable publication and test isolation
defects. No blocking defect was found in the inspected configurable encoder's
masking, scaling, training-data separation or CUDA execution. The complete
saved TEMPO-5 experiment passes the stronger audit: its numerical measurements
and generated Markdown report are exactly unchanged.

**No new quality experiment.** The review ran three small engineering models
for two CUDA updates each, two focused CUDA checkpoint checks, and an artificial
CPU classifier self-test. It performed zero quality encoder calls and zero
quality classifier fits. All historical model instances remain preserved.

## Findings and corrections

| Severity | Finding | Correction and regression evidence |
|---|---|---|
| P2 | Python `assert` disabled admission and completion checks under `python -O`. An incomplete 511-update fixture could be finalized. | Explicit runtime checks remain active normally and under optimization. Negative fixtures reject incomplete counters, altered receipts and inconsistent bindings; a valid fixture finishes once. |
| P2 | Quality launch trusted an admission filename; source capture could accept bytes changed between manifest capture and copying. | Check the actual three-profile CUDA admission receipt and frozen settings before quality generation. Bind every copied source to the initial manifest. Recheck original input hashes after the run. |
| P2 | Observation loading accepted undeclared archive fields and malformed packed source IDs. | Require exactly five legal observation fields and valid packed identities. Tests reject latent fields, odd IDs and wrong master metadata. Original quality inputs pass. |
| P2 | Observation and reader archive publication could replace a dangling symlink or a competing output. | Publish with no-replace filesystem operations. Regression fixtures preserve existing bytes and symlinks; two competing observation writers produce one complete winner. Historical shared publication semantics remain preserved. |
| P2 | Reconstruction audit could accept targets unrelated to original observations. A coordinated target/prediction shift of +0.25 evaded the old check. | Derive every target from its exact original observation and frozen TRAIN scaler, including exact float32 rounding. Check masks, source IDs, dataset, profile, point and provenance. The shifted fixture now fails. |
| P2 | Derived scoring artifacts were compared mainly with one another. | Bind scoring labels and source rows to original role archives, and native readout rows to saved CUDA exports. Future exports carry explicit per-view source IDs and CPU scaler witnesses. |
| P2 | Paired comparison intervals were published without independent replay. | Recompute all declared source-group differences from saved decisions, including direction, seed, population and interval bounds. Reject missing, duplicate and altered replay records. |
| P2 | Saved checkpoint review checked integer geometry but omitted floating settings and schema text. | Validate all six numeric settings, schema/tag/architecture/layout/device/dtype markers, semantic IDs, point and integer geometry. Mutation tests exercise each field and wrong numeric dtype. |
| P3 | Fixed temporary checkpoint-test filenames collided with interrupted or concurrent checks. | Use separate owned temporary directories with cleanup and same-basename isolation checks. Mirror the correction in `cuwacunu`; both focused CUDA tests pass. |

The reusable model arithmetic, configuration contract, signal law, classifier
recipes, masking rates and quality budgets are unchanged. Evaluation fixes remain
in `cuwacunu_embedding`; the reusable checkpoint-test correction is mirrored in
the clean `cuwacunu` integration repository. All 15 declared backport file pairs
match their SHA-256 values. `cuwacunu_torch` was not changed.

## Verification

The final build uses the managed `cuwacunu_embedding` container and separate
session `review-capacity-20261012`. Actual CUDA checks used the NVIDIA RTX A2000
8GB GPU. The completed quality capsule and its original build session were not
replaced.

- Generator/schema/information tests pass, including concurrent publication.
- All 11 capsule-state tests and 12 evaluation tests pass both normally and
  under `python -O`.
- Three engineering profiles pass two CUDA updates, finite gradients, changed
  weights, native export and exact checkpoint reload. CPU encoder calls: zero.
- The artificial reader self-test passes its three fitted repetitions and
  publication checks. These are engineering fixtures, not quality measurements.
- Saved-only replay passes 81 fitted classifier pairs, 243 prediction archives
  and 90 paired comparisons without fitting or encoder execution.
- Every one of the 36 original reconstruction archives and 18 checkpoint
  metadata bundles passes the stronger scaler/target/settings audit.
- Every original quality summary and training measurement is exactly equal to
  its preserved value. The regenerated report is byte-for-byte identical.
- All 815 entries in the original artifact manifest pass after the review.
  Original manifest SHA-256:
  `62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83`.

Original experiment source fingerprint:
`8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1`.
Corrected review-tool source fingerprint:
`a7b3929a358d2778f828a0ce6278ef5d54b95c30ea38adb29bccc74bb7253f9c`.
The corrected audit is separate evidence; it does not relabel the code that
produced the original measurements. Its 95-file source capture, binary hashes,
logs, engineering receipts and read-only regenerated tables are retained in
[the review evidence](results/configurable_capacity_review_2026_10_12).
The retained local engineering tensors remain in
`output/reviews/configurable-capacity-code-review-2026-10-12/review-24B3xP`.

## Results and continuation

The [agreed quality and training tables](results/configurable_capacity_review_2026_10_12/published-audit/report.md)
reuse the original nine CUDA trajectories. Dataset: **TEMPO-5**, timing,
designed complexity **5/5**, three streams, 128 candles and nine structured
features. Each cohort has 512 TRAIN examples from 256 paired sources and
256 VALIDATION examples from 128 paired sources; three independent cohorts,
batch 8, 512 updates, fixed ridge and tanh16 heads with three repetitions.
The preserved card is `configurable-capacity-v1`; this remains development
VALIDATION evidence, without TEST or market validation.

RPB-v20.alt-01 remains the compact W64/native32 continuation configuration.
Its intact Linear/Neural means remain 94.66%/94.31%, and additional-30%-deletion
means remain 90.10%/88.72%, with 100% coverage. This review supports the integrity
of those measurements; it does not establish additional model quality.

Continue with [the existing research direction](CONTINUATION_2026-10-12_AFTER_CONFIGURABLE_CAPACITY.md):
identify the usefulness of learned versus fixed coordinates, then predeclare
the longer-history and feature-specific challenges before market use. Preserve
the original fixed evaluations, keep new diagnostic head fits explicit, and
avoid changing width or classifier budgets merely to improve known scores.
The review leaves no known unresolved blocker in its inspected scope; it does
not exhaustively prove every future geometry, workload or market distribution.
