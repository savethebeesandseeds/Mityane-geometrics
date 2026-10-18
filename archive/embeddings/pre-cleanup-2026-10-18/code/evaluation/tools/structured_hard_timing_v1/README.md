# TEMPO-3 metadata report formatter

`format_report.py` formats the single passed
`structured-hard-timing-comparison-v2` audit and its exact producer JSONs.
It also preserves the separate failed v1 and passed v2 data-only information
admissions. It reads five explicitly SHA-bound JSON roles, with whole path and
inode admission before hashing. It runs no tensor decoder, encoder, head, PCA,
bootstrap or audit. These tools sit outside the compiled producer source closure.

Run only in the existing managed container after the independent audit passes.
Supply the actual audit/report/complete paths and hashes from that completed
capsule. The audit must certify the two exact producer hashes. The information
roles are the durable `doc/results/structured_hard_timing_information_v1.json`
and `structured_hard_timing_information_v2.json`, each with its actual hash.
The output must be an absent direct child `emitted-*` under
`/embedding/output/runs/rpb-structured-hard-timing/report-tools`.

```sh
python3 -B code/evaluation/tools/structured_hard_timing_v1/format_report.py \
  --audit "$audit_json" --audit-sha256 "$audit_sha" \
  --experiment "$capsule/results/report.json" --experiment-sha256 "$report_sha" \
  --complete "$capsule/results/complete.json" --complete-sha256 "$complete_sha" \
  --information-v1 /embedding/doc/results/structured_hard_timing_information_v1.json \
  --information-v1-sha256 "$information_v1_sha" \
  --information-v2 /embedding/doc/results/structured_hard_timing_information_v2.json \
  --information-v2-sha256 "$information_v2_sha" \
  --output "$new_emitted_leaf"
```

The result is a Markdown report and a JSON summary containing every producer
cohort, full training trace, readout/paired interval and original independent
audit record. Every table has its own TEMPO-3 designed-complexity 4/5 legend.
The level is ordinal and preassigned, not an accuracy-derived score.
Means retain all five masters and all three fixed head repetitions. Unsupported
fits or zero-support views remain undefined; interval bounds are never averaged.
GPU loop seconds include CPU trace capture and are distinguished from the
other mixed transfer, verification and I/O scopes.

After independent metadata QA, copy the two emitted files byte for byte to the
new durable report and `doc/results/structured_hard_timing_comparison_v2.json`.
Preserve the emission and its SHA records. This workflow uses direct formatting,
metadata QA and an exclusive durable copy; no reader rerun or flags-release
ladder is required by this formatter.

SOURCE-only artificial fixtures can be run with `--self-test`; they create no
data, model or report. The prequality reviewed source is
`314ea309600fbc904a827d1dfa4f3e79ff083951589f414a44c60dad41e1462d`.
Its 75,516-check/18-negative fixture result is bound to a source differing only
in one legend punctuation correction, with an exact reverse-byte proof. No
measured metadata was used to prepare or test this formatter.

The completed xrZMAS report is
[`STRUCTURED_HARD_TIMING_DIAGNOSTIC.md`](../../../encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_DIAGNOSTIC.md),
with [all verified metadata and traces](../../../../doc/results/structured_hard_timing_comparison_v2.json).
`update_registry.py` is the administrative updater for that exact completed
capsule, not a quality runner. It appends the two alt-05 instance groups and
their diagnostic while asserting every old design, instance and historical
top-level object is unchanged. Its new training records keep only tag, updates,
mean errors and loop seconds; complete per-master traces stay in the linked
durable JSON. The latest pointer changes without replacing original saved v7
or the formal v4 reference. It performs no numerical evaluation.
