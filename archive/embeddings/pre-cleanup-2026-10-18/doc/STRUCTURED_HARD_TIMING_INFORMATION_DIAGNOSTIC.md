# TEMPO-3 data-only information admissions

The separately specified cross-feature v2 information rule passed all intact and deleted-view gates on two fresh engineering cohorts. The original same-feature v1 failure remains preserved. Both attempts used the unchanged TEMPO-3 generator; neither fitted an encoder or classifier head.

The designed complexity is an ordinal preassigned challenge level, not an accuracy-derived score. Information admission establishes the declared observed-only rule and support, not encoder quality or Bayes optimality.

## Cross-feature v2: passed

Dataset: **TEMPO-3** · timing · **complexity 4/5** · variable delay, positive gains/offsets and 3-tick channel gaps · views: intact and extra 30% coordinate deletion.

| Engineering master | View | Valid / total | Analytic conditional accuracy % | Coverage % | Full-population correctness % | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 82585 | deleted | 1020/1024 | 100.00 | 99.6094 | 99.6094 | PASS |
| 82585 | intact | 1024/1024 | 100.00 | 100.0000 | 100.0000 | PASS |
| 83686 | deleted | 1024/1024 | 100.00 | 100.0000 | 100.0000 | PASS |
| 83686 | intact | 1024/1024 | 100.00 | 100.0000 | 100.0000 | PASS |

Dataset: **TEMPO-3** · timing · **complexity 4/5** · unchanged structured recipe · equal means across the two fresh engineering masters.

| View | Mean analytic conditional accuracy % | Mean coverage % | Mean full-population correctness % |
| --- | ---: | ---: | ---: |
| intact | 100.00 | 100.0000 | 100.0000 |
| deleted | 100.00 | 99.8047 | 99.8047 |

Each engineering cohort contains 512 TRAIN and 512 separate VALIDATION source pairs, yielding 1,024 scoring rows per view. The intact accuracy/coverage thresholds remain ≥99%/99%; deleted thresholds remain ≥98%/95%, checked independently for each master. Deleted support is 1,020 and 1,024 rows, exceeding the required 973. Every supported row was correct.

The v2 rule takes legal three-time differences independently within each channel, using the same eligible feature set for that channel’s left and right differences. Its determinant equals an average over legal cross-channel feature combinations. Positive gains and constant offsets preserve the noiseless sign. The four-centre support minimum and all numeric gates are unchanged; labels are used only for scoring.

The two attempts use different fresh engineering seeds. Their coverage numbers compare declared diagnostics on separate cohorts, not a paired improvement estimate. No generator, encoder, head or gate was tuned from model quality.

Evidence: [passed information record](../output/runs/rpb-structured-hard-timing/information-admission/info-AGAb6s/information-record.json), [v2 prospective card](../code/evaluation/cards/structured_hard_timing_comparison_v2.md), [v2 durable metadata](results/structured_hard_timing_information_v2.json). Receipt SHA-256: `6c5f295c7c3956d5ede11a8f1bc28a8402031ee1946c769f876ce9e457aee8b8`; source closure: `00a20994860bf361d47679a2f9eb7dd765c17e36250865e96d0192015a78666b`.

## Same-feature v1: preserved failure

The earlier rule failed deleted coverage at 950/1,024 and 900/1,024 rows despite 100% conditional accuracy. That failure remains valid and is not replaced by v2. See the [byte-preserved failed report](STRUCTURED_HARD_TIMING_INFORMATION_V1_FAILED.md) and [unchanged v1 durable metadata](results/structured_hard_timing_information_v1.json).

No encoder quality result is reported here. Report extension used bound JSON metadata only, with zero tensor archives, model/head fitting or audit reruns.
