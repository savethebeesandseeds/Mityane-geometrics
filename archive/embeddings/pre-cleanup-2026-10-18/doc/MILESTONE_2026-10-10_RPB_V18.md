# RPB-v18 milestone, 10 October 2026

RPB-v18 removes the weak linear-head result seen in the fresh v17 group on
TEMPO-3. Save this architecture and every measured instance separately from
[RPB-v17](MILESTONE_2026-10-10_RPB_V17.md). It is the strongest measured
TEMPO-3 timing design so far, with limits stated below.

RPB-v17.alt-01 serves20 learned waveform-shape numbers and12 fixed generic
signed timing numbers. RPB-v18 keeps that layout and normalizes each channel
pair's four timing numbers to unit length inside the encoder. It preserves
their signs and relative spacing while removing strength variation. No new
weights or classifier changes:226,877 total parameters,226,445 trainable and
432 frozen. Waveform Huber1 still trains through the sole native32 decoder.

Five independent source cohorts: masters80787/81888/82989/84090/85191,
TRAIN256/128 source pairs and VALIDATION128/64 source pairs per cohort.
V18 first screens80787/84090, deliberately including v17's weak linear case.
Only after its prospective gate passes does it run the remaining three.
This is **known-data development**, not a fresh confirmation or TEST result.
V17 scores below reuse its saved fresh-confirmation metadata; no reruns.

CUDA RTX A2000 training: context deletion .15, B8,512 unskipped updates,
five trajectories per design. Fixed heads: Ridge penalty1; neural tanh16,
Adam .01/100 updates; all three declared repetitions. Each TRAIN-fitted head
scores both views. No PCA after an encoder. Size32 heads have66/562 fitted
parameters. The full source, initial0 and trained512 results are preserved.

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable periods/small
signed delays, gains/offsets, natural10% missingness and three-tick channel
gaps · intact VALIDATION · five-cohort means at512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 100.00 | 100.00 | 100 |
| RPB-v18 | 32 | 100.00 | 100.00 | 100 |

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable periods/small
signed delays, gains/offsets and channel gaps · VALIDATION with extra30%
coordinate deletion · five-cohort means at512.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 32 | 82.19 | 99.90 | 100 |
| RPB-v18 | 32 | 100.00 | 99.95 | 100 |

The deleted Linear gain is17.8125 percentage points. Worst-cohort v18 deleted
Linear/Neural is100%/99.739583%; the matching v17 group is70.3125%/99.479167%.
Both designs retain100% coverage. Individual head repetitions and source-group
conditional uncertainty remain in the durable JSON.

Dataset: **TEMPO-3** · timing · **complexity4/5** · variable periods/small
signed delays, gains/offsets and channel gaps · fixed masked TRAIN/intact
VALIDATION queries · standardized mean absolute error, five-cohort means.

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v17.alt-01 | 512 | 0.705814 | 0.731698 | 21.82 |
| RPB-v18 | 512 | 0.692694 | 0.719535 | 21.89 |

Time is synchronized training-loop wall, including preparation/transfers and
excluding heads, extraction, queries and writing. It is descriptive. Full
512-entry optimization traces retain masked waveform Huber1 and gradients;
the table uses fixed-query standardized MAE, not the last minibatch loss.

The **untrained0 v18** already gives100%/100% intact and100%/99.895833%
deleted after its fixed heads fit TRAIN. Credit the generic architectural
prior for timing success. The learned shape and reconstruction route remain
useful research targets; near-perfect timing alone does not establish a
general representation. Unit normalization can amplify weak noisy relation
pairs and removes magnitude/confidence. It has only been measured on this
bounded signal family. Reconstruction remains a separate limitation.

The [full report](../code/encoders/raw_patch_bottleneck_mae/UNIT_TEMPORAL_RELATION_SCREEN.md),
[durable summary](results/unit_relation_screen_v1.json),
[prospective card](../code/evaluation/cards/unit_relation_screen_v1.md) and
[continuation](CONTINUATION_2026-10-10_AFTER_FRESH_V17.md) preserve the result.
The exclusive capsule is `output/runs/rpb-unit-relation-screen/admission-DiR2yH`:
source snapshots, card/SDK/input hashes, CUDA admission, initial/trained native
features, scalers, checkpoints, optimizer/RNG, fitted heads, queries, traces,
screen gate receipt and artifact inventory. Checkpoints stay in the local
ignored run directory; compact source/results/notes are versioned in Git.

- Compiled59-file SOURCE fingerprint: `d82d92864c87abb985fd0b1153927bcded70904b54e9c7ba74b0101cbe850993`.
- Card SHA256: `906b5ab75da3c12d243ebe915316c1f9cb4a46727f32e1896219ff66305997b9`.
- Final summary SHA256: `48d8beaa9fa0bdfd9440f291d0c8a5046f0dc9c747c7f03475a92ccdf15cc5ba`.
- Final report SHA256: `f0e5d6b9b284279e4a55eeb886ca4e9b5c34459a022a8bdfdea21a1442575962`.
- Capsule inventory SHA256: `30a4ae4a463c947c3ea49dcfd53fa1b11e167fef955a5577942e89518b445b4b`.

CUDA primitive and runner admissions passed. Saved CPU checks passed5,833,018
checks across180 witness archives and30,720 exact fixed relation values.
Independent final metadata review passed3,947 comparisons, including all means,
worst cohorts, initial controls, gates, source pins and historical registry
preservation. Final source guards and Python/shell syntax checks also passed.
CPU only fits the fixed heads or verifies saved arithmetic; it never reruns
the encoder or optimizer. No old model/head refits, post-encoder PCA, TEST,
stress runs, production promotion or default changes. All18 prior designs and
17 prior instance bundles remain unchanged; v18 is an additive new design.

Next: freeze and admit the coherent two-component challenge from
[this proposal](PROPOSED_TEMPO4_CHALLENGE.md). TEMPO-4 tests the slow component's
lead/lag, while separate AMP-2 tests component balance; both are proposed5/5
and unmeasured. Preserve v18's prior/initial control and screen one separately
specified frequency-sensitive candidate on two cohorts per task. Check that
legal observations carry the answer before quality and retain all failures.
Expand only a useful design; require gain over its own initial control before
crediting learned shape. Original saved v7, formal v4 and TEMPO-1 v10.alt-03
remain intact.
