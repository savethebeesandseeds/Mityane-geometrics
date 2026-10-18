# Continuation after the pooled-context diagnostic

Date: 9 October 2026. Preserve the original working **RPB-v7** bundle on
masters 4404/5505/6606/7707/8808, its checkpoints/scalers/readouts and optional
decoder-calibration capsule. **RPB-v4** remains the active reference. Compact
early **RPB-v10** is an investigation candidate; it is not promoted and replaces
none of those saved instances. Alternative groups identify separate runs.

This window completed six bounded milestones with frozen recipes and independent
saved-arithmetic verification:

| Milestone | Saved record | Independent checks | Interpretation |
| --- | --- | ---: | --- |
| Saved TRAIN reliability | [Diagnostic](../code/encoders/raw_patch_bottleneck_mae/SAVED_NATIVE_RELIABILITY_DIAGNOSTIC.md) | 1,693,154 | Weak timing can already be weak on TRAIN; geometry is mixed. |
| Frozen amplitude transfer | [Diagnostic](../code/encoders/raw_patch_bottleneck_mae/FROZEN_AMPLITUDE_TRANSFER_DIAGNOSTIC.md) | 38,784,404 | Strong untrained intact controls limit credit assigned to training. |
| Early mixer at 512 | [Diagnostic](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_RELIABILITY_DIAGNOSTIC.md) | 96,352,345 | Better timing means/worst cohorts, with cohort, reconstruction and transfer tradeoffs. |
| Continuous early-mixer curve | [Diagnostic](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_LEARNING_CURVE_DIAGNOSTIC.md) | 141,317,815 | All 0/512/1024/2048 points retained; reconstruction improves without monotone timing benefit. |
| Matched-target TRAIN gain | [Diagnostic](../code/encoders/raw_patch_bottleneck_mae/MATCHED_TARGET_GAIN_DIAGNOSTIC.md) | 88,154,488 | All six numeric guards fail; recipe stopped. |
| Pooled temporal width | [Diagnostic](../code/encoders/raw_patch_bottleneck_mae/POOLED_CONTEXT_DIAGNOSTIC.md) | 96,170,207 | All four timing numeric guards fail; both mean MAE guards pass; wider route stopped. |

The latest [durable JSON](results/pooled_context_v1.json) retains every fixed
head repetition, conditional interval, full training trace and all five cohorts
for RPB-v10.alt-03/RPB-v12. Both serve native32 with original .15 training context
deletion. The compact model projects each contextual temporal summary to D32;
v12 pools W64 summaries before the global bottleneck. Route and first-layer
capacity change together: 225,805 versus231,949 registered values. The candidate
has 2,080 inactive projection values and 229,869 reconstruction-reachable values.
Common 219,469 values were copied before AdamW. This does not isolate a causal
effect of the compact projection.

New timing TRAIN/known VALIDATION cohorts have 256/128 rows, 128/64 independent
source pairs, C3/H32/F3 and original missing rate 0.1. Timing masters are
53151/54252/55353/56454/57555. Separate amplitude-transfer masters are
58656/59757/60858/61959/63060, with the original timing scaler and no amplitude
encoder training. Each timing model completes 512 joint encoder/decoder updates
at batch 8. Heads are fixed Ridge penalty 1 and tanh16 with Adam 0.01 for 100 steps,
three repetitions 2701/2802/2903. Their weights fit separately once per method;
the same TRAIN fit scores intact and fixed additional 30% deleted views.

New timing intact, equal-master means:

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v10.alt-03 | 32 | 98.28125 | 99.94792 | 100 |
| RPB-v12 | 32 | 90.93750 | 96.51042 | 100 |

New timing deleted:

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v10.alt-03 | 32 | 97.03125 | 99.37500 | 100 |
| RPB-v12 | 32 | 83.12500 | 89.42708 | 100 |

New amplitude intact:

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v10.alt-03 | 32 | 98.59375 | 98.64583 | 100 |
| RPB-v12 | 32 | 100.00000 | 99.94792 | 100 |

New amplitude deleted:

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v10.alt-03 | 32 | 92.18750 | 90.05208 | 100 |
| RPB-v12 | 32 | 98.75000 | 97.86458 | 100 |

These excerpts describe the compact early control and wider candidate; the full
report retains all seven methods in each panel, including raw 576, mask 288, raw
PCA32 and two separately fitted native32 initial controls. Untrained intact
amplitude linear means are 98.4375%/99.0625%, so strong transfer cannot all be
credited to timing training. Do not combine timing and amplitude into one score.

All five timing Ridge pairs, compact/pooled:

| Master | Intact % | Deleted % |
| ---: | ---: | ---: |
| 53151 | 98.4375 / 97.65625 | 96.875 / 92.96875 |
| 54252 | 93.75 / 73.4375 | 89.0625 / 53.125 |
| 55353 | 100 / 99.21875 | 100 / 96.875 |
| 56454 | 99.21875 / 85.15625 | 99.21875 / 84.375 |
| 57555 | 100 / 99.21875 | 100 / 88.28125 |

Mean standardized original-query MAE and synchronized training-loop cost:

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v10.alt-03 | 512 | 0.079615491 | 0.080886286 | 13.827282377 |
| RPB-v12 | 512 | 0.075831372 | 0.078075544 | 13.886893782 |

These loops include CPU evidence/trace capture and exclude checkpoint writing,
quality forward/transfer/verification, CPU heads and interval work. They are not
pure CUDA kernel timings. There is zero extra decoder-only calibration; ordinary
512 training jointly updates the encoder and decoder. The candidate improves
mean reconstruction but loses timing Ridge in every cohort and both views.
All four timing numeric guards fail; both mean MAE and coverage guards pass.
The joint direction fails. Within-master source-pair intervals are conditional
on saved encoders/heads; their bounds are not averaged or interpreted as an
across-retraining interval.

Stop wider pooled width and matched-target gain without range/rate/budget/head
rescue. Keep the stopped native view-agreement recipe and all-pair auxiliary
proposal stopped; skip half-mixer. The next bounded action is **one fresh
fixed512 compact early-v10 confirmation against late-v7**, under a separate
prospective card with matched new sources, original .15/native32/B8 and fixed
heads. Retain all cohorts and report all views; no adaptive best-point selection.
This note creates no next-run result, tag, seed schedule or implementation.

The sole pooled audit passed 96,170,207 checks / 1,235 saved CPU archive decodes in
402.684740278s. It verified 210 pipelines/420 heads, 20 retained points, 120 distinct
native exports, 20 query writers / 80 necessary masked CUDA forwards and 40,960
sampled row exposures. The reader replays saved arithmetic and named CPU state;
CUDA training/inference and ordinary checkpoint bodies remain source/admission
and byte-bound. There was no CPU encoder/autodiff/optimizer/head/PCA rerun,
historical payload input, TEST/stress, selection or promotion.

Capsule Na1nEQ inventory SHA:
`66a86efd911b5c98485b8540120c348105fb7be6853e9180b9733c53e4b3d700`.
Source SHA:`be129553c8712415d3f009dce8b2131c88123ff2e39f3822baf3bb55e0310059`.
Audit SHA: `6820cd40ec4ff79a715ff7521a4ff1315c78908ae81c8d2fa8e27395b7f23964`.
Report SHA: `3fc19814bfb5976a04e7335df2f2c5cf6b5d1f4b38022496ab2ff73814378944`.
JSON SHA: `196c4197e7ab3b34ad91d4df0df58fcfb6ca50197a70c8f0156b4ef3515142e1`.
Every historical registry object/report and continuous curve point remains
unchanged. Root's once-only
[original-v7 preservation check](../output/runs/rpb-matched-target-gain/original-v7-preserved-20261009-end-window.json)
passed for all 1,710 files / 678,396,765 bytes: 1,483 original v7 files and 227
decoder-calibration files, with zero changed bytes/files. Their inventory SHAs
remain `d4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821`
and `51400954bc2e9cae483de2e2a784e83f93482b02a888a3bdf95581d207967442`.
Proof SHA is `46f97eea182aeb1d1e1bb44e572033545f66ea455eca7b079c9cca44e6773c84`.
The check decoded no tensors, mutated no payloads, executed no model/head and
used no original quality-analysis roles. Do not repeat this completed check
as a new quality measurement.

Use the existing managed container `1e8de9cfec33`, named `rpb-paired-pooling` build
session and NVIDIA RTX A2000 8GB Laptop GPU. LibTorch 2.6.0+cu124 C++11-ABI Linux
SDK/runtime is inside `/opt/cuwacunu_embedding/libtorch`; proof/inventory lives
under `/opt/cuwacunu_embedding/setup/sdk-*`. Internal runtime dependency checks
and the separate first-encoder CUDA smoke passed 8+2 artificial updates/export.
`setup.sh` installs reproducible dependencies only; the authoritative launcher
owns lifecycle and reuses the existing container. No host toolchain, new
container, deletion/pruning or undocumented setup is needed.

The [portable source tools](../code/evaluation/tools/pooled_context_v1/README.md)
include seven exact reviewed sources plus a managed-container stager. Its
separate SOURCE smoke created a fresh FALSE reader/fixture/seal triple, verified
existing source dependencies and passed 27,229 artificial checks without any
payload/model/audit execution. It did not replace current reader pointers or
reuse historical fixtures. These tools are outside the compiled closure.

Use the [journal](RESEARCH_SESSION_2026-10-09.md),
[registry](EMBEDDING_VERSIONS.md) and
[next advance](../code/encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md).
Preserve failed attempts, original FALSE sources, released-copy proofs, every
saved group and historical TEST semantics. No stopped recipe is an implicit
permission for a tuning sweep.
