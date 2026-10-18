# Continuation after the fresh early/late confirmation

Date: 9 October 2026. Preserve the original working **RPB-v7** bundle on
4404/5505/6606/7707/8808, its checkpoints/scalers/readouts and optional frozen
decoder-calibration capsule. **RPB-v4** remains the formal reference. Fresh
**RPB-v7.alt-04 / RPB-v10.alt-04** are separate late/early groups. Compact v10
remains unpromoted: timing means improve, but both worst-cohort scores fall.
Matched-target gain v11, wider pooled v12, native-view agreement v9 and the all-pair
auxiliary remain stopped; skip half-mixer and further rescue tuning.

The exact [confirmation report](../code/encoders/raw_patch_bottleneck_mae/EARLY_MIXER_CONFIRMATION_DIAGNOSTIC.md)
and [durable JSON](results/early_mixer_confirmation_v1.json) retain four full
seven-method panels, all five paired cohorts, three fixed head repetitions,
conditional intervals, both initial controls and full 512-step training traces.
Late mixing follows temporal encoding. Early mixing precedes it and retains a
separate independent local temporal pass. Both register 225,805 values and
serve native32; equal parameter count does not imply equal compute.

Timing masters are 64161/65262/66363/67464/68565. Each has 256 TRAIN rows from 128
source pairs and 128 known VALIDATION rows from 64 disjoint pairs, C3/H32/F3 and
natural missing rate .1. Each of ten trajectories completes 512 joint
encoder/decoder updates, batch 8, AdamW and the unchanged .15 training context
policy. Separate amplitude head-data masters are 69666/70767/71868/72969/74070;
amplitude never fits the encoder/scaler. Heads are Ridge 1 and tanh 16 with
Adam .01 for 100 steps, reps 2701/2802/2903. Every TRAIN fit scores intact and one
fixed additional 30% coordinate-deleted VALIDATION view.

New timing intact, equal-master means:

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-04 | 32 | 94.21875 | 96.71875 | 100 |
| RPB-v10.alt-04 | 32 | 95.00000 | 96.92708 | 100 |

New timing deleted:

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-04 | 32 | 88.28125 | 94.73958 | 100 |
| RPB-v10.alt-04 | 32 | 92.81250 | 95.88542 | 100 |

New amplitude intact:

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-04 | 32 | 99.84375 | 99.42708 | 100 |
| RPB-v10.alt-04 | 32 | 99.84375 | 99.79167 | 100 |

New amplitude deleted:

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-04 | 32 | 97.65625 | 92.86458 | 100 |
| RPB-v10.alt-04 | 32 | 96.87500 | 94.47917 | 100 |

These excerpts describe the trained native late/early pair. The full report
retains raw576, mask288, raw PCA32 and two separately fitted native32 initial
controls. Initial amplitude Ridge is already 99.0625%/99.21875% intact, limiting
credit assigned to training near this ceiling. Do not combine tasks into one
encoder accuracy or use secondary scores to erase timing regressions.

Three timing pairs improve and two worsen in both views. Every Ridge pair,
late/early:

| Master | Intact % | Deleted % |
| ---: | ---: | ---: |
| 64161 | 97.65625 /100 | 79.6875 /100 |
| 65262 | 96.875 /80.46875 | 97.65625 /75 |
| 66363 | 96.875 /100 | 92.96875 /98.4375 |
| 67464 | 95.3125 /94.53125 | 92.96875 /90.625 |
| 68565 | 84.375 /100 | 78.125 /100 |

Master 65262 loses 16.40625 and 22.65625 percentage points. Worst-cohort Ridge
falls 84.375%→80.46875% intact and 78.125%→75% deleted. Both mean accuracy
guards and both mean MAE guards pass; both worst-cohort guards fail, so the
joint descriptive direction fails. Coverage passes for every cohort. This is
not a promoted, uniformly reliable winner, nor evidence of a specific causal
information-loss or overfitting mechanism. The earlier 98.28125% compact-v10
score is retained on different cohorts; it is not rewritten by this run.

Standardized original-query MAE and synchronized update-loop means:

| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | ---: | ---: | ---: | ---: |
| RPB-v7.alt-04 | 512 | 0.077806255 | 0.080484228 | 15.832232 |
| RPB-v10.alt-04 | 512 | 0.073898432 | 0.075583393 | 16.602317 |

Loop seconds include CPU trace capture and exclude checkpoint writing, quality
inference/transfer/verification and CPU head/bootstrap work. Pure CUDA kernel
time is unmeasured. There is zero extra decoder-only calibration; ordinary
training jointly updates encoder and decoder. Every saved head is fitted once
and reused across views; native exports receive no PCA afterward.

The separate [harder timing benchmark](HARDER_TIMING_BENCHMARK_PLAN.md) has an
implemented reusable variable-period/delay generator and analytic data-only
fixtures. Harder encoder performance remains **unmeasured**. The next bounded
measurement keeps both late-v7 and compact early-v10 at fixed512 with unchanged
native32/.15/B8/heads, under its own prospective card/source/reader/CUDA gates.
Preserve original-difficulty results as an anchor. Additional headroom does not
resolve retraining variation, and it authorizes no head, width, gain, rate or
best-budget tuning. No TEST/stress is reopened.

The sole quality capsule uJ1Ack completes 10 trajectories, 20 retained 0/512 states,
40,960 row exposures, 210 pipelines/420 heads, 120 distinct native exports and 20
query writers with 80 necessary masked CUDA forwards. There are 10 shared driver
raw/PCA outer fits and 50 helper outer fits. The external cohort fit identity is
`early-mixer-confirmation-v1/lag_sign`; the unchanged implementation contract
truthfully remains `early-mixer-reliability-v1/lag_sign`. A fifth `.confirmation.pt`
and separate confirmation snapshot audit bind both identities to the four
original companions, exact original TRAIN/order/scaler/counters and source IDs.
No historical quality payload is an input.

Preserve failed admission FXJe90 and its exact SOURCE. Its new test used two
nonexistent checkpoint-with-optimizer overloads; the narrow repair calls the
authoritative optimizer loader. No production recipe changed and no quality
was generated during that failed compile. Admission 72hqzF then passed CUDA
tests with the internal SDK. The first v3 audit stopped at SOURCE provenance
after 18,502 checks and **zero archive decodes**, because incremental admission
reused three objects compiled in the preserved prior attempt.

The final composite v4 audit binds only those three absent compiler records,
requiring exact prior/current 36/115/160 SOURCE-subset equality, prior object
macro/target commands, current successful link/runtime markers and every
unchanged downstream typed/arithmetic check. It makes no claim of fresh current
inherited compilation. The base reader and tolerances are unchanged. Its
structured compiler-evidence limits and external SOURCE preservation are carried
in the exact durable JSON. The capsule is never edited or regenerated.

Final audit PASS: 96,153,106 checks/1,255 saved CPU archive decodes in
780.0044506200065s. Independent emission metadata QA passed 286,641 comparisons.
The one exact saver verified every summary field/Markdown and all eight input
bytes before the exclusive durable copy. A compatibility leaf copied the two
emitted files byte-for-byte without rerendering. During report emission, QA and
saving, no CPU encoder, optimizer, autodiff, head/PCA/bootstrap or audit rerun
occurred. There was no second quality generation.

Inventory:`d7f6db6dccad050eec984196d28e99b2cc1091c537edf40049ce2be117454846`.
Source:`fa1cbd0e67b4ce6226d6d3a0c5c5c39a71298764e64158dfdbc50decad7e1f12`.
Audit:`63826de216aff8a09b26b203d10fb9020fc08208276eec27347c9570e839957e`.
Composite reader:`979601d1c2acb2a7886b0d0f39070bf8738dd3f5788ce57e3dbdc6ac9e4a3c66`.
Base reader:`6ba1100b5e983a76ddf83369a893e07241a0193f009c1b6c00b340ab270ce0a0`.
Compiler evidence:`2084b926ca034e18f8d0ffa09d436d304be4c7735ab00a1558a1cca3d0bee1e8`.
Report:`d72b31decfb2a3545779b842f88da9303f7c6bd39c2a103cfa20da298a188ae6`.
JSON:`62b810c857148dd99bb44ab9d12e7db8c9708ff65f16d44a3d1b3c8354df7611`.

Portable [SOURCE tools](../code/evaluation/tools/early_mixer_confirmation_v1/README.md)
and truthful cached-source adjunct copies are outside the frozen 182 producer
closure. For a future admission, the reviewed five-source Make-W wrapper ensures
the unchanged checker captures its own compile records. Only syntax/artificial
argument forwarding was tested; no future cached admission has run. Its proof
SHA is `11040dee1b1c1fe323f55c6e20c5bd5e6f9952182ec7dc83998b7c722cdab5e6`.

Use the existing managed container 1e8de9cfec33 and named rpb-paired-pooling build,
RTX A2000 8GB Laptop GPU, and internal LibTorch 2.6.0+cu124 C++11-ABI Linux SDK
at `/opt/cuwacunu_embedding/libtorch`. SDK proofs/ldd and the separate first-
encoder artificial CUDA smoke remain completed; setup installs dependencies
only. No new container/toolchain installation, deletion or broad cleanup is
needed. The [earlier pooled continuation](CONTINUATION_2026-10-09_AFTER_POOLED_CONTEXT.md)
retains all six prior milestones and the completed once-only original-v7
preservation proof. Every historical version/instance object and all original
v7 files remain protected; do not repeat that byte check as new quality.
