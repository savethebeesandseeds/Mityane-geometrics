# After fresh spectral confirmation, 10 October 2026

The unchanged **RPB-v19** design passed fresh-source confirmation as the separate
**RPB-v19.alt-01** instance bundle. The [report](SPECTRAL_CONFIRMATION_V1.md)
and [durable JSON](results/spectral_confirmation_v1.json) retain all six
trajectories, initial controls, fitted heads, queries, costs and twelve gates.
The original [v19 screen](MULTIBAND_SCREEN_V1.md) is preserved byte-for-byte.
Formal v4, original v7, TEMPO-1 v10.alt-03 and TEMPO-3 v18 remain preserved.

Three fresh masters930905/930906/930907 per task; TRAIN256/128 source pairs and
VALIDATION128/64 pairs. Native32, B8,512 CUDA updates, original waveform Huber1,
context deletion.15 and unchanged Ridge1/tanh16 heads. Each instance ran once.
No old encoder/head, raw baseline or information-fitting execution.

Dataset legend: **TEMPO-4** = slow-component lead/lag in a coherent two-rhythm
mixture; **AMP-2** = relative slow/fast strength. Each has designed complexity
**5/5**, using `two-component-v1`. Below is extra30% deletion VALIDATION,
averaged over three fresh cohorts and their three fixed head repetitions.
Tasks are separate. RPB-v19.alt-01 retains20 learned shape coordinates and12
fixed generic low/high complex spectral relations inside the sole native32.

| Dataset | Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | ---: | ---: | ---: | ---: |
| TEMPO-4 | RPB-v19.alt-01 | 32 | 93.23 | 93.23 | 100.00 |
| AMP-2 | RPB-v19.alt-01 | 32 | 94.79 | 86.89 | 100.00 |

Intact Linear/Neural means are99.48%/98.44% timing and99.48%/98.87% balance.
All12 prospective cohort/view conditions pass both heads>=75% and coverage100%.
The weakest cohort/head is84.375% deleted AMP-2 Neural on master930906.

Initial0 deleted means are91.67%/89.76% timing and92.45%/85.42% balance.
Training adds1.5625pp/3.472222pp timing and2.34375pp/1.475694pp balance to
Linear/Neural means. Those small gains vary across cohorts; fixed rhythm
structure still supplies most of the strength. Fresh confirmation supports
repeatability in this declared family, not universal quality or promotion.

Dataset legend: **TEMPO-4** timing and **AMP-2** rhythm strength, each designed
complexity **5/5**. Fixed masked TRAIN/intact VALIDATION waveform queries;
errors are standardized MAE. Seconds are mean GPU training-loop time per
instance, excluding extraction, heads, query writing and saved-only checking.

| Dataset | Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| TEMPO-4 | RPB-v19.alt-01 | 512 | 0.717581 | 0.742483 | 20.69 |
| AMP-2 | RPB-v19.alt-01 | 512 | 0.726132 | 0.757934 | 21.30 |

The learned waveform path is the next target. These fresh cohorts confirm the
classification success, while waveform reconstruction still has room to improve.

Evidence capsule: `output/runs/rpb-spectral-confirmation/admission-hzyMa2`.
73-file SOURCE SHA256:
`170e77892d97116cae3b3af219a34beaef605fdb3d51803345f6c2a17db8da65`.
Card: `010138160358129f4c3a96ec8d6300e432130c5b66aeb7c43dd8ed74f91a5917`.
Input manifest: `0e05de823e02c46afd5a47bc1c3864f200dabc7e97da4aecd2148566a7061e8b`.
Summary: `21928e321a17b01b32d52a78c04bac3d4e9f755796c09a4ff6b2fc5112431eb1`.
Artifact inventory: `29a5ec0e18b596668b78452287cd8f3130a94b2495f56b74f69a782fc60401f1`.
The prospective source/card and actual CUDA admission were committed and pushed
as `00cd5fc` before quality data generation. The registry adds one bundle to the
existing20 designs/18 bundles, preserving every original entry and checkpoint.

Ledger:6 trajectories/3,072 updates/24,576 sampled rows;36 native exports,
36 head pipelines/72 heads,12 query writers/48 necessary CUDA forwards.
Six data-only generator calls created18 legal archives once. Saved-only checking
passed7,235,268 checks/216 CPU archives and36,864 byte-identical fixed coordinate
values at0/512. It performed zero encoder forwards/updates/head fits/PCA fits,
information fits or checkpoint body decoding. CPU checking is saved arithmetic;
the encoder was not run again on CPU. No skips, TEST, stress or default changes.

## One next architecture hypothesis

Keep the successful spectral12, exact encoder/shape20, data and fixed heads.
Replace the nonlinear patch-conditioned decoder with a **linear generic
harmonic-synthesis decoder** receiving only native32. Predict153 coefficients:
DC plus cosine/sine q1..8 for each of nine semantic channel/feature streams.
Synthesize on original time ticks0..31 with a fixed CUDA basis, then gather
semantic channels into the requested physical order. No coefficient targets,
hidden frequencies, information-fit outputs, labels or auxiliary losses.

The reusable route is a new isolated v19 wrapper. Construct the full unchanged
v19 first, then unregister backbone decoder_positions/decoder_channels/
decoder_first/decoder_second before AdamW; attach a32-to153 affine decoder and
explicit new decode/forward. LibTorch2.6 exposes public unregister_module.
This preserves the old encoder constructor/RNG draws and encode arithmetic,
without copying or changing frozen Model.h. Private holders retain about46KB
of unused storage; disclose it. Their11,528 values leave optimizer/checkpoint
registration. Adding5,049 values gives220,398 registered,219,966 trainable and
432 frozen. Preserve or avoid Torch RNG draws for the new head; choose one
nonzero deterministic initializer before outcomes, with no initialization search.

Before any candidate quality, freeze a separate card and source, a new design
tag, CUDA engineering and a two-cohort known-development screen on saved
930905/930906 data for both tasks. Reuse v19.alt-01 inputs and metadata; no old
encoder/head reruns or new quality data generation. Exact initial native32
parity against the saved v19 exports and unchanged spectral12 at0/512 are
mandatory. Save own initial queries as well as512 queries and all failures.

Proposed decision criteria to freeze before that run: each cohort/view/head
>=75% with100% coverage; all task/view/head means no worse than matched trained
v19; deleted AMP-2 Neural improves at least2pp over both its own initial and
matched trained v19; per-task mean TRAIN and VALIDATION original-query MAE each
lower than matched trained v19 and the candidate's own initial. Timing's
near-ceiling scores are preservation guards. Never average tasks or choose a
best seed/head, and stop a weak frozen recipe without tuning bands, masking,
heads or budget. Lower MAE alone does not establish useful class geometry.
These are proposed, unmeasured criteria; no new architecture tag, implementation
or quality allocation exists yet. A pass would justify fresh confirmation,
not automatic promotion. Off-grid truncation and limited shape rank remain
the decoder hypothesis's material risks. Preserve original Q and the successful
rhythm representation throughout.
