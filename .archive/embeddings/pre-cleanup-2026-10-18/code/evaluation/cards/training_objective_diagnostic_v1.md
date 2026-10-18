# TRAIN objective diagnostic v1

Protocol: `training-objective-diagnostic-v1`. Prospective diagnostic, frozen
before permitted payload hashing or decoding. RPB-v4 remains active. This is
neither a new encoder version nor a quality or promotion experiment.

## Boundary and identities

Inspect saved RPB-v4/v7/v8 checkpoints at 512 attempted/completed updates on
masters 4404/5505/6606/7707/8808: fifteen frozen instances, zero encoder updates,
zero optimizer creation/steps and zero classifier fits. Each has native global
32 export, mode2/mixer1, C3/H32/F3, patch8, width64, temporal layers3, heads4,
feed-forward256, decoder128 and 225,805 parameters. No PCA follows an encoder.
All existing weights, buffers, fits, optimizer archives and source artifacts
must be preserved byte for byte.

The metadata-only role planner declares exactly 195 TRAIN files: seven cohort
roles per instance (controlled TRAIN observations, checkpoint, checkpoint
audit, fitted scaler, raw TRAIN checkpoint companion, native TRAIN export and
TRAIN reconstruction) plus three saved native fits and three saved TRAIN
prediction archives per instance. Those roles declare 153,430,792 bytes.
They are explicitly named by `prepare-training-objective-inputs.py`, never
recursively discovered. Its pinned inventory and prior role metadata are
metadata, not permission to decode their other files. Preserve the distinction
between original checkpoint producer and physical V8 cached-head writer.

Parents are `context-replication-JjNEUc`, `lighter-validation-duyRU2` and
`balanced-validation-99UTEV`, with pinned inventories respectively
`13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9`,
`d4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821` and
`e0cdfbc729c8d989e16bb5e8515b5c5d73b7b445bdf91d2d938d88e09c969ac7`.
No VALIDATION, TEST, stress, point0, raw/PCA fits or mixed-split archive may be
opened. Labels are diagnosis/scoring data only; the encoder input has no labels.
Each TRAIN cohort is 256 rows from 128 opposite-label source pairs with original
10% natural missingness. Results are in-sample diagnostic evidence.

Actual CUDA admission, frozen source manifest and this card's SHA precede input
hashing. A new exclusive capsule binds all 195 input hashes before and after,
compiled source, admission logs and artifact inventory. Failures remain saved.
An independent sealed CPU arithmetic reader checks saved arrays and lineage;
it does not rerun encoders or classifier optimizers.

## Full-observation geometry and saved-head parity

First prove exact CPU native-export parity with the allowed TRAIN archive and
all three saved Ridge fits identical. Reproduce all three Ridge and tanh neural
TRAIN predictions without refitting. Record tolerances and observed errors.
Use full original observation O for geometry only. Inspect flattened contextual
BCD32 (96 coordinates) and served BD32. Standardize each geometry surface using
its own TRAIN mean/std, floor1e-8; this is analysis, never a served transform.

For each same-source pair let m=(z0+z1)/2 and d=(z1-z0)/2. Save midpoint
variance, mean squared half-difference, squared mean half-difference and
incoherent difference variance. Check the trace identity: centered row variance
equals midpoint variance plus mean squared half-difference. Report saved Ridge
signed margins along paired contrasts. A coherent direction can cancel with
random signal phase. These statistics describe linear geometry, not irreversible
information loss, recoverability or generalization.

## Eight frozen query banks per instance

Four original common-patch query banks each use all 256 TRAIN rows, original
Q=O&A, original eligibility and scaler. They reproduce the saved factual frozen
CUDA reconstruction first. Only these full-pair banks have decoder interventions:
factual native z, per-bank TRAIN mean z, opposite-label same-source sibling z,
and fixed cyclic cross-source derangement preserving label. Receiver metadata,
targets, Q and original loss denominators remain fixed. No classifier/decoder
refit. Save the donor indices explicitly. Interventions are counterfactual
diagnostics, not generated examples or quality scores.

Four actual-policy banks reproduce absolute attempted counters 512..515,
batch8, original sampled row and Torch/mask/context streams. v4 uses E=0, v7
uses rate0.15 every attempt, v8 uses ordinary even and rate0.30 odd attempts.
Original sampler indices/queries and repair are unchanged. The visible set is
V=O&~A&~E; all nonvisible storage is zero. There is no counter increment or
weight update. These four banks contain 32 row exposures, which need not be
32 distinct rows or independent source groups. Save exact masks, row indices,
support counts and policy. They have no donor interventions.

The full query bank's common A differs from actual training's independent
per-channel A. Preserve unsupported auxiliary examples; do not replace them,
change masks or condition sampling on support.

## One hypothetical auxiliary and gradients

Original reconstruction is hierarchical Huber delta1 in separately fitted
normalized channel units, averaging cells, eligible channels, eligible examples.
The single hypothetical auxiliary uses every unordered semantic channel pair:
Huber(((pred_i-pred_j)-(target_i-target_j))/sqrt(2)), on Q_i intersection Q_j
and original eligibility. Average supported cells, then supported pairs per
example, then supported examples. With no supported example it is differentiable
zero, and that absence is explicitly recorded. Original Huber remains unchanged.
Do not privilege channels0/1. This adds no new target information. It reweights
residual differences; two of three pairs include unrelated channel2. It is not
a phase-invariant lag objective. Common residual cancellation applies in the
separately normalized channel units, not necessarily raw units. Under full C3
support with identical complete query-cell supports in the quadratic region,
base plus weight1 auxiliary raises centered
channel residual weight by2.5 while leaving normalized common-mode weight fixed.

Use the first identical saved Ridge fit for gradients. Let u be its final
standardized coordinates, with both saved affine maps retained. Signed correct
margin is (2y-1)*(logit1-logit0). For a zero float64 CUDA leaf, construct decoder
z=z_frozen+(u*outer_scale*ridge_scale).to(model_dtype), preserving the exact
factual z at u=0. The reconstruction coordinate gradient multiplies both scales;
it never divides them. The margin gradient in u is (2y-1)*(w1-w0).

Save base, auxiliary and margin gradients at the latent surface and over every
non-decoder encoder parameter in lexical name order. Exclude all `decoder_`
parameters, including decoder target metadata. Unused encoder gradients are
explicit zero segments. Use autograd::grad without optimizer or backward update.
Record latent and full-encoder norm/dot/cosine evidence and parameter offsets.
First establish CPU export and frozen-CUDA query parity. Grad-enabled export
may select a different CUDA kernel: save max|z_graph-z_frozen| and reject above
1e-6. Use a separate real grad-enabled encode/decode path for full-encoder
gradients, without a straight-through bridge; only the latent leaf uses detached
frozen z.

Define descent alignment as cosine(-g_reconstruction,+g_signed_margin): positive
helps the current correct margin, negative opposes it. Zero norms are undefined,
never counted as helpful or conflicting. Combined means base+weight1 auxiliary.
This is a local plain-gradient direction, not an observed update, AdamW path,
training improvement or out-of-sample score.

## Predeclared advance-or-null decision

Only v7 actual-policy banks512..515 govern whether to consider a separate future
weight1 auxiliary encoder experiment. All of these conditions must hold:

- Each master has at least16 of32 auxiliary-supported row exposures.
- Mean whole-encoder combined-descent margin cosine improves over base by
  more than1e-6 on every master.
- Base mean descent-margin cosine is negative on at least three masters.
- Every master's mean original-Huber directional derivative under the unit
  combined descent is strictly negative:
  g_base dot (-(g_base+g_aux)/norm(g_base+g_aux)) <0.
- All required norms, support, parity, preservation and independent audit pass.

Otherwise record null for this loss hypothesis and stop it; do not force an
encoder run, tune auxiliary weights, search pair subsets or use VALIDATION to
rescue it. A pass merely motivates a separately frozen implementation/model card
and actual-CUDA admission before a prospective quality experiment. No promotion
or version registration follows from this diagnostic alone.
