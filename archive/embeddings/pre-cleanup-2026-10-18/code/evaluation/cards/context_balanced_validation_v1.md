# Balanced context validation v1 development card

Protocol: `context-balanced-validation-v1`. Policy: 1.2. Status: prospective,
implemented driver awaiting admission and measurement. No measured gain,
TEST/stress access, selection, confirmation or promotion is claimed.
RPB-v4 remains active. Freeze this card, source identities, parent role hashes
and coordinated actual-CUDA admission before fitting.

## One hypothesis and unchanged inference

RPB-v8 alternates ordinary visible context and existing 0.30 context deletion.
It retains the RPB-v4 mode-2/mixer-1 native global32 architecture and exact32
decoder bottleneck. All 225,805 parameters, C3/H32/F3, semantic IDs 0/1/2,
patch 8, width 64, temporal layers 3, heads 4, feed-forward 256 and decoder 128
remain unchanged. Inference has no schedule, deletion or repair. No PCA or
random projection follows the encoder. RPB-v7 is the same inference model
trained with 0.15 requests on every update; RPB-v4 has ordinary context only.

The [implementation plan](../../encoders/raw_patch_bottleneck_mae/BALANCED_CONTEXT_IMPLEMENTATION_PLAN.md)
motivates one view-distribution change after the lighter policy still failed
reconstruction. It is not evidence that the mixture improves the encoder.
Do not add another rate, budget, loss, teacher, decoder or classifier change.

## Exact schedule and training state

Use policy `rpb-training-context-balanced-030-v1`, tag RPB-v8. Let `a` be the
zero-based absolute attempted counter before incrementing. Even `a` uses
unchanged ordinary forward with E=0; odd `a` uses the original
`coordinate30_v1` context primitive at that same absolute `a`. Indices 0..511
give exactly 256 ordinary and 256 deletion updates. Do not reverse the phase,
renumber active attempts, shift after skips or add replacements. Persist the
exact schedule literal `absolute-attempt-even-ordinary-odd-coordinate30-v1`. Any skipped
attempt aborts this paired experiment before scoring.

Preserve initialization, sampled rows, original artificial masks and per-attempt
Torch streams. Odd attempts retain context stream `0x6374782d64726f70`,
semantic-canonical BCHF ordinal and the top53-bit request law. Ordinary
attempts consume no context or extra Torch RNG. This is an active rate 0.30
on half the updates, not uniform 0.15. Record branch counts and cumulative
requested/actual/restored coordinates; actual deletion need not equal v7.

Original O, A, Q=O&A, targets, eligibility, scaler and hierarchical Huber 1
cell/channel/example reductions remain exact. Repair clears only extra E to
retain two original visible patch groups in originally eligible channels.
Never restore natural missingness, queries or hidden target values. Zero
nonvisible storage before encode and decode solely the served global32.

Use learned_global.conf with actual CUDA, batch 8, dropout0, AdamW learning
rate0.001, weight decay0.0001 and clipping 1. Retain original mask ratio 0.25,
TRAIN scaler floor 1e-6, threads 1 and all resolved settings. Fit the scaler
only on original legal TRAIN. Save 0/512 on one live AdamW trajectory and
require attempted=completed=512, sampled rows4096, branch counts 256/256.
Checkpoint policy tags block ordinary resume. Only fresh live continuous
training and immutable snapshots are implemented. Ordinary tagged reload-resume
and the existing v6-only replay adapter reject v8; there is no balanced replay API.
Inspection restores CPU/all-CUDA RNG and threads.

## Two explicit retained parents and legal cohorts

Use known masters 4404/5505/6606/7707/8808, timing `lag_sign`, 128 TRAIN source
pairs/256 examples and 64 known VALIDATION pairs/128 examples per master,
with 10% natural missingness. Opposite-label examples stay paired by source.
Train five new candidate encoders, retain five v4 and five v7 encoders;
saved points and head repetitions do not add encoder runs.

Named parent `v4` is
`output/runs/rpb-context-replication/context-replication-JjNEUc`;
inventory SHA256 `13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9`.
Named parent `v7` is
`output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2`;
inventory SHA256 `d4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821`.

The driver exposes metadata-only `--source-id` and `--parent-roles` before
payload access. The exact 326 roles are 158 under v4/reference and 168 in v7:
158 under results/candidate-development plus the five saved view `.pt`/`.json`
pairs. Roles cover original TRAIN/VAL observations/source order, development
metadata, 0/512 checkpoints and companions, native features, point records,
fixed queries and positive native fits/predictions. No recursive discovery,
old TEST/stress/report.json access or preservation hashing of such payloads.
Runner metadata names parent_id,path,role,tag,master_seed,budget; the exact
SHA text binding is `SHA256  parent_id/relative_path`. Preserve every allowed
byte and both inventories. Input roles must be bound before any fit.

Pregenerate TRAIN/known VAL only and require byte-exact values/masks/labels/
source order against both parents. Encoder callbacks receive only legal TRAIN
observations/metadata under `native-development-v1/lag_sign`, never labels,
clean hidden values or heldout data. Retained v7 initial state is gated through
the unchanged explicit 0.15 gate against v4; candidate0 through the balanced
gate. Require all initialized parameters/buffers, scalers and common streams
to agree before 512 is admitted. Point0 remains an ordinary diagnostic.

## Reuse exact known-VALIDATION deletion, fixed fits

Copy the exact pinned v7 `validation-dropout-030.pt` and its JSON per master;
there is no new draw. Preserve old namespace `context-lighter-validation-v1`,
stream `0x636c763164726f70` (7164231061750312816), actual seed
`stream_seed(master,stream)`, source-paired erasures and physical channel
order0/1/2. Verify original values/support, labels/source order, requested
erasures, O'=O&~E, retained values and hidden zeros. No support repair.
Both views use the same exact model and ordinary TRAIN-fitted heads.

The positive matrix is 15 inputs: five masters times v4/v7/v8 at512. Refit
ordinary TRAIN maps and heads transparently, then require exact retained/cache
native fit and intact/TRAIN prediction parity. Native32 has no PCA. Raw576
is288 TRAIN-standardized observed values plus288 observation flags; standalone
raw PCA32 fits TRAIN only. Check identical raw/PCA maps/fits/predictions across
versions. Do not claim loaded old heads or old raw/PCA parity. Preserve every
unsupported reason; unsupported controls do not disable legal native scores.

Keep ridge 1 and tanh 16, Adam 0.01/100 updates, all base seeds 2701/2802/2903
and width-paired actual seeds. At width 32 heads have 66/562 parameters; raw576
heads have 1154/9266. No corrupted VALIDATION fit, best-head selection or tuning.
The positive matrix contains45 fitted pipelines per method when supported,
135 total, separate from candidate development fitting. Retain 45 native
parity records, 60 version/view/repetition records and 120 head effects.
Comparators are v8-v4 and v8-v7; bootstrap comparator IDs 4/7 are explicit.
Use exact per-row common support and full-population correctness with
abstentions as failures, 1000 source-group replicates/95% within-master
intervals. No mean CI bounds or pooled-master uncertainty.

## Admission, fixed reconstruction and decision

Actual CUDA admission must prove full common initialization, ordinary first
update parity, odd-attempt original 0.30 erasures, untouched queries/targets,
deterministic repair, exact32 route, and direct 0→4 versus split 0→1→2→4
weights/buffers/AdamW/scaler/counters/context-count parity. Preserve earlier
snapshot witnesses, disabled/0.30/0.15 regressions and tagged-resume rejection.
Generic native32 serving gate is necessary but does not replace this new
training-policy admission. The measure phase binds its passed source/log SHA.

Retain original unaugmented four-patch TRAIN/VAL query arrays and standardized
MAE/Huber reductions. Retained v4/v7 exports/queries must reproduce saved
values/support; candidate cache must also match its retained loader. Keep
per-master counters and synchronized loop costs, distinguish old versus new
timings and avoid kernel-only/inference-cost claims.

At 512 the joint development question requires mean **and worst-master** native
ridge accuracy in both views no lower than v7, equal coverage, and mean TRAIN
and VALIDATION MAE each no worse than paired v4. Infrastructure/admission/audit
must pass. Record every master and neural result; neural gains cannot rescue
a primary failure. Crossing-zero within-master intervals limit interpretation.
Finish all declared runs even if these score/error guards fail, then report
the failed tradeoff and stop this local schedule/rate/budget tuning sequence.
No automatic TEST draw, fresh confirmation or promotion follows even a pass.
