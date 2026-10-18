# Pooled temporal context v1

Prospective development protocol: `pooled-context-v1`. Freeze this card,
the complete source closure, actual CUDA admission and an independently reviewed
blocked reader before quality generation. One architecture comparison, with no
width, loss, deletion-rate, head or update-budget search.

## Question and groups

Does retaining each channel's learned W64 temporal summary until global pooling
improve the same native32 representation? The early architecture currently
projects each channel to D32 before its global MLP. A direction removed by that
projection cannot reach the MLP. The candidate changes that route and first-layer
capacity together; it does not isolate projection loss as a causal explanation.
The previous matched-target gain failed all six numeric guards and is stopped.

**RPB-v10.alt-03** is a fresh compact-summary early-mixing control.
**RPB-v12** pools full temporal summaries before the global bottleneck.
Both retain pretemporal aligned channel mixing, the separate independent local
temporal pass, mode2/mixer1/dropout0, learned temporal attention, and the exact
served global BD32 as the decoder's only observation signal. No token, local
export or W64 bypass reaches reconstruction. Decoder metadata stays unchanged.

The closed configuration discriminator is `global_pool_input_source`: 0 is the
historical projected-D input, 1 is the learned temporally pooled-W input.
Source1 is legal only with mode2/mixer1/early-placement1/dropout0. Canonicalize
both independent and contextual W summaries by configured semantic channel
order, zero unsupported slots, flatten values, append support bits, and apply
the same Linear/GELU/Linear global head. All-absent global exports are exact zero.
Per-channel D32 outputs remain diagnostics. Source0 preserves its old arithmetic,
settings text and archive envelopes; source1 requires its typed discriminator,
input semantics and architecture/output/reconstruction tags before weight load.
Its architecture ID is `aligned-mixer-before-temporal-pooled-context-width-v1`.

The first global matrix changes [64,99] to [64,195]. Registered parameter counts
are 225,805 control and 231,949 candidate, adding 6,144. Candidate diagnostic
`export_projection.weight` [32,64] and `.bias` [32] contain 2,080 entries and
receive no reconstruction gradient: they remain unchanged and have no AdamW
state. There are 229,869 candidate entries on the reconstruction graph; this
does not imply every scalar changes. Report actual named optimizer activity.

Timing masters: 53151,54252,55353,56454,57555. Separate amplitude masters:
58656,59757,60858,61959,63060, paired in that order. Generate each task/cohort
once with 128 TRAIN source pairs and 64 VALIDATION source pairs (256/128 rows),
C3/H32/F3, patch8, semantic IDs0/1/2, unitless features, interval1, endpoint31,
CPU float64 legal observations, natural missingness0.10 and zero absent storage.
Prefix IDs with this protocol/task. TRAIN and VALIDATION sources are disjoint.
No TEST/stress, historical quality inputs, trained initialization or selection.

## Initialization and training

Both label-free factories receive the same original timing TRAIN data/order
under `pooled-context-v1/lag_sign`; each fits its original scaler once. Amplitude
never fits an encoder/scaler. Require exact original raw, schema, source order,
scaler counts/means/scales, fit seed, resolved settings except input source, and
zero counters on the compact reference before candidate initialization.

Save compact point0 first. Construct the candidate with the same initialization
seed, then copy every same-name/same-shape parameter and buffer from that exact
zero-update reference before creating AdamW or the initial-state witness.
Only `global_pool_first.weight` differs in shape and retains its declared wider
fan-in initialization. Copy the first global bias and second global weight/bias
explicitly; a shared seed alone does not preserve later constructor draws.
Exactly 219,469 parameter entries are common. Bind the five reference checkpoint
roles and record complete copied/nonshared names, shapes and reference identity.
Reject positive, mismatched or altered references. Preserve ambient RNG/threads.
The candidate's optimizer starts empty and never imports control optimizer state.
Different initial global outputs require two separate untrained controls.

Both use the original `rpb-training-context-deletion-015-v1`, original Q and
eligibility, hierarchical Huber1 and deterministic E-only repair. No gain,
auxiliary loss, decoder calibration or scaler refit. Batch8, AdamW0.001 with
weight-decay0.0001, gradient clip1, one thread, full per-attempt trace, exactly512
completed CUDA updates, attempt cap1024. Abort on any skip or exhaustion.
Row/A/E/Torch streams match across the pair and do not depend on constructor
draw count. Ten live trajectories give5,120 updates/40,960 row exposures;
each has16 equivalent presentations of256 TRAIN rows, not guaranteed epochs.
Keep one model/scaler/AdamW per trajectory; absolute0/512 snapshots do not reset
the loop. No new checkpoint reload-resume API. Historical factories reject
source1 and ordinary train/resume keeps its early/tagged rejection.

## Fixed evaluation

Each task has intact VALIDATION and one common additional0.30 coordinate erasure
view without support repair. Timing uses `stream_seed(timing_master,
0x7063763174696d30ULL)` and `pooled-context-v1/lag_sign/validation-coordinate-dropout`;
amplitude uses `stream_seed(amplitude_master,0x70637631616d7030ULL)` and
`pooled-context-v1/amplitude/validation-coordinate-dropout`. Keep the existing
source-paired physical-grid law. All methods share observations, labels, order,
erasure masks and the fixed head recipes.

Seven method IDs: `raw`, `mask_metadata`, `pca_only`, `untrained_compact`,
`untrained_pooled`, `native_compact`, `native_pooled`. Sizes are respectively
576,288,32,32,32,32,32. PCA follows raw data only. Raw and PCA share one TRAIN
outer normalization; each probe has its unchanged TRAIN normalization.
Repetitions2701/2802/2903 use Ridge1 and tanh16/Adam0.01/100 updates under the
existing repetition/width seed law. Fit each head once on ordinary TRAIN and
reuse for both views. At32 heads have66/562 parameters; raw576 has1,154/9,266,
mask288 has578/4,658. Encoder rows use native features without PCA afterward.

Maximum210 pipelines/420 heads, including120 native pipelines,10 driver raw
outer fits and50 helper outer fits. Retain unsupported statuses, undefined
view scores, coverage, all five cohorts and all three repetitions, marginal
and within-master paired source intervals. Do not average interval endpoints
or treat head repetitions as independent encoder runs.

Retain twenty0/512 model points. Four native methods times three TRAIN/intact/
deleted surfaces times two tasks times five cohorts give120 distinct full CUDA
native exports. They are distinct architecture roles, with no shared-initial
counterpart argument. Positive timing original TRAIN/intact VALIDATION queries
add20 writers/80 Q-masked CUDA forwards under the unchanged four patch banks.
No point0 or amplitude reconstruction scores. No CPU encoder inference repeats.

## Admission and saved evidence

Artificial actual-CUDA admission proves source0 old literal encode/forward/
update and settings/archive parity; source1 support/semantic/hidden isolation;
exact BD32 decoder intervention; finite CUDA input/loss/gradients and actual
updates; reconstruction gradients through W/mixer/temporal/pool/global paths
and absent diagnostic-projection gradients; typed checkpoint/export roundtrip
and malformed-tag rejection before weights; and historical-factory rejection.
The engineering namespace is `pooled-context-engineering-v1/lag_sign`.

Trainer admission additionally proves exact shared initialization/scaler and
rejects wrong source/seed/namespace/config/positive reference; direct0→4 versus
0→1→2→4 model/buffer/AdamW/counter/trace equality; inactive projection unchanged
with no state; immutable snapshots/bytes; no RNG/thread disturbance; and abort
on decreasing/skipped/exhausted budgets. Preserve failed gates.

Each point has five roles: checkpoint, `.audit.pt`, `.scaler.pt`,
`.training-raw.pt`, `.continuation.pt`. New continuation kind is
`rpb_pooled_context_continuation_state_v1`. Typed companions and a paired initial
gate record bind role/input-source/config/policy/source IDs, complete named
initial/current parameters/buffers, original scaler/raw/source association,
copy-reference metadata, actual AdamW state presence/step/moments and counters.
No gained targets/manifests or invented optimizer moments.

The independent CPU checker validates the closed role/schema/shape/source matrix
before payload reads, initial common/nonshared and inactive state, original
data/scaler/support/query arithmetic, fixed saved heads/logits/argmax and
intervals. It performs no model inference, autograd, optimizer, head/PCA/SVD fit.
Checkpoint bodies are byte-bound without CPU CUDA-storage decoding. CPU state
witnesses plus actual CUDA admission establish association, not independent
AdamW or model trajectory replay. Freeze artificial fixtures and the FALSE reader
triple before quality; release a new flags-only copy after completed inventory.
Float64 replay uses additive abs2e-9+rel2e-9, float32 abs2e-6+rel2e-5;
support, discrete associations, zeros, own-logit argmax and equality are exact.

## Decision and reporting

Use the agreed quality/training tables, descriptions beside them, separate
timing/amplitude and both views. Fixed-query errors are original standardized
MAE; retain objective/gradient/eligibility traces separately. Report registered
and actual optimizer counts and separate synchronized CUDA-loop, native/query,
binding/I/O, CPU head/bootstrap and independent audit scopes. Pure kernel time
is unmeasured. Cost and quality are descriptive under this fixed protocol.

Joint direction requires timing linear means and worst-cohort scores no worse
in BOTH views, equal common coverage, and original TRAIN/VALIDATION query MAE
means no worse. Retain every cohort/interval and secondary neural/amplitude
tradeoff. A failure stops this route without width/rate/budget/head rescue.
A pass requires separately frozen fresh confirmation before promotion. Strong
untrained amplitude accuracy limits learning-credit claims. Preserve original
RPB-v7, active v4 and all alternatives; no automatic promotion or TEST/stress.

Generate quality only after source review, CUDA admission and the complete
prequality independent reader are ready within the remaining work window.
