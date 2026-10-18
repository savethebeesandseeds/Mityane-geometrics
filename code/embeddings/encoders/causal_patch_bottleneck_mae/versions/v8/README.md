# CPB-v8: four ordered temporal summaries at unchanged capacity

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
Inherited notices remain in every copied file and LICENSES/NOTICE.

This source-only candidate is pending independent admission, build and runtime
proof. No quality result or promotion is claimed. CPB-v3 remains the selected
clean baseline/application. Preserve all V3–V7 strengths and historical bytes.
The prospective experiment card is archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V8_CARD.json.

The version owns complete physical V6 copies under its own include paths,
namespace, settings, fixtures and typed kinds. The external versions/v8.h is a
full byte-identical public-header copy. No old implementation is included or
forwarded. Source ancestry does not import weights, optimizer state or fitted
statistics; all learned values and permitted TRAIN statistics start fresh.

The sole architecture change replaces each ordinary and early-contextual
single temporal softmax with four separately normalized original-position bins.
For H128/P8, bins contain original patch indices0–3,4–7,8–11,12–15, oldest first.
Each bin pools the same 64-wide causal hidden states, then uses only the
corresponding eight rows of the existing32×64 export_projection and its bias.
Concatenated slices retain32 coordinates. The existing global99→64→32 pool,
native_shape_projection20×32, ordered relation contribution, fixed spectral12,
mean4 and uncertainty4 banks and location/scale decoder are unchanged.
Original indices determine bin membership after packing/deletion/left padding.
Later-bin states can contain earlier causal context; bins are anchored summaries,
not independent windows or frequency bands. Global/shape layers remix them.

Empty bins use a finite softmax fallback, zero masked weights and zero slices
AFTER projection bias. Both actual temporal paths call one owned pooling
kernel. Temporal traces expose bool pool_bin_valid/contextual_pool_bin_valid
[active_rows,4]. Each nonempty bin has weight mass1; an empty bin has mass0.
The flat pool_weights trace therefore totals the number of nonempty bins,
not a fabricated global1. Legal profiles require padded original patch count
and export width divisible by4 and the served compact-export mode2/source0.

All constructor registration/draw order and tensor shapes remain V6's:
76 trainable tensors/294752 values plus one literal fixed prior/3888 values,
77 tensors/298640 total; native40 = parent32 + mean4 + uncertainty4. No new
parameter or random draw is introduced. Initial content20 intentionally changes;
initial fixed spectral12/mean4/uncertainty4 and five fresh-stat archives are the
paired numeric controls. Own initial fixed heads must still be evaluated.

Training remains V6's unweighted F64 Huber1 mean reconstruction plus unit-weight
Gaussian variance calibration on detached mean error, unchanged supported-cell,
eligible-channel/example reduction, AdamW .001/.0001, shared global norm clip1,
2048 updates/B8 and paired seeds. Only total objective is backpropagated;
ordinary Gaussian NLL remains a detached reporting value. Shared clipping and
future residual/statistic coupling remain; no optimizer-independence claim.

Inference outer, payload, TRAIN root and its nested encoder/payload bind
UTF-8 temporal_pooling_policy = original-position-four-bin-disjoint-export-rows-v1
and Long[5] temporal_pooling_partition_v1 = [4,K,D,K/4,D/4] (default[4,16,32,4,8]).
Public temporal_pooling_partition(config) returns std::vector<int64_t>. Missing,
altered, wrong-shape/dtype fields and old versions are refused alongside the
unchanged objective, geometry, state, optimizer and deterministic-stream binds.

Data-learning admission prospectively excludes ONLY the one-value common
pool_score.bias softmax-shift gauge from a nonzero-gradient requirement. It
remains registered, requires_grad and optimized; its observed gradient/movement
is logged without calling roundoff or AdamW decay data learning. The other75
tensors need actual nonzero CUDA data gradients and movement during the actual
trajectory. pool_positions has four within-bin constant-offset gauges, so all
four within-bin-centered position-gradient contrasts and final contrast changes
must also be witnessed. Each of four eight-row export weight/bias slices needs
nonzero data-gradient events and final movement; empty-bin steps are legitimate.
This makes no assertion that every scalar parameter is identifiable.

The owned temporal fixture checks independent scalar pooling/adjoint parity,
original-index gaps, empty/single/full bins, unsupported NaNs, after-bias zeros,
common/per-bin score-shift invariance and one actual AdamW step. F32 manual
absolute/relative tolerance is2e-5; gauge tolerance2e-6; per-bin mass2e-6 and
flat mass8e-6; empty/padding zeros and support membership are exact. It performs
one owned backbone encode for both real temporal paths, two direct actual-kernel
calls and zero public encoder calls/head fits. It does not establish whole-model
learning; complete causal/CUDA/resume and the one actual trajectory do that.
The retained objective fixture proves composite-loss gradients and shared-clip
coupling. No numerical fixture has yet been executed.
