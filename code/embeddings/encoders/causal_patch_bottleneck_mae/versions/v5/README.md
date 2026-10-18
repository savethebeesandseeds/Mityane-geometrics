# CPB-v5: independent mean and uncertainty coordinates

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
Inherited notices remain in the complete copied files and LICENSES/NOTICE.

Source implemented and frozen for independent preflight; no build, calibration,
model execution, training, causal proof or quality pass is claimed here. The
prospective protocol is archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FRESH_V5_CARD.json. CPB-v3 remains the clean
baseline and application encoder. V4 and every historical artifact remain intact.

This directory owns all48 files copied from the V4 architecture/parser closure,
with V5 namespace, include paths, configuration and typed artifact identities.
Only Torch and the C++ standard library are external runtime code dependencies.
The canonical include is embedding/encoders/causal_patch_bottleneck_mae/versions/v5.h
through this directory's include root. The sibling versions/v5.h is a complete
byte-identical public-header copy for source discovery, not a forwarding shim.
The provenance receipt pins the actual V4 source bytes and current owned copies.
Architecture copying imports no learned tensor, optimizer or fitted statistic.

Native40 consists of parent32 [0:32], mean4 [32:36] and uncertainty4 [36:40].
Both four-coordinate projections read the same frozen TRAIN-standardized raw
relative bank270, through independent learned matrices. The existing mean
projection and parent/residual decoder retain their live path. The new bias-free
uncertainty_projection has shape [4,270], adding1080 learned values. The conditional
variance head [27,4] plus27 biases reads live uncertainty4 only. Mean decoding does
not read uncertainty4; variance decoding does not read parent32 or mean4.

The numerical Gaussian NLL and its complete joint gradient remain in use.
In particular, the mean residual receives error/variance weighting; there is no
Huber replacement, detached likelihood or scalar-valued gradient surrogate.
The gradient policy is:

    separate-mean-and-uncertainty-bank-projections-live-gaussian-nll-v1

The new projection is registered and initialized after every existing learned
constructor and variance-head parameter initialization, using uniform bounds
+/-1/sqrt(270). Thus all preceding seeded constructor draws remain paired. Fresh
initialization contains76 learned tensors /294752 values, one fixed analytic
spectral prior /3888 values, and77 total tensors /298640 values. C3/H128/F9/P8,
mean4 geometry and the mean decoder remain unchanged.

fresh(Config,FrozenScaler,BankStandardizer,seed) is the sole initialization API.
It directly constructs the model under the supplied seed, with source_updates=0,
causal_updates=0 and pretrained_ancestor=none. No predecessor importer exists.
The caller freshly fits both statistics on declared TRAIN observations before
construction. The bank identity binds the fresh scaler. Training accepts only
those frozen statistics; separately declared frozen-inference scaler policies
can retain the unchanged market bank. No old fitted statistic is imported.

Raw observations use genuine CPU F64 values, boolean masks, semantic channel IDs,
actual uniform timestamps and explicit cutoffs. Cropping precedes validation,
casting, scaling, bank extraction and masking. Original patch-order temporal
attention is causal. Context deletion cannot create reconstruction targets.
Gaussian NLL arithmetic is CUDA F64 after promoting F32 distributions and targets.
Only literal prior parameters are fixed; inherited learned paths contain no
NoGrad guard. The retained standalone separated-change helper is live as well.
Frozen public inference and initialization keep their intentional NoGrad guards.

Ordinary inference accepts cpb_v5_inference_checkpoint_v1; resume accepts
cpb_v5_causal_training_state_v1; the nested architecture payload accepts
cpb_v5_location_scale_decoder_payload_v1. Each binds the V5 gradient policy,
explicit native40 layout and widths. Old V4 policy/layout and mismatched widths
are refused. Serialization retains seed/initial-state/statistic/model bindings;
resume also binds dataset/split/recipe, parameter order, optimizer moments and
presence/steps, update/attempt counters and deterministic streams. Saves require
matching live counters and never overwrite existing artifacts.

The complete owned family fixtures and Make fragment remain source available.
The prospective V5 research capsule explicitly compiles its independently owned
research/tests/v5_test.cpp and v5_separation_test.cpp, rather than this family's
fixture targets. Those active fixtures cover the full causal matrix, actual
CUDA learning, typed actual-trajectory resume and nonvacuous independent gradient
routing. Both actual initial and final saved checkpoints require frozen causal
proof before the fixed heads. Source creation alone proves none of those passes.
