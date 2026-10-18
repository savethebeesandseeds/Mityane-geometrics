> Source implemented, pending admission: complete owned CPB-v4 copy of CPB-v3 with only variance_input_gradient_stop selected. Forward distribution values and numerical Gaussian NLL are unchanged; the encoder receives the partial gradient through live mean/residual paths. The prospective card, input manifest and execution admission are pending; no build, calibration, learning, proof or quality run has occurred.

# CPB-v4: causal encoder trained from scratch

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
Inherited source notices remain in their copied files and LICENSES/NOTICE.

This version owns a complete physical copy of the architecture/parser closure
under its own namespace and include paths. Its public entry is
`embedding/encoders/causal_patch_bottleneck_mae/versions/v4.h`; its namespace is
`embedding::encoders::causal_patch_bottleneck_mae::v4`. Only Torch and the C++
standard library are external runtime code dependencies.

`fresh(config, TRAIN_scaler, TRAIN_bank, seed)` is the sole initialization API.
It constructs all75 learned tensors (293672 values) directly from the seeded
constructor. The3888-value literal spectral prior is deterministic and fixed.
No checkpoint path, pretrained model, tensor map, historical scaler identity or
historical normalizer identity enters this factory. There is no predecessor
import API. Source updates are0 and the pretrained ancestor is `none`.
Zero/one initialized biases, normalizations and heads are part of the fresh
constructor; they are not copied from a saved model.

Native36, C3/H128/F9/P8, all297560 parameter values and causal forward arithmetic
remain unchanged. The only mechanism change is variance_input_gradient_stop:
conditional variance uses the detached four change coordinates in its matrix
product. Its108 projection values and27 biases remain trainable; the mean and
residual decoder retain live change coordinates. Numerical Gaussian NLL is
unchanged, but the encoder receives its partial gradient through the live mean
and residual paths rather than the full joint NLL gradient. The75 learned tensors
(293672 values),3888 fixed prior values and fresh initialization remain intact.

The caller fits new input and relative-bank statistics on its own declared
TRAIN observations before calling `fresh`. The frozen bank must bind that
scaler's actual identity. Training requires exactly these frozen statistics;
frozen inference can use a separately declared external TRAIN scaler without
changing the checkpoint's market bank. No historical statistics are imported.

Raw observations are genuine CPU F64 with boolean masks, semantic channel IDs,
actual uniform timestamps and explicit cutoffs. The boundary crops values,
masks and target sources before validation, casting, scaling, banks or masking.
Temporal attention is causal in original patch order; channel mixing occurs
only at the same patch. Additional context deletion removes visible inputs and
never creates reconstruction targets. The unchanged Gaussian NLL is calculated
in CUDA F64 after promoting F32 distributions and standardized targets.

Ordinary inference accepts only `cpb_v4_inference_checkpoint_v1`; ordinary
resume accepts only `cpb_v4_causal_training_state_v1`. The nested payload also
uses a CPB-v4 kind/architecture and the exact gradient_policy. Both bind fresh initialization kind/seed/initial
state, zero source updates, current updates, dynamic TRAIN statistics and
model state. Resume additionally binds dataset/split/recipe, optimizer parameter
order, exact moment contents and presence/steps, counters and three streams.
No save overwrites an existing artifact.

The engineering command is:

`v4_test --settings CONFIG --output NEWLEAF --seed 943001`

It generates and fits its own32-row synthetic TRAIN-only fixture, constructs
the model and an independent direct same-seed constructor before opening any
checkpoint, compares every tensor, saves the first artifact, and verifies exact
reload. It then tests actual CUDA learning and exact next-update typed resume,
refuses malformed/ancestry/policy/statistics/moment/counter artifacts, and proves
all128 cutoffs/2350 future-value-or-mask cases/30 internal temporal cases before
and after two engineering updates. Its extra next-step witness is identified
separately; this is not the declared market training trajectory.

`v4_test --saved-checkpoint FILE --settings CONFIG --output NEWLEAF`

requires an own typed artifact with either0 or2048 causal updates and repeats
the complete frozen causal matrix with zero new updates. This mode is invoked
on both actual main-run initial and final checkpoints before quality.
`--final-checkpoint` is an alias requiring2048 updates.

This implementation is pending source admission/build/runtime verification.
No passing test or measured quality follows from source creation. The separate
prospective card is `doc/CAUSAL_FRESH_V4_CARD.md`. Earlier v1/v2 source,
checkpoints and measurements remain protected historical references. They are
not v4 training sources or quality targets.
