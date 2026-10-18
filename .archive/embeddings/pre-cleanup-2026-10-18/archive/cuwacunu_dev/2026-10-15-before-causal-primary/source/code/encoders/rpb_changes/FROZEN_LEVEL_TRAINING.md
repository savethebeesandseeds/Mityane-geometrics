# Frozen-level RPB-v23 training recipe

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

The additive `consecutive_change_frozen_level.h` helper trains the change
projection while retaining a typed v22 parent's entire level/relation model
and decoder. It changes training policy, not inference arithmetic or archives.
Use a distinct instance tag and recipe identity; do not relabel the parent.

Load the parent with `load_content_ordered_relation_checkpoint`, validate its
hash/config/scaler/update count, construct a zero-start v23, then call
`consecutive_change_training::warm_start_level(model,parent.model)`.
The helper copies parameter/buffer values by name into the registered child.
It rejects config mismatch and a nonzero change projection. It copies weights,
not parent optimizer state. Check initial native parity separately.

Construct a fresh standard AdamW over the complete registered parameter list,
then a `FrozenLevelGuard`. Preserve requires_grad flags: the model validators
and archive format depend on them, and the decoder must propagate gradients
through native32 to the new projection. After each backward call
`guard.discard_gradients(model,optimizer)` **before clipping and optimizer.step**.
This makes all child gradients undefined. Zero tensors would still permit
AdamW weight decay and moments, so they are insufficient protection.

Call `guard.validate(model,optimizer)` before training, at segment/checkpoint
boundaries and after the final step. It checks exact parameter/buffer values,
registration identities, flags, absence of child gradients and AdamW states.
Discarding gradients also rejects existing child optimizer state on every
update. Only the new projection should acquire moments. Full backward still
computes child gradients before discarding them; no compute-saving freeze is
claimed. Protection preserves child weights, not the complete exported
embedding: the learned20 coordinates can still change through the residual.

Save/load with the existing typed v23 inference/training helpers. Training
recipe identity must include the parent hash, inherited exposure, new budget
and freeze policy. After training-state load, rebuild a guard, validate the
parent values against the declared parent, and use this same discard/step
sequence. A generic resume loop that omits the guard would train the child.
Inference has the ordinary v23 semantics; instance lineage lives in the
recipe/run receipt. Its update count denotes this new branch segment, not the
sum of parent and branch exposure.

The focused artificial CUDA fixture verifies typed warm-parent parity,
nonzero/mismatched restart rejection, branch learning, exact protected values,
no child moments/decay, typed inference reload and exact next resumed loss and
weights. Integration target: `test-v23-frozen`; mirrored research target:
`test-rpb-v23-frozen`. A mirrored fixture is not a second test execution.

The separately declared market experiment is
[RPB-v23.market-01.alt-01](../../../doc/MARKET_FROZEN_CHANGE_CARD.md).
Its market loader, prospective card, runner and evaluation remain outside the
portable model. The primary v22 configuration and existing sources/archives
are preserved.
