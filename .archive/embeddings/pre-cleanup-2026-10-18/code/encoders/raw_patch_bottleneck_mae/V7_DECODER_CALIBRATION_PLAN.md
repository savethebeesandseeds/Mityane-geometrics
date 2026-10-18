# V7 decoder calibration diagnosis

Date: 2026-10-08. Prospective; not measured.

Continue from the five retained RPB-v7 instances, keeping RPB-v4 as the active
reference. V7 has strong native timing readouts but worse ordinary-context
reconstruction on every master. Before changing the encoder, test whether its
existing decoder can recover the archived v4 reconstruction threshold while
the embedding remains exactly unchanged.

Freeze every encoder parameter, buffer, configuration and TRAIN scaler. The
four `decoder_*` module families contain 11,528 trainable parameters; the other
214,277 parameters remain frozen. Use `model.eval()`, a no-gradient encoding
of original visible context V=O&~A, and gradient-enabled decoding of the exact
detached served32 export. Preserve Q=O&A and the original hierarchical Huber1.
No extra context deletion, agreement, projector or additional decoder capacity.

Starting from each saved v7@512, run exactly128 fresh decoder-only AdamW updates:
batch8, rate0.001, weight decay0.0001, clipping1, threads1, dropout0. Use original
absolute row/mask/Torch counters512..639; abort on any skip. This is a fresh
decoder optimizer, not a resumed encoder optimizer. Save decoder stage0/128,
moments, counters, loss/evidence and immutable parent binding in distinct
calibration artifacts. Original encoder updates stay512. Ordinary checkpoint
formats, training defaults and tagged-resume rejection remain untouched.

Reuse the shared four-patch TRAIN/known-VALIDATION reconstruction writer and
cached classification records. No classifier or PCA fitting; no CPU encoder
training. Needed model inference and decoder optimization use CUDA. Admission
must prove frozen encoder state/native32, CUDA decoder gradients/updates,
original query and scaler support, hidden-value isolation, direct/split live
optimizer continuity and immutable saved/reloaded calibration snapshots.

The separate frozen [card](../../evaluation/cards/v7_decoder_calibration_v1.md)
defines the exact85 permitted parent roles and one fixed question. Compare
stage128 mean original TRAIN/VALIDATION fixed-query MAE with archived v4@512.
All five masters and per-master tradeoffs stay reported. No equal-total-compute
claim, TEST/stress, selection, promotion, or rate/budget search.

A successful result shows decoder recoverability for these frozen v7 features
and this existing decoder. It does not improve the embedding or establish
generalization. A failed result does not prove encoding information loss under
a finite decoder budget. Do not assign a new embedding tag to an unchanged
encoder. Record this as an RPB-v7 decoder-calibration diagnostic.
