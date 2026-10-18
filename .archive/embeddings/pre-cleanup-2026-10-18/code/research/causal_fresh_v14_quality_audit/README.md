# Independent saved CPB-v14 quality audit: source preparation

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

Four whole completed V13 quality-auditor parents were preserved and copied before
editing. The inert archive reader remains byte-identical. This auditor uses only
standard-library scalar arithmetic and its own reader. It imports no producer,
reporter, comparator, training auditor, encoder or head implementation. Source
preparation admits no AST, imports, model/head calls or numerical audit body.

V13 supplies encoder SOURCE only. V14 replaces four disjoint original-position
local supports with nested trailing supports [12,16), [8,16), [4,16), [0,16),
ordered shortest to longest. The edge16 is the retained H128/P8 grid after the
public cutoff crop, retention and left padding; it never follows packed length,
valid count or last observed value. Sparse patches retain original positions.
Each present window retains the actual two-sum half-local/half-global attention
mean and attention-coupled centered variance, then actual F32
sqrt(v+2^-10)-1/32. Longest local and global coincide but remain two sums.
The global half and causal hidden states include earlier past, so these windows
are temporal emphases rather than isolated32/64/96/128-minute receptive fields.

Native40, parameter shapes/counts/registration/draws, raw-bank tail20,
objective, shared clip1, single AdamW, fresh TRAIN statistics, streams,
2048 successful updates/batch8 and fixed heads remain unchanged. No old learned
weights, optimizer or fitted statistics are imported. Both ordinary/contextual
paths retain the full live hidden/center/score/position adjoint and finite-F32
center-residual correction. Empty local windows zero all eight served rows after
bias. All-padding raw roots remain1/32; transformed dispersion is zero.

Current records bind the new pooling policy, geometry[4,16,32,8], trailing
lengths[4,8,12,16], mix[.5,.5], moment partition[4,4], export layout, transform
and fixed tau. Old disjoint partition metadata is refused. Inference root/payload
and TRAIN root/encoder/encoder.payload require exact String and typed Long4
geometry/length vectors, plus exact transform String and Float64[1] tau.
Wrong/missing/old/type/rank/shape/order/value cases refuse, including old extra
partition keys alongside otherwise valid new metadata.

Source-derived prospective counts are53 direct temporal-kernel calls,
18 F64 and18 CUDA-F32 finite-difference coordinates, four isolated gradients,
one backbone encode, zero public encoder/head calls and one synthetic AdamW
step. Research primary declares821 typed refusal cases:817 full saved corrupted
archives and four request refusals, with zero compact fixtures/restored controls.
Five cases explicitly exercise legacy extra partition metadata. Manual3e-5,
F64 finite-difference2e-8 at step1e-5, F32 finite-difference2e-3 at step2e-3,
gauge/bin-mass2e-6, flat8e-6 and analytic slope bound16 remain fixed.
The thirteen required new fixture flags include fixed-edge nested membership,
exact per-window local captures, longest-window two-sum behavior and padding
membership invariance. Counts and predicates here are source contracts, not passes.

Exactly two existing same-forward actual-step tensor fields now hold per-window
local weights [R,4,T] instead of the old aggregate SUM [R,T]. Field cardinality
remains60 with25 final-only fields; the aggregate pool-weight diagnostic remains
a SUM, not a single probability. The separate independent TRAIN auditor owns
full captured-adjoint arithmetic. No mixture inversion or extra forward is used.
Fixed-quarter position-contrast CSV groups remain original-quarter diagnostics;
they are not the new overlapping local supports.

Saved quality arithmetic retains112 TRAIN/32 known chronological VALIDATION
prediction checks, ridge coefficients/normalizers, physical MAE/RMSE, all three
neural repetitions and TEMPO cohort/group/interval/count/mean/coverage reductions.
Neural predictions are DATA, never forwarded. D40 head recipes, seeds and all
numerical scoring kernels are unchanged; no PCA, tuning or old model replay.

Thirteen initial references are saved V13 DATA: eight feature archives compare
native20:40 and complete support/IDs/labels, while five fresh statistic archives
compare whole logical contents. Learned0:20 is unpaired; no whole40 equality is
claimed. Eleven final controls V3-V13 and eight separate trust joins V6-V13 keep
their original semantics: whole40 for V6/V7/V10 and partial20:40 for
V8/V9/V11/V12/V13. Historical V9 report recovery remains opaque. Thirteen
comparison methods yield52 market rows and four TEMPO panels; the primary own/raw
report still has16 market rows.

The actual separately reviewed and applied prospective V3-V13 protocol must bind
the future main/audit cards. All22 ordered criteria copy the reviewed design:
five V13 floors tighten, all other17 whole criterion objects stay unchanged,
three positivity rules and1e-12 operators remain, ties retain earliest leaders.
TRAIN NLL adds no gate; historical counts, including V13's original8/22, are not
recomputed. The old866 protocol/review is ancestry, never current authority.

Future execution separately binds all four auditor sources, actual main card,
input/source manifest/ZIP, source count, seven binaries, eleven translation units,
five scientific plus two aggregation stages, complete mirror and explicit paths.
External hashes precede arithmetic; sources must remain unchanged. This source
work creates no audit card or experiment card. Runtime trust, partial initial
pairing, audits, coverage, quality and measured establishment remain distinct.
V3 remains the clean baseline and sole DEV encoder. Every old source, result,
checkpoint, failure and scoped leader is preserved.
