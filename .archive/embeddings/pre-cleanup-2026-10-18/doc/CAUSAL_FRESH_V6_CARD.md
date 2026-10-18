<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v6.market-01: fixed mean supervision and variance calibration

Declared before build, calibration, training or quality. The
[machine card](CAUSAL_FRESH_V6_CARD.json) is authoritative. CPB-v3.market-01
remains the clean baseline and sole dev encoder; every V4/V5 scoped strength,
source, failed attempt, result and local ignored checkpoint remains protected.
The prospective version descriptor is immutable and will be written after the
card hash freezes, before source admission or measurement. A later measured
instance record is additive.

The audited [saved TRAIN diagnosis](CAUSAL_V4_V5_TRAIN_DIAGNOSIS.md) motivates one
controlled objective change. V5 lowers all 112 TRAIN MAE but has less favorable
TRAIN NLL than V4; its first 8 witness moves differently. Inverse variance still
reweights mean pressure, and same-mean4 NLL/Huber/MAE directions differ while
remaining positively aligned. Volatility readout contributions shift toward
uncertainty4, while activity parent/mean contributions reinforce. This proves
neither a cause of held-out losses nor the next objective's effectiveness.

Keep V5's complete native40 forward geometry, parameter initialization order and
capacity: learned20 + literal spectral12 + mean4 + uncertainty4, with separate
270-to-4 bank projections. The mean decoder reads parent32+mean4; variance reads
live uncertainty4. There are 298640 parameter values: 294752 learned in 76 tensors
and 3888 fixed prior values, 77 tensors total. No learned capacity or readout seed
changes relative to V5. V3/V4 native36 capacity/seeds still differ.

Let R denote the original equal-example/equal-channel/supported-target-cell
reduction. Use the fixed objective

`L = R[Huber_delta1(mu-y)] + R[0.5*(stopgrad(mu-y)^2/v + log(v))]`.

Both coefficients equal 1, without tuning. Sanitize unsupported operands first;
promote each F32 operand separately to CUDA F64 before subtraction/arithmetic.
Keep the anchored positive softplus link, its constants and support masks.
Huber1 supplies the live mean/parent gradient. The variance term has zero direct
mean gradient while the independent uncertainty projection and variance head
stay live. This is a distinct objective, not the full joint Gaussian-NLL mean
gradient, a coefficient blend or a rescue of V5. The variance component and total
may be negative. Unchanged full Gaussian NLL and MAE are detached reporting
metrics; reporting NLL is never backpropagated.

Keep one global gradient-norm clip at 1 and one AdamW optimizer. Shared clipping
retains optimizer coupling; changing the mean also changes future residuals used
by variance calibration. Zero direct variance-NLL mean gradient does not make
optimization independent. Huber supervision can hurt rare large errors, NLL,
volatility or timing; all retention failures must remain visible.

Freshly construct every learned value under 943001 with the same V5 draw order.
Fit permitted TRAIN input/bank statistics from raw data anew; use a fresh AdamW
state. No old checkpoint, teacher, trained tensor, optimizer or fitted statistic
may initialize the model. source_updates=0. Learned paths remain trainable;
only the literal prior and newly fitted/frozen statistics are protected.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history reconstruction
and past-window retention · **complexity 4/5**. Keep 4096 TRAIN anchors January 1–21,
recent128/C3/F9/P8, width64/three layers/four heads/feedforward256/decoder128.
Fresh rawF64 input and causal bank calibration uses the unchanged population
recipe, scale floor1e-6, zero encoder calls and no held-out statistics. Keep 112
TRAIN/32 known chronological VALIDATION probes and saved 1/16 query masks.
January 29–31 and new periods remain reserved. No forecasting claim is made.

One fresh trajectory: 2048 successful CUDA:0 updates/B8, maximum 4096 attempts;
AdamW0.001/decay0.0001/global clip1/dropout0, query25%/context deletion15%, seeds
943001/943002/943003/943004. Save inference0/2048 and typed TRAIN512/1024/2048;
resume the actual trajectory at 512 with exact states/moments/counters/order and
no repeated updates. Record total objective, mean Huber, variance NLL, reporting
NLL and preclip/scale coupling per step, with first/last existing-forward
component witnesses. Report synchronized step time separately from I/O,
calibration, proofs, extraction, head fits and audits.

Before quality, require actual CUDA learning in every one of 76 learned tensors,
with 3888 literal prior values and all fresh statistics protected. Prove every native40 coordinate causal
at all 128 cutoffs/2350 future-value-or-mask cases/30 internal temporal cases,
before/after engineering updates and on actual saved initial0/final2048. Include
future nonfinite poison, positive past interventions and frozen-state witnesses.
Typed inference/training must refuse old kinds/layouts/policies and malformed
objective, delta, coefficient, detach, direct-route or reporting-NLL fields.

A nonzero CUDA fixture must prove unchanged numeric forward mean/variance,
live Huber mean gradients, zero direct variance-NLL mean gradients, zero direct
Huber uncertainty gradients, live uncertainty/head/projection learning, manual
composite-objective native40 gradient parity, shared clip coupling and an actual
optimizer step. This is not manual parity with the old full joint-NLL mean
gradient. Bind exact source/compiler/binary/input/checkpoint closure.

Both saved stages receive their own new embeddings and unchanged fixed heads;
no initial-score reuse, post-encoder PCA, head tuning or intermediate selection.
Require 13 numeric initial pairings against V5 DATA/statistic archives: all 40
coordinates in eight feature archives plus five newly fitted statistic archives.
This is saved-data comparison, not an old model call, initializer or statistic
import. Version/file metadata may differ. Pairing and independent training/
quality audits are mandatory trust before establishment; no pre-head parity
claim is made without an actual verified receipt.

Market heads are four-output ridge1 and D40→16tanh→4, Adam0.01/100 updates,
164/724 parameters. Repetitions 2701/2802/2903 keep seeds
6049515622977958931 /8262941649267994858 /2095371932330426401. Three head fits
on one encoder are not independent encoder replications.

Dataset: **TEMPO-5** · slow/fast nine-feature lag-sign timing · **complexity 5/5**.
Use saved cohorts 941101/941102/941103, each 512 TRAIN/256 VALIDATION examples from
256/128 paired source groups, zero synthetic encoder updates. Score intact and
extra30%-deleted views under newly fitted synthetic-TRAIN and market-TRAIN input
scalers. Keep ridge1/tanh16/Adam0.01/100, the same D40 seeds and 82/690 parameters.
Fresh cohort scalers are shared by the saved initial/final stages.

Advancement requires all 22 criteria in the separately reviewed
[V5-retention protocol](clean_causal_encoder_search_goal_v5_retention.md), exact
machine SHA256 `47fa165d325230ffca4f99c19bbec494d04f1daf0d99ae7e6bd197fdc47bec0d`,
plus complete trust and 100% coverage. Protect the best clean final V3/V4/V5
market/timing scopes and query MAE, strict V4 VALIDATION NLL improvement and
unchanged positivity rules. No TRAIN NLL gate or V1/V2/RPB oracle floor is added.
The explicit native40 declaration overrides the inherited generic36 default
under the protocol's future-dimensions rule. Historical pass counts retain their
old floors; do not relabel or compare them as identical criteria.

Use only the existing managed GPU cuwacunu container and exclusive source/build/
run leaves. Preserve concurrent Latent Roadmap work and dev's selected V3.
No coefficient/seed/rate/budget/head rescue, automatic promotion or commit/push
is authorized. Stop this exact recipe at its fixed final measurement. The clean
research goal continues across separately declared mechanisms.
