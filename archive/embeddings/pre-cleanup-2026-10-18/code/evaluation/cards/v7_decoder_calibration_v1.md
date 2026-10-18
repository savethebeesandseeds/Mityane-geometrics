# V7 decoder calibration v1

Date: 2026-10-08. Prospective frozen diagnostic; policy1.2.
Protocol: `v7-decoder-calibration-v1`.

Question: can the existing decoder of retained v7@512 recover archived v4@512
ordinary reconstruction after128 decoder-only updates, with the encoder exactly
unchanged? This is a sufficiency diagnosis, not an improved embedding,
equal-budget comparison, acceptance, checkpoint selection or promotion.

## Frozen recipe

Five known masters4404/5505/6606/7707/8808, timing `lag_sign`,256 TRAIN examples /
128 source pairs and128 known VALIDATION examples /64 pairs each. C3/H32/F3,
patch8, width64, three temporal blocks, four heads, feed-forward256, mixer1,
global mode2, decoder128, native32, all225805 parameters. Natural missingness0.1.
Original encoder training remains512 updates.

Freeze all214277 non-decoder parameters, every buffer, input/serving config and
original TRAIN scaler. Optimize only11528 parameters in decoder_positions,
decoder_channels, decoder_first and decoder_second. Start from their saved
v7@512 values. Fresh decoder AdamW0.001, weight decay0.0001, clipping1, batch8,
threads1, dropout0; exactly128 unskipped updates. Use original absolute counters
512..639 for sampled rows, whole-patch mask and Torch streams, with the original
actual training seed. No replacements, rate/weight/budget/seed search or E.

Original observations O, masks A, eligibility and Q=O&A are unchanged. Encoder
sees V=O&~A with hidden storage isolated, under evaluation mode and no gradients.
Detach its exact native global32; gradient-enabled decoder uses original
hierarchical Huber delta1 on Q. No classifier, projector, predictor, EMA or PCA
after the encoder. No model is trained on CPU.

Save stage0/128 as typed decoder-calibration assets composed with their exact
immutable parent. Record original encoder512 and decoder128 separately. Save
decoder AdamW moments, absolute sampled indices, support, losses, gradient norms
and synchronized training-loop seconds; include evidence-copy time explicitly.
Do not represent this as ordinary encoder resume or erase the v7 training tag.

## Exact legal inputs before payload access

Parent v7: `output/runs/rpb-context-lighter-validation/lighter-validation-duyRU2`,
inventory SHA256 `d4b38150814d65b21712a6371348b76bbc0d539a60009152932d293557b45821`,
producer `f22cc8d5f2ac8af3e6a6056c6ad2693ffcc82cf39e8744c32fb13038710c5311`.
Prefix `results/candidate-development`.
Parent v4: `output/runs/rpb-context-replication/context-replication-JjNEUc`,
inventory SHA256 `13b073116eab4f7da78e15eee93eb4a74e7ffda482d9369ed7615301c4ce14f9`,
producer `9cab6262d79ddb29bbd8d4d380a7153d11048720fe031fab7e1d4e9d05c0d828`.
Prefix `reference`.

Closed template, no recursive payload discovery:

- V7 prefix: native-development-card.json, development-complete.json,
  validation-report.json (three roles).
- Each V7 prefix/seed-M-lag_sign: controlled-training.pt,
  controlled-validation.pt, development-manifest.json, trainer-audit.json.
- Each V7 cohort/milestone-512: checkpoint.pt and its .audit.pt, .scaler.pt,
  .training-raw.pt companions; native-training.pt, native-validation.pt,
  point.json, training-reconstruction.pt, validation-reconstruction.pt.
- V4 prefix: native-development-card.json, validation-report.json (two roles).
- Each V4 prefix/seed-M-lag_sign/milestone-512: point.json,
  training-reconstruction.pt, validation-reconstruction.pt.

Exactly68 V7 and17 V4 roles,85 total. Generate metadata-only role plan first;
validate every role/redirect against the entire closed set before the first
payload hash/read. Bind allowed bytes to pinned parent inventories and verify
before/after. No point0, head-fit/prediction, TEST/stress or historical report.json
payload. No borrowing prior cards' input permissions. New outputs are exclusive,
source/card/admission identities frozen before quality fitting; retain failures.

Encoder fitting receives only legal TRAIN observations/metadata, no labels,
hidden clean references, VALIDATION inputs or evaluation query arrays. Verify
parent raw TRAIN/scaler/data/source order against the permitted split. VALIDATION
and reconstruction references are scoring only. Cached native features are
protected references, not decoder fitting inputs.

## Measurement and decision

Use unchanged original four-patch fixed queries and hierarchical standardized
MAE/Huber on TRAIN and intact known VALIDATION at decoder stages0/128. Require
stage0 exact same-backend query/support parity with saved v7 arrays. Prove all
non-decoder parameters/buffers/scaler unchanged and CUDA native32/support exact
before/after; use existing saved classifier results without any fits or new
accuracy computation. Encoder invariance and unchanged serving code justify
retained classification, explicitly labelled reused and state-bound.

Recovery requires mean stage128 TRAIN MAE <= paired v4@512 mean0.0675931436195981
AND VALIDATION MAE <=0.07130906590008225, with equal full support, actual CUDA
admission, preserved source/inputs, unchanged encoder and independent saved-query
arithmetic audit. Report all per-master outcomes and all stages. A fail stops this
finite diagnosis, with no decoder budget/rate/capacity rescue; it does not prove
encoding loss. Even a pass does not revise earlier fixed512 dispositions,
create a new embedding version, select a checkpoint, promote v7 or open TEST.

Report original encoder cost as retained, fresh decoder time separately, and
source/extraction/query/audit overhead separately where measured. No total-cost
or inference-speed comparison without its own benchmark. CUDA admission must
cover frozen modules and exact native outputs, zero encoder gradients,
decoder-only finite loss/gradients/updates, original Q/scaler, hidden-value
isolation, direct/split optimizer parity, immutable/reloaded snapshots, wrong
parent/data/policy/seed rejection and output-overwrite rejection. The independent
CPU reader replays saved query arithmetic, not the encoder or optimizer.
