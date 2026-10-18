<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->

# Causal training provenance correction — 15 October 2026

CPB-v1.market-01 remains the preserved pretrained historical reference and the
currently selected dev artifact. It provides context only. Its scores are not
pass/fail targets, retention floors or an oracle for a properly trained causal
encoder. CPB-v2.market-01 is likewise an inherited-weight historical experiment.

No clean, from-scratch causal training baseline has yet been established.
The source and saved initialization records confirm this ancestry:

- CPB-v1 imported every V30 parameter with zero new causal updates.
- V30 includes the trained v22 parent and a subsequently trained change branch.
  Those learned tensors came from the earlier noncausal encoder computation.
- CPB-v2 imported every v1 parameter, then added 2048 causal updates. A fresh
  optimizer and a seeded constructor immediately overwritten by checkpoint
  loading do not make that a fresh initialization.
- The future-value/mask proofs still establish the measured runtime input
  invariance for fixed weights. They do not establish causal training ancestry,
  erase pretraining, or measure a held-out leak quantity.

The previous proposal to freeze v1's parent and adapt only its change branch is
withdrawn: it would retain the same learned ancestry. Preserve its old proposal,
reports, cards, checkpoints and scoped measurements as historical records.

The next proper baseline is separately declared CPB-v3.market-01: native36 with
all learned blocks initialized directly from a declared seed, a fresh optimizer,
source_updates=0, and no imported model/teacher/parent/decoder weights. Rebuild
the literal analytic prior directly. Fit input and bank statistics freshly from
permitted TRAIN raw windows only, without old encoder calls; freeze all fitted
state before validation. Causal preprocessing, relation offsets and attention
must apply from the first training forward.

Save and evaluate both fresh initial and final artifacts with the unchanged
fixed heads. Report each market task and TEMPO-5 panel separately, together with
raw-data controls, coverage and reconstruction. Record technical trust checks
separately from useful task performance. Historical pretrained scores impose no
advancement gates. Do not select a checkpoint, head seed, recipe or stopping
point after inspecting scores.

This correction changes the interpretation and next research direction, not
any saved measurement. Dev's runtime selection remains v1 pending a separately
verified replacement. No encoder arithmetic, weights, data, frozen card/report
or existing source capsule is rewritten by this correction. No commit/push is
implied by it.

Full experiment details and the frozen prospective v3 protocol live in the
research repository's doc/CAUSAL_FRESH_V3_CARD.md and matching JSON. New research
executes only inside the existing GPU-enabled cuwacunu container.
