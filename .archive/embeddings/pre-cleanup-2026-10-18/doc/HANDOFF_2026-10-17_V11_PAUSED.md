<!-- Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at the repository root. -->
# Encoder research paused — 17 October 2026

The user asked to close with notes and continue later. Pause the research goal;
do not start further research until the user resumes it. This note supersedes
the pending-stage descriptions in the preserved 16 October handoff.

**CPB-v3.market-01 remains the clean baseline and sole cuwacunu_dev encoder.**
V11 has now been evaluated once and passes **8/22 frozen conditions, with 100%
coverage**. Independent TRAIN and quality audits pass, but the separate final
evidence join has not run. V11 is therefore not yet an established measured
candidate or a new overall best. V10 remains the latest fully established
candidate, with its original 3/22 result and no new scoped record. Preserve every
earlier scoped leader, checkpoint, result, source capsule, review and failure.

## What V11 changed

CPB-v11.market-01 is a fresh V9-source architectural trial. Each original temporal
bin's eight intermediate export rows change from eight weighted means to four
weighted means plus four componentwise centered variances, using the same
half-local/half-global attention. Actual two-sum mean arithmetic, full
attention-dependent variance derivatives, original bin positions and post-bias
empty-bin zeros are preserved. This intermediate variance is distinct from
native uncertainty4 and decoder conditional variance.

Native40 remains learned content20 + literal spectral12 + mean4 + uncertainty4.
Shapes, parameter count, registration/draw order, seeds, data/scalers/targets,
global clip1, single AdamW, 2048-update batch8 budget and fixed heads are unchanged.
The objective is V9's unit Huber1 mean plus detached-error Gaussian variance
calibration; V10's live joint mean-NLL route is excluded. All 294752 learned
values, optimizer state and TRAIN statistics start fresh, source_updates=0.
No pretrained/oracle weights, teacher or old fitted statistics were imported.

## Completed work; do not repeat it

| Stage | Actual outcome |
| --- | --- |
| Prospective freeze/build/fixtures | Completed once; 132 source files, 7 executables, 11 translation units; independent build closure passed |
| Engineering | Passed once: 360571 checks, 7372 CUDA encoder calls, 491 typed negative cases |
| Main fresh training | Completed once: 2048 successful updates / 2048 attempts, batch8, zero skips |
| Actual CUDA learning | All 75 identifiable tensors, eight moment subgroups, four export/position groups and three between-bin directions verified |
| Typed resume512 | Saved next loss, model and optimizer moments exact |
| Saved initial causal proof | Passed once: 179097 checks, 3685 CUDA calls, zero training updates |
| Saved final causal proof | Passed once: 179097 checks, 3685 CUDA calls, zero training updates |
| Independent saved TRAIN audit | Passed once: 516865 checks, zero mismatches |
| Initial/final quality | Completed once with unchanged fixed heads; 5 scientific + 2 aggregation stages, 1572 CUDA exports |
| Independent quality audit | Passed once: 13287 checks, zero mismatches |
| Initial DATA pairing | All 13 exact: 8 feature archives compare native20:40 only; 5 statistic archives compare whole values |
| Separate final trust join | **Pending; no join card or body has been generated/run** |
| Canonical reports / ledger finalization / final preservation | **Pending** |

Both saved proofs cover all 128 cutoffs, 2350 suffix cases and 30 internal cases:
changing future values **or masks** cannot change the embedding at the cutoff.
`causal_updates=0/2048` describes the saved initial/final state;
`new_training_updates=0` correctly records that proofs do not train. The common
pool-score bias is a registered softmax gauge, observed separately; 294751 is
identifiable tensor capacity, not proof that every scalar learned.

The TRAIN audit includes 16384 moment-subgroup rows and 8192 whole-bin identities.
Actual update1/2048 retained upstream gradients are pre-clip; apply the observed
clip factor once. Three local parameter adjoints pass their tight declared bound.
No full hidden/backbone/global-norm or other 2046 raw-gradient reconstruction is
claimed. The 190309 nested moment checks are included in 516865, not additional.

## Measured direction

The audited fixed-head numbers show five provisional scoped records, pending the
separate final trust join:

- Activity linear R²: **0.9094387755129657**; neural mean R²: **0.7324899092829259**.
- Relation neural mean R²: **0.4487745118258597**.
- TEMPO-5 synthetic-TRAIN-scaler intact linear accuracy: **0.98828125**.
- TEMPO-5 synthetic-TRAIN-scaler deleted-view neural accuracy: **0.9427083333333334**.

Volatility, trend, linear relation, six timing floors and reconstruction floors
still fail. V11 passes five retention floors plus three positivity conditions;
14/22 conditions fail. It sets no reconstruction record. Fixed standardized
query MAE is 0.25179751241660819 TRAIN and 0.32013411834078387 known VALIDATION;
reporting NLL is −0.36965938103736928 / 0.31234884646836464. TRAIN NLL is descriptive.
The exact measured V11 recipe must not receive seed/rate/budget/coefficient/head
or checkpoint rescue. Historical pass counts retain their own prospective floors.

The complete agreed four market tables, four TEMPO panels, dataset/complexity
legends, head repetitions, physical errors, reconstruction components and separate
cost scopes are saved in [RESULTS draft002](causal_fresh_v11/human-report-drafts-002/RESULTS.md),
[DETAIL draft002](causal_fresh_v11/human-report-drafts-002/DETAIL.md) and
[REVIEW draft002](causal_fresh_v11/human-report-drafts-002/REVIEW.md).
Generation001 remains preserved. These are drafts; canonical reports and leader
metadata must wait for the join. Known chronological VALIDATION and designed
TEMPO-5 evidence are not a fresh future-period forecast. Three head repetitions
are not three independent encoder trajectories. V3/V4 native36 versus native40
retains width, capacity and head-seed confounds.

## Saved evidence and checkpoints

Research repository: `C:\Work\cuwacunu\cuwacunu_embedding`.
Use only existing managed GPU container `cuwacunu`, immutable ID
`82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e`,
image `sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7`.
Research root: `/opt/cuwacunu/research/causal-fresh-v11/attempt-001`;
build: `/opt/cuwacunu/build/causal-fresh-v11-attempt001`.
The container was found stopped during today's resume and restarted unchanged
through the authoritative launcher. It is now running and idle: the close-up
process inventory showed only `/bin/bash`. Do not recreate or remove it.

Key authorities and exact SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| [Frozen card](CAUSAL_FRESH_V11_CARD.json) | `aa0c8cf06e10b4da7ce1ede0d2cc914bc05b244043ad4f0d3039305499a10aaa` |
| [Protocol preflight](causal_fresh_v11/protocol-preflight.json) | `2545cea625d4ad2ad5dfe371621da970eb21cb56f844acef61ead0e4fedae695` |
| [Complete mirror](causal_fresh_v11/complete-mirror-attempt-001.json) | `c18962bd15f876f4c0a2ec776bb2a00e687048673ba72f96d8b702c16c27fa26` |
| [TRAIN audit](causal_fresh_v11/independent-training-audit.json) | `96c11ddc1c9f5881319674c8262c8e81d780c0e2c0b96817b9ac9e9abc4687dd` |
| [Quality audit](causal_fresh_v11/quality-independent-audit.json) | `c37644fed46c8a0ec4916116fdb3eef5ab010e12383451e1213a0994cb74202b` |
| [Durable JSON copy receipt](causal_fresh_v11/durable-evidence-mirror.json) | `f4bfbfb4da0b2cbfae69eee0721ce0b375980a8cf31f2ab29ed32cf846ee8edd` |
| [Strongest V3–V9 22-condition protocol](clean_causal_encoder_search_goal_v9_retention.json) | `221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144` |

The main Complete mirror covers 819 files / 1402480119 bytes, including 721
ignored PT archives. These include typed negative fixtures, feature DATA and
fitted heads, not 721 main trained encoders. Earlier Engineering/Training mirrors
overlap this inventory; do not sum their sizes. The 22 durable JSON copies in
`doc/results/causal_fresh_v11/` are byte copies, not repeated measurements.

All five canonical V11 checkpoints remain at
`output/runs/causal-fresh-v11/run-001/checkpoints/` and in the container's matching
`run-001/checkpoints/`. They are Git ignored, about 14 MB combined.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| inference-0.pt | 1304440 | `f8a4372718060c0de1f2c533536064290228cbd95158e6cb18e1b424c469a626` |
| inference-2048.pt | 1305617 | `9325231356595e165b7729ebca3303b60bf83bce44748760a71b27b2b732df85` |
| training-512.pt | 3788147 | `a129adbd5aa9f944ad4bbf1a22db3aa8ea7a89ba711ab6fea6b42dca9dc5089c` |
| training-1024.pt | 3788799 | `6a81ebba14644436978ca72b392666f193ab89b564eb08418abecd1d017ad7ea` |
| training-2048.pt | 3788799 | `5a8498e2608df545f0e014aee377134837941dd98c0b5f1ba282dd1ddbe9565b` |

Selected V3 checkpoint in both repos stays 1290625 bytes, SHA256
`50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e`.
The machine-readable [pause record](causal_fresh_v11/pause-handoff-2026-10-17.json)
records close-up rehashes, including all 104 predecessor checkpoint pins and five
V11 checkpoints, V3 sources and the current evidence authorities. This close-up
verification does not substitute for the pending full final-preservation workflow.

## Exact resume point

1. Read both AGENTS files, reporting standard1.2, this handoff and the frozen V11
   card. Reverify local pins and existing container identity. No rebuild, fixture,
   engineering, training, saved proof, fixed head or audit rerun is needed.
2. Independently review the actual 14-role
   [establishment input proposal](causal_fresh_v11/establishment-input-plan.PROPOSED.json),
   SHA256 `60e6f63522479a1b1d6a85a4355766ebd720557af7bf1047ac95ed80cd181281`.
   It is explicitly **authorized=false**. No ADMITTED copy or join card exists.
   Check current pins and canonical/runtime aliases; `pairings` must alias the
   exact complete quality-audit file. Pairing scope is native20:40, not whole40.
3. Create a separate reviewed ADMITTED copy, then use the owned metadata helper
   to generate the join card. Review that actual card independently before the
   separate Stage/AST admission and once-only container JSON join. Mirror it and
   inspect measured-candidate/full-trust versus advancement/winner decisions.
   Preserve raw producer pending-trust fields unchanged.
4. Finish the three canonical human reports only after that join. Use the held
   reviewed finalization workflow: 43 inputs, proposal/review, then guarded seven
   research metadata writes with exact prior-byte snapshots. No DEV changes.
5. Finish the held full preservation workflow: 109 critical checkpoints,
   132-source capsule, 18 build/dependency members, all completed evidence roots,
   preparation/repair/failure history and today's notes/reviews/draft generations.
   Do not invent a nonexistent quality-audit-preflight-repairs directory.
6. If the five new scopes are established, preserve them and strengthen future
   prospective floors before another candidate. Then separately declare saved
   TRAIN-only diagnosis before selecting a new mechanism. No old encoder/head
   replay/refit, held-out diagnostic tensors, PCA or tuning. No next encoder has
   been selected or admitted.

Held source-review sets remain establishment8 / finalization10 / preservation7.
Inspect the current scripts' exact arguments before execution; source review is
not actual-card/body admission. The actual TRAIN and quality scope reviews are
saved under their respective `doc/causal_fresh_v11_*_audit/` directories.

Close-up is additive documentation and file/hash inspection only. The frozen
sources, producer outputs, live ledger/catalog/front matter and DEV were not
rewritten. Agents stopped; no scheduled continuation was created. No Git commit,
push, deletion, cleanup, promotion or container lifecycle change was performed
during close-up. Preserve concurrent DEV Latent Roadmap work. Research resumes
only when the user asks.
