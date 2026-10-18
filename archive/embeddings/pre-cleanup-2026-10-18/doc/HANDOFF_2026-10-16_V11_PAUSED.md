# Encoder research paused — 16 October 2026

The user asked to close with notes and continue later. The goal is paused, not
complete. No new scientific work should start until the user resumes it.

CPB-v3.market-01 remains the clean baseline and sole cuwacunu_dev encoder.
CPB-v11.market-01 is trained but **not quality evaluated or independently
established**. It is not a new winner. V10 remains the latest fully established
candidate: 3/22 conditions, no new scoped leader. Preserve all earlier leaders,
results, source copies, checkpoints, failures and reviews. CPB-v1 and the old RPB
lineage remain historical references; their trained weights are not clean targets.

## Completed V11 work

One architecture change from V9 source: each original temporal bin's eight
intermediate export rows now use four weighted-mean rows and four componentwise
centered-variance rows. Both use the same half-local/half-global attention.
The actual two-sum mean arithmetic, empty-bin zeros and full attention-dependent
variance derivatives are retained. This intermediate variance is distinct from
the native uncertainty4 and decoder conditional variance.

Native40, parameter shapes/counts, registration/draw order, seeds, data, heads,
global clip1 and 2048/B8 budget stay fixed. V9's unit Huber mean plus detached-error
Gaussian variance objective is used; V10's joint live-mean NLL is excluded.
All 294752 learned values, optimizer and TRAIN statistics start fresh. No old
weights, teacher, fitted statistics, old model execution or head tuning.

| Stage | Actual outcome |
|---|---|
| Prospective card/source freeze | Passed before build/testing; 22 unchanged conditions, D40 heads and tolerances declared |
| Source package | 132 files; 82 encoder/research owners |
| First build | Passed once, 85.6964 s; 7 executables, 11 translation units |
| Independent build closure | Passed; all source/ZIP/member hashes and V11-only dependencies verified |
| Objective CUDA fixture | Passed once |
| Mean/variance CUDA fixture | Passed once; full adjoints, F64/F32 finite differences, empty/constant/NaN cases and eight subgroup learning witnesses |
| Full engineering suite | Passed once, 408.8930 s; 360571 checks, 7372 CUDA encoder calls, 491 typed negative cases |
| Engineering causality | All 128 cutoffs, 2350 suffix and 30 internal cases per initialized/trained stage; future values AND masks excluded |
| Engineering resume | Next loss/model/optimizer moments bitwise exact |
| Main fresh training | Passed once: 2048 updates / 2048 attempts, batch8, zero skips, source_updates=0 |
| Main learning receipts | All 75 identifiable tensors, all eight mean/variance subgroups, four export/position groups and three between-bin directions pass producer checks |
| Main typed resume512 | Exact according to completed trajectory receipt |
| Main timing | 146.497975 synchronized GPU-step seconds; 205.102070 trajectory seconds; 214.856776 outer stage seconds |
| Initial/final saved-checkpoint proofs | **Not started** |
| Independent saved-training audit | Sources reviewed; actual audit card/body **not started** |
| Initial/final fixed-head quality | **Not started**; no V11 quality score or pass count exists |
| Quality audit / trust join / finalization | Sources reviewed; actual cards/bodies **not started** |

Producer learning/resume receipts still require the declared independent audit.
294751 describes identifiable tensor capacity, not every-scalar learning.

## Resume from saved artifacts; do not retrain

Repository: `C:\Work\cuwacunu\cuwacunu_embedding`.
Only the existing managed GPU container `cuwacunu` may execute project code.
Its ID is `82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e`;
image `sha256:ed0d8a5141f073d5c961b95e65e9b3528a66e122bda4971ec8f47d925d717de7`.
Keep its existing sole DEV bind mount/GPU/runtime configuration. Research lives
at `/opt/cuwacunu/research/causal-fresh-v11/attempt-001`, not in the DEV mount.
Build leaf: `/opt/cuwacunu/build/causal-fresh-v11-attempt001`.

Important frozen inputs:

| Artifact | SHA256 |
|---|---|
| [Frozen card](CAUSAL_FRESH_V11_CARD.json) | `aa0c8cf06e10b4da7ce1ede0d2cc914bc05b244043ad4f0d3039305499a10aaa` |
| [Card Markdown](CAUSAL_FRESH_V11_CARD.md) | `7cf07acfd3b1e5fc53d4b32d07e41814c6b05ff343ad0e2bf35e88fb080c72e1` |
| [Canonical descriptor](../code/encoders/causal_patch_bottleneck_mae/versions/CPB-v11.json) | `2cb25d674aa9cf9ebd136359c259e5498b0dd00786cf6606e07afb7e2f5b0db8` |
| [Protocol preflight](causal_fresh_v11/protocol-preflight.json) | `2545cea625d4ad2ad5dfe371621da970eb21cb56f844acef61ead0e4fedae695` |
| Source manifest, output/engineering/causal-fresh-v11/attempt-001 | `b9a701bbfa3e98e4719853dfb1b2fad8d7610b3a86b475e43398ab43fea15778` |
| Source ZIP, same directory | `7c76afbbce09eb84e17ce865d942ab340988d4127d5a06839ccb1634babf6a26` |
| [Completed training mirror](causal_fresh_v11/training-mirror-attempt-001.json) | `ff80472580000608229934b2f0c9a7301507b0abb0c2837488a43db9c3f44d20` |
| Training summary, output/runs/causal-fresh-v11/run-001 | `a0281af36e999c3e5e36d9247f65949f2a407cdb1b4e534e08d72863e46b2567` |
| [Current 22-condition protocol](clean_causal_encoder_search_goal_v9_retention.json) | `221f7cc748db8bc57bbb8dd066f569d957d16791d2a7cfe0ef0cc3785a579144` |
| [V10 final preservation, 104 predecessor checkpoints](causal_fresh_v10/final-preservation.json) | `fa6eb72bd45b9cb43d8dc7ea523ed70908eee072cc1c2e046e45995cd3a4c24d` |

All five main checkpoints are available both in the container's
`run-001/checkpoints` and locally at
`output/runs/causal-fresh-v11/run-001/checkpoints`. Local bytes were rehashed at
close and match the container inventory. They are Git ignored.

| File | Bytes | SHA256 |
|---|---:|---|
| inference-0.pt | 1304440 | `f8a4372718060c0de1f2c533536064290228cbd95158e6cb18e1b424c469a626` |
| inference-2048.pt | 1305617 | `9325231356595e165b7729ebca3303b60bf83bce44748760a71b27b2b732df85` |
| training-512.pt | 3788147 | `a129adbd5aa9f944ad4bbf1a22db3aa8ea7a89ba711ab6fea6b42dca9dc5089c` |
| training-1024.pt | 3788799 | `6a81ebba14644436978ca72b392666f193ab89b564eb08418abecd1d017ad7ea` |
| training-2048.pt | 3788799 | `5a8498e2608df545f0e014aee377134837941dd98c0b5f1ba282dd1ddbe9565b` |

The completed Training mirror covers 537 files / 1367870958 bytes, including
504 ignored PT artifacts. The Engineering mirror overlaps it: 506 files /
1343521422 bytes, including 490 ignored PT artifacts. **Do not add these sizes**.
Most of this volume is typed negative-case engineering evidence, not extra main
training runs. The five canonical training/inference checkpoints total about
14 MB. Nothing was deleted or cleaned during close-up.

## Next work when explicitly resumed

1. Read both AGENTS files, reporting standard, this handoff and frozen V11 card.
   Verify saved file hashes and the existing container identity. Keep V3 selected.
   Do not rerun draft generation, freeze, compilation, fixtures, engineering or
   main training; those completed once and their exclusive paths must remain.
2. Run the pending saved initial and final proofs using the frozen wrapper:

   ```powershell
   $v11Card = 'aa0c8cf06e10b4da7ce1ede0d2cc914bc05b244043ad4f0d3039305499a10aaa'
   $v11Preflight = '2545cea625d4ad2ad5dfe371621da970eb21cb56f844acef61ead0e4fedae695'
   & .\doc\causal_fresh_v11\run-container.ps1 -Phase InitialProof -Attempt 1 -CardSHA256 $v11Card -PreflightSHA256 $v11Preflight
   & .\doc\causal_fresh_v11\run-container.ps1 -Phase FinalProof -Attempt 1 -CardSHA256 $v11Card -PreflightSHA256 $v11Preflight
   ```

3. Prepare/review the actual TRAIN audit card with the completed mirror hash
   above. Owned workflow: `doc/causal_fresh_v11_training_audit`; auditor source
   `code/research/causal_fresh_v11_training_audit`. Its actual independent source
   review is `causal_fresh_v11/family-and-training-auditor-independent-static-review.json`,
   SHA `3a92cb4d4b6aebb22483dda9e8f7f90b5c7d622a9a9a47bdf0b6a90fb9b1b8b4`.
   Stage AST and run the independent audit only through its reviewed workflow.
   It reads saved data and can run alongside a saved GPU proof without an extra
   encoder/head call. Inspect exact helper parameters before invocation.
4. Audit 16384 subgroup rows and the actual update1/2048 56-field witnesses.
   Retained upstream gradients are PRECLIP: apply the recorded clip factor once.
   Tight three-parameter local adjoints use abs1e-9 + rel3e-4; require export W
   OR bias reconstructed maximum >1e-6 at each witness. Common score bias is
   finite descriptive evidence, not a tight near-zero parity/nonzero gate.
5. Only after required proofs/audit pass, run Quality once using the same wrapper,
   then Complete mirror and independent quality audit. Evaluate own initial0 and
   final2048 saved embeddings with unchanged fixed heads. Pair eight saved V9
   feature DATA archives only on native20:40, plus five whole fresh-statistic
   archives. Learned content0:20 is unpaired. No old encoder/head execution.
6. Join the separate final trust evidence; report all agreed MKT/TEMPO tables,
   repetitions, physical errors, reconstruction components and costs. Measured
   candidate, 22-condition advancement, overall winner and DEV selection are
   separate decisions. No new V11 quality result exists at pause.
7. Establishment8, finalization10 and preservation7 sources are already owned and
   independently reviewed, but no actual input/card/body is admitted for them.
   Finalization derives 43 input roles. Final preservation must include all 104
   old critical checkpoints plus the five actual V11 checkpoints (109), the
   132-file capsule, 18 binary/dependency members, whole evidence roots and all
   preparation/repair/failure history. Read the held reviews before new cards.

Saved-proof counters need care: `causal_updates` identifies the saved model's
learned state (0/2048); `new_training_updates=0` is correct because a saved proof
does not train. The typed-resume receipt is at the main run root,
`run-001/typed-resume-512.json`, not under witnesses.

## Preservation and scope notes

All draft generations and static findings remain preserved. The pre-execution
D32 layout guard, latest-V10 127-source rehash guard, and two inherited tolerance
summary corrections were completed before any scientific body. The original
2e-5 summary was preserved and aligned to the already declared 3e-5 fixture via
`causal_fresh_v11/draft-metadata-amendment-001.json`. Later preservation-source
repairs changed the stale first-binary selector to V11 and removed an
unconditional nonexistent failure-directory requirement; actual repair roots
still need protection. All have separate exact before snapshots and reviews.

Some ad-hoc host inspection predicates/field names were corrected without any
scientific replay. Their diagnostics are not failed encoder experiments. Keep
both the original and corrected draft review receipts.

Documentation changes are additive standalone handoff files. Close-up also
mirrors the already completed training evidence, without replay. Existing frozen
cards, source packages, research ledger, selected-version catalog and pinned
AGENTS/status front matter were not rewritten during pause. No DEV edits, promotion, Git
commit/push, deletion, cleanup or container lifecycle action was performed for
this close-up. Preserve concurrent DEV Latent Roadmap work. No research process
or scheduled continuation is intentionally left running.

Machine-readable close-up pins: `causal_fresh_v11/pause-handoff-2026-10-16.json`.
