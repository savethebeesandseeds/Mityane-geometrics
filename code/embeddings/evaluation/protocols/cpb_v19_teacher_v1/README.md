# V19: learned-state EMA teacher with unchanged evaluation

This completed comparison tested a **training-only teacher/predictor method**
on the configurable CPB component. Read the [protocol](PROTOCOL.md) and
[Operation Contract/Requirements-and-Evidence map](CONTRACT.md). They were saved
before V19 numerical execution. The [results](RESULTS.md),
[compact record](../../records/CPB-v19.teacher-001.json) and
[independent review](REVIEW.md) retain the completed evidence.

**The continuation criterion failed.** Median paired ΔR² was volatility
+0.053184 (3/3 improving), trend −0.029321 (1/3), activity −0.053271 (1/3),
and relation +0.109583 (2/3). All four trend/activity conditions failed while
volatility protection passed. No scoped linear record or top-three group
changed; V16 seed946103/final2048 remains the reference. All 3,072 new candidate
updates, full-state resumes and three new readout/audit pairs passed. Another
3,072 control updates and their existing heads/audits were reused. The method
and its scoped gains remain available research tools, without advancement or
application claims. The account below preserves the declared design.

The student keeps its original architecture, scattered mask, Gaussian NLL and
coefficient0.1 temporal objective. Add0.1 normalized-dot agreement between a
training-only predictor of masked learned-D coordinates and a detached fully
observed teacher at the same cutoff. D derives from the configured learned
shape width (20 in native36); fixed spectral/change coordinates are excluded.
The predictor is Linear(D,2D)→GELU→Linear(2D,D), with biases and no BN/dropout.
Normalize in F64 with norm floor1e-6. After each successful joint student/
predictor AdamW step, update the teacher by EMA0.99. The teacher receives no
gradients and never enters the optimizer; fixed state remains exact.

The auxiliary mean covers only original NLL-eligible, student-valid and
teacher-valid rows. Empty support gives graph-zero and does not cancel a
supported base update. Persist actual learned-state dispersion/norm/support
diagnostics; there is no variance/covariance regularizer, collapse rescue or
diagnostic-dependent budget. Inference remains the student alone, native36 at
the study configuration. Typed TRAIN state must carry the complete teacher,
predictor, optimizer, fitted state, seeds and progress needed for exact resume.

Reuse all three exact already completed scattered-control final1024 states and
their audited heads from [V18](../cpb_v18_span_v1/RESULTS.md). Train only three
fresh V19 candidates, seeds947101/2/3,0→512→1024 with actual typed resume and
unchanged TRAIN fitting/sampling. Completed new work is **3,072 updates, three
TRAIN fits, six training processes,12 target-head fits and three scalar audits**.
Inherited controls contribute another3,072 updates, three fits,12 heads and
three audits. Keep new/reused work distinct within6,144 logical comparison
updates, with engineering work counted separately.

All three new final candidates plus inherited control/head identities seal
before any new candidate head. The reused control heads already exist and
predate that seal. Evaluation remains the same112 TRAIN/32 known January
validation anchors, legacy targets, TRAIN-standardized ridge1, original scalar
audit and five trend/activity/volatility predicates. No intermediate gate,
rescue seed, hyperparameter sweep or outcome-dependent extension is used.

V17's weakened temporal term and V18's contiguous spans retain their completed
negative outcomes. This method is a different learning target, not evidence
that the earlier diagnoses establish its usefulness. January/February/March
are consumed development evidence. No fresh, neural, TEMPO, reconstruction,
full2,048-budget or application result is measured. **V16
seed946103/final2048 remains the selected research reference.**

The reusable family, configuration, full training recipe and each trained
instance remain separate. Carry construction, TRAIN fitting, teacher/predictor/
EMA rules, checkpoint/resume/inference code, tests, dependencies/licenses,
rationale and eventual results together. Numerical work remains C++ in the
managed Linux container with Node/Bash orchestration; no Python or host
toolchain is introduced. Applications retain substantial training and docking.
