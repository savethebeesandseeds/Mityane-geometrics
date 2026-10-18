# Clean causal research after CPB-v13.market-01

V13's exact fresh 2048-update, batch8 trajectory is closed. It passes causal,
CUDA-learning, typed-resume and independent saved TRAIN/quality checks, with
100% coverage and **8/22** frozen strongest conditions. It is not an overall
winner. CPB-v3.market-01 remains the clean baseline and the sole DEV encoder.
The research goal remains active.

This closeout reuses the saved measurements: **no new experiment**. The complete
per-task tables, physical errors, head repetitions, training costs and limitations
are in [results](CAUSAL_FRESH_V13_RESULTS.md), [detail](CAUSAL_FRESH_V13_DETAIL.md)
and [review](CAUSAL_FRESH_V13_REVIEW.md).

V13 keeps V11's model shapes and native40 layout. Its source change replaces the
attention-coupled centered-variance projection inputs with
`sqrt(v + 2^-10) - 1/32`, using the declared F32 add/sqrt/subtract order and live
derivatives. V11 supplied source only. Every learned weight, optimizer state and
TRAIN statistic starts fresh; there is no trained ancestor or oracle baseline.

Preserve its five scoped records: market activity and relation neural mean R2,
and TEMPO-5 synthetic-TRAIN-scaler intact linear, deleted linear and deleted
neural accuracy. All earlier complete leader records and displaced records stay
in the [ledger](clean_causal_encoder_research_ledger.json). Volatility, trend,
market-scaler transfer and reconstruction still miss retained leaders. Known
VALIDATION has guided architecture selection; no TEST or fresh-period result is
claimed.

The [applied metadata receipt](causal_fresh_v13_finalization/applied-metadata.json)
records the seven catalog/ledger/instance/front-note updates. The
[final preservation receipt](causal_fresh_v13/final-preservation.json) passed
140410 file/hash/history checks: **119 unique critical checkpoints = 114 earlier
plus 5 V13**, all ignored by Git. The 47 selected V3 source files in each
repository and its checkpoint hash remain exact. DEV's observed Git status is clean. No publication,
deletion, model/head replay or additional training occurred during preservation.

The first preservation attempt stopped because the intermediate AGENTS.md
generation had an existing whole snapshot but lacked a verifier alias. Preserve
the [failure diagnosis](causal_fresh_v13/preservation-failure-independent-diagnosis-001.json),
original verifier/card and both host logs. The separately owned, independently
reviewed repair registers only that exact path/size/hash snapshot before history
traversal. It changes no checkpoint, result, quality gate or scientific source.
The final receipt identifies the repair card and exact executed verifier.

The five main V13 checkpoints are under
`output/runs/causal-fresh-v13/run-001/checkpoints/`. Other saved `.pt` files include
heads, features, witnesses and proof fixtures; their count is not a count of
trained encoders. Preserve all historical market attempts, including RPB-v23,
RPB-v24 native40 and its separate native36 attempt.

Next, complete and separately admit the
[saved TRAIN-only V11/V13 diagnosis](causal_v11_v13_train_diagnosis/design-inventory-001.PROPOSED.json).
The proposal has passed independent design review, with actual preservation
context closure handled additively. Compute V13 descriptions only; carry the four
already audited V11 objects unchanged. Examine native blocks, each model's own
saved readout contributions, early56/late56 behavior, reconstruction, recorded
moment gradients and captured dispersion/root scales. No encoder/head execution,
refit, held-out diagnostic tensors, old raw reductions or trust replay. These
descriptions cannot establish branch importance or a validation cause. No V14
mechanism is selected; future training needs its own prospective declaration and
criteria protecting the updated scoped leaders.
