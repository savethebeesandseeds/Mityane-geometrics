# CPB-v9.market-01 — prospective declaration

Root reviewed and froze this separately named fixed half-local/half-global hypothesis at 2026-10-16T10:57:30.3250502Z, after the completed saved TRAIN-only V6/V8 diagnosis. The diagnosis selected no model; the research owner selected this separate experiment afterward. Machine card: [CAUSAL_FRESH_V9_CARD.json](CAUSAL_FRESH_V9_CARD.json). Exact source closure: 118 files including the generated preflight, with50 owned family files plus the full public header,31 research files,7 binaries and11 translation units. These are prospective source counts; the actual compiler/binary closure remains a required later proof. No AST/build/test/model/head body has run. Every future phase requires its explicit invocation, hash-bound passed preflight and exclusive managed-container workflow. The canonical immutable descriptor pins the frozen card without a cycle. Complete previous draft bytes are preserved in the freeze snapshot.

CPB-v3.market-01 remains the clean baseline and the sole active development
encoder. Preserve all six clean references V3–V8, every earlier checkpoint and
historical result, including RPB-v23.market-01 and RPB-v24.market-01. Preserve
the entire V8 workflow, failed host Stage, independently reviewed repaired
orchestration, and completed V6/V8 diagnosis/audit/failures. No old encoder or
head runs, refits or trained-state imports follow from this proposal.

## Selected forward change

V9 starts from complete physical V8 source copies. Both temporal routes retain
the same strictly causal Transformer and backward-only relation bank. For the
16 original patch positions, four anchored bins contain four positions each.
Each nonempty bin mixes half its local masked-softmax hidden vector and half
the shared masked-softmax hidden vector over the current valid history.
Apply that bin's eight disjoint rows of the single export weight tensor and
its bias once, after mixing. Concatenate the four eight-coordinate exports;
retain the existing downstream global mixing and content projection.

“Same export weights” means one **freshly initialized** tensor shared between
the two pooling routes, with unchanged shape, registration and random draw
order. It never means loading trained V8, V6 or any other historical values.
All 294752 learned values in 76 registered learned tensors, the optimizer and
all TRAIN-fitted statistics start fresh under the unchanged seeds. The analytic
3888-value prior remains fixed; total parameter capacity is 298640.

An empty anchored bin must zero its whole eight-coordinate export **after**
the one bias, even when other bins provide a nonempty global vector. Entirely
padded histories produce finite zero. Clear unsupported nonfinite hidden
values before arithmetic and retain original positions after compaction.

All four exported slices belong to the endpoint embedding at the current
requested cutoff `t`; they are not four embeddings stamped at bin endpoints.
The shared whole-history vector may use only valid states at or before `t`.
Later-in-window past may enter an older anchored slice within that endpoint
embedding. This does not relax the cutoff mask, Transformer causality or
backward relation-bank offsets. Changing any future values **or masks** must
leave every coordinate unchanged at every cutoff, both initialized and final.

Actual traces must expose local and global weights `[R,T]`, mixture weights
`[R,4,T]` and bin validity `[R,4]` in both routes. Each nonempty bin's mixture
has mass one; empty bins have mass zero. The old aggregate `pool_weights` is
the sum of those mixtures, with total mass equal to the number of nonempty
bins. It is not a probability vector of mass one.

## Typed and optimization contract

Bind these four fields at inference root/payload and TRAIN root/encoder/
encoder.payload:

| Field | Exact value/type |
|---|---|
| `temporal_pooling_policy` | UTF8 `original-position-four-bin-shared-global-half-mixture-v1` |
| `temporal_pooling_partition_v1` | Long `[4,16,32,4,8]` |
| `temporal_pooling_mix_f64` | Double `[0.5,0.5]`, ordered anchored local, whole history |
| `empty_local_bin_policy` | UTF8 `zero-whole-export-slice-after-bias-v1` |

Reject missing or altered policy/mix/empty/partition fields, incorrect types,
lengths and ordering at every typed location. Reject old artifact kinds.
Require actual typed resume at update512 with exact parameters, buffers,
moments, optimizer order, counters and data/recipe identities. Continue the
same actual trajectory; do not repeat updates. Engineering separately proves
the controlled next-step parity case.

Keep native40: content20 + literal spectral12 + mean4 + uncertainty4. Retain
V8's mean Huber(delta1) plus detached-error Gaussian variance objective, both
coefficients one, direct gradient separation, single AdamW, one shared global
norm2 clip1, learning rate0.001, weight decay0.0001, dropout0. Budget2048
successful updates, batch8, maximum4096 attempts. Seeds943001/2/3/4 remain
initialization/row/mask/context respectively. Fixed queries use the saved
1/16 query plans; no new query seed draws. Masks, context deletion, targets,
chronological populations and F64-before-subtraction support reductions remain
unchanged. Reporting full Gaussian NLL is detached and never the mean objective.

Fresh input and 270-coordinate bank normalizers use only the same4096 TRAIN
anchors/recent128 history. Fit each once with the unchanged rawF64 → sanitized
F32 → CPUF64 recipe, minimum support2 and scale floor1e-6. No old normalizer,
learned state or validation/test row enters calibration. Calibration runs no
encoder. The fresh constructor JSON intentionally lacks `gradient_policy`;
require absence only there while requiring all other objective and temporal
fields. Engineering, actual saved proofs, typed checkpoints and the training
summary bind the gradient policy strictly.

## Prospective learning evidence

Require observed nonzero CUDA data-gradient and movement events in75
identifiable learned tensors. Their294751 values describe the summed tensor
capacity, not proof that every scalar receives a nonzero gradient or moves.
The scalar `core.encoder.level.base.backbone.pool_score.bias` is the sole
softmax-gauge exemption; it remains registered in the same optimizer and is
observed separately. Weight decay or movement alone is not data learning.

Retain all2048 component/global-norm/clip-factor rows; all8192 four-group rows;
all2048 gauge rows; and the actual update1/2048 witnesses with the same twelve
before/after/clipped-gradient arrays. Each export weight/bias eight-row group
and each within-bin centered position group needs data-gradient and movement
events and final changes. Evidence uses existing backward/optimizer steps,
with no extra encoder forwards. Separate evidence overhead from compute cost.

Add `temporal-between-bin-position-learning.csv`, exactly2048 rows and nine
ordered columns: update, between-bin gradient L2, between-bin movement L2,
three signed gradient coefficients, three signed movement coefficients.
Use the unit full16 Hadamard basis: bin signs `++--`, `+-+-`, `+--+`, each
repeated four positions and divided by4. Promote operands to F64 before
projection or before/after subtraction; verify Parseval against the full
between-bin projection norm. The self-describing learning receipt binds this
third CSV and records all three event counts and signed final contrast changes.

Every direction must have at least one nonzero **data-gradient** event and
movement event. The final between-bin projection's overall change L2 must be
positive. Individual signed final coefficients are descriptive and need not
each be nonzero. Movement includes AdamW decay; decay never counts as a data
gradient. Do not infer a full encoder Jacobian, per-scalar learning or a
counterfactual old trajectory from these observed arrays.

The separate temporal CUDA fixture must prove independent manual mixture and
adjoint parity, both real temporal paths, original-position membership, empty
slice zero, truthful mixture traces, common softmax gauge invariance, relative
bin global sensitivity, all four export/local contrast groups and all three
between-bin contrasts. Frozen source currently declares one backbone encode,
five direct-kernel calls, one synthetic AdamW step, zero public Encoder calls
and zero head fits. Actual build/source/binary/TU counts have been derived and
reviewed from the final owners before admission; the later actual dependency audit must confirm
them. The unchanged composite-objective fixture has its own proof.

## Fixed evaluation and protected gains

Use exactly the reviewed future22 criteria from
`clean_causal_encoder_search_goal_v8_retention.json`, SHA256
`e7920442fbc5528cbe7e67f71ada65101974a6a28ad4d39bc57dea461c477699`.
The machine card copies all22 IDs, operators, floors and source labels
literally. Protect all V3–V8 scoped records. No TRAIN-NLL gate, older oracle
floors or retrospective recounting of historical cards is added. Strict
positivity still means all **three** neural trend repetitions and both relation
usefulness conditions. Do not tune or relax any failing criterion.

Export only saved initial0 and final2048,786 CUDA export calls each,1572 total.
Fit both checkpoints' own fixed heads once; no initial score reuse. Compare
only initial feature coordinates20:40 across eight saved DATA archives versus
V8; compare all declared valid masks/IDs/labels and five whole fresh-statistic
archives. Learned content0:20 may differ. All13 partial DATA/statistic pairings
must be exact before full trust. They do not load historical encoder weights,
assert whole40 equality or justify reused initial heads.

D40 head settings stay unchanged: ridge1; neural hidden16/100steps/lr0.01;
three repetitions2701/2802/2903 and exact dimension-dependent seeds
6049515622977958931,8262941649267994858,2095371932330426401. Market budgets remain
164 linear/724 neural. TEMPO budgets remain82 linear/690 neural. Use all three
cohorts941101/2/3, both fresh synthetic-TRAIN and fresh market-scaler policies,
512 TRAIN/256 intact validation/256 deleted validation, unchanged populations,
generator recipes, labels and support. No post-encoder PCA, head tuning,
intermediate checkpoint selection, old encoder training or old head refits.

Report four separate market task tables and four TEMPO panels, with standard
method/dimension/linear/neural/coverage columns, full neural repetitions and
all recipe/dataset/task/complexity legends. Include initial/final112/32 fixed
query MAE and diagnostic NLL, physical channel/unit errors, component losses,
coverage and cost. Complexity4/5 and5/5 remain ordinal recipe labels.
Six saved final controls V3–V8 and exact V6/V7/V8 final-trust joins may be reused;
their original pending producer bytes and historical pass counts stay exact.
Raw8 encoder-free controls may be reused from the byte-pinned source.

Runtime producers keep pairing/independent-audit trust pending. A separate
evidence join may establish a measured candidate only after full causality,
freshness, typed resume, actual learning, finite/full coverage, independent
training/quality audits and all13 partial pairings pass. Advancement requires
all22 conditions additionally. Winner selection and application promotion are
separate and remain false. A failed outcome gets no seed/rate/budget/blend/
checkpoint/head rescue.

## Preservation, execution boundary and limits

The pre-execution inventory
`causal_fresh_v9/preservation-before.json` verifies all94 existing critical
checkpoint files, both47-file V3 source copies, all113 V8 capsule sources,
all436 saved V8 main artifacts and additional immutable diagnosis/source/
result/failure evidence. The frozen V8 receipt continues to bind broader
historical inventories. Checkpoints stay local and ignored. Five future V9
main checkpoint files would bring the critical total to99 only **after**
actual availability/hash verification; no future artifact count is prefilled.

Execution may use only the same managed GPU cuwacunu container and exclusive
fresh source/build/output leaves after independent prospective admission.
Host scripts perform only byte/JSON orchestration, using native `docker.exe`.
The active wrapper is derived from V8's repaired native-Docker helper, not the
historical shadowed helper. There is no new V9 recovery event. Preserve every
failed attempt and all original bytes. No container recreation/deletion,
concurrent Latent Roadmap changes, DEV edits, Git mutation or publication is
authorized here.

Existing downstream global mixing already permits global temporal information.
The proposed blend changes routing, activation scales and computation. A good
or poor fixed-head result on these known development populations cannot prove
causation, forecasting quality, fresh-period generalization or an overall
winner. Those limitations remain even if all22 conditions eventually pass.
