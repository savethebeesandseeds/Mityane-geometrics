# Fixed-prior temporal relation screen, version 1

Protocol: `fixed-prior-relation-screen-v1`; new **RPB-v17**. Separate source,
model identity and outputs; retain all v14/v15/v16 measurements unchanged.
Known development data only, no TEST, stress or automatic promotion.

Dataset: **TEMPO-3**, timing, designed complexity **4/5**. Variable period and
small signed delay, positive feature gains, offsets, natural missingness .10,
three-tick channel gaps and the saved extra .30 coordinate-deletion validation
view. Reuse exact controlled archives from `structured-hard-timing-xrZMAS`.
Each master has TRAIN256 examples/128 independent pairs and VALIDATION128/64
pairs. Labels only fit/score fixed classifiers; the encoder receives no class,
generator period/delay, selected task pair/lag or hidden clean waveform.

## One focused recipe change

RPB-v16's two-cohort screen reaches100%/100% intact and83.20%/80.34% deleted
after512, but one deleted neural score is69.53%, failing its75% continuation
gate. Its untrained native32 scores100%/100% intact and89.06%/99.74% deleted.
Training degrades a useful timing prior. Stop v16 expansion under its own card.

RPB-v17 preserves exactly v16's native32 architecture:20 learned shape
coordinates and12 grouped odd relation coordinates. Freeze ONLY the432-value
bias-free grouped relation projection at its generic literal initializer:
all three semantic channel pairs, all nine ordered feature pairs and all
original-time spacings1–4, averaging feature pairs at each spacing. The20
shape coordinates, early backbone and original waveform decoder remain
trainable. Total registered226,877 parameters; trainable226,445, fixed432.
No RNG draw, new loss, supervision or post-encoder transformation is added.
The decoder receives only that same native32 and original fixed IDs.

This is a hybrid encoder: learned waveform shape plus fixed generic temporal
relations. Any improved timing score must be credited to that architectural
prior, not to the waveform objective learning the timing relation. Freezing
the projection guarantees its odd coordinates stay unchanged for the same
legal input/scaler; the learned shape block and fixed-head weights can still
change performance. Masking/deletion can change the available relation signal.
No selected feature, coefficient or classifier search. Exact initial native32
parity with v16 and fixed-weight/coordinate parity after updates/load are required.

## Budget and continuation gate

Train first on masters75272/76373, once from0 through512 completed CUDA updates.
Retain both points. B8, context deletion .15, patch8, hidden64, dropout0,
AdamW .001/weight decay .0001, clip1 and original hierarchical waveform Huber1
remain unchanged. Use exact original row/mask/Torch counter streams and common
initialization. Frozen weights must receive no gradient or optimizer update.

Use the shared Ridge penalty1 and tanh16/Adam .01/100-update heads, with paired
repetitions2701/2802/2903 under `stream_seed(repetition,width)`. Fit TRAIN once
per point/cohort/repetition; score both validation views with those same fits.
No PCA after the encoder, new baseline fits, head tuning or best seed selection.

Advance to known masters77474/78575/79676 only if trained512 Linear AND Neural
scores are at least75% in intact AND deleted VALIDATION for EACH initial cohort,
with100% coverage. Otherwise stop after two. Retain failures and untrained
controls. This cost gate is not promotion or learned-benefit evidence. The
remaining known cohorts are development checks after screening, not unseen
confirmation. A later recipe gets a new tag/card and separate artifacts.

## CUDA admission and saved checks

Build/test/run in the existing managed GPU container using a separate named
task session. Reuse the nine unchanged generic evaluation objects. Root runs
quality jobs sequentially. Admit actual GPU parameters/input/loss and finite
nonzero gradients/changed trainable weights, exact fixed432 weights and odd12
coordinates after AdamW, zero frozen gradients, semantic/mask invariance,
sole-native32 decoding and immutable save/reload with frozen flags preserved.

Capture this card, closed sources, exact input hashes and verified SDK proof;
keep model/AdamW/frozen TRAIN scaler/RNG/progress checkpoints. Export native
TRAIN/intact/deleted once at0/512 and original fixed four-bank TRAIN/VALIDATION
masked queries at512. The pinned standard-library reader verifies saved head
predictions, source/label association, coverage, query MAE and odd-coordinate
equality between0/512. It loads no CUDA model or optimizer and fits no heads.
Initial budget:2 trajectories×512,12 native exports,12 head pipelines/24 heads,
4 query writers/16 masked forwards. Admission fixtures are separate.
