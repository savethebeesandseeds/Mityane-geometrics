# Generic multiband representation screen v1

Prospective new CUDA architecture screen. Freeze all source, this card, the
passed information evidence, input archives and verified SDK before quality.
No TEST, stress suite, checkpoint selection, head tuning or automatic promotion.
Preserve the TEMPO-3 RPB-v18 milestone and all original saved encoders.

## Data and instances

**TEMPO-4**, task `slow_lag_sign`, designed complexity **5/5**: slow-component
lead/lag in a coherent two-rhythm mixture. **AMP-2**, task `component_balance`,
separately designed complexity **5/5**: relative slow/fast strength. Both use the
unchanged `two-component-v1` recipe, C3/H32/F3, low noise, positive affine gains,
natural missingness and channel gaps. Never average their accuracies together.

The exact signal and draw law is the frozen information card
`code/evaluation/cards/two_component_information_v1.md`, SHA256
`ee92e33107d97a247eea383736d44db570ebc9d6d6ddba83420f41ac467d0b5b`.
Its engineering admission passed all12 task/cohort/view gates before this screen:
`doc/results/two_component_information_v1.json`, SHA256
`04fda53a38d8ead378dec5b276c62eba322ec5ea0361daab789a02ee3c03569f`.
That diagnostic is not an encoder feature or training target.

Fresh quality masters are exactly **920903 and920904**, each for both tasks.
Every task/cohort has TRAIN256/128 source pairs and VALIDATION128/64 pairs.
Assign sources before variants and randomize opposite-label paired row order.
Generate four cohorts once; save12 legal CPU archives (TRAIN, intact VALIDATION,
VALIDATION with pair-shared extra30% deletion). Source IDs, labels and retained
values in the deleted view are unchanged. No clean waveform or latent parameter
is exported. Bound all12 roles, independent tasks/splits/sources, types and
file hashes before constructing a candidate.

**RPB-v18.alt-01** names newly fitted instances of unchanged v18 on these data.
**RPB-v19** names the new mixed spectral architecture below. Each model runs
once per task/master, retains its own initial0 and trained512 exports, and
fits its own TRAIN scaler and fixed heads. Old TEMPO-3 scores are motivation;
they cannot serve as matched new-dataset controls.

## One architecture change

V18 retains20 learned waveform-shape coordinates and12 fixed unit-normalized
generic time-odd coordinates. V19 keeps the identical shape20/backbone/decoder,
registered initialization, frozen432 aggregation and native32 allocation, but
replaces the relation calculus with a generic fixed harmonic bank.

For every semantic channel pair01/02/12 and all9 feature pairs, use the **same
currently visible original-time intersection J** for both signals. Sanitize
hidden values before arithmetic. Centre both signals and each sine/cosine
basis on J. Fixed frequencies are q=1..8 at2*pi*q/32; this is the same generic
grid for every pair and feature, not a true-frequency search or selected task
pair. Apply an analytic2x2 mask-only Gram correction to the harmonic projection.

Freeze eligibility: |J|>=4; each of the8 centred Gram matrices has trace>1e-12
and determinant>1e-8*trace^2. This rotation-invariant rank rule preserves time
reversal eligibility. Clamp inactive inverse denominators at1e-24. Both signals'
total eight-bin coefficient energy must exceed1e-12; clamp before square root.
Unsupported feature-pair contributions are exactlyzero.

Write complex coefficient Z=beta_cos-i*beta_sin. Normalize each signal by its
total energy across all8 coefficients. Form conj(Za)*Zb. Sum bins1..3 as low
and4..8 as high. Four coordinates per feature pair are
**[ReLow, ImLow, ReHigh, ImHigh]**. The imaginary sign is positive for the second
signal using t+positiveDelay. The unchanged frozen1/9 aggregation averages
all9 feature pairs into these four slots per semantic pair. Jointly normalize
each four-slot vector to unit length with norm floor1e-12 (squared clamp1e-24).
Separate per-band unit normalization is forbidden because it discards balance.

The12 coordinates are **mixed even/odd spectral relations**: real parts are
time-reversal even; imaginary parts are odd. Do not call them all time-odd.
Keep total226,877 registered parameters,226,445 trainable and432 frozen;
no new weights or random draws. The same20 learned shape numbers plus12 fixed
relations are the sole served32 and the sole decoder input. Fully CUDA Torch
operations; no CPU encoder arithmetic, interpolation, hidden frequencies,
phase/lag labels, analytic information-fit outputs or auxiliary targets.
Finite-window/mask leakage and noisy unit normalization remain explicit risks;
this bank does not promise perfect separation of arbitrary mixtures.

## Fixed training and evaluation

Use the identical C3/H32/F3/P8/W64/D32 configuration, three encoder layers,
four heads, FF256, decoder128, one early mixer, learned global mode2 and zero
dropout. Common initialization uses the existing mixed(master XOR
0x7270622d696e6974) policy for both candidate designs. Each task is a separate
training trajectory; task/labels/IDs cannot enter the model interface.

Original observed-waveform masked Huber1, context deletion .15, batch8,
512 completed AdamW updates, learning rate .001, weight decay .0001, gradient
clip1. Retain every attempt, skip, target count, gradient and loss component.
No label-dependent batching or auxiliary, early stop, replay or budget search.
Actual CUDA parameters, inputs, loss, gradients and changed weights must pass
small engineering admission before expensive quality training.

Export exact native32 only, with each model's TRAIN-fitted normalizer.
Ridge penalty1; neural16 tanh units/Adam .01/100 updates. All three repetitions
2701/2802/2903 use the unchanged stream_seed(rep,32) policy. For each point,
fit on that task/cohort's TRAIN and reuse the same heads for both validation
views. No PCA after an encoder. Retain fitted assets, predictions, source-group
intervals, populations, unsupported fits and initial controls; no best seed.

Retain original fixed masked TRAIN/intact VALIDATION waveform queries and
standardized MAE, sole32 reconstruction, synchronized training-loop cost and
separate extraction/head/query/I/O cost. The initial0 control determines
whether a result comes from the fixed prior or learned shape.

## Continuation rule and interpretation

This is a small two-cohort screen for each task. V19 may advance only if every
task/cohort/view achieves>=75% in **both** fixed heads with100% coverage, the
two-cohort mean deleted TEMPO-4 Linear improves at least5 percentage points
over matched v18.alt-01, and deleted AMP-2 mean Linear is no worse than that
control. Retain all results even when a condition fails; stop this recipe's
expansion rather than changing frequency cutoffs, coefficients, masks or heads.
No remaining quality seeds are allocated by this card.

Publish both tasks and views separately. A point0 success credits a fixed
architectural prior. Require a trained gain over the same instance's initial
control before claiming learned-shape utility; verify each model's fixed12
values are identical at0/512 on identical inputs. Even joint success only
establishes the declared two-rhythm family. Keep formal v4, original v7,
TEMPO-1 v10.alt-03 and saved TEMPO-3 v18 evidence and defaults intact.
