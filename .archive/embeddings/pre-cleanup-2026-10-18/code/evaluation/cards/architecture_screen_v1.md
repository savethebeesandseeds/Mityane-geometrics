# Temporal architecture screen, version 1

Protocol: `architecture-screen-v1`. This is a bounded architecture/objective
screen on known development data, not TEST, a fresh-data confirmation or a
production promotion. The user redirected work from the prospective data-size
comparison to architecture changes on 10 October 2026. Historical cards,
checkpoints and measured evidence remain unchanged.

Dataset: **TEMPO-3**, timing, designed ordinal complexity **4/5**;
`structured-hard-timing-v1`: varying period and small signed delay, positive
per-feature gains, offsets, natural missingness .10, three-tick channel gaps
and a separate saved extra .30 coordinate-deletion validation view. Use the
original controlled archives from `structured-hard-timing-xrZMAS`, without
regenerating observations, source order, splits or deletion masks. Each master
has TRAIN256 examples/128 independent pairs and VALIDATION128/64 pairs.
No TEST or stress access; labels enter only fixed head fitting and scoring.

## Two fixed changes

**RPB-v14** preserves the compact early-v10 backbone and adds a generic
cross-channel temporal relation bank before the served native32. For every
semantic channel pair, ordered feature pair and original-time spacing1–4,
intersect the currently visible three-point increment support. On that identical
support, form normalized antisymmetric and symmetric increment products and a
support fraction:108+108+108 values. Empty or zero-energy products are zero;
actual support remains explicit. A trainable bias-free324→32 projection adds to
the backbone's native32; only that combined native32 enters the original decoder.
The new projection has a deterministic nonzero initialization, with no backbone
RNG draw. Total236,173 parameters, including10,368 new values. No analytic
classifier, selected lag, class sign, generator period/delay or hidden waveform
enters the model. The architecture contains a deliberate generic relation prior;
its separate untrained result is required.

**RPB-v15** preserves the original early-v10 native32 input path and adds a
training-only bias-free32→18 decoder. Its targets are even and odd lag profiles
for all three semantic channel pairs and spacings1,2,4, using centered normalized
cross-feature correlations on identical legal plus/minus endpoint support.
Require at least four supported times and nonconstant energies; unsupported
target dimensions do not contribute. Original legally observed TRAIN values,
including subsequently masked training targets, supply detached targets. Fit
the target mean/scale once on TRAIN only. Loss is original hierarchical waveform
Huber1 plus equally example/dimension-averaged standardized dynamics Huber1,
fixed1:1 without a weight search. The576-value linear decoder has deterministic
nonzero initialization without advancing backbone RNG. Total226,381 parameters.
Serving and waveform reconstruction still use only the original native32.

Both designs use semantic IDs rather than physical channel order. Sanitize
unseen values before subtraction/statistics, use current actual context for
input relations, and keep naturally unobserved values out of auxiliary targets.
Fixed finite-energy floors scope affine-invariance claims to floor-inactive
cases. Neither design changes scoring eligibility or classifier capacity.

## Budget and advancement rule

First measure both designs on masters75272 and76373, once each from0 to512
completed CUDA updates. Retain point0 and512. B8, context deletion .15, native32,
patch8, hidden width64, dropout0, AdamW .001/weight decay .0001, clip1 and original
masked reconstruction targets remain fixed. Log both objective components where
applicable. Match the original row/mask counter streams and backbone initialization
when the producer admits them; record exact resolved streams in artifacts.

Fit the unchanged shared classifiers separately for each design/point/cohort:
Ridge penalty1 and tanh16/Adam .01/100 updates, repetitions2701/2802/2903 with
the existing width-paired seed policy. Fit TRAIN once and score both validation
views. No PCA after an encoder, head tuning, best repetition or baseline refits.
Existing baseline/early-v10/v13 numbers may be reused with their own cohort
population identified; never mix a two-cohort candidate mean with a five-cohort
reference mean without saying so.

An individual trained design advances to the remaining three retained masters
77474/78575/79676 only if BOTH its mean fixed Linear and Neural scores are at
least75% in BOTH intact and deleted validation for EACH screen cohort, with
100% coverage. Retain all screen results, including failures and untrained
controls. This gate bounds further comparison cost; it is not a formal encoder
promotion or proof of learned improvement. If the untrained version explains
the benefit, credit the architecture prior explicitly. Any subsequent recipe
change receives a new tag/card and separate outputs; do not rewrite this screen.
Additional known cohorts are broader development checks, not independent
fresh-data confirmation of the screen-selected design.

## Execution and evidence

Root coordinates sequential GPU jobs in the existing managed
`cuwacunu_embedding` container. Compile/test there only. Before training, verify
actual CUDA parameters, inputs, loss, finite nonzero gradients and changed
weights, native32-only decoder input, semantic ordering, absent-value isolation,
support/energy handling and save/reload behavior. No CPU encoder fallback.

Capture the card and a compact closed source inventory, existing SDK identity,
input file hashes, resolved settings/seeds, checkpoints, AdamW state, frozen
scaler, CPU/CUDA RNG state, update/loss traces and synchronized training costs.
Extract native TRAIN/intact/deleted features once per retained point. Fit saved
features on CPU with the shared fixed-head helper. Write the original fixed
TRAIN/VALIDATION masked-query predictions at512 and retain their reductions.
A compact independent saved-tensor check verifies predictions, labels/source
association, coverage and query MAE. It must not rerun encoders or classifiers.

Initial maximum:4 trajectories×512,24 native exports,24 fixed-head pipelines
(48 heads),8 original-query writers/32 necessary masked forwards. Parameter
admission and tiny engineering fixtures are separate from quality runs. Record
actual timings and counts; preserve failed attempts. Do not reproduce the prior
full forensic archive matrix for this small exploratory screen.
