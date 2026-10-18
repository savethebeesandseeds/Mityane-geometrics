# Partitioned temporal relation screen, version 1

Protocol: `partitioned-relation-screen-v1`. New design **RPB-v16**; separate
outputs and checkpoints. Known development data only, with no TEST access or
automatic promotion. Preserve the measured v14/v15 screen and all earlier models.

Dataset: **TEMPO-3**, timing, designed complexity **4/5**. Variable period and
small signed delay, positive feature gains, offsets, natural missingness .10
and three-tick channel gaps. Reuse the exact controlled TRAIN/intact/deleted
archives from `structured-hard-timing-xrZMAS`; deleted VALIDATION adds the saved
.30 coordinate deletion. Each master has TRAIN256 examples/128 independent
pairs and VALIDATION128/64 pairs. No generator, period, delay, hidden clean
waveform or class labels enter the encoder.

## Architecture

RPB-v14's relation bank substantially improves timing accuracy, but its learned
residual mixes antisymmetric timing, symmetric shape and support coordinates.
Its trained scores are below its untrained scores; missing-data robustness is
also weaker across the five known cohorts. RPB-v16 tests structural separation.

Preserve the early-v10 backbone and its exact common initialization. Serve one
native32 vector: **20 learned shape coordinates plus12 learned odd relation
coordinates**. The shape block is a bias-free32→20 projection initialized with
the first20 identity rows. The relation block uses only the existing generic
bank's108 antisymmetric values, grouped into the three semantic channel pairs.
Each pair has a bias-free36→4 projection, initialized to average its nine
cross-feature pairs at each original-time spacing1–4. No raw, symmetric or
support coordinates enter these12 outputs; pairs cannot mix with other pairs.
Total parameters:225,805 backbone+640 shape+432 relation=**226,877**.
Literal initializers consume no backbone CPU/CUDA RNG draw.

Joint reversal of observations and visibility masks negates the relation block
for any trained projection weights. This property does not guarantee that
learning preserves the useful cue: weights may cancel or shrink odd relations.
The20-coordinate shape allocation may harm waveform reconstruction. Neither
the projection nor this allocation is fitted to class labels. All semantic
pairs, ordered feature pairs and spacings are retained; no selected task pair
or selected lag. The waveform decoder receives only the concatenated native32
and its original fixed IDs. Original hierarchical waveform Huber1 is the sole
training loss, without a new auxiliary or weight search.

## Fixed screen and gate

Train masters75272 and76373 once from0 through512 actual CUDA updates, keeping
both untrained0 and trained512 results. B8, training context deletion .15,
patch8, hidden64, dropout0, AdamW .001/weight decay .0001 and clip1 remain fixed.
Reuse the exact original row, mask and Torch counter streams. Record all loss
traces, masks/configuration identities and completed/skipped update counts.

Use the unchanged shared heads: Ridge penalty1 and tanh16/Adam .01/100 updates,
repetitions2701/2802/2903 with `stream_seed(repetition,width)`. Fit TRAIN once
per point/cohort/repetition and score both validation views with that same fit.
No PCA after the encoder, classifier tuning, baseline refits or best seed.

Advance this design to the remaining known masters77474/78575/79676 only if its
trained512 Linear AND Neural scores reach75% in intact AND deleted VALIDATION
on EACH starting cohort, with100% coverage. Otherwise stop this version after
the two-cohort screen. This rule bounds evaluation cost, not learned benefit
or promotion. Credit a strong untrained result to the architecture prior.
Additional known cohorts are screen-selected development evidence, not unseen
confirmation. A later design or training recipe gets a new tag and card.

## Admission and evidence

Build/test/run only inside the existing managed `cuwacunu_embedding` container.
Root schedules sequential GPU jobs. Admit actual CUDA parameters, inputs,
finite loss/gradients and weight updates; test semantic-channel invariance,
absent-value/hidden-target isolation, arbitrary-weight oddness, pair isolation,
native32-only decoding, save/reload and deterministic split-run AdamW parity.
Reuse unchanged compiled shared evaluation objects without recompiling them.

Capture this card, closed sources, exact input hashes and verified SDK proof.
Keep model/AdamW/frozen TRAIN scaler/RNG/progress checkpoints; export native
TRAIN/intact/deleted once at0 and512. At512 write the original fixed four-bank
masked TRAIN/VALIDATION queries. The pinned standard-library saved-tensor
reader checks fitted predictions, labels/source association, coverage and query
MAE without loading or rerunning the CUDA model or refitting heads.

Initial budget:2 trajectories×512 updates,12 native exports,12 fixed-head
pipelines/24 heads,4 query writers/16 necessary masked forwards. Tiny admission
fixtures are separate. Preserve failed attempts and all retained quality points.
