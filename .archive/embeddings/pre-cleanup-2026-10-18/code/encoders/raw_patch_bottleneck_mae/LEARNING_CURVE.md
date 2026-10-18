# RPB-MAE GPU learning curve

This development experiment tests whether the previous 128-update training
budget was too short. It changes the training budget while keeping the encoder
architecture, training-source count and optimizer recipe fixed. It does not
establish consumer acceptance under the evaluation policy.

## Frozen recipe

- Task: controlled `lag_sign`, C=3, H=32, F=3, raw float64 observations with
  10% coordinate missingness and source-paired opposite labels.
- Development masters: 901, 1002, 1103; 32 training source pairs (64 rows) and
  64 validation pairs (128 rows). These are new source cohorts.
- Architectures: unchanged independent RPB and one-layer aligned channel mixer,
  using `evaluation.conf` and `channel_mixer.conf` with an explicit CUDA override.
- One continuous AdamW training path per architecture/master, preserving optimizer
  moments, frozen training scaler, row/mask streams and absolute counters.
- Exact initialized checkpoint at zero; completed-update checkpoints at 128,
  512 and 2048. Batch 8 gives nominal 16, 64 and 256 training-row presentations
  per row respectively; sampling is with replacement.
- Learning rate .001, weight decay .0001, gradient clipping 1, dropout 0,
  patch length 8, width 64, compact channel width 32; no schedule or loss changes.
- Model initialization uses the existing hashed initialization stream. Common
  parameters start identically across the two architectures. Training row,
  artificial-mask and Torch streams use the declared master and absolute attempt.

Before the full run, a separate four-update check must verify that all parameters,
the transformed training input and scalar loss are on CUDA, gradients are finite,
and optimizer updates actually change weights for each architecture. Encoder
training and reconstruction inference use CUDA. Frozen feature extraction and
the small normalization/PCA/ridge/nonlinear readouts use CPU.

## Measurement and selection

The shared driver owns source generation, reconstruction target masks, readout
fits, scoring and budget selection. The RPB adapter receives only training
observations and metadata for fitting. It owns model training, serialization and
decoding through its exact exported compact vector.

At every checkpoint, enumerate each original patch as a reconstruction target
across eligible channels. Keep natural missing cells absent; score only observed
target cells. Every selected channel must retain at least two visible patches.
Report fixed-target training and validation hierarchical standardized Huber/MAE,
separately from the changing minibatch training-loss trace. Both losses use the
same training-fit scaler and the same frozen checkpoint as the feature probes.

Fit normalization, centered PCA and fixed ridge/tiny readouts on valid training
rows separately at every checkpoint. Report native global/concatenation surfaces,
global PCA12 and concatenation PCA36. The primary selector is the mean validation
PCA36 ridge accuracy over both architectures and all three masters, using the
candidate/comparator common valid population within each master. Select one
common nonzero budget; exact ties choose the smallest. A budget with an unsupported
primary entry is ineligible, rather than averaging a successful subset.

Persist `selection.json` before creating any testing observations. Final testing
uses 64 fresh source pairs per master under the named test seed
`stream_seed(master, 0x6375727665746573)`. Apply the selected exact checkpoints
and already-fitted readouts without refitting or additional training. Retain all
checkpoint, validation, trace and final-test artifacts.

## Interpretation

Falling training loss alone does not establish generalization. Falling validation
reconstruction error establishes improvement on the reconstruction objective;
validation/test lag readouts establish whether that information is accessible
through the exported vector and fixed probes. A plateau can still reflect limited
source diversity, the objective, pooling or the readout. This experiment isolates
training duration; it does not settle those other explanations.

Source-group bootstrap intervals are conditional on each fitted checkpoint and
readout. They do not quantify variation across retraining seeds. This remains
synthetic development evidence with no consumer evaluation or acceptance claim.

## Evidence

The experiment completed on 6 October 2026 in the existing managed Linux
container. The [run capsule](../../../output/runs/rpb-implementation/learning-curve-49fd5f8227/)
retains its [recipe frozen before scoring](../../../output/runs/rpb-implementation/learning-curve-49fd5f8227/frozen-recipe.md),
[exact invocations](../../../output/runs/rpb-implementation/learning-curve-49fd5f8227/invocations-before-scores.json),
[GPU gate](../../../output/runs/rpb-implementation/learning-curve-49fd5f8227/gpu-check/gpu-check.json),
[validation scores](../../../output/runs/rpb-implementation/learning-curve-49fd5f8227/results/validation-report.json),
[selection](../../../output/runs/rpb-implementation/learning-curve-49fd5f8227/results/selection.json)
and [fresh final-test report](../../../output/runs/rpb-implementation/learning-curve-49fd5f8227/results/report.json).
The frozen recipe remains unchanged; this document adds the measured outcome.

### GPU and execution

The GPU was an NVIDIA RTX A2000 8GB Laptop GPU, using staged LibTorch
2.6.0+cu124. Before the full run, each architecture completed four real optimizer
updates. All 167,341 independent-model and 217,325 mixer-model parameter values,
training inputs and scalar losses were on CUDA. Gradients were finite, weights
changed, and saved-checkpoint reconstruction was verified on CUDA. Live
`nvidia-smi` sampling during the main run recorded 42% GPU utilization.

Six continuous paths completed 2,048 updates each: 12,288 GPU updates, 98,304
sampled row presentations and 24 measured checkpoints including initialization.
The full run took about 5 minutes 48 seconds; summed synchronized model-training
time was 281.43 seconds. Feature extraction and small probe fits used CPU as
declared. No extra container, dependency or host development toolchain was added.

### Validation curves

These are descriptive means over the three masters. The primary surface is
channel concatenation reduced by training-fit PCA to 36 dimensions, followed by
the fixed ridge readout. Initialization is diagnostic and is ineligible for
budget selection.

| Completed updates | Independent ridge | Mixer ridge | Common selector mean |
| --- | ---: | ---: | ---: |
| 0 | 52.08% | 51.56% | Ineligible |
| 128 | 54.43% | 76.82% | 65.63% |
| 512 | 50.52% | 70.31% | 60.42% |
| 2,048 | 54.17% | 60.42% | 57.29% |

The selector chose **128 updates**. The mixer's primary score declined after
128 in each of the three masters, rather than only in their mean. The shared
driver persisted the selection before generating any final-test observations.

Fixed-target reconstruction continued to improve on both training and validation
sources. MAE below is in each model's frozen training-scaler units, aggregated
over target cells, channels and examples according to the card.

| Architecture | Updates | Training MAE | Validation MAE |
| --- | ---: | ---: | ---: |
| Independent | 128 | 0.04785 | 0.05603 |
| Independent | 512 | 0.03427 | 0.04555 |
| Independent | 2,048 | 0.02193 | 0.03761 |
| Mixer | 128 | 0.05093 | 0.05892 |
| Mixer | 512 | 0.03693 | 0.04697 |
| Mixer | 2,048 | 0.02286 | 0.03868 |

Initialization validation MAE was approximately 0.905 for both architectures.
Validation standardized Huber also fell: independent 0.00267 to 0.00133 and
mixer 0.00293 to 0.00135 between 128 and 2,048 updates. This is improvement on
held-out reconstruction, despite the weaker compact lag readout.

Longer training did help the independent model's **native 96-dimensional
concatenation with the nonlinear readout**: validation accuracy rose from
74.48% at 128 updates to 80.73% at 512 and 83.33% at 2,048. The mixer's equivalent
scores were 82.81%, 82.55% and 83.07%. The global PCA12 ridge surface remained
weak: mixer 59.38%, 61.72%, then 48.96%; independent approximately 48–50%.
Consequently, this experiment does not establish that the bottleneck contains no
lag information. Its accessibility depends on the exported surface, compression
and readout.

### Fresh testing at the selected budget

Each master supplied 64 fresh testing source pairs. The exact 128-update
checkpoints and their retained fitted readouts were reused, with no refit or
additional training. All tested surfaces and reconstruction targets had full
expected coverage in this fixture.

| Development master | Independent PCA36 ridge | Mixer PCA36 ridge | Mixer minus independent, percentage points | Conditional 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 901 | 53.91% | 62.50% | +8.59 | [−0.78, +17.19] |
| 1002 | 46.09% | 85.16% | +39.06 | [+28.91, +49.22] |
| 1103 | 55.47% | 71.09% | +15.63 | [+4.69, +26.56] |
| Descriptive mean | 51.82% | 72.92% | +21.09 | No across-master interval |

Two within-master source-group intervals favor the mixer; one crosses zero.
The raw lag oracle scored 100% and mask/metadata ridge scored 50% in every
fresh test cohort. The task is solvable, but those controls do not guarantee
that every learned export and probe will recover its label.

Native-width concatenation with the nonlinear readout averaged **85.16% for the
mixer versus 77.86% for the independent model**. Global PCA12 ridge averaged
57.03% versus 52.08%, reinforcing that pooled-global lag access remains weaker.

### Engineering checks and limits

The new [shared curve interface](../../shared/include/embedding/shared/learning_curve.h)
and driver own evaluation, while the
[RPB CUDA adapter](src/learning_curve_adapter.cpp) owns training and exact
checkpoint export/decoding. Existing feature/reconstruction adapters retain their
previous CPU behavior. Both encoder implementations and their configurations
were preserved; the longer run does not change their architecture or objective.

Managed-container checks passed shared generator regressions, development/test
split separation, CUDA continuation and optimizer resume, immutable checkpoint
snapshots, hidden-target isolation, fixed reconstruction coverage, training-only
readout fits, selection tie/unsupported-entry rules, and artifact preservation.
Snapshot tests match frozen parameter flags when comparing CUDA reconstructions,
following the checkpoint-parity lesson from the earlier mixer work. Sparse loss
traces retain queried milestone endpoints as later training proceeds.
An [independent artifact audit](../../../output/runs/rpb-implementation/learning-curve-49fd5f8227/validation.json)
passed 12,132 assertions over all 24 points, six paths, 96 surface/tier entries
and 480 disjoint source pairs.

This is a new development cohort and a separate validation-selected card; its
scores must not be substituted into older fixed-budget reports. Three masters
and within-master bootstrap intervals do not establish uncertainty across
retraining. Equal updates do not match parameter count or compute cost. Only
64 training rows per path were used, so larger or more diverse training cohorts
remain untested. There is no consumer acceptance claim.

The evidence does not support increasing training duration alone as the fix for
the current compact/global lag benchmark. Reconstruction generalizes and some
full-width nonlinear access improves with duration, while compact linear access
declines. A subsequent declared experiment should distinguish objective,
pooling, compression and readout limitations before choosing a larger budget.
