# RPB MAE compression and global bottleneck experiments

This is the completed historical experiment record. Its compressed tiers and
selection rules remain part of that evidence. **RPB-v4 — Learned global
bottleneck** is the active experimental version; future primary comparisons use
its native32 output without PCA afterward. See the
[version registry](../../../doc/EMBEDDING_VERSIONS.md),
[reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md), and
[next advance plan](NEXT_ADVANCE.md). Historical `current_mixer`, `mean_global`,
and `learned_global` names map to RPB-v2, RPB-v3-mean, and RPB-v4 respectively.

These development experiments target weak compact and global lag readouts after
the [GPU learning curve](LEARNING_CURVE.md). Native channel vectors retain useful
nonlinear lag access, while longer reconstruction training does not restore the
compact linear result. The experiments separate compression, the reconstruction
signal path, pooling and source diversity. Evaluation remains in shared code;
encoder adapters receive training observations and schema metadata without labels.

## Frozen compression diagnostic

Use only saved training and validation features from the earlier learning curve:
masters 901/1002/1103, independent and mixer models, checkpoints 128/512/2,048.
No testing features are accepted and no encoder is retrained.

Fit an outer normalizer on valid training rows, then compare centered PCA and a
fixed orthonormal Gaussian random projection at widths 12/24/36/48. PCA numerical
rank determines supported widths for both methods. Keep the native 96-vector
reference. Ridge penalty is 1; the secondary tanh probe has hidden width 16,
100 Adam steps and learning rate .01, with its own training-only normalizer.
Native and dimension-sweep nonlinear fits have different input parameter counts;
the same-width PCA/random comparison keeps readout capacity matched.

Every input runs all three declared repetitions: projection seeds 1701/1802/1903
and probe seeds 2701/2802/2903. Actual probe initialization is
`stream_seed(probe_seed,width)`, shared between methods at that width. Random
maps are shared across inputs with the same dimensions/repetition. Report all
repetitions without choosing a best seed. These reused validation cohorts supply
development diagnosis, not fresh confirmation.

## Frozen global bottleneck comparison

Keep the existing one-layer channel mixer, width 64, three temporal blocks,
patch length 8, channel width 32, decoder hidden width 128, batch 8, AdamW
learning rate .001/weight decay .0001, gradient clipping 1 and dropout 0.

| Variant | Global pooling | Sole reconstruction signal |
| --- | --- | --- |
| Current mixer, mode 0 | Equal mean over valid contextual channels | Per-channel contextual vectors, 3 by 32 |
| Mean global, mode 1 | Equal mean over valid contextual channels | Exact served global vector, 32 |
| Learned global, mode 2 | Semantic-ordered contextual vectors and observed-support bits through a two-layer MLP | Exact served global vector, 32 |

Modes 1/2 broadcast that one global vector to recipient decoders, together with
fixed patch/channel metadata. No channel-vector or token bypass reaches decoding.
Mode 2 adds 8,480 parameters: linear 99 to 64, GELU, linear 64 to 32. Register
its modules after all common modules, preserving common initialization. Local
and concatenated exports remain observed summaries of cotrained weights.

Current versus mean global changes reconstruction routing and total bottleneck
capacity from 96 to 32; it is not a pure objective comparison. Learned versus
mean global isolates pooling under the same global width and reconstruction path,
with the additional parameters and runtime disclosed.

All-absent globals are exact zero and invalid. Global support requires at least
one observed channel; complete concatenation requires every declared channel.
Semantic IDs determine learned-pool order; storage permutations cannot change
the global output. Missing channels cannot acquire valid local vectors from peers.
The encoder sees the final visible mask and never receives hidden target values;
the reconstruction objective receives its detached held-out observed targets.

Fresh masters are 1401/1502/1603. Each of reversal, level, amplitude and lag sign
has 32 training source pairs, 64 validation pairs and 64 testing pairs, C=3/H=32/F=3
float64 observations with 10% coordinate missingness. Train each task separately.
One continuous path preserves the scaler/optimizer/attempt streams through
checkpoints 0/128/512. The primary selector is mean lag validation PCA12 global
ridge accuracy over all three variants and masters on their common valid rows.
Choose one common nonzero budget, with exact ties favoring 128. An unsupported
primary entry excludes the whole budget. Store selection before generating any
testing sources; testing uses the named fresh `0x67626f7474657374` stream.

Final scores reuse selected checkpoints and fitted assets without refitting.
Report native global32/concatenation96 and matched PCA12/PCA36 ridge/nonlinear
scores, raw-oracle and mask/metadata controls, fixed-target training/validation/test
standardized MAE/Huber, and within-master source-group paired intervals.
The selected checkpoints also reuse the shared fixed-readout 12-case missingness
sweep. Parameter counts and synchronized training time are reported.

Before full training, a four-update gate checks every parameter, input and loss
on CUDA, finite gradients, changed weights, exact saved-checkpoint reconstruction
parity and the declared decoder input rank for all three variants. No CPU training
fallback is permitted. Feature extraction and fixed small probes use CPU.

## Source diversity and objective follow-up

After the global comparison, freeze a separate 32-versus-128 training-pair
experiment at 128/512 updates with fresh masters and unchanged validation/test
budgets. Match the architecture and recipe so distinct sources can be separated
from repeated training exposure. Preserve every variant's result; do not select
an architecture from its final testing score and describe that score as confirmation.

The [frozen diversity plan](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/diversity-plan.json)
uses masters 1701/1802/1903, all four tasks and all three variants. Both source
counts report validation at 128/512 and use the preregistered 512-update budget
for fresh testing, regardless of validation utility. Generate a 128-pair training
reservoir and take the first 32 complete pairs for the smaller cohort. Generate
the identical 64-pair validation cohort through the independent
`0x676276616c` stream; reuse the named fresh testing stream for 64 pairs after
selection. This keeps the training sets nested and the held-out populations
identical. The historical generator shuffles source roles based on total counts,
so merely changing its ordinary training-pair argument would not meet that contract.

This primary matrix changes the complete training cohort: encoder weights, input
scaler, feature normalization/PCA and readouts fit on 64 versus 256 rows. Its
32-to-128 difference measures the combined pipeline effect. An additive,
archive-only lag diagnosis fixes readout fitting to the same first 32 complete
source pairs for both frozen pipelines, with identical probe recipes and seeds.
It compares their shared validation predictions at 512 updates, without model
training, budget selection or testing access. Input-scaler differences remain
part of the frozen pipeline; even this follow-up does not isolate weights alone.

Only if compact/global access remains weak after those diagnostics, test one
additional representation objective against reconstruction-only. Masked views
must come from the same observation row; opposite-label observations sharing a
source group are never positive pairs. Permitted views must preserve the task
distinctions, and an anti-collapse constraint must prevent constant outputs.
That objective requires its own frozen card and correctness gates before training.

All stages are synthetic development evidence. Source-group intervals condition
on fitted models/readouts and do not measure retraining uncertainty. A useful
change should improve compact/global lag access across paired seeds while retaining
the other tasks and support contracts; consumer acceptance remains governed by
the [evaluation policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md).

## Recorded outcomes

The [new evidence capsule](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/)
retains the source/artifact inventory captured before edits, the frozen recipe,
container checks and experiment results.

### Compression diagnostic

All 18 archived inputs completed the declared widths and three repetitions in
119.48 seconds. Every training feature matrix had centered numerical rank 63,
so all four widths were supported. The following are descriptive validation
means over three masters and all three declared repetitions at width 36:

| Encoder | Updates | PCA ridge | Random ridge | PCA nonlinear | Random nonlinear |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent | 128 | 54.43% | 52.26% | 53.99% | 82.03% |
| Independent | 512 | 50.52% | 52.86% | 52.52% | 85.76% |
| Independent | 2,048 | 54.17% | 52.26% | 55.21% | 86.89% |
| Mixer | 128 | 76.82% | 66.75% | 75.26% | 82.73% |
| Mixer | 512 | 70.31% | 64.06% | 68.58% | 86.11% |
| Mixer | 2,048 | 60.42% | 58.42% | 60.94% | 87.33% |

Random projection improves nonlinear access substantially at this width, but
does not repair the weak linear result. PCA usually preserves the mixer's
linear lag access better. The small nonlinear probe also benefits from lower
PCA widths: the independent 128-update PCA12 mean is 82.38%, versus 53.99%
at PCA36. Compression, nuisance dimensions and fixed-probe fitting interact;
these scores do not establish information loss from PCA alone. Native-width
probes have different parameter counts and are separate references. Repetitions
are repeated readout fits, not independent encoder retrainings, and no best
projection/probe seed was selected.

The [diagnostic report](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/projection-diagnostic/report.json)
retains every input, map, fitted probe and conditional paired interval. These
validation sources were used previously, so this is development diagnosis.

### Fresh global bottleneck comparison

All 36 continuous CUDA paths completed 512 updates (18,432 total), preserving
108 checkpoints. Validation selected 128 updates: the equal-weight primary
utility was 71.35%, versus 68.49% at 512. Selection was saved before the fresh
testing cohorts were generated. The full run took 1,041.54 seconds including
feature fitting, reconstruction and the fixed-readout missingness sweep.

Fresh lag means over three masters at the selected common budget are:

| Variant | Global PCA12 ridge | Global native32 ridge | Global native32 nonlinear | Channel PCA36 ridge |
| --- | ---: | ---: | ---: | ---: |
| Current mixer | 66.67% | 66.93% | 57.29% | 92.71% |
| Mean global | 70.05% | 72.66% | 67.97% | 88.54% |
| Learned global | 78.65% | 83.59% | 80.73% | 88.54% |

Learned-minus-current global PCA12 ridge differences are +12.50 percentage
points [3.91, 21.88] for master 1401, +22.66 [14.06, 31.25] for 1502 and
+0.78 [-10.94, 11.72] for 1603. These are conditional 95% source-group intervals,
not uncertainty across retraining. Mean-minus-current intervals cross zero for
all three masters. Learned versus mean has one positive interval and two
crossing zero, so the additional pooling head is not a universal improvement.

Global PCA12 ridge reversal and level means are 100% for every variant;
amplitude is 100% for current/mean and 98.44% for learned. The narrower
reconstruction route has a cost: validation standardized MAE averaged over the
12 task/master paths at 128 is 0.16444 / 0.40296 / 0.37815 for current/mean/learned,
with full target coverage. Better global readouts do not imply better reconstruction.
The current-to-global comparison also changes decoder-input capacity.

Current and mean have 217,325 parameters; learned has 225,805 (+8,480, 3.90%).
Mean synchronized CUDA training time through 128 updates is 3.82 / 4.05 / 4.29
seconds per path. These are descriptive timings, not a repeated resource benchmark.
Native and matched probes have different capacities; the channel concatenation
is a distinct cotrained diagnostic and is not the sole global reconstruction vector.

The [full report](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/global-comparison/report.json)
and [independent audit](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/validation-main-20261006T163238886-60e024a2.json)
retain all tasks, surfaces, unsupported-fit handling, paired controls and stress.
The audit passes 153,929 assertions with no pending results. Defaults remain unchanged.

The learned head's validation improvement exceeds its modest average initial
advantage. Its global PCA12 ridge mean is 51.04% at zero updates, 77.34% at 128
and 79.17% at 512; native32 ridge is 54.95% / 84.11% / 90.10%. The last master
already has some random-initialization accessibility, and these are validation
curves with a train-fitted probe at each checkpoint. Untrained fresh-test
performance was not measured. Raw-oracle fresh-test accuracy is 100% and
mask/metadata ridge/nonlinear accuracy is 50% on all 12 task/master runs.

Severe fixed-readout corruption remains unresolved: with 90% additional random
deletion, global PCA12 ridge means are 51.04% / 50.26% / 49.48% for
current/mean/learned despite full observed support coverage. All-absent signal
surfaces correctly abstain. This sweep establishes measured support behavior,
not robustness acceptance.

### Fixed-budget source-diversity matrix

Both predeclared cohorts completed all four tasks, three variants and three
masters at 512 updates, adding 36,864 CUDA updates and 216 checkpoints. Their
validation, fresh testing and stressed observations are exactly shared; the
smaller training cohort is the first 32 complete pairs of the same 128-pair
reservoir. The 32-pair run took 883.04 seconds; the 128-pair run took 1,214.77
seconds. Both use the preregistered 512-update testing budget.

Fresh lag means over three masters, with full ordinary coverage, are:

| Variant | Training pairs | Global PCA12 ridge | Global PCA12 nonlinear | Global native32 ridge | Global native32 nonlinear |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current mixer | 32 | 61.46% | 61.20% | 65.36% | 58.07% |
| Current mixer | 128 | 65.63% | 62.76% | 69.27% | 62.50% |
| Mean global | 32 | 62.24% | 66.67% | 63.54% | 67.45% |
| Mean global | 128 | 59.38% | 70.83% | 70.31% | 60.94% |
| Learned global | 32 | 75.78% | 86.46% | 86.98% | 92.45% |
| Learned global | 128 | 76.56% | 98.70% | 94.79% | 100% |

Native learned-global nonlinear predictions are correct on all 384 fresh lag
rows in the larger cohort. PCA12 nonlinear scores by master are 99.22% / 100% /
96.88%. This supplies strong synthetic information-access evidence; a nonlinear
probe is part of that result. Compact ridge remains mixed: its 32-to-128 changes
by master are +15.63 / +5.47 / -18.75 percentage points. Learned-minus-current
compact ridge intervals are positive for two masters in each cohort and cross
zero for the third. More sources do not uniformly solve the linear benchmark.

Reversal and level global PCA12 ridge means remain 100%; amplitude is at least
99.48% across variants/cohorts. Raw-oracle accuracy is 100% and mask/metadata
accuracy is 50%. Learned channel PCA36 ridge/nonlinear means rise from
90.10% / 82.81% to 95.57% / 96.61%, as a separate cotrained diagnostic.

At 512, learned-global lag reconstruction is much closer to the current route
than in the 128-update comparison. Fresh standardized MAE current/mean/learned
is 0.04845 / 0.31264 / 0.07535 with 32 pairs and
0.04524 / 0.31124 / 0.07130 with 128 pairs, all with full target coverage.
The learned route still has a reconstruction cost versus the 96-value route.
Each cohort uses its own frozen training metric scaler; cross-cohort MAE values
are descriptive rather than a common-unit paired effect.

These are combined pipeline results: readout fitting also increases from 64 to
256 rows. The [combined analysis](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/combined-analysis.json)
retains every master, paired architecture comparison, control, cost and stress
result. The [full audit](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/validation-complete-20261006T171015540-bda05949.json)
passes 433,165 assertions with no failures or pending cohorts.

### Matched-size readout diagnosis

The [separate frozen plan](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/matched-readout-frozen-plan-v2.json)
and capsule-only helper keep production sources unchanged. They refit on the
same 64 training rows for both frozen pipelines, with identical probe seeds and
budgets, then score the same 128 validation rows. No encoder/checkpoint loading,
training, testing access or selection occurs. All 36 native feature cases
complete in 162.64 seconds; the report retains 324 paired metric entries.

Learned-global descriptive validation means over all three masters and declared
readout repetitions are:

| Readout, fixed 32 fitting pairs | Model trained on 32 pairs | Model trained on 128 pairs |
| --- | ---: | ---: |
| Global PCA12 ridge | 75.52% | 77.60% |
| Global PCA12 nonlinear | 87.85% | 85.68% |
| Global native32 ridge | 84.38% | 86.46% |
| Global native32 nonlinear | 92.53% | 88.28% |

PCA12 ridge's first-repetition paired changes are +17.97 [9.38, 25.78] percentage
points for master 1701, +5.47 [-4.69, 14.06] for 1802 and
-17.19 [-25.00, -9.38] for 1903. More encoder training sources therefore do not
produce a uniform gain under a fixed readout population. The near-perfect
full-pipeline results cannot be attributed solely to additional encoder data.
Input preprocessing differs too; this analysis compares frozen encoder/scaler
pipelines. It is additional validation diagnosis, not fresh confirmation.

The [paired diagnosis report](../../../output/runs/rpb-implementation/encoder-advance-b88ddd4b38/matched-readout-validation/report.json)
retains native/PCA/random results and all seeds; no favorable repetition is chosen.

### Decision after the recorded experiments

Keep learned semantic global pooling as an opt-in candidate through
[learned_global.conf](config/learned_global.conf), with mode 0 still the default.
It improves served-global access under equal readout budgets, and the larger
pipeline reaches 100% native nonlinear / 98.70% compact nonlinear synthetic lag
accuracy. The focal compact-linear benchmark remains uneven, reconstruction
has a cost, and strong corruption remains unresolved.

Do not add the proposed auxiliary consistency loss in this round: native/global
information access is now strong, and the remaining compact-linear result does
not establish an absence of learned information. A new objective would need a
separate hypothesis and frozen comparison. The immediate research target is
compact linear accessibility and independent confirmation on consumer data.
Every architecture remains reported; these test scores do not promote a default.

### Active research direction

For the 128-pair, fixed-512-update timing cohort, native32 linear/neural means are
69.27% / 62.50% for RPB-v2 and 94.79% / 100% for RPB-v4. These use the encoder
outputs directly. The former's reconstruction uses 96 contextual signal values;
the latter's uses the exact 32-number global embedding. Reconstruction errors
therefore describe different bottleneck capacities, while native32 head inputs
have equal size.

Next compare RPB-v4 to raw data and standalone raw PCA32 with fixed heads, then
run a fresh native-only protocol. Keep RPB-v2 as a frozen historical reference
and compatibility input rather than retraining it in every development cycle.
