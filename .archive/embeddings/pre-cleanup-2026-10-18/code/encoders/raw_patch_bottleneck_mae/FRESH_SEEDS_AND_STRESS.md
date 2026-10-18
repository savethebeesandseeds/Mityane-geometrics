# RPB-MAE fresh seeds and fixed-readout stress

Date: 2026-10-06. Evidence stage: `development`. Status: Implemented, validated
and measured under the declared recipe. Four of five fresh primary intervals
are positive; seed 704 crosses zero. Heavy missingness erodes the advantage.

This experiment repeats the [aligned-mixer comparison](CHANNEL_MIXER_ADVANCE.md)
on five fresh development seeds and measures survival of the same fitted
readouts under observation deletion. The previous three seeds guided development
and remain archived. Five new seeds do not by themselves establish consumer
confirmation: acceptance thresholds, resource limits, relevant consumer data and
across-training-run/multiplicity rules remain incomplete under the
[shared policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md).

The focal base comparison remains **lag-sign ridge at joint PCA width 36,
trained contextual mixer versus trained independent RPB (`pair-1`)**. Stress
outcomes are diagnostics. No task, width, probe, training budget or stopping rule
will be changed after scores, and a secondary result cannot replace the primary.

## Frozen model and base recipe

| Item | Declared choice |
| --- | --- |
| Providers | Independent `rpb` and contextual `rpb_mixer`, each with its matched untrained initialization |
| Configurations | Unchanged [evaluation.conf](config/evaluation.conf) and [channel_mixer.conf](config/channel_mixer.conf); mixer layers 0 and 1 respectively |
| Geometry | CPU float64 raw BCHF, C3/H32/F3, P8, D32 local/contextual; IDs `0,1,2`, unitless features, interval 1 and endpoint 31 |
| Training | 128 completed masked-Huber updates per provider/task/seed, replacement batches of 8; existing AdamW/gradient-clip recipe and 10,000-attempt limit |
| Matched exposure | Same source observations, common-parameter initialization and per-attempt row/mask streams; separate optimizer trajectories |
| Base population | `paired-controls-v1`, noise standard deviation `0.005`, class-independent coordinate missingness `0.1`, same observation mask within each source pair |
| Splits | 32 training, 16 validation and 64 testing source pairs, giving 64/32/128 variant rows |
| Tasks | Reversal, level, amplitude and lag sign, fixed before generation |
| Fresh seeds | **401, 502, 603, 704, 805**; no reuse of 101/202/303 as fresh evidence |
| Dimensions | Native global 32 and concatenation 96; matched global 12 and joint concatenation 36 |
| Probes | Ridge penalty 1; separate tanh-16, 100 full-batch Adam updates at `0.01`, one fixed surface/tier-derived initialization, no restarts |
| Selection | Final fixed-update model; no validation checkpoint/probe selection, width adaptation or test-driven early stopping |
| Inference | Frozen weights/preprocessing, evaluation mode, one Torch CPU thread |

Each task/seed fits fresh models on permitted training observations only; this
is not transfer testing of one universal checkpoint. Labels reach shared probe
fitting/scoring only. Encoder/scaler fit callbacks receive no labels, hidden
clean values or held-out observations. Equal updates and sampled-row exposure
do not match the mixer and independent model's parameter counts or compute.

The original seven declared pairs remain fixed in this order:

| ID | Candidate | Comparator | Tier |
| --- | --- | --- | --- |
| pair-1 | `rpb_mixer_trained_contextual_channel_concatenation` | `rpb_trained_channel_concatenation` | `matched_channels` (36) |
| pair-2 | `rpb_mixer_trained_contextual_channel_concatenation` | `rpb_mixer_untrained_contextual_channel_concatenation` | `matched_channels` (36) |
| pair-3 | `rpb_trained_channel_concatenation` | `rpb_untrained_channel_concatenation` | `matched_channels` (36) |
| pair-4 | `rpb_mixer_trained_contextual_global` | `rpb_trained_global` | `matched_global` (12) |
| pair-5 | `rpb_mixer_trained_contextual_channel_concatenation` | `rpb_trained_channel_concatenation` | `native` (96) |
| pair-6 | `rpb_mixer_trained_contextual_channel_concatenation` | `rpb_mixer_untrained_contextual_channel_concatenation` | `native` (96) |
| pair-7 | `rpb_mixer_trained_channel_concatenation` | `rpb_trained_channel_concatenation` | `matched_channels` (36) |

Pair-7 is the independent pre-mixer local readout of contextually trained shared
weights, not a separately supervised local branch. All pairs besides the focal
base lag/ridge result, all other tasks and nonlinear scores remain diagnostic.

## Fixed-readout side protocol

The shared side protocol is `fixed-readout-stress-v1`, selected with
`--stress-sweep fixed-readout-v1`. It is off by default; the existing
[controlled-pairs-v2 base protocol](../../evaluation/cards/controlled_pairs_v2.md)
and historical reports retain their meanings.

Fit the model/scaler, raw-control scaler, all feature normalization stages, PCA,
ridge, nonlinear and shuffled-label probes **once** on permitted base training
data. Apply the exact fitted providers, transforms, ridge and nonlinear readouts
to each corrupted testing view. The shuffled-label probe remains fitted once and
reported in the unchanged base evaluation; it is not reevaluated for each stress
case. No stress condition refits assets, reconstructs a provider, adds training
updates, changes PCA width or trains a replacement readout. Validation is
reported without selection and is not part of the stress sweep. Base-unsupported
compression remains unsupported in every case.

There are exactly twelve cases for every testing population:

| Case | Observation deletion |
| --- | --- |
| Intact | Original testing observations, including their existing 10% nominal natural missingness |
| Additional dropout 10% | Delete base-observed coordinates where the shared uniform value is below `0.1` |
| Additional dropout 30% | Same uniform field, threshold `0.3` |
| Additional dropout 60% | Same uniform field, threshold `0.6` |
| Additional dropout 90% | Same uniform field, threshold `0.9` |
| Temporal blackout 25% | Delete 8 consecutive steps across every channel and feature |
| Temporal blackout 50% | Delete 16 consecutive steps across every channel and feature |
| Temporal blackout 75% | Delete 24 consecutive steps across every channel and feature |
| Semantic channel 0 absent | Delete its entire history/features |
| Semantic channel 1 absent | Delete its entire history/features |
| Semantic channel 2 absent | Delete its entire history/features |
| All channels absent | Delete every observed coordinate |

For each independent source, draw one class-independent uniform field `[C,H,F]`
for dropout. Reuse it across severities, both transformed variants and every
provider, so the dropout masks are nested. The rates describe **additional**
deletion of already observed data; report actual retained cells as well as the
requested rate rather than calling them the final missing fraction.

For temporal blackouts, draw one integer anchor uniformly from `0..H-1` per
source and reuse it across severities. For length L, use
`start = clamp(anchor - floor(L/2), 0, H-L)` and delete `[start,start+L)` across
all C/F. At H32, L is 8, 16 or 24. These centered, edge-clamped windows are
nested; their start positions are **not uniformly distributed**. Save the
anchors/intervals or an equivalent reproducible mask manifest.

Named corruption streams depend on run seed, task and source identity, never
labels, variant order or provider identity. Complete paired variants receive
identical corruption masks. Channel deletion follows semantic IDs rather than
an assumed storage slot. New visibility is the original observation mask AND
the retained-support mask; it never adds observations. Set hidden raw storage to
zero while preserving visible values, IDs, units, timestamps and endpoints.
Save exact masks/source mappings and checksums. All providers receive isolated
copies of the same corrupted observations.

## Support and scoring

External eligibility remains all 128 testing variant rows from 64 source groups,
fixed before method validity. Report the following separately for each
surface/tier/probe/task/case:

| Quantity | Definition |
| --- | --- |
| Coverage | `N_valid / N_total` |
| Conditional accuracy | `N_correct / N_valid`; null/unsupported if no predictions are valid |
| Full-population correctness | `N_correct / N_total`, counting invalid/abstained signal predictions as failures |
| Support accounting | Eligible/valid/abstained rows and source groups, class counts, observed/absent semantic channels, retained cells and exceptions |

Only valid predictions contribute to `N_correct`; an argmax from an invalid zero
placeholder is not a prediction. Report failed/unsupported outcomes and numerical
exceptions explicitly rather than silently removing them or assigning a finite
conditional score.

Global RPB surfaces require at least one observed-valid channel. Complete channel
concatenation requires every declared constituent channel and uses semantic
order `0,1,2`. Thus removing one whole channel gives concatenation zero coverage
and null conditional accuracy, while globals can remain valid from observed
neighbors. The contextual mode does not infer observed support for absent
channels. With all channels absent, signal globals and concatenations have zero
coverage, null conditional accuracy and zero full-population correctness.
Mask/metadata controls retain their separately declared support; their scores
cannot be reported as observed-signal coverage.

Each declared method pair has its own corruption-specific common-valid
intersection, with class/source counts and coverage. Unrelated providers do not
change it. Conditional paired effects on that intersection and full-population
correctness answer different questions; high common-valid accuracy alone does
not establish robustness. Empty intersections remain unsupported. Any base-to-
stress comparison also uses the same source rows and declares its population.

The base legal-observation oracle gate remains a construction check. Under heavy
stress, oracle accuracy/support is descriptive: the run does not require 95%
oracle recovery after deleting relevant observations. Raw and mask controls and
the initialized models remain visible alongside the trained surfaces in stress
results; the existing row-shuffled-label diagnostic remains in the base report.

## Uncertainty and fresh-data scope

Nominal paired 95% intervals use 1,000 bootstrap replicates over complete testing
source groups, retaining transformed variants together. Conditional effects use
the pair's declared common-valid population; full-population measures retain
abstentions in the original denominator. There are 64 testing sources, not 128
independent rows, and twelve views of a source are not twelve independent
trajectories. Do not pool stress cases into an enlarged independent sample.

Report each fresh seed separately. Intervals condition on its fitted model,
compression and probe instances; they do not include uncertainty across training
seeds or nonlinear initializations. Surface-name-derived nonlinear seeds differ
between providers. Descriptive seed means or spreads are not an across-run
confidence interval. This development track assigns no new multiplicity-adjusted
decision or consumer threshold.

The fresh master seeds change generator draws/splits, model training and named
probe streams. This is fresh paired development replication, not an experiment
varying only training initialization. Archive source IDs/checksums and confirm
their declared seed namespace is disjoint from earlier development runs. Complete
the full five-seed/four-task budget regardless of intermediate results; retain
failed or unsupported outcomes, and do not stop after the first favorable seed.

## Pre-score evidence capsule and reproduction

Before generation or fitting, save `experiment-plan.json` and an immutable copy
of this document. The companion plan supplies the complete resolved training
and stress recipe; the generic card alone does not capture every adapter budget.

| Capsule field | Required retained evidence |
| --- | --- |
| Scope | Date, policy/protocol/stress versions, development stage, primary task/pair/tier/probe and diagnostic roles |
| Exact invocation | Binary path, complete CLI arguments, unique output directory and intended full run budget |
| Configurations | Independent/mixer raw configuration contents and SHA256 hashes; actual 128-update overrides, batch size and actual per-run seeds |
| Source/runtime identity | Evaluation producer, RPB core/writer and minimum-driver fingerprints with algorithm/scope; Git HEAD/dirty state when available; runtime/device/precision/thread identity |
| Population | Generator/version, shape, IDs/units/grid/endpoint, split budgets, fresh source/seed namespaces, natural missingness and noise |
| Corruptions | Twelve cases, thresholds and anchor formula, named seed policy, exact support masks/checksums and source/variant mappings |
| Frozen assets | Model/scaler identities and fit-source manifests, update/skip counts, feature normalization/PCA/probe files and unsupported fits; no stress refitting |
| Analysis | Individual and pair-common populations, full-population abstention rule, within-run source-group uncertainty and explicit missing acceptance contracts |

Use the existing managed container and full evaluation binary. Allocate a new
parent/results path; root orchestration saves the capsule before this command:

```bash
eval_bin=/opt/cuwacunu_embedding/build/rpb-implementation/embedding_evaluate
mkdir -p /embedding/output/runs/rpb-implementation
run_parent="$(mktemp -d /embedding/output/runs/rpb-implementation/fresh-stress-XXXXXX)"
# Save experiment-plan.json and an immutable pre-score recipe before this call.
"$eval_bin" evaluate --output "$run_parent/results" --encoders rpb,rpb_mixer \
  --rpb-config /embedding/code/encoders/raw_patch_bottleneck_mae/config/evaluation.conf \
  --rpb-steps 128 \
  --rpb-mixer-config /embedding/code/encoders/raw_patch_bottleneck_mae/config/channel_mixer.conf \
  --rpb-mixer-steps 128 --seeds 401,502,603,704,805 \
  --tasks reversal,level,amplitude,lag_sign \
  --channels 3 --history 32 --features 3 --channel-ids 0,1,2 \
  --units unitless,unitless,unitless \
  --train-pairs 32 --validation-pairs 16 --test-pairs 64 \
  --matched-global-width 12 --matched-channel-width 36 --threads 1 \
  --card-id controlled-pairs-v2 --stress-sweep fixed-readout-v1 \
  --compare rpb_mixer_trained_contextual_channel_concatenation,rpb_trained_channel_concatenation,matched_channels \
  --compare rpb_mixer_trained_contextual_channel_concatenation,rpb_mixer_untrained_contextual_channel_concatenation,matched_channels \
  --compare rpb_trained_channel_concatenation,rpb_untrained_channel_concatenation,matched_channels \
  --compare rpb_mixer_trained_contextual_global,rpb_trained_global,matched_global \
  --compare rpb_mixer_trained_contextual_channel_concatenation,rpb_trained_channel_concatenation,native \
  --compare rpb_mixer_trained_contextual_channel_concatenation,rpb_mixer_untrained_contextual_channel_concatenation,native \
  --compare rpb_mixer_trained_channel_concatenation,rpb_trained_channel_concatenation,matched_channels
```

## Completed measurements

The [base report](../../../output/runs/rpb-implementation/fresh-stress-24846234a7/results/report.json)
contains all 20 declared task/seed runs and 140 base comparisons. Every base pair
has 128 testing rows from 64 complete source groups. The legal-raw oracle scores
100% in all runs; measured mask-only probes score 50%. There are 580 measured
feature/tier entries and 20 explicitly unsupported mask-only PCA36 fits.

The focal base lag/PCA36 ridge result is:

| Fresh seed | Contextual mixer accuracy | Independent accuracy | Mixer minus independent, percentage points | Nominal within-run 95% interval, percentage points |
| --- | ---: | ---: | ---: | ---: |
| 401 | 87.50% | 42.97% | +44.53 | [+35.16, +53.91] |
| 502 | 79.69% | 48.44% | +31.25 | [+21.88, +40.63] |
| 603 | 85.16% | 51.56% | +33.59 | [+24.22, +42.19] |
| 704 | 60.16% | 53.13% | +7.03 | [-1.56, +14.84] |
| 805 | 85.94% | 47.66% | +38.28 | [+28.13, +49.22] |

The descriptive seed means are 79.6875% versus 48.75%. Four primary intervals
are positive; the fifth crosses zero. The contextual trained-versus-own-
initialization PCA36 ridge diagnostic is positive in all five seeds. The
independent trained-versus-initialization PCA36 ridge intervals cross zero in
all five. Other primary tasks are at 100% for both models and distinguish no
architecture benefit in this population.

The global PCA12 comparison is weaker: its descriptive lag means are 57.97%
versus 49.53%, with two strictly positive base intervals and one interval whose
lower endpoint is zero. A concatenated channel-vector advantage does not establish
equivalent quality in one pooled global vector. Native96 nonlinear architecture
comparisons are also mixed: two positive intervals, two crossing zero and one
whose upper endpoint is zero. They remain separate diagnostics.

## Fixed-readout stress results

The [stress report](../../../output/runs/rpb-implementation/fresh-stress-24846234a7/results/stress-report.json)
contains all 240 run/case views, 6,960 measured feature/tier entries, 240 retained
unsupported base PCA fits and 1,680 pair entries. The table gives descriptive
five-seed lag means for pair-1's frozen PCA36 ridge readout. Correctness uses the
complete test population, including abstentions as failure.

| Testing case | Mixer full-population correctness | Independent full-population correctness | Coverage of each model |
| --- | ---: | ---: | ---: |
| Intact | 79.69% | 48.75% | 100% |
| Additional coordinate deletion 10% | 74.69% | 51.41% | 100% |
| Additional coordinate deletion 30% | 65.31% | 49.84% | 100% |
| Additional coordinate deletion 60% | 54.84% | 51.56% | 100% |
| Additional coordinate deletion 90% | 50.94% | 50.31% | 99.6875% |
| Temporal blackout 25% | 61.41% | 52.50% | 100% |
| Temporal blackout 50% | 51.25% | 52.97% | 100% |
| Temporal blackout 75% | 51.88% | 52.34% | 100% |
| Semantic channel 0 absent | 0% | 0% | 0%; conditional accuracy null |
| Semantic channel 1 absent | 0% | 0% | 0%; conditional accuracy null |
| Semantic channel 2 absent | 0% | 0% | 0%; conditional accuracy null |
| All channels absent | 0% | 0% | 0%; conditional accuracy null |

At 10% and 30% additional deletion, four of five pair-1 stress intervals are
positive. At 60%, one is positive; at 90%, none is positive. A 25% temporal
blackout retains two positive intervals; 50% retains none, and 75% includes one
negative interval. The difference between intact and corrupted predictions is
separately measured on matched populations in the reports. These correlated
diagnostics do not create a new acceptance rule.

Whole-channel deletion correctly eliminates complete-concatenation coverage.
Global surfaces retain 100% observed support when another channel remains, but
their lag correctness stays near chance when channel 0 or 1 is removed. Removing
the nuisance channel 2 gives global pair-4 descriptive means 59.38% versus 48.75%,
with two positive intervals. All-absent signal globals and concatenations have
zero support, null conditional accuracy and zero full-population correctness.
Mask-only controls retain their declared control support and 50% accuracy.

High observed coverage under heavy random deletion or temporal gaps therefore
does not imply that the frozen readout preserves lag information. This recipe
supports compact linear accessibility with light missingness and exposes a
robustness limitation under larger losses. It supplies no first-encoder
superiority, cost equivalence, real-consumer performance, formal confirmation
or promotion claim.

## Validation and retained evidence

The [evidence capsule](../../../output/runs/rpb-implementation/fresh-stress-24846234a7/experiment-plan.json)
retains the exact invocation/configurations and producer identities. Its immutable
`frozen-recipe.md` is the pre-score document, SHA256
`6FF2A20BC6C50231EA1BA8A2B6F84E9B89642C35EC20E916D812D49C6C7DA930`.
Results and masks are local ignored artifacts; a clean checkout must reproduce
the runs to resolve those links.

Container checks pass shared fitting/orchestration, generic stress primitives
and both RPB adapter modes. Stress on/off preserves ordinary reports/cards,
manifests, provider audits and fitted tensors exactly. Tests reproduce a
corrupted prediction from the ordinary saved normalization/PCA/probe tensors,
check immutable fit guards, fit/save-once behavior, provider-clone isolation,
nested source-shared label-independent masks, hidden NaN zeroing and insufficient
support/null-interval handling.

The [seed 502 repeat](../../../output/runs/rpb-implementation/fresh-stress-24846234a7/repeat-validation.json)
reproduces its entire base and stress run JSON and all 16 JSON sidecars exactly.
The [default-path regression](../../../output/runs/rpb-implementation/fresh-stress-24846234a7/default-path-regression.json)
preserves all 30 archived seed202 feature entries, seven pairs and 1,087
non-source tensor buffers; only expected producer fingerprints differ.
The [first-encoder regression](../../../output/runs/rpb-implementation/fresh-stress-24846234a7/baseline-regression-validation.json)
preserves 25 prior baseline/independent feature entries, three pairs, fitted
assets and optimizer state, and validates baseline all-missing support through
the same generic sweep. Optimizer process-address serialization keys are mapped
through saved parameter-group order for semantic checkpoint comparison.

The [validation manifest](../../../output/runs/rpb-implementation/fresh-stress-24846234a7/validation.json)
checks all source identities, completed budgets, disjoint fresh source namespaces,
coverage and artifact counts. All 1,685 prior archived files are byte-preserved.
All 56 preexisting encoder source/test/config/script files remain unchanged;
only shared evaluation integration/tests, the new generic module and CLI/build
composition changed. Ordinary RPB still has no evaluator/baseline dependency;
the minimum evaluator still has no RPB dependency. Resource measurements,
consumer contracts and quantitative contextual decoder reliance remain separate
work.
