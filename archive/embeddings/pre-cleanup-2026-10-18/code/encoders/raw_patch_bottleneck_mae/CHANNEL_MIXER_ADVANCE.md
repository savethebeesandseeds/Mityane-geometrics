# RPB-MAE aligned channel-mixer advance

Date: 2026-10-05. Evidence stage: `development`. Status: Implemented, validated
and measured under the frozen recipe. The primary compact linear lag comparison
has positive paired intervals in all three development seeds.

This extension tests whether explicit cross-channel processing before temporal
pooling improves compact linear lag accessibility. The
[independent-parent experiment](FROZEN_PROBE_ADVANCE.md) did not establish its
primary PCA36 ridge benefit over initialization, while native96 nonlinear
readouts showed positive training effects. That evidence motivates this mixer
experiment; it does not establish that independent exports lack lag information.
The [shared policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md),
[implementation guidelines](RPB_MAE_implementation_guidelines.md#aligned-mixer-experiment)
and [controlled-pairs-v2 card](../../evaluation/cards/controlled_pairs_v2.md)
govern the contracts and evidence limits.

## Lessons from the first encoder

The [baseline encoder](../mtf_jepa_mae_vicreg/include/embedding/encoders/mtf_jepa_mae_vicreg/encoder.h#L90)
computes a masked mean over all visible time/frequency/channel tokens, projects
that mean and broadcasts it into token updates at every layer. Its channel
vectors are therefore contextual before pooling. This mechanism weights visible
tokens, rather than averaging channels equally, and it does not align channel
interactions at a particular original patch position. An aligned channel mean
would be a new control, not a reproduction of that baseline mechanism.

The baseline [MAE decoder](../mtf_jepa_mae_vicreg/include/embedding/encoders/mtf_jepa_mae_vicreg/encoder.h#L218)
cross-attends to context token states. Its reconstruction objective consequently
does not require decoding solely from the served pooled vectors. RPB retains its
explicit compact bottleneck: target predictions receive the exact active D32
export and target metadata, with no token-state or raw-context bypass.

The baseline [tokenizer](../mtf_jepa_mae_vicreg/include/embedding/encoders/mtf_jepa_mae_vicreg/tokenization.h#L141)
uses its declared channel-slot geometry for channel embeddings and metadata.
RPB instead carries semantic channel IDs and maps them to fitted scaler and
embedding-table entries. Moving complete records together must permute
channel-indexed outputs and preserve set-pooled globals; moving values while
leaving IDs fixed is a different input. The mixer introduces no storage-position
embedding that would defeat this contract.

Both implementations mask invalid states after biased updates. The mixer extends
that discipline to query eligibility and entirely invalid attention groups.
Observed validity is not inferred from a neighbor or from a nonzero learned
bias. These are implementation lessons, not evidence that the baseline mixer
improves the new task: the earlier short shared-driver baseline runs supplied
development/wiring comparisons, not a causal channel-mixer ablation.

## Implementation contract

`Config.channel_mixer_layers` defaults to zero. The independent model remains the
default, with its existing local/global outputs and mixer-off parameter layout.
The declared extension has one additional pre-normalized residual channel block,
width 64, four attention heads, feed-forward width 256 and dropout zero. It reuses
the existing temporal encoder, precision-preserving frozen scaler, tokenizer,
readout and small decoder.

~~~text
visible temporal states + original positions      [B*C,L_max,64]
  -> scatter original patch indices; preserve visibility
temporal grid + observed-visible patch support     [B,C,4,64], [B,C,4]
  -> group by original patch position
channel attention                                 [B*4,C,64]
  -> residual block; restore grid; valid temporal pooling
contextual exports                                [B,C,32]
  -> exact-export decoder + target channel/patch metadata
~~~

Only retained visible temporal states enter the scatter; it never rebuilds
states from unmasked observations. Unequal packed layouts such as `{0,2}` and
`{1,2}` meet at original patch 2, not at their first packed slot. The current
uniform schema provides a common endpoint and sampling interval across channels.
This does not support independently shifted grids or irregular sampling without
a separately tested alignment adapter.

Both query and key eligibility equal observed-visible patch support. Entirely
invalid groups are excluded before attention, and invalid queries are zeroed
after residual/feed-forward biases. Pooling excludes invalid queries. Absent
channels remain finite zero/invalid even when other channels are observed; an
entirely absent sample also remains zero/invalid. There is no inferred-channel
mode. Temporal states already see the permitted history, so an interaction at
patch k does not create a per-patch causal output; the endpoint contract remains.

`z_local [B,C,32]` remains the independent pre-mixer readout, and `z_global`
remains its equal-valid-channel mean. The extension separately exposes
`z_contextual [B,C,32]` and its equal-valid-channel `z_contextual_global`.
Contextual and local readouts share pooling/projection parameters. With mixing
enabled, reconstruction trains through exactly `z_contextual`; the local output
is a diagnostic of jointly trained weights, not an independently supervised
branch. The decoder does not concatenate both exports as extra hidden capacity.

Mixer-off construction must preserve common parameters, registration order and
random draws. Mixer-on construction retains identical initialization of common
parameters for the paired independent/mixer comparison. Contextual checkpoint
and export semantics must be identified and validated rather than silently
written as independent outputs. Unchanged raw input/scaler mathematics remain
reusable through explicit schema compatibility. Correctness evidence must cover
old independent checkpoint loading, round-trip inference and resume, in addition
to the new contextual mode.

Required checks include original-position alignment with unequal layouts;
missing-channel/all-missing safety; semantic-record permutations; pre-mixer local
independence and contextual dependence; fixed-asset hidden-target perturbation
and gradient isolation; gradients through the mixer and exact active bottleneck;
and exact-export decoding. Passing these establishes the tested contracts,
not representation quality.

## Frozen development recipe

| Item | Declared choice |
| --- | --- |
| Providers | Fresh independent `rpb` and fresh contextual `rpb_mixer`, each with its matched untrained initialization |
| Configuration | Independent [evaluation.conf](config/evaluation.conf); contextual [channel_mixer.conf](config/channel_mixer.conf) with `channel_mixer_layers=1` |
| Geometry | CPU float64 raw BCHF, C3/H32/F3, P8, local/contextual D32; semantic IDs `0,1,2`, unitless features, interval 1 and endpoint 31 |
| Training | 128 completed masked-Huber updates per provider/task/seed, replacement batch size 8, AdamW learning rate `0.001`, weight decay `0.0001`, gradient clipping 1, attempt limit 10,000 |
| Common initialization/exposure | Same common-parameter initialization, source observations, per-attempt row sampling and mask streams; each provider keeps its own optimizer trajectory |
| Population | `paired-controls-v1`, measurement noise standard deviation `0.005`, class-independent coordinate missingness `0.1`, shared mask within each transformed pair |
| Splits | 32 training, 16 validation and 64 testing source pairs: 64/32/128 variant rows |
| Tasks/seeds | Reversal, level, amplitude and lag sign; development seeds 101, 202 and 303 |
| Feature preparation | Frozen encoder/scaler; valid-training-only feature normalization and centered PCA; matched global width 12, joint channel-concatenation width 36 |
| Readouts | Primary ridge, penalty 1; fixed secondary tanh-16, 100 full-batch Adam updates at `0.01`; one surface/tier-derived initialization, no restarts |
| Selection | Fixed final training budget; no validation checkpoint/probe selection, test-driven width changes or sequential stopping |
| Inference | Evaluation mode, frozen parameters, one Torch CPU thread |

Both models receive equal update and sampled-row budgets, not matched compute or
parameter budgets. The mixer adds parameters and channel attention. Any quality
effect cannot support an efficiency or cost claim without separately measured
resources. Each task/seed fits its own fresh models; this is not transfer
evaluation of a universal checkpoint.

## Declared comparisons

The seven pairs below are fixed in this order. The **first pair's lag-sign ridge
effect at PCA width 36** is the focal primary comparison. All other pair/task
results and nonlinear readouts are diagnostics or secondary evidence. A
secondary win cannot rescue a primary result that does not establish benefit.

| ID | Candidate | Comparator | Tier |
| --- | --- | --- | --- |
| pair-1 | `rpb_mixer_trained_contextual_channel_concatenation` | `rpb_trained_channel_concatenation` | `matched_channels` (36) |
| pair-2 | `rpb_mixer_trained_contextual_channel_concatenation` | `rpb_mixer_untrained_contextual_channel_concatenation` | `matched_channels` (36) |
| pair-3 | `rpb_trained_channel_concatenation` | `rpb_untrained_channel_concatenation` | `matched_channels` (36) |
| pair-4 | `rpb_mixer_trained_contextual_global` | `rpb_trained_global` | `matched_global` (12) |
| pair-5 | `rpb_mixer_trained_contextual_channel_concatenation` | `rpb_trained_channel_concatenation` | `native` (96) |
| pair-6 | `rpb_mixer_trained_contextual_channel_concatenation` | `rpb_mixer_untrained_contextual_channel_concatenation` | `native` (96) |
| pair-7 | `rpb_mixer_trained_channel_concatenation` | `rpb_trained_channel_concatenation` | `matched_channels` (36) |

Pair-7 measures the mixer model's independent pre-mixer local readout after
contextual training. It helps distinguish changes in shared local weights from
the contextual surface, but is not an independently trained local branch or a
complete causal decomposition. Native global width is 32 and native concatenation
width is 96. Concatenation uses semantic order `0,1,2`, requires every observed-valid
constituent and is one joint probe, not separate per-channel accuracy.

Each pair has its own fixed candidate-and-comparator validity intersection;
unrelated controls cannot change that denominator. Report individual coverage,
the common population's class/source counts and unsupported outcomes. PCA
requires adequate valid training rows and numerical rank. Unsupported widths
remain unsupported rather than being narrowed after scores.

The shared engine also reports observed-input oracles, legal raw/support and
mask/metadata controls, and its fixed row-shuffled-training-label ridge
diagnostic. Baseline embeddings/descriptors are outside this selected provider
set. No result from this experiment establishes superiority over the first
encoder or the strongest reproducible same-budget descriptor. Oracle solvability
and shortcut controls must accompany interpretation, particularly when raw
linear lag probes are weak.

Paired nominal 95% intervals use 1,000 bootstrap replicates over 64 testing source
groups, retaining both transformed variants together. The 128 variant rows are
not independent units. Intervals condition on the fitted models, compression and
probes; they do not include between-training-run or readout-initialization
uncertainty. Nonlinear seeds are derived from surface/tier names, so the compared
providers receive different fixed probe initializations. Report every seed
separately; these exploratory intervals are not multiplicity-adjusted acceptance
tests.

## Reproduction and evidence status

Use the existing managed container and the full evaluation binary. Save the
complete companion `experiment-plan.json` before protocol generation/fitting,
including exact CLI, configuration contents, source hashes, resolved update
budgets, seed formulas and primary role. Preserve an immutable copy of this
pre-score document. The generic `evaluation-card.json` fixes shared geometry,
population, probes and pairs but does not capture the full adapter training
recipe by itself.

```bash
eval_bin=/opt/cuwacunu_embedding/build/rpb-implementation/embedding_evaluate
mkdir -p /embedding/output/runs/rpb-implementation
run_parent="$(mktemp -d /embedding/output/runs/rpb-implementation/channel-mixer-XXXXXX)"
# Save experiment-plan.json and an immutable pre-score recipe before this call.
"$eval_bin" evaluate --output "$run_parent/results" --encoders rpb,rpb_mixer \
  --rpb-config /embedding/code/encoders/raw_patch_bottleneck_mae/config/evaluation.conf \
  --rpb-steps 128 \
  --rpb-mixer-config /embedding/code/encoders/raw_patch_bottleneck_mae/config/channel_mixer.conf \
  --rpb-mixer-steps 128 --seeds 101,202,303 \
  --tasks reversal,level,amplitude,lag_sign \
  --channels 3 --history 32 --features 3 --channel-ids 0,1,2 \
  --units unitless,unitless,unitless \
  --train-pairs 32 --validation-pairs 16 --test-pairs 64 \
  --matched-global-width 12 --matched-channel-width 36 --threads 1 \
  --card-id controlled-pairs-v2 \
  --compare rpb_mixer_trained_contextual_channel_concatenation,rpb_trained_channel_concatenation,matched_channels \
  --compare rpb_mixer_trained_contextual_channel_concatenation,rpb_mixer_untrained_contextual_channel_concatenation,matched_channels \
  --compare rpb_trained_channel_concatenation,rpb_untrained_channel_concatenation,matched_channels \
  --compare rpb_mixer_trained_contextual_global,rpb_trained_global,matched_global \
  --compare rpb_mixer_trained_contextual_channel_concatenation,rpb_trained_channel_concatenation,native \
  --compare rpb_mixer_trained_contextual_channel_concatenation,rpb_mixer_untrained_contextual_channel_concatenation,native \
  --compare rpb_mixer_trained_channel_concatenation,rpb_trained_channel_concatenation,matched_channels
```

## Completed development measurements

The [completed report](../../../output/runs/rpb-implementation/channel-mixer-dc13a548aa/results/report.json)
contains all 12 task/seed runs, 30 feature/tier entries per run and 84 declared
pair comparisons. The
[pre-generation plan](../../../output/runs/rpb-implementation/channel-mixer-dc13a548aa/experiment-plan.json),
[immutable pre-score recipe](../../../output/runs/rpb-implementation/channel-mixer-dc13a548aa/frozen-recipe.md)
and [instantiated card](../../../output/runs/rpb-implementation/channel-mixer-dc13a548aa/results/evaluation-card.json)
retain the frozen choices. Every pair was measured on its complete testing
population: 128 variant rows, 64 source groups and 64 rows per class, with 100%
common coverage. All requested RPB PCA fits were supported. The 12 unsupported
entries are mask/metadata-only PCA36 fits with insufficient numerical training
rank; they were retained without narrowing the requested width.

The focal primary, `pair-1`, is lag-sign ridge on matched 36-dimensional channel
concatenation. Testing accuracies and paired mixer-minus-independent effects are
shown below. Differences and 95% intervals are percentage points; the report
retains unrounded values.

| Seed | Mixer contextual PCA36 ridge | Independent PCA36 ridge | Difference (95% interval), pp |
| --- | ---: | ---: | ---: |
| 101 | 90.63% | 52.34% | +38.28 (+30.47 to +45.31) |
| 202 | 76.56% | 48.44% | +28.13 (+17.19 to +39.06) |
| 303 | 63.28% | 52.34% | +10.94 (+1.56 to +19.53) |

All three primary lower bounds are positive. The arithmetic development-seed
means are **76.8229% mixer versus 51.0417% independent**, a descriptive difference
of 25.7813 percentage points. These means are not confidence intervals across
training runs. The effect varies substantially by seed, and the weakest lower
bound is 1.56 percentage points.

This supplies development evidence that the declared contextual mixer improves
compact linear lag accessibility relative to the trained independent model
under this fixed synthetic recipe. It does not establish consumer acceptance,
superiority over the first encoder or matched computational cost.

## Diagnostic results and limits

The contextual PCA36 ridge also improved over its own initialized mixer control
in each lag run (`pair-2`): effects were +46.09 pp (95% interval +38.28 to +53.91)
for seed 101, +30.47 (+20.31 to +39.84) for 202, and +17.19 (+9.38 to +25.78) for
303. This distinguishes the measured trained contextual result from performance
already present in the extra randomly initialized block; it does not isolate
every mechanism of the gain.

Matched global PCA12 ridge (`pair-4`) showed a smaller positive lag advantage:

| Seed | Mixer contextual global ridge | Independent global ridge | Difference (95% interval), pp |
| --- | ---: | ---: | ---: |
| 101 | 60.94% | 46.88% | +14.06 (+4.69 to +23.44) |
| 202 | 67.19% | 43.75% | +23.44 (+14.84 to +32.03) |
| 303 | 62.50% | 46.09% | +16.41 (+9.38 to +23.44) |

Reversal, level and amplitude reached 100% for both trained models on the
matched-concatenation ridge comparison. They remain accessibility diagnostics
at ceiling, rather than evidence of a mixer improvement. The legal observed-input
lag oracle achieved 100% accuracy and coverage in each seed, and native mask-only
ridge scored 50%. Support counts and unsupported compression remain separate
from conditional task scores.

Native96 tanh-16 lag results (`pair-5`) were uneven:

| Seed | Mixer contextual native96 tanh-16 | Independent native96 tanh-16 | Difference (95% interval), pp |
| --- | ---: | ---: | ---: |
| 101 | 71.88% | 60.94% | +10.94 (+1.56 to +20.31) |
| 202 | 69.53% | 83.59% | -14.06 (-21.88 to -6.25) |
| 303 | 93.75% | 90.63% | +3.13 (-0.78 to +7.81) |

The mixer is worse under this diagnostic in seed 202, and seed 303's interval
includes zero. The primary compact linear gain is therefore not a universal
improvement across surfaces, seeds or readout families. Nonlinear fits use the
fixed but different surface-name-derived initializations disclosed above; the
intervals condition on those particular fits.

The co-trained pre-mixer local diagnostic (`pair-7`) remained weak on lag.
Its PCA36 ridge accuracies were 45.31%, 54.69% and 55.47%; effects versus the
independent model were -7.03 pp (-14.06 to 0.00), +6.25 (-2.34 to +14.84) and
+3.13 (-3.13 to +9.38). None established a positive lower bound. That pattern
is consistent with the observed benefit being accessible through the contextual
surface rather than an established local diagnostic improvement. Shared weights,
readout training and extra model capacity prevent a complete causal decomposition.
It does not prove that local exports contain no lag information or that PCA
alone explains differences between tiers.

## Validation and retained assets

Model/support/bottleneck checks, shared feature tests, feature and reconstruction
adapter checks in both modes, independent/contextual workflow round trips,
legacy loading/resume/export, and CPU/CUDA mixer smoke passed inside the managed
container. The [validation record](../../../output/runs/rpb-implementation/channel-mixer-dc13a548aa/validation.json)
records their scope and artifact checks. Ordinary trainable loading reproduced
contextual exports within maximum absolute difference `1.19209e-7`; matching the
provider's frozen gradient flags reproduced them exactly. These checks establish
compatibility within their tested execution conditions, not downstream quality.

Each task/seed retains both providers' model/scaler assets, actual training counts
and seeds, split/support manifests, raw observations, features, PCA and probe
fits. See the [seed-101 lag mixer assets](../../../output/runs/rpb-implementation/channel-mixer-dc13a548aa/results/seed-101-lag_sign/provider-1/)
and [audit](../../../output/runs/rpb-implementation/channel-mixer-dc13a548aa/results/seed-101-lag_sign/provider-1/provider-audit.json).
Frozen-feature model archives are distinct from ordinary resumable checkpoints.
The plan/audits identify the full evaluation producer as
`45b298fe18183a0ba121cba8a97cc03c5c8155d137405dffcd2a2d0bb8d6031e`
and the RPB core/writer as
`dfdbe30014906bab141b8a1ab89d06f86ff82847c15834affd312adc4108292a`.
The immutable recipe snapshot remains unchanged.

Retain the optional contextual mode and this scoped development evidence while
keeping the independent model available. Confirmation, across-training-run
uncertainty, multiplicity-adjusted decisions, resource measurement, real-data,
forecasting and structured-outage evaluation remain outside this experiment.
There is no consumer acceptance or cost-equivalence claim. Subsequent experiments
need their own frozen hypotheses; the primary cannot be retrospectively changed.
Generated artifacts remain local under ignored `output/` and must be reproduced
in a clean checkout.
