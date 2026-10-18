# RPB-MAE frozen-feature advance

Date: 2026-10-05. Evidence stage: `development`. Status: Completed under the
frozen recipe. The primary comparison did not establish consistent compact
linear lag benefit; a separate native-width nonlinear diagnostic showed benefit.

This experiment asks whether a longer, fixed unsupervised training budget makes
the existing compact exports more useful to frozen probes than their matched
initialization. It follows the [held-out decoder-reliance track](DECODER_RELIANCE.md),
which established useful reconstruction information within its synthetic
population. Reconstruction quality does not answer this frozen-feature question.
The [shared policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md) and
[controlled-pairs-v2 protocol](../../evaluation/cards/controlled_pairs_v2.md)
govern the comparison.

The earlier [four-update v2 report](../../../output/runs/rpb-implementation/rpb-evaluation-9nX9MV/results/report.json)
showed reversal and level at ceiling for trained and untrained RPB exports,
amplitude mostly near ceiling, and weak or variable lag-sign readouts. The focal
diagnostic is therefore lag sign on the complete channel concatenation at matched
width 36. Ridge remains the primary probe family. Other surfaces and tasks are
reported separately; their scores are not averaged into a single encoder score.
No acceptance threshold is assigned by this experiment.

## Frozen recipe

| Item | Declared choice |
| --- | --- |
| Encoder | Existing independent-channel RPB-MAE; no mixer, new loss, readout change or export normalization |
| Geometry | CPU float64 raw `[B,3,32,3]`; semantic channel IDs `0,1,2`; feature units `unitless,unitless,unitless`; uniform interval 1, endpoint 31 |
| Architecture | P8, encoder width 64, three pre-normalized temporal blocks, four heads, feed-forward width 256, local export D32, decoder hidden width 128 |
| Numeric preparation | Training-fit frozen per-ID/feature float64 scaler, scale floor `1e-6`; center before float32 neural computation |
| Model training | Fresh initialization per task/seed; 128 completed masked-Huber updates; replacement minibatches of eight; AdamW learning rate `0.001`, weight decay `0.0001`, gradient clipping 1 |
| Mask/loss | Existing whole-patch mask ratio `0.25`, Huber delta 1 and equal eligible-example/channel reduction |
| Attempt limit | 10,000; skip entirely ineligible batches without an optimizer update; report attempted and completed counts |
| Configuration | Explicit [evaluation.conf](config/evaluation.conf), with CLI `--rpb-steps 128` overriding its stored update default; actual per-run training seed recorded separately |
| Tasks | Reversal/order, level, amplitude and signed cross-channel lag, fixed before scores |
| Population | `paired-controls-v1` generator: independent sources, measurement noise standard deviation `0.005`, class-independent coordinate missingness rate `0.1`, same mask within each transformed pair |
| Splits | 32 training, 16 validation and 64 testing source pairs per task/seed: 64/32/128 variant rows |
| Seeds | Development seeds 101, 202 and 303; separate generator, split, mask, ordering, initialization, sampling and probe streams |
| Inference | Frozen scaler and encoder, evaluation mode, no parameter updates; one Torch CPU thread |
| Selection | Fixed final 128-update checkpoint; no validation selection, nonlinear restarts, hyperparameter search or test-driven stopping |

The missingness setting differs from the fully observed reconstruction track.
Increasing testing pairs also changes the generator's total source pool and split
assignment relative to the earlier 32-test-pair v2 report. This is a new
development comparison against each run's own untrained control, not an isolated
estimate of the effect of increasing four updates to 128 on unchanged data.

Fit callbacks receive only permitted training observations and schema metadata.
They receive no labels, hidden clean values or validation/test observations. The
trained and untrained models start from identical weights and share the same
training-fit preprocessing. Only the trained model receives optimizer updates.
Training on each task's source population is disclosed; this is not transfer
evaluation of one universal checkpoint across all four tasks.

## Surfaces and comparisons

| Fixed comparison, in declared order | Candidate | Comparator | Probe width |
| --- | --- | --- | ---: |
| Channel concatenation, matched | `rpb_trained_channel_concatenation` | `rpb_untrained_channel_concatenation` | 36 |
| Global, native | `rpb_trained_global` | `rpb_untrained_global` | 32 |
| Global, matched | `rpb_trained_global` | `rpb_untrained_global` | 12 |
| Channel concatenation, native | `rpb_trained_channel_concatenation` | `rpb_untrained_channel_concatenation` | 96 |

The first comparison's lag-sign ridge result is the focal primary comparison.
All remaining pair tiers, the other three tasks and tanh-16 are diagnostic or
secondary evidence. These roles remain fixed after scores are produced.

The global surface is the equal mean of valid independent local channel exports;
it requires at least one observed-valid channel. Concatenation preserves semantic
order `0,1,2` and requires every constituent channel to be observed-valid. This is
one joint probe, not separate per-channel accuracy or a contextual-channel export.

Feature normalization and centered PCA fit only valid training rows. Every
matched tier applies PCA separately to each surface; width 36 is joint
concatenation PCA, not per-channel PCA or a narrower trained model. The 64
training rows imply a centered rank bound of at most 63, with an additional
numerical-rank requirement. Unsupported fits or compression remain reported;
the protocol does not lower widths, pad missing components or remove a failed
run after seeing scores.

Each declared pair uses its own fixed candidate-and-comparator validity
intersection. Report both individual coverage and the common population's row,
class and source-group counts. Adding raw or mask controls does not change an
existing pair's denominator. Coverage is measured rather than assumed from this
light-missingness recipe.

## Controls, probes and uncertainty

The shared engine also reports the legal observed-input oracle, raw histories
plus visibility, mask/metadata-only features, and shuffled-training-label ridge
controls. Raw input contains 288 signal coordinates plus 288 mask coordinates;
the mask control has 288 coordinates. They receive native and applicable matched
tiers, with the same PCA support checks. The oracle receives legal observations
and support, never hidden clean values, generator state or labels. Its coverage
and solvability result must be checked before interpreting an encoder failure.

Baseline embeddings and baseline descriptors are outside this run's selected
provider set. Consequently, a trained-versus-untrained advantage would establish
training benefit in this recipe, not superiority over the baseline or the
strongest same-budget descriptor. Near-ceiling tasks may confirm accessibility
while providing little room to demonstrate an improvement.

Ridge uses penalty 1 and the fixed training-label budget. The separate interaction
tier is a tanh-16 probe trained for 100 full-batch Adam updates at learning rate
`0.01`, with one fixed initialization per surface/tier/run. It has no restarts or
selection using validation or test results. Surface-name-derived seeds differ
between trained and untrained probes and are persisted. Validation results are
reported without choosing a checkpoint or probe.

The report emits trained-minus-untrained accuracy effects and paired intervals
for both ridge and tanh-16 on each declared pair. It also reports each surface's
tanh-minus-ridge effect. It does not estimate a difference-of-differences between
the two training benefits. A nonlinear advantage is a separately scoped
accessibility result; it cannot rescue a failed primary linear claim. Conversely,
weak linear lag performance, especially when raw linear probes are also weak,
does not prove that local exports have lost all timing or lag information.

The existing shuffled-label control permutes labels among eligible training
rows, preserving overall class counts. It is a fixed shortcut diagnostic, not a
source-pair-preserving randomization test. Its one reported accuracy does not
provide a statistical comparison with chance by itself.

Uncertainty uses 1,000 percentile bootstrap replicates and nominal 95% intervals
over independent testing source groups, retaining both transformed variants
together. There are 64 testing source groups, not 128 independent observations.
Unlike source-swapping reconstruction, this probe track needs no two-source
exchange blocks. Intervals condition on the fitted encoder, compression and
probe instances; they do not include variability across training seeds or
nonlinear initializations. Keep the three seed results separate. The many
task/surface/tier/probe intervals are exploratory and not multiplicity-adjusted
confirmation decisions.

## Reproduction and retained evidence

Use the existing managed container and the full `embedding_evaluate` executable.
Allocate a new results directory and retain `experiment-plan.json` in its parent
before generation or fitting. The plan records the exact command, configuration
content, resolved training budget/seeds, source fingerprints and seed formulas;
`evaluation-card.json` fixes the shared population, probes, dimensions and pairs.
The generic card alone does not serialize the encoder's entire training recipe.

```bash
eval_bin=/opt/cuwacunu_embedding/build/rpb-implementation/embedding_evaluate
mkdir -p /embedding/output/runs/rpb-implementation
run_parent="$(mktemp -d /embedding/output/runs/rpb-implementation/frozen-probe-XXXXXX)"
# Save the complete experiment-plan.json here before invoking the evaluator.
"$eval_bin" evaluate --output "$run_parent/results" --encoders rpb \
  --rpb-config /embedding/code/encoders/raw_patch_bottleneck_mae/config/evaluation.conf \
  --rpb-steps 128 --seeds 101,202,303 \
  --tasks reversal,level,amplitude,lag_sign \
  --channels 3 --history 32 --features 3 --channel-ids 0,1,2 \
  --units unitless,unitless,unitless \
  --train-pairs 32 --validation-pairs 16 --test-pairs 64 \
  --matched-global-width 12 --matched-channel-width 36 --threads 1 \
  --card-id controlled-pairs-v2 \
  --compare rpb_trained_channel_concatenation,rpb_untrained_channel_concatenation,matched_channels \
  --compare rpb_trained_global,rpb_untrained_global,native \
  --compare rpb_trained_global,rpb_untrained_global,matched_global \
  --compare rpb_trained_channel_concatenation,rpb_untrained_channel_concatenation,native
```

Explicit comparisons are necessary: the RPB-only CLI default declares only the
matched global trained-versus-untrained pair. Retain the complete card, plan,
report, split/support checksums and manifests, fitted scaler/model assets,
pretraining counts, features, PCA and probe archives. Check every requested run,
including unsupported outcomes, instead of selecting only successful seeds.

## Completed measurements

The [completed report](../../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/results/report.json)
contains all 12 task/seed runs and 48 declared pair comparisons. The
[pre-generation experiment plan](../../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/experiment-plan.json),
[immutable recipe snapshot](../../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/frozen-recipe.md)
and [instantiated card](../../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/results/evaluation-card.json)
retain the choices made before scores. Every requested pair was measured with
100% testing common-population coverage: 128 rows, 64 complete source pairs and
64 rows per class. Provider audits record 128 attempted/completed updates for
each trained model and zero model updates for its untrained control.

All requested RPB compression tiers were supported. The 12 unsupported feature
entries are mask/metadata-only PCA at width 36, where numerical training rank
was insufficient. Their unsupported status remains in the report; no width was
adapted. Native mask/metadata ridge and tanh scores were 50% on each lag run. The
legal observed-input lag oracle achieved 100% accuracy and coverage for all
three seeds, while raw native ridge scores remained near chance. Thus the
fixture was solvable under its legal support, but its linear raw readout was
not a strong positive control for the interaction.
Raw PCA12 with the fixed nonlinear probe scored 76.56%, 88.28% and 73.44%
on lag for seeds 101/202/303, consistent with nonlinear accessibility.
Shuffled-label controls were not uniformly near chance: a single training-row
permutation can align or invert a strongly separable class direction. They remain
shortcut diagnostics, not chance-calibrated tests or a blanket pass/fail gate.

The preregistered primary is lag-sign ridge on joint PCA width 36 (`pair-1`).
Percentages below are testing conditional accuracy; differences and intervals
are percentage points of trained minus untrained accuracy. The report supplies
the unrounded values and paired source-group 95% intervals.

| Seed | Trained PCA36 ridge | Untrained PCA36 ridge | Difference (95% interval), pp |
| --- | ---: | ---: | ---: |
| 101 | 52.34% | 47.66% | +4.69 (-5.47 to +14.06) |
| 202 | 48.44% | 39.84% | +8.59 (-2.34 to +18.75) |
| 303 | 52.34% | 53.13% | -0.78 (-9.38 to +7.81) |

Every primary interval includes zero. This run therefore failed to establish
consistent compact linear lag benefit over the matched initialization. It does
not establish equivalence, a reliable improvement, or an acceptance outcome.
Reversal and level remained at ceiling for both models on this matched tier;
amplitude was at or near ceiling. Those diagnostic results cannot compensate
for the primary finding.

The fixed secondary tanh-16 probe on native 96-dimensional concatenation
(`pair-4`) produced the following diagnostic lag results on the same complete
testing populations:

| Seed | Trained native96 tanh-16 | Untrained native96 tanh-16 | Difference (95% interval), pp |
| --- | ---: | ---: | ---: |
| 101 | 60.94% | 48.44% | +12.50 (+1.56 to +23.44) |
| 202 | 83.59% | 46.09% | +37.50 (+26.56 to +47.66) |
| 303 | 90.63% | 59.38% | +31.25 (+21.88 to +40.63) |

All three native nonlinear paired intervals have positive lower bounds. The
arithmetic development-seed means are **78.3854% trained versus 51.3021%
untrained**, a descriptive difference of 27.0833 percentage points. These means
are not confidence intervals across training runs. The per-seed variability and
the single fixed, surface-specific probe initializations remain visible.

This diagnostic supports a native-width nonlinear accessibility benefit within
the declared synthetic recipe. It does not replace the primary PCA36 ridge
comparison. The contrast changes compression/dimensionality and probe family
together, so it cannot identify PCA alone as the cause of the weaker primary
result. Native trained ridge also remained weak: 47.66%, 49.22% and 48.44% for
seeds 101, 202 and 303. The experiment neither proves that independent local
exports lack lag information nor isolates which architecture or readout change
would improve compact linear access.

## Decision and retained assets

Retain the existing encoder and the complete development evidence without a
quality promotion. The primary improvement remains unestablished; the nonlinear
result is a separately scoped diagnostic. Any subsequent compression, readout or mixer
experiment needs a newly declared recipe rather than changing this run's
comparison after scores.

Each task/seed directory retains training/validation/testing observations,
source/support manifests, features, scaler/model assets and fitted probes/PCA.
For example, the [seed-101 lag assets](../../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/results/seed-101-lag_sign/provider-0/)
and [provider audit](../../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/results/seed-101-lag_sign/provider-0/provider-audit.json)
record training counts, actual seeds and preprocessing identities. Model assets
are frozen-feature model archives, not resumable ordinary-workflow checkpoints.
The [source fingerprint record](../../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/source-fingerprints.txt)
distinguishes the evaluation producer from the core encoder/writer identity.
The pre-score recipe snapshot is unchanged.
The [validation manifest](../../../output/runs/rpb-implementation/frozen-probe-8b6e501bee/validation.json)
records the exact seed202 lag repeat, unchanged four-update regression and
80 preserved preexisting code/configuration files. The adapter's dedicated test
validates saved model/scaler/seed provenance and held-out extraction isolation.

This track adds no forecasting, structured-outage, real-data, latency/memory,
confirmation or consumer-acceptance evidence. Its intervals remain conditional
within-run, exploratory and unadjusted for multiplicity, with no uncertainty
across training seeds. The earlier decoder measurements retain their own
population and uncertainty interpretation. Generated artifacts are retained
locally under ignored `output/`; a clean checkout must reproduce them.
