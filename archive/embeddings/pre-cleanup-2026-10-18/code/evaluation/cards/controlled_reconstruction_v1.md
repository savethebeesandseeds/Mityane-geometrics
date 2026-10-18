# Controlled reconstruction: protocol v1

Date: 2026-10-05. Policy: `EMBEDDING_EVALUATION_POLICY.md` version 1.1.
Evidence stage: `development`. Implemented and measured on 2026-10-05; results
and validation are recorded in [RPB decoder reliance](../../encoders/raw_patch_bottleneck_mae/DECODER_RELIANCE.md).
This protocol does not issue acceptance
decisions and is separate from frozen-feature controlled-pairs protocols.

The instantiated reconstruction card fixes geometry, metadata, source budgets,
tasks, seeds, uncertainty budget and the complete resolved adapter recipe before
generation or fitting. An explicit encoder configuration must agree with card
geometry; it does not choose the evaluation population.

| Item | Frozen recipe |
| --- | --- |
| Population | Independent controlled source trajectories; both transformed variants stay in one split |
| Defaults | CPU float64 BCHF, C=3/H=32/F=3, semantic IDs 0/1/2, unitless coordinates, uniform interval one and endpoint H-1 |
| Tasks | Reversal/order, level, amplitude and signed cross-channel lag; all selected before scores |
| Splits | 32 training, 16 validation and 64 testing source pairs per task/run |
| Observation support | Fully observed histories; broader natural missingness requires another declared track |
| Held-out targets | One complete original patch per channel; trial r targets patch (r+c)%K in channel c; enumerate all K positions, P=8 by default |
| Primary model | Fresh RPB model and training-fit frozen scaler; 128 completed updates, replacement minibatches of eight by default |
| Mask/metadata control | Same initial frozen encoder fed normalized zeros plus actual visible masks, semantic IDs and patch positions; separately train the identical decoder head only |
| Control budget | Same completed-update budget as primary by default; an explicit override is persisted as a recipe deviation |
| Training protocol | Adapter-owned configuration, optimizer, counter-seeded sampling/masks and attempt limit; no labels, hidden clean signals or held-out observations in fitting |
| Primary metric | Training-standardized masked MAE, with equal eligible-example/channel weighting |
| Secondary metric | Training-standardized masked Huber, delta one, with the same support and reduction |
| Interventions | Normal compact exports, support-compatible shuffled exports, separately zeroed exports, mask/metadata-only predictions, matched untrained model and training-mean reference |
| Shuffle | Disjoint swaps of complete independent source groups; preserve receiver targets/metadata and record donor mapping |
| Uncertainty | Percentile bootstrap over complete two-source exchange blocks, retaining paired variants and repeated masks; 1,000 replicates by default |
| Randomness | Separate source, ordering, adapter initialization/training, swap and bootstrap streams; development seeds 101/202/303 |
| Decision | Report measurements and paired degradation; no consumer promotion or confirmation claim |

The shared engine receives predictions in raw units, then applies its saved
training-only per-channel/feature population-standard-deviation scaling with
floor 1e-8 to predictions and targets. It scores only
the fixed originally observed, intentionally hidden coordinates. It never drops
nonfinite target/prediction cells to make a result appear finite.

Target cells are averaged within each eligible channel, eligible channels within
each example, and eligible examples within the measured population. Repeated
patch trials remain attached to their source/exchange block; they are not extra
independent samples. Reports retain eligible counts and the actual uncertainty
unit. An insufficient population is unsupported rather than a zero error.

Shuffling replaces the compact vector used by the frozen trained decoder while
retaining receiver target positions, channel identities, support and targets.
The donor must be a different source trajectory, not the other transformed
variant of the same source. Complete source groups are paired in disjoint swaps,
so uncertainty resamples the resulting two-source exchange blocks rather than
treating dependent donors and recipients as independent.

Zero vectors are reported separately because they can be outside the learned
embedding distribution. The mask/metadata control is independently trained;
it is not the trained decoder evaluated at zero z. Its frozen encoder processes
no observed signal values, but can retain visibility, identity and position
information. The decoder architecture and legal metadata match the primary.

Fitted scalers/control assets, configurations, initialized/trained model
identities, seed policies, split/mask/swap manifests and per-example measurements
are saved under a unique output directory. Validation and test observations may
affect their measurements but cannot change fitted training assets. No recipe
selection uses test results.

Use the separate `embedding_evaluate reconstruct` command described in
[RPB decoder reliance](../../encoders/raw_patch_bottleneck_mae/DECODER_RELIANCE.md).
The minimum `feature_harness` executable rejects reconstruction because it does
not register RPB. Encoder-specific fitting/decoding stays in the RPB adapter;
population generation, metric scaling, swaps, uncertainty and reports stay in
the shared reconstruction engine.

## Completed development track

The [twelve-run report](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/report.json)
and [instantiated card](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/reconstruction-card.json)
use the frozen defaults above: three seeds, four tasks and 128 main/control
updates. Every testing mask/metadata-minus-intact and shuffled-minus-intact
comparison has a positive lower 95% interval bound for both standardized MAE
and Huber, with full example/channel/target coverage. Task means, per-run paired
intervals, assets and repeat/resume validation are recorded in
[RPB decoder reliance](../../encoders/raw_patch_bottleneck_mae/DECODER_RELIANCE.md).

Each testing run contains 64 source groups with 128 paired-variant rows and four
repeated target trials. Bootstrap resampling uses 32 disjoint two-source exchange
blocks; neither the rows nor repeated trials are independent uncertainty units.
These are within-run intervals, not uncertainty across training seeds.

Held-out reconstruction reliance does not establish useful frozen-feature
classification, lag recovery, forecasting, robustness under outages or consumer
acceptance. Those require their own declared comparisons and evidence.
