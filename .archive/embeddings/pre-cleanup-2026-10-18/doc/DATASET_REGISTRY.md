# Dataset registry

Registered 9 October 2026. These codenames identify signal-generation recipes,
separately from model tags, trained instances, random seeds and split budgets.
Future result tables carry a short adjacent legend with the codename and numeric
complexity. Historical frozen reports/cards/source captures stay unchanged.

| Codename | Task | Designed complexity | Short recipe | Status |
| --- | --- | ---: | --- | --- |
| TEMPO-1 | Timing lead/lag sign | 1/5 | Period16, absolute delay2, shared phase | Original recipe implemented and measured |
| TEMPO-2 | Timing lead/lag sign | 2/5 | PeriodU[12,20), absolute delayU[.5,1.5) | Implemented; data-only fixtures pass; encoder quality unmeasured |
| TEMPO-3 | Timing lead/lag sign | 4/5 | PeriodU[10,24), delayU[.25,1), gains/offsets and 3-tick channel gaps | Implemented; cross-feature v2 information gates pass; fresh fixed512 comparison measured and audited |
| AMP-1 | Amplitude contrast | 1/5 | Period16 sine scaled0.5×/2× | Original separate task implemented and measured |
| TEMPO-4 | Slow-component timing lead/lag | 5/5 | Two coherent rhythms; slow lag sign, unrelated fast lag, gains/offsets and gaps | Implemented; information admission and separate two-cohort CUDA encoder screen completed |
| AMP-2 | Relative component balance | 5/5 | Swap slow/fast strength at equal component energy; gains/offsets and gaps | Implemented; information admission and separate two-cohort CUDA encoder screen completed |
| TEMPO-5 | Slow-component timing lead/lag | 5/5 | C3/H128/F9; slow/fast feature groups, structured distractors, gains/offsets and 5-tick gaps | Information admission and three-cohort/nine-trajectory CUDA capacity experiment completed |
| MKT-HIST-1 | Four past-history regressions | 4/5 | BTC/ETH/SOL January2024 minute candles, nine raw features, H128, chronological TRAIN/VALIDATION | Bounded pilot in sibling integration; [card](../../cuwacunu_dev/doc/MARKET_HISTORY_PILOT_CARD.md), [results](../../cuwacunu_dev/doc/MARKET_HISTORY_PILOT_RESULTS.md); no forecasting |
| MKT-HIST-2 | Four recent128 history regressions | 4/5 | Same January sources; common H256-legal endpoints, H128/H256 inputs, one full-H256 TRAIN scaler;112/32 panel | Matched2048-update dimension study complete in integration; [card](../../cuwacunu_dev/doc/MARKET_CAPACITY_STUDY_CARD.md), [results](../../cuwacunu_dev/doc/MARKET_TRAINING_STUDY_RESULTS.md); no forecasting |

Complexity is an **ordinal designed challenge level on1–5**, assigned before
measurement. It is not measured entropy, information content, accuracy,
observability, coverage or a mathematical difficulty ratio. Level4 does not mean
twice the challenge of level2. Different tasks with the same level are not
proved equally difficult. Level3/5 remains unassigned. Keep achieved accuracy
and legal-observation information checks as separate evidence.

The new [two-component recipe registry](two_component_dataset_registry.json)
extends the original frozen [machine registry](dataset_registry.json) without
changing its entries or bytes. TEMPO-4 and AMP-2 each received level5/5 before
engineering outcomes. The [information admission](TWO_COMPONENT_INFORMATION_V1.md)
passes all task/cohort/view gates on two engineering masters, with100% accuracy
and coverage. These are legal-observation information scores, not encoder or
fixed-head accuracy. New encoder experiments use a separate prospective card
and fresh source cohorts; they cannot reuse TEMPO-3 scores as matched controls.
The [completed multiband screen](MULTIBAND_SCREEN_V1.md) compares new
RPB-v18.alt-01 and RPB-v19 instances on the same four quality cohorts. Its
joint prospective continuation gate passes; this remains development evidence.

TEMPO-5 is separately specified in the
[configurable capacity registry](configurable_capacity_dataset_registry.json)
and [prospective card](../code/evaluation/cards/configurable_capacity_v1.md).
All nine quality role/view archives pass the observed-only information gate:
3,072 decisions,100% correct and supported. The known signal-family rule is
a solvability witness, not an encoder score. The
[completed capacity comparison](CONFIGURABLE_CAPACITY_V1.md) tests W64/D32,
W128/D32 and W64/D64 on the same three fresh cohorts. H256 has engineering
information admission only; no H256 encoder quality is measured. Longer history
can supply more cycles and aid observability, so larger tensor size alone does
not establish a harder learning task.

The original codenames below apply to closed C3/H32/F3 recipes; TEMPO-5 instead
has its own declared C3/H128/F9 geometry. Altering the signal
law, shape, noise or natural observation mechanism requires an explicit new
recipe identity before generation. New seeds or split counts identify new
cohorts of the same recipe. An additional scored deletion view is recorded
separately and does not silently alter the dataset's complexity label.

## Exact existing recipes

TEMPO-1 uses the shared controlled `lag_sign` generator: period16, absolute
delay2 on channel1 relative to channel0, source-shared phase, and unrelated
channel2 phase. Each source has opposite delay-sign variants. Features have
multipliers1/1.1/1.2. Gaussian nuisance noise has SD.005 and natural coordinate
missingness is.10. The two variants share nuisance noise and masks; paired row
order is randomized. Relevant source is
[feature_harness.cpp](../code/shared/src/feature_harness.cpp). Do not change that
historical generator to increase difficulty.

TEMPO-2 uses `variable-delay-timing-v1`: each source draws period uniformly on
[12,20) and positive delay on[.5,1.5). Opposite signs form the two variants;
phase, noise and masks are source-shared. Feature multipliers, noise SD.005,
natural missingness.10 and unrelated channel2 remain as above. The closed
[generator](../code/evaluation/benchmarks/variable_delay_timing/variable_delay_timing.h)
and [plan](HARDER_TIMING_BENCHMARK_PLAN.md) separate legal-observation analytic
information checks from encoder scores. Data-only fixtures are completed; no
TEMPO-2 encoder quality has been measured.

AMP-1 is the shared controlled `amplitude` task: period16 sine observations with
paired multipliers0.5 and2.0, source-shared phase/noise/masks, feature multipliers
1/1.1/1.2, noise SD.005 and natural missingness.10. It is a separate task, not a
timing level. Historical transfer fits amplitude heads on its own TRAIN data
while retaining an encoder/scaler trained on TEMPO-1. State both identities.

TEMPO-3 uses the separately specified `structured-hard-timing-v1` recipe.
For each source, period is uniform on[10,24) and positive delay on[.25,1).
Channels0/1 retain the paired opposite lead/lag relation. Positive channel and
feature gains vary independently, with offsets constant across time. The
unrelated third channel has its own period on[6,12). Noise SD stays.005;
natural missingness stays.10 and each channel also has one3-tick missing gap.
Gains, offsets, noise and masks are identical within each source pair. Each
channel/feature independently draws a base gain from[.5,1.5) on channels0/1
or[1,2) on channel2, multiplied by1+.1*f, and a constant offset from[-.75,.75).
Gap starts are0..29, with all three features absent for those three ticks.
The [generator](../code/evaluation/benchmarks/structured_hard_timing/structured_hard_timing.cpp)
and [frozen v2 card](../code/evaluation/cards/structured_hard_timing_comparison_v2.md)
define the exact draw order. Its level4/5 is a designed challenge, not a
measured outcome. The first same-feature analytic information admission
[failed deleted coverage](STRUCTURED_HARD_TIMING_INFORMATION_V1_FAILED.md);
no encoder or head was fitted. A separately specified cross-feature analytic
rule retains the exact data recipe and gates. Its fresh 82585/83686 information
admission [passed both views](STRUCTURED_HARD_TIMING_INFORMATION_DIAGNOSTIC.md):
deleted coverage was 1,020/1,024 and 1,024/1,024, with every supported row
correct. The subsequent [fresh encoder comparison](../code/encoders/raw_patch_bottleneck_mae/STRUCTURED_HARD_TIMING_DIAGNOSTIC.md)
completed once and passed its independent audit. Its two standard quality
panels retain all seven methods, five cohorts and three fixed head repetitions.
The analytic rule also solves all TRAIN/intact/deleted rows in the actual
quality cohorts, with100% coverage. This rule has no fitted heads and is
reported separately from fixed-head classification.

The completed single-task comparison is
`structured-hard-timing-comparison-v2`, with new masters
75272/76373/77474/78575/79676 and separate measured groups
RPB-v7.alt-05/RPB-v10.alt-05. It keeps512/B8/native32 and fixed heads; these
budgets do not define the dataset complexity. Separate data-only engineering
masters 80383/81484 and then 82585/83686 check usable information using legal
observed values, without head fitting. The two information attempts use separate
fresh seeds; their change is an analytic support rule, not a generator or gate
change. The [durable JSON](results/structured_hard_timing_comparison_v2.json)
retains both histories. The machine recipe JSON remains the frozen specification;
measured status is recorded here and in result files. Neither this registry nor
the completed comparison authorizes another quality run.

## Compact legends for new reports

Place these immediately beside each applicable result table:

- `Dataset: TEMPO-1 · timing · complexity1/5 · P16, |delay|2; intact VALIDATION.`
- `Dataset: TEMPO-2 · timing · complexity2/5 · P12–20, |delay|.5–1.5; intact VALIDATION.`
- `Dataset: TEMPO-3 · timing · complexity4/5 · variable delay, gains/offsets, 3-tick gaps; intact VALIDATION.`
- `Dataset: AMP-1 · amplitude · complexity1/5 · 0.5×/2× sine; extra30% deletion VALIDATION.`
- `Dataset: TEMPO-4 · slow-component timing · complexity5/5 · two rhythms, gains/offsets and gaps; intact VALIDATION.`
- `Dataset: AMP-2 · component balance · complexity5/5 · relative slow/fast strength; extra30% deletion VALIDATION.`
- `Dataset: TEMPO-5 · slow lead/lag, nine grouped features and structured distractors · complexity5/5 · intact VALIDATION.`
- `Fit: TEMPO-1(1/5); score: AMP-1(1/5); timing encoder frozen.`

Always state split/source counts and classifier/update budgets separately.
For a single dataset retain the standard five table columns. When rows mix
several timing datasets, add Dataset and list the individual numeric levels
beside the table; do not average task/dataset scores or difficulty levels.

## Reusable pipeline boundary

Generate each source-paired legal observation cohort once with its own dataset
protocol/seed namespace. Save observations, masks, scoring-only labels and
ordered source IDs together with fixed recipe metadata and source/artifact
hashes. Split independent sources before paired variants; use identical saved
rows/views for both encoder designs. If hidden generator truth is retained,
keep it in a distinct audit-only asset, never encoder/scaler/PCA/head inputs.

Reuse the existing controlled-dataset contract, source-paired deletion law,
TRAIN-only preprocessing and fixed-feature readouts. Fit each method's head
once and reuse it across declared views. A registry codename requires no new
encoder dependency, objective, head or adaptive data path. Persist metadata
fields `dataset_id`, `recipe_id`, `task`, `designed_complexity_level`,
`complexity_scale_max:5`, and separate fitting/scoring dataset identity. These
are reporting/provenance requirements; this documentation adds no data runtime.

The [JSON registry](dataset_registry.json) is the machine-readable mapping.
Follow the [reporting standard](RESULTS_REPORTING_STANDARD.md). A new recipe/card
and its independent source/fixture/CUDA admission gates must precede measurement;
the registry itself authorizes no quality generation or TEST access.
