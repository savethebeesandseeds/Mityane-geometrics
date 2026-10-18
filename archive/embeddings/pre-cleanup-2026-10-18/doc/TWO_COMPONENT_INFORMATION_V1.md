# Two-component information admission

Data-only engineering, not encoder or classifier accuracy. Both new datasets
use two coherent sinusoidal components, positive gains/offsets, low noise,
natural missingness and three-tick channel gaps. The labels describe a
physical component property; they are not derived from a summed margin.

Two engineering masters,910901/910902, per task. Each has TRAIN256/128
source pairs and VALIDATION128/64 pairs; an extra30% pair-shared deletion
view preserves the validation rows. No quality seeds or TEST data generated.

The fixed observed-only information rule estimates two frequencies and
affine sinusoidal coefficients from legal cells. It receives no labels,
source IDs or hidden frequencies. This fitting is not an encoder feature,
head or training target. The supported accuracy and coverage below judge
whether the declared challenge retains usable task information.

Dataset: **TEMPO-4** · timing · slow-component lead/lag in a two-rhythm mixture · **complexity5/5** · validation-intact.

| Method | Information accuracy % | Coverage % | Supported / total |
| --- | ---: | ---: | ---: |
| Observed-only information check | 100.00 | 100.00 | 256 / 256 |

Dataset: **TEMPO-4** · timing · slow-component lead/lag in a two-rhythm mixture · **complexity5/5** · validation-deleted.

| Method | Information accuracy % | Coverage % | Supported / total |
| --- | ---: | ---: | ---: |
| Observed-only information check | 100.00 | 100.00 | 256 / 256 |

Dataset: **AMP-2** · component balance · relative slow/fast strength in a two-rhythm mixture · **complexity5/5** · validation-intact.

| Method | Information accuracy % | Coverage % | Supported / total |
| --- | ---: | ---: | ---: |
| Observed-only information check | 100.00 | 100.00 | 256 / 256 |

Dataset: **AMP-2** · component balance · relative slow/fast strength in a two-rhythm mixture · **complexity5/5** · validation-deleted.

| Method | Information accuracy % | Coverage % | Supported / total |
| --- | ---: | ---: | ---: |
| Observed-only information check | 100.00 | 100.00 | 256 / 256 |

All12 task/cohort/view gates passed.
TRAIN/intact gates require>=99% supported accuracy and>=99% coverage;
deleted gates require>=98% accuracy and>=95% coverage on every cohort/task.
No rows or failures are removed from the total scoring population.

Generator invariants passed116,963 checks and40 malformed-input cases.
The information primitive passed54 timing and54 component-balance affine
noiseless fixtures, legal-mask and abstention cases. Actual full noisy
engineering uses four generator calls and2,048 information rows.
Saved CPU verification passed262,400 checks/24 archives;
it replayed selected-fit normal equations/residuals/phase or ratio margins
and exact deletion streams, without rerunning frequency search or generation.

Evidence is in `output/runs/two-component-information-v1/admission-V339vw`.
The [durable summary](results/two_component_information_v1.json) retains
the card/source/SDK pins, per-cohort counts, costs and artifact inventory.
The [prospective card](../code/evaluation/cards/two_component_information_v1.md)
and [protocol](../code/evaluation/protocols/two_component_information_v1/README.md)
retain the exact pre-outcome rules. Compact source/results are versioned;
the full CPU archives remain in the local ignored capsule.

This supports a separate small CUDA architecture screen. It does not
establish encoder quality on the new datasets, universal observability
outside this family, or production promotion. Keep TEMPO-3 and RPB-v18
milestones intact; compare new dataset instances under unchanged fixed heads.
