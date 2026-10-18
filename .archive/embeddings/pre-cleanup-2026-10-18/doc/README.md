# Paused research and migration handoff — 18 October 2026

Start with [the migration handoff](HANDOFF_2026-10-18_V14_PAUSED_MIGRATION.md)
and [current status](CURRENT_STATUS.md). The latest measured candidate is
CPB-v14.market-01 (full trust, 3/22 protected conditions); CPB-v3.market-01 remains
the clean baseline. Research is paused. The handoff identifies the saved results,
ignored artifacts to copy, unfinished metadata work and steps requiring a resume.
The dated index below is retained as historical context.

---

# Documentation index

Start with [CURRENT_STATUS.md](CURRENT_STATUS.md), reviewed on 13 October 2026.
RPB-v20 is implemented as a configurable spectral API with recorded engineering
checks, the completed synthetic capacity experiment and saved-coordinate diagnostic.
RPB-v4 remains the formal evaluated reference and original RPB-v7 is preserved.
The completed RPB-v22 content-only relation candidate passes its two-cohort
screen and third development check. The preceding RPB-v21 remains stopped.
Selected v20 settings and every historical checkpoint remain preserved.

## Using the repository

| Document | Use |
| --- | --- |
| [Current status](CURRENT_STATUS.md) | Implemented contracts, measured evidence, repository boundaries and open work |
| [Root guide](../README.md) | Build entry points, retained CLI and source layout |
| [Environment](ENVIRONMENT.md) | Existing managed container, named task sessions and artifact locations |
| [RPB guide](../code/encoders/raw_patch_bottleneck_mae/README.md) | RPB family, historical CLI and adapters |
| [RPB-v22 explained](RPB_V22_ENCODER_GUIDE.md) | Plain input/output ranks, added metadata, hybrid architecture, training targets and separate evaluation |
| [Content-only relation API](../code/encoders/raw_patch_bottleneck_mae/CONTENT_ORDERED_RELATION.md) | RPB-v22 formulas, configurable native export, settings and typed inference checkpoints |
| [Configurable spectral API](../code/encoders/raw_patch_bottleneck_mae/CONFIGURABLE_SPECTRAL.md) | RPB-v20 C/H/F/D, masks, scaler, file binding and immutable inference checkpoints |
| [Core settings](../code/shared/config/README.md) | Reusable file/settings syntax and parser provenance |
| [MTF-v1 guide](../code/encoders/mtf_jepa_mae_vicreg/README.md) | Retained multiscale JEPA/MAE/VICReg encoder |
| [Shared components](../code/shared/README.md) | Architecture-independent utilities and engine ownership |
| [Evaluation integration](../code/evaluation/README.md) | Adapter registration, historical drivers and protocol-specific entry points |

## Versions, protocols and completed evidence

The latest [content-only relation report](CONFIGURABLE_CONTENT_RELATION_V1.md),
[instance registry](configurable_content_relation_embedding_versions.json),
[saved result](results/configurable_content_relation_v1.json) and
[continuation](CONTINUATION_2026-10-13_AFTER_CONTENT_RELATION.md) retain the three
RPB-v22.alt-01 development cohorts, exact gates and independent audits.
Fresh synthetic confirmation is proposed and has not been allocated.

Use [EMBEDDING_VERSIONS.md](EMBEDDING_VERSIONS.md) and its
[machine-readable registry](embedding_versions.json) for design tags; the
[capacity registry extension](configurable_capacity_embedding_versions.json)
identifies the new v20 instance bundles.
[DATASET_REGISTRY.md](DATASET_REGISTRY.md) defines dataset codenames and designed
complexity. New reporting follows [RESULTS_REPORTING_STANDARD.md](RESULTS_REPORTING_STANDARD.md);
evaluation follows [EMBEDDING_EVALUATION_POLICY.md](EMBEDDING_EVALUATION_POLICY.md).

The preceding stopped candidate is
[CONFIGURABLE_LEARNED_RELATION_V1.md](CONFIGURABLE_LEARNED_RELATION_V1.md),
with its [saved result](results/configurable_learned_relation_v1.json),
[instance registry](configurable_learned_relation_embedding_versions.json) and
[13 October continuation](CONTINUATION_2026-10-13_AFTER_LEARNED_RELATION.md).
Its learned timing improves, but its two-cohort deletion gate fails. The preceding
[LEARNED_TRAIN_V1.md](LEARNED_TRAIN_V1.md) diagnoses saved TRAIN variation without
model execution or head fitting.

The preceding completed capacity report is
[CONFIGURABLE_CAPACITY_V1.md](CONFIGURABLE_CAPACITY_V1.md), with its
[saved result](results/configurable_capacity_v1.json) and
[12 October continuation](CONTINUATION_2026-10-12_AFTER_CONFIGURABLE_CAPACITY.md).
It compares compact, wider-backbone and larger-output RPB-v20 instances on
TEMPO-5 at C3/H128/F9/P8, complexity 5/5 and a fixed 512-update budget.
H256 encoder quality and market quality remain unmeasured.

[CONFIGURABLE_BRANCH_V1.md](CONFIGURABLE_BRANCH_V1.md) and its
[saved record](results/configurable_branch_v1.json) separate the usefulness of
fixed12 and learned20 while reusing complete32 results. The encoder remains unchanged; saved source-parity checks cover all15 reusable
file pairs.
Use the [historical coordinate continuation](CONTINUATION_2026-10-12_AFTER_CONFIGURABLE_BRANCH.md)
for the focused learned-path investigation; retain the capacity continuation as
the preceding milestone.

The preserved [SPECTRAL_CONFIRMATION_V1.md](SPECTRAL_CONFIRMATION_V1.md) and
[saved result](results/spectral_confirmation_v1.json) record RPB-v19.alt-01. The preceding
[MULTIBAND_SCREEN_V1.md](MULTIBAND_SCREEN_V1.md) and
[TWO_COMPONENT_INFORMATION_V1.md](TWO_COMPONENT_INFORMATION_V1.md) retain the
screen and information admission. Synthetic confirmation does not establish
arbitrary-geometry or market quality.

[EVALUATION_IMPLEMENTATION_REVIEW.md](EVALUATION_IMPLEMENTATION_REVIEW.md) and
[RPB implementation history](../code/encoders/raw_patch_bottleneck_mae/IMPLEMENTATION_STATUS.md)
preserve earlier engineering stages; their dated contents are not the latest
repository status. Dated `CONTINUATION_*`, `MILESTONE_*`, `RESEARCH_SESSION_*`,
experiment reports, frozen cards and saved JSON receipts retain their original
meaning. Use current status for the next direction; do not run an old recipe
merely because its historical note says "next".

## Preserved MTF-v1 outreach brief

`representation-encoder.pdf` is the one-page MTF-JEPA-MAE-VICReg research brief.
Editable sources are `representation-encoder.tex`, `ieee-preamble.tex` and
`references.bib`. It describes the retained sensor-history encoder; it is not
an RPB-v19/v20 overview or a market/robotics performance report.

The unmodified IEEEtran conference class supplies US Letter, two columns,
numbered sections and IEEE references. This is a research brief, not a claim
of IEEE publication or endorsement. The bibliography cites original JEPA,
masked-autoencoder and VICReg papers as methodological background; their image
benchmarks are not measurements for this encoder.
See [vendor provenance](vendor/PROVENANCE.md).

From this `doc/` directory, use the configured paper builder with its existing
LaTeX container running:

```powershell
.\build.ps1 -Render
```

The builder requires the existing `documents-latex` container with
latexmk/pdfLaTeX, Poppler and qpdf. It snapshots the paper sources into container
`/tmp`, copies the PDF and previews back, and requires exactly one page.
Paper packages are separate from the managed embedding environment.
The outreach source/PDF is preserved during this documentation refresh.
