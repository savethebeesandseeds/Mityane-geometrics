# Representation providers and model evaluation

This directory owns the imported MTF, RPB and causal patch bottleneck encoders,
their model-specific training/evaluation, provider integration definitions, and
the separately versioned State Representation Network. Fusion families live in
[`../fusion/`](../fusion/README.md). The predecessor is
`cuwacunu_embedding`, with recorded upstream base revision
`b15702b29753188482b11ca85e997006b503d0f0`; migration file inventories identify
the actual imported bytes. Model family/version tags, native input/output
contracts and existing C++ include names remain unchanged. Maintained source
has new path-bound source identities after integration.

| Path | Responsibility |
| --- | --- |
| `encoders/` | Encoder families, provider definitions, adapters and model-owned workflows |
| `state_representation/` | Mapping observation embeddings to task-dependent latent states |
| `evaluation/` | Encoder/model protocols, cards, benchmarks and evaluation adapters |
| `scripts/` | Maintained model and evaluation runners |

Shared representation/evaluation utilities are in
[`../shared/representation/`](../shared/representation/), common settings in
[`../shared/settings/`](../shared/settings/), and immutable model metadata in
[`../shared/artifacts/`](../shared/artifacts/). Observation archives and market
timeline/array/target definitions live in [`../data/`](../data/README.md).
Latent dynamics, predictive training, direct-embedding forecast controls,
forecast issuance and their evaluation live in
[`../operations/forecasting/`](../operations/forecasting/).

The [project charter](../../doc/operations_over_representations_design_spec_v0.1.0.md)
distinguishes observation embeddings, learned latent states and forecast states.
Source placement does not establish prediction skill or downstream utility.
External/open models and API embeddings remain first-class representation sources.

## Encoder Hero development direction

**Hx15–Hx18 develop reusable encoder families and provider tools.** Hx20 and
Hx21 live in [`../fusion/`](../fusion/README.md), not in this directory. Read the
[shared encoder instructions](encoders/AGENTS.md) and the assigned Hero record.
The deliverable includes supported configuration, callable interfaces, portable
workflows and explicit constraints/coverage; a selected checkpoint is evidence
for an instance. Separate family/provider, configuration, recipe and instance.
Locally trainable components expose fresh construction, TRAIN fitting, training,
export and resume where implemented, with missing pieces stated. External
providers expose their actual upstream capabilities. Cuwacunu and other
applications own substantial application training and final docking; bounded
architecture research and tool verification remain here.

**Hx15 MTF is retained for reference and is not currently used in downstream
applications.** Its [portable family guide](encoders/mtf_jepa_mae_vicreg/family/README.md)
and [completed engineering record](encoders/mtf_jepa_mae_vicreg/family/verification/RESULTS_2026-10-22.md)
cover configurable CPU/F32 construction, explicit identity preparation,
training/resume, inference and independently verified source delivery. This
coverage establishes component behavior within the recorded fixtures, without
an encoder-quality or application-utility result.

**Hx16 RPB now supplies a configurable fresh-v22 family and its complete training
method.** The [family guide](encoders/raw_patch_bottleneck_mae/family/README.md)
covers TRAIN fitting, masked updates, checkpoint/resume and inference/embedding
export in the declared Linux CUDA/F32 runtime. The
[delivery receipt](encoders/raw_patch_bottleneck_mae/family/migration/delivery-2026-10-22.json)
binds the 117-file source package and isolated acceptance, including the final
documentation-only amendment. Two configurations passed bounded engineering
checks; historical research results, temporal limits and application training
retain their separate scope.

## Current research references

The current research default is **V16 candidate seed 946103, final update 2048,
native36**, explicitly chosen on 22 October 2026 in the family-owned
[research reference](encoders/causal_patch_bottleneck_mae/research-reference.json).
It is the strongest saved trend checkpoint, with March trend/activity gains over
its matched fresh V4 control. Seed 946102 is the neural-activity specialist;
seed 946101 is the deleted-view, market-scaled TEMPO-5 linear timing specialist.
The [live CPB ranking](encoders/causal_patch_bottleneck_mae/causal_patch_bottleneck_mae.ranking.md)
owns these separately identified checkpoint standings, and the
[research guide](../../doc/EMBEDDINGS.md) preserves the search history.
V4 remains the historical benchmark and known-validation reconstruction leader;
V3 is the clean baseline. RPB-v24.market-01.alt-01 native36 retains its historical
balanced-leader role. V14 is the latest catalogued historical instance, separate
from the new research-reference choice. The
[29 October handoff](encoders/causal_patch_bottleneck_mae/family/migration/HANDOFF_2026-10-29.md)
records the shared-clock development instance: width 128, seed 949102, native36,
2048 updates, published 1m/1h/1d bars, log-return view. The 29 October search is
[closed](../data/market/experiments/timeframe_published_bars_v1/CLOSE_2026-10-29.md)
on that instance. The packed-channel instance offered to Cuwacunu is the
[31 October handoff](encoders/causal_patch_bottleneck_mae/family/migration/HANDOFF_2026-10-31.md),
seed 949501. Further Mityane training on that packed layout is closed. Both leave the research reference unchanged.

The latest [V19 EMA-teacher study](evaluation/protocols/cpb_v19_teacher_v1/RESULTS.md)
completed three new 1,024-update candidates against the three preserved
scattered controls from the V18 study, with unchanged linear evaluation and three passing new
arithmetic audits. Volatility improved in 3/3 pairs (median ΔR² +0.053184) and
relation in 2/3 (+0.109583), while trend and activity each improved in 1/3 and had negative medians
(−0.029321/−0.053271). Four primary conditions failed; volatility protection
passed. No scoped linear record or reference changed. The
[record](evaluation/records/CPB-v19.teacher-001.json) distinguishes 3,072 new
updates from 3,072 reused control updates. The complete teacher/predictor training
method, configurable construction and typed resume travel with the
[component](encoders/causal_patch_bottleneck_mae/versions/v19/README.md);
inference remains the original-size student.

The preceding [V18 contiguous-mask study](evaluation/protocols/cpb_v18_span_v1/RESULTS.md)
completed three matched 1,024-update pairs with the established linear evaluation
and six passing arithmetic audits. It failed every continuation condition:
median R² changes were trend −0.221846, activity −0.090620 and volatility
−0.057857; relation improved +0.015733. No new scoped linear record or reference
was established. The [record](evaluation/records/CPB-v18.span-001.json) separates
5,376 new updates from 768 reused control-prefix updates. Configurable
construction, complete training and typed resume/export remain part of the
delivered component.

The [V16 study](evaluation/protocols/cpb_v16_temporal_v1/RESULTS.md) completed on
21 October 2026 under its [prospective protocol](evaluation/protocols/cpb_v16_temporal_v1/PROTOCOL.md).
Four January candidate record scopes span linear/neural trend (946103), neural
activity (946102), and deleted-view, market-scaled linear TEMPO-5 accuracy (946101).
Fresh V4 controls 946101/946103 also exceed the historical neural-trend result;
their gains remain separate from the candidate effects. Both March primary
targets pass: trend median paired linear R² gain **+0.047276** (2/3 positive
pairs), activity **+0.193252** (3/3). The declared screen is **mixed**:
volatility median ΔR² **−0.033500** misses the −0.01 protection, and no candidate
retains all eight strongest timing conditions. Candidate 946101 passes 1/8 and
the others 0/8; all fresh controls also pass 0/8. All six logical 2048-update
trajectories, twelve independent arithmetic audits and twelve checkpoint causal
proofs pass. These quality floors and causal proofs are separate. The research
choice establishes no application deployment or broader forecasting utility.

V3–V16 start from fresh learned weights. V15/V16 branch from V4's native36 source
design with fresh optimizer state and TRAIN statistics; V5–V14 use native40.
Candidate 946103 continued from its own saved 512 typed state after interruption,
with discarded-tail work repeated. Full execution cost, exact repetition count
and cumulative prefix parameter evidence remain unknown; measured suffix
evidence is separately scoped. The
[recovery declaration](evaluation/protocols/cpb_v16_temporal_recovery_v1/RECOVERY.md)
and [compact study record](evaluation/records/CPB-v16.temporal-001.json) bind that
history.

The [V15 objective study](evaluation/protocols/cpb_v15_objective_v1/RESULTS.md)
remains complete with its mixed February screen. Five of its six earlier scoped
records, from candidates and fresh controls, still lead; V16 now exceeds its
neural-activity record. The saved V14 TRAIN study remains an unexecuted draft.

## Development

The commands below build and inspect the retained baseline CLI. For the portable
MTF API and `mtf_family_train`, use its [family guide](encoders/mtf_jepa_mae_vicreg/family/README.md).

From the repository root in PowerShell:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', '-j2', 'all', 'test')
.\container.ps1 -Action exec -Command @('/opt/mityane-geometrics/embeddings/build/baseline/embedding', '--help')
```

Inside the existing managed container, use `/mityane` as the source root:

```bash
cd /mityane
bash code/scripts/task.sh baseline -j2 all test
```

The root Makefile, task runner, dependency lock and setup are authoritative.
Named builds and locks use `/opt/mityane-geometrics/embeddings/build/<session>`;
new task results use `/mityane/.temp/embeddings/runs/<session>`. See the
[environment guide](../../doc/ENVIRONMENT.md) for the existing container and SDK.

## Research and preservation

The saved 18 October handoff records the original pause boundary. Start with
the [current research guide](../../doc/EMBEDDINGS.md) for the user's later
reviewed V14 metadata application and the completed V15/V16 studies. Archived next-step notes do not authorize
new experiments or bypass the current scope and review requirements.

The complete pre-cleanup source/documentation component is retained byte-for-byte under
[`../../.archive/embeddings/pre-cleanup-2026-10-18/`](../../.archive/embeddings/pre-cleanup-2026-10-18/).
Historical ignored `output/`, `.build/` and `.external/` remain in this directory.
Frozen paths, source capsules, cards, checkpoints and receipts retain their
identities. Read the [archive guide](../../.archive/embeddings/README.md) before
following a historical source or reproduction pointer.

The repository [MIT license](../../LICENSE) covers the imported project code;
[third-party notices](../../THIRD_PARTY_NOTICES.md) retain upstream obligations.
