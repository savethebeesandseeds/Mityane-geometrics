# Preserved embeddings research

[`pre-cleanup-2026-10-18/`](pre-cleanup-2026-10-18/) is the byte-preserved
`code/embeddings/` source/documentation tree immediately before structural cleanup. It
retains the original `code/`, `doc/`, `archive/` and top-level files, including
the duplicate environment/build entries, license, original instructions,
frozen protocols, research source copies, reports, failures and receipts.
Its contents are evidence; do not edit them to follow current source paths.

The [cleanup receipt](../../doc/EMBEDDINGS_CLEANUP_2026-10-18.json) links the
[detailed preservation and source manifest](CLEANUP_MANIFEST_2026-10-18.json).
Existing ignore rules still apply to local render/build artifacts retained in
the snapshot; those files need their original local preservation copy on another
machine. No datasets, checkpoints or SDK binaries were added to source control.

Historical ignored `output/`, `.build/` and `.external/` remain at their
existing locations under `code/embeddings/`, including local datasets,
checkpoints and SDK transfer input. These directories were not relocated or
deleted as cleanup. The initial migration's source Git recovery directory
now lives at ignored `.temp/migration/embeddings-2026-10-18/source.git` after
the root artifact-directory rename; dated receipts retain the original
`output/migration/...` path identity.

## Current ownership

Paths on the left are relative to the old standalone component. Paths on the
right are relative to the current repository root.

| Original path | Maintained location |
| --- | --- |
| `code/encoders/` | `code/embeddings/encoders/` |
| `code/networks/state_representation/` | `code/embeddings/state_representation/` |
| `code/evaluation/` except `latent_dynamics/` and historical `tools/` | `code/embeddings/evaluation/` |
| `code/evaluation/tools/` | Archive only: completed-run audits, repairs and registration helpers |
| `code/evaluation/latent_dynamics/` | `code/operations/forecasting/evaluation/` |
| `code/networks/latent_dynamics/`, `predictive_training/`, `direct_embedding_control/` | Corresponding directories in `code/operations/forecasting/` |
| `code/latent_market/forecast_issuance/` | `code/operations/forecasting/issuance/` |
| `code/latent_market/versions/` | `code/data/market/versions/` |
| `code/shared/include/embedding/shared/data.h`, `src/data.cpp`, `tests/data_test.cpp` | Corresponding include/source/test subtrees in `code/data/observations/` |
| `code/shared/config/` and its owning `cuwacunu/config/` / `piaabo/parse/` parser headers | `code/shared/settings/` |
| Remaining `code/shared/` | `code/shared/representation/` |
| `code/artifacts/` | `code/shared/artifacts/` |
| Operational `code/scripts/` | `code/embeddings/scripts/` |
| Component Makefile/task entry | Root `Makefile` and `code/scripts/task.sh` |
| Component setup/container/lock | Root `setup.sh`, `container.ps1`, `dependencies.lock` |
| Component license/notices | Root `LICENSE` and `THIRD_PARTY_NOTICES.md` |
| `code/research/`, `doc/`, `archive/` | Preserved in this snapshot |

This table is a navigation aid, not a historical hash alias or reproduction
admission. Original receipt paths, source-file lists, source closures and model
identities remain unchanged. Maintained source adaptations produce different
source identities; never present them as the source bytes pinned by an old run.
Relative links within this full snapshot retain their original source context.
Absolute paths and references to ignored artifacts can still require an explicit,
verified mapping before historical reproduction.

## Instructions and pause boundary

Use the root [README](../../README.md), [environment guide](../../doc/ENVIRONMENT.md),
[current research guide](../../doc/EMBEDDINGS.md) and current `AGENTS.md` for
maintenance. Archived AGENTS, setup scripts, container launchers, commands and
status notes describe their dated context and must not be used as current
lifecycle instructions.

The [V14 handoff](pre-cleanup-2026-10-18/doc/HANDOFF_2026-10-18_V14_PAUSED_MIGRATION.md)
records the paused scientific work and unfinished metadata review/application.
Earlier proposed experiments and "next" notes do not resume that work.
Preserve negative results and superseded proposals alongside successful records.
