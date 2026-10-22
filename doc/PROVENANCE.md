# Source provenance

## Project charter

The [Operations over Representations design specification](operations_over_representations_design_spec_v0.1.0.md),
`ORR-DESIGN-001` v0.1.0, defines Mityane geometrics' scope, capability families,
contracts and evidence standards. Imported research and interface components
retain the source identities below. Their provenance does not limit the project
to one encoder family, representation source or application domain.

## Dashboard backbone

On 18 October 2026, the interface, model, schemas, bundling helpers, model tests
and schema guide from `web/dashboard/` in
`https://github.com/savethebeesandseeds/non-verba`, revision
`b67d8a1e025dacfed8ac7aba2225d76cfa48f7e7`, were imported into
`web/dashboard/`. [The import receipt](DASHBOARD_IMPORT_2026-10-18.json) records
the original copied-file hashes. The source interface files had no local changes;
the source's locally edited Hero data and generated HTML were excluded.

The adaptation preserves the layout, styles and navigation. It changes project
names, documentation links, container orchestration and branding input; allows
an empty manifest; adds an explicit empty-project view; and keeps synthetic
interaction records confined to tests. No Heroes, project milestones or
Non-verba artwork were copied. The dashboard retains AGPL-3.0-only, with the
[full license](licenses/non-verba-AGPL-3.0.txt). It is a local artifact, with no
publication configured.

Linux Node 22.23.3 uses the exact cached official archive and SHA256 pinned by
the source project's `code/setup.sh`. The inspected source container was
`fcb9461f2ccc1d76871ee40713fcc93752b72e51a454890188cf9b1f052d7d4e`, image
`sha256:2fb1e6b1c3250aae15625a15c4496326bceb8487b26ac9f66cd94f7c2462bc1e`.
Only its 31,001,304-byte Node archive was streamed into the Mityane container's
`/opt/mityane-geometrics/web/downloads/` cache. The pinned binary and license are
installed under `/opt/mityane-geometrics/web/tools/`. No source-container data,
image or package was changed. See the dashboard guide for the artifact identity
and dependency-only setup procedure.

## Embeddings and environment

The research predecessor was `C:\Work\cuwacunu\cuwacunu_embedding`. On
18 October 2026 its complete working tree was moved to `code/embeddings/`.
The recorded upstream base is revision
`b15702b29753188482b11ca85e997006b503d0f0` of
`https://github.com/savethebeesandseeds/cuwacunu_embedding`. That base revision
alone is not the complete imported-tree identity; the migration's per-file
hash inventories identify the actual bytes and local operational adaptations.
Its MIT license,
third-party notices, model family/version identities, configurations, reports,
datasets, checkpoints and failed attempts are retained. This migration adds no
research training, fitted preprocessing or new quality measurement.

The original Git directory is retained, with its Windows ownership unchanged,
at ignored `.temp/migration/embeddings-2026-10-18/source.git`. It is recovery
history; root `output/` was subsequently renamed to `.temp/` with all contents
preserved. Dated receipts retain the original path identity. The component is
ordinary source in Mityane's existing repository,
without a nested repository or submodule. The complete before/after file
inventory and hashes are in that same local migration directory. Original
versions of adapted operational files were retained under the component's
`archive/mityane-migration-2026-10-18/`. The structural cleanup described below
preserves that tree in the complete pre-cleanup snapshot.

The development environment uses the already installed embedding snapshot made
on 12 October 2026. `environment-image.json` preserves its immutable image ID,
source container identity and original Debian base digest. A new tag reuses the
same layers. The original `cuwacunu` container is preserved. The obsolete
`cuwacunu_embedding` container is retired by the migration after local recovery
preservation and component verification; see the
[migration record](EMBEDDINGS_MIGRATION_2026-10-18.md).

Environment support comes from the MIT-licensed Waajacu source projects:

- `cuwacunu_dev` revision `221ce90d25336f8ce81b9d7f272928191f584324`:
  adapted `container.ps1`, offline `setup.sh`, and
  `code/scripts/install-libtorch.py` (only its project entry guard and descriptive
  name change). The verifier retains its original SDK proof semantics.
- `cuwacunu_embedding` revision `3316fdda42de103282c62f47cfd0300cbf2d272b`:
  `dependencies.lock` and the retained MIT license. The dependency lock is also
  byte-identical to the `cuwacunu_dev` copy, SHA256
  `b8f946448de4c2a6b0b2d94794a096266644dece00fd8fd27ef5a0a61e8684d3`.

These revisions identify the inspected repositories; the copied support files
are recorded by SHA256 in `environment-image.json` because local source edits
can exist independently of a Git revision.

The [18 October Python maintenance migration](PYTHON_MIGRATION_2026-10-18.md)
adds Node replacements for current SDK/layout verification and source identity,
and a Node/C++ replacement for dashboard artwork preparation. These are new
maintained implementations with their own source identities. The original
Python support sources remain available only as archived evidence after the
[working-copy removal](PYTHON_MIGRATION_REMOVAL_2026-10-18.json); replacements do not
stand in for source bytes pinned by earlier research. SDK verification preserves
the existing immutable proof/inventory without modifying the SDK or its profile.

When importing another embedding, record its family/version, checkpoint hash,
input layout, native output layout, masks, scaler and training provenance,
evaluation protocol, source revision and license. Keep evidence from the
predecessor intact and distinguish inherited measurements from new results.

## Structural cleanup, 18 October 2026

The initial import retained a standalone project hierarchy. The subsequent
cleanup integrates maintained source by responsibility: encoder/state models
and model evaluation in `code/embeddings/`, forecasting and its evaluation in
`code/operations/forecasting/`, data handling in `code/data/`, and shared
settings, tensor/evaluation utilities and artifact metadata in `code/shared/`.
Root environment/build entries replace duplicate component configuration. The
component MIT license was byte-identical to the root license; attribution and
third-party obligations are consolidated in root `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

The full source/documentation component immediately before this cleanup is retained
byte-for-byte at `.archive/embeddings/pre-cleanup-2026-10-18/`, including its
`code/`, `doc/`, `archive/` and top-level files. Historical ignored `output/`,
`.build/` and `.external/` stay in place under `code/embeddings/`. Original
source capsules, evidence, model IDs, hashes and old path identities are not
rewritten. Maintained source references/build inputs change with ownership;
new source identities must not be substituted into historical receipts.

The [archive guide](../.archive/embeddings/README.md) and
[current research guide](EMBEDDINGS.md) distinguish preserved records from
current instructions. The earlier migration record and verification receipts
remain dated accounts of that import. Cleanup supplies no new scientific
measurement, metadata promotion or authorization to resume paused research.

## External multimodal provider

On 18 October 2026, Google's stable `gemini-embedding-2` service was registered as
a documented integration target in
[the Google multimodal provider folder](../code/embeddings/encoders/google_multimodal_embedding/README.md).
Its adjacent `provider.json` records official model/API references, supported
modalities, the proposed output contract and current integration state. This
registration imports no weights or SDK and establishes no local operation
quality. The API adapter and evaluation remain proposed.

A hosted service model ID does not identify immutable model weights. Future
requests must retain configuration, returned revision when available, execution
date, source identity and response artifact identity. Service terms and dataset
transmission permissions remain separate from the repository's source license.

## NSW archival reference data

On 20 October 2026, the owner authorized selection of a documented open randomized
dataset for Hx06's two new profile bindings. The
[NSW data adapter](../code/data/observations/open_benchmarks/nsw_wooldridge_v1/README.md)
retains `wooldridge::jtrain2`, package 1.4-7, source revision
`427d117f3888cd3a630ee22a1aedc8f036d128ff`, with the publisher's GPL-3 declaration,
full license text and attribution to Justin M. Shea, LaLonde and Dehejia/Wahba.
Seven original publisher/documentation artifacts and all data-only preparation
receipts are retained under ignored `.temp/data/hx06-nsw-reference-v1/`.
The exact served CSV is 37,008 bytes, SHA256
`3dab3d903e07549343ae3aa51ce6530afc490c5861f6ebec76360e48fed9f9df`.
It is a separately identified R-universe serialization; no independent numerical
equivalence with the retained RData binary was asserted.

The [two profile instances](../code/operations/counterfactuals/profiles/instances/nsw_reference_v1/README.md)
bind the actual 445-row roster and unfitted `[re74]` raw descriptor, source and
representation hashes, permissions, distinct protocols and limitations. Data
import/schema checks and baseline-only materialization do not execute matching,
causal estimation, model fitting or scientific evaluation. The archival benchmark
is publicly exposed, has unknown original information-availability/design details
and establishes no fresh protected assessment or causal result. Raw/derived
individual records were not added to Git or sent to an embedding provider.
