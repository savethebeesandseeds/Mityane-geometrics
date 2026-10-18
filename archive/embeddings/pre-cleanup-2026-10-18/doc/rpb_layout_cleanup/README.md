# RPB source-layout cleanup — 15 October 2026

The existing work was committed and pushed before the cleanup, at the user's
explicit request: integration `c5d1c7f3a203180c89e1255ece676a1c20fe0852` and
embedding `0ab44b7628344815a85959310441269a9b857230`. The layout cleanup remains
uncommitted and unpushed.

The existing [version index](../../code/encoders/raw_patch_bottleneck_mae/versions/README.md)
is the navigation authority. Its existing `index.json` has an additive
`live_layout` section. All 30 original design records and the original index
fields retain their historical declarations and hashes.

```text
code/encoders/raw_patch_bottleneck_mae/
  README.md
  include/                  compatibility public includes
  shared/
    README.md
    include/                existing common model/core/scaler and adapters
    src/ tests/ config/     existing shared files, where available
  versions/
    README.md index.json RPB-v1.json ... RPB-v30.json
    vN/
      README.md             current entry, config, sharing and availability
      include/ src/ tests/ config/   only components that already existed
```

Integration's old `rpb_*` include roots remain compatibility includes; original
import provenance and historical module documentation stay in place. The
configuration paths used by existing CLIs and application defaults retain
their exact bytes. Eleven integration canonical configuration copies use the
same version paths and bytes as embedding. This avoids changing the default
encoder selection or historical command semantics.

The physical moves cover 120 integration files and 225 embedding files.
All 318 moved implementation, test and configuration files retain their exact
bytes. The remaining 27 moved files are Make fragments or shell launchers:
their differences are build/configuration paths and the two shell launchers'
repository-root traversal. Existing include spellings resolve through small
forwarding headers. No encoder body, API, namespace, model tag, default,
arithmetic, RNG consumption, registered name or checkpoint key was edited.

## Audit records

- [relocation.json](relocation.json): every old/new move path, original and
  final SHA-256, sizes, compatibility headers/configs and build-path changes.
- [configuration-aliases.json](configuration-aliases.json): unchanged selected
  application settings and their canonical version-folder copies.
- [additional-build-path-edits.json](additional-build-path-edits.json): canonical
  source/configuration globs and private test-support provenance inputs.
- [consumer-path-edits.json](consumer-path-edits.json): active preparation and
  check scripts' source-path substitutions. Historical receipt data stays intact.
- [synchronization.json](synchronization.json): all 99 inherited portable pairs
  match their original hashes, 85 common canonical headers match, and all 11
  canonical configuration copies match across repositories.
- [preservation-before.json](preservation-before.json) and
  [preservation-after.json](preservation-after.json): full protected-artifact
  inventory, including ignored checkpoints, result trees, scalers, fitted heads,
  capsules, provenance, causal source and selected defaults. The one preexisting
  dangling Linux test symlink is represented by its raw reparse payload.
- [verification.json](verification.json): bounded container build, checkpoint,
  output, registered-state and initialization-RNG checks and their exact scope.

All compatibility checks passed in the existing container. Ten saved typed
checkpoints (v22–v30, both v24 widths), eleven constructors and forty batch-two
views give byte-identical before/after integration and embedding witnesses:
26,433,986 bytes, SHA-256
`5d217382c4bae71862bf70f4f4b0b27c4a72121dbc83f4d32d3c891ed577b268`.
The witness includes native/decoder/validity state, registered parameter/buffer
names and bytes, CPU/CUDA RNG and v30 variance. Rebuilt v19 matches 128 saved
windows exactly. Embedding v21 and the legacy workflow build successfully
without execution. All 23 edited consumer scripts pass container syntax checks.
The full preservation comparison passes for 80,104 regular files and the
preexisting symlink: zero changed/missing files or unexpected additions.

Large before/after witnesses and container build logs remain under integration's ignored
`output/engineering/rpb-layout/`; no historical output was replaced. Build
products and the standalone source snapshot stay inside the existing managed
`cuwacunu` container. No training, scaler/head fitting, quality rerun or benchmark
is part of this cleanup.

## Historical mappings

Current shared `Model` options and policy descriptors are the available entry
for several early designs. They do not establish byte-exact equivalence to
every historical v1–v19 training definition. Original registries, cards and
frozen capsules remain authoritative; no historical implementation was
reconstructed. Integration lacks dedicated v9/v11/v15/v21 components; their
entry links lead to embedding. The existing index gives each version's actual
availability and dependencies.

RPB-v22 stays primary, native36 stays the balanced reference, and
`causal_patch_bottleneck_mae` remains unchanged. These are layout and
compatibility checks, with no new encoder-quality claim.
