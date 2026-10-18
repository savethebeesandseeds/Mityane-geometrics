# Embeddings component migration, 18 October 2026

The complete former `C:\Work\cuwacunu\cuwacunu_embedding` project is migrated
to `C:\Work\Mityane-geometrics\code\embeddings` as the embeddings component of
Mityane geometrics. Source revision is
`b15702b29753188482b11ca85e997006b503d0f0` from
`https://github.com/savethebeesandseeds/cuwacunu_embedding`; the source working
tree was clean before migration. Its MIT license and third-party notices remain.
The integration and verification receipt is recorded alongside this guide.

## Preservation

The directory is moved on the same NTFS volume, including all ignored data,
checkpoints, results, archived sources, `.build/` evidence and the staged
`.external/` bundle. No scientific model identity, frozen card, checkpoint,
measured report or research source capsule is renamed internally or rewritten
for the new project name. The component retains its internal `code/`, `doc/`
and `archive/` layout and relative links.

The original `.git` directory is moved to ignored
`output/migration/embeddings-2026-10-18/source.git`, retaining its original
`WAAJACU\santi` ownership. Mityane keeps its existing Git repository. The
component is ordinary source, with no nested repository or submodule. No
external account, repository owner, remote, authentication or global Git
configuration is changed.

The deeper component location makes a preserved archived header exceed Windows'
legacy path limit. Repository-local `core.longpaths=true` is enabled after a
Git dry run confirmed the failure; the working repository remains owned by
`WAAJACU\santi`. No source was staged or committed by the migration.

The local migration directory contains `source-complete.jsonl`, `source-summary.json`,
container inventories, adapted-file records and validation logs. The original
placeholder embeddings README is preserved there too. Original bytes of adapted
operational files are retained in the component at
`archive/mityane-migration-2026-10-18/`; source history also retains them.
The imported `.gitattributes` preserves component working-tree bytes under the
parent repository, including historical line endings.

## Integration

The root `container.ps1` remains authoritative and accepts
`-Component embeddings` for component `exec` and `shell` working directories.
The component `container.ps1`, `setup.sh` and SDK installer entry delegate to
the root launcher and offline verifier. No standalone component environment is
created, and the component cannot silently pull the former Debian image.

`code/scripts/embeddings.sh` is the parent task entry. Component Makefile and
operational runner build defaults now use
`/opt/mityane-geometrics/embeddings/build/<session>`; new named task results use
`/mityane/output/embeddings/runs/<session>`. Direct legacy research runners retain
component-relative output defaults where bound to a procedure. The original
installed SDK remains at `/opt/cuwacunu_embedding/libtorch`; its proof, inventory,
runtime profile and original setup path are unchanged. The historical staged
host bundle is preserved as an ignored transfer input.

Only operational defaults, delegates, current entry documentation and Git
attributes are adapted. Consequently new operational source fingerprints can
differ from historical receipts. Frozen sources and evidence retain the old
absolute paths and hashes; historical reproduction requires separately reviewed
path mapping and admission. This migration performs no research training,
quality rerun, model promotion or publication.

## Containers

The existing `mityane-geometrics` container is reused:
`381c6b78f4fdb301cd132050ffd8095587c0794829a7a73a547fdbc0484630e1`.
Its pinned environment image, `/bin/bash` command, single project bind mount,
loopback port 4874, GPU configuration, 1 GiB shared memory and restart policy
remain as documented in the parent environment guide.

The obsolete stopped `cuwacunu_embedding` container is
`1e8de9cfec33591fb06b2f182bff04f6e9fecf2ddeb0aa865f34eeac3639c2ca`.
Before its authorized removal, its writable filesystem is preserved in the local
recovery image `mityane-geometrics:embeddings-retired-2026-10-18`, immutable ID
`sha256:c718134b046e555c82a6d240bafba40577cd2a0ed27842d7e4e587772ed05030`.
This recovery image is not published or substituted for the development image.
Its bind-mounted files are preserved by the separate directory move. The old
container has no named or anonymous volume. Only this exact container is removed
after component verification; other containers, images, caches and volumes are
preserved. The local recovery image intentionally retains the old writable data,
so removing the container is not a disk-space cleanup.

The exact old container was removed after the checks passed. All 27 other
preexisting containers remain, and the existing Mityane container is running.

## Completed verification

- All 110,029 moved files, totaling 35,727,855,476 bytes, were compared with the
  source inventory: regular-file SHA256 and reparse metadata, zero mismatches.
- All 61 adapted operational files have byte-exact originals preserved. No C++
  model or header file was changed.
- All preserved Git descendants have the intended Windows owner. Ordinary user
  Git status, complete source visibility, Git import dry run and whitespace
  checks passed. No files were staged or committed.
- The existing environment verified all 321 locked packages and the original
  SDK inventory without installation, downloads or SDK mutation.
- Both PowerShell entries, 55 adapted shell/Python syntax checks, component
  working directories and SDK verifier delegation passed.
- A fresh baseline CLI build, all eight baseline test executables and CLI
  startup passed in the existing managed container. These are engineering
  fixtures, including small synthetic training/resume/evaluation checks, not
  new research measurements or a replay of historical quality studies.

The [verification receipt](EMBEDDINGS_MIGRATION_VERIFICATION_2026-10-18.json)
records identities, hashes, validation scope and final container state.
