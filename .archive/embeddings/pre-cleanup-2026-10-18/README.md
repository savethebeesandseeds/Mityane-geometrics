# Mityane geometrics: embeddings

The embeddings component contains the complete research project formerly named
`cuwacunu_embedding`, imported from revision
`b15702b29753188482b11ca85e997006b503d0f0`. It includes the MTF, RPB and causal
patch bottleneck encoder families, model-owned training/evaluation, shared tensor
utilities, research protocols, documentation and preserved historical evidence.
Model family/version tags and native input/output contracts keep their identities.

The parent project's [Operations over Representations charter](../../doc/operations_over_representations_design_spec_v0.1.0.md)
defines the broader scope. This component supplies representation providers and
model-owned research; reusable operations and their evidence live separately in
`../operations/`. Custom/frozen encoders and externally supplied representations
can both serve an operation when their contracts and evidence requirements hold.

Start with [current research status](doc/CURRENT_STATUS.md), the
[documentation index](doc/README.md) and the [component environment guide](doc/ENVIRONMENT.md).
The [parent environment guide](../../doc/ENVIRONMENT.md) is authoritative for
container lifecycle, dependencies and SDK verification. The
[migration record](../../doc/EMBEDDINGS_MIGRATION_2026-10-18.md) records provenance.

## Development

From the Mityane repository root in PowerShell:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/embeddings.sh', 'baseline', '-j2', 'all', 'test')
.\container.ps1 -Component embeddings -Action exec -Command @('/opt/mityane-geometrics/embeddings/build/baseline/embedding', '--help')
.\container.ps1 -Component embeddings shell
```

Inside the existing `mityane-geometrics` container:

```bash
cd /mityane/code/embeddings
bash code/scripts/task.sh baseline -j2 all test
```

The component's PowerShell launcher delegates to the parent with this working
directory. Its setup delegates to the parent's offline dependency verifier.
Use named sessions to isolate build products and locks under
`/opt/mityane-geometrics/embeddings/build/<session>`. New named task outputs go to
the parent's `output/embeddings/runs/<session>`. Direct legacy runners retain
their historical component-relative output defaults where bound to a protocol;
read the relevant procedure before using them. The installed, verified Linux
SDK remains at `/opt/cuwacunu_embedding/libtorch`.

## Preserved layout and evidence

`code/` owns encoder, shared utility, network and research source; `doc/` owns
the scientific guides, cards, ledgers and reports; `archive/` retains earlier
source. Historical ignored `output/`, `.build/` and `.external/` were moved intact.
No research training, quality rerun or reference promotion is part of this migration.

The complete [pre-migration README](archive/mityane-migration-2026-10-18/README.md)
retains detailed historical research/API guidance. Its old container commands
and filesystem locations are historical. Frozen source capsules, hashes and
records keep their original bytes and paths; use a separately reviewed mapping
for historical reproduction. Operational build defaults have been adapted, so
new operational source fingerprints can differ from historical receipts.

The original [MIT license](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md)
remain applicable. Checkpoints, generated results and SDK binaries stay ignored
by Git. Source history is preserved in the parent migration recovery directory;
this component is ordinary source in the parent's existing repository.