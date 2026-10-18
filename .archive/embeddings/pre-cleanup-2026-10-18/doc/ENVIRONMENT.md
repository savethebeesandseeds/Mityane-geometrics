# Embeddings development environment

This is a component of Mityane geometrics. Read the authoritative
[parent environment guide](../../../doc/ENVIRONMENT.md) and
[component README](../README.md) for current operations. The sole development
container is `mityane-geometrics`, mounting the parent repository at `/mityane`.
Its existing image, GPU access, port, SDK and container identity are reused.
Component container/setup entries delegate to the parent.

From the parent PowerShell directory:

```powershell
.\container.ps1 status
.\container.ps1 -Action exec -Command @('bash', 'setup.sh', '--no-download')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/embeddings.sh', 'baseline', '-j2', 'all', 'test')
.\container.ps1 -Component embeddings shell
```

| Location | Purpose |
| --- | --- |
| `/mityane/code/embeddings` | Component working directory |
| `/opt/mityane-geometrics/embeddings/build/<session>` | Build products, task lock and temporary files |
| `/mityane/output/embeddings/runs/<session>` | New named task results |
| `/mityane/code/embeddings/output` | Preserved historical local artifacts; some direct legacy runner defaults |
| `/mityane/code/embeddings/.build` | Preserved extraction reference and installation evidence |
| `/mityane/code/embeddings/.external` | Preserved Linux SDK transfer input; not an active toolchain |
| `/opt/cuwacunu_embedding/libtorch` | Shared installed SDK, unchanged original proof identity |

Session names isolate build products, not source or GPU resources. Never rewrite
frozen cards, measured reports, model versions or source capsules for a path
cleanup. The original [environment guide](../archive/mityane-migration-2026-10-18/doc/ENVIRONMENT.md)
retains historical research procedures; old container names and absolute paths
in it are superseded for ordinary development. Historical reproduction requires
its own source verification and reviewed path mapping. Reuse saved measurements
for reports rather than rerunning completed research.