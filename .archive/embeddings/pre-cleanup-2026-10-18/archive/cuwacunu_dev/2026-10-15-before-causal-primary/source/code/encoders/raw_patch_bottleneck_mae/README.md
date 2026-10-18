# Raw patch bottleneck MAE

Start with the [existing version index](versions/README.md) for current version entries, canonical configuration paths, shared dependencies and repository availability. Read [shared components](shared/README.md) for the preserved dependency structure. Implementation files are organized under versions/vN/ and shared/; old public include roots remain compatibility forwards.

RPB-v22 stays primary. config/encoder.conf and historical defaults stay unchanged. RPB-v24.market-01.alt-01 stays the balanced native36 reference. The separate causal_patch_bottleneck_mae family stays unchanged. This cleanup changes source layout only.

Each vN/README.md identifies an actually present current implementation/helper or shared recipe. Missing historical definitions and integration-only availability gaps remain explicit. No historical Model body is reconstructed or populated by copying today's implementation.

The original rpb_v19 import manifest and other retained provenance remain at their original paths. The original 30 design records, instance registries, checkpoints, scalers, heads, results, capsules and frozen receipts remain authoritative and unchanged. code/scripts/rpb-layout-plan.json records old-to-new canonical paths; cleanup verification records before/after hashes. No training, head fitting, quality rerun or benchmark is part of this task.
