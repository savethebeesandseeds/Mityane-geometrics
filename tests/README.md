# Tests

`test-settings` covers the shared settings parser independently of an encoder.
`python3 -B code/scripts/check-layout.py` verifies the source/evidence archive,
relocations, ignored historical artifacts and maintained documentation links.
Use `--links-only` while editing documentation. Run these through the managed
container from `/mityane`.

Run builds and tests through the managed container, using the repository-root
Makefile and named task entry:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'baseline', '-j2', 'all', 'test')
```

Tests live beside the source they exercise: encoder/model checks in
`code/embeddings/`, forecasting checks in `code/operations/forecasting/`, data
checks in `code/data/`, and common utility checks in `code/shared/`. The root
Makefile integrates their existing targets. Build products remain under
`/opt/mityane-geometrics/embeddings/build/<session>`.

Operation tests cover declared input/output contracts, space compatibility,
provenance and failure/abstention conditions. The forecasting
[contract](../code/operations/forecasting/CONTRACT.md) and
[evidence record](../code/operations/forecasting/EVIDENCE.md) distinguish
artificial engineering checks from unmeasured scientific claims.

Evaluate scientific claims with separately admitted protocols and evidence
records under the [project charter](../doc/operations_over_representations_design_spec_v0.1.0.md).
Engineering fixtures do not establish operation or composed-pipeline utility.
Routine build/test execution and dependency verification do not resume the
paused research protocols in the [archive](../archive/embeddings/README.md).
