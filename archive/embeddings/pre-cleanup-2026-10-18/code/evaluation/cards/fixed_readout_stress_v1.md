# Fixed-readout stress v1

Protocol: `fixed-readout-stress-v1`. Evidence stage: `development`.
Enable with `--stress-sweep fixed-readout-v1` on the controlled-pairs-v2 driver.
The driver saves its instantiated stress card before generation or fitting.

Fit the encoder, raw scaler, feature normalizer, compression and probes on the
ordinary permitted training split once. Corrupt testing observations only and
reuse every fitted object. This track measures fixed-readout robustness; it does
not fit an outage-aware encoder or readout and does not supply consumer acceptance.

| Case | Fixed construction |
| --- | --- |
| Intact | Original observed testing batch |
| Coordinate deletion | Additional 10/30/60/90% deletion; nested source-specific uniform draws over C×H×F |
| Temporal blackout | 25/50/75% of H, rounded up; all channels/features share the interval; centered on a source-specific anchor and clamped at the edges |
| Whole channel absent | One case for each declared semantic channel ID |
| All absent | Remove every observed cell |

One mask is shared by both variants of a source and all providers. Named streams
depend on source/task identity and never labels or provider order. Corruption
only removes observations and stores newly hidden raw cells as zero. Metadata,
units, semantic IDs, endpoints and visible values are preserved. Masks are saved
with batch/source mapping, checksums and actual retention counts.

For each supported base fit, report coverage, conditional accuracy and
full-population correct/total with abstentions treated as failure. Unsupported
base fits remain unsupported. No observations means no signal support; complete
channel concatenation needs all required channels. Metadata controls may retain
validity when signal surfaces abstain and are labeled as controls.

Architecture comparisons use each pair's stressed common-valid population.
Stress-versus-intact effects use their common-valid population. Report paired
source-group intervals conditional on the fixed models/readouts, with null
intervals when source support is insufficient. Full-population effects compare
correctness on the complete fixed test population, including abstentions.
Cases are correlated diagnostics; their rows/cases are not independent runs.

The base legal-raw oracle remains a construction gate. Under corruption its
score and coverage are descriptive, since signal removal can defeat a legal
oracle. There is no new stress acceptance threshold or test-driven selection.

The engine writes root `stress-card.json` and `stress-report.json`, plus each
seed/task's `stress/` artifacts. The ordinary v2 report remains separate.
The [evaluation policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md) defines the
remaining domain, confirmation, cost and consumer contracts.
