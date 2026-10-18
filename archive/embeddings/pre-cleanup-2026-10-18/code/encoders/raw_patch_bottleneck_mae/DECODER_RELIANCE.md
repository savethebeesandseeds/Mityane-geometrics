# RPB-MAE held-out decoder reliance

Date: 2026-10-05. Status: Implemented and measured under the frozen development
recipe; useful held-out decoder reliance established within this synthetic track.

This experiment advances the bottleneck evidence required by the
[implementation guidelines](RPB_MAE_implementation_guidelines.md) and shared
[evaluation policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md). Existing model
tests establish exact-export decoding, gradient reachability and the absence of
a token-state bypass. Held-out measurements test whether decoding uses useful
trajectory information beyond masks, channel identities and positions.

The [controlled reconstruction v1 card](../../evaluation/cards/controlled_reconstruction_v1.md)
defines the population, target masks, reduction and uncertainty. This is a
separate reconstruction protocol, not a reinterpretation of archived
controlled-pairs classification scores.

## Build and run

Use the existing managed container and a named session from the project root:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'reconstruction', '-j2', 'evaluation')
.\container.ps1 -Action exec -Command @('/opt/cuwacunu_embedding/build/reconstruction/embedding_evaluate', 'reconstruct', '--help')
```

Inside the container, allocate a unique parent and a new results directory:

```bash
eval_bin=/opt/cuwacunu_embedding/build/reconstruction/embedding_evaluate
mkdir -p /embedding/output/runs
run_parent="$(mktemp -d /embedding/output/runs/reconstruction-XXXXXX)"
"$eval_bin" reconstruct --output "$run_parent/results" \
  --seeds 101,202,303 --tasks reversal,level,amplitude,lag_sign \
  --channels 3 --history 32 --features 3 --patch-length 8 \
  --train-pairs 32 --validation-pairs 16 --test-pairs 64 \
  --rpb-steps 128 --metadata-steps 128 --batch-size 8 \
  --threads 1 --bootstrap-replicates 1000
```

`--output` must name a directory that does not exist. `--rpb-config FILE` supplies
an explicit architecture/training configuration; its dimensions, IDs, patch
length and uniform schema must agree with the card. `--channel-ids` accepts
signed int64 semantic IDs; `--units` declares one unit per feature. The resolved
recipe is recorded before fitting. When `--metadata-steps` is omitted, its budget
equals `--rpb-steps`. The command uses fresh training and accepts no checkpoint
input. The minimum `feature_harness` binary rejects this command.

Defaults are the existing independent-channel core: P8/H32, encoder width 64,
three pre-normalized temporal blocks, D32 local exports, a small two-layer
decoder and masked Huber training. The experiment adds no mixer, objective,
export normalization or alternate bottleneck capacity. Every task/seed gets
fresh fitted assets under the declared recipe.

## Predictions and controls

| Prediction | Information available |
| --- | --- |
| Normal | Trained encoder's exact compact local export from visible observations; fixed trained decoder and target metadata |
| Shuffled export | Another independent source's support-compatible export; receiver metadata and targets remain fixed |
| Zeroed export | Numeric zero in place of the export; same trained decoder and metadata, reported separately |
| Mask/metadata only | Frozen matched initial encoder receiving normalized numeric zeros, actual visible masks, semantic IDs and positions; independently trained identical decoder head |
| Matched untrained | Original initialized encoder/decoder and the same training-fit preprocessing; no model updates |
| Training mean | Training-only per-channel/feature mean in raw units; no held-out fitting |

The mask/metadata control updates only decoder parameters. Its encoder runs in
evaluation mode with gradients and dropout disabled and receives no observed
signal values. It can encode mask/position structure, so it is a stronger control
than feeding latent zeros into the trained decoder. Its compact representation
and decoder width match the primary; budgets and source exposure are recorded.

The adapter returns raw-unit CPU float64 predictions after float32 neural
decoding and float64 inversion of its frozen scaler. Shared evaluation owns
training-fit metric standardization, fixed targets, MAE/Huber aggregation,
source swaps, block bootstrap and reports. The adapter fit interface exposes
training observations/metadata only, never labels or held-out observations.
The ordinary RPB prepare/train/embed executable remains separate.

## Interpretation and evidence

Primary measurements are standardized masked MAE; Huber is secondary. The same
hidden observed coordinates and hierarchical channel/example reduction apply
to every prediction. Enumerated patch masks provide repeated measurements of
each source, not independent trajectories.

Shuffle complete source groups through disjoint swaps. Both transformed
variants and every repeated target mask stay in their exchange block during
uncertainty estimation. A donor cannot come from the receiver's source, and the
recorded mapping preserves channel/support compatibility. The engine reports
paired reconstruction degradation and unsupported populations explicitly.

A normal decoder's held-out advantage over the independently trained control,
together with degradation under valid shuffling, supplies evidence of reliance
on trajectory information. A zeroed-z effect alone does not establish that
claim. Loss reduction during training, passing structural tests, or an isolated
point estimate cannot substitute for this comparison and its uncertainty.

No acceptance threshold or representation-quality promotion is assigned by
this development recipe. It does not resolve whether lag information supports
the declared frozen probes, and it does not validate structured outages or real
sensor data.

## Measured development results

The [completed report](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/report.json)
contains twelve task/seed runs with the
[instantiated card](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/reconstruction-card.json):
C3/H32/F3, P8, 32/16/64 training/validation/testing source pairs, seeds
101/202/303, and 128 completed updates for both the primary model and
mask/metadata decoder. Training uses minibatches of eight and one Torch CPU
thread. All testing cases have 100% example, channel and target coverage.

Testing standardized MAE is shown below; lower is better. Entries are arithmetic
means across the three development seeds, not confidence intervals across
training runs.

| Task | Intact | Mask/metadata only | Shuffled | Untrained |
| --- | ---: | ---: | ---: | ---: |
| Reversal | 0.1839 | 0.8556 | 1.0154 | 0.8637 |
| Level | 0.3119 | 0.9318 | 1.1471 | 0.9310 |
| Amplitude | 0.0391 | 0.7806 | 1.1403 | 0.7842 |
| Lag-sign fixture | 0.0279 | 0.9130 | 1.1402 | 0.9072 |

Every run's mask/metadata-minus-intact and shuffled-minus-intact error
difference has a positive lower bound for both primary MAE and secondary Huber.
The primary paired effects and 95% intervals below come from the report; links
open the corresponding saved paired measurement arrays. Positive values favor
intact exports.

| Task | Seed | Mask/metadata minus intact (95% interval) | Shuffled minus intact (95% interval) |
| --- | ---: | --- | --- |
| Reversal | 101 | [0.651 (0.627–0.676)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-reversal/testing-metadata_only-minus-intact.pt) | [0.871 (0.626–1.123)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-reversal/testing-shuffled-minus-intact.pt) |
| Reversal | 202 | [0.702 (0.691–0.713)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-202-reversal/testing-metadata_only-minus-intact.pt) | [0.690 (0.446–0.937)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-202-reversal/testing-shuffled-minus-intact.pt) |
| Reversal | 303 | [0.663 (0.635–0.689)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-303-reversal/testing-metadata_only-minus-intact.pt) | [0.934 (0.659–1.185)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-303-reversal/testing-shuffled-minus-intact.pt) |
| Level | 101 | [0.634 (0.629–0.639)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-level/testing-metadata_only-minus-intact.pt) | [0.785 (0.548–1.060)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-level/testing-shuffled-minus-intact.pt) |
| Level | 202 | [0.622 (0.618–0.626)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-202-level/testing-metadata_only-minus-intact.pt) | [0.901 (0.653–1.150)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-202-level/testing-shuffled-minus-intact.pt) |
| Level | 303 | [0.604 (0.596–0.613)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-303-level/testing-metadata_only-minus-intact.pt) | [0.820 (0.532–1.065)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-303-level/testing-shuffled-minus-intact.pt) |
| Amplitude | 101 | [0.746 (0.737–0.755)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-amplitude/testing-metadata_only-minus-intact.pt) | [1.143 (1.059–1.217)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-amplitude/testing-shuffled-minus-intact.pt) |
| Amplitude | 202 | [0.742 (0.734–0.750)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-202-amplitude/testing-metadata_only-minus-intact.pt) | [1.155 (1.071–1.233)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-202-amplitude/testing-shuffled-minus-intact.pt) |
| Amplitude | 303 | [0.736 (0.713–0.758)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-303-amplitude/testing-metadata_only-minus-intact.pt) | [1.006 (0.869–1.127)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-303-amplitude/testing-shuffled-minus-intact.pt) |
| Lag-sign fixture | 101 | [0.888 (0.872–0.904)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-lag_sign/testing-metadata_only-minus-intact.pt) | [1.056 (0.927–1.182)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-lag_sign/testing-shuffled-minus-intact.pt) |
| Lag-sign fixture | 202 | [0.877 (0.866–0.886)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-202-lag_sign/testing-metadata_only-minus-intact.pt) | [1.131 (0.997–1.267)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-202-lag_sign/testing-shuffled-minus-intact.pt) |
| Lag-sign fixture | 303 | [0.890 (0.861–0.922)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-303-lag_sign/testing-metadata_only-minus-intact.pt) | [1.150 (1.049–1.253)](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-303-lag_sign/testing-shuffled-minus-intact.pt) |

Each testing run has 64 independent source groups, 128 paired-variant rows and
four target trials. Its uncertainty unit is 32 disjoint two-source exchange
blocks. Rows and trials are dependent measurements within those blocks; they
are not 128 independent observations or four times that sample size. The 1,000
bootstrap replicates give within-run intervals only. No interval across training
seeds or independent confirmation result is reported.

These results establish that the trained compact local exports carry trajectory
information used by the frozen decoder on these fully observed synthetic
signals. Good reconstruction on the lag-sign fixture is not a lag-classification
probe result or proof that cross-channel relationships are useful to a consumer.
Structured outages, real-data utility, costs and acceptance remain unmeasured
by this track. Zeroed-z and training-mean controls are retained in the full report.

## Validation and retained assets

Shared reconstruction-engine and RPB reconstruction-adapter tests passed inside
the managed container. Encoder/scaler mathematics and exported surface meanings
remain unchanged. The trained model, frozen scaler, independent decoder control,
untrained model, raw training archive and producer provenance are retained in
each run's provider assets; see the
[seed-101 reversal assets](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-reversal/provider-assets/)
and its [audit](../../../output/runs/rpb-implementation/reconstruction-2jehCR/results/seed-101-reversal/provider-assets/provider-audit.json).

A smaller four-update level run was repeated with identical
[reports](../../../output/runs/rpb-implementation/reconstruction-1Rf6x9/results/report.json)
and [cards](../../../output/runs/rpb-implementation/reconstruction-1Rf6x9/results/reconstruction-card.json)
in the [second run](../../../output/runs/rpb-implementation/reconstruction-O4oVf3/results/report.json).
Ordinary loading restored trainable parameters; comparison with frozen-provider
exports differed by at most `7.45058e-8`, while matching frozen gradient flags
restored exact equality. The ordinary CLI then resumed that four-update model
for one update and exported from the resulting five-update
[checkpoint](../../../output/runs/rpb-implementation/reconstruction-validation-HZp50W/resumed.pt)
and [embedding archive](../../../output/runs/rpb-implementation/reconstruction-validation-HZp50W/embeddings.pt).
The [source fingerprint record](../../../output/runs/rpb-implementation/reconstruction-validation-HZp50W/source-fingerprints.txt)
identifies the finalized executable sources and distinguishes adapter training
from the ordinary checkpoint writer.
The [validation manifest](../../../output/runs/rpb-implementation/reconstruction-validation-HZp50W/validation.json)
records the test, repeat, checkpoint and report-integrity checks.

Generated evidence is retained locally under `output/` and excluded from Git.
A clean checkout must reproduce these runs to populate the linked artifacts.
