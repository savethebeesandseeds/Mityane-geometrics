# Primary encoder selection: RPB-v22

On 13 October 2026 the user selected the strongest completed encoder design
for normal `cuwacunu` integration. RPB-v22 is now the primary architecture.
This is an application selection change, not a new architecture, trained
instance, synthetic quality experiment or market result.

## Application entry point

```cpp
#include "cuwacunu/encoder/primary_encoder.h"
#include "cuwacunu/integration/configurable_encoder_input.h"
namespace enc = cuwacunu::encoder::primary;

auto c = enc::read_settings("config/encoder.conf");
auto raw = cuwacunu::integration::configurable_encoder_input(batch, channel_ids, c);
auto scaled = enc::transform(train_fitted_scaler, raw, c);
auto native = enc::native_embedding(model->encode(scaled), c);
```

`primary` is a compile-time alias of the existing `content` API. Model tag,
typed settings, checkpoint kind, layout, normalization, masks and arithmetic
are all the existing RPB-v22 implementation. No runtime model registry is added.
Applications still need to construct a matching model or load its typed
checkpoint and reuse its TRAIN-fitted scaler.

| Role | API | Settings |
| --- | --- | --- |
| Primary architecture | `cuwacunu::encoder::primary` | `config/encoder.conf` — RPB-v22 |
| Explicit same architecture | `cuwacunu::encoder::content` | `config/encoder_v22.conf` — RPB-v22 |
| V20 base and historical reference | `cuwacunu::encoder::configurable` | `config/encoder_v20.conf` — RPB-v20 |
| Frozen V19 reference | Existing V19 API | Its original geometry/checkpoints |

Primary geometry remains C3/H128/F9/P8/W64/D32, with 20 learned coordinates and
12 fixed spectral coordinates. Input values/masks are `[B,C,H,F]`; native output
is one `[B,D]` vector per window. The shared raw nine-feature past-only adapter
and core file parser stay independent of the encoder choice.

## Compatibility and source preservation

The original v20 selected settings were copied byte-for-byte into
`config/encoder_v20.conf` before changing `config/encoder.conf`. Their SHA256 is
`c136f42f4cf89e50d00b17f6d9ad13bbbe39c2f59ba71dc180eb1c63ef077c8b`.
The live [v20 backport manifest](CONFIGURABLE_ENCODER_BACKPORT.json) now names
that relocated settings path and retains its original hash. All 15 v20 and
six v22 portable files continue to match the standalone repository exactly.
V19 imported sources and historical checkpoints are unchanged.

Existing callers that pair the v20 API with `config/encoder.conf` must either
switch to the primary API or explicitly use `config/encoder_v20.conf`.
The typed readers reject a wrong model tag instead of silently constructing
another architecture. V20 and v22 checkpoint loaders continue to reject one
another's artifacts. No old checkpoint is relabelled or overwritten.

The [initial optional-sync record](CONTENT_ENCODER_SYNC.json) remains historical
evidence of the prior integration and its CUDA check. Its selected-v20 fields
describe that earlier moment, not the current primary setting. This document
and [the selection receipt](PRIMARY_ENCODER_SELECTION.json) own the current
application choice.

No portable model arithmetic, scaler or core settings improvement was introduced
by this selection. The v22 implementation already exists in
`cuwacunu_embedding`; only application-facing wiring, checks and documentation
change here. Standalone documentation records the new application selection;
market loading and this application's primary alias remain here.

## Focused validation

The coordinated managed-container check passed with exit code 0 in the existing
`cuwacunu` GPU container. Seven check executables passed using:

```bash
bash code/scripts/task.sh primary-v22 -j2 test-primary test-v20
```

The successful run is `output/runs/primary-v22-selection/check-wXBnWV`.
Exact logs and before/after settings hashes are preserved under
[the durable capture](results/primary_encoder_selection/check-wXBnWV/build-and-test.log).
The [selection receipt](PRIMARY_ENCODER_SELECTION.json) records all 21 verified
portable pairs and captured-file hashes. The selected settings did not change
during the run. The first failed compile log remains separate and preserved.

The new primary witness reads the actual selected settings and constructs the
full default model. Artificial CSV candles pass through the real mmap loader,
raw nine-feature batch, shared mask adapter, TRAIN-fitted scaler, CUDA native
encoder and immutable typed checkpoint reload. It checks candle/per-value gaps,
future exclusion, masked storage, allmissing output and cross-family refusal.
It trains no model and fits no classifier.

The existing H129 content witness checks selected/explicit-v22 settings parity,
one actual artificial CUDA optimizer update, nonzero relation gradients,
fixed-tail preservation and typed reload. V20 checks exercise the preserved
reference settings, configurable model, loader integration and checkpoint
integrity. These are engineering witnesses, not market or synthetic accuracy
measurements.

Default `all`, `test` and `test-stability` include the primary v22 witnesses.
`test-encoder` selects the primary check; `test-v20`, `test-v19` and
`test-reference` retain explicit historical checks. A dry run inspects default
target routing without replaying the whole reference suite.

The first build attempt exposed a const model-holder declaration in the new
primary checkpoint witness. The declaration was corrected; its failed build log
is preserved separately. No encoder quality trajectory ran in that attempt.

## Trained weights and next work

The [saved RPB-v22.alt-01 development evidence](../../cuwacunu_embedding/doc/CONFIGURABLE_CONTENT_RELATION_V1.md)
supports this architecture choice on known TEMPO-5 data. It does not select
trained market weights. No synthetic checkpoint is copied or selected for
market inference, and no market-trained checkpoint exists under this choice.

Market preparation still needs source/cadence selection, chronological splits,
bounded TRAIN statistics, resource measurements and a tagged pilot instance.
Follow [the continuation](CONTINUATION.md). Fresh synthetic confirmation remains
a separate prospective research experiment in `cuwacunu_embedding`.
