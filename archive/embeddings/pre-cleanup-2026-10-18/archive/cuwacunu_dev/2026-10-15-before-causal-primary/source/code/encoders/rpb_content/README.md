# Primary content-only ordered-relation encoder — RPB-v22

This API adds normalized odd/even cross-channel relations to the
learned portion of the existing v20 embedding. Support counts control
normalization and validity, without additive observation-density inputs.
The five portable headers and settings are exact copies from the separately
tagged standalone v22 design; see [the sync manifest](../../../doc/CONTENT_ENCODER_SYNC.json).

The primary application entry point is `cuwacunu::encoder::primary`, selected
through [config/encoder.conf](../../../config/encoder.conf). It aliases the
existing `content` API; there is no new model arithmetic or runtime registry.
V20 remains explicit through `cuwacunu::encoder::configurable` and the exact
preserved [config/encoder_v20.conf](../../../config/encoder_v20.conf). Its 15
portable file pairs and the frozen v19 reference remain unchanged, with only
the live v20 settings-pair path relocated. The explicit `content` namespace and
[config/encoder_v22.conf](../../../config/encoder_v22.conf) remain supported.
No trained market checkpoint is copied or chosen. See
[the selection record](../../../doc/PRIMARY_ENCODER_SELECTION.md).

The additive [streaming TRAIN support](../../../doc/STREAMING_TRAINING_SUPPORT.md)
provides bounded TRAIN statistics and a distinct v22 model/scaler/AdamW resume
archive with source/split/recipe identities and deterministic attempt counters.
Its focused CUDA continuation check passes; the inference API below remains
separate and unchanged.

The native contract remains CUDA float32 `[B,C,H,F] -> [B,D]`, with the same
v20 geometry rules. At C3/F9/D32, 20 learned coordinates receive a zero-start
20x1944 content projection; the final 12 fixed spectral coordinates remain
literal and frozen. All F² cross-channel feature pairs and original tick lags
1/2/4/8 participate. Odd/even moments use the original common-support CUDA
float64 arithmetic, with count/energy guards and masked-storage sanitization.
Zero initialization consumes no RNG. Changes in masks can still alter moment
estimates or remove information; this is not arbitrary deletion invariance.

The existing past-only adapter accepts the same Config/Input types, so data
loading, nine-feature order, timestamp alignment and scaling are shared:

```cpp
#include "cuwacunu/encoder/primary_encoder.h"
#include "cuwacunu/integration/configurable_encoder_input.h"
namespace enc = cuwacunu::encoder::primary;

auto c = enc::read_settings("config/encoder.conf");
auto input = cuwacunu::integration::configurable_encoder_input(batch, channel_ids, c);
auto scaler = enc::fit_train_scaler(train_input, c); // TRAIN only
auto scaled = enc::transform(scaler, input, c);
enc::Encoder model(c);
auto native = enc::native_embedding(model->encode(scaled), c);
enc::save_checkpoint(new_path, model, scaler, completed_updates);
auto loaded = enc::load_checkpoint(new_path, c.device, &c);
```

Use the same configured H and sampling interval for every loader source. The
file's interval 1 is a test default, not a market cadence. History,
feature count, semantic IDs and native width are fixed per model/checkpoint;
B may vary. Reconstruction and masks use the unchanged v20 decoder and sole
waveform Huber objective. Typed v20/v22 settings and checkpoints refuse one
another; immutable saves require a new filename. Optimizer/RNG resume is
outside this inference API.

Focused engineering, exclusively inside the managed `cuwacunu`
container after project instructions and coordination:

```bash
bash code/scripts/task.sh primary-v22 -j2 test-primary
```

The primary mmap witness follows the real loader mechanics with artificial
CSV candles and the full default network, then checks native CUDA output and
typed checkpoint reload. The content witness uses artificial values and one
CUDA update. It
checks all nine adapter coordinates, row/per-value masks, future exclusion,
nondivisible H129, initial v20-child bytes, active learned gradients, exact
fixed-tail preservation, poison/allmissing behavior and immutable typed
checkpoint reload. It imports no v21 model or research test, fits no classifier
and performs no market or synthetic quality experiment. Default build/test
targets include both primary v22 witnesses. V20 reference checks use their
explicit settings file. The original optional-sync receipt remains historical
evidence; the current selection record owns this default change and validation.
