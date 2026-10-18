# Configurable spectral encoder — RPB-v20

RPB-v20 retains the selected v19 mechanism and generalizes its input and native
embedding sizes. V19 and its six confirmed checkpoints remain frozen in
`../rpb_v19`; this is a separate design, with no measured market accuracy.

RPB-v22 is now the primary application encoder. This v20 API remains its
reusable base and an explicit reference. Use the
[primary guide](../rpb_content/README.md) for ordinary application integration.

The additive [streaming TRAIN support](../../../doc/STREAMING_TRAINING_SUPPORT.md)
fits the existing scaler without retaining the full TRAIN tensor. It also
documents the separate v22 optimizer-resume archive; this v20 inference format
and its portable model headers remain unchanged.

The public contract is float32 CUDA `[B,C,H,F] -> [B,D]`. Batch size can change
between calls. C/H/F/D are fixed by each constructed model and checkpoint.
The market adapter supplies all nine ordered numeric kline features; timestamps
are sampling metadata. The reusable model can accept another positive F.

The default is C3/H128/F9/P8/D32: 20 learned shape coordinates followed by 12
fixed spectral coordinates. For C channels there are C(C−1)/2 semantic pairs,
each using four fixed coordinates. D must exceed 4 times that pair count;
the remainder is learned. C1 uses all D coordinates for learned shape.

H must be at least 17, because this design retains eight harmonic bins.
Nondivisible history lengths are supported: H129/P8 uses a private H136
backbone, with added cells masked false; spectral calculations and loss use
the original H, and reconstruction is cropped to H129. Harmonics cover q1–8
cycles across the configured window, so changing H changes physical frequencies.
This is not a model accepting variable H between calls or a frequency study.

Read reference model settings from [encoder_v20.conf](../../../config/encoder_v20.conf), including
`HISTORY_LENGTH`, `INPUT_WIDTH`, `EMBEDDING_SIZE`, `CHANNEL_COUNT`, stable
`CHANNEL_IDS`, and `SAMPLING_INTERVAL`. Fixed architecture switches cannot be
silently changed through this file. Unknown settings, duplicate names, invalid
dimensions and CPU devices are rejected. The market adapter requires F9 and
one common, regular timestamp grid for all channels.

```cpp
#include "cuwacunu/encoder/configurable_artifact.h"
#include "cuwacunu/integration/configurable_encoder_input.h"
namespace enc = cuwacunu::encoder::configurable;
auto c = enc::read_settings("config/encoder_v20.conf");
// For every source: dataset.add_dataset(csv, c.history_length, future, "none", false);
auto input = cuwacunu::integration::configurable_encoder_input(batch, channel_ids, c);
auto scaler = enc::fit_train_scaler(train_input, c); // TRAIN only
auto scaled = enc::transform(scaler, input, c);     // CPU statistics, CUDA model input
enc::Encoder model(c);
auto native = enc::native_embedding(model->encode(scaled), c);
enc::save_checkpoint(new_checkpoint_path, model, scaler, training_updates);
auto loaded = enc::load_checkpoint(new_checkpoint_path, c.device, &c);
```

Use the transform helper, rather than calling the preserved scaler's methods
with a nondivisible logical config. Its validation copy uses patch length 1;
the model and training masks continue using the actual configured patch length.

Artificial training masks hide complete patches, including a partial final
patch. A channel needs at least three observed patches for reconstruction
eligibility. Check `eligible_example_count` and skip ineligible batches. Natural
per-value gaps and NaNs are handled by the adapter's observed mask and zeroing.
The fixed spectral mean includes every F² feature pair with coefficient 1/F²;
its literal weights stay frozen through training and loading.

The inference checkpoint binds the tag, architecture/layout, every resolved
model dimension and numerical setting, semantic channel IDs, scaler identity
and update count. Loading reconstructs the model from saved geometry; optional
expected settings reject mismatches even when H shares the same padded size.
CUDA ordinal may change at load time. The format does not save optimizer/RNG
state or replace complete training-run provenance. Saving requires a new filename
and never replaces an existing file. A complete same-directory staging file is
published atomically in the managed Linux runtime. Loading and saving reject
nonfinite weights and model tensors whose shapes/dtypes/devices disagree with
the declared architecture. Publication does not promise power-loss durability.

Focused CUDA checks cover masks, all features, C1–4, H17–257, D8–64, a disposable
optimizer update, checkpoint round trips and exact v19 arithmetic at its old
shape. They establish compatibility and execution, not generalized quality.
Reusable headers/settings/checkpoints and tests also live in the standalone
embedding repository; see [the backport record](../../../doc/ENCODER_BACKPORT.md).
