# RPB-v20 configurable spectral encoder

For repository-wide evidence and open work, read
[current status](../../../doc/CURRENT_STATUS.md). This guide describes the
implemented v20 API; historical CLI examples in the RPB guide use a separate
workflow and checkpoint format.

This is the reusable backport from the clean `cuwacunu` integration project,
requested to keep representation research independent of market ingestion.
The v19 source and every historical checkpoint/card remain unchanged. RPB-v20
has CUDA engineering checks and a separate
[completed synthetic capacity experiment](../../../doc/CONFIGURABLE_CAPACITY_V1.md).
Three fresh TEMPO-5 cohorts at C3/H128/F9 compare W64/D32, W128/D32 and W64/D64
at512 updates. Compact W64/D32 is the continuation configuration; larger profiles
show no consistent classification gain at this budget. No market instance is trained.

Default geometry is `[B,3,128,9] -> [B,32]`. Configurable C/H/F/D are fixed per
constructed model and saved checkpoint; B can vary between calls. H must be
at least 17. Native D contains four fixed spectral coordinates for each of
C(C−1)/2 semantic channel pairs; the remaining coordinates are learned, so
D must be greater than four times the number of pairs. C1 has only learned
coordinates. The default keeps 20 learned + 12 fixed, without PCA afterward.

All F features reach the learned backbone and all F² feature pairs enter the
fixed mean. H need not be a multiple of P: H129/P8 pads the private backbone to
H136, masks the extra cells, retains original-H harmonics and crops decoder
output back to H129. Harmonic q1–8 frequencies are relative to the requested
history, so changed geometry still needs separate quality evaluation.

Portable headers under `include/embedding/encoders/raw_patch_bottleneck_mae/`:

- `configurable_spectral_config.h`: configuration, dimension constraints and widths.
- `configurable_spectral_relation.h`: mask-aware generalized spectral bank.
- `configurable_spectral_model.h`: CUDA model, masks and scaler helpers.
- `configurable_spectral_settings.h`: core file/settings binding.
- `configurable_spectral_checkpoint.h`: full resolved model/scaler inference archive.
- `configurable_spectral_checkpoint_io.h`: staged, atomic publication to a new filename.

The core stdlib-only file module lives under `code/shared/include/cuwacunu/config/`
and the original parser under `code/shared/include/piaabo/parse/`. See
[core config syntax/provenance](../../shared/config/README.md).

```cpp
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_settings.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/configurable_spectral_checkpoint.h"
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
// Caller supplies config_file, TRAIN/input tensors and a new output path.
auto c = rpb::read_configurable_spectral_settings(config_file);
auto scaler = rpb::fit_configurable_spectral_scaler(training_input, c);
rpb::ConfigurableSpectralModel model(c);
// Initialization-only round trip. For trained weights, save the model after
// CUDA training with its actual completed update count.
rpb::save_configurable_spectral_checkpoint(new_path, model, scaler, 0);
auto loaded = rpb::load_configurable_spectral_checkpoint(new_path, c.device, &c);
loaded.model->eval();
torch::NoGradGuard no_grad;
auto scaled = rpb::transform_configurable_spectral_input(loaded.scaler, input, loaded.config);
auto native = loaded.model->encode(scaled).z_contextual_global;
```

Edit [config/configurable_spectral.conf](config/configurable_spectral.conf).
Actual CUDA float32 execution is required. Use the configurable scaler helper
for nondivisible H, since preserved direct scaler methods validate the older
patch divisibility rule. Fit only on TRAIN; streaming fitting remains future work.
Training masks hide full patches, including a partial final patch. Channels with
fewer than three observed patches are ineligible; skip batches with no eligible
examples. The inference archive saves every configuration field, stable channel
IDs, scaler identity and update count; it does not claim optimizer/RNG resume.
Saving requires a new filename and never replaces an existing file. The managed
Linux runtime stages beside the destination and publishes atomically. Loading
and saving reject nonfinite weights and tensor shapes/dtypes/devices that do
not match the declared architecture. These checks preserve model arithmetic
and archive schema; they do not promise power-loss durability.

Inside the existing managed `cuwacunu_embedding` container:

```bash
source /etc/profile.d/embedding.sh
bash code/scripts/task.sh rpb-v20 -j2 test-rpb-v20
```

The four checks cover core config syntax, CUDA model/old-shape parity,
typed configuration/scaler/checkpoint round trip, and malformed model state
with overwrite protection. Tested shapes include C1–4,
H17/65/129/130/257, F3/F9 and D8/16/32/48/64. Each model test makes one disposable
optimizer update to verify CUDA gradients and unchanged literal frozen weights;
it does not repeat historical quality training. No maximum VRAM capacity is claimed.

`rpb-v20` builds those four check executables and `test-rpb-v20` runs them.
There is no standalone v20 training/serving CLI target in this module.

The new model remains an explicit API; historical CLI flags, defaults, readers,
research/evaluation drivers and measured reports keep their original semantics.
Market experiments declare their own observation grid, TRAIN statistics,
chronological evaluation and resource budget independently of this model API.
