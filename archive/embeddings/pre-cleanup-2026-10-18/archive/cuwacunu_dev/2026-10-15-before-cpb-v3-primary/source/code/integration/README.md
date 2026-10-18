<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../LICENSE. -->

# Selected causal encoder integration

The application uses **CPB-v1.market-01**, native36, through
`cuwacunu/encoder/encoder.h` in namespace `cuwacunu::encoder::primary`.
`primary_encoder.h` includes that same API. There is one active encoder family.
The complete evaluated implementation is owned by
`code/encoders/causal_patch_bottleneck_mae/versions/v1/`.

`config/encoder.conf` contains its exact geometry and CUDA:0 settings.
`config/encoder_runtime.conf` selects its typed checkpoint by path, instance
and SHA-256. Paths in the runtime file resolve relative to that file.
`primary::load_selected()` rejects a missing, altered or incompatible checkpoint.
It loads the existing frozen scaler and bank normalizer; it does not fit either.

The input contract is raw CPU Float64 `[B,3,N,9]`, Boolean observations,
actual common timestamps and semantic channel IDs. Use the loader's explicit
Float64 option when constructing samples: converting rounded Float32 back to
Float64 does not recover the source doubles. Normalized caches are rejected.
Masks may be per row or per feature. Retained observed NaN/Inf values are errors;
unavailable coordinates require explicit false observation masks.

Every encoding requires explicit CPU Float64 `[B]` cutoffs in the timestamp
units. `causal_encoder_input.h` passes raw history to the version's cutoff
boundary. Future sample fields never enter the encoder. Later observations
and masks are cropped before value validation, scaling or bank construction.
Do not reuse a final vector as earlier-time embeddings: call with each time's
own cutoff. The result provides `native` `[B,36]`, `valid` `[B]` and `cutoffs`.
Invalid rows retain their validity marker; downstream code must honor it.

For a raw market cache, the convenience loader reads exactly three semantic
channels in BTC/ETH/SOL order (IDs 0,1,2), H128, zero future horizon, and requires
the cutoff to be an actual common candle-close key. CSV arguments require an
existing fresh raw cache. It opens existing data read-only.

```cpp
#include "cuwacunu/integration/causal_market_input.h"
namespace primary = cuwacunu::encoder::primary;
namespace integration = cuwacunu::integration;
auto loaded = primary::load_selected();
const int64_t close_ms = 1706486399999LL;
const std::array<std::filesystem::path, 3> sources = {
    "data/mkt-hist-1/csv/BTCUSDT-1m-2024-01.cache.bin",
    "data/mkt-hist-1/csv/ETHUSDT-1m-2024-01.cache.bin",
    "data/mkt-hist-1/csv/SOLUSDT-1m-2024-01.cache.bin"};
auto sample = integration::load_market_history(sources, close_ms, loaded.config);
auto ids = torch::tensor(loaded.config.level.channel_ids, torch::kInt64);
auto cutoff = torch::tensor({static_cast<double>(close_ms)}, torch::kFloat64);
auto result = integration::encoder_input(loaded, sample, ids, cutoff);
```

Inside the existing `cuwacunu` container:

```bash
bash code/scripts/task.sh causal-app -j2 test
bash code/scripts/task.sh causal-app encode-history ARGS='--cutoff 1706486399999 --source data/mkt-hist-1/csv/BTCUSDT-1m-2024-01.cache.bin --source data/mkt-hist-1/csv/ETHUSDT-1m-2024-01.cache.bin --source data/mkt-hist-1/csv/SOLUSDT-1m-2024-01.cache.bin'
```

The CLI prints instance, cutoff, validity and the 36 native coordinates.
No post-encoder PCA is applied. Research versions and retired callers are in
the [source archive](../../doc/CAUSAL_PRIMARY_INTEGRATION.md); historical quality
reports remain evidence for their original experiments.
