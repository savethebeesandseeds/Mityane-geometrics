# Typed observation-window datasets

[observation_windows.h](include/embedding/data/observation_windows.h) defines
an encoder-independent C++/LibTorch dataset boundary in
`embedding::data::observation_windows`. It carries genuine raw observations,
support, timeframe channels and explicit per-row temporal cutoffs. One window
is one instrument. Each channel is one time resolution of that instrument, and
its id is a cuwacunu_torch kline-interval ordinal
(`embedding/encoders/timeframe_channel.h`). No encoder
weights, fitted preprocessing, targets or market-specific schema are embedded.

The archive kind is **`observation_window_dataset_v1`**. Use the C++ `Dataset`,
`write()`, `read()` and checked `subset()` APIs; no Python conversion is needed.
The [CPB training example](../../embeddings/encoders/causal_patch_bottleneck_mae/family/examples/TRAINING.md)
shows one consumer. Other consumers must declare their own shape, grid and
support requirements.

## Schema

All tensors are stored/read on CPU. B is the number of source windows, C the
number of channels per window, N the supplied raw ticks and F the feature width.
All four dimensions must be positive.

| Archive field | Type / shape | Meaning |
| --- | --- | --- |
| `artifact_kind` | U8 text | Exact kind above |
| `source_identity` | U8 text | Explicit source/manifest identity |
| `split_identity` | U8 text | Explicit split/manifest identity |
| `role` | U8 text | `TRAIN`, `VALIDATION` or `TEST` |
| `values` | F64 `[B,C,N,F]` | Genuine raw observation values, before parent/model F32 rounding |
| `observed` | Boolean `[B,C,N,F]` | Cellwise support mask |
| `channel_ids` | I64 `[C]` or `[B,C]` | Timeframe channel IDs, unique within each row. One window is one instrument |
| `timestamps` | F64 `[N]` or `[B,N]` | Shared or row-specific finite timestamp grid |
| `cutoffs` | F64 `[B]` | Explicit finite cutoff per source row |
| `row_ids` | I64 `[B]` | Unique stable IDs within this source dataset |
| `sampling_interval` | F64 scalar | Finite positive interval, in the same units as timestamps/cutoffs |

Source/split/role strings must be nonempty, NUL-free and at most 1024 bytes.
IDs are torch kline-interval ordinals, not array positions, class labels or
instrument ordinals. Consumers map timeframe channels against their own
configuration. The timestamp grid is one shared anchor. It does not mean every
channel is that interval. N is the supplied history length;
it need not equal an encoder's retained history H. The consumer may retain a
causally cropped window, including masked padding when its contract allows it.

The memory-mapped kline loader in `code/dataloaders/numeric_series/` is a separate
source. One sample there keeps a history length and a past-key row per
resolution. This observation archive still stores one shared timestamp grid,
so it does not carry those per-channel keys.

`validate()` checks schema, metadata, finite declared time fields and uniqueness.
It deliberately does **not** inspect raw value or mask contents across the
whole supplied history. A consumer must first crop at each cutoff, then validate
its retained observed values and support. CPB additionally requires an ordered
uniform grid, an aligned in-range cutoff, matching sampling interval and the
configured semantic channels/features. Dataset schema validity alone does not
establish those consumer-specific constraints or useful training support.

A `TRAIN` role is a caller declaration, not independent proof of population or
split membership. Applications must bind the real source/split manifests and
verify their sampling/calendar/retention policy before fitting or training.
The CPB CLI accepts only declared TRAIN archives and binds their exact bytes and
identities; it does not infer true membership from the observations.

## C++ creation and reading

```cpp
#include "embedding/data/observation_windows.h"
namespace ow = embedding::data::observation_windows;

// Supply tensors with the schema above and identities from the actual manifests.
ow::Dataset dataset{
    raw_f64, observed_bool, semantic_ids_i64, timestamps_f64,
    cutoffs_f64, unique_row_ids_i64, sampling_interval,
    source_manifest_identity, split_manifest_identity, "TRAIN"};
ow::write(new_archive_path, dataset);
auto restored = ow::read(new_archive_path);
auto batch = ow::subset(restored, {0, 3, 0}); // with-replacement selection
```

`subset()` checks index bounds and retains source row IDs and metadata. Repeated
indices are allowed for sampled batches; this does not change the requirement
that a complete source dataset have unique `row_ids`. A repeated-row batch is
not a new independently validated source dataset.

`write()` refuses an existing destination. It writes a reserved same-directory
temporary payload and publishes by hard link without replacing an existing
file, including one created concurrently. The filesystem must support that
publication operation. If writing/publication fails, preserve any pending
payload for inspection; it is not a completed dataset. The archive bytes are
the content identity: consumers should hash/bind them and check for changes
while loading or executing.

## Bounded synthetic fixture

Build and run inside the existing managed container, through the root launcher:

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/task.sh', 'observation-window-demo-2026-10-22',
  'build-observation-window-fixture'
)
.\container.ps1 -Action exec -Command @(
  '/opt/mityane-geometrics/embeddings/build/observation-window-demo-2026-10-22/observation_windows_fixture',
  '--output', '/mityane/.temp/embeddings/runs/observation-window-demo-2026-10-22/synthetic-TRAIN.pt',
  '--channels', '2', '--history', '40', '--features', '2', '--rows', '8',
  '--sampling-interval', '1000'
)
```

Use an unused output filename when repeating this command. The fixture supplies
timeframe channel IDs 11 and 18 (`4h` and `1M`), row IDs 0–7, a shared grid ending at 39000 and corresponding
per-row cutoffs. It generates only artificial TRAIN observations and does not
read real application data or establish quality. Generation is bounded to ten
million cells; it is not a production ingestion or sharding tool.

The current implementation reads the full archive into memory before selecting
batches. Production readers, streaming/sharding, actual manifest membership and
durable data orchestration remain application/data-workstream responsibilities.
Keep raw archives and generated outputs under ignored `.temp/`, preserving
applicable source access, retention and license conditions.
