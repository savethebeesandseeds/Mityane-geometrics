# Shared market data loading

This is a self-contained extraction of the memory-mapped loader from
`cuwacunu_torch` at commit `4abd3a4f544eeb5ab97a690281d54f13b27e29cd`.
Its source/cache preparation, causal anchor lookup, window construction,
padding, collation and sampling algorithms are retained. The original graph
label, merged graph/source contract overloads, exchange JSON APIs, statistics
helpers, and terminal progress UI are omitted. Logging is a small stderr helper.
Neither sibling repository is needed to compile or run it.

Read-only mappings are owned by `ReadOnlyMappedFile` in
`read_only_mapped_file.h`. `MemoryMappedDataset` uses this same storage owner.
CSV preparation remains a separate operation from mapping a cache that is
already published.

The public namespace is `mityane::data`. Include `dataloader.h` from this
tree's `include/`. The library has one compilation source: `src/record_types.cpp`.
The remaining implementation is header based. Add this tree's `include/` and LibTorch includes,
and link the same LibTorch libraries as the encoder. POSIX memory mapping means
builds and execution belong in the Linux development container.

## Data contract

`sample_t` is an alias of the imported `edge_sample_t`:

| Field | Batched shape | Meaning |
|---|---|---|
| `features` | `[B,C,H,F]` | Past input window |
| `mask` | `[B,C,H]` | Valid source rows; padding and missing rows are false |
| `past_keys` | `[B,C,H]` | Actual source timestamps |
| `future_features` | `[B,C,Hfuture,F]` | Optional future targets, separate from encoder input |
| `future_mask`, `future_keys` | `[B,C,Hfuture]` | Future row validity and timestamps |

Samples returned directly by a dataset lack the leading batch dimension.
Klines retain nine ordered features: open, high, low, close, volume,
quote volume, trades, taker buy base volume, taker buy quote volume.
The historical generic record types are retained to preserve the loader's
template implementation; this integration initially uses klines.

CSV input uses the existing twelve-column Binance kline format. Numeric tokens
must be consumed completely; ordinary surrounding whitespace is accepted.
Millisecond
timestamps are canonicalized from sufficiently large microsecond values. A
buffered raw binary cache retains the latest duplicate and fills regular-grid
gaps with invalid records. CSV preparation with policy `none` stores the raw
klines in `<stem>.cache.bin`. Policy `log_returns` writes a separate normalized
cache. Mapping a published raw cache does not write either file and does not
change the stored values.
Cache files are native-layout, native-endian snapshots, not portable interchange
files. Preparation writes private files beside the destination and publishes a
complete cache with atomic replacement; a failed rebuild preserves the previous
cache. A persistent `<raw-cache>.lock` serializes raw/normalized writers. Existing
mapped readers retain their original snapshot across refresh. Fresh cache reuse
requires no directory write access. Do not directly mutate/truncate mapped files;
the lower-level `normalize_binary_file` operation is explicitly in place and is
only used on a private staging copy during cache preparation. Keep the source
CSV immutable during preparation, and create/reset readers after a refresh when
the new snapshot is needed.

In this torch-derived loader, one edge is one instrument and each channel is
one sampling resolution of that edge. Many instruments are many samples.
All channels in one edge dataset and all samples in one collated batch must use
the same raw/normalized state. Mixed states are rejected, so already normalized
values cannot silently cross the encoder's raw-input boundary. Horizon sizes are
checked before materialization; causal source lookup rejects anchors before the
first record. Do not add datasets while a loader is iterating in worker threads.

The global anchor grid defaults to the finest source cadence. Each channel uses
its latest available source key at or before the anchor and retains its native
timestamps. The loader does not interpolate or resample different cadences.
Different history lengths are left-padded with false masks; future windows are
right-padded. An encoder adapter must explicitly validate or align this geometry.

## API

Use `MemoryMappedEdgeDataset<kline_t>` and `add_dataset(csv_path, history_length,
future_length, "none", force_rebuild_cache)` for an explicit CSV dataset.
`read_loader_settings` fills `source_forms` and `channel_forms` from
`SOURCE_DEFAULTS`, `KLINE_SOURCE_SET`, and `CHANNEL_SET`. A published raw
CACHE88 opens through `add_raw_cache` or `open_raw_caches` with its path and
SHA256. That door does not call `add_dataset`. Alternatively populate
`source_materialization_request_t.source_forms` and `.channel_forms`,
then call `make_sequential_edge_dataloader<kline_t>(instrument_signature, request,
options)` or `make_random_edge_dataloader<kline_t>(...)`.
`edge_dataloader_options_t` sets batch size (default 64), workers (default 0), and
cache rebuilding. Only requested windows/batches are materialized into tensors;
the loader does not expand the entire dataset into overlapping tensors in RAM.
No maximum tensor capacity has been benchmarked in this repository yet.

Feature output defaults to Float32. Consumers that require original record
precision can construct `MemoryMappedEdgeDataset<kline_t>(torch::kFloat64)` or
set `edge_dataloader_options_t::feature_dtype = torch::kFloat64` before using a
source factory. A single-file dataset accepts the dtype as its fourth constructor
argument; the lower-level dataset/loader factories accept it as their last
argument. Only Float32 and Float64 are accepted. Float64 copies the original
double feature fields directly, including future fields, rather than upcasting
rounded Float32 tensors. Masks remain Boolean and timestamps retain their actual
Int64 or Float64 key grids. This option does not change normalization policy.

The preserved iterator yields a vector of window samples for each batch.
Use `sample_t::collate_fn(samples)` to obtain the batched tensors in the table:

```cpp
auto loader = edge_dataloader_t<kline_t>(dataset, {.batch_size = 8, .workers = 0});
for (const auto &samples : loader) {
  const auto batch = sample_t::collate_fn(samples);
  // batch.features is [B,C,H,9]; batch.mask is [B,C,H].
}
```

After exhausting an epoch, another `begin()` starts the next sampler epoch and
reuses the LibTorch worker pool. Explicit `reset()` also works before iteration
or after a partial epoch: it abandons that epoch, waits for already scheduled
work, and replaces the LibTorch loader and its workers. It does not fetch the
remaining dataset. Shared mapping owners keep the source snapshot alive without
reopening or copying the mapped file. All previous iterators, batch references
and references returned by `inner()` are invalid after reset. Start iteration
again with `begin()`; set `torch::manual_seed(...)` before that call when a
repeatable random-sampler order is required. Reset alone does not start an epoch
or draw a new permutation. This is an epoch restart, with no mid-epoch resume
or saved random-generator state API.

## Verification

`tests/test_memory_mapped.cpp` adapts the original eleven storage tests. It checks
cache freshness without rebuilding frozen inputs, feature order, timestamp
canonicalization, normalization, causal anchors, catchable storage errors, zero
future horizons, padding/ranges, CSV sanitation, collation, and loader batching.
Additional sampler tests check sequential and seeded random coverage, repeated
epochs, reset before/full/partial epochs, and short final batches with zero and
two workers. A purpose-built dataset verifies that both workers execute `get()`
off the main thread, delays the first batch behind later work while checking
ordered delivery, and bounds partial-reset reads to existing prefetch work.
The same public sequential/random wrappers are exercised on mapped kline data.
Compile it with `src/record_types.cpp`, the include root above and LibTorch; the
root Makefile supplies the managed-container command.

`tests/test_settings_and_raw_cache.cpp` checks the settings reader and the
published raw-cache door, including a wrong SHA256 and a normalized filename.
`tests/test_data_safety.cpp` adds failure-safe publication, live raw/normalized
mapping refresh, concurrent writer serialization, mixed-normalization guards,
causal anchor and horizon validation, strict numeric parsing, and shared feature
statistics when batch size equals feature width. These are storage correctness
checks, not market throughput or encoder quality measurements.
