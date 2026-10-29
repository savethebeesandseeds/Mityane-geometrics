# Numeric series dataloader

This directory is the time-series dataloader. It is the first form under
`code/dataloaders/`. Other forms get their own directories. This tree does not
include an encoder, an observation archive, or a training recipe.

The implementation is the cuwacunu_torch memory-mapped kline loader, in the
form already extracted by cuwacunu_dev. [PROVENANCE.md](PROVENANCE.md) is that
extraction's contract, including torch commit
`4abd3a4f544eeb5ab97a690281d54f13b27e29cd`. One sample is one instrument. Each
channel is one resolution of that instrument, with its own history and past
keys. A shorter channel is left-padded. The anchor grid is the finest cadence.

The public entry is `include/dataloader.h`, namespace
`mityane::data`. Build products go to
`/opt/mityane-geometrics/dataloaders/build/numeric-series`. From the managed
container:

```bash
make -C /mityane/code/dataloaders/numeric_series test
```

`read_loader_settings` in `include/settings_reader.h` fills `source_form_t` and
`channel_form_t` from `SOURCE_DEFAULTS`, `KLINE_SOURCE_SET`, and `CHANNEL_SET`.
A checked-in channel table is not in this directory. A published raw cache
opens by path and SHA256 through `add_raw_cache`. That call does not prepare
a CSV.

The causal patch bottleneck family places one edge sample per instrument onto
its shared anchor clock in `history_from_edge_samples`. That adapter lives with
the encoder.
