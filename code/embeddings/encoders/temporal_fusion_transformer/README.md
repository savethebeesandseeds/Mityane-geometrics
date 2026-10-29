# Temporal Fusion Transformer

Hx20 owns this forecasting family. The library is a C++17 Temporal Fusion
Transformer on LibTorch. It forecasts one target over a horizon as quantiles.
It also returns variable-selection weights and causal attention weights.

This is a model family, kept with the other model-owned code under
`code/embeddings/encoders/`. It is not an observation-embedding encoder, and it
does not use the patch-bottleneck channel contract. Hx11 still owns operations
that forecast latent states. A result from this library does not close that
operation.

## Contract

Static, past and known-future inputs stay in separate groups. Each group can
hold continuous values, categorical ids and, when its width is positive, one
pooled vector. Future targets are an argument of the loss. They are not an
input of `forward`.

A boolean time mask may mark a hole. A false step holds the LSTM state, is not
an attention key, and is left out of the loss. An observed zero still advances.
Non-finite values are rejected. Scaling, splitting and feature provenance stay
with the caller. The library does not fit preprocessing.

The public headers live in `include/praesidium-humanitatis/`. The C++ namespace
remains `praesidium_humanitatis`. That name is the imported interface. This
directory is the maintained source identity.

Include `praesidium-humanitatis/tft.h` and link the CMake target
`praesidium_humanitatis::tft`. `quantile_loss` scores the returned predictions.
`checkpoint.h` saves and restores configuration, Adam state and progress.
`evaluation.h` reports forecast metrics against a persistence baseline.
`calibration.h` fits and applies per-horizon intervals on a separate labeled
split. `Config::ordered_quantiles` keeps quantile forecasts from crossing.

Checkpoint format version 4 stores the pooled-vector widths. Older checkpoints
load with those widths at zero.

## Build

The managed container already provides the compiler and LibTorch at
`/opt/cuwacunu_embedding/libtorch`. That image has no CMake, so the Makefile
compiles the imported sources with g++. `CMakeLists.txt` remains the imported
description of the same targets. From the container:

```bash
make -C /mityane/code/embeddings/encoders/temporal_fusion_transformer test
```

`make test` runs the CPU suites and the demo continuation check. Products go to
`/opt/mityane-geometrics/embeddings/build/temporal-fusion-transformer`.
`make test-cuda` runs the same suites with `--device cuda`. A skipped CUDA
test does not show that GPU execution works.

The examples `tft_demo`, `tft_uncertainty`, `tft_compare_reference` and
`tft_train_electricity` build with the library. Running one is a chosen
experiment, not part of `make test`.

## What this import does not claim

The [provenance](PROVENANCE.md) names the two source trees. Upstream synthetic
checks and the unfinished Electricity comparison belong to those programs.
They are not measurements of this tree. The replication directory keeps that
comparison harness, including its Python reference scripts. Its README and
`verify.sh` still name the upstream layout. They are not the build entry here.

The architecture follows Lim, Arik, Loeff and Pfister,
[Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
Forecasting](https://arxiv.org/abs/1912.09363), checked against the authors'
reference implementation. This C++ library is not an official release by the
authors. The MIT license is in [LICENSE](LICENSE).
