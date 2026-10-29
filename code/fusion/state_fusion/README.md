# State fusion models

Hx21 owns this directory. The family was named on 29 October 2026. It fuses an
unordered collection of records into one fixed market-state tensor. A later
Nonverba projection can fill `SetBatch` from domain records and call `encode`
without changing this package.

This is not the State Representation Network in
`code/embeddings/state_representation/`. That network maps one observation
embedding to a predictive state. It is not the Temporal Fusion Transformer in
the sibling directory. A result here does not close Hx11 or Hx20.

## Architecture

The encoder is a Set Transformer (Lee et al., ICML 2019), written here against
LibTorch. Two induced set attention blocks (ISAB) use 16 learned inducing
points. Pooling by multihead attention (PMA) uses one seed vector and returns
`[batch, seeds, d_model]`. The width is 64, with 4 heads and a feedforward
width of 128. `Config` accepts a width from 64 through 128, inducing points
from 16 through 64, and one or two seeds. Dropout defaults to 0 so evaluation
is deterministic.

There is no positional or temporal feature in this version. The set output of
an ISAB is equivariant. The seed output of PMA is invariant. The weights are
created in the constructors. Nothing in this package is loaded from a
checkpoint.

`juho-lee/set_transformer` is an architectural reference. That repository is
MIT-licensed. This package does not copy its source and does not port its
weights. It does not use TabPFN or any other pretrained tabular model.

## Input and holes

Each set element is one record. `SetBatch` carries:

- scalar features and a boolean observed flag for each feature
- categorical ids for place, kind, record type, side, and currency, each with
  its own observed flag
- an optional semantic vector and one observed flag
- one observed flag for the whole record

A false flag is a hole. The stored number under that flag is ignored, whether
it is zero or non-finite. `torch::where` selects the observed value or the
absent path, so a non-finite hole is not an operand. An observed zero stays a
zero and changes the state. Missing scalars are concatenated with their flag
before the projection. A missing category uses a learned absent embedding. A
missing semantic vector uses a learned absent token drawn from a small normal
distribution, so the absent token is not the zero vector. A fully unobserved
record is not read. Its token is that absent-field encoding, and the
key-padding mask drops it. It is not replaced by a zero token. Attention does
not add a zero slot. Each set needs at least one observed record, because a
fully masked key row makes softmax undefined.

The default categorical cardinalities are synthetic bounds for this package.
They are not a domain ontology. `semantic_width` is configuration; the
synthetic panels use 16.

## CUDA

Attention and the training path run on CUDA. CPU batches are rejected.

The managed image provides CUDA toolkit 12.4 (V12.4.131) and the LibTorch
2.6.0 CUDA build at `/opt/cuwacunu_embedding/libtorch`. This package has no
`.cu` files. GPU kernels come from that LibTorch build. `CMakeLists.txt`
requests Torch 2.6 and fails if `libtorch_cuda` is absent. The image has no
`cmake` binary. The Makefile compiles with g++ using that include path, C++17,
`_GLIBCXX_USE_CXX11_ABI=1`, and `-ltorch_cuda`.

Products go to `/opt/mityane-geometrics/fusion/build/state-fusion`.

```text
make -C /mityane/code/fusion/state_fusion test
make -C /mityane/code/fusion/state_fusion train
```

`test` runs the GPU invariance and hole checks, the fidelity checks, and then
the training binary. The fidelity binary replays the paper wiring on the
package blocks: inducing points are the first ISAB queries, PMA seeds are the
pooling queries, and each residual is applied before LayerNorm. It also
requires a non-zero finite gradient on the inducing points, the seeds, and
both absent tokens, and it rejects quadratic growth from set size 16 to 256.
`train` runs masked reconstruction on one fixed synthetic panel and prints
`device=cuda:0`, the LibTorch version, `initial_masked_mse`, and
`final_masked_mse`. It stops if the final error is not finite and below half
the initial error.

## Synthetic objective

`make_panel` builds the panels in C++. Records in one set share a latent
scalar vector, plus noise, real zeros, field holes, record holes, and absent
semantic vectors. A fraction of the originally observed scalars is withheld.
The visible batch stores garbage in holes and in withheld cells, and those
cells are marked unobserved. The loss is the mean squared error on the
withheld cells only.

These panels validate the pipeline. They are not a market, and the printed
error is not a forecast score.

## Recorded check

On 29 October 2026, `make test` ran in the managed container on an NVIDIA RTX
A2000, CUDA toolkit 12.4, and LibTorch 2.6.0. The invariance test passed on
`cuda:0`: `permutation_max_abs=3.57628e-07`, hole, non-finite-hole, and
unobserved-record gaps of 0, `padding_max_abs=4.76837e-07`,
`two_seed_permutation_max_abs=7.15256e-07`, and
`observed_zero_gap=0.0210351`. The training binary then ran 400 Adam steps on
one fixed synthetic panel and printed `initial_masked_mse=1.25887` and
`final_masked_mse=1.85048e-06`. The fidelity binary then passed on the same
GPU: the block wiring matched exactly, absent-token and inducing gradients
were non-zero, and encoder time from set size 16 to 256 was 3.16 ms, 2.84 ms,
and 4.68 ms per call. The per-record ratio was 0.092, against about 16 for
quadratic growth. Fixed CUDA launch cost still dominates that range. CMake
was not executed. These checks are pipeline validation on synthetic panels.
