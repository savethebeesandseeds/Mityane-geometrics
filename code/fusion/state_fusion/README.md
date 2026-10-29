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

The encoder is a Set Transformer [1], written here against the LibTorch C++
API from PyTorch 2.6.0 [4]. Two induced set attention blocks (ISAB) use 16
learned inducing points. Pooling by multihead attention (PMA) uses one seed
vector and returns `[batch, seeds, d_model]`. The width is 64, with 4 heads
and a feedforward width of 128. `Config` accepts a width from 64 through 128,
inducing points from 16 through 64, and one or two seeds. Dropout defaults to
0 so evaluation is deterministic. The multihead block follows the
residual-then-LayerNorm form of Vaswani et al. [3], which is the MAB used by
Lee et al. [1].

There is no positional or temporal feature in this version. The set output of
an ISAB is equivariant. The seed output of PMA is invariant. The weights are
created in the constructors. Nothing in this package is loaded from a
checkpoint.

The official implementation [2] is an architectural reference. That repository
is MIT-licensed. This package does not copy its source and does not port its
weights. It does not use TabPFN [7] or any other pretrained tabular model.

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
key-padding mask drops it. In PyTorch, and therefore in this LibTorch build, a
true entry in that mask ignores the key [4]. The record is not replaced by a
zero token. Attention does not add a zero slot. Each set needs at least one
observed record, because a fully masked key row makes softmax undefined.

The default categorical cardinalities are synthetic bounds for this package.
They are not a domain ontology. `semantic_width` is configuration; the
synthetic panels use 16.

## Query

`query(state, query_batch)` answers one or more questions against a market
state already returned by `encode`. The set is not encoded again. `state` is
`[batch, seeds, d_model]`. `query_batch` is a `SetBatch` whose set dimension
is the query dimension, so a later projection fills the same fields it fills
for records and calls `query` without changing this package.

Each query row uses the shared token encoder. Scalars keep their observed
flags, categories use the same embeddings, and a missing semantic vector uses
the same learned absent token. A hole is ignored, including a non-finite
stored value. An observed zero stays a zero. A query row with
`record_observed` false is the all-absent token and is still answered; it is
not dropped, because a query is not a key.

The query token is the query side of one multihead attention block. The keys
are the pooled state. The block is the same MAB primitive as the rest of the
package. It is not a second induced-set encoder. The answer is
`[batch, queries, d_model]`.

Mixture density sampling from that answer is deferred. This package does not
draw a sample, and it does not add further PMA seeds or separate readout
surfaces. The masked-reconstruction loop does not train the query block.
The query checks below are structural.

## CUDA

Attention and the training path run on CUDA. CPU batches are rejected.

The managed image provides CUDA toolkit 12.4, V12.4.131 [5], and the LibTorch
2.6.0 CUDA build [4] at `/opt/cuwacunu_embedding/libtorch`. This package has no
`.cu` files. GPU kernels come from that LibTorch build. `CMakeLists.txt`
requests Torch 2.6 and fails if `libtorch_cuda` is absent. The image has no
`cmake` binary. The Makefile compiles with g++ using that include path, C++17,
`_GLIBCXX_USE_CXX11_ABI=1`, and `-ltorch_cuda`.

Products go to `/opt/mityane-geometrics/fusion/build/state-fusion`.

```text
make -C /mityane/code/fusion/state_fusion test
make -C /mityane/code/fusion/state_fusion train
```

`test` runs the GPU invariance and hole checks, the fidelity checks, the query
checks, and then the training binary. The fidelity binary replays the paper wiring on the
package blocks: inducing points are the first ISAB queries, PMA seeds are the
pooling queries, and each residual is applied before LayerNorm. It also
requires a non-zero finite gradient on the inducing points, the seeds, and
both absent tokens, and it rejects quadratic growth from set size 16 to 256.
`train` runs masked reconstruction on one fixed synthetic panel and prints
`device=cuda:0`, the LibTorch version, `initial_masked_mse`, and
`final_masked_mse`. It stops if the final error is not finite and below half
the initial error. The optimizer is Adam [6].

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

The query checks ran on the same GPU. A place-and-kind query was unchanged by
permuting the encoded set (`permutation_max_abs=4.76837e-07`, and
`two_seed_permutation_max_abs=3.57628e-07`). Editing a query hole left the
answer unchanged. An observed zero moved the answer by `0.209496`, and the
place-and-kind query differed from an empty query by `3.25368`. A cloned
market state reproduced the answer exactly. These checks do not train the
query block.

## References

[1] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and
Yee Whye Teh. Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks. Proceedings of the 36th International
Conference on Machine Learning, PMLR 97:3744–3753, 2019.
<https://proceedings.mlr.press/v97/lee19d.html>. Preprint
[arXiv:1810.00825](https://arxiv.org/abs/1810.00825). ISAB and PMA are defined
in this paper. The proceedings page links the same code repository as [2].

[2] Juho Lee. Official PyTorch implementation of [1].
<https://github.com/juho-lee/set_transformer>. The repository states the MIT
license at
<https://github.com/juho-lee/set_transformer/blob/master/LICENSE>. Pattern
reference only. This package does not copy that source and does not load its
weights.

[3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention Is All You Need.
Advances in Neural Information Processing Systems 30, 2017.
<https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>.
Preprint [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). The MAB residual
and LayerNorm order follows this block.

[4] PyTorch 2.6.0, including the LibTorch C++ API linked by this package.
<https://github.com/pytorch/pytorch/releases/tag/v2.6.0>.
`torch.nn.MultiheadAttention` documents the key-padding mask: a true value
ignores that key.
<https://pytorch.org/docs/2.6/generated/torch.nn.MultiheadAttention.html>.

[5] NVIDIA CUDA Toolkit 12.4. The managed image reports compiler V12.4.131.
<https://docs.nvidia.com/cuda/archive/12.4.1/>.

[6] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization.
International Conference on Learning Representations, 2015.
[arXiv:1412.6980](https://arxiv.org/abs/1412.6980).

[7] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter.
TabPFN: A Transformer That Solves Small Tabular Classification Problems in a
Second. International Conference on Learning Representations, 2023.
[arXiv:2207.01848](https://arxiv.org/abs/2207.01848). The later tabular
foundation model is Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun
Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank
Hutter, Accurate predictions on small data with a tabular foundation model,
Nature, 9 January 2025, <https://doi.org/10.1038/s41586-024-08328-6>. The 2023
preprint names <https://github.com/automl/TabPFN>, which now resolves to
<https://github.com/PriorLabs/TabPFN>. This package does not load that code or
those weights.
