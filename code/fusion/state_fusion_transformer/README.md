# State fusion transformer

Hx21 owns this directory. The family was named on 29 October 2026. It is the
set-to-state machinery for Nonverba.

Nonverba fills `SetBatch` from its records, calls `encode` and `query`,
trains the weights, and owns the checkpoint. This package supplies that
machinery and ships no trained weights. The records and the query answers
stay with Nonverba.

`encode` fuses one unordered set of records into one fixed state,
`[batch, seeds, d_model]`. Order does not change that state. `query` asks
in the same record language against the cached state and does not encode
the set again.

The reading this family is built for is a set of evidence, performance,
attestation, or outcome records, fused once and then queried. A query can
support a track record, a decaying vote, a dispute check, or a verification.
Which records those are, and what each query returns, remains open in this
package. Binding them is the remaining handoff.

The synthetic panels, the capacity protocol, and the LEGO study only check
that path: missing values, permutation invariance, and a query against a
cached state. They are checks. They are not the purpose of the family.

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
created in the constructors. `save_checkpoint` writes the config and those
weights. `load_checkpoint` restores them onto CUDA. This package does not
ship a trained checkpoint.

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

The categorical columns are named place, kind, record type, side, and
currency. Those names are the slot labels used by the checks in this package.
They are not Nonverba's schema. A projection chooses which Nonverba field
occupies each slot, and it may raise the cardinalities. The default
cardinalities are synthetic bounds. `semantic_width` is configuration; the
synthetic panels use 16.

## Query

`query(state, query_batch)` answers one or more questions against a state
already returned by `encode`. The set is not encoded again. `state` is
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

`query_scalars(state, query_batch)` applies one linear map to that answer and
returns `[batch, queries, scalar_features]`. It is the probe readout used by
the capacity protocol. A training run that owns its task head can ignore it
and attach that head to `query()`.

Mixture density sampling from that answer is deferred. This package does not
draw a sample, and it does not add further PMA seeds. The masked-reconstruction
loop in `train` does not train the query block. The capacity protocol does.
The query checks below are structural.

## CUDA

Attention and the training path run on CUDA. CPU batches are rejected.

The managed image provides CUDA toolkit 12.4, V12.4.131 [5], and the LibTorch
2.6.0 CUDA build [4] at `/opt/cuwacunu_embedding/libtorch`. This package has no
`.cu` files. GPU kernels come from that LibTorch build. `CMakeLists.txt`
requests Torch 2.6 and fails if `libtorch_cuda` is absent. The image has no
`cmake` binary. The Makefile compiles with g++ using that include path, C++17,
`_GLIBCXX_USE_CXX11_ABI=1`, and `-ltorch_cuda`.

Products go to `/opt/mityane-geometrics/fusion/build/state-fusion-transformer`.

```text
make -C /mityane/code/fusion/state_fusion_transformer test
make -C /mityane/code/fusion/state_fusion_transformer train
make -C /mityane/code/fusion/state_fusion_transformer capacity
make -C /mityane/code/fusion/state_fusion_transformer lego
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

This section is a pipeline check. `make_panel` builds the panels in C++.
Records in one set share a latent scalar vector, plus noise, real zeros,
field holes, record holes, and absent semantic vectors. A fraction of the originally observed scalars is withheld.
The visible batch stores garbage in holes and in withheld cells, and those
cells are marked unobserved. The loss is the mean squared error on the
withheld cells only.

These panels validate the pipeline. The printed error is the masked error on
that one synthetic panel.

## Training handoff

Nonverba trains this model on its own records. The callable surface is
`StateFusionModel`, `reconstruction_loss`, `query`, `query_scalars`,
`save_checkpoint`, and `load_checkpoint`. `src/train.cpp` and
`src/capacity.cpp` show that surface on synthetic panels. Those panels are
not Nonverba's data.

Construct `StateFusionModel` from `Config`, move it to CUDA, and fill
`SetBatch` on that device. `reconstruction_loss` is the masked scalar error
on cells that the caller has already turned into holes. `query` reads a
cached `encode` result and does not encode the set again. `query_scalars` is
a linear probe on that answer. Replace it with a task head on `query()` when
the target is not a scalar vector. Optimize with Adam [6]. `save_checkpoint`
writes the config and the weights to a path the caller owns.
`load_checkpoint` reads that archive and restores the module onto CUDA.

`make test` still runs the one-panel reconstruction gate. That gate memorizes
one synthetic panel and does not train the query block. `make capacity` is a
separate protocol. It is not part of `make test`.

## Capacity protocol

This protocol checks that one pooled state can answer a query. Nonverba's
records are not in this run. `make_capacity_panel` builds a fresh CUDA panel
on every step. The set size is even. Records come in pairs. A pair shares a place in `{0, 1, 2, 3}` and a
kind in `{0, 1}`, and no two pairs in a set share that combination. Scalar
feature 0 is the shared latent plus `0.5 * (place - 1.5) + 0.4 * (kind - 0.5)`
plus noise of scale 0.05. The other scalar features are the shared latent
plus that noise. Records 0 and 1 stay observed. Place and kind stay observed
on every observed record. Feature 0 of record 0 is hidden. Feature 0 of
record 1 stays visible, so the pair shift is in the state and is not written
into the hidden cell.

Each set has two queries. The empty query has every field unobserved. Its
target is the mean clean scalar over observed records. The filter query
observes only the place and kind of record 0. Its target is the conditional
expectation: latent feature 0 plus that pair's shift, and the shared latent
on the other features.

The loss is `reconstruction_loss` on the visible panel plus the mean squared
error of `query_scalars(encode(visible), queries)`. The state is not
detached, so the query block and the encoder both receive gradients. Each
step draws a new panel. Seeds `2102` through `3701` are the 1600 training
panels. The 32 held-out panels use seeds `50000` through `50031` and are
never training seeds. Metrics are computed in eval mode. The baselines are a
prediction of zero on hidden cells, and the set mean of the visible clean
scalars.

The program prints `note=synthetic_capacity`. A passing run has held-out
reconstruction below half its untrained value, held-out empty-query error
below half its untrained value, held-out filter feature-0 error below the
held-out set-mean baseline, and a reloaded checkpoint whose `query_scalars`
answers match within `1e-4`. This is a synthetic capacity measurement. It
does not close Hx11 or Hx20.

On 31 October 2026, `make capacity` ran in the managed container on the same
RTX A2000, with TensorFloat-32 off and LibTorch 2.6.0. The untrained held-out
scores were reconstruction `1.44461`, empty-query `1.26757`, and filter
feature 0 `1.76328`. The zero baseline on hidden cells was `1.05143`. The
set-mean baseline was `0.0642019` for reconstruction and `0.348477` for
filter feature 0. The first step put a finite gradient on the query block
(`139.105`) and on the probe head (`32.2642`). After 1600 Adam steps the
held-out scores were reconstruction `0.0994731`, empty-query `0.0820172`,
and filter feature 0 `0.0925034`. Reloading the checkpoint reproduced
`query_scalars` with max abs `0`. That probe file is written under `.temp/`
and is not part of the source tree. The binary printed
`state_fusion_capacity: pass`.

Filter feature 0 beat the set-mean baseline, so one pooled seed can be
indexed by place and kind. That error is still above the variance of the
0.05 observation noise. The curve was still falling at step 1600.
Reconstruction beat zero and fell below half of its untrained value. It
remains above the visible set-mean baseline of `0.0642019`. These figures
are the synthetic capacity measurement.

## LEGO boxes

`make lego` checks the same path on one catalog of boxes. It is not part of
`make test`, and it is not the purpose of the family. One situation is one
LEGO set. One record is one loose part in one color. The covered pile
is left out of the state. The query names its color, its part category, and
its part number. The probe answers with a brick count. Spare pieces are
excluded. Pieces inside a minifig are not opened. The year on the box is unused.

The rows come from the Rebrickable catalog download on 31 October 2026,
acknowledged as required by Rebrickable. The gzip hashes are themes
`52d3ac93001638c3960641af8560327e890969aacdd507bc9baec7156ced6ef1`, sets
`54c62a658164608472d556c47383b70ed7a0695c0f19dc7264af4136264035ab`, inventories
`09497d1c1c842994b7729357f8e57ce25dd932ce7a497d8dccf4fca58361f452`, parts
`78eb0c23a339f3fd8dbdd146963c7d6f5aa26ced8d458be2845a0c3cc4653e44`, part
categories
`c6cbd2306885609f496f4bc486938b911ee1ab9c759e4f59f21d1691e04b26cb`, and
inventory parts
`640d26717de2d4511283313637b0f7357a1b74904ffe15b07b9c9bd090338db6`.
Each set keeps its highest inventory version. A box is kept when it has 8
through 24 distinct part-and-color piles. That leaves 4,895 boxes. The
held-out theme is the largest top-level theme with at least 40 such boxes and
at most 15 percent of them. That rule selects Duplo, 714 boxes. Of the
remaining boxes, those whose set number has FNV-1a hash divisible by 5 are a
seen-theme holdout, 859 boxes. The other 3,322 train. Color, category, and
part ids, and the log-quantity scale, are fit on the training boxes only.
An unseen id is an unknown token. The loss is the mean squared error of
`query_scalars` in that scaled space. The reported error is the mean absolute
brick count after the scale is inverted.

On the same RTX A2000, TensorFloat-32 off, LibTorch 2.6.0, seed 2101, 4,000
Adam steps at `1e-3`. The first step put a finite gradient on the query block
(`240.119`) and on the probe head (`18.5838`). On Duplo, 10,959 covered piles,
the final mean absolute errors were model `0.717937`, the average of the other
piles in that box `1.04814`, zero `1.94452`, and the training-catalog mean of
that part and color `2.0558`. Of those piles, 4,472 were a part and color
already seen in training. On the seen-theme holdout, 13,646 piles, the same
errors were model `1.47262`, box average `1.77394`, zero `2.82925`, and
catalog `2.14712`, with 11,421 known part-and-color pairs. The Duplo curve was
`0.915783`, `1.08713`, `0.974694`, `1.00655`, then `0.717937` at steps 800
through 4,000. Reloading the checkpoint reproduced `query_scalars` with max
abs `0`. The probe file is under `.temp/` and is not part of the source tree.
The binary printed `state_fusion_lego: pass`.

The model is closer than the other piles in the same box, on a kind of box it
did not train on and on other boxes from kinds it did. The catalog mean is a
weak opponent here: many Duplo part-and-color pairs never appear in the
training boxes, and even on seen themes the box average is closer than the
catalog. An error of about 0.7 bricks on Duplo, where the average pile is
about 1.9 bricks, still misses by a large share of the count. This is a count
inside a box.

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
state reproduced the answer exactly. These checks do not train the
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
