# Fusion families

Hx20 and Hx21 own this directory. It is separate from `code/embeddings/`.
Encoders and the State Representation Network stay there.

| Family | Path | What is here |
| --- | --- | --- |
| Hx20 · Temporal Fusion Transformer | [temporal_fusion_transformer](temporal_fusion_transformer/README.md) | Quantile forecasts, and a temporal state `[batch, horizon, hidden]` before that head |
| Hx21 · State fusion transformer | [state_fusion_transformer](state_fusion_transformer/README.md) | Set Transformer (set fusion) on CUDA. The goal is state fusion |

The method for Hx21 is a Set Transformer (set fusion) ([Lee et al., ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)).
The goal is state fusion: one unordered set becomes one fixed state, and a
query reads that cached state. Hx20 keeps its quantile forecast and also
returns the temporal state in front of that head, shaped
`[batch, horizon, hidden]`. A result from either family does not close Hx11.

On 11 November 2026 both families are paused at those surfaces. Each guide
has a section named Where this stands: what is built, what the checks showed,
and what to consider on a later return. That section does not start a run.
