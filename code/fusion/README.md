# Fusion families

Hx20 and Hx21 own this directory. It is separate from `code/embeddings/`.
Encoders and the State Representation Network stay there.

| Family | Path | What is here |
| --- | --- | --- |
| Hx20 · Temporal Fusion Transformer | [temporal_fusion_transformer](temporal_fusion_transformer/README.md) | The imported quantile-forecast library |
| Hx21 · State fusion transformer | [state_fusion_transformer](state_fusion_transformer/README.md) | Set Transformer state encoder ([Lee et al., ICML 2019](https://proceedings.mlr.press/v97/lee19d.html); ISAB, PMA) on CUDA |

A forecast from either family does not close Hx11.
