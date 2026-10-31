# OR01-RET-ABS 0.2.0 local text encode

Encoded on 31 October 2026 inside `mityane-geometrics` with `encode_interior.py`. The model is `google/embeddinggemma-2` revision `f9972ac22e696a105c482202646979e33e4b5a1e`, text tower only, dtype bfloat16. The prompt is `task: sentence similarity | query: `. The surface is the first member, a colon, and the first definition. Pooling is the mean of every token, including the prompt, then L2. The metric is cosine. The native width is 768.

The corpus snapshot id is `or01-ret-abs-0.2.0-local-text`. The space id is `local-text-768-cosine-l2-sentence-similarity`. The file has 17,191 rows, one per interior source id, in snapshot order. Its sha256 is `6cf4cbbff14809791f49851472b6e9df28d3fda736eba28f7abce7ffa77bfd8e`.

The ten leaf queries are a separate file. Their snapshot id is `or01-ret-abs-0.2.0-leaf-queries`. Their sha256 is `f4164ad59c68aa795a7119974789f1cf48dc065a927e5f98403cf667460e1963`. None of those source ids is an interior row.

The 0.1.0 files were checked before this encode and left unchanged: abstract `0096a2d4338d03ba94973d2679ad0604a0003958985208171acc13f2da535a81`, concrete `6c8e7bccf3b2e16cdb01abe1a80489d266b167954dfe035271ee30e8fe90e885`.

Vector bytes and the plaintext IdyDB file stay under ignored `.temp/` or the container build disk. This record does not admit a relevance result. The encode does not train the model and does not rank neighbors.

Reproduction, from the repository root, through the managed container:

```text
/usr/bin/python3 code/operations/retrieval/bindings/oewn-2025/encode_interior.py \
  --snapshot code/operations/retrieval/bindings/oewn-2025/interior-snapshot.json \
  --queries code/operations/retrieval/bindings/oewn-2025/leaf-queries.json \
  --out-dir /mityane/.temp/operations/retrieval/or01-ret-abs-0.2.0-local-text \
  --prior-dir /mityane/.temp/operations/retrieval/or01-ret-abs-0.1.0-local-text
```

The producer process needs the existing EmbeddingGemma prefix on `PYTHONPATH`, `HF_HOME` pointed at that prefix's cache, and `HF_HUB_OFFLINE=1`.
