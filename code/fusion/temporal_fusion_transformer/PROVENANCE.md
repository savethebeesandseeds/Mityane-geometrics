# Provenance

Imported on 29 October 2026 into
`code/embeddings/encoders/temporal_fusion_transformer/`.
Moved the same day to `code/fusion/temporal_fusion_transformer/`.
The maintained path is the fusion directory. The earlier embeddings path is
only the first landing place.

The library bytes come from the Non-verba copy at
`C:\Work\Non-verba\private-source\code\server\tools\temporal-fusion-transformer`.
That copy's `ORIGIN.md` says it was taken the same day from
`C:\Work\post-profit-economics\tools\temporal-fusion-transformer`.
A directory comparison the same day showed what the Non-verba copy adds:
one pooled vector per input group, a boolean time mask, checkpoint format
version 4, and `tests/tft_mask_test.cpp`. The post-profit-economics tree is
the earlier library, without that mask surface.

These files match the Non-verba tree at the copy. Their hashes are in
[checksums.sha256](checksums.sha256):

- `CMakeLists.txt`
- `LICENSE`
- `include/praesidium-humanitatis/`
- `src/`
- `tests/`
- `examples/`
- `replication/`

Not copied:

- `environment/`, the other project's container and dependency installer.
  Mityane builds in `mityane-geometrics` and links
  `/opt/cuwacunu_embedding/libtorch`.
- `tasks.sh`, which refuses to run without that project's profile.
- `ORIGIN.md` and `README.md`. This file and the family README replace them.
- `__pycache__/`.

The MIT license in `LICENSE` stays with the imported files. Copyright (c) 2026
Waajacu. Upstream notes that point outside the tool directory remain notes of
the post-profit-economics program and were not imported.

The C++ namespace `praesidium_humanitatis` is unchanged. The fusion path is
the maintained source identity. Do not treat a file here as the file recorded
by an upstream receipt, or as the file first landed under `code/embeddings/`.

No Electricity run, synthetic study or CUDA suite was replayed for this copy.
