# Third-party notices

The imported representation implementation and supporting code come from
Waajacu's `cuwacunu_embedding` project, including adaptations of
`cuwacunu_torch`. Their MIT copyright and permission notice is retained in the
root [LICENSE](LICENSE), which also covers copied and adapted project code.
The duplicate component license had the same bytes. Original notices and
configuration remain in the [pre-cleanup snapshot](archive/embeddings/pre-cleanup-2026-10-18/).
See [source provenance](doc/PROVENANCE.md) for revisions and import boundaries.

The dashboard includes code imported from Non-verba under AGPL-3.0-only.
Its [full license](doc/licenses/non-verba-AGPL-3.0.txt) and dashboard notices
remain applicable; the repository MIT license does not relicense that code.

The preserved ignored `code/embeddings/.external/libtorch` transfer bundle is
PyTorch/LibTorch `2.6.0+cu124`, including CUDA and cuDNN runtime libraries.
The installed SDK remains at `/opt/cuwacunu_embedding/libtorch` inside the
managed container. Those components retain their respective upstream licenses.
The project MIT license does not relicense them, and the binary bundle remains
excluded from source control.

Debian compiler/runtime package copyright notices are available inside the
container under `/usr/share/doc/<package>/copyright`. Exact installed dependency
versions are recorded in root `dependencies.lock`. Cached Node and paper
vendor notices retain their own provenance and licenses.

Redistributions including third-party binaries must include the corresponding
upstream license texts and notices.
