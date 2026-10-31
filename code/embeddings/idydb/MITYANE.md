# IdyDB copy for Hx18

This directory is a source copy of `C:\Work\idydb`, brought in on 30 October 2026 so Hx18 has a local file for saved embeddings. The upstream license is MIT, copyright 2025 Waajacu, in [LICENSE](LICENSE). The upstream readme is [README.md](README.md). The library itself is [idydb/](idydb/).

The copy is the C/C++ engine, its public header, CMake build, and the tests that sit next to that engine. Windows addon binaries, the `idydb-dev` container, and the `build/` tree were not copied. Two upstream fixture files, `portable-v1.db` and `portable-v1.encrypted.db`, are mode `0600` and owned outside this workspace, so this copy does not include their bytes. Tests that open those fixtures are not part of the Mityane check.

Mityane builds the library inside the `mityane-geometrics` container and links the image OpenSSL Crypto. It does not use the IdyDB development container. A database file that holds vectors stays under ignored `.temp/` or the container build disk. It is not a Git object.

IdyDB stores text and float vectors and can rank a column by cosine or L2. That rank is the library's own nearest-neighbor query. It is not OR01-COMB, and it is not the Gemini Embedding 2 adapter. Hx18's API adapter stays proposed. No live provider call is authorized by this copy.
