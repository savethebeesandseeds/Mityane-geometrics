# Source provenance

The research predecessor is `C:\Work\cuwacunu\cuwacunu_embedding`. Its models,
measurements, datasets and checkpoints remain there. Initial setup imports no
embedding model, trained weights, fitted preprocessing, datasets or application.

The development environment uses the already installed embedding snapshot made
on 12 October 2026. `environment-image.json` preserves its immutable image ID,
source container identity and original Debian base digest. A new tag reuses the
same layers. Both original research containers are preserved.

Environment support comes from the MIT-licensed Waajacu source projects:

- `cuwacunu_dev` revision `221ce90d25336f8ce81b9d7f272928191f584324`:
  adapted `container.ps1`, offline `setup.sh`, and
  `code/scripts/install-libtorch.py` (only its project entry guard and descriptive
  name change). The verifier retains its original SDK proof semantics.
- `cuwacunu_embedding` revision `3316fdda42de103282c62f47cfd0300cbf2d272b`:
  `dependencies.lock` and the retained MIT license. The dependency lock is also
  byte-identical to the `cuwacunu_dev` copy, SHA256
  `b8f946448de4c2a6b0b2d94794a096266644dece00fd8fd27ef5a0a61e8684d3`.

These revisions identify the inspected repositories; the copied support files
are recorded by SHA256 in `environment-image.json` because local source edits
can exist independently of a Git revision.

When importing an embedding later, record its family/version, checkpoint hash,
input layout, native output layout, masks, scaler and training provenance,
evaluation protocol, source revision and license. Keep evidence from the
predecessor intact and distinguish inherited measurements from new results.
