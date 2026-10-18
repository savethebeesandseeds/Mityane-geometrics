# Dev encoder replacement archive — 15 October 2026

Before selecting CPB-v3.market-01, the complete former dev CPB-v1 implementation,
family metadata, application facade/settings and affected live documents were
copied byte for byte into `source/`. `manifest.json` maps all67 original dev
paths, sizes and SHA256 hashes. Its SHA256 is
`fa7019bc998ffbdc14a6a1aeebaeba280f1fe07ba176ba57b1741acb035bf031`.
`consumer-manifest.json` separately maps five original downstream consumer files
under `consumers-before/`; these include the latent capture fixture, its build
binding and current explanatory docs.

The old active v1 implementation and its two catalog entries were removed from
dev only after all47 implementation files matched both this archive and the
unchanged portable research v1 source. This archive is historical source, not
an active build tree. Preserve inherited copyright and license notices; archiving
does not change ownership or licensing. All original checkpoints/results remain
in place, including the original dev v1 checkpoint. No weights are in this archive.

CPB-v3 was trained from fresh learned weights and fresh TRAIN-only statistics;
v1 inherited noncausally trained V30 weights and is historical context only.
The current integration evidence is in `doc/cpb_v3_primary_integration/` and the
sibling dev `doc/CAUSAL_PRIMARY_INTEGRATION.md`. Research results are preserved
in `doc/CAUSAL_FRESH_V3_RESULTS.md`; this replacement performs no new quality run.
