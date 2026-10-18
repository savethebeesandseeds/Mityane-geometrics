# Mityane geometrics branding

The 24 full-resolution transparent PNGs are in `individual/`. Open
[the collection gallery](index.html) to browse them. The exact ImageGen prompts,
reference identities and artifact hashes are in [generation.json](generation.json).
An earlier version of the mixed-bouquet character remains in `explorations/`.

[dashboard-assignments.json](dashboard-assignments.json) selects one global
character and one distinct character for every dashboard Hero. Cosmic Synthesis
is the global image. These are visual assignments; the Hero records retain their
existing scope and progress.

Run `web/dashboard/scripts/refresh.ps1` from the repository root after changing
assignments. It uses the managed Linux container to prepare compact transparent
WebP copies, verify full Hero coverage and regenerate the standalone dashboard.
The original PNGs remain unchanged. The prepared copies and their hash records
live in `web/dashboard/assets/branding/`.
