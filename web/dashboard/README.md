# Mityane geometrics project dashboard

The [project design specification](../../doc/operations_over_representations_design_spec_v0.1.0.md)
defines the research scope: operations over learned and externally supplied
representations. Its capability catalogue guides research organization;
owner-authored Heroes, Goals and Milestones record confirmed work separately.

## Open the dashboard

Open [public/index.html](public/index.html) directly in a browser. The page embeds
its artwork, interface and last validated JSON snapshot. Opening it needs no
server, container, network connection or file picker.

This is the Non-verba dashboard backbone: the same cream/sage/sand styling,
overview → one Hero → one Goal navigation, milestone details, status filters,
search, progress rings, evidence cards and dependency jumps. The manifest contains
**18 Heroes**, defined from the owner's requested work segmentation: the thirteen
OR capability families, shared foundations, three existing encoder families and
Google's external multimodal encoder. See [the work map](WORKSTREAMS.md) for their
boundaries. No Non-verba progress or artwork is included, and synthetic test
records never enter the maintained page.

## Maintain Hero records

Read [SCHEMA.md](SCHEMA.md) and edit the relevant file in `heroes/`. Each Hero has
one JSON registered by ID/file in `manifest.json`. Add further Heroes only when
the owner defines their scope. Keep the three levels **Hero → Goal → Milestone**.
Only Milestones store states; summaries count recorded Milestones automatically.
Empty Heroes and Goals are supported. Record only confirmed scope and evidence.

The Hero JSON files hold the current task details, acceptance criteria and
states; the refreshed dashboard calculates their counts. Keep those details in
the Heroes rather than duplicate them in guides. Counts describe recorded scope,
not effort or overall project readiness. The causal encoder's research pause and
pending registry review are preserved; the Google provider record does not imply
a working API adapter.

After changing JSON, interface code or branding, refresh from the repository root:

```powershell
.\web\dashboard\scripts\refresh.ps1
```

The helper uses the existing `mityane-geometrics` container through
`container.ps1`. It runs focused model and bundling checks, validates the data
and evidence paths, then regenerates `public/index.html`. It refuses conflicting
refreshes and detects source changes during generation. Reload the HTML afterward;
browser file security prevents fetching adjacent JSON automatically.

To verify freshness without changing the HTML:

```powershell
.\web\dashboard\scripts\refresh.ps1 -Check
```

## Maintain the branding

The dashboard uses the shared [cosmic character collection](../branding/index.html).
Cosmic Synthesis is the global mark, introductory portrait and fallback image;
each of the 18 Heroes has its own distinct character. Assignments are keyed by
the existing Hero IDs in [dashboard-assignments.json](../branding/dashboard-assignments.json).
Change that file to select another original PNG, then run the normal refresh.
Adding a Hero also requires a distinct image assignment.

Refresh prepares transparent WebP copies in `assets/branding/`: the global image
is 384 × 384 and Hero images are 256 × 256. It regenerates
[branding.json](assets/branding/branding.json) and records source/output hashes,
dimensions and conversion settings in [provenance.json](assets/branding/provenance.json).
The full-resolution PNGs and generation records stay in `web/branding/`.

Preparation runs inside the managed container, using Python's standard library
and the already locked GdkPixbuf and libwebp libraries. It installs nothing and
keeps the existing 512 KiB combined artwork budget. The finished HTML embeds all
19 selected images and needs no adjacent image files or service. `-Check` checks
the prepared copies and standalone HTML without rewriting them. Visual theme
tokens remain at the top of `src/dashboard.css`.

## Container dependencies

Opening the finished HTML needs no development runtime. Refreshing uses Linux
Node **22.23.3**, with no npm packages. The optional dependency-only helper is:

```powershell
.\container.ps1 -Action exec -Command @('bash', 'web/dashboard/scripts/setup.sh')
```

It verifies and installs only the Node binary and full upstream license from the
already cached archive:

- Container archive: `/opt/mityane-geometrics/web/downloads/node-v22.23.3-linux-x64.tar.xz`
- Container runtime: `/opt/mityane-geometrics/web/tools/node-v22.23.3-linux-x64/`
- Upstream artifact: `https://nodejs.org/dist/v22.23.3/node-v22.23.3-linux-x64.tar.xz`
- SHA256: `df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de`

The initial cache was streamed directly from the verified existing Non-verba
container's exact pinned archive. No sibling mount, host toolchain, package
installation, runtime download or image change was needed. The helper never
downloads a missing archive, changes an existing mismatched runtime or modifies
system packages. Restore that exact cache explicitly on another environment.
Temporary files and the refresh lock stay under `/opt/mityane-geometrics/web/`.

## Source and license

Imported from Non-verba revision `b67d8a1e025dacfed8ac7aba2225d76cfa48f7e7`.
The dashboard retains **AGPL-3.0-only**; the repository's MIT license does not
relicense the imported interface. See the
[retained license](../../doc/licenses/non-verba-AGPL-3.0.txt),
[import receipt](../../doc/DASHBOARD_IMPORT_2026-10-18.json) and
[provenance](../../doc/PROVENANCE.md).

This is a local review dashboard, marked `noindex`. Scripts use generated CSP
hashes and make no network requests. The style-element exception supports the
Codex review selector; script policy and style-attribute restrictions remain strict.
No hosting, publication or release export is configured.
