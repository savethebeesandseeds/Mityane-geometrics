# Heroes Goals and Milestones schema

Version 2. An empty `heroes: []` manifest is valid until the owner defines the
first Hero. The hierarchy has exactly three levels:

**Hero → Goal → Milestone**

- **Hero:** a broad area of work, defined by the project owner.
- **Goal:** a concrete outcome within that Hero.
- **Milestone:** a checkable step toward the Goal, with its own state.

Edit the relevant Hero JSON in `heroes/`, such as
`heroes/01-example.json`. Only Milestones store states. The dashboard calculates Hero
and Goal summaries automatically from its last validated snapshot.

## Record only confirmed scope

This is a partial project record. Add a Goal or Milestone when an explicit owner
request, agreed task record or implementation evidence supports its scope and
state. Broad research notes and missing-feature lists support the Hero outline;
they do not establish a specific task's acceptance criteria, dependencies or
active status. Read the linked evidence before recording an outcome.

Leave `goals: []` when a Hero's breakdown is unconfirmed. A confirmed Goal can
have `milestones: []` until its steps are defined. Empty records mean no
Milestones are recorded, rather than no work exists. Keep known prototypes,
synthetic checks, physical trials and deployment outcomes precisely scoped.
Omit unknown dates and ownership. A done record completes only its stated scope.

## Minimal Hero file

```json
{
  "schemaVersion": 2,
  "id": "example",
  "number": 1,
  "title": "Example workstream",
  "summary": "Replace this example with confirmed scope.",
  "updatedAt": "2026-10-14",
  "goals": [
    {
      "id": "example-reliable-capture",
      "title": "Reliable sensor capture",
      "milestones": [
        {
          "id": "example-permission-recovery",
          "title": "Handle denied permissions and recovery",
          "status": "proposed"
        }
      ]
    }
  ]
}
```

## Fields

| Level | Required fields | Optional fields |
| --- | --- | --- |
| Hero | `schemaVersion: 2`, `id`, `number`, `title`, `summary`, `updatedAt`, `goals` | None |
| Goal | `id`, `title`, `milestones` | `description` |
| Milestone | `id`, `title`, `status` | `description`, `owner`, `updatedAt`, `completedAt`, `acceptance`, `evidence`, `dependsOn` |

IDs are stable lowercase slugs, unique across all Heroes, Goals and Milestones.
Prefix Goal and Milestone IDs with their Hero ID. Titles, descriptions and
criteria use ordinary text. `goals` and `milestones` are arrays; there is no
recursive `children` field. Do not put `status` on a Hero or Goal.

Dates use `YYYY-MM-DD`. Update the Hero's `updatedAt` when changing its file.
Milestone dates are optional; a recorded completion date must be known, belong
to a done Milestone and not follow its update date (or the Hero's update date).
Unknown completion dates can be omitted or `null`. `owner` is optional text or
`null`. Missing arrays are treated as empty.

## Milestone states

| JSON value | Dashboard color | Meaning |
| --- | --- | --- |
| `proposed` | Gray | Scope or approach needs definition or agreement |
| `ready` | Blue | Defined, with prerequisites resolved; available to start |
| `in_progress` | Amber | Work has begun, supported by the current record |
| `blocked` | Red | A specific unresolved dependency prevents progress |
| `done` | Green | Acceptance criteria are achieved and evidence is linked |

These states describe work lifecycle. Readiness does not grant execution
authority or remove a research pause; each experimental run needs its applicable
Operation Contract, linked Requirements-and-Evidence Record and permissions.
A valid completed evaluation may have
a negative or inconclusive scientific verdict and still meet its milestone's
acceptance criteria. Failed, invalid or unexecuted attempts remain visible but do
not alone complete an operation-quality evaluation.

A blocked Milestone requires a `description` explaining the blocker. A done
Milestone requires nonempty `acceptance` and `evidence` arrays:

```json
{
  "id": "example-permission-recovery",
  "title": "Handle denied permissions and recovery",
  "status": "done",
  "completedAt": "2026-10-14",
  "acceptance": ["The agreed denial and recovery cases pass."],
  "evidence": [
    {"label": "Recorded validation", "path": "doc/VALIDATION.md"}
  ]
}
```

Each evidence entry has a `label` and a repository-relative file `path`, with an
optional heading anchor. Link authored records supporting the actual outcome.
For proposed work, charter and planning links can explain the design; they do not
serve as completion receipts. Completing work requires evidence of its actual
acceptance criteria.
`dependsOn`, if used, is an array of other Milestone IDs. Dependencies must exist
and cannot form a cycle; they cannot point to a Hero or Goal.

Hero and Goal summaries count their Milestones once: all done means done, an
unresolved blocker takes precedence, active work means in progress, and a mixture
of completed and proposed/ready work is labelled **Partially complete**. Otherwise
they show ready or proposed. Counts describe tracked Milestones, not effort or
production readiness. Empty Goals are allowed while planning.

## Updating a milestone

1. Read this schema and the relevant Hero JSON; preserve other sessions' changes.
2. Update the Milestone's state and description. For done, add its actual
   acceptance criteria and evidence; for blocked, explain the blocker.
3. Update the Hero date. Add optional Milestone dates, owner or dependencies when
   useful. Keep existing IDs when moving a Milestone between Goals.
4. From the repository root, refresh the standalone dashboard:

```powershell
./web/dashboard/scripts/refresh.ps1
```

The helper runs through the root `container.ps1` launcher against the existing
`mityane-geometrics` container and its `/mityane` bind mount. It runs focused tests,
validates the JSONs and regenerates `public/index.html`. It detects source changes
during generation. Hero JSONs remain the source of truth; never hand-edit the HTML.
For a container-side invocation, use `bash web/dashboard/scripts/refresh.sh`.

The adjacent `manifest.json` needs an edit only when adding/removing a Hero or
changing its file/ID/order, and has no shared update date. Ordinary progress work
edits one Hero JSON and refreshes the generated HTML.

Open [public/index.html](public/index.html) directly in a browser after a successful refresh.
It opens populated without a picker or server. Browser file-open security prevents
automatic fetching of adjacent JSONs, so the HTML includes the last validated
snapshot. JSON edits become visible only after refresh and reopening/reloading
the HTML. `./web/dashboard/scripts/refresh.ps1 -Check` validates data and HTML freshness
without changing the HTML. Focused model and bundling checks run on every refresh.

The formal definitions are [hero.schema.json](schemas/hero.schema.json) and
[manifest.schema.json](schemas/manifest.schema.json). Runtime validation also checks real
calendar dates, date ordering, global ID uniqueness, dependency references/cycles
and existing evidence files. See [opening instructions](README.md#open-the-dashboard).
