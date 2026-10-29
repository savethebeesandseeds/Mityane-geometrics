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

## Shared version-2 contract

All three dashboards use the same Hero, Goal and Milestone field contract in
`schemas/hero.schema.json`. Optional fields do not require a data migration:
omit them when no confirmed assessment or distinction has been recorded.
Completion counts describe recorded Milestones, not total project scope, effort
or deployment readiness. Only Milestones store work states.

Manifest metadata retains two explicit profiles. Mityane geometrics and
Non-verba use the flat envelope `{schemaVersion, title, description, heroes}`.
Cuwacunu uses `{schemaVersion, project, heroes}`, where `project.title` maps to
the shared title and `project.summary` maps to description. Its `project.name`
supplies branding. Optional `focusMilestones` in the flat profile, or
`project.focusMilestones` in the project profile, lists existing Milestone IDs
in declared campaign order. It grants no execution authority and does not
require a campaign panel in every project interface. This is a metadata adapter;
it does not change Hero records or give `schemaVersion: 2` a different
work-state meaning. Preserve the local envelope
when editing a manifest. Mityane permits an empty Hero list before scope is
defined; Non-verba and Cuwacunu require at least one Hero.

## Hero references

The canonical reference for a Hero is **HxNN**: `Hx` followed by its `number`,
padded to at least two digits. Use the exact casing in the interface and maintained
prose: **Hx01**, **Hx06**, **Hx18**. Keep the number stable when reordering Heroes.
The dashboard derives this reference for labels, navigation and search; do not
add a separate reference field or repeat it in the Hero's descriptive `title`.

The existing `id` slugs and JSON filenames remain stable machine references.
For example, Hx06 has `number: 6`, ID `or-06-comparisons`, and the title
"Matched comparisons, sensitivity, and counterfactual analysis". It is displayed
as **Hx06 · Matched comparisons, sensitivity, and counterfactual analysis**.

**OR-NN** identifies a capability family in the Operations over Representations
charter. Hx01–Hx13 currently map to OR-01–OR-13; Hx14 owns shared foundations,
Hx15–Hx18 own encoders/providers, Hx19 owns dataloaders, and Hx20 owns the
Temporal Fusion Transformer. Use Hx references when discussing the Heroes,
their Goals or their Milestones. Use OR references for operation contracts and
scientific classification. See the [complete mapping](WORKSTREAMS.md).

Archived documents, frozen protocols and hash-bound evidence retain their
original names. Historical references such as "Hero 17" or "Hero17" mean Hx17;
they do not create another Hero or require rewriting preserved records.

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
| Hero | `schemaVersion: 2`, `id`, `number`, `title`, `summary`, `updatedAt`, `goals` | `purpose`, `boundary` |
| Goal | `id`, `title`, `milestones` | `description`, `type` |
| Milestone | `id`, `title`, `status` | `description`, `owner`, `updatedAt`, `completedAt`, `acceptance`, `evidence`, `dependsOn`, `verification`, `outcome`, `requiresOutcomes`, `requiresCurrentEvidence` |

IDs are stable lowercase slugs, unique across all Heroes, Goals and Milestones.
Prefix Goal and Milestone IDs with their Hero ID. Titles, descriptions and
criteria use ordinary text. `goals` and `milestones` are arrays; there is no
recursive `children` field. Do not put `status` on a Hero or Goal.

Dates use `YYYY-MM-DD`. Update the Hero's `updatedAt` when changing its file.
Milestone dates are optional; a recorded completion date must be known, belong
to a done Milestone and not follow its update date (or the Hero's update date).
Unknown completion dates can be omitted or `null`. `owner` is optional text or
`null`. Missing arrays are treated as empty.

## Goal types

Optional Hero `purpose` describes its enduring mission and `boundary` describes
its responsibility alongside other Heroes. `summary` records its current
position. Optional Goal `type` defaults to `outcome` for a bounded outcome or
question. `ongoing` identifies a continuing responsibility: completing its
recorded Milestones does not exhaust that responsibility or imply active work.
`conditional` identifies an optional direction, without committing it to current
work. Leave unknown Goal and Milestone breakdowns empty.

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

## Completion, evidence applicability and advancement

`status` records work completion. Preserve a historical completion when later
changes mean its evidence no longer applies to the current system. A done
Milestone and a ready Milestone must have all direct `dependsOn` prerequisites
done. Proposed work can still need agreement or admission after its recorded
prerequisites are satisfied; a dependency-satisfied candidate is not execution
authorization.

Optional `verification` records evidence applicability with `status` (`current`,
`historical` or `unverified`), `scope` and `reason`. A current assessment also
requires a repository-relative `receipt` and its lowercase `receiptSha256`.
The receipt contains a nonempty `sourceFiles` array of `{path, sha256}` records.
Refresh checks the receipt and source bytes and generates
`verificationAssessments`; missing or changed pins yield stale applicability
without changing historical completion. No assessment means current
applicability is unverified. Evidence and receipt/source paths must resolve to
regular files within the repository. These checks establish only the recorded
scope, not full-system or research quality.

Optional `outcome` records an explicitly evidenced decision on done work:
`supported`, `negative`, `inconclusive`, `no_eligible_candidate`, `favorable`,
`eligible` or `rejected`. It is never inferred from `done`; engineering work does
not require a scientific outcome.

`dependsOn` requires completion. Optional `requiresOutcomes` adds explicit
advancement conditions, for example
`[{"milestone":"example-decision","outcomes":["supported"]}]`.
`requiresCurrentEvidence` lists dependency IDs whose generated applicability
must be current. Both must name distinct direct `dependsOn` dependencies.
Missing or disallowed outcomes and stale or unverified required evidence keep
current advancement closed, including through completed intermediate
dependencies. These gates do not erase completed work. Independent work needs
only its own declared prerequisites.

Ready states are checked in two phases: structural validation checks completed
direct prerequisites, then refresh and dashboard loading check all outcome and
current-evidence gates through the dependency ancestry. A normal refresh uses
the verification written in the Hero JSON. `-Audit` re-reads receipt and source
bytes first and can mark that recorded current evidence stale before the same
gate check. A ready record whose gates no longer pass must be reconciled before
that refresh can succeed; the validator does not change its state automatically.
Historical done records retain their completion when gates drift.

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
`mityane-geometrics` container and its `/mityane` bind mount. It reads the Hero
JSON and regenerates `public/index.html`. It detects changes to those inputs
during generation. Hero JSONs remain the source of truth; never hand-edit the HTML.
For a container-side invocation, use `bash web/dashboard/scripts/refresh.sh`.

The adjacent `manifest.json` needs an edit only when adding/removing a Hero or
changing its file/ID/order, and has no shared update date. Ordinary progress work
edits one Hero JSON and refreshes the generated HTML.

Open [public/index.html](public/index.html) directly in a browser after a successful refresh.
It opens populated without a picker or server. Browser file-open security prevents
automatic fetching of adjacent JSONs, so the HTML includes the last validated
snapshot. JSON edits become visible only after refresh and reopening/reloading
the HTML. `./web/dashboard/scripts/refresh.ps1 -Check` compares the HTML with the
Hero JSON without changing it. `-Audit` re-reads evidence files and current
receipts. `-Test` runs the focused model and bundling checks. `-Branding` prepares
artwork.

The formal definitions are [hero.schema.json](schemas/hero.schema.json) and
[manifest.schema.json](schemas/manifest.schema.json). Runtime validation also checks real
calendar dates, date ordering, global ID uniqueness and dependency references/cycles.
`-Audit` also checks that evidence paths resolve to files and that current receipts
still match. See [opening instructions](README.md#open-the-dashboard).
