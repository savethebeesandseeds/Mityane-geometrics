# Dashboard work segmentation

The owner requested an explicit work map on 18 October 2026: operation
capabilities, a Hero for each existing encoder family, and an external Google
multimodal encoder. There are **18 Heroes**, each maintained as one JSON file in
`heroes/` and registered in `manifest.json`.

Use **Hx01–Hx18** as their canonical references: `Hx` plus the zero-padded Hero
number. These references identify the workstreams and are independent of their
stable JSON IDs and file paths. The [schema](SCHEMA.md#hero-references) defines
the naming convention, including how to read historical references.

## Operation capabilities

Hx01–Hx13 own work for the thirteen capability families defined by the
[project charter](../../doc/operations_over_representations_design_spec_v0.1.0.md).
OR is the charter's Operations over Representations capability prefix; it is
retained for scientific classification and operation contracts.
Each separates contract definition, a reference implementation and evaluation.
The Hero JSON files hold current milestones, acceptance criteria, dependencies
and priorities. Specific datasets, artifacts and execution admission must be
bound before the dependent experiment; this work map does not select them or
start an experiment.

| Hero | Capability family | Ownership |
| --- | --- | --- |
| Hx01 | OR-01 | Retrieval and similarity matching: ranked source references and relevance |
| Hx02 | OR-02 | Structure discovery and density modeling: neighborhoods, groups and concentration |
| Hx03 | OR-03 | Anomaly and novelty scoring: departures from an explicit reference |
| Hx04 | OR-04 | Distribution shift monitoring: cohort change under a versioned space |
| Hx05 | OR-05 | Task prediction and readouts: external target estimates |
| Hx06 | OR-06 | Matched comparisons, sensitivity and counterfactual analysis: explicitly scoped contrasts |
| Hx07 | OR-07 | Latent transformations and relational operations: validated candidate transformations |
| Hx08 | OR-08 | Compression, aggregation and summarization: compact codes with retained utility |
| Hx09 | OR-09 | Decoding and conditional generation: reconstructions and generated observations |
| Hx10 | OR-10 | Representation alignment and interoperability: cross-space mappings |
| Hx11 | OR-11 | Latent dynamics and forecasting: future latent states and distributions |
| Hx12 | OR-12 | Decision making, planning and control: actions under objectives and constraints |
| Hx13 | OR-13 | Information acquisition and exploration: budgeted selection of new evidence |

## Shared foundations and encoders

| Hero | Ownership and evidence boundary |
| --- | --- |
| Hx14 · Shared foundations | Observation/space/run/evidence contracts, shared datasets, fair comparisons, pipeline semantics and reproducibility |
| Hx15 · MTF-JEPA-MAE-VICReg | Reusable MTF family components and provider tools; retain the MTF-v1 source and baseline engineering evidence with their original scope |
| Hx16 · RPB-MAE | Reusable RPB family components and provider tools; preserve version-specific geometry, historical leaders and causal-eligibility limits |
| Hx17 · Causal patch bottleneck MAE | [Configurable causal family and portable training APIs](../../code/embeddings/encoders/causal_patch_bottleneck_mae/family/README.md); the [Hero record](heroes/17-encoder-cpb.json) owns current engineering coverage and the V16 mixed research reference alongside V4/V3 history |
| Hx18 · Google multimodal encoder | Reusable external-provider adapter, configuration and validation tools within the capabilities recorded in the [provider guide](../../code/embeddings/encoders/google_multimodal_embedding/README.md); local adapter availability remains explicitly scoped |

The provider folders under `code/embeddings/encoders/` each have one Hero.
The [current research summary](../../doc/EMBEDDINGS.md) distinguishes these
leading successes, baseline roles, task-specific records and formal promotion.
Versions and trained instances stay within the corresponding family Hero;
they are not separate top-level Heroes.

### Encoder Hero instructions

Hx15–Hx18 follow the shared [encoder component instructions](../../code/embeddings/encoders/AGENTS.md).
Their deliverables are reusable encoder components and tools that applications
can configure and compose. Mityane researches architecture families, portability
and scoped representation evidence; application projects such as Cuwacunu own
substantial application training and final docking.

Keep the family definition, supported configuration, training recipe and trained
instance separate. A checkpoint or benchmark describes one instance and does not
set the only constructible geometry or establish a finished application. Owned
trainable families need explicit constraints and portable construction,
preprocessing, training, export and resume coverage, with implementation gaps
identified. External providers expose their actual upstream configuration and
capability limits; this does not imply access to weights or local training.
The Hero records hold the agreed work and evidence for these interfaces. This
guidance does not claim that every family already supplies them or authorize
new training, experiments or external API calls.

Shared foundations own cross-component contracts and reproducibility. Encoder
Heroes own representation production and its provenance. Operation Heroes own
their input/output semantics, controls and evidence of usefulness. Combining
them does not transfer an encoder's historical scores into a new operation.

## Exposing each Hero as a toolset

On 20 October 2026 the owner requested a toolset-exposure goal for every Hero,
following the broader Hx06 goal. Each goal maps the Hero's developed components
and selected future public capabilities, then exposes and verifies the agreed
building blocks through consistent command-line and reusable library interfaces.
The [initial common tool boundary](../../code/operations/TOOLS.md) provides a
starting convention; its first registered tool does not establish toolset
completion for the other Heroes.

Operation Heroes expose their own semantics across compatible representation
providers. Encoder Heroes expose configurable family and provider tools,
including supported representation and training workflows, through common
consumer interfaces. Shared foundations expose reusable
contracts, validators, artifact/evidence handling and composition mechanisms.
These goals retain component ownership, explicit compatibility and claim limits.
Their scope, milestone states and acceptance criteria remain in the Hero JSONs;
planning entries do not start experiments, live API calls or deferred work.

## Reading progress

Completed milestones link existing evidence for their exact scope, including
preservation, baseline checks, completed research and portable component checks.
Small engineering training fixtures remain separate from research measurements.
Dated research pause instructions, frozen recipes and original registry
proposals remain historical evidence; subsequent authorized work has its own
records. Current states and limitations belong in the Hero JSONs. Future
scientific execution remains separately scoped.

No owner or deadline is guessed. Only milestones store states. Proposed work
does not imply a selected dataset, running implementation, live API access or
measured operation quality. Dependency links connect specific milestones; the
thirteen capability families are not a compulsory sequence.

After an authored record changes, run the documented refresh. The generated
HTML is a snapshot; JSON remains the editable source of truth. See
[SCHEMA.md](SCHEMA.md) for the fixed Hero → Goal → Milestone hierarchy.
