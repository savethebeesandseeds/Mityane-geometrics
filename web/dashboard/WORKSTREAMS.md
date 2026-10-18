# Dashboard work segmentation

The owner requested an explicit work map on 18 October 2026: operation
capabilities, a Hero for each existing encoder family, and an external Google
multimodal encoder. There are **18 Heroes**, each maintained as one JSON file in
`heroes/` and registered in `manifest.json`.

## Operation capabilities

These thirteen Heroes preserve the IDs and scope of the
[project charter](../../doc/operations_over_representations_design_spec_v0.1.0.md).
Each separates contract definition, a reference implementation and evaluation.
The Hero JSON files hold current milestones, acceptance criteria, dependencies
and priorities. Specific datasets, artifacts and execution admission must be
bound before the dependent experiment; this work map does not select them or
start an experiment.

| Hero | Family and ownership |
| --- | --- |
| 01 · OR-01 | Retrieval and similarity matching: ranked source references and relevance |
| 02 · OR-02 | Structure discovery and density modeling: neighborhoods, groups and concentration |
| 03 · OR-03 | Anomaly and novelty scoring: departures from an explicit reference |
| 04 · OR-04 | Distribution shift monitoring: cohort change under a versioned space |
| 05 · OR-05 | Task prediction and readouts: external target estimates |
| 06 · OR-06 | Matched comparisons, sensitivity and counterfactual analysis: explicitly scoped contrasts |
| 07 · OR-07 | Latent transformations and relational operations: validated candidate transformations |
| 08 · OR-08 | Compression, aggregation and summarization: compact codes with retained utility |
| 09 · OR-09 | Decoding and conditional generation: reconstructions and generated observations |
| 10 · OR-10 | Representation alignment and interoperability: cross-space mappings |
| 11 · OR-11 | Latent dynamics and forecasting: future latent states and distributions |
| 12 · OR-12 | Decision making, planning and control: actions under objectives and constraints |
| 13 · OR-13 | Information acquisition and exploration: budgeted selection of new evidence |

## Shared foundations and encoders

| Hero | Ownership and evidence boundary |
| --- | --- |
| 14 · Shared foundations | Observation/space/run/evidence contracts, shared datasets, fair comparisons, pipeline semantics and reproducibility |
| 15 · MTF-JEPA-MAE-VICReg | Retained MTF-v1 implementation; imported source and baseline engineering verification; operation contract work remains proposed |
| 16 · RPB-MAE | Existing raw patch bottleneck family and versioned evidence; choose explicit instances without changing historical references |
| 17 · Causal patch bottleneck MAE | CPB-v3 clean baseline and saved later candidates; V14 measurements are recorded, while its paused registry review remains unfinished |
| 18 · Google multimodal encoder | [Gemini Embedding 2 provider record](../../code/embeddings/encoders/google_multimodal_embedding/README.md); source registration complete, API adapter and operation evaluation proposed |

The provider folders under `code/embeddings/encoders/` each have one Hero.
Versions and trained instances stay within the corresponding family Hero;
they are not separate top-level Heroes.

Shared foundations own cross-component contracts and reproducibility. Encoder
Heroes own representation production and its provenance. Operation Heroes own
their input/output semantics, controls and evidence of usefulness. Combining
them does not transfer an encoder's historical scores into a new operation.

## Reading progress

Completed milestones link existing evidence for their exact scope: charter
adoption, verified preservation/environment, retained baseline tests, saved V14
records and the new external-provider documentation. Small engineering training
fixtures remain separate from research measurements. Existing research pause
instructions, frozen recipes and pending registry proposals are preserved.

No owner or deadline is guessed. Only milestones store states. Proposed work
does not imply a selected dataset, running implementation, live API access or
measured operation quality. Dependency links connect specific milestones; the
thirteen capability families are not a compulsory sequence.

After an authored record changes, run the documented refresh. The generated
HTML is a snapshot; JSON remains the editable source of truth. See
[SCHEMA.md](SCHEMA.md) for the fixed Hero → Goal → Milestone hierarchy.
