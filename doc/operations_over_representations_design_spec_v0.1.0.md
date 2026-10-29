---
title: "Operations over Representations"
subtitle: "A Map of Representation-Space Capabilities"
document_id: ORR-DESIGN-001
version: 0.1.0
status: "Working design specification — conceptual baseline"
date: 2026-10-18
scope: "Domain-agnostic operations on learned and externally supplied representations"
---

# Operations over Representations

**A Map of Representation-Space Capabilities**  
**Design specification · ORR-DESIGN-001 · Version 0.1.0 · 18 October 2026**

> **Governing question:** Given a representation of observations, what operations can we perform on it, what do those operations produce, and what evidence shows that the representation is suitable for each one?

## Document status and intended use

This document establishes the conceptual backbone of a long-running research and engineering project. It is **a design specification for organizing work**, not a claim that any capability has already been demonstrated, a commitment to one model architecture, or a complete implementation plan. The thirteen capability families and their identifiers are the stable initial taxonomy; methods, data sources, metrics, and implementations may evolve independently.

Current priorities, dependencies, acceptance criteria and work states are maintained in the [Hero records](../web/dashboard/README.md). The program and demonstrators below are illustrative; they do not impose additional prerequisites on that work plan.

Hero references use **Hx01–Hx21**, derived from their Hero numbers. **OR-01–OR-13**
are the Operations over Representations capability-family IDs; Hx01–Hx13 currently
own work for those respective families, Hx14 owns shared foundations, Hx15–Hx18
own encoders/providers, Hx19 owns dataloaders, Hx20 owns the Temporal Fusion
Transformer forecasting family, and Hx21 owns state fusion models. The [Hero work map](../web/dashboard/WORKSTREAMS.md) records
the mapping; Hx references name workstreams while OR references classify operations.

**Intended audiences:** researchers building representation models; engineers integrating open, pretrained, or custom encoders; people designing experiments, exploratory interfaces, indexes, agents, or downstream applications.

**Reading guide:** Sections 1–4 define the purpose, vocabulary, and concise map. Section 5 is the normative capability catalogue. Sections 6–10 specify composition, interfaces, evaluation, application lanes, and the incremental program of work. Appendices preserve the design decisions, original-to-revised mapping, templates, and references.

**Meaning of normative terms:** **MUST** means required for a reproducible, evidence-bearing result in this project; **SHOULD** is a strong default with documented exceptions; **MAY** is optional. These terms apply to experiments and project artifacts, not to the underlying scientific truth.

---

## 1. Purpose, thesis, and boundaries

### 1.1 Thesis

An embedding or learned latent state is not itself a solution. It is a representation on which operations may become possible: searching, describing structure, estimating outcomes, comparing alternatives, transforming, reconstructing, predicting evolution, selecting actions, or choosing what evidence to acquire next.

The project studies **operations over representations**, not merely the production of vectors. It asks whether a given representation contains and organizes the information needed for a specified operation and whether working in that representation improves the result relative to suitable alternatives.

**Core research unit:**

> **Operation + input/output contract + required representational properties + additional evidence or models + explicit test.**

This unit is intentionally smaller than a product and larger than a mathematical function: an operation can be evaluated, composed with others, and reused across domains and encoders.

### 1.2 Goals

1. Maintain a **coherent, extensible map** of major representation-space capability families.
2. Define **reusable contracts** for input, output, dependencies, evidence, and failure conditions.
3. Make **representation suitability operation-specific**, rather than assigning a single vague measure of embedding quality.
4. Enable the same catalogue to evaluate **frozen self-supervised encoders, external/open pretrained models, learned latent states, and aligned multimodal spaces**.
5. Support **composition**: individual capability families are building blocks, not isolated products.
6. Keep separate claims about **geometric properties, statistical prediction, causal effects, and decision quality**.
7. Preserve **reproducibility, data lineage, and historical comparability** as implementations evolve.
8. Provide a research agenda that can grow in breadth without requiring every branch to depend on all the others.

### 1.3 Non-goals and exclusions

- No claim that an embedding is a complete or invertible description of its source.
- No universal assumption that cosine similarity, Euclidean distance, or any single metric preserves the meaning relevant to every task.
- No claim that visually close points are equivalent, that low density implies an abnormal real-world event, or that latent arithmetic generally obeys symbolic logic.
- No causal claim inferred from proximity alone.
- No assumption that a more accurately predictable latent state must yield better task estimates or better decisions.
- No obligation to train an encoder: externally supplied embeddings are first-class inputs.
- No requirement for an interactive visualizer, vector database, autonomous agent, or any particular technology stack as part of the conceptual specification.
- No premature commitment to a domain-specific objective, including finance. Existing market/time-series work is a valuable application lane, not the definition of the project.

### 1.4 Design principles

**P1 — Question and output first.** Classify an operation by the question it answers and the result it produces, not by whichever algorithm implements it.

**P2 — Representation is relative to a task.** Suitability is demonstrated for a specified operation, population, metric, and operating condition.

**P3 — Interfaces before architecture.** A frozen encoder, an API embedding model, and a trained state model may each satisfy an operation's input contract, but not necessarily its evidence requirements.

**P4 — Separate observation, inference, and intervention.** Descriptive comparisons, model perturbations, and causal effects support different claims.

**P5 — Compose explicitly.** Record when one operation supplies data, candidates, state, uncertainty, or decisions to another.

**P6 — Preserve raw evidence.** Embeddings are indexed to source observations, not substitutes for provenance or the source whenever retention is lawful and practical.

**P7 — Learn from negative results.** An embedding's inability to support an operation is itself a useful finding if demonstrated against appropriate controls.

**P8 — Grow without taxonomy drift.** Stable capability identifiers survive renaming, implementation changes, and additions of new specializations.

---

## 2. Conceptual model and vocabulary

Let an observation (document, signal window, event, image, sensor reading, trajectory, or set) be denoted by \(x\), with contextual metadata \(c\). An encoder \(E\) maps it to a representation:

\[
z = E(x,c) \in \mathcal{Z}_E.
\]

An operation \(O_i\) may consume one or more representations, metadata, a comparison population \(D\), a metric \(d\), target labels \(y\), action descriptions \(a\), or an additional learned model \(M\):

\[
r_i = O_i(z, D, d, c, y, a, M;\theta_i).
\]

This is a **schema of possible inputs**, not a requirement that every operation consume every argument. The operation-specific contract defines the actual signature. Some operations produce another representation, some produce a score or estimate, some return source references, and others produce actions or generated observations.

### 2.1 Terminology

| Term | Definition used by this project |
|---|---|
| **Observation** | The source item or historical event being represented; may include time and context. |
| **Embedding / representation** | An encoded state of an observation or collection. A vector is common but not mandatory; a representation can be structured or probabilistic. |
| **Representation space** | The domain of valid representations with its associated encoder, preprocessing, geometry, and version. |
| **Latent state** | A representation intended to summarize state relevant to some downstream task, often temporal or decision-oriented. Not synonymous with every encoder output. |
| **Geometry** | A specified similarity/distance, neighborhood, topology, kernel, or learned relational structure; not an intrinsic guarantee of semantic meaning. |
| **Reference population** | The data and operating context against which comparisons, densities, anomalies, or changes are assessed. |
| **Readout / probe** | A mapping from a representation to an externally defined target; a limited-capacity probe is also a test of accessible information. |
| **Dynamics model** | A mapping that predicts subsequent representations or their distribution, conditioned on history and optionally actions. |
| **Decision policy** | A mapping from available state and decision context to an action. Its value is measured by realized or appropriately simulated objectives. |
| **Evidence record** | An immutable description of a claim, its data and controls, and the result of a reproducible evaluation. |
| **Capability family** | One of the thirteen categories below, defined by its primary question/output rather than a specific method. |

### 2.2 Distinguish three representation objects

The system MUST allow the following to be represented separately, even when an experiment chooses the same vector for more than one role:

1. **Observation embedding \(z_t^E\):** what an encoder extracts from the available observation or history.
2. **Task-dependent latent state \(s_t^G\):** what an optional state model preserves or transforms for a specified prediction or decision task.
3. **Predicted future state \(\hat{s}_{t+h}\):** a forecast produced by a dynamics model, not an observed encoding.

Avoid treating their distances, likelihoods, and coordinates as interchangeable by default. A cross-space mapping or an explicit shared geometry is necessary before comparisons between spaces can be interpreted.

### 2.3 Models and representation sources

The catalogue is compatible with:

- **Custom/frozen encoders:** time series, multimodal signals, domain-specific histories, or learned embeddings retained for repeated experiments.
- **Externally supplied/open models:** word, text, image, audio, and multimodal embeddings obtained from published models or APIs.
- **Additional learned heads:** readouts, clustering models, transition functions, state abstractions, decoders, adapters, policies, and acquisition rules.

A project SHOULD make it possible to evaluate the same operation against different encoders using the same admissible observations and splits, with declared information and capacity controls. Comparing task results does not require shared coordinates; an operation that combines or directly compares representations needs its own compatibility argument or mapping.

---

## 3. The capability map — one-page view

**The thirteen families are building blocks, not stages that must be completed in order.** Composition is a first-class use case: retrieval can ground generation; density can inform anomaly scoring; dynamics can inform planning; and exploration can select which examples or trajectories improve another family.

### A. Describe and organize observations

| ID | Capability | Question answered | Primary output |
|---|---|---|---|
| **OR-01** | Retrieval and similarity matching | Which existing observations are relevant or similar? | Ranked items, neighborhoods, matches |
| **OR-02** | Structure discovery and density modeling | How are observations organized in the space? | Partitions, prototypes, distributions |
| **OR-03** | Anomaly and novelty scoring | How unusual is this observation or transition? | Departure scores, flags |
| **OR-04** | Distribution shift monitoring | How has the represented population changed? | Shift statistics, change signals |

### B. Estimate outcomes and compare alternatives

| ID | Capability | Question answered | Primary output |
|---|---|---|---|
| **OR-05** | Task prediction and readouts | What externally defined quantity can be estimated from the representation? | Labels, values, ranks, predictive distributions |
| **OR-06** | Matched comparisons, sensitivity, and counterfactual analysis | What differs between comparable cases or specified alternatives? | Contrasts, sensitivities, justified effects |

### C. Transform representations and connect systems

| ID | Capability | Question answered | Primary output |
|---|---|---|---|
| **OR-07** | Latent transformations and relational operations | How can candidate representations be transformed or related? | Candidate embeddings, transformations, paths |
| **OR-08** | Compression, aggregation, and summarization | What compact representation retains information needed downstream? | Codes, aggregates, summaries |
| **OR-09** | Decoding and conditional generation | What observations can be reconstructed or produced under a representation? | Reconstructions, generated outputs |
| **OR-10** | Representation alignment and interoperability | How can two representation systems communicate or preserve correspondences? | Mappings, shared coordinates, validated links |

### D. Model evolution, choose actions, and acquire evidence

| ID | Capability | Question answered | Primary output |
|---|---|---|---|
| **OR-11** | Latent dynamics and forecasting | How does the represented state evolve? | Next states, trajectories, transition distributions |
| **OR-12** | Decision making, planning, and control | What action should be chosen under objectives and constraints? | Actions, plans, policies |
| **OR-13** | Information acquisition and exploration | What should be observed, labeled, revisited, or tried next? | Selected samples, experiments, exploration policies |

**Classification rule:** if an algorithm serves multiple questions, document the corresponding operations separately and connect them in a pipeline. For example, a vector index is an implementation for OR-01; it does not define the meaning or validity of OR-05 or OR-06.

---

## 4. Common contract and evidence standard

Each operation's permanent documentation MUST include an **Operation Contract** and a linked **Requirements-and-Evidence Record**. This keeps the concise map readable while locating methodological cautions beside the test they affect.

Reusable contracts can be drafted before selecting an experimental instance. Each experimental execution, including a diagnostic probe, needs the applicable Operation Contract and linked Requirements-and-Evidence Record bound to its actual inputs, artifacts, protocol and permissions; compact records can reference shared rules.

### 4.1 Operation Contract — minimum fields

| Field | Required content |
|---|---|
| **Operation ID and revision** | Stable family ID, particular operation name, semantic revision |
| **Primary question** | One question the operation intends to answer |
| **Input contract** | Representation references, context, history, comparison set, models, target/action definitions as needed |
| **Output contract** | Data type, semantics, units/scale, uncertainty if applicable |
| **Required representation structure** | Properties of the space necessary for the operation to make sense |
| **Other dependencies** | Labels, paired observations, actions, temporal ordering, objective, generator, decoder, reference distribution, etc. |
| **Scope of claim** | Exploratory description, task association/prediction, controlled model perturbation, or identified causal/decision effect |
| **Failure and abstention conditions** | Conditions under which the output should not be interpreted, or the operation should abstain |

### 4.2 Requirements-and-Evidence Record — minimum fields

| Field | Required content |
|---|---|
| **Hypothesis** | A falsifiable statement about a representation/operation on a defined population |
| **Data and provenance** | Dataset snapshot, observation identity, encoder/preprocessing/metric versions, temporal or group boundaries |
| **Controls and baselines** | Trivial, raw-feature, random/alternative representation, and task-specific baselines where meaningful |
| **Evaluation protocol** | Splits, leakage controls, resampling, negative controls, and how hyperparameters are selected |
| **Measures of success** | Primary metric(s), thresholds or comparison criterion defined before examining held-out results |
| **Robustness checks** | Repeats/seeds, uncertainty, subgroups, shift, sensitivity to geometry or representation version |
| **Results and limitations** | Numeric evidence, observed failure modes, and the strongest justified claim |
| **Reproduction pointer** | Code/config revision, model version, data manifest, seed and artifact locations |
| **Decision** | Promoted, exploratory, inconclusive, disproved in tested setting, or retired, with reason |

A failure to beat a strong baseline is **an admissible result**, not a reason to discard the experiment. Aesthetic 2D/3D plots and intuitive nearest neighbors are valuable for investigation but insufficient as performance evidence on their own.

Work completion, execution disposition and scientific verdict are separate. A valid negative or inconclusive evaluation can complete its scoped work; failed, invalid and admission-rejected attempts remain discoverable without being treated as negative scientific evidence.

### 4.3 General evaluation invariants

- **Comparable inputs:** compare encoders on the same admissible observations and information horizon, or explicitly account for differences.
- **Chronology/group separation:** avoid cross-time leakage, repeated-entity contamination, overlapping windows, and future-derived normalization, labels, or metadata when those would invalidate the question.
- **Dependency-aware time boundaries:** assess source availability and fitted-artifact eligibility under the declared claim mode; event timestamps alone do not establish temporal validity. Derived outputs retain their applicable input restrictions.
- **Versioned geometry:** metric, normalization, pooling, dimensionality, projector layer, and model version are part of the representation identity.
- **Visible baselines:** compare against raw descriptors, simple nonembedding alternatives, random encoders, or task-specific conventional methods as appropriate.
- **Uncertainty-aware evaluation:** report uncertainty and variability, especially when sample sizes are limited or data are dependent.
- **Explicit operating regime:** success on a validation population does not establish performance under arbitrary shift.
- **No magical semantics:** meaning attributed to an axis, cluster, transformation, or projection must be grounded in an external test.
- **No automatic promotion of causal claims:** descriptive or predictive evidence does not establish an effect of intervention.

---

## 5. Detailed capability contracts

These entries define **what belongs** in each family. Specific estimators are illustrations, not mandatory implementation choices.

### OR-01 — Retrieval and similarity matching

**Purpose.** Retrieve, filter, or rank historical/source items based on a query representation and a specified relevance relation. Enables semantic search, candidate deduplication, case-based reasoning, and retrieval-augmented workflows.

- **Input → output:** `query representation + candidate collection + similarity/index configuration → ranked source references with scores and metadata`.
- **Required structure:** a neighborhood or ranking relation that tracks task-relevant similarity; consistent query/candidate encoding or validated cross-space alignment.
- **Additional dependencies:** a candidate corpus, stable source references, inclusion/exclusion rules, duplicate policy, optionally relevance judgments.
- **Evidence:** recall@k, precision@k, nDCG/MRR where labels exist; inspected matches; index approximation loss; subgroup and temporal robustness; retrieval quality versus lexical/raw-feature and other baselines.
- **Failure checks:** proximity may reflect a nuisance feature; high-dimensional hubness; near-duplicates that reveal future labels; cross-version index mismatch; missing source evidence.
- **Starter experiment:** use a fixed set of queries with adjudicated relevant historical observations and compare embedding-based retrieval against a simple raw-feature or lexical baseline.

### OR-02 — Structure discovery and density modeling

**Purpose.** Describe neighborhoods, modes, partitions, connectivity, prototypes, and population concentration. Enables segmentation, cohort analysis, representative selection, and candidate regime discovery.

- **Input → output:** `representation collection + chosen geometric/statistical method → clusters, neighborhood graph, density model, prototypes or soft assignments`.
- **Required structure:** neighborhoods or distributional variation that is stable enough to study; a defensible metric and scale.
- **Additional dependencies:** sampling frame, normalization/metric, possibly reference metadata to interpret found groups.
- **Evidence:** cluster stability under seeds/subsampling; held-out neighborhood consistency; density calibration when feasible; external attribute enrichment; comparison with simple feature spaces.
- **Failure checks:** arbitrary partitions from a chosen algorithm; spurious density from dimensionality; visualization artifacts; accidental interpretation of a static cluster as a temporal regime.
- **Starter experiment:** compare clustering and local-neighborhood stability between learned embeddings and simple descriptors on a held-out population.

### OR-03 — Anomaly and novelty scoring

**Purpose.** Rank observations or transitions by their degree of departure from a defined reference distribution, local neighborhood, normal model, or expected transition.

- **Input → output:** `candidate + reference population/model + scoring rule → anomaly or novelty score; optional calibrated threshold/flag`.
- **Required structure:** a signal of novelty that is meaningfully related to the operation's definition of unusualness.
- **Additional dependencies:** reference population and its time/context, operational cost of false positives, known anomalies or designed perturbations when available.
- **Evidence:** precision–recall or ROC performance on known/simulated anomalies; false-alert rate; threshold stability; drift and subgroup testing; comparison with simple anomaly detectors.
- **Failure checks:** low likelihood/density is not automatically abnormality; rarity may be normal; high likelihood can be assigned to unfamiliar inputs; an anomaly score is not a fault diagnosis.
- **Starter experiment:** inject controlled perturbations and compare their rank under distance-, density-, and prediction-residual methods, reporting missed cases.

### OR-04 — Distribution shift monitoring

**Purpose.** Measure changes between reference and current populations of representations, with an explicit distinction between input-space shift and changes in downstream targets.

- **Input → output:** `reference cohort + comparison cohort + statistical discrepancy method → shift statistic, test/interval, explanation candidates`.
- **Required structure:** a stable coordinate/representation definition across cohorts, or a validated alignment allowing the comparison.
- **Additional dependencies:** chronological/group labels, sampling design, windows, test power expectations, optional outcome labels to study conditional drift.
- **Evidence:** sensitivity to known shifts; false-alarm rate under no-shift controls; bootstrapped/permutation tests where assumptions hold; correlation with downstream degradation as a separate evaluation.
- **Failure checks:** a shift in \(P(z)\) does not by itself imply a change in \(P(y\mid z)\), and stable \(P(z)\) does not exclude changing target relationships; model upgrades can mimic drift.
- **Starter experiment:** compare intentionally shifted versus stable held-out cohorts under a frozen encoder, then repeat after an encoder version change to measure spurious alerts.

### OR-05 — Task prediction and readouts

**Purpose.** Estimate an explicitly defined external target from representations. Includes classification, regression, ranking, probabilistic estimation, future outcome/risk estimation, and limited-capacity probes of accessible information.

- **Input → output:** `representation/history + target definition + readout → target estimate, ranking or predictive distribution`.
- **Required structure:** accessible information about the target and a readout with sufficient but controlled capacity.
- **Additional dependencies:** target labels, horizon, measurement/label validity, task loss, fair baseline conditions, optionally calibration data.
- **Evidence:** appropriate held-out task metrics, calibration and error decomposition, capacity-matched probes, chronological/generalization testing, comparison against raw features and simple alternatives.
- **Failure checks:** a powerful readout can hide weaknesses of a representation; overlap/leakage can create false predictability; target performance is not equivalent to future-latent prediction.
- **Starter experiment:** freeze each encoder, fit the same small readout on identical permitted observations, and compare held-out learning curves and calibration.

### OR-06 — Matched comparisons, sensitivity, and counterfactual analysis

**Purpose.** Compare observations or hypothetical alternatives that differ in specified ways. **Three claim levels must remain distinct:**

1. **Matched observational contrast:** similar cases with different observed features/outcomes; association only.
2. **Controlled model perturbation:** modify an input or condition and observe how the encoder/readout behaves; a statement about the model.
3. **Identified intervention effect:** estimate what changing a factor would cause under defensible design/identification assumptions; a causal statement about the system studied.

- **Input → output:** `cases + match constraints or perturbation/intervention definition + comparison rule → contrasts, sensitivities, conditional effects`.
- **Required structure:** task-relevant matching/sensitivity, plus additional causal assumptions for level 3.
- **Additional dependencies:** measured covariates, source context, intervention semantics, outcome measurement, identification strategy as applicable.
- **Evidence:** matching balance/overlap; perturbation validity and stability; negative controls; causal diagnostics and sensitivity analysis if a causal effect is claimed.
- **Failure checks:** nearest neighbors need not differ only in a controlled factor; source factors may be omitted by the encoder; a model response need not predict a real intervention.
- **Starter experiment:** define a known synthetic intervention with ground truth and separately report neighbor contrast, controlled perturbation response, and actual identifiable effect estimates.

### OR-07 — Latent transformations and relational operations

**Purpose.** Construct or compare representations by interpolation, extrapolation, analogy, composition, or learned transformations. Investigate whether meaningful attributes or relations can be manipulated in latent coordinates.

- **Input → output:** `one or more representations + transformation/path specification → candidate representations, paths, or relational scores`.
- **Required structure:** a demonstrated correspondence between the proposed geometric operation and a change in the represented object or relation.
- **Additional dependencies:** anchor pairs, valid-domain constraints, decoders or external validators when output meaning must be checked.
- **Evidence:** relation accuracy on held-out examples; valid-intermediate rate; locality/smoothness where appropriate; comparison with transformations in input space and learned baselines.
- **Failure checks:** Euclidean line segments may leave the data manifold; analogies are not universal vector laws; rotated, rescaled, or nonlinear coordinate changes can alter naive arithmetic.
- **Starter experiment:** compare simple vector arithmetic with a learned transformation on relation pairs, verifying results against source-level labels or decoded outputs.

### OR-08 — Compression, aggregation, and summarization

**Purpose.** Store or compute with a more compact representation of an observation or collection, retaining the information necessary for a defined downstream task. Includes pooling, quantization, sketches, and set/trajectory aggregation.

- **Input → output:** `observations/representations + compression or aggregation rule → compact code, summary or representative vector/set`.
- **Required structure:** enough task-relevant information survives the transformation; code size and costs are actually reduced if efficiency is claimed.
- **Additional dependencies:** target task, acceptable distortion, serialization/storage conventions, reconstruction target when relevant.
- **Evidence:** memory and compute savings, rate–distortion tradeoff, impact on retrieval/readouts/dynamics, robustness to aggregation scale and ordering.
- **Failure checks:** dimensionality alone is not a bit-cost guarantee; a compact representation may irreversibly discard timing, phase, rare events, or individual distinctions.
- **Starter experiment:** vary embedding quantization and pooling depth; measure storage footprint and changes in downstream retrieval and readout utility.

### OR-09 — Decoding and conditional generation

**Purpose.** Reconstruct source observations or generate new observations conditioned on representations; include completion and controlled synthesis when a suitable model exists.

- **Input → output:** `representation + decoder/generator + optional condition/seed → reconstruction or generated observation/distribution`.
- **Required structure:** the generator has learned an interpretable relationship between its conditioning space and feasible outputs.
- **Additional dependencies:** paired training examples or generative model, constraints, source-domain validator, provenance policy.
- **Evidence:** reconstruction error for reconstruction tasks; realism, diversity, constraint satisfaction and conditional faithfulness for generation; leakage/memorization checks.
- **Failure checks:** multiple possible sources may share an embedding; plausible generation is not proof that a latent point corresponds to an observed real-world state; unsupported conditioning may hallucinate detail.
- **Starter experiment:** reconstruct known items at different compression levels and separately evaluate conditional generation against source-grounded constraints.

### OR-10 — Representation alignment and interoperability

**Purpose.** Map between encoders, versions, modalities, or training runs so that relevant observations can be compared or consumed across systems.

- **Input → output:** `source representation + alignment mapping/anchors + target space definition → target-space representation or correspondence`.
- **Required structure:** enough shared identifiable information and correspondence to preserve the relevant operation after mapping.
- **Additional dependencies:** anchor/paired data, correspondence assumptions, normalization and version metadata, target-space evaluation sets.
- **Evidence:** retrieval/neighbor preservation, downstream task retention, mapping residuals, per-subgroup behavior, preservation across updates and unseen anchors.
- **Failure checks:** information lost by the source encoder cannot be recreated by alignment alone; globally small mapping error may still damage nearest-neighbor ranking or decisions.
- **Starter experiment:** align two frozen encoder versions using an anchor subset; test cross-version retrieval and a held-out readout before and after alignment.

### OR-11 — Latent dynamics and forecasting

**Purpose.** Estimate how a represented state evolves across a specified time horizon, conditioned on the available history and external signals/actions when applicable. Includes deterministic and probabilistic state-transition models.

- **Input → output:** `state/history + horizon + permitted exogenous variables/actions + dynamics model → predicted future latent state(s) or distribution`.
- **Required structure:** enough predictive state information for the horizon; compatible training and evaluation representations.
- **Additional dependencies:** chronological trajectories, horizons, transition pairing, action histories if action-conditioned, loss and uncertainty conventions.
- **Evidence:** one-step and multi-step errors; uncertainty calibration; stability of rollouts; comparisons to persistence, linear and direct-encoder baselines; downstream relevance assessed separately.
- **Failure checks:** a low latent-coordinate error need not imply good task utility; unobserved factors and partial observability limit predictable evolution; long rollouts compound error.
- **Starter experiment:** compare a simple state-history transition rule, persistence, and direct prediction of future encoder embeddings on strictly held-out chronological trajectories.

### OR-12 — Decision making, planning, and control

**Purpose.** Select actions under objectives, constraints, uncertainty, and information available at decision time. May be direct-policy or planning-based, with or without a learned dynamics model.

- **Input → output:** `state/decision context + action set + objective/constraints + optional predictive model → chosen action, policy, or plan`.
- **Required structure:** information sufficient for the decision; when planning through a model, useful action-conditioned transitions and objective estimates.
- **Additional dependencies:** action semantics, costs/rewards/risks, feasibility and safety constraints, evaluation environment or logged-action assumptions.
- **Evidence:** realized or validly simulated decision performance; policy baselines; constraint violations; robustness to state/dynamics error; off-policy evaluation assumptions where applicable.
- **Failure checks:** visually attractive latent trajectories can be physically infeasible; correlation does not imply actionability; predictive state accuracy does not guarantee superior control.
- **Starter experiment:** test a small action-selection task with known dynamics/objective and compare direct policy, model-based planning, and simple heuristic controls.

### OR-13 — Information acquisition and exploration

**Purpose.** Select what to observe, label, sample, revisit, or experiment on next to improve knowledge or performance under a finite budget. Includes active learning, diverse sampling, replay curation, and intrinsic-motivation exploration.

- **Input → output:** `candidate observations/actions/trajectories + current representation/model + acquisition budget/criterion → selection priority, sample batch, experiment, or exploration action`.
- **Required structure:** geometry or uncertainty linked to actual learning value, coverage, or exploration objective.
- **Additional dependencies:** candidate pool or environment, label/measurement costs, replay constraints, diversity/uncertainty estimates, evaluation budget.
- **Evidence:** learning curves per acquired label/interaction; coverage versus redundancy; generalization after replay; regret or downstream improvement where meaningful; comparison with random/stratified acquisition.
- **Failure checks:** geometric diversity need not be behavioral diversity; unpredictable randomness may yield persistent curiosity rewards without learning; exploration can be unsafe or expensive.
- **Starter experiment:** compare random, uncertainty, and embedding-diversity sampling at identical budgets, recording held-out improvement rather than selection score alone.

---

## 6. Composition: capabilities as building blocks

The map deliberately supports multi-operation systems. A **pipeline** is a versioned directed graph of operation instances. Its nodes specify inputs/outputs, assumptions, and evidence; edges specify which outputs are passed downstream and what transformations occur between them.

### 6.1 Reference composition patterns

| Pattern | Pipeline | Resulting system and primary evaluation |
|---|---|---|
| **Grounded generation** | OR-01 → OR-09 | Retrieve source evidence, condition generation, and evaluate answer fidelity/grounding separately from retrieval quality. |
| **Novelty monitoring** | OR-02 → OR-03 → OR-04 | Model structure, score new cases, then track distributional change and operational false alerts. |
| **Case-based estimation** | OR-01 → OR-06 → OR-05 | Retrieve comparable cases, inspect contrasts, then estimate a target; distinguish predictive and causal claims. |
| **Compression-preserving search** | OR-08 → OR-01 | Quantize or aggregate representations while monitoring retrieval degradation versus savings. |
| **Cross-encoder retrieval** | OR-10 → OR-01 | Align spaces and measure target-side relevance, not only mapping distance. |
| **Predictive state** | OR-05 + OR-11 | Separately evaluate external-target estimates and latent-state forecasting from a common state representation. |
| **Model-based action** | OR-11 → OR-12 | Use action-conditioned dynamics to rank feasible actions and evaluate downstream decisions. |
| **Exploration and replay** | OR-02 + OR-03 + OR-13 → model update | Use coverage/novelty to select training or replay samples; evaluate post-update learning, not just diversity. |
| **Transform-and-validate** | OR-07 → OR-09 → OR-05 | Transform a latent point, decode it, and check semantic/task properties with independent validators. |

### 6.2 Pipeline design rules

1. **Every edge MUST state its data semantics.** A latent state, a target probability, and a generated observation are not interchangeable values.
2. **A downstream capability MUST inherit applicable restrictions** of its upstream sources (e.g., access control, temporal horizon, encoder version, missing variables).
3. **A pipeline's final claim MUST be tested end to end.** Passing isolated unit tests does not establish overall utility.
4. **Intermediate evidence SHOULD be retained.** It makes failure analysis possible when downstream performance degrades.
5. **Loops are allowed.** For example, OR-13 can select new evidence that improves OR-05 or OR-11, but the feedback loop needs a new data/model/version record.

### 6.3 Minimal example — search, compare, predict

```text
Source observations + context
            │
            ▼
    Encoder E (versioned)
            │
     Representation z
      ┌─────┴─────────┐
      ▼               ▼
 OR-01 retrieve    OR-05 readout
      │               │
      ▼               ▼
 OR-06 compare     target estimate
      │               │
      └───────┬───────┘
              ▼
      Evidence record
```

The retrieved cases and model prediction may inform one another without being treated as equivalent evidence. A similar pipeline can use a temporal state model and dynamics without changing the conceptual contracts.

---

## 7. Data model and interoperability contracts

These are **logical records**, not a mandated database schema, programming language, or serialization format. Implementations can begin in files and progress to indexes/services as needed.

### 7.1 Core records

**ObservationRef**

```yaml
observation_id: immutable content- or registry-based ID
source_uri: pointer to source or access-controlled archive
domain: text | time_series | image | audio | multimodal | other
subject_or_group_id: optional isolation/grouping identity
event_time: optional observation time
available_at: optional time the information became available
context_schema: versioned definition of available metadata
content_fingerprint: source/version checksum
access_policy: source restrictions, where applicable
```

**SpaceRef**

```yaml
space_id: stable ID
encoder_id: selected model/provider identity + immutable revision/hash when exposed
input_preprocessing_id: preprocessing version
layer_or_head: exported representation point
pooling_or_aggregation: method, if applicable
dimension_and_shape: coordinate specification
normalization: e.g. unit norm, z-score, none
metric: cosine | L2 | learned | other (explicit parameters)
training_scope: known train population/time or declared unknown
known_limitations: recorded domain/coverage restrictions
```

When a hosted revision is undisclosed, record that uncertainty and the evidence scope explicitly. Space/configuration identity remains distinct from fitted-artifact and execution/response identity. Replaying retained response artifacts does not establish that a later service call uses the same hidden revision or returns the same outputs.

**EmbeddingRecord**

```yaml
embedding_id: immutable ID
observation_id: link to ObservationRef
space_id: link to SpaceRef
representation_artifact: vector, structured state, or distribution values / artifact reference
context_and_time: source-aligned metadata
encoder_run_id: execution provenance
quality_flags: missingness, truncation, OOD, etc. (when available)
```

**OperationRun**

```yaml
operation_run_id: immutable ID
operation_family: OR-01 ... OR-13
operation_spec_version: versioned instance contract
input_references: observations, embeddings, models, cohorts
parameters: including metric and calibration choices
output_references: generated artifacts/results
execution_provenance: code, environment, seeds, model revisions
warnings: detected assumption violations or abstentions
```

**EvidenceRecord**

```yaml
evidence_id: immutable ID
claim: falsifiable question and interpretation level
operation_run_ids: runs being compared
population_and_split_id: exact held-out protocol
baseline_ids: comparison definitions
primary_metrics: results and uncertainty
robustness_checks: repeats and adverse conditions
limitations: known failure modes and untested claims
status: exploratory | supported_in_scope | inconclusive | refuted_in_scope | retired
artifacts: reproducible results and report pointers
```

### 7.2 Compatibility rules

- An embedding MUST never be identified by vector dimension alone.
- Re-embedding the same source with a different encoder/preprocessing version creates a **new embedding record**; it must not silently overwrite the earlier one.
- Search, density, and shift comparisons MUST declare which `SpaceRef` and metric they use.
- An operation combining spaces MUST provide a defined alignment/adapter or document why direct comparison is valid.
- Temporal data MUST distinguish **event time** from **information-availability time** when leakage or decision timing matters.
- Results SHOULD reference original observations to allow investigation, while respecting retention and privacy restrictions.
- Generated data MUST be labeled as generated and not silently inserted into real-observation cohorts.
- Annotations and target definitions SHOULD be versioned independently from embeddings.

### 7.3 Security and stewardship

Embeddings can preserve sensitive or identifying information even if the source text or signal is not directly visible. Treat representation storage as a data-governance question, not inherently anonymized data. Access controls, retention policy, consent/license scope, dataset provenance, and safe deletion paths belong to each dataset/space. New experiments involving identifiable, confidential, or security-sensitive material require an explicit review appropriate to their context.

---

## 8. Representation suitability: cross-cutting research program

There is **no single universal embedding quality score**. The quality question is always relative to an operation and an evidence standard. Geometry is a hypothesis about useful relationships, not proof of them.

### 8.1 Property-to-operation examples

| Property or question | Particularly relevant families | Evidence to seek |
|---|---|---|
| Do relevant items share neighborhoods? | OR-01, OR-02, OR-06 | Retrieval labels, matching diagnostics, neighborhood preservation |
| Does local density track useful notions of familiarity? | OR-02, OR-03, OR-13 | Novelty labels, false alert rates, information gained after selection |
| Are representation changes comparable across cohorts? | OR-04, OR-10 | Version-aware shift tests, stable anchors, outcome checks |
| Are target variables recoverable with little capacity/data? | OR-05, OR-08 | Frozen low-capacity probe curves; matched-input baselines |
| Can controlled changes be represented and distinguished? | OR-06, OR-07 | Paired data, synthetic interventions, semantic validity checks |
| Is necessary information retained under compression? | OR-08, OR-09 | Rate–distortion and downstream utility |
| Do representations support valid generation? | OR-07, OR-09 | Constraint satisfaction, source-based validation |
| Are states predictable at useful horizons? | OR-11 | Time-separated transition/rollout tests |
| Does using the state improve decisions? | OR-12 | Decision outcomes versus policy baselines, constraint satisfaction |
| Does acquisition improve future learning per unit cost? | OR-13 | Budget-matched learning curves and held-out improvement |

### 8.2 Geometric cautionary examples

**Proximity is not equivalence.** Two observations can be close because an encoder retains a shared nuisance property while ignoring the variable relevant to the task.

**Density is not semantic normality.** A model's density score need not correspond to human familiarity or fault probability; this has been observed for deep generative models.

**Interpolation is not guaranteed to remain valid.** If valid states lie on a curved manifold (imagine a circle), the straight segment between valid points can traverse invalid regions.

**Alignment cannot recover erased distinctions.** If a source encoder maps two different observations to the same code, a deterministic adapter receiving only that code cannot distinguish them without additional information.

**Projected plots are lenses, not the space itself.** PCA, UMAP, t-SNE, and other low-dimensional views can expose patterns but may alter distances/neighborhood relationships. All critical findings require validation in the operational representation, not just the display projection.

### 8.3 Evidence levels

Use restrained, scoped language:

1. **Exploratory:** a pattern appears in visualizations or informal examples.
2. **Measured:** a predefined task metric was computed under a recorded protocol.
3. **Supported in scope:** robust improvements survive appropriate controls, held-out tests, and uncertainty assessment in the stated operating regime.
4. **Transferred:** the same kind of result persists across multiple independent populations, conditions, or encoders.

These levels do not automatically imply causality or safe autonomous use. A result can be supported at one level and remain exploratory for another claim.

---

## 9. Distinctions the project MUST preserve

### 9.1 Three noninterchangeable interfaces

| Interface | Typical mapping | Answers | Evidence |
|---|---|---|---|
| **Target readout (OR-05)** | \(z_{\le t}\mapsto\hat{y}_{t+h}\) | What external target/outcome can be estimated? | Target accuracy, calibration, robustness and baseline lift |
| **Latent dynamics (OR-11)** | \(s_{\le t},a_{\le t}\mapsto\hat{s}_{t+h}\) | How will represented state evolve? | Multi-horizon latent prediction and rollout quality |
| **Decision policy (OR-12)** | \((s_{\le t},\mathcal{A},J,C)\mapsto a_t\) | What action best serves objective \(J\) under constraints \(C\)? | Realized/validly simulated outcomes and violations |

The signatures are representative, not fixed architectures: readouts may need context, dynamics may not be action-conditioned, and policies may act directly without a latent world model.

**Key consequence:** reliable future-state prediction is not a prerequisite for every useful target readout; and good future-state prediction does not establish policy improvement. Each interface needs separate baselines and outcome measures.

### 9.2 Three levels of comparisons and counterfactual claims

| Level | What is changed/compared? | Valid claim without further assumptions |
|---|---|---|
| **Observed comparison** | Matched historical observations | Descriptive association/contrast |
| **Model response** | Controlled change to input or condition | Sensitivity of the specified model |
| **Identified intervention** | Intervention under a valid experiment/causal design | Causal effect within identification scope |

**Rule:** OR-06 MUST label its claim level. Merely finding a nearby embedding does not isolate a controlled factor.

### 9.3 “Suitable for which operation?” is a required question

Before declaring a representation good, record:

1. The operation and target output.
2. The population/horizon and permitted information.
3. The geometry or model being relied on.
4. The strongest relevant baseline.
5. The validity test and an observable failure criterion.

A result that answers only “the points look nicely separated” does not yet answer the research question.

---

## 10. Program of work and initial implementation slices

The project should grow as a **portfolio of bounded experiments**, with shared contracts and reusable datasets. The following phases are illustrative rather than a mandated dependency graph or calendar commitment. Use the current Hero records for priorities and acceptance criteria; implement only the shared prerequisites needed by an admitted path.

### Phase 0 — Foundational charter and registry

**Deliverables:** this design specification; capability identifiers; at least one dataset/observation manifest; a `SpaceRef`; experiment/evidence templates; a clearly defined benchmark protocol.

**Exit criterion:** a second person can tell exactly which observations, encoder version, metric, and split produced an embedding collection.

### Phase 1 — Static geometry and retrieval

**Candidates:** OR-01, OR-02, OR-03, OR-08. Build indexes, neighborhoods, prototypes, anomaly baselines, and a minimal representation inspector.

**Exit criterion:** report reproducible search/structure/novelty metrics with source-grounded examples and at least one nonembedding baseline.

### Phase 2 — Stability and external targets

**Candidates:** OR-04, OR-05, OR-10. Compare frozen encoders/versions, monitor shifts, and benchmark limited-capacity probes/readouts.

**Exit criterion:** establish which practical operation benefits from each representation and which comparisons are invalidated by version/metric change.

### Phase 3 — Controlled comparisons and transformation

**Candidates:** OR-06, OR-07, OR-09. Start with synthetic or paired data where semantics and interventions can be checked.

**Exit criterion:** clearly separate descriptive, model-perturbation, and causal claims; demonstrate valid transformations/generation or report where they fail.

### Phase 4 — Temporal and decision-making operations

**Candidates:** OR-11, OR-12, OR-13. Study state evolution, action-conditioned transitions where available, budgeted acquisition, and decision quality with appropriate environments.

**Exit criterion:** demonstrate separate results for target estimation, state prediction, and decision outcomes. Do not use the same metric as proof of all three.

### Phase 5 — Composed applications and portability

**Deliverables:** end-to-end pipelines; cross-encoder or cross-domain transfer tests; stable interfaces; case studies with explicit limits; a research backlog updated from evidence.

**Exit criterion:** an end-to-end task can be reproduced, diagnosed at each operation boundary, and compared fairly to a baseline system.

### 10.1 Suggested first end-to-end demonstrator

Keep the first demonstration deliberately small:

1. Select a manageable dataset with source references and a groundable notion of relevance or target labels.
2. Obtain embeddings from **two different representation sources** (for example, a custom frozen encoder and a suitable external model, or two versions of one model).
3. Register each as a distinct `SpaceRef`, preserving preprocessing and metric.
4. Execute **OR-01 retrieval**, **OR-02 neighborhood/structure analysis**, and **OR-05 small readout** under shared, valid splits.
5. Record requirements, metrics, negative examples, and a reproducibility manifest.
6. Demonstrate at least one composition—e.g. retrieval feeding a comparison or prediction—and evaluate the final result.

This is one possible demonstration, not the minimum required first path. A smaller admitted composition can establish its own bounded engineering and research conclusions without implementing every operation or provider shown here.

### 10.2 Application lanes (examples, not restrictions)

**Lane A — Time-series/state representations.** Frozen self-supervised signal embeddings, optional task-specific state heads, temporal histories, state transitions, risk readouts, replay sampling. Particularly relevant families: OR-02/03/04/05/11/12/13. Keep the earlier market-dynamics research plan independently scoped; this project supplies general contracts and evaluation ideas, not a replacement for that roadmap.

**Lane B — Text and open-model embeddings.** Words, phrases, documents, and other externally embedded items for search, clustering, analogy tests, summarization, semantic exploration, and cross-encoder alignment. Particularly relevant: OR-01/02/05/07/08/09/10.

**Lane C — Multi-modal and sensor observations.** Compare images, signals, text, or trajectories across compatible spaces; test cross-modal retrieval, alignment, anomaly detection, and contextual prediction. Particularly relevant: OR-01/03/05/09/10/11.

The same catalogue applies to each lane, but the evidence is not automatically transferable between them.

### 10.3 Interfaces and visual exploration

An optional interactive explorer may display neighborhoods, clusters, source examples, projected views, analogical transitions, temporal paths, and pipeline provenance. It SHOULD show the **actual metric, encoder version, and projection method**, and SHOULD offer inspection of original items. A 2D/3D view is a useful instrument for hypothesis generation, not a substitute for evaluation in the original feature space.

---

## 11. Research governance and evolution

### 11.1 Stable identifiers and changes

- The **OR-01 ... OR-13** identifiers belong to the capability families and SHOULD remain stable across documentation versions.
- New specific algorithms are **operation instances** under a family, not automatically new families.
- A new family is justified when its **primary question and output** cannot be expressed coherently by the existing categories; relatedness alone is insufficient.
- A future split or merge MUST include a migration table and preservation of old evidence references.
- This specification uses document versions separately from model, dataset, operation, and pipeline versions.

### 11.2 Research decision log

Maintain an append-only log per material decision:

```text
Decision ID:
Date:
Question:
Options considered:
Chosen position and reason:
Evidence currently available:
What would change the decision:
Affected capability IDs:
```

### 11.3 Claims and reporting

A project report SHOULD open with the operation being tested, the population and representation, and the strongest result and limitation. Prefer statements such as **“OR-01 retrieval improved recall@k on dataset X under encoder Y and split Z”** to **“the embedding understands X.”** An open problem, failed assumption, or negative baseline result is a valid research artifact.

### 11.4 Known open decisions — deliberately unresolved

- Which first datasets and evaluation slices have sufficient relevance/target labels?
- Which encoder sources are available, with what rights, costs, and reproducibility constraints?
- How broad should “representation” be in practice: fixed vectors first, then structured sequences/probabilistic states?
- Which operations require calibrated uncertainty, and how should calibrated scores be serialized?
- At what point do density indices, approximate search, and experiment storage deserve dedicated infrastructure?
- How should partially overlapping modalities/encoders be aligned when no fully paired dataset exists?
- What is the minimum empirical evidence required to promote an operation to a reusable component?
- Which real interventions or controlled simulations are available for OR-06 and OR-12?
- What are the resource budgets and privacy constraints for exploratory pipelines?

The answers belong in future project decisions, not speculative commitments in version 0.1.0.

---

## Appendix A — Mapping from the original eleven themes

The revised map **preserves all eleven original themes**. It clarifies two boundaries that had been combined or omitted: task prediction is added, and latent world-model dynamics is separated from planning/control. Thus, **11 original categories → 13 families**.

| Original theme | Revised family / treatment |
|---|---|
| Retrieval and nearest-neighbor operations | **OR-01** Retrieval and similarity matching |
| Clustering and density estimation | **OR-02** Structure discovery and density modeling |
| Anomaly and outlier detection | **OR-03** Anomaly and novelty scoring |
| Distribution drift monitoring | **OR-04** Distribution shift monitoring |
| Interpolation, extrapolation, and analogical reasoning | **OR-07** Latent transformations and relational operations |
| Counterfactual comparison | **OR-06** Matched comparisons, sensitivity, and counterfactual analysis; separates three levels of claims |
| Compression and summarization | **OR-08** Compression, aggregation, and summarization |
| Conditional generation and decoding | **OR-09** Decoding and conditional generation |
| World-model dynamics and planning | Split into **OR-11** Latent dynamics and **OR-12** Decision making, planning, and control |
| Intrinsic motivation and curiosity | Broadened to **OR-13** Information acquisition and exploration |
| Representation alignment | **OR-10** Representation alignment and interoperability |
| *(Missing as an independent family)* | Added **OR-05** Task prediction and readouts |

**Why the segmentation changed:** The four top-level groups orient the reader by purpose. Families are classified by the question they answer and the output they produce. This avoids placing similarity algorithms, predictive models, and decision objectives at a single undifferentiated level, while still allowing them to compose.

---

## Appendix B — Blank experiment card

Copy this template once for every serious operation test. An operation may have many experiment cards across datasets and representations.

```markdown
# Experiment: <name>

- Experiment ID:
- Date / owner:
- Capability family (OR-XX):
- Operation implementation / version:
- Decision question:
- Hypothesis (falsifiable):
- Claim level: descriptive | predictive | model-response | causal | decision

## Inputs and dependencies
- Observation population / source manifest:
- Allowed input information and temporal horizon:
- SpaceRef / encoder / normalization / metric:
- Auxiliary targets, actions, metadata, or models:
- Input/output contract:

## Required representation properties
- Which structure is hypothesized to matter?
- What evidence would falsify its suitability?

## Evaluation
- Train/validation/test split and contamination controls:
- Baselines and equal-information controls:
- Primary metric and success criterion:
- Secondary metrics and calibration:
- Seeds, confidence intervals, robustness/subgroup checks:
- Data/code/model/output provenance:

## Results and interpretation
- Main result (with uncertainty):
- Negative examples / failure modes:
- Alternative explanations and limitations:
- Strongest claim justified by the experiment:
- Decision: exploratory | supported_in_scope | inconclusive | refuted_in_scope | retired
- Reproduction references:
- Follow-on experiment:
```

---

## Appendix C — Worked evidence-card example

### OR-01 historical signal retrieval

**Question:** Does a self-supervised signal embedding retrieve historical windows that are more relevant to the current window than simple signal descriptors?

| Requirement | Example contract |
|---|---|
| **Inputs** | Query window; frozen encoder embedding; historical candidate windows with timestamps |
| **Outputs** | Top-k window IDs, similarity scores, retained source evidence |
| **Required structure** | Similarity must relate to externally defined signal behavior, not merely amplitude or metadata |
| **Additional information** | Allowed preprocessing, candidate chronology, permitted subject/asset overlap, relevance definition |
| **Baselines** | Raw standardized descriptors and a trivial recent-window match; optionally random encoder |
| **Primary evidence** | Held-out recall@k or relevance ranking plus source-level inspection |
| **Leakage control** | Exclude overlapping future windows and any data unavailable at query time |
| **Failure interpretation** | Good visual neighborhoods without relevance lift do not establish retrieval utility |

This example is illustrative. The experiment becomes admissible only once relevance is operationally defined and the protocol is instantiated with actual data and results.

---

## Appendix D — Related prior research and references

These sources are **starting points for method design and critical reading**, not evidence that a particular encoder in this project has succeeded. They are cited for the concepts indicated and do not prescribe a technology stack.

1. Chen et al. (2020), *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). Representation learning and downstream probing. <https://proceedings.mlr.press/v119/chen20j.html>
2. Gretton et al. (2012), *A Kernel Two-Sample Test*. Comparing distributions using maximum mean discrepancy and statistical tests. <https://www.jmlr.org/papers/v13/gretton12a.html>
3. Nalisnick et al. (2018), *Do Deep Generative Models Know What They Don't Know?* Limits of likelihood as a proxy for semantic novelty. <https://arxiv.org/abs/1810.09136>
4. Shalit, Johansson, and Sontag (2017), *Estimating Individual Treatment Effect: Generalization Bounds and Algorithms*. Representation learning for causal estimation under assumptions. <https://proceedings.mlr.press/v70/shalit17a.html>
5. Kingma and Welling (2013), *Auto-Encoding Variational Bayes*. Latent-variable modeling and decoding. <https://arxiv.org/abs/1312.6114>
6. Conneau et al. (2017), *Word Translation Without Parallel Data*. Example of cross-space alignment. <https://arxiv.org/abs/1710.04087>
7. Hafner et al. (2019), *Learning Latent Dynamics for Planning from Pixels*. Action-conditioned dynamics and planning as distinguishable components of a larger system. <https://proceedings.mlr.press/v97/hafner19a.html>
8. Pathak et al. (2017), *Curiosity-driven Exploration by Self-supervised Prediction*. Exploration with prediction error in learned features. <https://arxiv.org/abs/1705.05363>
9. Burda et al. (2018), *Large-Scale Study of Curiosity-Driven Learning*. Study of prediction-based intrinsic rewards and limitations in stochastic settings. <https://arxiv.org/abs/1808.04355>
10. Sener and Savarese (2018), *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. Example of geometric coverage for sample selection. <https://arxiv.org/abs/1708.00489>

---

## Appendix E — Decisions preserved from the design conversation

These are the design decisions adopted for this starting document:

1. **Name:** *Operations over Representations*; subtitle *A Map of Representation-Space Capabilities*. The working name foregrounds operations rather than vectors in isolation.
2. **Count and structure:** thirteen families in four groups, characterized by primary question and output.
3. **Map versus methods:** short, approachable category definitions in the map; caveats, assumptions, and evaluation requirements live in detailed contracts and evidence cards.
4. **Composition:** the families are interoperable building blocks, with concrete examples and explicit pipeline-edge requirements.
5. **Three separate predictive/decision interfaces:** external target readout, latent-state dynamics, and action policy remain independently testable.
6. **Counterfactual discipline:** distinguish nearby observational comparisons, controlled changes to a model, and identified intervention effects.
7. **Operation-specific suitability:** every claim must answer “suitable for which operation?” against a defensible baseline.
8. **Embeddings from different sources:** frozen custom self-supervised encoders and external/open pretrained models both fit the common framework.
9. **Evolution:** this is an enduring project charter with stable IDs, templates, and explicit unresolved decisions—not a declaration of completed research.

---

## Change log

| Version | Date | Status | Summary |
|---|---|---|---|
| **0.1.0** | 2026-10-18 | Working conceptual baseline | Established title, thirteen capability families, four-group segmentation, contracts, evidence standard, composition patterns, data model, application lanes, roadmap, and research decision rules. |
| **0.1.0 clarification** | 2026-10-18 | Aligned with maintained Hero planning | Clarified illustrative plans, execution binding, outcome distinctions, dependency-aware time boundaries and hosted revision uncertainty; capability IDs and logical record field names are unchanged. |

**End of initial design specification.**
