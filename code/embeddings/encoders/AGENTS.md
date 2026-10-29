# Shared encoder Hero instructions

Applies to **Hx15–Hx18** and every encoder family/provider maintained here.
Read this alongside the repository instructions, the assigned Hero JSON and
the family README. Hero JSON holds current goals, acceptance and work states;
this file defines the shared development direction.

The [component delivery and application ownership contract](COMPONENT_DELIVERY.md)
defines the producer/consumer boundary, reusable learning semantics, fully bound
application recipes, raw versus application artifacts and release acceptance.

## Develop reusable components

Mityane discovers, compares and develops encoder families and representation
providers as **reusable tools**. Deliver usable components, supported
configuration, explicit contracts and evidence that another researcher or
application can construct or call them. A finished application model or a
particular trained checkpoint does not define this workstream's deliverable.

Mityane actively researches and compares encoder architectures and training
methods, developing new versions to seek better representations. Configurable
construction, training and export support that continuing discovery and make
its results reusable. Generalizing a version for delivery does not end the
versioned research search or establish a new quality winner. Meaningful
architecture or learning-method candidates retain explicit version identities;
configuration choices, seeds and checkpoints retain their separate identities.

Mityane owns research training and comparative evaluation for that discovery,
alongside bounded engineering verification. Application projects such as Cuwacunu own
substantial application training, final docking and end-to-end application
evaluation. Keep the portable training machinery with its encoder component;
application data, budgets and orchestration belong to the consumer.

## Channel axis

cuwacunu_torch `interval_type_e` is the standard for the causal and raw patch
bottleneck encoders, in every version. One window is one instrument. The
channel axis holds that instrument's time resolutions, and each channel id is
the torch ordinal of a kline interval. Different instruments are different
batch rows. The contract is `include/embedding/encoders/timeframe_channel.h`.
A historical checkpoint that stored channel ids `{0,1,2}` keeps that withdrawn
instrument binding. Do not relabel its scores as timeframe measurements, and
do not accept that binding in a new family settings file.

## Keep four objects separate

- **Family/provider definition:** architecture or upstream service, public
  interfaces, actual constraints, supported operations and dependencies.
- **Configuration:** supported input geometry, latent layout/width, model
  capacity and execution choices; for external providers, upstream-supported
  model/request/modality/output settings.
- **Recipe:** initialization, preprocessing fit, objective and optimizer,
  sampling/masking and progress rules; or the external request/task policy.
- **Instance/result:** trained weights and fitted state, or a particular
  provider revision/request/response, with exact identities and provenance.

Checkpoints, benchmark configurations and scoped rankings are evidence for
particular instances. Preserve their results and limitations without turning
their dimensions, seeds, budgets or parameter totals into universal family
requirements. A reference checkpoint remains a comparison reference.

## Configuration and portable workflows

For locally trainable families, derive dimensions and parameter counts from
configuration. Validate genuine architectural constraints, supported ranges,
numerical/device requirements and memory limitations. Supported geometry,
latent-width or capacity changes should use configuration; they should not
require a new source version just to bypass a frozen benchmark assertion.
Existing weights still require their saved configuration and fitted state.

The reusable surface should cover supported fresh construction, TRAIN-only
preprocessing fitting, forward/encoding, objective/backward/optimizer steps,
checkpoint/resume and inference export/loading. State which pieces are
implemented, verified or missing. Bind configuration, recipe, source/split,
initialization, fitted statistics and progress to the resulting artifacts.
Document stochastic-state requirements for continuation. Do not claim a
complete training pipeline from an inference loader alone.

For external/open providers, expose the capabilities actually available:
loading or service calls, supported configuration, validation, output records
and provenance. Respect upstream limits and access/license conditions. Mark
unavailable local weights, training or dimension controls explicitly; do not
invent them to imitate a locally trainable family.

Expose agreed building blocks through documented reusable APIs and CLI where
applicable, with input/output schemas, discovery/examples, failure/abstention
behavior and explicit compatibility. Keep operation and application logic with
their owners. Deliver a portable source/dependency/artifact handoff rather than
an application-specific copy of an entire research environment.

For locally trainable encoders, the training method is part of the component
delivery. Carry the complete numerical dependency closure, runnable training
example, preprocessing/mask/target rules, objective and gradient treatment,
optimizer and sampling recipe, stochastic state and resume semantics,
configuration, tests, dependencies/licenses, research rationale and known
failures. Keep historical, portable and application recipes distinct. Provide
an explicit inventory and byte-bound origin/destination mapping; an inference
checkpoint or network header alone is an incomplete trainable-component handoff.
External providers document the training capabilities actually available and
identify unavailable parts.

## Evidence and preservation

Before claiming configurable construction/training, retain focused engineering
evidence for meaningful supported alternative shapes or capacities: construction,
forward, real updates, export and the declared resume behavior. External-provider
checks instead cover supported requests/outputs, adapter compatibility and
failure handling. Scale checks to the changed surface; these instructions do
not require replaying completed experiments or launching expensive studies.

Distinguish implemented interfaces, engineering portability, measured research
quality and application utility. Keep missing coverage and application docking
work explicit. One passing configuration or strong checkpoint score does not
establish every configuration or downstream use.

Preserve frozen versions, original protocols, checkpoints, source hashes and
historical receipts. Adapt maintained interfaces under a new source identity;
do not rewrite evidence to make an old benchmark look configurable. Retain
failures and earlier references. Documentation and proposed milestones do not
authorize experiments, live provider calls, application training or publication.
