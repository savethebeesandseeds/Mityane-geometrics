# Encoder discovery, delivery and application ownership

Mityane is the discovery and reusable-component owner. Applications instantiate,
train, select and use those components for their own populations and tasks.
The training method travels with a locally trainable encoder: the network,
learning rules and implementation together make the reusable tool. A particular
checkpoint is a separately identified instance and research reference.

Discovering better encoders remains an ongoing Mityane responsibility. It
includes proposing, training and comparing new architecture and learning-method
versions under explicit research protocols. Generalizing an existing version
for configuration or export is an engineering activity within that program;
it does not conclude the search. Later research candidates can reuse the
portable interfaces while retaining their own source, recipe and result
identities. Application ownership of substantial application training does not
transfer this research responsibility away from Mityane.

These boundaries apply to Hx15–Hx18 within each provider's actual capabilities.
Hosted providers retain upstream limits on weights, training and configuration.
Hero JSONs hold current priorities and work states. Recipe, import and artifact
records hold the actual execution identities and evidence.

This contract incorporates the [22 October Cuwacunu consumer review](causal_patch_bottleneck_mae/family/migration/CONSUMER_REVIEW_2026-10-22.md).

## Who owns what

| Concern | Mityane component owner | Application owner, initially Cuwacunu |
| --- | --- | --- |
| Family and interfaces | Architecture/provider, numerical input/output contract, supported configuration and genuine constraints | Actual features, units, semantic IDs, availability and domain adapters |
| Learning semantics | Fitting arithmetic, masks/targets/support, losses/reductions, gradient routing, optimizer primitives, stochastic and progress rules | Fully bound application recipe using supported choices; explicit identity for any adaptation |
| Recipes | Versioned mechanism definitions, documented defaults/reference recipes, rationale, comparisons and known failures | Real population and TRAIN/calibration membership, batch/sampling/context policy, hyperparameters, seeds, budgets and selection |
| Execution | Reusable fitting, update, save/resume, inference and loading APIs; runnable bounded examples | Substantial training, scaling, scheduling, interruption handling, durable orchestration and deployment |
| Artifacts | Raw family inference/training-state formats, numerical identities and declared compatibility | Authenticated publication, sidecars, ModelRef, selected-artifact records, capture and exact downstream parents |
| Evidence | Mechanism and portability checks; scoped research findings | Imported-component acceptance, data admission, operational capacity and end-to-end application usefulness |

Mityane can train research and engineering instances to discover or validate
methods. Applications can experiment with variants. Substantial training in an
application does not make its data or operating policy part of the generic
family; useful general improvements can be proposed back to Mityane.

## Mechanism versus application recipe

The component defines the meaning and supported choices of its training rules.
An application records a complete recipe binding those rules to its data and
execution. Choosing a supported learning rate, geometry, mask ratio or capacity
creates a new configuration/recipe/instance as appropriate. It need not create
a new architecture implementation.

Changing fitting arithmetic, masking/target eligibility, loss reduction,
gradient flow, optimizer algorithm or stochastic behavior creates an explicitly
identified mechanism/recipe variant. Record its source, changed semantics and
appropriate evidence. An application may maintain a local variant and later
contribute it upstream; this does not require a long intersession approval
chain. Inherited equivalence and quality claims retain their original scope.

Some policies have reusable implementations and application-specific choices.
For example, Mityane may supply a deterministic sampler or context-deletion
primitive, while the application selects and binds its actual policy. Name
the algorithm as well as its seed: different sampling algorithms can share
the same seed and produce different training trajectories. The actual TRAIN
and calibration membership remains the application's responsibility even when
the fitter enforces every numerical and causal input rule.

## Two artifact layers

The raw family layer stores the model, configuration, fitted preprocessing and,
for resumable state, optimizer/progress/stochastic identities. Its owner declares
what loads and what supports exact continuation, including runtime constraints
and missing state. Inference and resumable-training artifacts stay distinct.

The application layer authenticates and selects those artifacts, records data
and exposure declarations, publishes captures and enforces downstream lineage.
It can wrap the raw state without changing the encoder's numerical meaning.
Its authenticated declarations still require independent data admission.
Equal width or successful raw loading cannot establish compatible learned
coordinates. A changed trained encoder requires regenerated representations
and compatible fitted descendants under their respective owners.

## Small, explicit delivery and acceptance

A source release travels as one independently verified inventory containing:

1. Complete numerical dependency closure, source identities, dependencies and
   licenses, public API/configuration and raw artifact contracts.
2. Training method and reference recipe, including preprocessing, targets,
   masking, objective/gradients, optimizer, stochastic state and resume behavior;
   a runnable example plus known limitations and research failures.
3. Change and compatibility notes separating API, configuration, raw artifact,
   numerical behavior and exact-resume compatibility. Identify any required
   refit, retraining or consumer changes.
4. Evidence for the advertised behavior under named configurations/runtime:
   construction/causality, fresh fitting and updates, export/load, declared
   resume behavior and relevant incompatible-input refusals. Keep tested and
   merely supported configurations distinguishable.

The consumer verifies the delivery pin and its origin/destination mapping,
integrates through its own managed environment/build, and checks the imported
mechanism and artifact adapter with bounded fixtures. It separately owns actual
data/split/calibration admission, source availability, training recipe/budget,
resource/recovery behavior, selected-artifact/capture bindings, descendant
compatibility and application utility. Acceptance of source does not select
a market checkpoint or admit a scientific campaign.

Scale verification to the change. Documentation-only releases can retain
unchanged numerical evidence with exact source bindings; they need no repeated
training merely to refresh prose. Changed learning behavior needs its own
appropriate checks. Preserve old receipts and imports rather than relabeling
them as evidence for new paths or algorithms.

## Current CPB example

The [CPB migration package](causal_patch_bottleneck_mae/family/migration/README.md)
supplies complete portable training source, method notes and historical context.
Its maintained fitter and generic sampling/context policies differ from the
historical V16 study. Cuwacunu already owns its separate application training
and checked-publication pipeline at `code/encoder_training/versions/v1/`.
These are distinct recipes and artifact layers over shared family primitives.
The generic example permits increased total budgets on resume; Cuwacunu's
application recipe binds its original declared budget. Exact continuation is
evaluated against each owner's explicit recipe, rather than presumed from a
shared raw training-state codec.

An import is a reviewed local source dependency with recorded provenance.
Production/fresh application training uses the delivered local component rather
than a runtime dependency on a sibling Mityane checkout. Historical checkpoints
and study outcomes retain their original configurations, recipes and verdicts.
