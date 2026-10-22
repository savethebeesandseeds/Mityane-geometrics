# Operations over representations

Follow the [project design specification](../../doc/operations_over_representations_design_spec_v0.1.0.md),
`ORR-DESIGN-001`, version `0.1.0`. Its OR-01 through OR-13 families classify
operations by their primary question and output. Specific algorithms are
versioned operation instances within those families; pipelines compose instances
with explicit input/output semantics.

The [forecasting implementation](forecasting/README.md) contains the inherited
OR-11 latent-state/direct-native forecasts, predictive training, checked
issuance and their evaluation. Its [Operation Contract](forecasting/CONTRACT.md)
and linked [Requirements-and-Evidence Record](forecasting/EVIDENCE.md) also
distinguish downstream OR-05 risk readouts. The direct-embedding "control" is a
comparison baseline; its name establishes no OR-12 action-selection capability.
These implementations and artificial engineering fixtures do not establish
forecast skill, market utility or every catalogue capability.

## Adding an operation

The [operation toolset](TOOLS.md) supplies a common command-line and reusable
library boundary for discovery, preparation, execution and verified result reuse.
Hx06's recorded paired-response comparator is its first registered tool. Each tool retains its
own payload and representation requirements; model-owned encoding stays with
the provider. Other existing interfaces remain available independently.

The [counterfactuals component](counterfactuals/README.md) owns Hx06 / OR-06
comparison work. Its entry guide links the callable paired-response tool,
contracts, synthetic evaluations, readout decision and separately authored
observational/intervention profiles. The [tool guide](counterfactuals/TOOL.md)
and its [advanced commands](counterfactuals/TOOL.md#sweep-a-scenario-space) cover recorded-response
workflows, sweeps, result differences and retained-study verification. The sole
callable claim remains `controlled_model_response`; broader profiles need their
own implementations and scientific evidence.

Begin with an Operation Contract and a linked Requirements-and-Evidence Record
from section 4 of the specification. State the question, inputs, outputs,
required representation structure, additional dependencies, claim level and
failure or abstention conditions. Use the experiment card in appendix B to
declare the hypothesis, population/splits, baselines, metrics and reproduction
references before evaluating a claim.

Reusable contracts may be drafted before a dataset or trained instance is
selected. Bind the actual inputs, artifacts, protocol and permissions before
experimental execution, including diagnostic probes. Current priorities and
acceptance criteria belong in the [Hero records](../../web/dashboard/README.md);
shared work blocks only the executions that depend on it.

Inputs may come from custom/frozen encoders, external models or API embeddings.
Identify observations and representation spaces independently; shape alone is
not a representation identity. Keep encoder/preprocessing versions, geometry,
source references and information-availability limits attached to each run.

Keep operation-specific code and protocols together, with local evidence records
or links to their canonical records. Model-owned
training and historical encoder evaluation remain in `../embeddings/`; reusable
data handling and cross-component utilities belong in `../data/` and `../shared/`.
Run implementation checks through the root managed container and the root
Makefile/task runner; see each operation's guide for its focused targets.
Historical source, cards, protocols and receipts remain unchanged in the
[pre-cleanup archive](../../.archive/embeddings/README.md). New protocols and
evidence remain owned by their operation and must distinguish current source
identities from historical receipts.

Keep records discoverable under the adopted repository layout. A proposed
shared registry is not an implemented storage requirement; adopting one must
preserve operation ownership and links to the existing evidence.

## Claims and composition

Evaluate suitability for a stated operation against meaningful controls. Keep
target readout quality, latent-state prediction and decision outcomes separate.
Label observational comparisons, controlled model responses and identified
intervention effects explicitly. A composed pipeline needs its own end-to-end
evidence, alongside the evidence for its individual operations. Preserve
negative results and untested limitations. Record execution disposition
separately from the scientific verdict: valid negative or inconclusive results
can complete their scoped evaluation, while failed or invalid attempts remain
visible without becoming negative scientific evidence.
