# Embedding evaluation policy

Policy version: 1.2

Last updated: 2026-10-06

Scope: every encoder evaluated in `cuwacunu_embedding`, including the existing
MTF-JEPA-MAE-VICReg encoder, RPB-MAE, and future local or external encoders.

This is the repository's reference for evaluation ownership, experimental
validity, reporting, and acceptance. Encoder documents define architecture,
training, and applicable behavioral contracts. An evaluation card defines the
particular task, comparison, and acceptance thresholds. A run report records
what actually happened. Keeping these separate allows a new encoder to enter
the same evaluation without redefining the benchmark around its training loss.

The requirements below apply to new acceptance claims. Historical reports retain
their original protocol and limitations; they are not retroactively relabeled as
passing this policy. A policy requirement, source file, or Make target is not
evidence that an implementation has passed it.

Version 1.2 adopts the [results reporting standard](RESULTS_REPORTING_STANDARD.md)
and [stable model tags](EMBEDDING_VERSIONS.md). New primary encoder comparisons
use native exports without PCA afterward, with fixed classifier recipes and
standalone raw/PCA controls. It removes the prior requirement to make a
compressed encoder tier primary. Historical version-1.1 cards/drivers retain
their declared protocols and measurements. The new
[shared archive readout](../code/evaluation/cards/archive_readout_v1.md) implements
the validation-only native/raw/PCA comparison. The separate native-only
training/selection runner is implemented in the
[shared native curve](../code/shared/src/native_curve.cpp), with measured
development evidence in the
[RPB-v4 record](../code/encoders/raw_patch_bottleneck_mae/NATIVE_CURVE_ADVANCE.md).

## 1. Evidence layers and claim scope

Every report separates the following evidence. Scores from different layers are
never averaged into a single encoder score.

| Layer | Question | Evidence and allowed claim |
| --- | --- | --- |
| Correctness | Does the implementation satisfy its declared contract? | Applicable gates pass, fail, or remain unverified. Passing establishes behavior within the tested scope. |
| Optimization | Did the declared training procedure run and use the intended paths? | Loss components, gradients, update/skip counts, teacher or optimizer diagnostics. Falling loss establishes optimization progress. |
| Representation | What declared distinctions can a frozen export support? | Held-out probes, controls, interventions, coverage, and uncertainty under a named protocol. |
| Acceptance | Is this configuration suitable for the declared use? | A frozen primary comparison, required-task thresholds, correctness gates, coverage, and resource limits. |

A claim identifies the encoder/configuration/checkpoint, exported surface,
task and population, probe family, dimension tier, and evidence stage. Synthetic
classification does not establish forecasting, robotics, market performance,
or general information preservation. A finite embedding or healthy projector
does not establish a useful served representation.

Evidence stages are `correctness`, `development`, and `confirmation`.
Confirmation outcomes are `accepted`, `failed`, `inconclusive`, or `invalid`.
Missing required evidence prevents acceptance. A broad information-preservation
claim must name the distinctions tested and the permitted invariances; no finite
probe suite certifies lossless retention of arbitrary inputs.

## 2. Ownership and separation of concerns

| Owner | Responsibilities | Boundary |
| --- | --- | --- |
| This policy | Evidence layers, shared validity rules, reporting, and acceptance process | Does not choose an encoder architecture, loss, dataset, or consumer-specific task. |
| Task/protocol card | Population, source groups, legal support, labels, controls, probes, budgets, thresholds, and uncertainty | Is frozen before confirmation; does not change to favor an encoder after seeing test scores. |
| Shared harness, primarily `code/shared/` | Protocol validation, generic fitting/probing, dimension controls, scoring, coverage, diagnostics, and report contracts | Consumes typed features/support/provenance; core probe code does not instantiate an encoder or interpret its training loss. |
| Encoder adapter, under `code/encoders/<encoder>/` or an explicit integration layer | Load exact preprocessing/checkpoints, map inputs, freeze inference, expose named feature surfaces and support, save adapter assets | May depend on shared APIs and its encoder; cannot silently redefine labels, splits, probe fitting, or missing-row policy. |
| Encoder training and model tests | Architecture, optimization, encoder-specific masks, gradients, checkpoint state, and applicable behavioral gates | Training objectives and test fixtures do not choose the representation acceptance metric. |
| Runner/build integration, `code/scripts/` and `Makefile` | Explicit encoder/protocol dispatch, managed-container execution, unique output paths, and manifests | A build session isolates artifacts; it does not by itself select an encoder or certify a result. |
| Report and decision | Measurements, provenance, limitations, comparison, and scoped outcome | An acceptance report does not silently replace defaults or rewrite prior evidence. |

Generic harness logic belongs behind encoder-independent interfaces. The shared
provider/card/engine contract is in `code/shared/feature_evaluation` headers,
source and tests, and explicit integration lives under `code/evaluation/`.
Adapters belong to their encoders. Fitting callbacks receive permitted training
observations and declared metadata without labels, clean hidden references or
held-out data. Surface kinds, support/order and compared-method sets are explicit
contracts rather than conventions inferred from feature names.

Existing baseline code retains its historical evaluator while the new common
driver is introduced. An extraction/refactor requires legacy regression evidence
and an implementation change; this policy alone does not authorize changing
model behavior or reinterpret archived evidence. Current implementation scope
and remaining gaps are recorded in the
[separation review](EVALUATION_IMPLEMENTATION_REVIEW.md).

## 3. Evaluation card before a comparison

Each development or confirmation run references a versioned card. The card may
be a saved section/file or an immutable manifest, but must contain:

- Intended use and claim; task definitions, label availability, metrics, and
  whether a task is required, diagnostic, or outside the encoder's contract.
- Input schema, units, channel identities, time/endpoint semantics, legal
  observation support, precision/preparation, and permitted invariances.
- Dataset/generator version, source-group and split manifests, eligible
  population, corruption definitions, and independent sampling unit.
- Named global/local/contextual/channel readouts, validity rules, native
  dimensions, and any declared matched-size controls.
- Legal controls, their input/pretraining/metadata access, and how the comparator
  is selected without using confirmation scores.
- Encoder, checkpoint, preprocessing and compression assets; initialization,
  training, sampling, masking, split, corruption, and probe seed namespaces.
- Training and label budgets, probe family and hyperparameter/selection budget,
  checkpoint selection, and handling of failed or skipped runs.
- One primary comparison; required-task/noninferiority thresholds; coverage and
  cost limits; uncertainty method, confidence level, multiplicity and stopping
  rules; and conditions that make a result inconclusive or invalid.

Encoder-specific tasks need a reason tied to the claim. Absolute level retention
is required when level is a declared signal of interest, and may be inapplicable
to an explicitly level-invariant returns encoder. Task scope is declared before
scores are inspected. A narrower contract cannot support a broader claim.

## 4. Applicable correctness gates

All encoders require tests for declared shapes/types, valid support, observed
zero versus missing data, finite valid outputs, public configuration validation,
padding behavior, and compatible archive/checkpoint loading. Missing storage
values cannot influence outputs when the input contract says they are absent.
Numerical tests cover relevant constants, offsets, small variations, scale
floors, and nonfinite observed versus masked values. Precision-sensitive
preparation must occur before a cast that loses the distinctions being tested.

Other gates apply only when the architecture or consumer contract requires them:

- Endpoint causality: changing observations unavailable at the endpoint cannot
  change that endpoint's export. Per-timestep causal claims need their own tests;
  bidirectional processing of already available history is a different contract.
- Channel semantics: a declared local export cannot depend on another channel.
  Reordering complete channel records with IDs, masks, units, and timestamps
  follows the declared equivariance; a declared set-pooled global output is
  invariant. Fixed-order encoders must disclose and validate their order contract
  rather than inherit an inapplicable set-invariance gate.
- Mask isolation: forbidden targets cannot enter encoder-visible tensors through
  normalization, transforms, cached statistics, or overlapping support. With
  fitted assets fixed, perturb hidden finite targets and test invariance; test
  forbidden-cell gradients where the path is differentiable.
- Teacher/EMA contracts: freeze and update rules apply to encoders that use them.
- Bottleneck contracts: reconstruction gradients reach the trained export and
  the decoder has no forbidden token-state bypass.
- Workflow contracts: round-trip inference, resume/RNG/update-counter behavior,
  interrupted periodic recovery, and device portability within promised
  tolerances. Entirely ineligible batches must follow the declared skip policy.

Applicable structural failures invalidate the corresponding claim. Statistical
decoder/probe performance is evaluated with thresholds and uncertainty, rather
than called a deterministic unit-test gate.

## 5. Splits, fitting, and selection

Assign source trajectories, sessions, generator draws, and paired transformations
to splits before constructing windows. Keep a trajectory's transformed pair and
all derived windows together. Grouped and temporal evaluation address dependence
that ordinary random row splits overlook; the choice depends on the deployment
question. See [grouped and temporal cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

For chronological confirmation, specify context and label support intervals and
purge boundary overlap accordingly. Main support-disjoint results and operational
walk-forward reuse are different named protocols. Future-prediction labels occur
strictly after the declared endpoint; choose horizons in advance and audit actual
availability timestamps, preprocessing support, and split-boundary labels.

Fit encoder pretraining, scaling, imputers, feature normalization, PCA, and probe
parameters only on their permitted training partitions. Window-local transforms
may use legal context under an explicit contract. Validation selects among
declared alternatives; confirmation data never refit assets. The same rule
applies to transformations such as PCA, as explained in [data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Unlabeled confirmation observations remain confirmation data. Any transductive
access is a separately declared track with matching controls; it cannot support
an unchanged inductive claim.

Use a fixed training-budget checkpoint or a preregistered validation selection
rule. Training-loss-based stopping/selection is permitted as a disclosed training
recipe or optimization diagnostic; it cannot establish representation utility
and cannot be compared with an unlimited downstream-tuned search. Report every
method's selection budget. Confirmation acceptance still requires the declared
representation comparison.

Fresh confirmation trajectories/seeds remain separate from those used to choose
widths, masking, losses, probes, or tasks. Once their scores guide a change, they
become development evidence. Do not repeatedly inspect a fixed confirmation set
and retain its label as fresh evidence.

## 6. Frozen surfaces, support, and dimensions

Freeze encoder weights and declared preprocessing during feature extraction.
Use the declared inference/evaluation mode and disable training updates; state
whether the served features use online, teacher/EMA, or another checkpoint state.
Numerical reproducibility tolerances and any stochastic inference are explicit.
Report at least the applicable surfaces below with explicit semantics:

| Surface | Meaning | Primary support rule |
| --- | --- | --- |
| Global | One vector summarizing the declared input/channel set | Valid declared global support; usually at least one observed channel, unless the contract is stricter. |
| Local channel | One channel's vector using only that channel and permitted metadata | That semantic channel has adequate support. |
| Contextual channel | Channel-indexed vector that may use other channels | Declare observed versus inferred support and the inference policy. |
| Concatenated channels | One joint probe on a defined ordering of channel vectors | Every required constituent is valid in the complete-channel tier. |
| True per-channel probes | Separately fitted/scored probes for eligible semantic channels | Task is meaningful per channel, with declared minimum train/test/class support. |

Use the name `channel_concatenation` for the joint surface. Do not call its score
per-channel accuracy. For true per-channel probes, report each channel's score,
support and omissions, plus the declared macro or weighted aggregate. Neither
that aggregate nor concatenation can certify cross-channel interaction by itself.

Use the native served encoder export as the primary tier. Do not apply PCA or
random projection after an encoder in new primary comparisons. Count appended
masks, IDs, normalization statistics, and other metadata in the feature budget.
Give controls the same legal metadata or identify a separate privileged track.
Keep head architectures, training budgets and declared seed policies fixed;
fit each head's weights separately using its training representation. Disclose
input/head parameter counts when native dimensions differ.

PCA remains a standalone raw-data comparator, labelled "PCA only — no encoder",
with its width matched to the active native export for the compact comparison.
Fit scaling/compression on valid training rows and persist its assets. Raw data
with all declared coordinates is a larger-input reference. Training a narrower
encoder is a separate architecture/configuration experiment. Additional
compression or random-projection diagnoses are separately declared development
tracks; historical cards retain their compressed primary tiers unchanged.

For centered PCA require requested width no greater than input width, valid
training rows minus one, and numerical rank. Mark unsupported compression rather
than padding zero-rank components or silently changing widths. The current
baseline widths 12/global and 36/concatenated, and RPB-MAE's native 32/global
and historical 96/concatenated, are configuration examples, not universal policy constants.

## 7. Task validity, controls, and probes

For each task report a legal raw-observation reference, descriptors where
applicable, a matched untrained encoder, the candidate frozen export, and the
strongest reproducible relevant control. Include mask/metadata-only and
shuffled-training-label controls for shortcut detection. Label-shuffling uses a
declared source-group/label-permutation scheme; compare it statistically with
the chance or prevalence reference rather than demanding an exact chance score.
If a control is unavailable, disclose why and limit the claim; a minimum harness
stage can run before a new encoder exists.

Constructed synthetic contrasts require an analytic or independently verified
raw-input solvability reference. An oracle receives only legal observations and
metadata, and reports coverage. It cannot see a hidden clean signal, generator
state, unavailable future, or labels. Noise/missingness may limit recoverability;
do not require perfect oracle accuracy regardless of the task. Real-world tasks
usually need label audits and strong legal positive controls instead of an
unavailable analytic oracle.

Descriptors are empirical controls, not an information ceiling. Compare native
and capacity-matched versions. A compact export need not beat a much larger
descriptor vector to justify compression, cost, transfer, or robustness under
the card's stated claim. If a same-budget legal control dominates the candidate
on the primary comparison, that candidate cannot pass an improvement claim.

The initial controlled task family covers reversal/order, declared level,
amplitude, and signed cross-channel lag. Reversal pairs can match global
mean/variance/Fourier magnitudes without colliding under the complete overlapping
tokenizer; compare actual permitted tokens and exported features. Independent
channel vectors may retain phase/timing for a joint readout, so mixer-off lag
failure is neither automatically expected nor automatically excused. Compare
local/global/joint surfaces and mixer variants according to their contracts.

Use frozen linear/ridge probes as the default accessibility tier. Fix their label
and selection budgets. A small fixed nonlinear probe is a separate interaction
tier with its own training/validation budget. It can establish a nonlinear-access
claim when preregistered, but cannot retroactively rescue a failed primary linear
claim. Failure of a raw linear probe while an oracle/nonlinear raw control solves
the task is evidence about the readout's capacity, not proof of encoder loss.

Frequency, multiscale structure, event timing, and future prediction are separate
task families added for relevant claims. Do not average future prediction into
morphology by default. Any aggregate requires preregistered weights/scales and
still reports each required task's result.

## 8. Robustness, coverage, and abstention

Preserve the existing fixed-readout stress sweep: random dropout at 10/30/60/90%,
contiguous outages at 25/50/75%, one whole channel absent, and all channels absent.
Reuse the training-fitted probe and preprocessing; this measures survival of an
already fitted readout. A refitted-readout track uses corrupted training data
only and is reported separately with its additional training budget.

Use identical paired corruptions across methods. Keep class-independent
missingness as the controlled reference. Add regime-dependent or realistic
missingness shifts when the claim needs them, with the mask/metadata baseline
and explicit direction of the shift. Missingness can be legitimate operational
information; distinguish that contribution from signal retention rather than
calling every mask-based prediction leakage.

Define the evaluation population and external eligibility before method-specific
validity is known. Exclude invalid placeholders from fitted assets and conditional
statistics. For every method/task/corruption report:

- Total eligible examples/source groups, valid predictions, invalid/abstained
  outcomes, exceptions, and counts by class/channel/signal family.
- Coverage and conditional task score with its denominator.
- A paired common-valid comparison and the size/composition of that intersection.
- The card's full-population utility or failure rule; common-valid accuracy alone
  cannot establish robustness or acceptance.

Define the common-valid intersection for each declared comparison and freeze
its compared-method set. Adding another provider must not silently change an
earlier comparison's population; report changed denominators and subgroup mix.

For classification, with total N, valid N_valid, and correct N_correct, coverage
is N_valid/N and conditional accuracy is N_correct/N_valid. Conditional accuracy
is null/unsupported when N_valid is zero. If the card counts abstention as failure,
full-population correctness is N_correct/N; name that measure explicitly. An
all-missing case in the signal-supported primary tier keeps zero coverage and
null conditional accuracy, without fabricating observed support from a zero
placeholder. A separately named metadata/inference track may declare meaningful
inferred support and is scored under its own card; it cannot be reported as
observed-signal coverage. Other tasks declare their own abstention utility before
confirmation.

Training skip counts describe a separate cohort effect. Report attempts,
completed updates, retained/reduced/infeasible targets and skip rates by family.
Strict masking that disproportionately excludes difficult families changes
exposure; compare cohorts and effective budgets as well as scores.

## 9. Diagnostics, interventions, and ablations

Measure served exports before training-only projectors: finite rate, validity,
coordinate standard deviations, norms, numerical/effective rank, and relevant
correlations/similarity. Report projector measurements separately. Retain
per-semantic-channel statistics across independent trajectories and distinguish
within-channel from between-channel variation. Unsupported fewer-than-two-row
statistics are flagged. A centered covariance from n valid rows of dimension D
has algebraic rank at most min(D, n-1). Report independent source-group counts
separately: paired variants and overlapping windows do not provide independent
observations. This diagnostic bound is separate from training-fit PCA support.
Diagnostics identify problems and guide hypotheses, but do not accept an encoder.

Mask audits document exact legal raw support across scales/transforms and target
retention. Conservative same-channel interval overlap is a scoped measurement;
cross-channel statistical dependence is measured separately.

For a claimed useful bottleneck decoder, compare held-out reconstruction with a
matched metadata-only decoder. Shuffle exports across support-compatible
examples and separately zero them, preserving target/mask semantics and
reporting reconstruction degradation with uncertainty. No token bypass and
gradient reachability are structural gates; measured reconstruction advantage
and shuffle sensitivity are statistical evidence. Out-of-distribution zero
vectors alone do not demonstrate reliance on input information.

Ablate one declared factor when possible: objectives, tokenizer, mixer,
preprocessing, export width, or predictive branch. Keep data, selection budgets,
readout capacity, and corruption manifests matched; disclose composite changes.
A token-decoder success with a failed bottleneck motivates capacity/readout/loss
diagnosis, not a predetermined mandatory width increase or tokenizer verdict.
Negative results do not prescribe either removal or another loss automatically;
record the hypothesis and justify the next bounded experiment separately.

## 10. Replication and uncertainty

One deterministic correctness run can validate harness wiring. Three paired
development seeds are the default exploratory budget; the historical 64-test-
trajectory/three-seed benchmark remains development/regression evidence. New
confirmation comparisons default to at least five paired runs unless a card
justifies a different design. These are project budgets, not guarantees of power
or adequate uncertainty.

Pair methods on identical source splits and corruption draws. Separate training
randomness from data/probe randomness and report which vary. Do not require
matching parameter fingerprints across different architectures. Preserve each
method's source/seed manifests and include failed runs in the declared outcome
accounting rather than keeping only successful seeds.

Report paired effect sizes and intervals, their confidence level, assumptions,
and the variability included. Resample independent trajectories with all paired
variants/windows together; use time blocks and an explicit dependence design
for temporal data. If generalizing across training randomness, include between-run
variation through a declared hierarchical/paired analysis rather than treating
all seed-by-window rows as independent. Merely bootstrapping overlapping windows
or reporting seed standard deviation is not a confidence interval. These reporting
requirements align with the [NeurIPS experimental uncertainty checklist](https://neurips.cc/public/guides/PaperChecklist).

Freeze margins, comparator selection, required-task multiplicity treatment,
sample-size rationale, and any sequential stopping rule before confirmation.
A clear violation of a required criterion under the declared decision rule is
`failed`; insufficient precision across its decision margin is `inconclusive`.
Protocol violations make the affected claim `invalid`. A missing required
measurement remains unverified and prevents acceptance. Five seeds do not
override any of these outcomes.

## 11. Acceptance and practical cost

Acceptance is scoped to a configuration and evaluation card. It requires:

1. Every applicable correctness/protocol gate passes with referenced evidence.
2. The primary probe meets the preregistered effect and uncertainty rule against
   the frozen strongest relevant reproducible control, or a declared validated
   control-selection procedure.
3. Every required task meets its declared threshold/noninferiority rule; an
   aggregate win cannot hide a required-task failure.
4. Coverage and full-population utility meet their declared floors, including
   robustness cohorts required for the use.
5. Cost is within budget: report parameter count, training data/updates/compute,
   extraction latency/throughput, and memory under stated hardware, precision,
   batch size, warmup, timing/synchronization, and missingness conditions.
6. Configuration/selection choices are frozen, confirmation is uncontaminated,
   and artifacts identify a reproducible result.

An efficiency/compression claim may preregister quality noninferiority plus a
cost improvement instead of quality superiority. State that objective in the
card; it is not an exception invented after a failed accuracy comparison.
Training loss, projector geometry, and a four-regime diagnostic win cannot
substitute for these requirements.

The decision names the selected configuration, comparator, effect/interval,
required-task outcomes, coverage, costs, remaining limitations, and artifact IDs.
Acceptance concerns this declared use; a different domain or readout needs its
own evidence. Adaptive head routing or online model selection is a separate
consumer/system evaluation, even when it uses an accepted frozen encoder.

## 12. Domain and external-pretraining tracks

Domain evaluation has its own card and does not inherit a synthetic score as a
domain result. For a market track declare the instrument panel, sample grid,
session calendar, units, endpoint and publication/availability times, labels,
horizon, chronology, and permitted metadata. Distinguish scheduled closure/halts
from unexpected missing observations without inventing a third mask state in an
existing binary schema. The adapter/card must explain how those states are
represented.

Choose returns, ranges, prices, or other signals according to the intended use;
raw level is not universally forbidden or required. Use legal context for
context-derived scales and disclose frozen train-fit assets. Compare relevant
causal controls such as lagged inputs, range and volatility under the same
support/metadata contract. Report missingness, skip rates and coverage by
predeclared liquidity/session/regime groups. Forecasting, routing, and trading
claims require their own task/system protocols and costs.

External checkpoints, including possible MOMENT/TS2Vec/forecasting-derived
controls, are a separate pretraining-data track. Pin model/code versions,
checkpoint hashes, normalization, stream mapping, padding, extraction/readout,
actual support, and known/unknown training overlap. Unknown overlap prevents a
clean inductive claim, even if the external score is useful context. Appended
normalization statistics create a separately named/dimensioned feature surface.

## 13. Artifacts, compatibility, and maintenance

Every run persists a card/protocol/policy version, source revision plus dirty
source hashes, environment/toolchain/device, configuration and all seed streams,
data/split/support/corruption manifests, checkpoint and preprocessing identities,
compression/probe assets or reproducible fit instructions, gate evidence,
per-task/per-run measurements, coverage/failures/skips, costs, and decision.
Use unique run directories under `output/`; never overwrite an earlier evidence
bundle. Content hashes establish immutable asset identity; label noncryptographic
fingerprints as reproducibility checks rather than security/association proofs.

Keep the baseline checkpoint's exact original preprocessing. A saved scaler
beside a checkpoint does not by itself prove their association or training-data
independence. If that provenance cannot be established, identify a newly trained
control under the permitted protocol; do not apply a new per-channel scaler and
present the result as unchanged historical weights.

The four-regime benchmark is a useful synthetic quality diagnostic and regression
alarm, distinct from the tiny optimization smoke fixture. Regression expectations
refer to its own documented behavior/tolerances. Losing to native high-dimensional
descriptors is not an automatic stop rule; acceptance uses the card's fair
comparison. Historical AULC/FSPA-4 numbers require matching artifact provenance,
metric axes, integration/aggregation rules, label budgets, and checkpoint/data
identity. A new label-budget learning curve gets its own metric name unless those
definitions match.

Maintain this file when a new encoder/surface, task, domain, gate, metric, protocol,
support rule, or acceptance decision exposes a missing shared contract. Change
its version/date and log the rationale. Record policy deviations explicitly in
development reports; an unresolved mandatory confirmation deviation blocks an
acceptance claim. Changes to labels, fitting, denominators, dimensions, corruption,
selection or metrics require a new protocol version and fresh confirmation where
the interpretation changes. Never edit archived reports to conform retroactively.

### Repository references and evidence map

| Reference | Role and evidence boundary |
| --- | --- |
| [Shared feature harness APIs](../code/shared/include/embedding/shared/feature_harness.h), [implementation](../code/shared/src/feature_harness.cpp), and [tests](../code/shared/tests/feature_harness_test.cpp) | Generic valid-row fitting, PCA, linear/nonlinear probes, controlled pairs, and raw references. Source/tests alone do not certify a complete comparative run. |
| [Shared provider/card contract](../code/shared/include/embedding/shared/feature_evaluation.h), [engine](../code/shared/src/feature_evaluation.cpp), and [integration guide](../code/evaluation/README.md) | Encoder-independent development orchestration with explicit adapters, typed surfaces, saved cards and declared paired populations; confirmation/acceptance remains outside its implemented scope. |
| [Shared representation diagnostics](../code/shared/src/evaluation.cpp) | Existing synthetic generation and served-vector diagnostics. |
| [Legacy evaluator](../code/encoders/mtf_jepa_mae_vicreg/src/evaluation.cpp) and [retained benchmark summary](../output/evaluation/goal-01a107a5/summary.json) | Established four-regime protocol, native controls, corruptions, mask audit and historical development evidence. Local generated evidence may be absent in another checkout. |
| [RPB-MAE specification](../code/encoders/raw_patch_bottleneck_mae/RPB_MAE_architecture_and_evaluation_spec.md) and [implementation guidelines](../code/encoders/raw_patch_bottleneck_mae/RPB_MAE_implementation_guidelines.md) | Encoder-specific architecture, support, training and staged implementation. Shared evaluation/claim requirements are governed by this policy. |
| [Environment instructions](ENVIRONMENT.md) | Authoritative managed-container execution and artifact/session coordination. Validation status comes from dated run evidence, not target existence or stale prose. |

### Decisions incorporated from the review notes

| Suggested rule | Policy decision and reason |
| --- | --- |
| Separate correctness, probes, and decisions | Adopt; optimization is also explicitly separated so its measurements remain useful without becoming acceptance scores. |
| Descriptor score is a ceiling; stop whenever the encoder loses | Replace with native/matched empirical controls and claim-specific thresholds. Dimensional budget, support and intended efficiency matter. |
| Only linear probes can make a representation claim | Use a default linear tier and a separately preregistered nonlinear tier. A changed readout cannot rescue the original failed claim. |
| Never select a checkpoint using training loss | Permit a declared training recipe/diagnostic, with equal disclosed selection budgets; it supplies no acceptance evidence. |
| Every task needs a perfect analytic oracle | Require a legal solvability reference for synthetic contrasts; real/noisy tasks need appropriate audited controls and recoverability limits. |
| Mixer-off lag failure is expected | Evaluate the declared surface/readout; local timing information can support joint lag inference. |
| Bottleneck reconstruction interventions are all deterministic gates | Separate structural no-bypass/gradient gates from held-out statistical reliance evidence. |
| Five seeds are enough for acceptance | Keep five paired runs as the default minimum budget; uncertainty and power still determine the outcome. |
| All encoders must pass the same permutation, level and teacher gates | Make gates/tasks conditional on declared semantics; unexplained exclusions cannot support broader claims. |
| Market inputs must exclude price level; synthetic success is a universal prerequisite | Require a separate domain card with causal availability and suitable controls; input/task choices follow the intended claim. |
| Negative results mandate removal or a wider bottleneck | Record evidence and choose a justified next experiment; evaluation policy does not dictate architecture changes. |

### Change log

- 2026-10-05, version 1.1: documented the extracted shared engine, label-free
  training interface, typed surfaces and independent integration boundary;
  added the implementation/gap review. The development driver uses protocol v2;
  acceptance rules and archived v1 measurements are unchanged.
- 2026-10-05, version 1.0: established the shared policy; reviewed the supplied
  notes against current code and encoder plans; defined ownership, conditional
  gates, fair probes, coverage, confirmation, reporting, and maintenance.
