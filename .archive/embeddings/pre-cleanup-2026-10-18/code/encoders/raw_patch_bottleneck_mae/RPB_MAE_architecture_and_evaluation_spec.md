# Raw Patch Bottleneck MAE (RPB-MAE): compact, information-preserving time-series representations

Date: 2026-10-04
Original status (4 October): research-informed proposal, before implementation.
Preserved proposal: the RPB family has since been implemented and evaluated in
separate stages. Use [current status](../../../doc/CURRENT_STATUS.md), the
[model guide](README.md) and [RPB-v20 API](CONFIGURABLE_SPECTRAL.md) for supported
code and measured evidence. The proposal below is not the v20 API specification;
unimplemented extensions retain their prospective status.

Shared evaluation and claim requirements are governed by the
[embedding evaluation policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md).
Where this proposal overlaps with that policy, use the policy for experimental
validity and acceptance and this document for encoder-specific architecture.

## 1. Decision

Build a compact raw-patch representation learner with explicit temporal and channel semantics. Establish the held-out evaluation harness before changing the learner. Train through the compact exported representation, not exclusively through internal token states or a training-only projector.

The first candidate is a **bottleneck masked autoencoder**, not a replacement stack containing every recent self-supervised objective. Preserve the existing implementation as a reproducible control. Add channel attention, multiscale/time-frequency features, and future prediction through independent, removable experiments. Do not reinstate the old masked-JEPA branch merely because newer JEPA papers exist.

This proposal uses the user's pasted Codex static review as the description of the current checkout. It is not a fresh repository audit. No local source, current checkpoint, or benchmark was executed for this report. Historical FSPA-4 and AULC references must be mapped to the actual checkpoint, dataset, metric definition, and commit before comparison with this checkout. Do not copy historical numerical results into a new report as newly reproduced results.

## 2. Research decisions

| Work | Relevant evidence | Decision for this project |
|---|---|---|
| PatchTST, ICLR 2023 [R1] | Patching, shared channel-independent temporal processing, and self-supervised pretraining. | Borrow ordered raw patches and weight sharing; not a claim of universal embedding superiority. |
| MOMENT, ICML 2024 [R2] | General-purpose pretrained time-series model with patch reconstruction and embedding extraction. | First external frozen-embedding baseline. |
| TimeMAE, WSDM 2026; first preprint 2023 [R3] | Visible/masked processing is separated; full method includes learned codewords and momentum-target regression. | Borrow visible-patch encoding; do not call the complete method a simpler JEPA replacement. |
| TimeMixer++, ICLR 2025 [R4] | Explicit multiscale/multiresolution processing and cross-variable interactions. | Architectural reference for a later scale/mixer ablation, not a mandatory replacement. |
| TS-JEPA, September 2025 preprint [R5] | Time-series masked latent prediction evaluated on classification and forecasting. | Evidence for a family of methods, not evidence that the old project branch is repaired. |
| CF-JEPA, June 2026 preprint [R6] | Crop-based, multi-horizon future-latent prediction; distinguishes online and EMA deployment features. | A later predictive experiment; do not copy its entire recipe or normalization indiscriminately. |
| LeJEPA, November 2025 preprint [R7] | SIGReg-based distribution regularization without the conventional EMA/stop-gradient setup. | Optional separate research recipe, not a default replacement for VICReg. |
| Forecasting feature-extractor study, 2025 [R8] | Directly evaluates frozen forecasting models as classification feature extractors. | Add one forecasting-derived embedding baseline after MOMENT. |
| TiRex-2, July 2026 preprint [R9] | Multivariate, covariate-aware, recurrent streaming forecasting. | Relevant future system comparator; forecasting results do not establish compact-embedding quality. |
| TS2Vec, AAAI 2022 [R10] | Hierarchical temporal representation learning across several task families. | Optional train-on-our-data representation baseline. |

These papers use different datasets, training corpora, extraction methods, and evaluation protocols. This document does not declare a single universal state-of-the-art model or rank results across incompatible benchmarks.

## 3. Consumer contract and information semantics

Preserve the existing global and per-channel interfaces, including the previously discussed `[B, 3, 32]` per-channel configuration. General notation is `[B, C, D_export]`; do not hard-code three channels into the encoder. The scalar time-series notation below is `[B, C, T]`. Any existing extra feature or entity axes belong in an explicit adapter, not an undocumented reshape.

Codex reports that the current per-channel outputs are contextualized: other channels affect them through global-mean mixing. A new channel-independent output is therefore **not** a semantically identical replacement, even when its shape matches.

Distinguish:

- `z_local`: summary computed from one channel's observations and its declared metadata.
- `z_contextual`: channel-indexed summary that may depend on other channels.
- `z_global`: global summary with a documented pooling policy.

Use explicit API names or a versioned mode. Keep an independent-channel diagnostic path, but do not silently redefine the existing public output. If a local diagnostic is generated with shared readout parameters rather than trained independently, report that limitation.

Every output includes or accompanies validity/support information. An absent channel must not masquerade as a valid zero-valued signal. Document whether other channels may support an inferred contextual embedding for an absent channel; mark it inferred, not observed. Define deterministic behavior for an entirely invalid input.

Permutation contract: moving complete channel records together with their IDs, masks, units, and timestamps should move channel-indexed outputs correspondingly; a set-pooled global output should be unchanged. Exchanging values while keeping semantic channel IDs fixed is a different input and need not preserve the embedding.

## 4. Proposed data flow

```text
raw history, observation mask, timestamps, channel metadata
    -> select permissible context and pretraining targets
    -> fixed training-fit scaling and finite-safe input preparation
    -> ordered non-overlapping raw patches + validity/time metadata
    -> small shared temporal encoder, independently applied per channel
    -> optional explicit residual cross-channel mixer
    -> position-aware readout and compact export bottleneck
       z_contextual [B,C,D_export], with documented global readout
    -> lightweight decoder using exported embeddings and target metadata
       (training only; no shortcut through unrestricted token states)
```

Multiscale/time-frequency information is an optional later branch computed from the same permissible context. It is not the sole gateway through which observations enter the model.

### 4.1 Numerical behavior

Use centered masked variance rather than `E[x*x] - E[x]*E[x]`. Preserve precision at ingestion: subtracting a suitable offset or using higher-precision preprocessing must occur before a float32 cast has already rounded away the variations of interest. Stable variance cannot recover information already lost to quantization.

For the first experiment, fit per-channel scaling on the pretraining training partition and freeze it. Record scale parameters, units, and channel identities. Do not normalize each training example using its hidden target or future tail. Window-local normalization can be tested later, with visible-context statistics only. Where absolute level and amplitude matter, retain them through the model's declared inputs and evaluate their recoverability.

Reject nonfinite configuration values through the public C++ API, not only the CLI. Check loss weights, numerical stabilizers, and any newly introduced numerical controls. Test constants, large offsets, tiny variations, all-missing inputs, singleton support, and nonfinite input handling.

### 4.2 Tokenization

Start with a single patch size and stride equal to patch size. A reasonable smoke-test configuration, not a literature optimum, is patch length 16, encoder width 64, three temporal blocks, four attention heads, and export width 32.

A patch contains ordered normalized samples, per-sample validity, and declared time information. Add relative patch position and semantic channel metadata. For irregular sampling, include elapsed times; sequence index alone is not elapsed time. Keep patch boundaries explicit when removing masked patches.

Begin with a linear patch projection. If a convolutional tokenizer is compared, confine its receptive field to permissible observations and record that support. Masked values must be removed before any convolution, smoothing, resampling, or frequency transform that could otherwise expose them.

Mean/std and spectral magnitudes are useful descriptors but are not lossless. Whether the complete existing multiscale tokenizer loses a particular distinction must be tested on its actual outputs, because overlapping windows and positional metadata can recover some distinctions. Test descriptor collisions rather than assuming every phase/order distinction is lost.

### 4.3 Temporal and channel processing

Temporal processing must exchange information across ordered patches, not rely solely on a single global average. Use a small pre-normalized temporal Transformer as the first candidate. A dilated temporal-convolution alternative is a later backbone control, not another mandatory branch.

The channel mixer is explicit and switchable. Compare independent channels, the legacy global-mean approach where feasible, and a small residual attention block across channel tokens. For phase/lag tasks, place mixing before final temporal pooling; adding attention only after all timing information has been pooled may be too late.

Attention across channels receives semantic identities and masks. Storage position is not a substitute for channel identity. Entirely absent channels need safe masking and declared inference behavior.

### 4.4 Readout and bottleneck

A compact exported vector is the product. Reconstruction from a large token sequence does not, by itself, prove that a subsequently pooled 32-dimensional vector is useful.

For the initial bottleneck candidate, use position-aware learned pooling followed by the export projection. The decoder receives only that compact channel-indexed representation plus target channel/time metadata. It must not cross-attend to unrestricted encoder token states. This is a project-specific hypothesis: training directly through the export may improve its utility, but the bottleneck can also be too restrictive.

Keep a conventional token-decoder control available for diagnosis. If it reconstructs well while the bottleneck candidate fails, compare export widths and task probes before declaring the entire objective or raw tokenizer unsuccessful. Keep the standard-width comparison primary; larger-width experiments are explicitly capacity diagnostics.

Do not add an untrained learned global readout. A deterministic validity-aware pool is a simple starting point, but global consumers still need their own probes. Any newly learned exported surface must receive an identified training signal.

## 5. Masking and causality contract

Maintain two distinct masks:

- O: originally observed and valid sample.
- A: intentionally hidden pretraining target.

Encoder-visible support is `O & ~A`; supervised reconstruction support is `O & A`. Naturally absent data is never converted into a target with value zero.

For each encoder input, track the raw observations that can influence it, including normalization, multiscale aggregates, FFTs, rolling features, convolutions, and cached statistics. For the strict reconstruction experiment, the union of these influences must exclude deliberately hidden observations. A target encoder or reconstruction label may legitimately see target observations; that fact alone is not leakage. The forbidden path is target information entering the predictor input or fit procedure contrary to the evaluation protocol.

Use span/patch masks sampled independently of synthetic class. Start with a moderate masking rate, such as 25%, and log actual valid target and context counts. This is a starting configuration, not a universal optimum. Sample-level or class-conditioned visibility can otherwise become a shortcut.

No silent overlap fallback in the strict mode. If sufficient legal context is unavailable, resample within a fixed budget and then skip with an explicit reason. Track skip rates by signal family and missingness to detect a changed benchmark population. Preserve legacy behavior only under a named compatibility mode.

For an endpoint embedding `z_t`, all encoder observations must have timestamps at or before t. Bidirectional attention within that already observed history is permissible. A per-timestep output claimed to be causal requires causal attention, appropriate convolution support, or recomputation from the corresponding prefix.

For a future-prediction experiment, future values are training labels only. Target windows and their labels must remain inside the training split. Known-future covariates require an explicit availability contract; future observed target values are not known-future covariates.

## 6. Objectives

### 6.1 Initial candidate: one reconstruction objective

Let `x_scaled` use the declared frozen training-fit scale, `z = E(x_visible)`, and `D(z, q)` decode a target at position/channel metadata q.

```text
L_mask = sum_{O & A} Huber(D(z, q), x_scaled[q]) / count(O & A)
```

Average across eligible examples/channels according to a fixed policy so that dense channels do not unintentionally dominate. Skip zero-target examples rather than assigning a fictitious successful zero loss. Record eligible counts.

Use a small decoder. No JEPA teacher, EMA updates, VICReg projector, codebook, or forced time-frequency alignment in this first candidate. Reconstruction reduces the need to stabilize a jointly learned target, but it does not guarantee useful embeddings or eliminate all degenerate solutions. Check the output.

### 6.2 First predictive extension: predict observations before predicting latents

Once the core passes its probes, attach a small horizon-conditioned head to the same exported z. Predict fixed, observed future targets with a properly masked regression objective. Keep the data scale consistent with the context. Choose horizons from the actual application's sampling and decision times.

```text
L = L_mask + lambda_future * L_future
```

Compare against the successful parent, not just the original failing stack. This separates the value of a predictive training signal from the stability of a learned latent target. Pure future prediction may neglect useful but unpredictable details; the multi-task probes must detect that tradeoff.

### 6.3 Deferred latent prediction and regularization

A later JEPA branch must use a new target/masking specification and an independently evaluated objective. Do not merely rename or reopen the previous branch.

Treat conventional EMA/stop-gradient JEPA and a LeJEPA-style symmetric, regularized recipe as distinct alternatives. Do not mix fragments and assume the cited analysis still applies. Test teacher and online features separately if both are deployed; choose on validation, not the final test.

VICReg or SIGReg is not a mandatory upgrade. Their distributional preferences are not evidence of downstream utility. Decorrelation of learned coordinates does not logically imply removal of raw cross-channel relationships, but compatibility with this structured output must be measured. A covariance estimate from n rows has rank at most n-1; do not label that sampling limit as model collapse. For variance/covariance diagnostics, aggregate enough meaningfully independent examples and explicitly flag singleton/low-support batches.

## 7. Evaluation protocol

### 7.1 Split before generating overlapping examples

For synthetic data, separate source trajectories, generative parameter draws, and random seeds before cropping. Randomize missingness independently of signal family for the controlled benchmark, and add separate realistic missingness-shift tests.

For real temporal data, use chronological training, validation, and final testing. Prevent future-label overlap across the cutoff. Keep a strict support-disjoint evaluation for the main claim; if operational walk-forward evaluation legitimately reuses older observed context, document it as a different protocol. Fit pretraining, scaling, PCA, and probes only on their permitted partitions.

Unlabeled test observations are still test observations for an inductive evaluation. External pretrained checkpoints belong in a separately labeled external-pretraining track, with known or unknown pretraining overlap disclosed.

### 7.2 Baselines and fair readouts

Mandatory local controls are current raw descriptors, raw-history features under a declared compression policy, an untrained encoder with matched architecture, and the frozen current trained encoder. Include FSPA-4 only after resolving its reproducible artifact and metric definition.

Add frozen MOMENT embeddings. Next, choose either TS2Vec trained on the same training data or one forecasting-model feature-extractor baseline, rather than implementing every named paper.

Use a fixed linear/ridge/logistic probe family, plus the already accepted tiny probe where relevant. Fix its optimization and validation budget. Report both native-dimensional results and dimension-matched results; fit any PCA/compression on the training split only. Give baselines access to the same declared metadata. Report parameter count, extraction latency, memory, and pretraining data budget separately.

Retain the historical definition of AULC for historical comparisons. Add label-budget learning curves as a separately named metric if the existing AULC instead measures optimization steps. Specify integration axis, score orientation, normalization, and task weights before comparing runs.

### 7.3 Task and structural probes

The initial suite should include signal morphology/order; phase-sensitive contrasts; frequency and multiscale recovery; amplitude/level recovery where required; cross-channel lag/lead relationships; future-target prediction; and robustness under sensor dropout or missingness shifts.

Use deliberately difficult controlled pairs: a waveform and its reversal with matched global mean/std and Fourier magnitudes; signals with matched marginal statistics but different channel dependence; shifted events with relevant timing labels. Verify which pairs truly collide under the complete legacy tokenizer rather than relying on the single-window argument alone.

Separate structural requirements:

- Changing samples after endpoint t must not change `z_t`.
- Reordering complete channel records with their metadata must satisfy the declared equivariance.
- A time reversal or change of meaningful channel identity should remain detectable, not automatically invariant.
- Invalid channels must be explicitly flagged and must not produce undefined operations.

Add a missingness/metadata-only baseline and shuffled-label controls to identify shortcuts. A label-independent synthetic mask is a controlled benchmark choice, not a claim that real-world missingness is never informative.

### 7.4 Diagnostics and acceptance

Measure native, pre-projector exported vectors: per-dimension standard deviation, norms, effective rank, correlations/similarity, finite rate, and valid support. Report projector statistics separately when a projector exists. Healthy projector geometry does not certify exported task utility; a deterministic projector also cannot create input-specific information from identical inputs.

Use three seeds for smoke tests and at least five for final candidate comparisons, as proposed budget settings. Use paired splits and seeds. Quantify uncertainty at the trajectory or time-block level, not by pretending millions of overlapping windows are independent.

Accept a candidate only when a preregistered primary representation metric improves over the strongest reproducible relevant control, required structural checks pass, and practical cost is acceptable. Define any noninferiority margins before inspecting results. Do not let an average score hide failure on a required task. Negative or inconclusive experiments should remove a branch, not automatically add another loss.

## 8. Implementation sequence for Codex

### Phase A: evidence and correctness

1. Record commit, configuration, data/generator version, seeds, and current extraction outputs. Preserve the legacy implementation and checkpoints.
2. Build the split-aware frozen-probe runner and baseline extractors before learner changes.
3. Harden input precision, centered masked variance, C++ config validation, and no-support behavior. Separate these changes from architectural experiments.
4. Implement support diagnostics and an explicit strict masking mode. Report baseline versus strict mode, including cohort/skip changes.

### Phase B: one compact candidate

5. Implement raw patches, a small shared temporal encoder, and the compact bottleneck decoder behind a new version/configuration. Start without extra losses and with explicit independent-channel semantics for the diagnostic candidate.
6. Run matched descriptor-versus-raw input variants on the same new backbone where feasible. Compare the exported vectors, not only reconstruction.
7. Compare independent processing with explicit residual channel mixing. Keep the public contextualized export contract versioned.
8. Add the existing multiscale/time-frequency branch only if it improves the successful parent. Compute it from allowed support; add no alignment objective by default.

### Phase C: predictive branch only after a useful core

9. Compare a small future-observation head against its parent.
10. Consider latent future prediction or geometric regularization only as new, separately justified experiments.

Deliver a results table with per-task metrics, learning curves, uncertainty, structural pass/fail flags, invalid/skip counts, costs, and exact artifact references. State clearly which configuration is selected and why. Do not change a production default merely because a new training loss decreases.

## 9. Primary sources

[R1] Nie et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023. arXiv:2211.14730.

[R2] Goswami et al. MOMENT: A Family of Open Time-series Foundation Models. ICML 2024, PMLR 235:16115–16152. Official embedding implementation: moment-timeseries-foundation-model/moment.

[R3] Cheng et al. TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders. WSDM 2026. arXiv:2303.00320v4, 27 February 2026.

[R4] Wang et al. TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis. ICLR 2025. arXiv:2410.16032.

[R5] Ennadir, Golkar, Sarra. Joint Embeddings Go Temporal. arXiv:2509.25449, 29 September 2025.

[R6] Lee and Sim. CF-JEPA: Mask-free forward prediction with asymmetric encoder utilization for time-series representation learning. arXiv:2606.07031, 5 June 2026. Preprint; manuscript says submitted to Knowledge-Based Systems.

[R7] Balestriero and LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544, 11 November 2025.

[R8] Auer et al. Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors for Time Series Classification. arXiv:2510.26777, 30 October 2025; NeurIPS 2025 time-series foundation-model workshop.

[R9] Podest et al. TiRex-2: Generalizing TiRex to Multivariate Data and Streaming. arXiv:2607.01204, 1 July 2026.

[R10] Yue et al. TS2Vec: Towards Universal Representation of Time Series. AAAI 2022. arXiv:2106.10466.

Source locations, recorded for reproducibility:

```text
R1  https://arxiv.org/abs/2211.14730
R2  https://proceedings.mlr.press/v235/goswami24a.html
    https://github.com/moment-timeseries-foundation-model/moment
R3  https://arxiv.org/html/2303.00320v4
R4  https://openreview.net/forum?id=1CLzLXSFNn
    https://arxiv.org/abs/2410.16032
R5  https://arxiv.org/abs/2509.25449
R6  https://arxiv.org/abs/2606.07031
R7  https://arxiv.org/abs/2511.08544
R8  https://arxiv.org/abs/2510.26777
R9  https://arxiv.org/abs/2607.01204
R10 https://ojs.aaai.org/index.php/AAAI/article/view/20881
```

## Important future note

Future work: regime aware specialist bank. Hypothesis - Instead of one continually adapted model, maintain a bank of frozen specialist encoders, each paired with its own prediction head. At run time, score specialists in parallel on recent data using prequential evaluation. Generate predictions, observe outcomes, then update only after the fact. Switch only if a challenger shows sustained, significant advantage under a pre-specified drift test. Keep the old specialist frozen. Adapt a copy or train a new one. And validate carefully against baselines: fixed model, continuously adapted model, shared encoder with multiple heads. Track routing errors, delay sensitivity, and switching cost. This is a future direction, not part of the first candidate.

Refinement note: the regime may change the head, not the encoder. Some regime shifts don't invalidate the representation. They invalidate how we interpret it. For example, an encoder that captures strong upward momentum could remain correct across regimes. But whether that predicts continuation or reversal might flip. In that case, one shared encoder with a bank of regime specific heads might be enough. That's simpler and more stable than swapping whole encoders. The test is whether head switching explains performance differences without changing the encoder. So the refinement is, keep representations stable when possible, adapt interpretations first.
