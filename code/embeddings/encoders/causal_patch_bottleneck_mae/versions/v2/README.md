# CPB-v2: end-to-end causal fine-tuning

## Channel axis

cuwacunu_torch `interval_type_e` is the standard. One window is one instrument. Each channel is one time resolution of that instrument: minutes, hours, weeks, and the other kline intervals. The channel id is that interval's torch ordinal. Different instruments are different rows. A stored id set `{0,1,2}` is the withdrawn instrument binding from historical checkpoints. It is not this axis, and scores measured on it stay scores of that binding. The contract is `code/embeddings/encoders/include/embedding/encoders/timeframe_channel.h`.

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
Inherited notices remain in their copied files and LICENSES/NOTICE.

This independent version owns all 42 implementation/parser headers copied
from frozen CPB-v1. No v1, RPB, sibling checkout or shared mutable header is
required. The public header is
`embedding/encoders/causal_patch_bottleneck_mae/versions/v2.h`; its namespace is
`embedding::encoders::causal_patch_bottleneck_mae::v2`.

The sole initializer is the pinned CPB-v1.market-01 artifact, SHA256
54c06e1c9af9215bd3477c343579feb63bce092c0f7dedf3c7355fc21d761895.
`import_v1_checkpoint` reads data through this version's owned parser,
validates the exact source family/provenance/state and creates a fresh causal
update count of zero. It executes no v1 encoder. Initial model/scaler/bank
values must match the source exactly. Initial saved-feature parity is checked
by the separately admitted research runner before any optimizer update.

Forward arithmetic and native36 geometry are unchanged at initialization.
Training now permits gradients through the causal temporal parent, parent
reconstruction decoder, four-coordinate change branch and variance head.
The literal spectral prior remains frozen. The learned parent32 may change
after training; CPB-v1 remains the immutable application baseline.

`RawHistory`, `encode_at` and `trace_at` retain the explicit raw CPU-F64,
boolean-mask, semantic-ID, actual-timestamp and per-row cutoff contract.
Observations after the cutoff are cropped before value validation, scaling,
banks or tokenization. The same causal temporal attention, same-time channel
mixing and retained-window pooling apply. External fixed scaler policies are
permitted for frozen inference without changing the checkpoint scaler or bank.

`forward_at(raw, cutoffs, hidden_query, scaler, context_deleted)` crops the raw
observation AND target source before processing masks. Targets use only the
original query-hidden observed cells; context deletion removes additional
query-visible inputs and never creates targets. Training requires train mode
and enabled gradients. Fixed-query reconstruction requires eval mode under a
NoGradGuard. The Gaussian waveform NLL, variance link and precision order are
unchanged. No task labels, auxiliary loss or synthetic training are added.

The `training` namespace provides explicit fresh learned-only AdamW creation,
complete typed save/load and counter-based progress. The optimizer excludes
only fixed parameters. Serialized parameter order, presence/steps and moments,
source/split/recipe identities, causal policy, counters and seeds are bound.
Inactive learned parameters may have no optimizer state; their exact absence
is preserved. The old four-tensor trainer and its zero-head assumptions are
not this version's training contract.

Ordinary inference accepts only `cpb_v2_inference_checkpoint_v1`; ordinary
resume accepts only `cpb_v2_causal_training_state_v1`. Initializer import,
inference and resume are distinct APIs. The private nested payload retains the
copied decoder's tensor schema, while the mandatory outer v2 envelope selects
the causal/training semantics. Publication never overwrites an existing file.

The engineering test exercises all128 cutoffs,2350 suffix interventions and30
internal temporal cases both before and after two artificial CUDA updates.
It checks every imported parameter/scaler/bank, target masking, actual CUDA
learning categories, fixed-prior preservation, exact typed next-update resume
and refusal of malformed or mismatched artifacts, including altered finite optimizer moments. A separate `--final-checkpoint` mode requires the actual 2048-update typed artifact and repeats the complete causal matrix with zero new updates before quality. These are engineering
updates, not the one separately declared2048-update market quality trajectory.

Implementation is pending admission/build/proof; this README asserts no
measured quality or passed runtime check. See the prospective research card
`.archive/embeddings/pre-cleanup-2026-10-18/doc/CAUSAL_FINETUNE_V2_CARD.md` and its machine-readable companion.
