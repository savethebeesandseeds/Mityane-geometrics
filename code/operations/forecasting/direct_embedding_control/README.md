# Direct embedding control

Latent State Learning owns this independent frozen-Encoder control. Version v1
implements separately fitted horizon-specific residual MLPs in native embedding
coordinates. It does not update an Encoder or a State Representation Network.

The prospective recipe is registered in
[the numeric card](../../../../.archive/embeddings/pre-cleanup-2026-10-18/doc/latent_state_control_v1/numeric-card.json), before
implementation. Its D36/K4/tanh32 geometry has 5,828 parameters per horizon and
17,484 across horizons 1, 16 and 128, within the 18,000 parameter budget.
Per-horizon input means/stds and residual RMS scales use exactly the weighted,
complete, mature TRAIN associations accepted by the saved-array baselines.
Hidden weights use local seeded Xavier uniform initialization; biases and output
weights begin at zero. The initial native-coordinate forecast is persistence.

`versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/control.h`
provides `Config`, `Identity`, `Trainer`, input-only chronological `PredictionSet`
issuance and distinct immutable inference/training archives. Initializer and
sampler engines are local: none of these APIs consumes a global Torch RNG stream.
Training archives preserve named AdamW moments, sampler state, fitting arrays,
frozen transforms, identities and counters for exact continuation. Inference
archives carry frozen parameters, buffers and fit provenance, without optimizer
or training arrays. Checked publication and comparison receipts live alongside
the numerical implementation in `artifact.h`; consumers must use that adapter
with independent sidecar and ModelRef pins.

The registered fit uses CPU float64, 2,048 AdamW updates per horizon, microbatch32,
uniform replacement sampling, sample-weighted reference-scaled MSE, learning
rate0.001, betas0.9/0.999, epsilon1e-8, weight decay0.0001 and gradient clip1.
Repetition seeds are944101/944202/944303; each horizon's initializer and separately
serialized sampler use repetition seed plus horizon. There is no validation
selection, early stopping or search. Smaller explicitly artificial test recipes
exercise engineering only; they do not execute the market protocol or establish
point skill, state utility or portfolio value.

Run the focused witnesses only in the documented managed container:

```sh
bash code/scripts/task.sh control-v1 \
  test-direct-control-v1 test-direct-control-artifact
```

Existing evaluated modules and shared artifact v1 remain unchanged. This control
owns its separate publication extension and exact Encoder-parent contract.

Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../LICENSE.
