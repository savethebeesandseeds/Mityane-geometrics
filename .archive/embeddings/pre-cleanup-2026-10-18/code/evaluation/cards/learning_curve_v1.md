# Validation-selected learning curve v1

This is a synthetic development diagnostic for training duration. The shared
engine owns source splits, fixed reconstruction target masks, readout fits,
validation budget selection and final testing. Encoder adapters own label-free
training and exact checkpoint export/decoding.

`CurveTrainerFactory` receives `ProviderFitInput`, which contains permitted
training observations and schema metadata only. Its monotonic `train_to` callback
preserves one optimizer, scaler and absolute attempt stream. `snapshot` loads an
independent copy of each saved checkpoint and exposes global/concatenation feature
surfaces plus standardized compact reconstruction.

The current card fixes three fresh masters, 32 training source pairs, 64 validation
pairs, 64 final-test pairs, 10% natural missingness, and milestones 0/128/512/2048.
Fixed reconstruction enumerates each original patch, scores observed target cells
only and requires at least two retained visible patches per selected channel.
The zero checkpoint preserves the exact initialization used by the trained path.

Normalization, centered PCA and fixed ridge/tiny probes fit training rows only,
separately per checkpoint. Selection chooses a common nonzero update budget by
mean validation concatenation/PCA36 ridge accuracy across both architectures and
all masters, using within-master common valid rows. Unsupported primary entries
exclude the entire budget. Exact ties favor the smallest budget.

The driver persists `selection.json` before generating any final-test observations.
It then applies the retained selected checkpoints and fitted readouts to fresh
named test streams, without further training or fitting. All budgets and seeds
remain archived, including sampled losses and fixed train/validation reconstruction.

The development runner is `embedding_learning_curve`, built with `make learning-curve`.
Its RPB registrations explicitly require CUDA training; small frozen feature/probe
calculations run on CPU. `--gpu-check true` performs four updates per architecture
and verifies actual CUDA parameters/input/loss, finite gradients and changed weights.
Run that check before the full experiment.

Equal update budgets do not equalize model parameters or runtime. Source-group
intervals are conditional on each fitted checkpoint/readout. The card provides no
consumer acceptance, multiplicity correction or uncertainty over retraining seeds.

See the [RPB recipe and evidence](../../encoders/raw_patch_bottleneck_mae/LEARNING_CURVE.md)
and [evaluation policy](../../../doc/EMBEDDING_EVALUATION_POLICY.md).
