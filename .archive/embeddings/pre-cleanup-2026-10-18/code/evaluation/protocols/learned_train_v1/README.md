# Saved TRAIN diagnostic

This stdlib-only diagnostic reads selected saved TRAIN coordinates and decoder
parameters without constructing or executing an encoder. Its prospective scope
and interpretation limits are fixed in
[the card](../../cards/learned_train_v1.md). The lazy reader enumerates tensor
descriptors but decodes only explicitly allowed fields, excluding validation,
labels, classifier weights and predictions.

Use the existing managed container; this is saved-array CPU arithmetic:

```sh
source /etc/profile.d/embedding.sh
bash code/evaluation/protocols/learned_train_v1/run_train.sh
```

The runner locks a separate named internal session and preserves a unique
capsule with frozen source/card/input digests, engineering checks, every
declared numerical result, selected-field inventory and original-artifact hash
checks before/after. No build, dependency installation or GPU operation occurs.
