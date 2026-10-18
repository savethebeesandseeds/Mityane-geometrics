# RPB-v22 content-only ordered relations

RPB-v22 is a separately tagged candidate around the unchanged
[RPB-v20 configurable model](CONFIGURABLE_SPECTRAL.md). It retains the
zero-start learned ordered-relation route introduced by v21, but projects only
its normalized odd/even content. Common counts determine normalization and
validity; observation fractions never become additive projection inputs.
V20 and the stopped v21 model, settings, checkpoints and results are preserved.

For a plain explanation of the input/output ranks, additional metadata, three
architecture routes, exact training targets and separate evaluation, see
[RPB-v22 explained](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/RPB_V22_ENCODER_GUIDE.md). The
[completed development report](../../../../archive/embeddings/pre-cleanup-2026-10-18/doc/CONFIGURABLE_CONTENT_RELATION_V1.md)
records RPB-v22.alt-01's passed screen and third check; the guide reuses those
measurements and performs no new experiment.

This tests a hypothesis suggested by the exploratory TRAIN support audit:
v21's additive support input can produce an offset when observation density
changes. Removing that route does not guarantee deletion invariance. The
remaining finite-sample moments and the backbone can still change when masks
change, and missing observations can erase information.

## Geometry and arithmetic

Input is values/masks `[B,C,H,F]`, with configured semantic channel IDs.
Output remains native `[B,D]`: `L=D-4*C*(C-1)/2` learned coordinates followed
by the unchanged four fixed spectral coordinates per semantic channel pair.
The v20 geometry constraints apply, including nondivisible logical H and
CUDA float32 model execution. C1 has no cross-channel relation bank or added
projection values. Configuration and geometry are fixed per checkpoint; B
may vary.

All configured channel pairs and all F² cross-channel feature pairs use
original tick lags 1/2/4/8. Each comparison requires four actually observed
endpoints at `t` and `t+lag`, with shared common support. Both endpoint banks
are centered jointly across their two positions. For centered endpoints
`a0,a1,b0,b1`:

- `Ea=mean(a0²+a1²)` and `Eb=mean(b0²+b1²)`.
- Odd content is `mean(a0*b1-a1*b0)/sqrt(Ea*Eb)`.
- Even content is `mean(a0*b0+a1*b1)/sqrt(Ea*Eb)`.

These are exactly the v21 odd/even formulas and operation order. They run in
CUDA float64, then export float32. Counts below 2 or either energy at most
1e-12 make both moments zero. The denominator clamps before square root,
including inactive rows. Masked storage is sanitized before arithmetic.
There is no interpolation, packed-token-time substitution, hidden-target
access or label input.

Let `N=C*(C-1)/2 * F² * 4`. Values have shape `[B,2*N]`, ordered as an odd
block followed by an even block; each block orders canonical semantic pair,
feature-i, feature-j, lag. Separate `[B,N]` integer counts and bool eligibility
are bookkeeping only. At C3/F9, values are 1,944 wide and the projection is
20x1944: 38,880 new trainable values. V21's 972 additive support inputs and
19,440 associated weights are absent.

The model adds `W*content` only to the base learned coordinates. W is
bias-free and literal zero initially, consuming no initialization RNG.
Identical seeds/inputs therefore permit exact initialized v20 native values,
validity and reconstruction. The fixed spectral tail remains literal and
frozen. Waveform decoding, training masks, TRAIN-fitted scaling and the sole
hierarchical waveform Huber objective remain unchanged.

## API and checkpoints

Include `content_ordered_relation_model.h` and construct
`ContentOrderedRelationModel(c)`. It exposes `.base` as the preserved
`ConfigurableSpectralModel` and `.relation_projection->weight`, with methods
`encode`, `decode`, `forward`, `config`, `validate_state` and
`validate_fixed_prior`.

Use `read_content_ordered_relation_settings(path)` with
`config/content_ordered_relation.conf`; its strict reader accepts only
`MODEL_TAG=RPB-v22`. The reusable `Config` settings remain independent of
the experiment's proposed RPB-v22.alt-01 run recipe.

`save_content_ordered_relation_checkpoint` and
`load_content_ordered_relation_checkpoint` bind tag, architecture/layout,
original lags, support/energy rules, full resolved geometry, semantic IDs,
scaler and completed updates. The loaded object has `.config`, `.model`,
`.scaler` and `.training_updates`. It rejects malformed/nonfinite state and
uses the existing atomic no-overwrite publisher. V20, v21 and v22 typed
loaders reject one another's archives. This is inference serialization;
optimizer/RNG resume remains outside the API.

Architecture ID:
`configurable-fixed-spectral-plus-zero-start-content-ordered-lag-v1`.
Checkpoint kind: `rpb_content_ordered_relation_inference_checkpoint_v1`.
The exact native-layout string is declared in
`content_ordered_relation_config.h` and stored in every typed checkpoint.

## Engineering and interpretation

Inside the existing managed container, after environment initialization:

```bash
source /etc/profile.d/embedding.sh
bash code/scripts/task.sh rpb-v22-core -j2 test-rpb-v22
```

The test uses an independent CPU scalar array witness for arithmetic and
actual CUDA for every encoder/decoder/optimizer execution. It checks odd/even
time reversal, energy/count guards, mask poison and allmissing behavior,
semantic permutations, C1 and generic nondivisible C/H/F/D, initialized
native/reconstruction/RNG parity, finite gradients and two optimizer updates,
fixed-tail preservation, exact reload and strict malformed/cross-tag refusal.

A separate artificial witness removes every sixteenth tick from alternating
perfectly correlated/anticorrelated signals. Counts change at every lag, but
the normalized nonzero even content remains equal. A deterministic nonzero
projection must produce equal relation contributions; a zero projection
would be vacuous. The full base encoder is not required to stay equal because
mask changes legitimately affect its input. This witness does not establish
invariance under arbitrary deletion.

Engineering is not quality evidence. Initial/trained controls, unchanged
heads and reconstruction/coverage guards belong to a separate prospective
card and quality protocol. No default, checkpoint promotion, unified-market
backport or new quality result follows from this module alone. Stop a failed
frozen recipe rather than tuning its heads, rate, lags, loss or budget.
