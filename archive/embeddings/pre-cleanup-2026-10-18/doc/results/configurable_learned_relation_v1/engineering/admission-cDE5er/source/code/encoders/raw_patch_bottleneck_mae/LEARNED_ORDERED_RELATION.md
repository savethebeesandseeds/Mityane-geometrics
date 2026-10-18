# Isolated RPB-v21 learned ordered-relation candidate

This separate candidate adds a trainable relation route to the unchanged
[RPB-v20 configurable model](CONFIGURABLE_SPECTRAL.md). It is provisional:
engineering execution and a prospective quality card are separate requirements.
No quality result or production/default change is implied by this module.

The v20 child keeps its backbone, shape projection, frozen spectral prior,
decoder, masks, scaler and geometry. A new bias-free projection starts at
literal zero without consuming initialization RNG. It adds only to the learned
`L = D - 4*C*(C-1)/2` coordinates. The fixed tail remains unchanged. At C3/F9/D32,
the route adds58,320 trainable values and still serves20 learned +12 fixed.
All F features and F² cross-channel feature pairs participate. C1 has no
cross-channel bank or additional projection values.

## Original-time common-support moments

For every semantic channel pair and feature pair, compare original ticks
`t` and `t+lag` at fixed physical lags1,2,4,8. A row contributes only when
all four feature endpoints are actually observed. Both endpoint banks use the
same support. Center each channel by its mean over both endpoint positions;
write the centered endpoint values as `a0,a1,b0,b1` and the common count as `n`.

`Ea = mean(a0²+a1²)` and `Eb = mean(b0²+b1²)` define the common denominator.
The odd moment is `mean(a0*b1-a1*b0)/sqrt(Ea*Eb)`, and the even moment is
`mean(a0*b0+a1*b1)/sqrt(Ea*Eb)`. Both are zero if `n<2` or either energy is
at most1e-12. Support is recorded separately as `n/(H-lag)`, including supported
constant inputs. The bank order is odd block, even block, then support block;
within each block: configured semantic pair, feature-i, feature-j, lag.

Masked storage is sanitized before subtraction and multiplication. Moments
run in CUDA float64 and exports in CUDA float32. The denominator clamps before
square root even for inactive rows. No missing tick is interpolated; packed
token rank, hidden target values and scoring labels never enter this route.
The odd moment changes sign under joint time reversal or semantic pair swap;
the even moment and support stay unchanged. Positive affine gains/offsets
cancel except for the declared absolute energy guard. Allmissing samples remain
invalid exact-zero native exports.

The bank is compared before its learned linear compression into L numbers:
`learned = base_learned + W*bank`, with `W=0` initially. Input geometry follows
the v20 C/H/F/D constraints, including nondivisible H and configured semantic
order. Lags are original ticks, independent of patch length or history padding;
quality at changed history/sampling interval requires separate evidence.

## API and serialization

Include `learned_ordered_relation_model.h` and construct
`LearnedOrderedRelationModel(c)`. Methods are `encode`, `decode`, `forward`,
`config`, `validate_state` and `validate_fixed_prior`. The historical child is
`.base`; the new parameter is `.relation_projection->weight`.
The forward uses the original visible training context for both routes and
keeps hierarchical waveform Huber as its sole objective.

The separate settings reader accepts only `MODEL_TAG=RPB-v21`; use
`config/learned_ordered_relation.conf`. The separate
`save_learned_ordered_relation_checkpoint` / `load_learned_ordered_relation_checkpoint`
functions bind the new tag, architecture, layout, lag/support rules, full v20
resolved config, semantic IDs, scaler and completed updates. Loaded fields are
`.config`, `.model`, `.scaler` and `.training_updates`. Saves use the existing
atomic no-overwrite helper. V20 and v21 typed loaders refuse each other's
archives. This is inference serialization, not optimizer/RNG training resume.

## What the change can establish

Zero initialization gives a same-seed, same-input initial v20 control. A later
gain in learned-only fixed-head scores would demonstrate useful learning under
this declared relation architecture. The covariance calculation is still an
engineered inductive prior; it is not a generic guarantee of market utility.
The original v14–v18 results showed that waveform learning can weaken timing
cues. Protecting the working fixed tail avoids reopening those stopped recipes.

At initialization `dLoss/dW = sum_b(g_learned(b)*bank(b)^T)`, so the shared
decoder can supply a first-step gradient despite W starting at zero. Finite
nonzero gradients establish connectivity only. Global normalized moments omit
absolute phase/amplitude, and uniform phase can weaken their expected waveform
gradient; the decoder must combine them with the existing learned shape route.
The fixed tail may provide a shortcut, and addition can dilute the learned cues.
Keep initial/trained controls and stop a failing frozen quality recipe without
loss, head, budget or coefficient rescue.

Focused artificial engineering command, inside the existing managed container:

```bash
source /etc/profile.d/embedding.sh
bash code/scripts/task.sh rpb-v21-engineering -j2 test-rpb-v21
```

The test requires actual CUDA for model execution. Its scalar CPU array witness
checks arithmetic only. It covers masks/poison, odd/even reversal, support and
energy guards, semantic permutations, generic/nondivisible dimensions, byte
parity and RNG at zero, actual Huber gradients and optimizer movement, immutable
prior/native tail, typed reload and malformed state/tag rejection. It does not
train a quality cohort or fit a classifier.
