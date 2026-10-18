# Saved TRAIN variation and decoder-weight diagnostic

Protocol: `learned-train-v1`. Prospective, before numerical TRAIN calculations.
This is an arithmetic diagnostic of existing RPB-v20.alt-01 evidence, not a new
model, quality evaluation, architecture selection gate, or promotion.

## Closed inputs and execution

Use only original profile A, masters 941101/941102/941103, points 0 and 512,
from `output/runs/rpb-v20-capacity/admission-0gN4uD`. Original artifact manifest
SHA256 is `62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83`.
Original source fingerprint is
`8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1`.

Numerically decode only `training_values`, `training_valid`, and provenance
metadata from the six `features.pt` files. These historical archives predate
explicit source IDs: bind their stored TRAIN row order to the original hashes
and completed branch diagnostic's saved TRAIN normalizer source IDs. Decode only geometry,
TRAIN scaler, first decoder linear layer weights, and provenance metadata from
the corresponding six `checkpoint.pt` files. Read only the saved TRAIN mean,
scale and row-count normalizer fields in repetition2701 fit archives for
fixed12, learned20 at0 and learned20 at512 from the completed branch capsule
`output/runs/rpb-v20-branches/diagnostic-YA3Nws`. Do not decode fitted head
weights, labels, predictions, held-out numeric tensor storage, or query banks.
Lazy pickle tensor descriptors must prevent decoding unselected storage.
Reject selected storage aliased by any unselected tensor descriptor. Decode
only logical elements at the selected offset/strides, including bounded
noncontiguous views; record storage size and selection spans. Reading bytes to
extract a field does not authorize decoding other storage elements.
Archive key/shape enumeration and whole-file hashing are permitted.

Verify the original and branch manifest/completion bindings. Freeze this card,
all diagnostic source, the pinned independent archive codec, and selected input
hashes before numerical calculation. Verify all815 original artifact hashes
before and after the diagnostic. Use one new output capsule, retain failures,
and publish files exclusively. Run inside the existing managed embedding
container with its existing dependencies. Use Python stdlib only: no torch
import, encoder/module construction, forward call, head fitting, optimization,
GPU computation, new training, validation scoring, TEST, or market input.

## Exact numerical outputs

Use valid TRAIN rows only; preserve all512 rows and256 adjacent source pairs
when valid. Pair order is stored row1 minus row0, without reading labels or
orienting differences to a target. Reject malformed/nonfinite geometry,
nonunique paired source IDs, changed point-to-point row IDs or support.
Native indices0..19 are learned20; indices20..31 are fixed12.

For every coordinate, master and point, report mean, population standard
deviation (denominator N), minimum, maximum, RMS, exact distinct count, exact
constant flag, and whether population std is below the existing normalizer
floor1e-8. Report branch mean standard deviation, total centered RMS and
covariance participation rank `(trace(C))^2 / trace(C*C)`; zero covariance has
rank0. This is a variance-concentration statistic, not an information metric.

For every source pair, master, point and branch, retain the full signed
difference vector and its L2 norm/RMS. Report branch mean and median pair norm,
mean absolute coordinate difference, signed coordinate means, and exact zero
pair count. Compare0 to512 on the same rows: exact changed-cell count,
maximum absolute change, per-coordinate signed mean change/RMS change,
per-row L2 changes, and full per-pair signed difference-change vectors and
norms. Require exact fixed12 values and TRAIN validity at0/512.

Read original TRAIN scaler means/scales/counts/floor flags for all C3/F9
coordinates and require exact equality between0/512. Recalculate branch TRAIN
mean/std and compare each selected saved outer normalizer with mean and
`max(population std,1e-8)`, using absolute/relative tolerance2e-9. Report the
saved outer, ridge and neural normalization scales, flags and fitted-row
counts; inspect only these normalization fields, with no scoring or fitting.

The first decoder matrix has128 outputs and64 inputs: learned20, fixed12,
16 patch-position metadata inputs, and16 channel metadata inputs. Report each
column's L2 weight norm and block Frobenius norms at0/512. For embedding
columns, also report `TRAIN std * column L2 norm`, its block root-sum-square,
and the covariance-aware centered linear contribution RMS
`sqrt(sum_h sum_jk W[h,j]*C[j,k]*W[h,k])` for each embedding block. Metadata
gets raw weight norms only; metadata norms are not directly comparable to
scale-adjusted embedding norms. These are potential first-layer linear use
under full-context TRAIN exports. They are not decoder activation, gradients,
masked-training use, reconstruction causality, or demonstrated timing use.

## Interpretation and limits

Retain every declared metric/master/point, without adaptive thresholds,
coordinate selection, classifier changes, or new numerical probes. TRAIN
exports use full legal context, unlike masked contexts during the original
training objective. Pair differences may contain useful signal, nuisance or
both. Small variance/rank or decoder column norms alone do not prove collapse,
information loss, causal reliance, or the cause of near-chance timing results.
Same-row training movement is evidence of learning-related change, not timing
improvement. Do not describe this arithmetic as an encoder rerun on CPU.

Report zero encoder calls, zero updates, zero head fits, zero validation values
decoded, read-field inventory, source/card/input bindings, engineering tests
under normal and optimized Python, and exact original-hash preservation.
