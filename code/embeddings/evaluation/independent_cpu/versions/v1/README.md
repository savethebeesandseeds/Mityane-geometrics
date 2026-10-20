# Independent saved CPU arithmetic, candidate v1

Optional C++ foundations for common arithmetic in saved-result validators.
On 18 October 2026 this source compiled with GCC 12.2 and passed its engineering
arithmetic checks in the managed `mityane-geometrics` container. It has not been
admitted as historical parity evidence. Frozen readers, fixtures, protocols,
model/source identities and completed results are retained.

The implementation is independent of Torch, model deserialization, fitted
producer code, encoder training and producer arithmetic libraries. Its inputs
are already decoded finite CPU values. Protocol adapters still own their exact
archive schemas, TRAIN-only bindings, role admissions, source/hash ledgers,
expected fitted recipes, protection floors, failure policy and report format.
These helpers do not establish any model quality or experimental result.

Implemented foundations:

- Decimal uint64 seed parsing without a floating JSON round trip; wrapping
  stream/SplitMix seeds; FNV-1a byte hashes and length-prefixed UTF-8 source IDs.
- A separate MT19937_64 state engine and exact uint64 multiply/reject uniform
  mapping. The implementation uses explicit portable 32-bit limbs for its
  128-bit product; test source compares to direct Linux wide multiplication.
- Finite-only float32 conversion and error-free partial-expansion summation
  with final half-even correction. The numeric profile checks IEEE binary32/
  binary64, round-to-nearest and gradual underflow. It refuses nonfinite inputs
  and intermediate sum overflow. Build with `-fno-fast-math -ffp-contract=off`.
- Saved-map normalization, independently summed linear logits, exact first-class
  tie argmax, masked scores, sorted source accuracy/paired groups and frozen
  1000-replicate whole-source bootstrap intervals (order indices 24 and 974).
  Empty and singleton group intervals retain undefined bounds.
- TRAIN-only population mean/scale helpers, observation deletion, query-patch
  eligibility/masks, and the original sequential channel MAE/Huber accumulation
  followed by compensated per-example/population aggregation.

The finite sum is intended to replace finite `math.fsum` paths, not to change
their tolerance or permit using approximate equality for saved class decisions.
Exact cross-platform libm behavior, especially `tanh`/PCA covariance replay,
and Python integer/float formatting are outside this foundation. The small
nonlinear-head adapter and full scaler asset identity encoding have not been
ported here. This foundation does not fit heads or run an SVD. FNV functions
hash supplied bytes; protocol adapters must serialize dtype enums, signed64
dimensions, little-endian raw values and floor bytes in the frozen order.
`source_manifest` expects already validated UTF-8 IDs, as supplied by the
restricted text reader; it does not normalize Unicode or reorder IDs.

Test source includes fixed FNV/SplitMix/MT goldens, uint64 boundary seeds,
multi-twist comparisons to the standard engine, frequent-rejection ranges,
float32 midpoint rounding, cancellation/halfway/subnormal sums, tie decisions,
TRAIN support/abstention and source-bootstrap/query reduction fixtures.
Compilation and the engineering test passed without compiler warnings.
Comparison to archived numerical fixtures and protocol integration remain
pending. No Python execution path has been removed on the strength of these
engineering checks.

Use the isolated root target `check-independent-cpu-v1`, invoked through
`.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'migration-arithmetic', 'check-independent-cpu-v1')`
in the existing managed Linux container. It
does not link Torch and keeps products under `/opt/mityane-geometrics`.
The root exposes candidate migration goals explicitly. Normal maintenance/model
targets do not run these tests. Recipe guards require the container mount and
canonical named output paths; compiler/tests share a directory lock and named
`TMPDIR`, and compiled binaries are published by atomic rename. The validation
log is `/opt/mityane-geometrics/embeddings/build/migration-native-validation/validation.log`.
