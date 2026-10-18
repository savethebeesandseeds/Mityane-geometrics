# Archive readout v1

Policy: 1.2. Stage: synthetic development, validation only.

This encoder-independent protocol compares legal raw observations, standalone
raw PCA and exact native archived embeddings. It neither loads an encoder nor
trains one. There is no TEST input, discovery, checkpoint selection or acceptance
decision. Historical producer reports retain their original protocol.

## Frozen definitions

Each input explicitly declares TRAIN and VALIDATION observation/feature archives,
task, model tag, master, completed encoder updates, encoder producer fingerprint
and audited cohort/checkpoint lineage. The evaluator writes its instantiated
card before loading tensors or fitting anything. Its source identity is separate
from the archived encoder's identity. Four expected archive SHA256 values and an
expected feature provenance string can bind the input declaration to exact files.
All actual input SHA256 values and exact byte preservation are recorded.

Observation archives contain `observed`, bool `feature_mask`, binary int64
`labels_scoring_only` and `source_ids_json`. Only observed values supply signal;
hidden clean targets are never supplied to preprocessing, heads or scoring.
Native archives contain `features`,
bool `valid` and `provenance`, in the original observation archive row order.
Native archives do not contain source IDs: their row association is inherited
from audited producer lineage and immutable hashes, not recovered from vectors.
TRAIN/VALIDATION source groups must be disjoint and each source retains its
declared pair. Raw/PCA support requires at least one observed coordinate. Native
support is the encoder's archived declaration and may be a subset of those rows;
it cannot claim support for empty observations. Each method fits on its own
valid TRAIN rows; an unsupported native fit does not disable the raw controls.

1. **Raw data — no encoder:** TRAIN-fitted shared `ObservationScaler`, float64
   per channel/feature across rows/history, population standard deviation with
   the existing 1e-8 floor. Flatten normalized observed values (missing values
   zero), followed by original visibility flags. Shape 3×32×3 gives 576 inputs.
2. **PCA only — no encoder:** TRAIN-valid-row feature normalization and centered
   PCA of that same legal raw representation, to the declared compact width.
   Insufficient rank/support is explicit, never repaired by changing the width.
3. **Registered encoder tag:** exact native archived features at that compact
   width. No PCA or random projection follows the encoder.

Both classifiers fit weights and their own valid-TRAIN-row normalizers separately:
ridge penalty 1; neural head with 16 tanh hidden units, Adam 0.01, 100 updates.
At width 32 they fit 66 and 562 parameters; at raw width 576 they fit 1154 and
9266. Initial neural weights use `stream_seed(repetition.probe_seed, input_width)`:
equal-width methods share initialization. Three repetitions use 2701/2802/2903;
all repetitions are retained and averaged, never selected by validation scores.
These repetitions are classifier fits, not independent encoder trainings.

Validation reports include each support/count, accuracy on valid rows, full
population correctness, per-source predictions and 1000-resample source-group
conditional intervals. Pair comparisons use only their common valid validation
population, using each pair's validity intersection: native minus PCA-only,
native minus raw, and PCA-only minus raw.
Intervals are conditional on the fitted encoder and heads. Variation over
independent encoder masters is reported separately; tasks are not averaged.
Save fitted scalers, PCA, classifiers and predictions. Preserve ambient CPU/CUDA
RNG state. Every output directory and artifact must be new.

## Current phase 1 declaration

The first application uses all four tasks (`reversal`, `level`, `amplitude`,
`lag_sign`) and masters 1701/1802/1903 from the saved diversity-128 cohort.
Each input has 256 TRAIN examples (128 source pairs) and 128 VALIDATION examples
(64 source pairs), 10% natural missingness, shape 3×32×3. Native RPB-v4 exports
come from the saved learned-global milestone at 512 completed CUDA updates,
batch size 8. The compact width is 32. No encoder is retrained, and no TEST
archive is opened. Readout fits run on CPU with one thread.

See [the reporting standard](../../../doc/RESULTS_REPORTING_STANDARD.md),
[the continuation plan](../../encoders/raw_patch_bottleneck_mae/NEXT_ADVANCE.md),
and [evaluation integration](../README.md).
