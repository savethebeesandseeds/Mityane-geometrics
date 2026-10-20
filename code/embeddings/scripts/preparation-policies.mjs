/** Shared source-name constants and fixed recipe fragments, never mutable policy globals. */
export const POLICY_SOURCE = 'code/embeddings/scripts/preparation-policies.mjs';
export const RPB = 'code/embeddings/encoders/raw_patch_bottleneck_mae';
export const REPRESENTATION = 'code/shared/representation';
export const FIXED_SOURCES = [
  `${REPRESENTATION}/include/embedding/shared/fixed_feature_readouts.h`,
  `${REPRESENTATION}/src/fixed_feature_readouts.cpp`, `${REPRESENTATION}/tests/fixed_feature_readouts_test.cpp`,
];
export const EARLY_SOURCES = [
  ...FIXED_SOURCES, 'code/embeddings/evaluation/include/frozen_role_guard.h',
  `${RPB}/shared/include/embedding/encoders/raw_patch_bottleneck_mae/config.h`,
  `${RPB}/shared/include/embedding/encoders/raw_patch_bottleneck_mae/model.h`,
  `${RPB}/shared/include/embedding/encoders/raw_patch_bottleneck_mae/early_mixer_adapter.h`,
  `${RPB}/shared/src/early_mixer_adapter.cpp`, `${RPB}/shared/src/learning_curve_adapter.cpp`, `${RPB}/shared/src/workflow.cpp`,
  `${RPB}/versions/v10/tests/early_mixer_model_test.cpp`, `${RPB}/shared/tests/early_mixer_adapter_test.cpp`,
];
export const GAIN_SOURCES = [
  `${RPB}/versions/v11/include/embedding/encoders/raw_patch_bottleneck_mae/training_source_gain.h`,
  `${RPB}/versions/v11/src/training_source_gain.cpp`,
  `${RPB}/shared/include/embedding/encoders/raw_patch_bottleneck_mae/matched_target_gain_adapter.h`,
  `${RPB}/shared/src/matched_target_gain_adapter.cpp`, `${RPB}/shared/tests/matched_target_gain_adapter_test.cpp`,
];
export const EARLY_MARKERS = ['Early mixer model CUDA admission passed', 'Early mixer CUDA adapter admission passed'];
export const READOUT_MARKERS = ['Fixed feature readout tests passed', 'Frozen role guard checks passed'];
export const EARLY_GATES = [
  'default/early CUDA model contracts and typed architecture archives',
  'unchanged continuous coordinate15 CUDA updates and immutable snapshots',
];
export const READOUT_GATE = 'fixed generic paired readouts and whole-role guard fixtures';
export const INITIAL_PLAN = {
  encoder_trajectories: 10, decoder_updates: 0, batch_size: 8, train_pairs: 128, validation_pairs: 64,
  test_pairs: 0, shape: [3, 32, 3], native_width: 32, head_repetitions: [2701, 2802, 2903], deletion_rate: .30,
  encoder_device: 'CUDA', prior_quality_input_roles: [], quality_generated: false,
  testing_accessed: false, stress_accessed: false, selection: false, promotion: false,
};
export const POINT_PLAN = {
  ...INITIAL_PLAN, encoder_updates_each: 512, attempt_limit: 1024, log_every: 1,
  skipped_attempts_permitted: false, retained_points: 20, sampled_rows: 40960,
  planned_native_export_calls: 120, planned_query_writer_calls: 20, planned_query_forwards: 80,
  driver_raw_outer_fits: 10, raw_outer_fit_shared_with_PCA: true, quality_gain: false,
};
export const POINT_COMPLETE = {
  cohorts: 5, tasks_each: 2, encoder_trajectories: 10, encoder_updates_each: 512, attempt_limit: 1024,
  sampled_rows: 40960, retained_points: 20, skipped_attempts: 0, full_native_export_calls: 120,
  query_evaluation_calls: 20, necessary_query_forwards: 80, driver_raw_outer_fits: 10,
  selection: false, promotion: false, CPU_encoder_training: false, CPU_encoder_forward: false,
  testing_accessed: false, stress_accessed: false,
};
export const NO_EXTRA_DECODER = {
  extra_decoder_calibration_updates: 0,
  decoder_update_scope: 'joint reconstruction during encoder updates; no extra decoder-only calibration',
};

export function protocolSources(stem, card, extra, { setup = true } = {}) {
  return [POLICY_SOURCE, card, 'Makefile', 'dependencies.lock', ...(setup ? ['setup.sh', 'code/scripts/verify-libtorch.mjs',
    'code/scripts/node.sh', 'code/scripts/node-runtime.sh', 'code/scripts/setup-node.sh'] : []),
    `code/embeddings/evaluation/src/${stem.replaceAll('-', '_')}_main.cpp`, ...extra,
    'code/embeddings/scripts/prepare-fresh-decoder-replication.mjs', `code/embeddings/scripts/prepare-${stem}.mjs`,
    `code/embeddings/scripts/check-${stem}.sh`, `code/embeddings/scripts/evaluate-${stem}.sh`];
}
