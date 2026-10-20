// Closed metadata admission. Docker-stop source edits; no execution evidence.
import { fs, path, check, directory, same, records, preserved, document, text, hash, copyMatrix,
  emptyTarget, capture, verifySources, freezeScopes, compileCommands, command, inventory,
  utc, walk, cli, appendAdmitted, options } from '../node/v1/saved_run_support.mjs';
import { writeJSON } from '../../../../shared/artifacts/node/v1/files.mjs';

export const PROTOCOL = 'structured-hard-timing-comparison-v2';
export const RUN_ROOT = '.temp/embeddings/runs/rpb-structured-hard-timing';
export const CODE = 'code/embeddings/evaluation/protocols/structured_hard_timing_v1';
export const CARD = 'code/embeddings/evaluation/cards/structured_hard_timing_comparison_v2.md';
export const CARD_SHA = '35f7aa9987f293bafbe735506cc86cd2af02f2ab39bec42dd1dc2d2dedfa758c';
export const DATA_ROOT = 'code/embeddings/evaluation/benchmarks/structured_hard_timing';
export const DATA_BINARY = '/opt/mityane-geometrics/embeddings/build/structured-hard-timing-data/code/benchmarks/structured_hard_timing/cross_feature_test';
export const QUALITY_BINARY = '/opt/mityane-geometrics/embeddings/build/rpb-structured-hard-timing/embedding_structured_hard_timing';
export const RULE = 'observed-cross-feature-determinant-v2';
const config = { protocol: PROTOCOL, code: CODE, card: CARD, cardSHA: CARD_SHA, scopeTarget: 'print-structured-hard-timing-scope-inputs' };
const helperSources = ['code/scripts/verify-libtorch.mjs', 'code/scripts/node.sh', 'code/scripts/node-runtime.sh',
  'code/embeddings/evaluation/protocols/node/v1/saved_run_support.mjs', 'code/shared/artifacts/node/v1/files.mjs', `${CODE}/prepare_saved_run.mjs`];

export function informationSources(repo) {
  const names = ['Makefile', 'dependencies.lock', 'setup.sh', CARD, ...helperSources,
    ...['Makefile', 'structured_hard_timing.h', 'structured_hard_timing.cpp', 'cross_feature_test.cpp', 'cross_feature_solvability.h', 'cross_feature_solvability.cpp'].map(name => `${DATA_ROOT}/${name}`),
    'code/shared/representation/src/feature_harness.cpp'];
  names.push(...walk(path.join(repo, 'code/shared/representation/include')).filter(name => name.endsWith('.h')).map(name => path.relative(repo, name).split(path.sep).join('/')));
  return [...new Set(names)].sort();
}
export function validateInformation(value) {
  same(Object.keys(value).sort(), ['protocol', 'dataset_id', 'rule_id', 'complexity', 'complexity_maximum', 'seeds', 'passed', 'encoder_or_head_fits', 'TEST_generated'].sort(), 'closed information schema');
  check(value.protocol === 'structured-hard-timing-information-v2' && value.rule_id === RULE && value.dataset_id === 'TEMPO-3' &&
    value.complexity === 4 && value.complexity_maximum === 5 && value.encoder_or_head_fits === 0 && value.TEST_generated === false, 'information recipe/zero fit scope');
  check(value.seeds.length === 2, 'both engineering seeds'); let passed = true;
  for (let i = 0; i < 2; ++i) {
    const row = value.seeds[i]; same(Object.keys(row).sort(), ['seed', 'training_source_pairs', 'validation_source_pairs', 'intact', 'deleted'].sort());
    check(row.seed === [82585, 83686][i] && row.training_source_pairs === 512 && row.validation_source_pairs === 512, 'fixed engineering source population');
    for (const [view, accuracyMinimum, coverageMinimum] of [['intact', .99, .99], ['deleted', .98, .95]]) {
      const score = row[view]; same(Object.keys(score).sort(), ['total', 'valid', 'correct', 'accuracy', 'coverage', 'full_population_correctness', 'passed'].sort());
      const { total, valid, correct } = score;
      check([total, valid, correct].every(Number.isSafeInteger) && total === 1024 && 0 <= correct && correct <= valid && valid <= total, 'information integer bounds');
      for (const [name, expected] of [['accuracy', valid ? correct / valid : 0], ['coverage', valid / total], ['full_population_correctness', correct / total]])
        check(typeof score[name] === 'number' && Number.isFinite(score[name]) && Math.abs(score[name] - expected) <= 2e-15, `exact rational information score: ${name}`);
      const ok = score.accuracy >= accuracyMinimum && score.coverage >= coverageMinimum;
      check(score.passed === ok, 'fixed information gate'); passed = passed && ok;
    }
  }
  check(value.passed === passed, 'whole information gate'); return passed;
}
export function expectedPlan() {
  return { protocol: PROTOCOL, timing_master_seeds: [75272, 76373, 77474, 78575, 79676], dataset_codename: 'TEMPO-3', dataset_generator_protocol: 'structured-hard-timing-v1', dataset_id: 'TEMPO-3',
    recipe_id: 'structured-hard-timing-v1', complexity_is_designed_not_accuracy_derived: true, designed_complexity_level: 4, complexity_scale_max: 5, task: 'lag_sign', tags: ['RPB-v7.alt-05', 'RPB-v10.alt-05'],
    encoder_trajectories: 10, encoder_updates_each: 512, attempt_limit: 1024, log_every: 1, skipped_attempts_permitted: false, retained_points: 20, decoder_updates: 0, extra_decoder_calibration_updates: 0,
    decoder_update_scope: 'joint reconstruction during encoder updates; no extra decoder-only calibration', batch_size: 8, sampled_rows: 40960, train_pairs: 128, validation_pairs: 64, test_pairs: 0, shape: [3, 32, 3],
    native_width: 32, parameter_count_each: 225805, channel_mixer_placements: [0, 1], global_pool_input_sources: [0, 0], checkpoint_roles_per_point: 5,
    external_fit_protocol: PROTOCOL + '/lag_sign', implementation_fit_protocol: 'early-mixer-reliability-v1/lag_sign', point_binding_suffix: '.structured-timing.pt', head_repetitions: [2701, 2802, 2903],
    planned_pipelines: 105, planned_heads: 210, deletion_rate: .30, methods_each_task: 7, separate_initial_controls: 2, planned_unique_quality_native_exports: 60, planned_initial_counterpart_exports: 0,
    planned_native_export_calls: 60, planned_query_writer_calls: 20, planned_query_forwards: 80, driver_raw_outer_fits: 5, helper_outer_train_fits: 25, raw_outer_fit_shared_with_PCA: true,
    quality_gain: false, analytic_solver_fits: 0, analytic_individual_heads: 0, analytic_surface_evaluations: 15, information_rule_id: RULE, encoder_device: 'CUDA', prior_quality_input_roles: [], quality_generated: false,
    testing_accessed: false, stress_accessed: false, selection: false, promotion: false, human_card_sha256: CARD_SHA };
}
export function validatePlan(value) {
  const { source_fingerprint: id, ...plan } = value; same(plan, expectedPlan(), 'closed actual producer plan');
  check(typeof id === 'string' && /^[0-9a-f]{64}$/.test(id), 'actual compiled source fingerprint'); return plan;
}
export function verifyInformation(repo, recordPath) {
  check(path.basename(recordPath) === 'information-record.json' && path.dirname(path.dirname(recordPath)) === path.join(repo, RUN_ROOT, 'information-admission'), 'closed information binding path');
  const record = document(recordPath);
  check(record.protocol === 'structured-hard-timing-information-binding-v2' && record.rule_id === RULE && record.passed === true && record.status === 'passed' && record.card_sha256 === CARD_SHA &&
    record.source_preserved_after === true && record.archive_reads === 0 && record.encoder_or_head_fits === 0, 'passed information record');
  check(validateInformation(record.information), 'fixed information gates'); preserved(record.inputs);
  same(document(path.join(path.dirname(recordPath), 'information.json')), record.information, 'raw information association');
  check(verifySources(repo, path.dirname(recordPath)).source_sha256 === record.source_sha256, 'information SOURCE closure'); return record;
}
export function qualityRequirements(binary) {
  const base = path.dirname(binary), old = 'code/embeddings/encoders/raw_patch_bottleneck_mae/shared/src/';
  const od = `${base}/code/encoders/raw_patch_bottleneck_mae`, nd = `${base}/code/protocols/structured_hard_timing_v1`;
  return [[old + 'workflow.cpp', `${od}/workflow.o`, 'RPB_SOURCE_ID', 'core_writer'],
    [old + 'learning_curve_adapter.cpp', `${od}/learning_curve_adapter.o`, 'EVALUATION_SOURCE_ID', 'curve_training'],
    [old + 'early_mixer_adapter.cpp', `${od}/early_mixer_adapter.o`, 'EARLY_MIXER_ADAPTER_SOURCE_ID', 'early_adapter'],
    [`${CODE}/structured_hard_timing_adapter.cpp`, `${nd}/structured_hard_timing_adapter.o`, 'STRUCTURED_HARD_TIMING_ADAPTER_SOURCE_ID', 'structured_adapter'],
    [`${CODE}/paired_timing_run.cpp`, `${nd}/paired_timing_run.o`, 'EVALUATION_SOURCE_ID', 'structured_adapter'],
    [`${CODE}/structured_hard_timing_main.cpp`, `${nd}/structured_hard_timing_main.o`, 'EVALUATION_SOURCE_ID', 'structured_adapter'],
    [`${DATA_ROOT}/structured_hard_timing.cpp`, `${nd}/structured_hard_timing_generator.o`, 'EVALUATION_SOURCE_ID', 'structured_adapter'],
    [`${DATA_ROOT}/cross_feature_solvability.cpp`, `${nd}/cross_feature_solvability.o`, 'EVALUATION_SOURCE_ID', 'structured_adapter']];
}
function readerBundle(root) {
  directory(root); const seal = document(path.join(root, 'reader-seal.json'));
  check(seal.status === 'passed' && seal.measured_execution_enabled === false, 'reviewed blocked reader');
  const names = seal.files.map(row => row.path); same(names, [...new Set(names)].sort());
  check(names.every(name => typeof name === 'string' && name.length > 0 && !/[\\/]/.test(name) && !['.', '..'].includes(name) && !name.endsWith('.pt')), 'closed reader SOURCE names');
  check(['validate_structured_hard_timing.py', 'source-fixtures.json', 'saved_cpu_math.py', 'archive_codec.py'].every(name => names.includes(name)), 'complete independent reader');
  const actual = records(names.map(name => path.join(root, name)));
  actual.forEach((row, i) => check(row.bytes === seal.files[i].bytes && row.sha256 === seal.files[i].sha256, 'reader bytes'));
  const source = text(path.join(root, 'validate_structured_hard_timing.py'));
  check(source.split('\nREVIEWED_SCHEMA = False\n').length === 2 && source.split('\nMEASURED_IMPLEMENTATION = False\n').length === 2, 'prequality false flags');
  const fixture = document(path.join(root, 'source-fixtures.json')); check(fixture.status === 'passed' && fixture.measured_execution_enabled === false && fixture.archive_reads === 0, 'artificial-only fixtures');
  return [...names, 'reader-seal.json'];
}
export function run(action, args) {
  const declarations = {
    'self-test': [[], []],
    'freeze-information': [['repo_root', 'target', 'card_sha256', 'sources'], ['repo_root', 'target']],
    'admit-information': [['repo_root', 'target', 'binary'], ['repo_root', 'target']],
    'freeze-admission': [['repo_root', 'target', 'card_sha256', 'sources'], ['repo_root', 'target', 'card_sha256', 'sources']],
    admit: [['repo_root', 'target', 'binary', 'compiled_source_id', 'information'], ['repo_root', 'target', 'binary', 'compiled_source_id', 'information']],
    'freeze-run': [['repo_root', 'target', 'admission', 'reader_dir'], ['repo_root', 'target', 'admission', 'reader_dir']],
    finalize: [['repo_root', 'target'], ['repo_root', 'target']]
  };
  check(Object.hasOwn(declarations, action), 'closed structured metadata action'); options(args, ...declarations[action]);
  if (action === 'self-test') {
    const score = { total: 1024, valid: 1024, correct: 1024, accuracy: 1, coverage: 1, full_population_correctness: 1, passed: true };
    const value = { protocol: 'structured-hard-timing-information-v2', dataset_id: 'TEMPO-3', rule_id: RULE, complexity: 4, complexity_maximum: 5,
      seeds: [82585, 83686].map(seed => ({ seed, training_source_pairs: 512, validation_source_pairs: 512, intact: { ...score }, deleted: { ...score } })), passed: true, encoder_or_head_fits: 0, TEST_generated: false };
    check(validateInformation(value), 'information positive fixture'); return { status: 'passed', archive_reads: 0, encoder_or_head_fits: 0, real_SDK_reads: 0, source_fixture_only: true };
  }
  const repo = directory(args.repo_root ?? '/mityane');
  if (action === 'freeze-information') {
    const target = emptyTarget(repo, args.target, RUN_ROOT, 'information-admission'), required = informationSources(repo), names = args.sources ?? required;
    check(required.every(name => names.includes(name)), 'all information compilation/helper dependencies'); check((args.card_sha256 ?? CARD_SHA) === CARD_SHA, 'fixed card');
    const id = capture(repo, target, names, config); writeJSON(path.join(target, 'freeze.json'), { protocol: 'structured-hard-timing-information-v2', rule_id: RULE, source_sha256: id, card_sha256: CARD_SHA, utc: utc(), quality_generated: false });
    return { status: 'frozen', target, source_sha256: id, sources: names.length };
  }
  if (action === 'freeze-admission') {
    check(args.card_sha256 === CARD_SHA, 'fixed card'); const target = emptyTarget(repo, args.target, RUN_ROOT, 'admission');
    const id = freezeScopes(repo, target, args.sources, config, ['core_writer', 'curve_training', 'early_adapter', 'structured_adapter'], 'structured_adapter');
    writeJSON(path.join(target, 'freeze.json'), { protocol: PROTOCOL, source_sha256: id, card_sha256: CARD_SHA, utc: utc(), quality_generated: false }); return { status: 'frozen', target, source_sha256: id };
  }
  const target = directory(args.target);
  if (action === 'admit-information') {
    const binary = args.binary ?? DATA_BINARY; check(path.dirname(target) === path.join(repo, RUN_ROOT, 'information-admission') && binary === DATA_BINARY, 'fixed information family/binary');
    const inputs = records(['information.json', 'build-and-tests.log', 'source-manifest.json', 'source-inputs.sha256', 'source-fingerprint.txt', 'freeze.json'].map(name => path.join(target, name)).concat(binary));
    const manifest = verifySources(repo, target), log = text(path.join(target, 'build-and-tests.log'));
    check(log.includes('Cross-feature timing data-only fixtures passed: 54 affine noiseless cases; no encoder/head fitting or quality evaluation'), 'actual data fixture marker');
    const compiler = compileCommands(log, ['structured_hard_timing', 'cross_feature_test', 'cross_feature_solvability'].map(name => [`${DATA_ROOT}/${name}.cpp`, `${path.dirname(binary)}/${name}.o`]), binary);
    const information = document(path.join(target, 'information.json')), passed = validateInformation(information); verifySources(repo, target); preserved(inputs);
    writeJSON(path.join(target, 'information-record.json'), { protocol: 'structured-hard-timing-information-binding-v2', rule_id: RULE, status: passed ? 'passed' : 'failed', passed, utc: utc(),
      card_sha256: CARD_SHA, source_sha256: manifest.source_sha256, inputs, compiler_evidence: compiler, information, source_preserved_after: true, encoder_or_head_fits: 0, archive_reads: 0, quality_generated: false });
    check(passed, 'information failed; preserve evidence and do not train'); return { status: 'passed', path: path.join(target, 'information-record.json'), sha256: hash(path.join(target, 'information-record.json')) };
  }
  if (action === 'admit') {
    check(path.dirname(target) === path.join(repo, RUN_ROOT, 'admission') && args.binary === QUALITY_BINARY, 'fixed quality admission/binary');
    const logPath = path.join(target, 'build-and-tests.log'), bound = records([QUALITY_BINARY, logPath]);
    const information = verifyInformation(repo, args.information), manifest = verifySources(repo, target), id = command(QUALITY_BINARY, ['--source-id']);
    check(id === args.compiled_source_id && id === manifest.source_sha256, 'actual compiled enclosing SOURCE');
    const log = text(logPath);
    check(['Early mixer model CUDA admission passed', 'Early mixer CUDA adapter admission passed', 'Structured hard timing CUDA admission passed', 'Frozen role guard checks passed', 'Fixed feature readout tests passed', 'Container SDK proof:'].every(marker => log.includes(marker)), 'all actual CUDA/readout/role/SDK gates');
    const compiler = compileCommands(log, qualityRequirements(QUALITY_BINARY), QUALITY_BINARY, document(path.join(target, 'source-scopes.json')));
    for (const source of ['structured_hard_timing_adapter.cpp', 'paired_timing_run.cpp', 'structured_hard_timing_main.cpp']) check(compiler.commands[`${CODE}/${source}`].includes(id), 'new producer scope');
    const plan = JSON.parse(command(QUALITY_BINARY, ['--plan'])); validatePlan(plan);
    const names = ['information-record.json', 'information.json', 'build-and-tests.log', 'source-manifest.json', 'source-inputs.sha256', 'source-fingerprint.txt', 'freeze.json'];
    copyMatrix(names.map(name => path.join(path.dirname(args.information), name)), names.map(name => path.join(target, 'information', name)));
    preserved(bound);
    // Append only to the admitted canonical log. Its final identity is bound below.
    appendAdmitted(logPath, `TEMPO-3 observed-only information admission passed: ${hash(args.information)}\n`);
    verifySources(repo, target); verifyInformation(repo, args.information); preserved([bound[0]]); writeJSON(path.join(target, 'recipe-plan.json'), plan);
    writeJSON(path.join(target, 'passed.json'), { protocol: PROTOCOL, status: 'passed', source_sha256: manifest.source_sha256, card_sha256: CARD_SHA,
      binary: bound[0], log: records([logPath])[0], compiler_evidence: compiler, information_record: records([args.information])[0], information_source_sha256: information.source_sha256, quality_generated: false, utc: utc() });
    return { status: 'passed', admission: target, source_sha256: manifest.source_sha256 };
  }
  if (action === 'freeze-run') {
    const approved = directory(args.admission); check(path.dirname(approved) === path.join(repo, RUN_ROOT, 'admission'), 'approved quality admission');
    const passed = document(path.join(approved, 'passed.json')); check(passed.status === 'passed' && passed.protocol === PROTOCOL && passed.card_sha256 === CARD_SHA, 'actual quality admission');
    const manifest = verifySources(repo, approved); preserved([passed.binary, passed.log]); verifyInformation(repo, passed.information_record.path); preserved([passed.information_record]);
    const names = readerBundle(args.reader_dir), readerRows = records(names.map(name => path.join(args.reader_dir, name)));
    emptyTarget(repo, target, RUN_ROOT, '', 'structured-hard-timing-'); check(capture(repo, target, manifest.files.map(row => row.path), config) === passed.source_sha256, 'frozen current SOURCE');
    const admissionPaths = walk(approved), before = records(admissionPaths);
    copyMatrix(admissionPaths, admissionPaths.map(name => path.join(target, 'admission', path.relative(approved, name))));
    copyMatrix(readerRows.map(row => row.path), names.map(name => path.join(target, 'independent-reader', name)));
    preserved(before); preserved(readerRows); verifySources(repo, approved); verifySources(repo, target);
    writeJSON(path.join(target, 'launch-plan.json'), { protocol: PROTOCOL, source_sha256: passed.source_sha256, card_sha256: CARD_SHA, binary: passed.binary,
      admission: approved, admission_sha256: hash(path.join(approved, 'passed.json')), information_record: passed.information_record, reader_original: args.reader_dir,
      reader_files: readerRows, historical_payload_roles: [], quality_generated_before_freeze: false, utc: utc() }); return { status: 'frozen', capsule: target, source_sha256: passed.source_sha256 };
  }
  check(action === 'finalize' && path.dirname(target) === path.join(repo, RUN_ROOT) && path.basename(target).startsWith('structured-hard-timing-'), 'completed capsule family');
  const complete = document(path.join(target, 'results/complete.json'));
  for (const [name, value] of Object.entries({ protocol: PROTOCOL, status: 'complete', cohorts: 5, encoder_trajectories: 10, encoder_updates_each: 512, sampled_rows: 40960 })) same(complete[name], value, 'full fixed measurement');
  const launch = document(path.join(target, 'launch-plan.json')); verifySources(repo, target); preserved([launch.binary]); verifyInformation(repo, launch.information_record.path); preserved([launch.information_record]);
  preserved(launch.reader_files); readerBundle(path.join(target, 'independent-reader'));
  writeJSON(path.join(target, 'source-preserved-after.json'), { status: 'passed', source_sha256: launch.source_sha256, binary_preserved: true, reader_preserved: true, information_preserved: true }); return inventory(target, PROTOCOL);
}
cli(import.meta.url, run);
