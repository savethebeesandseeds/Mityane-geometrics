// Metadata-only replacement; engineering fixtures run in the managed container.
import { fs, path, check, directory, same, admission, records, preserved, document, text, hash,
  copyMatrix, safeSource, emptyTarget, capture, verifySources, freezeScopes, compileCommands,
  command, manifestText, inventory, cli, options } from '../node/v1/saved_run_support.mjs';
import { hashAdmitted, writeJSON, writeNew } from '../../../../shared/artifacts/node/v1/files.mjs';

export const PROTOCOL = 'visible-difference-v1';
export const CODE = 'code/embeddings/evaluation/protocols/visible_difference_v1';
export const CARD = 'code/embeddings/evaluation/cards/visible_difference_v1.md';
export const CARD_SHA = '75e30dce63ddadaeeaf9daa6b25fd540ed02f9e0e11374e8787a7f846eaf78bc';
export const MASTERS = [75272, 76373, 77474, 78575, 79676];
export const RUN_ROOT = '.temp/embeddings/runs/rpb-visible-difference';
export const BINARY = '/opt/mityane-geometrics/embeddings/build/rpb-visible-difference/embedding_visible_difference';
const config = { protocol: PROTOCOL, code: CODE, card: CARD, cardSHA: CARD_SHA, scopeTarget: 'print-visible-difference-scope-inputs' };
const parentRelative = 'output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS';
const parentPins = {
  'artifact-integrity.json': '9bd9efe7c1e6c536ba57970c8db4f5549f3a06480a2407ab8d2d5ff38efc08aa',
  'parent-validation.json': 'cf79abdca511f22778622d2b08e62b74b592c10618d1090ea2bd53eefd84539c',
  'parent-summary.json': '1f5641212ce329a449073822a98d0db5c7e99a818532dfdca857bc78b9e17cd4'
};
function parentPaths(repo, args) {
  check(args.historical_parent_root, 'explicit --historical-parent-root required');
  const root = directory(args.historical_parent_root), parent = path.join(root, parentRelative);
  return { parent, metadata: {
    'artifact-integrity.json': path.join(parent, 'artifact-integrity.json'),
    'parent-validation.json': path.join(root, 'output/runs/rpb-structured-hard-timing/audit-tools/run-xrZMAS-v3-ordering/validation.json'),
    'parent-summary.json': path.join(repo, '.archive/embeddings/pre-cleanup-2026-10-18/doc/results/structured_hard_timing_comparison_v2.json')
  } };
}
export function parentRoles() {
  const out = [];
  for (const master of MASTERS) {
    const base = `results/seed-${master}-lag_sign/`;
    out.push(...['controlled-training.pt', 'controlled-validation.pt', 'controlled-validation-deleted.pt', 'initial-pair.pt'].map(name => base + name));
    out.push(...['', '.audit.pt', '.scaler.pt', '.training-raw.pt', '.structured-timing.pt'].map(suffix => `${base}early/point-0/checkpoint.pt${suffix}`));
    out.push(...['training', 'validation-intact', 'validation-deleted'].map(view => `${base}lag_sign/readouts/untrained_early/${view}-features.pt`));
    for (const method of ['native_late', 'native_early']) for (const rep of [2701, 2802, 2903])
      for (const view of ['validation-intact', 'validation-deleted']) out.push(`${base}lag_sign/readouts/${method}/rep-${rep}/${view}-predictions.pt`);
    out.push(...['training', 'validation'].map(view => `${base}early/point-512/${view}-reconstruction.pt`));
  }
  check(out.length === 130 && new Set(out).size === 130, 'closed130 retained roles'); return out.sort();
}
export function selectedRoles(value) {
  check(value.status === 'complete' && Array.isArray(value.files), 'complete parent inventory');
  const index = new Map();
  for (const row of value.files) { check(typeof row.path === 'string' && !index.has(row.path), 'unique parent role'); safeSource(`code/${row.path}`); index.set(row.path, row); }
  return parentRoles().map(name => {
    const row = index.get(name); check(row && Number.isSafeInteger(row.bytes) && row.bytes > 0 && /^[0-9a-f]{64}$/.test(row.sha256), 'typed selected parent role');
    same(Object.keys(row).sort(), ['bytes', 'path', 'sha256']); return row;
  });
}
function parentMetadata(repo, args) {
  const paths = parentPaths(repo, args), before = records(Object.values(paths.metadata));
  Object.keys(paths.metadata).forEach((name, i) => check(before[i].sha256 === parentPins[name], 'immutable parent metadata'));
  const values = Object.fromEntries(Object.entries(paths.metadata).map(([name, filename]) => [name, document(filename)]));
  check(values['parent-validation.json'].status === 'passed', 'passed parent audit'); preserved(before);
  return { ...paths, values, before };
}
export function progressText(audit, summary) {
  check(audit.per_master.length === 5 && summary.cohorts.length === 5, 'five parent records');
  const a = new Map(audit.per_master.map(row => [row.timing_master, row])), s = new Map(summary.cohorts.map(row => [row.timing_master, row]));
  same([...a.keys()].sort(), MASTERS); same([...s.keys()].sort(), MASTERS);
  const lines = ['master\tattempted\tcompleted\tsampled_rows\trequested\tactual\trestored\ttarget_cells'];
  for (const master of MASTERS) {
    const encoders = a.get(master).encoders.filter(row => row.role === 'early');
    const points = s.get(master).encoder_points.filter(row => row.placement === 1);
    check(encoders.length === 1 && points.length === 1, 'one matched parent encoder');
    const e = encoders[0], p = points[0].encoder_progress;
    same([e.attempted, e.completed, e.sampled_rows], [512, 512, 4096]); same([p.attempted, p.completed, p.sampled_rows], [512, 512, 4096]);
    check(e.context_counts.length === 3 && e.context_counts.every(v => Number.isSafeInteger(v) && v >= 0), 'parent cumulative context triple');
    check(p.losses.length === 512 && p.losses.every((row, i) => row.length === 5 && row[0] === i + 1 && row[1] === i + 1 && Number.isSafeInteger(row[2]) && row[2] > 0), 'full parent target-count trace');
    lines.push([master, 512, 512, 4096, ...e.context_counts, p.losses.map(row => row[2]).join(',')].join('\t'));
  }
  return lines.join('\n') + '\n';
}
export function expectedPlan() {
  return { protocol: PROTOCOL, dataset_id: 'TEMPO-3', recipe_id: 'structured-hard-timing-v1', task: 'lag_sign', timing_master_seeds: MASTERS,
    designed_complexity_level: 4, complexity_scale_max: 5, candidate_tag: 'RPB-v13', matched_control_tag: 'RPB-v10.alt-05', other_control_tag: 'RPB-v7.alt-05',
    parent_inventory_sha256: parentPins['artifact-integrity.json'], reused_payload_roles: 130, encoder_trajectories: 5, encoder_updates_each: 512, batch_size: 8,
    sampled_rows: 20480, attempt_limit: 1024, retained_points: 10, checkpoint_roles_per_point: 6, candidate_parameter_count: 228877, common_parameter_count: 225805,
    difference_parameter_count: 3072, train_pairs: 128, validation_pairs: 64, shape: [3, 32, 3], native_width: 32, temporal_difference_input: 1, head_repetitions: [2701, 2802, 2903],
    planned_pipelines: 15, planned_heads: 30, helper_outer_train_fits: 5, initial_parity_exports: 15, unique_quality_native_exports: 15, full_native_export_calls: 30,
    query_writer_calls: 10, necessary_query_forwards: 40, baseline_or_control_refits: 0, parent_model_forwards: 0, generator_calls: 0, PCA_fits: 0,
    extra_decoder_calibration_updates: 0, selection: false, promotion: false, testing_accessed: false, stress_accessed: false, human_card_sha256: CARD_SHA };
}
export function compileRecords(log, binary, scopes) {
  const old = 'code/embeddings/encoders/raw_patch_bottleneck_mae/shared/src/', base = path.dirname(binary);
  const od = path.join(base, 'code/encoders/raw_patch_bottleneck_mae'), nd = path.join(base, 'code/protocols/visible_difference_v1');
  return compileCommands(log, [[old + 'workflow.cpp', `${od}/workflow.o`, 'RPB_SOURCE_ID', 'core_writer'],
    [old + 'learning_curve_adapter.cpp', `${od}/learning_curve_adapter.o`, 'EVALUATION_SOURCE_ID', 'curve_training'],
    [old + 'early_mixer_adapter.cpp', `${od}/early_mixer_adapter.o`, 'EARLY_MIXER_ADAPTER_SOURCE_ID', 'early_adapter'],
    ['code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v13/src/visible_difference_adapter.cpp', `${nd}/visible_difference_adapter.o`, 'VISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID', 'visible_adapter'],
    [`${CODE}/visible_difference_main.cpp`, `${nd}/visible_difference_main.o`, 'VISIBLE_DIFFERENCE_SOURCE_ID', 'visible_adapter']], binary, scopes);
}
function readerBundle(root) {
  directory(root); const seal = document(path.join(root, 'reader-seal.json'));
  check(seal.status === 'passed', 'prospectively reviewed reader seal'); const names = seal.files.map(row => row.path);
  same(names, ['archive_codec.py', 'legacy_implementation.py', 'saved_cpu_math.py', 'source-fixtures.json', 'validate_visible_difference.py']);
  const actual = records(names.map(name => path.join(root, name)));
  actual.forEach((row, i) => check(row.bytes === seal.files[i].bytes && row.sha256 === seal.files[i].sha256, 'sealed reader bytes'));
  const fixtures = document(path.join(root, 'source-fixtures.json')); check(fixtures.status === 'passed' && (fixtures.archive_reads ?? 0) === 0, 'artificial-only reader fixtures');
  return [...names, 'reader-seal.json'];
}
function verifyRetained(target, manifest, parent) {
  same(manifest.files.map(row => row.path), parentRoles(), 'exact retained130');
  const names = manifest.files.map(row => row.path), actual = records([...names.map(name => path.join(parent, name)), ...names.map(name => path.join(target, 'retained-inputs', name))]);
  manifest.files.forEach((row, i) => { for (const got of [actual[i], actual[i + names.length]]) check(got.bytes === row.bytes && got.sha256 === row.sha256, 'parent originals/copies preserved'); });
}
export function run(action, args) {
  const declarations = {
    'self-test': [['output'], []],
    manifest: [['repo_root', 'historical_parent_root', 'output'], ['historical_parent_root', 'output']],
    'freeze-admission': [['repo_root', 'target', 'sources', 'card_sha256'], ['repo_root', 'target', 'sources', 'card_sha256']],
    admit: [['repo_root', 'target', 'binary', 'compiled_source_id'], ['repo_root', 'target', 'binary', 'compiled_source_id']],
    'freeze-run': [['repo_root', 'target', 'admission', 'reader_dir', 'historical_parent_root'], ['repo_root', 'target', 'admission', 'reader_dir', 'historical_parent_root']],
    finalize: [['repo_root', 'target'], ['repo_root', 'target']]
  };
  check(Object.hasOwn(declarations, action), 'closed visible metadata action'); options(args, ...declarations[action]);
  if (action === 'self-test') {
    same(selectedRoles({ status: 'complete', files: parentRoles().map(name => ({ path: name, bytes: 7, sha256: 'a'.repeat(64) })) }).length, 130);
    const value = { status: 'passed', source_fixture_only: true, archive_reads: 0, models_or_fits: 0, quality_generated: false };
    if (args.output) writeJSON(args.output, value); return value;
  }
  const repo = directory(args.repo_root ?? '/mityane');
  if (action === 'manifest') {
    const metadata = parentMetadata(repo, args), files = selectedRoles(metadata.values['artifact-integrity.json']);
    writeJSON(args.output, { protocol: PROTOCOL, parent_inventory_sha256: parentPins['artifact-integrity.json'], files }); return { status: 'passed', payload_reads: 0, roles: files.length, manifest_sha256: hash(args.output) };
  }
  if (action === 'freeze-admission') {
    check(args.card_sha256 === CARD_SHA, 'fixed card CLI pin'); emptyTarget(repo, args.target, RUN_ROOT, 'admission', 'admission-');
    const id = freezeScopes(repo, args.target, args.sources, config, ['core_writer', 'curve_training', 'early_adapter', 'visible_adapter'], 'visible_adapter');
    return { status: 'frozen', target: args.target, source_sha256: id };
  }
  const target = directory(args.target);
  if (action === 'admit') {
    check(path.dirname(target) === path.join(repo, RUN_ROOT, 'admission') && args.binary === BINARY, 'fixed current admission/binary');
    const source = verifySources(repo, target), logPath = path.join(target, 'build-and-tests.log');
    const bound = records([BINARY, logPath]), log = text(logPath);
    check(['Visible difference retained input loader fixtures passed', 'Visible difference model CUDA admission passed', 'Visible difference adapter CUDA admission passed',
      'Fixed feature readout tests passed', 'Frozen role guard checks passed', 'Container SDK proof:'].every(marker => log.includes(marker)), 'actual input/CUDA/shared/SDK gates');
    const compiler = compileRecords(log, BINARY, document(path.join(target, 'source-scopes.json'))), id = command(BINARY, ['--source-id']);
    check(id === args.compiled_source_id && id === source.source_sha256, 'actual compiled SOURCE identity');
    const plan = JSON.parse(command(BINARY, ['--plan'])); same(plan, { ...expectedPlan(), source_fingerprint: id });
    verifySources(repo, target); preserved(bound); writeJSON(path.join(target, 'recipe-plan.json'), plan);
    writeJSON(path.join(target, 'passed.json'), { protocol: PROTOCOL, status: 'passed', source_sha256: id, card_sha256: CARD_SHA,
      binary: bound[0], log: bound[1], compiler_evidence: compiler, quality_generated: false }); return { status: 'passed', admission: target, source_sha256: id };
  }
  if (action === 'freeze-run') {
    const approved = directory(args.admission); check(path.dirname(approved) === path.join(repo, RUN_ROOT, 'admission'), 'approved admission family');
    const passed = document(path.join(approved, 'passed.json')); check(passed.status === 'passed' && passed.protocol === PROTOCOL && passed.card_sha256 === CARD_SHA, 'actual CUDA admission');
    const source = verifySources(repo, approved); preserved([passed.binary, passed.log]);
    const readerNames = readerBundle(args.reader_dir), readerRows = records(readerNames.map(name => path.join(args.reader_dir, name)));
    const metadata = parentMetadata(repo, args), rows = selectedRoles(metadata.values['artifact-integrity.json']);
    const manifest = { protocol: PROTOCOL, parent_inventory_sha256: parentPins['artifact-integrity.json'], files: rows };
    same(document(path.join(repo, CODE, 'retained-roles.json')), manifest);
    const admitted = admission(rows.map(row => path.join(metadata.parent, row.path))); check(admitted.every((row, i) => row.bytes === rows[i].bytes), 'all130 payload sizes before hashing');
    admitted.forEach((row, i) => check(hashAdmitted(row) === rows[i].sha256, 'selected original SHA pin'));
    emptyTarget(repo, target, RUN_ROOT, '', 'visible-difference-'); check(capture(repo, target, source.files.map(row => row.path), config) === passed.source_sha256, 'copied current SOURCE');
    const admissionPaths = (awaitlessWalk(approved)), admissionRows = records(admissionPaths);
    copyMatrix(admissionPaths, admissionPaths.map(name => path.join(target, 'admission', path.relative(approved, name))));
    copyMatrix(readerRows.map(row => row.path), readerNames.map(name => path.join(target, 'independent-reader', name)));
    copyMatrix(Object.values(metadata.metadata), Object.keys(metadata.metadata).map(name => path.join(target, 'parent-metadata', name)));
    copyMatrix(admitted.map(row => row.path), rows.map(row => path.join(target, 'retained-inputs', row.path)));
    writeJSON(path.join(target, 'retained-inputs-manifest.json'), manifest); writeNew(path.join(target, 'retained-inputs.sha256'), manifestText(rows));
    writeNew(path.join(target, 'parent-progress.tsv'), progressText(metadata.values['parent-validation.json'], metadata.values['parent-summary.json']));
    verifyRetained(target, manifest, metadata.parent); verifySources(repo, target); preserved(readerRows); preserved(admissionRows); preserved(metadata.before);
    writeJSON(path.join(target, 'launch-plan.json'), { protocol: PROTOCOL, source_sha256: passed.source_sha256, card_sha256: CARD_SHA,
      binary: passed.binary, log: passed.log, admission: approved, admission_sha256: hash(path.join(approved, 'passed.json')), reader_files: readerRows,
      parent_metadata: metadata.before, historical_parent_root: args.historical_parent_root, retained_payload_roles: 130,
      parent_progress_sha256: hash(path.join(target, 'parent-progress.tsv')), quality_generated_before_freeze: false }); return { status: 'frozen', capsule: target, source_sha256: passed.source_sha256 };
  }
  check(action === 'finalize' && path.dirname(target) === path.join(repo, RUN_ROOT) && path.basename(target).startsWith('visible-difference-'), 'completed capsule family');
  const complete = document(path.join(target, 'results/complete.json'));
  const fixed = { protocol: PROTOCOL, status: 'complete', cohorts: 5, encoder_trajectories: 5, encoder_updates_each: 512, skipped_attempts: 0, sampled_rows: 20480,
    retained_points: 10, checkpoint_roles_per_point: 6, retained_payload_roles: 130, planned_pipelines: 15, planned_heads: 30, quality_native_exports: 15, initial_parity_exports: 15,
    full_native_export_calls: 30, query_writer_calls: 10, necessary_query_forwards: 40, baseline_or_control_refits: 0, parent_model_forwards: 0, generator_calls: 0, PCA_fits: 0,
    extra_decoder_calibration_updates: 0, CPU_encoder_forward: false, CPU_encoder_training: false, selection: false, promotion: false, testing_accessed: false, stress_accessed: false };
  for (const [name, value] of Object.entries(fixed)) same(complete[name], value, `fixed candidate budget: ${name}`);
  check(complete.head_pipelines <= 15 && complete.individual_heads === 2 * complete.head_pipelines && complete.helper_outer_train_fits <= 5, 'only declared fits');
  const launch = document(path.join(target, 'launch-plan.json')); verifySources(repo, target); preserved([launch.binary, launch.log]); preserved(launch.reader_files); readerBundle(path.join(target, 'independent-reader'));
  const metadata = parentMetadata(repo, { historical_parent_root: launch.historical_parent_root }); same(metadata.before, launch.parent_metadata);
  verifyRetained(target, document(path.join(target, 'retained-inputs-manifest.json')), metadata.parent);
  for (const [name, pin] of Object.entries(parentPins)) check(hash(path.join(target, 'parent-metadata', name)) === pin, 'captured metadata exact');
  check(hash(path.join(target, 'parent-progress.tsv')) === launch.parent_progress_sha256, 'parent progress preserved');
  writeJSON(path.join(target, 'source-preserved-after.json'), { status: 'passed', source_sha256: launch.source_sha256, parent_inputs_preserved: true, reader_preserved: true, binary_preserved: true }); return inventory(target, PROTOCOL);
}
// Kept separate to make directory traversal an explicit metadata operation.
import { walk as awaitlessWalk } from '../node/v1/saved_run_support.mjs';
cli(import.meta.url, run);
