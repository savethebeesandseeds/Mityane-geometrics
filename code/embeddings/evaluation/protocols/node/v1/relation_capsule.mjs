// Shared receipt mechanics for the two explicitly configured relation families.
// This module does not fit, load tensors, allocate a cohort, or run a command.
import fs from 'node:fs';
import path from 'node:path';
import {
  check, equal, join, lines, sha, exists, text, document, direct, regular, write, copy,
  finalize, finite, assertNewRoot, implementationReceipt, captureSources,
  verifyCapturedSources, verifyConsumed, checksumLog, retainedAdmissions, CAPSULE_CLI_SOURCES,
} from './capsule_support.mjs';
import { admitFiles } from '../../../../../shared/artifacts/node/v1/files.mjs';

const ORIGINAL_MANIFEST = '62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83';
const ORIGINAL_SOURCE = '8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1';
const ORIGINAL_INPUTS = '6121773c0b97e5490f191a9f8ff013e90f27a9a557376ac8256bdbb2d6af6091';
const BRANCH_MANIFEST = 'cefa413c560bdf2ab1eee5be2e3f34feca6e2f723a99627f9eeae683c4c11d93';
const PREVIOUS_MANIFEST = '6b138475bbeac3bde35cc45f331f21f5e10cff64b5971fb883f60121c20562bf';

export function createRelationCapsule(options, family) {
  check(options && typeof options === 'object', 'explicit migration path/source binding required');
  const { repositoryRoot, buildDirectory, originalCapsule, branchCapsule, previousCapsule } = options;
  check(typeof repositoryRoot === 'string' && typeof buildDirectory === 'string' &&
    typeof originalCapsule === 'string' && typeof branchCapsule === 'string', 'explicit current and historical paths');
  const runRoot = join(repositoryRoot, '.temp/embeddings/runs', family.runName);
  check(buildDirectory.startsWith('/opt/mityane-geometrics/') && path.normalize(buildDirectory) === buildDirectory,
    'new isolated build session stays under /opt/mityane-geometrics');
  if (family.content) check(typeof previousCapsule === 'string', 'explicit preserved v21 path mapping');
  const historical = [
    { path: originalCapsule, recorded: '/embedding/output/runs/rpb-v20-capacity/admission-0gN4uD',
      manifest: ORIGINAL_MANIFEST, protocol: 'configurable-capacity-v1', frozenName: 'original-artifacts.sha256' },
    { path: branchCapsule, recorded: '/embedding/output/runs/rpb-v20-branches/diagnostic-YA3Nws',
      manifest: BRANCH_MANIFEST, protocol: 'configurable-branch-v1', frozenName: 'branch-artifacts.sha256' },
  ];
  if (family.content) historical.push({ path: previousCapsule,
    recorded: '/embedding/output/runs/rpb-v21-learned-relation/admission-0PGNmn',
    manifest: PREVIOUS_MANIFEST, protocol: 'configurable-learned-relation-v1', frozenName: 'v21-artifacts.sha256' });
  const mapping = historical.map(role => ({ recorded_path: role.recorded, current_path: role.path,
    artifacts_manifest_sha256: role.manifest }));
  const policy = { ...options, binaries: family.binaries,
    sourceIdBinaries: family.binaries.slice(0, 2), requiredSources: [
      ...CAPSULE_CLI_SOURCES,
      `code/embeddings/evaluation/protocols/${family.directory}/capsule.mjs`,
      `code/embeddings/evaluation/protocols/${family.directory}/gate.mjs`,
      'code/embeddings/evaluation/protocols/configurable_learned_relation_v1/gate.mjs',
      'code/embeddings/evaluation/protocols/node/v1/relation_capsule.mjs',
      'code/embeddings/evaluation/protocols/node/v1/capsule_support.mjs',
      'code/shared/artifacts/node/v1/files.mjs',
      family.card, family.config,
    ] };
  const capsulePath = root => assertNewRoot(root, runRoot);
  const gatePath = directory => join(repositoryRoot, 'code/embeddings/evaluation/protocols', directory, 'gate.py');
  function historicalGateSHA() {
    const reference = sha(gatePath('configurable_learned_relation_v1'));
    if (family.content) check(reference === sha(gatePath('configurable_content_relation_v1')),
      'v22 retains the exact unchanged historical 54-condition v21 gate');
    return reference;
  }
  function verifyHistorical() {
    const files = historical.flatMap(role => [join(role.path, 'artifacts.sha256'), join(role.path, 'complete.json')]);
    files.push(join(originalCapsule, 'freeze/receipt.json'), join(originalCapsule, 'freeze/inputs.sha256'));
    const rows = new Map(admitFiles(files, { singleLink: true }).map(row => [row.path, row]));
    for (const role of historical) {
      direct(role.path);
      check(sha(rows.get(join(role.path, 'artifacts.sha256'))) === role.manifest, 'pinned historical manifest');
      const done = document(rows.get(join(role.path, 'complete.json')));
      check(done.status === 'complete' && done.protocol === role.protocol &&
        done.artifacts_manifest_sha256 === role.manifest, 'historical completion receipt');
    }
    check(document(rows.get(join(originalCapsule, 'freeze/receipt.json'))).source_id === ORIGINAL_SOURCE &&
      sha(rows.get(join(originalCapsule, 'freeze/inputs.sha256'))) === ORIGINAL_INPUTS,
    'original source and exact nine quality role inputs');
    return rows;
  }
  function freeze(root) {
    capsulePath(root); direct(buildDirectory); const historicalRows = verifyHistorical();
    check(typeof options.gitHead === 'string' && /^[a-f0-9]{40,64}$/.test(options.gitHead), 'explicit captured Git revision');
    const unchangedGate = family.content ? historicalGateSHA() : undefined;
    const { out, sourceId, sourceFiles, hashes, sourceRows } = captureSources(root, policy);
    check(sourceId !== ORIGINAL_SOURCE, 'new implementation cannot reuse a historical source identity');
    for (const [name, source] of [['card.md', family.card], ['config.conf', family.config]]) {
      copy(sourceRows.get(source), join(out, name));
      check(sha(join(out, name)) === hashes.get(source), 'frozen card/settings exact source bytes');
    }
    copy(historicalRows.get(join(originalCapsule, 'freeze/inputs.sha256')), join(out, 'inputs.sha256'));
    for (const role of historical) copy(historicalRows.get(join(role.path, 'artifacts.sha256')), join(out, role.frozenName));
    const receipt = { protocol: family.protocol, source_fingerprint: sourceId,
      card_sha256: sha(join(out, 'card.md')), config_sha256: sha(join(out, 'config.conf')),
      input_manifest_sha256: ORIGINAL_INPUTS,
      original_capsule: historical[0].recorded, original_source_fingerprint: ORIGINAL_SOURCE,
      original_artifacts_manifest_sha256: ORIGINAL_MANIFEST,
      branch_capsule: historical[1].recorded, branch_artifacts_manifest_sha256: BRANCH_MANIFEST,
      ...(family.content ? { preserved_v21_capsule: historical[2].recorded,
        preserved_v21_artifacts_manifest_sha256: PREVIOUS_MANIFEST, unchanged_gate_sha256: unchangedGate } : {}),
      source_files: sourceFiles, build_directory: buildDirectory, git_head: options.gitHead,
      migration_implementation: implementationReceipt(sourceId, mapping) };
    write(join(out, 'receipt.json'), receipt);
    return receipt;
  }
  function verifyFrozen(root) {
    capsulePath(root);
    const receipt = document(join(root, 'freeze/receipt.json'));
    check(receipt.protocol === family.protocol &&
      sha(join(root, 'freeze/sources.sha256')) === receipt.source_fingerprint &&
      sha(join(root, 'freeze/card.md')) === receipt.card_sha256 &&
      sha(join(root, 'freeze/config.conf')) === receipt.config_sha256 &&
      sha(join(root, 'freeze/inputs.sha256')) === receipt.input_manifest_sha256 &&
      receipt.input_manifest_sha256 === ORIGINAL_INPUTS, 'frozen source/card/settings/input binding');
    check(receipt.build_directory === buildDirectory &&
      equal(receipt.migration_implementation, implementationReceipt(receipt.source_fingerprint, mapping)),
    'frozen new source identity, encoding, path mapping and named build session');
    if (family.content) {
      check(sha(join(root, 'freeze/v21-artifacts.sha256')) === receipt.preserved_v21_artifacts_manifest_sha256 &&
        receipt.preserved_v21_artifacts_manifest_sha256 === PREVIOUS_MANIFEST, 'preserved v21 inventory remains exact');
      check(historicalGateSHA() === receipt.unchanged_gate_sha256, 'historical prospective gate remains unchanged');
    }
    verifyCapturedSources(root, policy, receipt);
    return receipt;
  }
  function verifyEngineering(root) {
    const receipt = verifyFrozen(root);
    check(equal(document(join(root, 'admission.json')), { protocol: family.protocol, status: 'admitted',
      quality_encoder_runs: 0, source_fingerprint: receipt.source_fingerprint, card_sha256: receipt.card_sha256 }),
    'actual engineering admission');
    check(equal(document(join(root, 'runner-engineering/complete.json')), { status: 'passed',
      protocol: family.protocol, master: 941000, profiles: ['A'], updates_each: 2, CPU_encoder_calls: 0 }),
    'actual two-update CUDA admission');
    check(equal(document(join(root, 'readout-engineering/complete.json')), { protocol: `${family.protocol}-engineering`,
      status: 'passed', fitted_pairs: 6, encoder_calls: 0 }), 'artificial saved-head engineering scope');
    const progress = document(join(root, 'runner-engineering/A/progress.json'));
    check(progress.protocol === family.protocol && progress.master === 941000 &&
      progress.source_fingerprint === receipt.source_fingerprint && progress.card_sha === receipt.card_sha256 &&
      progress.config_sha === receipt.config_sha256 && progress.attempted === 2 && progress.completed === 2 &&
      progress.skipped_attempts === 0 && progress.CPU_encoder_calls === 0 && progress.training_device === 'cuda:0' &&
      progress.parameter_count === progress.cuda_parameter_count && progress.weights_changed &&
      progress.relation_gradient_updates > 0 && finite(progress.relation_weight_max_absolute_change) &&
      progress.relation_weight_max_absolute_change > 0,
    'actual two-update CUDA trace trains the projection under frozen source/card/settings');
    check(Array.isArray(progress.losses) && progress.losses.length === 2 &&
      progress.losses.filter(row => row.relation_gradient_norm > 0).length === progress.relation_gradient_updates &&
      progress.losses.every(row => finite(row.relation_gradient_norm) && row.relation_gradient_norm >= 0 &&
        finite(row.total_loss) && row.gradient_norm > 0), 'actual finite artificial CUDA gradient and loss witnesses');
    return receipt;
  }
  function admissionFields(root, receipt) {
    return { protocol: family.protocol, status: 'admitted', source_fingerprint: receipt.source_fingerprint,
      card_sha256: receipt.card_sha256, config_sha256: receipt.config_sha256, input_manifest_sha256: ORIGINAL_INPUTS,
      runner_complete_sha256: sha(join(root, 'runner-engineering/complete.json')),
      runner_progress_sha256: sha(join(root, 'runner-engineering/A/progress.json')),
      runner_checkpoint_sha256: sha(join(root, 'runner-engineering/A/checkpoint.pt')),
      readout_complete_sha256: sha(join(root, 'readout-engineering/complete.json')),
      binaries_manifest_sha256: sha(join(root, 'freeze/binaries.sha256')),
      producer_binary_sha256: sha(join(buildDirectory, family.binaries[0])) };
  }
  function bindQuality(root) {
    const receipt = verifyEngineering(root);
    const admission = admissionFields(root, receipt);
    write(join(root, 'quality-admission.json'), admission);
    return admission;
  }
  function verifyAdmission(root) {
    const receipt = verifyEngineering(root);
    check(equal(document(join(root, 'quality-admission.json')), admissionFields(root, receipt)),
      'immutable quality admission binds actual CUDA progress/checkpoint and exact binary');
    return receipt;
  }
  function verifyStage(root, stage) {
    check(stage === 'screen' || stage === 'third', 'closed relation stage');
    const receipt = verifyAdmission(root), masters = stage === 'screen' ? [941101, 941102] : [941103];
    const readout = join(root, stage, 'readouts'), published = join(root, stage, 'published');
    const done = document(join(readout, 'complete.json')), report = document(join(readout, 'report.json'));
    const summary = document(join(published, 'summary.json')), gate = document(join(published, 'gate.json'));
    check(done.protocol === family.protocol && done.status === 'complete' &&
      done.saved_arithmetic_fits_replayed === 6 * masters.length && done.head_fitted_pairs === 6 * masters.length &&
      done.report_sha256 === sha(join(readout, 'report.json')) &&
      done.arithmetic_audit_sha256 === sha(join(readout, 'arithmetic-audit.json')), 'completed readout artifact/counter binding');
    check(equal(report.masters, masters) && equal(summary.masters, masters) &&
      report.source_fingerprint === receipt.source_fingerprint && summary.source_fingerprint === receipt.source_fingerprint &&
      report.card_sha === receipt.card_sha256 && summary.card_sha256 === receipt.card_sha256 &&
      report.head_fitted_pairs === 6 * masters.length && summary.new_head_fit_pairs === 6 * masters.length &&
      summary.new_classifiers === 12 * masters.length && summary.encoder_trajectories === masters.length &&
      summary.encoder_updates === 512 * masters.length && summary.CPU_encoder_calls === 0 &&
      summary.publisher_encoder_calls === 0 && summary.publisher_head_fits === 0,
    'independently audited stage scope/source/card');
    check(equal(gate, summary.gate) && equal(gate.masters, masters) &&
      Array.isArray(gate.conditions) && gate.conditions.every(condition => typeof condition.passed === 'boolean') &&
      gate.passed === gate.conditions.every(condition => condition.passed),
    'prospective gate is exactly the audited numerical decision');
    verifyConsumed(join(published, 'consumed-files.sha256'));
    for (const master of masters) {
      const candidate = join(root, 'candidates', `master-${master}`), complete = document(join(candidate, 'complete.json'));
      check(complete.protocol === family.protocol && complete.status === 'complete' && complete.master === master &&
        complete.updates === 512 && complete.source_fingerprint === receipt.source_fingerprint &&
        complete.card_sha === receipt.card_sha256 && complete.config_sha === receipt.config_sha256 &&
        complete.input_manifest_sha === ORIGINAL_INPUTS && complete.report_sha256 === sha(join(candidate, 'report.json')),
      'completed candidate trajectory receipt');
      for (const point of [0, 512]) for (const name of ['checkpoint.pt', 'features.pt', 'optimizer-witness.pt',
        'scaler-witness.pt', 'reconstruction-training.pt', 'reconstruction-validation.pt', 'query-report.json', 'progress.json'])
        regular(join(candidate, `point-${point}`, name));
    }
    return { receipt, summary };
  }
  function admitThird(root) {
    const { receipt, summary } = verifyStage(root, 'screen');
    check(summary.gate.passed === true && !exists(join(root, 'candidates/master-941103')) && !exists(join(root, 'third')),
      'joint screen pass before allocating third cohort');
    const evidence = { protocol: family.protocol, status: 'admitted', master: 941103,
      source_fingerprint: receipt.source_fingerprint, card_sha256: receipt.card_sha256,
      screen_summary_sha256: sha(join(root, 'screen/published/summary.json')),
      screen_gate_sha256: sha(join(root, 'screen/published/gate.json')),
      consumed_manifest_sha256: sha(join(root, 'screen/published/consumed-files.sha256')) };
    write(join(root, 'third-admission.json'), evidence);
    return evidence;
  }
  function finish(root) {
    check(!exists(join(root, 'artifacts.sha256')) && !exists(join(root, 'complete.json')), 'new completion receipt');
    const retained = retainedAdmissions(root);
    const { receipt, summary: screen } = verifyStage(root, 'screen'), masters = [941101, 941102];
    let disposition;
    if (screen.gate.passed) {
      const admission = document(join(root, 'third-admission.json'));
      check(admission.source_fingerprint === receipt.source_fingerprint && admission.card_sha256 === receipt.card_sha256 &&
        admission.screen_summary_sha256 === sha(join(root, 'screen/published/summary.json')) &&
        admission.screen_gate_sha256 === sha(join(root, 'screen/published/gate.json')) &&
        admission.consumed_manifest_sha256 === sha(join(root, 'screen/published/consumed-files.sha256')),
      'third allocation preserves exact audited screen');
      const { summary: third } = verifyStage(root, 'third'); masters.push(941103);
      disposition = third.gate.passed ? 'third-development-pass' : 'third-development-fail';
    } else {
      check(!exists(join(root, 'third-admission.json')) && !exists(join(root, 'third')) &&
        !exists(join(root, 'candidates/master-941103')), 'failed screen does not allocate third');
      disposition = 'screen-fail-stop';
    }
    const candidateNames = fs.readdirSync(join(root, 'candidates')).sort();
    check(equal(candidateNames, masters.map(master => `master-${master}`)), 'exact allocated trajectory inventory');
    verifyHistorical();
    checksumLog(root, 'original-after-quality.log', join(originalCapsule, 'artifacts.sha256'), 815);
    checksumLog(root, 'branch-after-quality.log', join(branchCapsule, 'artifacts.sha256'),
      lines(text(join(branchCapsule, 'artifacts.sha256'))).length);
    if (family.content) checksumLog(root, 'v21-after-quality.log', join(previousCapsule, 'artifacts.sha256'),
      lines(text(join(previousCapsule, 'artifacts.sha256'))).length);
    checksumLog(root, 'source-after-quality.log', join(root, 'freeze/sources.sha256'), receipt.source_files);
    checksumLog(root, 'binary-after-quality.log', join(root, 'freeze/binaries.sha256'), family.binaries.length);
    const completion = { protocol: family.protocol, status: 'complete', disposition, masters,
      encoder_trajectories: masters.length, completed_updates_per_run: 512,
      quality_head_fit_pairs: 6 * masters.length, quality_classifiers: 12 * masters.length,
      engineering_head_fit_pairs: 6, CPU_encoder_calls: 0, control_head_fits: 0,
      source_fingerprint: receipt.source_fingerprint, card_sha256: receipt.card_sha256,
      original_artifacts_manifest_sha256: ORIGINAL_MANIFEST, branch_artifacts_manifest_sha256: BRANCH_MANIFEST,
      ...(family.content ? { preserved_v21_artifacts_manifest_sha256: PREVIOUS_MANIFEST } : {}),
      migration_implementation: receipt.migration_implementation };
    finalize(root, completion, { admittedFiles: retained });
    return completion;
  }
  return Object.freeze({ freeze, verifyHistorical, verifyFrozen, verifyEngineering, admissionFields,
    bindQuality, verifyAdmission, verifyStage, admitThird, finish });
}
