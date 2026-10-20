// Metadata CLI; tensor admission remains a separately verified scientific boundary.
import { main as capsuleMain } from '../node/v1/capsule_cli.mjs';
import { isMain } from '../node/v1/metadata.mjs';
import {
  check, equal, join, sha, exists, document, direct, regular, write, copy, finalize,
  finite, assertNewRoot, implementationReceipt, captureSources, retainedAdmissions, CAPSULE_CLI_SOURCES,
} from '../node/v1/capsule_support.mjs';
import { admitFiles, hashAdmitted, readAdmitted, parseJSON } from '../../../../shared/artifacts/node/v1/files.mjs';

export const MASTERS = Object.freeze([941101, 941102, 941103]);
export const PROTOCOL = 'configurable-capacity-v1';
export const ORIGINAL_CODEC = '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d';
export const BINARIES = Object.freeze(['embedding_configurable_capacity', 'embedding_configurable_capacity_data',
  'embedding_configurable_capacity_readout', 'code/protocols/configurable_capacity_v1/generator_test']);

export function createCapsule(options) {
  check(options && typeof options.repositoryRoot === 'string' && typeof options.buildDirectory === 'string',
    'explicit migration path/source binding required');
  const { repositoryRoot, buildDirectory } = options;
  check(buildDirectory.startsWith('/opt/mityane-geometrics/'), 'new isolated managed build session');
  const runRoot = join(repositoryRoot, '.temp/embeddings/runs/rpb-v20-capacity');
  const capsulePath = root => assertNewRoot(root, runRoot);
  function verifyAdmission(root) {
    capsulePath(root);
    check(equal(document(join(root, 'admission.json')), { protocol: PROTOCOL, status: 'admitted', quality_encoder_runs: 0 }),
      'completed engineering admission required before quality generation');
    check(equal(document(join(root, 'runner-engineering/complete.json')), { status: 'passed', protocol: PROTOCOL,
      master: 941000, profiles: ['A', 'B', 'C'], updates_each: 2, CPU_encoder_calls: 0 }),
    'actual two-update CUDA admission required for all three profiles');
    const receipt = document(join(root, 'freeze/receipt.json'));
    check(receipt.protocol === PROTOCOL && receipt.source_id === sha(join(root, 'freeze/sources.sha256')) &&
      receipt.card_sha256 === sha(join(root, 'freeze/card.md')) && receipt.config_sha256 === sha(join(root, 'freeze/config.conf')),
    'admission source/card/settings receipt differs');
    check(equal(receipt.migration_implementation, implementationReceipt(receipt.source_id, [])),
      'new source identity and receipt encoding');
    return receipt;
  }
  function freeze(root) {
    capsulePath(root); direct(buildDirectory);
    check(typeof options.gitHead === 'string' && /^[a-f0-9]{40,64}$/.test(options.gitHead), 'explicit captured Git revision');
    const copies = [['card.md', 'code/embeddings/evaluation/cards/configurable_capacity_v1.md'],
      ['config.conf', 'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v20/config/configurable_spectral.conf']];
    const { out, sourceId, sourceFiles, hashes, sourceRows } = captureSources(root, { ...options, binaries: BINARIES,
      sourceIdBinaries: BINARIES.slice(0, 1), requiredSources: [
        ...CAPSULE_CLI_SOURCES,
        'code/embeddings/evaluation/protocols/configurable_capacity_v1/capsule.mjs',
        'code/embeddings/evaluation/protocols/node/v1/capsule_support.mjs',
        'code/shared/artifacts/node/v1/files.mjs',
        ...copies.map(([, source]) => source),
      ] });
    for (const [name, source] of copies) {
      copy(sourceRows.get(source), join(out, name));
      check(sha(join(out, name)) === hashes.get(source), 'frozen card/settings exact captured source bytes');
    }
    const receipt = { protocol: PROTOCOL, source_id: sourceId, card_sha256: sha(join(out, 'card.md')),
      config_sha256: sha(join(out, 'config.conf')), build_directory: buildDirectory, source_files: sourceFiles,
      git_head: options.gitHead, migration_implementation: implementationReceipt(sourceId, []) };
    write(join(out, 'receipt.json'), receipt);
    return receipt;
  }
  function verifiedReader() {
    const reader = options.reader;
    check(reader && typeof reader.loadAdmitted === 'function' && reader.kind === 'independent-cpp-restricted-archive-v1' &&
      /^[a-f0-9]{64}$/.test(reader.implementation_sha256 ?? '') && typeof reader.equivalence_receipt === 'string' &&
      /^[a-f0-9]{64}$/.test(reader.equivalence_receipt_sha256 ?? ''),
    'capacity bind requires a hash-bound independently verified C++ reader; no fallback is bound');
    const certificateRow = regular(reader.equivalence_receipt);
    check(hashAdmitted(certificateRow) === reader.equivalence_receipt_sha256, 'reader equivalence receipt bytes');
    const equivalence = document(certificateRow);
    check(equivalence.status === 'equivalent' && equivalence.protocol === PROTOCOL &&
      equivalence.artifact_scope === 'capacity-information' && equivalence.original_codec_sha256 === ORIGINAL_CODEC &&
      equivalence.implementation_sha256 === reader.implementation_sha256 && equivalence.boolean_verdicts_exact === true,
    'reader equivalence is certified for the exact information roles and discrete decisions');
    return { reader, certificateRow };
  }
  function bind(root) {
    capsulePath(root);
    // No observation/witness payload is read before the reader binding and the
    // complete 18-file role matrix have both passed admission.
    const { reader, certificateRow } = verifiedReader(), data = join(root, 'quality-data');
    const informationRow = regular(join(data, 'information-report.json'));
    const information = parseJSON(readAdmitted(informationRow), { integers: 'number', duplicates: 'last', allowBOM: false });
    check(information.passed === true && Array.isArray(information.masters) && information.masters.length === 3,
      'all three information cohorts must pass');
    check(equal(information.masters.map(master => master.master), MASTERS), 'information master order differs');
    for (const master of information.masters) {
      check(master.history === 128 && master.passed === true && Array.isArray(master.views) && master.views.length === 3,
        'information history/population differs');
      check(equal(master.views.map(view => view.view), ['training', 'validation', 'validation-deleted']),
        'exactly TRAIN, intact and deleted information views required');
      for (const view of master.views) {
        const expected = view.view === 'training' ? 512 : 256;
        check(view.total === expected && view.supported === expected && view.correct === expected && view.passed === true,
          'every declared information row must be supported and correct');
      }
    }
    const roles = MASTERS.flatMap(master => ['training', 'validation', 'validation-deleted'].map(view =>
      `master-${master}/${view}.pt`));
    check(roles.length === 9, 'nine observation roles required');
    const files = roles.flatMap(role => [join(data, role), join(data, role.replace('.pt', '-information.pt'))]);
    const admitted = admitFiles(files, { root: data, singleLink: true });
    const evidence = [join(data, 'information-report.json')];
    for (let index = 0; index < roles.length; index += 1) {
      const role = roles[index], rows = role.endsWith('/training.pt') ? 512 : 256;
      const original = reader.loadAdmitted(admitted[2 * index], { role: 'capacity-observations', rows });
      const decisions = reader.loadAdmitted(admitted[2 * index + 1], { role: 'capacity-information', rows });
      check(decisions.valid.dtype === 'BoolStorage' && equal(decisions.valid.shape, [rows]), 'information validity schema differs');
      check(Array.from(decisions.valid.values).every(Boolean), 'unsupported information decision');
      for (const key of ['labels_scoring_only', 'source_ids']) {
        check(original[key].dtype === 'LongStorage' && decisions[key].dtype === 'LongStorage' &&
          equal(original[key].shape, [rows]) && equal(decisions[key].shape, [rows]), `information label/source schema differs: ${key}`);
        check(equal(Array.from(original[key].values), Array.from(decisions[key].values)), `information role association differs: ${key}`);
      }
      check(equal(Array.from(decisions.predictions.values), Array.from(original.labels_scoring_only.values)),
        'information prediction differs from scoring-only label');
      check(Array.from(decisions.frequency_pairs_evaluated.values).every(value => value === 119 || value === 119n),
        'closed information frequency search differs');
      const diagnostics = Array.from(decisions.diagnostics.values);
      check(equal(decisions.diagnostics.shape, [rows, 4]) && diagnostics.every(finite), 'finite information diagnostics required');
      for (let row = 0; row < rows; row += 1) {
        const [slow, fast, residual, margin] = diagnostics.slice(4 * row, 4 * row + 4);
        check(Array.from({ length: 17 }, (_, n) => n + 32).some(n => Math.abs(slow - 1 / n) < 1e-15) &&
          Array.from({ length: 7 }, (_, n) => n + 8).some(n => Math.abs(fast - 1 / n) < 1e-15) &&
          residual >= 0 && residual <= .0025 && Math.abs(margin) >= .12 && Math.abs(margin) <= .60,
        'information period/residual/phase gate differs');
      }
      evidence.push(admitted[2 * index + 1].path);
    }
    const hashes = new Map(admitted.map(row => [row.path, hashAdmitted(row)]));
    hashes.set(informationRow.path, hashAdmitted(informationRow));
    check(hashAdmitted(certificateRow) === reader.equivalence_receipt_sha256, 'reader equivalence changed during information validation');
    write(join(root, 'freeze/inputs.sha256'), [...roles].sort().map(role => `${hashes.get(join(data, role))}  ${role}\n`).join(''));
    write(join(root, 'freeze/information.sha256'), evidence.map(file =>
      `${hashes.get(file) ?? sha(file)}  ${file}\n`).join(''));
    write(join(root, 'freeze/independent-reader.json'), { original_codec_sha256: ORIGINAL_CODEC,
      implementation_sha256: reader.implementation_sha256, equivalence_receipt: reader.equivalence_receipt,
      equivalence_receipt_sha256: reader.equivalence_receipt_sha256 });
  }
  function finish(root) {
    capsulePath(root);
    check(!exists(join(root, 'artifacts.sha256')) && !exists(join(root, 'complete.json')),
      'completed capsule cannot be finalized again');
    const retained = retainedAdmissions(root);
    const receipt = document(join(root, 'freeze/receipt.json'));
    check(equal(receipt.migration_implementation, implementationReceipt(receipt.source_id, [])),
      'new source identity and receipt encoding');
    const bindings = { source_fingerprint: receipt.source_id, card_sha: receipt.card_sha256,
      config_sha: receipt.config_sha256, input_manifest_sha: sha(join(root, 'freeze/inputs.sha256')) };
    for (const profile of ['A', 'B', 'C']) for (const master of MASTERS) {
      const run = join(root, 'profiles', profile, `master-${master}`), result = document(join(run, 'complete.json'));
      check(result.status === 'complete' && result.updates === 512 && result.protocol === PROTOCOL &&
        result.profile === profile && result.master === master && result.report_sha256 === sha(join(run, 'report.json')) &&
        Object.entries(bindings).every(([key, value]) => result[key] === value), 'completed producer counter/report/bindings differ');
      for (const point of [0, 512]) for (const name of ['checkpoint.pt', 'features.pt',
        'reconstruction-training.pt', 'reconstruction-validation.pt']) regular(join(run, `point-${point}`, name));
    }
    const audit = document(join(root, 'readouts/complete.json'));
    check(audit.status === 'complete' && audit.protocol === PROTOCOL && audit.saved_arithmetic_fits_replayed === 81 &&
      audit.report_sha256 === sha(join(root, 'readouts/report.json')), 'completed saved-reader counter/report differs');
    const summary = document(join(root, 'summary.json'));
    check(summary.protocol === PROTOCOL && summary.saved_arithmetic_fits_replayed === 81 &&
      summary.source_fingerprint === bindings.source_fingerprint && summary.card_sha === bindings.card_sha &&
      summary.input_manifest_sha === bindings.input_manifest_sha, 'completed publisher/report bindings differ');
    regular(join(root, 'report.md'));
    const completion = { protocol: PROTOCOL, status: 'complete', encoder_trajectories: 9,
      quality_masters: [...MASTERS], completed_updates_per_run: 512, saved_head_fits: 81,
      migration_implementation: receipt.migration_implementation };
    finalize(root, completion, { admittedFiles: retained });
    return completion;
  }
  return Object.freeze({ freeze, verifyAdmission, bind, finish });
}
if(isMain(import.meta.url)) await capsuleMain('configurable_capacity_v1',createCapsule);
