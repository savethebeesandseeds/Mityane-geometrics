// Metadata CLI for new saved-coordinate diagnostic receipts.
import { main as capsuleMain } from '../node/v1/capsule_cli.mjs';
import { isMain } from '../node/v1/metadata.mjs';
import {
  check, equal, join, lines, sha, exists, text, document, direct, write, copy, finalize,
  timing, assertNewRoot, implementationReceipt, captureSources, retainedAdmissions, regular, CAPSULE_CLI_SOURCES,
} from '../node/v1/capsule_support.mjs';
import { admitFiles } from '../../../../shared/artifacts/node/v1/files.mjs';

export const ORIGINAL_MANIFEST = '62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83';
export const ORIGINAL_SOURCE = '8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1';
export const PROTOCOL = 'configurable-branch-v1';
export const BINARY = 'embedding_configurable_branch_readout';

export function createCapsule(options) {
  check(options && typeof options.repositoryRoot === 'string' && typeof options.buildDirectory === 'string' &&
    typeof options.originalCapsule === 'string', 'explicit migration path/source binding required');
  const { repositoryRoot, buildDirectory, originalCapsule } = options;
  check(buildDirectory.startsWith('/opt/mityane-geometrics/'), 'new isolated managed build session');
  const runRoot = join(repositoryRoot, '.temp/embeddings/runs/rpb-v20-branches');
  const recordedOriginal = '/embedding/output/runs/rpb-v20-capacity/admission-0gN4uD';
  const mapping = [{ recorded_path: recordedOriginal, current_path: originalCapsule,
    artifacts_manifest_sha256: ORIGINAL_MANIFEST }];
  const capsulePath = root => assertNewRoot(root, runRoot, 'diagnostic-');
  function freeze(root) {
    capsulePath(root); direct(originalCapsule); direct(buildDirectory);
    check(typeof options.gitHead === 'string' && /^[a-f0-9]{40,64}$/.test(options.gitHead), 'explicit captured Git revision');
    const historicalRows = new Map(admitFiles(['artifacts.sha256', 'complete.json', 'freeze/receipt.json']
      .map(name => join(originalCapsule, name)), { singleLink: true }).map(row => [row.path, row]));
    check(sha(historicalRows.get(join(originalCapsule, 'artifacts.sha256'))) === ORIGINAL_MANIFEST, 'original manifest differs');
    const done = document(historicalRows.get(join(originalCapsule, 'complete.json')));
    check(done.protocol === 'configurable-capacity-v1' && done.status === 'complete' &&
      done.artifacts_manifest_sha256 === ORIGINAL_MANIFEST, 'original completion differs');
    const originalReceipt = document(historicalRows.get(join(originalCapsule, 'freeze/receipt.json')));
    check(originalReceipt.source_id === ORIGINAL_SOURCE, 'original source receipt differs');
    const card = 'code/embeddings/evaluation/cards/configurable_branch_v1.md';
    const { out, sourceId, sourceFiles, hashes, sourceRows } = captureSources(root, { ...options, binaries: [BINARY], requiredSources: [
      ...CAPSULE_CLI_SOURCES,
      'code/embeddings/evaluation/protocols/configurable_branch_v1/capsule.mjs',
      'code/embeddings/evaluation/protocols/node/v1/capsule_support.mjs',
      'code/shared/artifacts/node/v1/files.mjs',
      card,
    ] });
    check(sourceId !== ORIGINAL_SOURCE, 'new implementation cannot reuse historical source identity');
    copy(sourceRows.get(card), join(out, 'card.md'));
    check(sha(join(out, 'card.md')) === hashes.get(card), 'frozen card exact captured source bytes');
    copy(historicalRows.get(join(originalCapsule, 'artifacts.sha256')), join(out, 'original-artifacts.sha256'));
    const binding = { protocol: PROTOCOL, original_capsule: recordedOriginal,
      original_source_fingerprint: ORIGINAL_SOURCE, original_artifacts_manifest_sha256: ORIGINAL_MANIFEST,
      original_card_sha256: originalReceipt.card_sha256, original_config_sha256: originalReceipt.config_sha256 };
    write(join(out, 'original-evidence.json'), binding);
    const receipt = { ...binding, source_fingerprint: sourceId, card_sha256: sha(join(out, 'card.md')),
      source_files: sourceFiles, build_directory: buildDirectory, git_head: options.gitHead,
      migration_implementation: implementationReceipt(sourceId, mapping) };
    write(join(out, 'receipt.json'), receipt);
    return receipt;
  }
  function finish(root) {
    capsulePath(root);
    check(!exists(join(root, 'complete.json')) && !exists(join(root, 'artifacts.sha256')),
      'completed diagnostic cannot be finalized again');
    const retained = retainedAdmissions(root);
    const receipt = document(join(root, 'freeze/receipt.json'));
    check(equal(receipt.migration_implementation, implementationReceipt(receipt.source_fingerprint, mapping)),
      'new source identity, encoding and historical path mapping');
    const reader = document(join(root, 'readouts/complete.json'));
    const report = document(join(root, 'readouts/report.json'));
    const summary = document(join(root, 'published/summary.json'));
    check(reader.protocol === PROTOCOL && reader.status === 'complete' && reader.encoder_calls === 0 &&
      reader.head_fitted_pairs === 27 && reader.saved_arithmetic_fits_replayed === 27 &&
      reader.report_sha256 === sha(join(root, 'readouts/report.json')) &&
      reader.arithmetic_audit_sha256 === sha(join(root, 'readouts/arithmetic-audit.json')),
    'reader completion or artifact binding differs');
    check(summary.protocol === PROTOCOL && summary.encoder_calls === 0 && summary.encoder_updates === 0 &&
      summary.new_head_fit_pairs === 27 && summary.new_classifiers === 54, 'audited diagnostic scope differs');
    check(summary.source_fingerprint === receipt.source_fingerprint && summary.card_sha256 === receipt.card_sha256 &&
      report.source_fingerprint === receipt.source_fingerprint && report.card_sha === receipt.card_sha256 &&
      summary.original_source_fingerprint === ORIGINAL_SOURCE && summary.original_artifact_manifest_sha === ORIGINAL_MANIFEST &&
      summary.branch_report_sha === sha(join(root, 'readouts/report.json')), 'published original/source/card/report binding differs');
    check(sha(join(root, 'freeze/sources.sha256')) === receipt.source_fingerprint &&
      sha(join(root, 'freeze/card.md')) === receipt.card_sha256 &&
      sha(join(root, 'freeze/original-artifacts.sha256')) === ORIGINAL_MANIFEST, 'captured evidence differs from freeze');
    for (const [name, count] of [['original-before-diagnostic.log', 815], ['original-after-diagnostic.log', 815],
      ['source-after-diagnostic.log', receipt.source_files], ['binary-after-diagnostic.log', 1]]) {
      const actual = lines(text(join(root, name)));
      check(actual.length === count && actual.every(line => line.endsWith(': OK')), `incomplete checksum verification: ${name}`);
    }
    const engineering = document(join(root, 'engineering/complete.json'));
    check(engineering.status === 'passed' && engineering.fitted_pairs === 9 && engineering.encoder_calls === 0,
      'artificial engineering scope differs');
    const stages = Object.fromEntries(['audit-test', 'audit-test-optimized', 'engineering', 'readout', 'arithmetic-audit', 'publish']
      .map(stage => [stage, timing(root, stage)]));
    write(join(root, 'stage-timings.json'), { unit: 'wall seconds', stages, quality_head_fits: 27,
      artificial_engineering_head_fits: 9, encoder_calls: 0, encoder_updates: 0 });
    retained.push(regular(join(root, 'stage-timings.json')));
    const completion = { protocol: PROTOCOL, status: 'complete', new_head_fit_pairs: 27, new_classifiers: 54,
      encoder_calls: 0, encoder_updates: 0, artificial_engineering_head_fit_pairs: 9,
      artificial_engineering_classifiers: 18, original_artifacts_manifest_sha256: ORIGINAL_MANIFEST,
      migration_implementation: receipt.migration_implementation };
    finalize(root, completion, { admittedFiles: retained });
    return completion;
  }
  return Object.freeze({ freeze, finish });
}
if(isMain(import.meta.url)) await capsuleMain('configurable_branch_v1',createCapsule);
