// SPDX-License-Identifier: MIT
// Container-only orchestration and evidence binding. Numerical work is C++.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';

if (process.platform !== 'linux' || !fs.existsSync('/.dockerenv')) {
  throw new Error('Use the managed Linux container and root task entry.');
}
const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i], value = process.argv[i + 1];
  if (!['--test', '--trainer', '--source-root', '--output'].includes(key) ||
      !value || args.has(key)) throw new Error(`Invalid argument ${key}`);
  args.set(key, value);
}
for (const key of ['--test', '--trainer', '--output']) {
  if (!args.has(key)) throw new Error(`Required ${key}`);
}
const root = path.resolve(args.get('--source-root') ?? process.cwd());
const output = path.resolve(args.get('--output'));
const test = path.resolve(args.get('--test'));
const trainer = path.resolve(args.get('--trainer'));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const fileSha = file => sha(fs.readFileSync(file));
const write = (name, value) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2) + '\n');
const family = 'code/embeddings/encoders/mtf_jepa_mae_vicreg/family';
const core = 'code/embeddings/encoders/mtf_jepa_mae_vicreg';
const headers = `${core}/include/embedding/encoders/mtf_jepa_mae_vicreg`;
const shared = 'code/shared/representation';
const files = [
  ...['config.h', 'types.h', 'tensor_ops.h', 'tokenization.h', 'masking.h',
    'encoder.h', 'objectives.h', 'model.h', 'workflow.h', 'evaluation.h'].map(x => `${headers}/${x}`),
  ...['workflow.cpp', 'evaluation.cpp'].map(x => `${core}/src/${x}`),
  ...['types.h', 'tensor_ops.h', 'objectives.h', 'evaluation.h'].map(x => `${shared}/include/embedding/shared/${x}`),
  `${shared}/src/evaluation.cpp`,
  'code/data/observations/include/embedding/shared/data.h', 'code/data/observations/src/data.cpp',
  ...['sidecar.h', 'tensor_state.h'].map(x => `code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/${x}`),
  `${family}/include/embedding/encoders/mtf_jepa_mae_vicreg/family.h`,
  `${family}/src/family.cpp`, `${family}/tests/family_test.cpp`, `${family}/tests/cli_example.mjs`,
  `${family}/config/small.conf`, `${family}/config/alternative.conf`,
  `${family}/family.mk`, `${family}/verification/PROTOCOL_2026-10-22.md`,
  'code/embeddings/scripts/mtf_family_train.cpp', 'code/embeddings/scripts/verify-mtf-family-delivery.mjs',
  'code/embeddings/scripts/export-mtf-family.mjs', 'code/embeddings/scripts/tests/export-mtf-family.test.mjs',
].sort();
const sourceFiles = files.map(relative => {
  const file = path.join(root, relative);
  if (!fs.lstatSync(file).isFile() || fs.lstatSync(file).isSymbolicLink()) {
    throw new Error(`Regular source required: ${file}`);
  }
  return {path: relative, sha256: fileSha(file)};
});
const sourceSha = sha(JSON.stringify(sourceFiles));
if (fs.existsSync(output)) throw new Error(`Preserve existing attempt; choose new output: ${output}`);
fs.mkdirSync(output, {recursive: true});
const environment = {...process.env, CUDA_VISIBLE_DEVICES: '', OMP_NUM_THREADS: '1'};
const preRun = {
  schema: 'mtf-family-engineering-admission-v1', recordedAt: new Date().toISOString(),
  authorization: 'Owner requested pursuit of entire Hx15 portable-training-delivery Goal on 2026-10-22.',
  protocol: `${family}/verification/PROTOCOL_2026-10-22.md`,
  scope: 'Bounded artificial CPU engineering only; no model-quality or application finding.',
  sourceRoot: root, sourceSha256: sourceSha, sourceFiles,
  binaries: [{path: test, sha256: fileSha(test)}, {path: trainer, sha256: fileSha(trainer)}],
  limits: {nativePhysicalUpdates: 48, cliPhysicalUpdates: 6, gpuAccess: false, threads: 1},
};
write('pre-run.json', preRun);
const commands = [];
function run(name, binary, argv) {
  const result = spawnSync(binary, argv, {cwd: root, env: environment, encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024, timeout: 180000});
  fs.writeFileSync(path.join(output, `${name}.stdout.log`), result.stdout ?? '');
  fs.writeFileSync(path.join(output, `${name}.stderr.log`), result.stderr ?? '');
  commands.push({name, executable: binary, arguments: argv, exitStatus: result.status,
    signal: result.signal, stdoutSha256: fileSha(path.join(output, `${name}.stdout.log`)),
    stderrSha256: fileSha(path.join(output, `${name}.stderr.log`))});
  write('commands.json', commands);
  if (result.error || result.status !== 0) {
    write('failure.json', {schema: 'mtf-family-engineering-failure-v1', status: 'failed',
      sourceSha256: sourceSha, failedCommand: name, error: result.error?.message ?? result.stderr,
      physicalUpdateAccounting: 'Inspect retained native/CLI logs and partial artifacts; do not infer completed work from a failed exit.'});
    throw new Error(`${name} failed; retained ${output}\n${result.stderr ?? result.error?.message}`);
  }
}
run('exporter-integrity', process.execPath, ['--test',
  path.join(root, 'code/embeddings/scripts/tests/export-mtf-family.test.mjs')]);
run('native-proof', test, ['--config-directory', path.join(root, family, 'config'),
  '--output', path.join(output, 'native'), '--source-sha256', sourceSha]);
run('cli-example', process.execPath, [path.join(root, family, 'tests/cli_example.mjs'),
  '--trainer', trainer, '--config', path.join(root, family, 'config/alternative.conf'),
  '--output', path.join(output, 'cli'), '--source-sha256', sourceSha]);
const native = JSON.parse(fs.readFileSync(path.join(output, 'native/result.json'), 'utf8'));
const cli = JSON.parse(fs.readFileSync(path.join(output, 'cli/cli-result.json'), 'utf8'));
if (native.status !== 'passed' || cli.status !== 'passed' ||
    !Number.isSafeInteger(native.physical_updates) || native.physical_updates > 48 ||
    !Number.isSafeInteger(cli.physical_updates) || cli.physical_updates > 6 ||
    native.source_sha256 !== sourceSha || cli.source_sha256 !== sourceSha) {
  throw new Error('Proof did not satisfy its declared binding/budget.');
}
for (const item of sourceFiles) {
  if (fileSha(path.join(root, item.path)) !== item.sha256) throw new Error(`Source changed during proof: ${item.path}`);
}
for (const item of preRun.binaries) {
  if (fileSha(item.path) !== item.sha256) throw new Error(`Binary changed during proof: ${item.path}`);
}
function outputs(directory, prefix = '') {
  return fs.readdirSync(directory).sort().flatMap(name => {
    const absolute = path.join(directory, name), relative = path.posix.join(prefix, name), stat = fs.lstatSync(absolute);
    if (stat.isSymbolicLink()) throw new Error(`Unexpected output symlink ${absolute}`);
    return stat.isDirectory() ? outputs(absolute, relative) : [{path: relative, bytes: stat.size, sha256: fileSha(absolute)}];
  });
}
const summary = {schema: 'mtf-family-delivery-engineering-run-v1', status: 'passed',
  recordedAt: new Date().toISOString(), sourceRoot: root, sourceSha256: sourceSha,
  sourceFiles, binaries: preRun.binaries, commands, runtime: native.runtime,
  nativePhysicalUpdates: native.physical_updates, cliPhysicalUpdates: cli.physical_updates,
  totalPhysicalUpdates: native.physical_updates + cli.physical_updates,
  nativeProof: {path: 'native/result.json', sha256: fileSha(path.join(output, 'native/result.json'))},
  cliProof: {path: 'cli/cli-result.json', sha256: fileSha(path.join(output, 'cli/cli-result.json'))},
  files: outputs(output),
  limitations: ['CPU/F32/single-thread and exact same executable/SDK/hardware/backend scope',
    'Committed update boundaries; no crash rollback or durable journal claim',
    'Identity preparation; no fitted statistics', 'No scientific quality or application utility result']};
write('run-summary.json', summary);
process.stdout.write(JSON.stringify({status: summary.status, sourceSha256: sourceSha,
  totalPhysicalUpdates: summary.totalPhysicalUpdates, summary: path.join(output, 'run-summary.json')}) + '\n');
