// Engineering request bundles, independent of any encoder implementation.
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeExampleRequest } from './fixture.mjs';
import { createExclusiveDirectory, writeNew } from '../../../shared/artifacts/node/v1/files.mjs';

const PROVIDERS = {
  'toy-a': { providerId: 'example-provider-A', dimensions: 4 },
  'toy-b': { providerId: 'example-provider-B', dimensions: 7 },
};
const OUTPUT_ROOT = '/mityane/.temp/operations/counterfactuals';

export async function writeExample({ provider = 'toy-a', output }) {
  const selected = PROVIDERS[provider];
  if (!selected) throw new Error(`Unknown engineering provider ${provider}; use toy-a or toy-b.`);
  if (typeof output !== 'string' || !path.isAbsolute(output)) {
    throw new Error('Example output must be an absolute new directory.');
  }
  const destination = path.resolve(output);
  if (!destination.startsWith(`${OUTPUT_ROOT}/`)) {
    throw new Error(`Example output must be a new directory within ${OUTPUT_ROOT}/.`);
  }
  // Retain the common admission checks for every ancestor, including the
  // operation root itself; a symlink must not redirect example publication.
  createExclusiveDirectory(destination, { root: '/mityane' });
  await mkdir(path.join(destination, 'artifacts'));
  const fixture = makeExampleRequest(selected);
  const artifacts = [];
  for (const [index, artifact] of fixture.artifacts.entries()) {
    const relative = `artifacts/${String(index + 1).padStart(4, '0')}-${artifact.sha256}.json`;
    writeNew(path.join(destination, relative), artifact.bytes);
    artifacts.push({ id: artifact.id, sha256: artifact.sha256, path: relative });
  }
  const request = {
    schema_version: 'mityane.operation.request/v1',
    operation: { id: 'OR06-PCR', revision: '0.1.0' },
    input: fixture.input,
    artifacts,
  };
  const requestFile = path.join(destination, 'request.json');
  writeNew(requestFile, `${JSON.stringify(request, null, 2)}\n`);
  return {
    schema_version: 'mityane.operation.example/v1',
    operation: request.operation,
    provider,
    representation_dimensions: selected.dimensions,
    request_file: requestFile,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const options = {};
    for (let i = 2; i < process.argv.length; i += 2) {
      const key = process.argv[i];
      if (!['--provider', '--output'].includes(key) || !process.argv[i + 1] || options[key.slice(2)] !== undefined) {
        throw new Error('Usage: example.mjs --provider toy-a|toy-b --output NEW_DIRECTORY');
      }
      options[key.slice(2)] = process.argv[i + 1];
    }
    process.stdout.write(`${JSON.stringify(await writeExample(options), null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 2;
  }
}
