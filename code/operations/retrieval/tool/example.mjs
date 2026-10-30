// Engineering request bundles of saved vectors. No encoder is loaded.
import path from 'node:path';
import {mkdir} from 'node:fs/promises';
import {engineeringRequest} from './fixture.mjs';
import {createExclusiveDirectory, writeNew} from '../../../shared/artifacts/node/v1/files.mjs';

const OUTPUT_ROOT = '/mityane/.temp/operations/retrieval';

export async function writeExample({provider = 'toy-a', output}) {
  if (typeof output !== 'string' || !path.isAbsolute(output))
    throw new Error('Example output must be an absolute new directory.');
  const destination = path.resolve(output);
  if (!destination.startsWith(`${OUTPUT_ROOT}/`))
    throw new Error(`Example output must be a new directory within ${OUTPUT_ROOT}/.`);
  createExclusiveDirectory(destination, {root: '/mityane'});
  await mkdir(path.join(destination, 'artifacts'));
  const fixture = engineeringRequest({provider});
  const artifacts = [];
  for (const [index, artifact] of fixture.artifacts.entries()) {
    const relative = `artifacts/${String(index + 1).padStart(4, '0')}-${artifact.sha256}.json`;
    writeNew(path.join(destination, relative), artifact.bytes);
    artifacts.push({id: artifact.id, sha256: artifact.sha256, path: relative});
  }
  const request = {
    schema_version: 'mityane.operation.request/v1',
    operation: {id: 'OR01-RET', revision: '0.1.0'},
    input: fixture.input,
    artifacts,
  };
  const requestFile = path.join(destination, 'request.json');
  writeNew(requestFile, `${JSON.stringify(request, null, 2)}\n`);
  return {
    schema_version: 'mityane.operation.example/v1', operation: request.operation, provider,
    representation_dimensions: fixture.input.space.dimensions, request_file: requestFile,
  };
}
