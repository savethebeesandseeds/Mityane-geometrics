// Saved-vector adapter. No encoder or provider client is imported.
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {encodeRequest, collectArtifactRefs, inputSchema} from './request.mjs';
import {writeExample} from './example.mjs';
import {nativeResultSchema, vectorArtifactSchema} from './schemas.mjs';
import {validateSchema} from '../../../shared/representation/tools/v1/schemas.mjs';
import {ROOT, keys} from '../../../shared/representation/tools/v1/runtime.mjs';
import {admitFiles, readAdmitted, sha256, parseJSON, jsonBytes, requireCondition as require}
  from '../../../shared/artifacts/node/v1/files.mjs';

const sources = [
  'code/operations/tools.mjs',
  'code/operations/retrieval/tool/adapter.mjs',
  'code/operations/retrieval/tool/request.mjs',
  'code/operations/retrieval/tool/native.cpp',
  'code/operations/retrieval/tool/schemas.mjs',
  'code/operations/retrieval/include/mityane/operations/retrieval/exact_search.h',
  'code/operations/retrieval/src/exact_search.cpp',
  'code/operations/retrieval/retrieval.mk',
  'code/shared/representation/tools/v1/runtime.mjs',
  'code/shared/representation/tools/v1/schemas.mjs',
  'code/shared/artifacts/node/v1/files.mjs',
];

const parse = bytes => parseJSON(bytes, {duplicates: 'reject', integers: 'number', allowBOM: false});
const sameValues = (left, right) => Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
  left.every((value, index) => value === right[index]);

function vectorMatches(bytes, point, label) {
  const saved = parse(bytes);
  keys(saved, ['source_id', 'representation_kind', 'values'], label);
  require(saved.source_id === point.source_id && saved.representation_kind === point.representation_kind &&
    sameValues(saved.values, point.values), `${label} does not match the inline vector`);
}

export const retrievalTool = {
  outputRoot: '.temp/operations/retrieval',
  descriptor: {
    tool_schema_version: 'mityane.operation.tool/v1', id: 'OR01-RET', revision: '0.1.0', family: 'OR-01',
    title: 'Rank saved vectors by exact search',
    question: 'Which saved source vectors are nearest to a query under a declared metric and normalization?',
    claim_levels: ['exploratory_description'],
    commands: ['describe', 'example', 'prepare', 'check', 'run', 'inspect', 'read'],
    unsupported_profiles: [
      {claim_level: 'task_association', title: 'Relevance evaluation',
        contract: 'code/operations/retrieval/CONTRACT.md', callable: false,
        note: 'Needs a bound independent relevance rule. The engineering tool does not score relevance.'},
      {claim_level: 'controlled_model_response', title: 'Controlled model response',
        contract: 'code/operations/counterfactuals/CONTRACT.md', callable: false,
        note: 'Retrieval does not state a contrast or an effect.'},
    ],
    input_schema: inputSchema,
    response_artifact_schema: vectorArtifactSchema,
    native_result_schema: nativeResultSchema,
    result_schema_version: 'mityane.operation.result/v1',
    output: 'Ranked source references, reported skips, and distinct empty, abstained, or failed outcomes',
    compatibility: 'Exact SpaceRef: provider, revision, dimensions, normalization, metric, and task role. Equal lengths do not match.',
    model_execution: false,
    check_behavior: 'Validate saved-vector bytes, then rank them in memory. Publish no files and call no provider.',
    limits: {request_bytes: 67108864, artifact_bytes: 268435456, corpus: 4096, dimensions: 8192, native_timeout_ms: 30000},
    limitations: [
      'This tool ranks caller-supplied saved vectors. It does not call Gemini, train an encoder, or judge relevance.',
      'A completed invocation is an engineering execution outcome. It does not establish retrieval quality.',
      'Source-window overlap and as-of availability are applied before exact search and are reported rather than dropped silently.',
    ],
  },
  encode: encodeRequest,
  references: collectArtifactRefs,
  example: writeExample,
  async check(wire) { return this.execute(wire); },
  context(input) { return {operation_run_id: input.operation_run_id, claim_level: input.claim_level}; },
  outcome(result, input) {
    validateSchema(nativeResultSchema, result, {label: 'native retrieval result'});
    require(result.engineering_only === true && result.operation_run_id === input.operation_run_id &&
      result.claim_level === input.claim_level && result.profile === input.profile &&
      result.space_id === input.space.space_id && result.snapshot_id === input.space.snapshot_id &&
      result.metric === input.space.metric && result.normalization === input.space.normalization &&
      result.code_sha256 === input.code_sha256 && result.configuration_sha256 === input.configuration_sha256 &&
      result.fitted_transform_id === input.space.fitted_transform.id, 'native lineage does not match the request');
    const summary = result.summary;
    const parents = new Map(input.corpus.map(row => [row.source_id, row.parent.id]));
    if (!result.admitted) {
      require(result.outcome === 'rejected' && result.hits.length === 0 && result.skipped.length === 0 &&
        result.eligible_count === 0 && summary.requested === 1, 'rejected admission invented rows');
    } else {
      require(summary.requested === 1, 'native requested count mismatch');
      const seen = new Set();
      result.hits.forEach((hit, index) => {
        require(hit.rank === index + 1 && parents.get(hit.source_id) === hit.parent_id && !seen.has(hit.source_id),
          'native hit lineage or rank mismatch');
        seen.add(hit.source_id);
      });
      for (const skip of result.skipped) {
        require(parents.get(skip.source_id) === skip.parent_id && !seen.has(skip.source_id),
          'native skip lineage mismatch');
        seen.add(skip.source_id);
      }
      const expected = {ranked: 'succeeded', empty: 'succeeded', abstained: 'abstained', failed: 'failed'};
      require(Object.hasOwn(expected, result.outcome), 'unknown native outcome');
      require(summary[expected[result.outcome]] === 1, 'native outcome count mismatch');
      require(result.outcome === 'ranked' ? result.hits.length >= 1 : result.hits.length === 0, 'native hit list mismatch');
      require(result.outcome !== 'empty' || result.eligible_count === 0, 'empty result reported eligible members');
    }
    const details = [`${input.operation_run_id}: ${result.outcome}`];
    for (const diagnostic of result.admission_diagnostics) details.push(`Admission ${diagnostic.code}: ${diagnostic.message}`);
    for (const hit of result.hits) details.push(`${hit.source_id}: rank ${hit.rank}`);
    for (const skip of result.skipped) details.push(`${skip.source_id}: skipped ${skip.reason}`);
    const status = !result.admitted ? 'rejected' : summary.failed || summary.unresolved ? 'partial' : 'completed';
    return {summary, details, status};
  },
  verify(input, artifacts) {
    const space = parse(artifacts.get(input.space.artifact.id));
    keys(space, ['space_id', 'snapshot_id', 'provider_id', 'revision_id', 'dimensions', 'normalization',
      'metric', 'task_role', 'representation_kind'], 'space artifact');
    for (const key of Object.keys(space)) require(space[key] === input.space[key], `space artifact field ${key} is stale`);
    const provider = parse(artifacts.get(input.space.provider.id));
    keys(provider, ['provider_id'], 'provider artifact');
    require(provider.provider_id === input.space.provider_id, 'provider artifact is stale');
    const revision = parse(artifacts.get(input.space.revision.id));
    keys(revision, ['revision_id'], 'revision artifact');
    require(revision.revision_id === input.space.revision_id, 'revision artifact is stale');
    const fitted = parse(artifacts.get(input.space.fitted_transform.id));
    keys(fitted, ['available_at'], 'fitted artifact');
    require(fitted.available_at === input.space.fitted_available_at, 'fitted artifact availability is stale');
    vectorMatches(artifacts.get(input.query.parent.id), input.query, 'query artifact');
    for (const row of input.corpus) vectorMatches(artifacts.get(row.parent.id), row, `corpus artifact ${row.source_id}`);
    require(jsonBytes(input.relevance_rule) && input.relevance_rule === null, 'relevance rule must stay unbound');
  },
  async execute(wire) {
    const binary = process.env.MITYANE_RETRIEVAL_TOOL_BINARY ||
      '/opt/mityane-geometrics/embeddings/build/baseline/retrieval/exact_search_tool';
    require(path.isAbsolute(binary) && binary.startsWith('/opt/mityane-geometrics/') && path.normalize(binary) === binary,
      'native binary must be an exact /opt/mityane-geometrics/ build path');
    const binaryRow = admitFiles([binary])[0];
    require(binaryRow.bytes < 64 * 1024 * 1024, 'native binary exceeds byte limit');
    const binaryHash = sha256(readAdmitted(binaryRow));
    const sourceRows = admitFiles(sources.map(name => path.join(ROOT, name)), {root: ROOT});
    const sourceFiles = sourceRows.map((row, index) => ({path: sources[index], sha256: sha256(readAdmitted(row))}));
    const child = spawnSync(binary, [], {input: wire, encoding: 'utf8', timeout: 30000,
      maxBuffer: 64 * 1024 * 1024, killSignal: 'SIGKILL', cwd: ROOT});
    require(!child.error && child.signal === null && child.status === 0,
      `native retrieval failed: ${child.error?.message || child.stderr || child.signal || child.status}`);
    require(sha256(readAdmitted(binaryRow)) === binaryHash, 'native executable changed during invocation');
    sourceRows.forEach((row, index) => require(sha256(readAdmitted(row)) === sourceFiles[index].sha256,
      'tool source changed during invocation'));
    return {
      result: parse(child.stdout),
      native_binary: {path: binary, sha256: binaryHash}, source_files: sourceFiles,
    };
  },
};
