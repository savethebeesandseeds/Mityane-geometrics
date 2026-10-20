// Provider-neutral recorded-response adapter. No encoder or service is imported.
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {encodeRequest, collectArtifactRefs, inputSchema} from './request.mjs';
import {writeExample} from './example.mjs';
import {ROOT, keys} from '../../../shared/representation/tools/v1/runtime.mjs';
import {admitFiles, readAdmitted, sha256, parseJSON, jsonBytes, requireCondition as require}
  from '../../../shared/artifacts/node/v1/files.mjs';

const sources = [
  'code/operations/tools.mjs', 'code/operations/counterfactuals/tool/adapter.mjs',
  'code/operations/counterfactuals/tool/request.mjs', 'code/operations/counterfactuals/tool/native.cpp',
  'code/operations/counterfactuals/include/mityane/operations/counterfactuals/paired_response.h',
  'code/operations/counterfactuals/src/paired_response.cpp',
  'code/operations/counterfactuals/counterfactuals.mk',
  'code/shared/representation/tools/v1/runtime.mjs',
  'code/shared/artifacts/node/v1/files.mjs',
];
export const pairedResponseTool = {
  outputRoot: '.temp/operations/counterfactuals',
  descriptor: {
    tool_schema_version: 'mityane.operation.tool/v1', id: 'OR06-PCR', revision: '0.1.0', family: 'OR-06',
    title: 'Compare recorded paired model responses',
    question: 'How does a frozen numerical model response change under a declared alternative?',
    claim_levels: ['controlled_model_response'],
    commands: ['describe', 'example', 'check', 'run', 'inspect'],
    input_schema: inputSchema,
    response_artifact_schema: {
      type: 'object', additionalProperties: false, required: ['axes', 'values'],
      properties: {axes: {type: 'array', description: 'Exactly the branch axes, in order'},
        values: {type: 'array', items: {type: 'number'}, description: 'Exactly the recorded response_values'}},
    },
    result_schema_version: 'mityane.operation.result/v1',
    output: 'Per-pair alternative-minus-baseline contrasts, typed diagnostics and complete state counts',
    compatibility: 'Exact declared SpaceRef, response axes, output selector and frozen pipeline; equal widths alone do not suffice.',
    model_execution: false,
    check_behavior: 'Validate the request and all artifact bytes, then calculate the pure recorded comparison in memory. Publish no files and call no model.',
    limits: {request_bytes: 67108864, artifact_bytes: 268435456, pairs: 10000, native_timeout_ms: 30000},
    limitations: [
      'This tool compares retained numerical responses; it does not execute or train the embedding or response model.',
      'Artifact bytes and response-value bindings are verified. Provider execution, controls, support and semantic attestations remain caller-supplied.',
      'A completed tool invocation is an execution outcome; it does not establish scientific validity, usefulness or a causal effect.',
    ],
  },
  encode: encodeRequest,
  references: collectArtifactRefs,
  example: writeExample,
  // Read-only by contract: this adapter performs only pure recorded arithmetic.
  async check(wire) { return this.execute(wire); },
  context(input) { return {operation_run_id: input.operation_run_id, claim_level: input.claim_level}; },
  outcome(result, input) {
    require(result && typeof result.admitted === 'boolean' && Array.isArray(result.pairs) &&
      Array.isArray(result.admission_diagnostics), 'invalid native result envelope');
    const summary = result.summary;
    if (result.admitted) {
      require(summary.requested === input.pairs.length && result.pairs.length === input.pairs.length, 'native roster mismatch');
      const counts = {success: 0, abstained: 0, failed: 0, unresolved: 0};
      result.pairs.forEach((row, index) => {
        require(row.pair_index === index && row.pair_id === input.pairs[index].id && Object.hasOwn(counts, row.state),
          'native pair state/order mismatch');
        counts[row.state]++;
        require(row.state === 'success' ? row.contrast !== null : row.contrast === null, 'native contrast/state mismatch');
      });
      require(counts.success === summary.succeeded && counts.abstained === summary.abstained &&
        counts.failed === summary.failed && counts.unresolved === summary.unresolved, 'native states/counts mismatch');
    } else require(result.pairs.length === 0 && summary.requested === input.pairs.length, 'rejected admission invented executed rows');
    const details = [];
    for (const diagnostic of result.admission_diagnostics) details.push(`Admission ${diagnostic.code}: ${diagnostic.message}`);
    for (const pair of result.pairs) {
      details.push(`${pair.pair_id}: ${pair.state}${pair.contrast ? `; delta = [${pair.contrast.delta.join(', ')}]` : ''}`);
      for (const diagnostic of pair.diagnostics) details.push(`  ${diagnostic.code}: ${diagnostic.message}`);
    }
    return {summary, details, status: !result.admitted ? 'rejected' : summary.failed || summary.unresolved ? 'partial' : 'completed'};
  },
  verify(input, artifacts) {
    for (const pair of input.pairs) for (const name of ['baseline', 'alternative']) {
      const branch = pair[name];
      if (branch === null) continue;
      const saved = parseJSON(artifacts.get(branch.response.id), {duplicates: 'reject', integers: 'number', allowBOM: false});
      keys(saved, ['axes', 'values'], `response artifact ${branch.response.id}`);
      require(jsonBytes(saved.axes).equals(jsonBytes(branch.axes)) &&
        jsonBytes(saved.values).equals(jsonBytes(branch.response_values)),
      `response artifact does not match inline axes/values: ${branch.response.id}`);
    }
  },
  async execute(wire) {
    const binary = process.env.MITYANE_COUNTERFACTUALS_TOOL_BINARY ||
      '/opt/mityane-geometrics/embeddings/build/operation-tools-v1/counterfactuals/paired_response_tool';
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
      `native comparison failed: ${child.error?.message || child.stderr || child.signal || child.status}`);
    require(sha256(readAdmitted(binaryRow)) === binaryHash, 'native executable changed during invocation');
    sourceRows.forEach((row, index) => require(sha256(readAdmitted(row)) === sourceFiles[index].sha256,
      'tool source changed during invocation'));
    return {
      result: parseJSON(child.stdout, {duplicates: 'reject', integers: 'number', allowBOM: false}),
      native_binary: {path: binary, sha256: binaryHash}, source_files: sourceFiles,
    };
  },
};
