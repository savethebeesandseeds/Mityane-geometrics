#!/usr/bin/env node
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {parseArgs} from 'node:util';
import fs from 'node:fs';
import {pairedResponseTool} from './counterfactuals/tool/adapter.mjs';
import {retrievalTool} from './retrieval/tool/adapter.mjs';
import {listTools, describeTool, checkTool, runTool, prepareTool, readResult, inspectResult, exitCode, outputPath, ROOT, createToolset}
  from '../shared/representation/tools/v1/runtime.mjs';

export const tools = Object.freeze([pairedResponseTool, retrievalTool]);
export const operations = createToolset(tools);
export async function main(args) {
  if (args.length === 0 || args[0] === '--help' || args[0] === 'help') {
    process.stdout.write('Operation tools v1\nlist\ndescribe OPERATION\nexample OPERATION --provider toy-a|toy-b --output NEW_DIRECTORY\nprepare --spec FILE --output NEW_DIRECTORY\ncheck --request FILE\nrun --request FILE --output NEW_DIRECTORY\ninspect --result FILE\nread --result FILE\nworkflow OPERATION --provider toy-a|toy-b --output NEW_DIRECTORY\nsweep OPERATION --request FILE --space FILE [--catalog FILE] --output NEW_DIRECTORY\ndiff OPERATION --left RESULT_JSON --right RESULT_JSON --output NEW_DIRECTORY [--abs-tolerance NUMBER] [--rel-tolerance NUMBER]\nverify-study OPERATION --attempt RETAINED_DIRECTORY --output NEW_DIRECTORY\n');
    return 0;
  }
  if (process.platform !== 'linux' || ROOT !== '/mityane' || !fs.existsSync('/.dockerenv'))
    throw new Error('Use code/operations/tools.ps1 through the managed Linux container.');
  const command = args[0];
  const {positionals, values} = parseArgs({args: args.slice(1), allowPositionals: true, strict: true,
    options: {request: {type: 'string'}, spec: {type: 'string'}, output: {type: 'string'}, result: {type: 'string'}, provider: {type: 'string'},
      space: {type: 'string'}, catalog: {type: 'string'}, left: {type: 'string'}, right: {type: 'string'}, attempt: {type: 'string'},
      'abs-tolerance': {type: 'string'}, 'rel-tolerance': {type: 'string'}}});
  const exact = (names, count = 0, optional = []) => {
    if (positionals.length !== count || Object.keys(values).some(name => ![...names,...optional].includes(name)) ||
      names.some(name => !Object.hasOwn(values, name))) throw new Error(`invalid arguments for ${command}; use --help`);
  };
  let result, code = 0;
  if (command === 'list') { exact([]); result = listTools(tools); }
  else if (command === 'describe') { exact([], 1); result = describeTool(tools, positionals[0]); }
  else if (command === 'example') {
    exact(['provider', 'output'], 1);
    const tool = tools.find(tool => tool.descriptor.id === positionals[0]);
    if (!tool) throw new Error(`unknown operation: ${positionals[0]}`);
    result = await tool.example({provider: values.provider, output: outputPath(tool, values.output)});
  } else if (command === 'prepare') {
    exact(['spec', 'output']); result = await prepareTool(tools, values.spec, values.output);
  } else if (command === 'workflow') {
    exact(['provider', 'output'], 1);
    const tool = tools.find(tool => tool.descriptor.id === positionals[0]);
    if (typeof tool?.workflow !== 'function') throw new Error(`operation supplies no workflow: ${positionals[0]}`);
    result = await tool.workflow({operations, provider: values.provider, output: outputPath(tool, values.output)});
    code = exitCode(result.status);
  } else if (command === 'check') {
    exact(['request']); result = await checkTool(tools, values.request); code = exitCode(result.status);
  } else if (command === 'run') {
    exact(['request', 'output']); result = await runTool(tools, values.request, values.output); code = exitCode(result.status);
  } else if (command === 'inspect') {
    exact(['result']); process.stdout.write(inspectResult(values.result, tools)); return 0;
  } else if (command === 'read') {
    exact(['result']); result = readResult(values.result, tools); code = exitCode(result.record.status);
  } else if (command === 'sweep') {
    exact(['request', 'space', 'output'], 1, ['catalog']);
    result = await operations.sweep(positionals[0], {request: values.request, space: values.space,
      ...(Object.hasOwn(values, 'catalog') ? {catalog: values.catalog} : {}), output: values.output});
    code = exitCode(result.status);
  } else if (command === 'diff') {
    exact(['left', 'right', 'output'], 1, ['abs-tolerance', 'rel-tolerance']);
    const tolerance = name => {
      if (!Object.hasOwn(values,name)) return 0;
      if (!/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(values[name]))
        throw new Error(`${name} must be a finite nonnegative number`);
      const number = Number(values[name]);
      if (!Number.isFinite(number)) throw new Error(`${name} must be a finite nonnegative number`);
      return number;
    };
    result = await operations.diff(positionals[0], {left: values.left, right: values.right, output: values.output,
      absoluteTolerance: tolerance('abs-tolerance'), relativeTolerance: tolerance('rel-tolerance')});
    code = exitCode(result.status);
  } else if (command === 'verify-study') {
    exact(['attempt', 'output'], 1);
    result = await operations.verifyStudy(positionals[0], {attempt: values.attempt, output: values.output});
    code = exitCode(result.status);
  } else throw new Error(`unknown command: ${command}`);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  return code;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { process.exitCode = await main(process.argv.slice(2)); }
  catch (error) { process.stderr.write(`Operation tool: ${error.message}\n`); process.exitCode = [2,3,4].includes(error.exitCode) ? error.exitCode : 2; }
}
