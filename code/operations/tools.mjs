#!/usr/bin/env node
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {parseArgs} from 'node:util';
import fs from 'node:fs';
import {pairedResponseTool} from './counterfactuals/tool/adapter.mjs';
import {listTools, describeTool, checkTool, runTool, inspectResult, exitCode, outputPath, ROOT, createToolset}
  from '../shared/representation/tools/v1/runtime.mjs';

export const tools = Object.freeze([pairedResponseTool]);
export const operations = createToolset(tools);
export async function main(args) {
  if (args.length === 0 || args[0] === '--help' || args[0] === 'help') {
    process.stdout.write('Operation tools v1\nlist\ndescribe OPERATION\nexample OPERATION --provider toy-a|toy-b --output NEW_DIRECTORY\ncheck --request FILE\nrun --request FILE --output NEW_DIRECTORY\ninspect --result FILE\n');
    return 0;
  }
  if (process.platform !== 'linux' || ROOT !== '/mityane' || !fs.existsSync('/.dockerenv'))
    throw new Error('Use code/operations/tools.ps1 through the managed Linux container.');
  const command = args[0];
  const {positionals, values} = parseArgs({args: args.slice(1), allowPositionals: true, strict: true,
    options: {request: {type: 'string'}, output: {type: 'string'}, result: {type: 'string'}, provider: {type: 'string'}}});
  const exact = (names, count = 0) => {
    if (positionals.length !== count || Object.keys(values).some(name => !names.includes(name)) ||
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
  } else if (command === 'check') {
    exact(['request']); result = await checkTool(tools, values.request); code = exitCode(result.status);
  } else if (command === 'run') {
    exact(['request', 'output']); result = await runTool(tools, values.request, values.output); code = exitCode(result.status);
  } else if (command === 'inspect') {
    exact(['result']); process.stdout.write(inspectResult(values.result)); return 0;
  } else throw new Error(`unknown command: ${command}`);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  return code;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { process.exitCode = await main(process.argv.slice(2)); }
  catch (error) { process.stderr.write(`Operation tool: ${error.message}\n`); process.exitCode = 2; }
}
