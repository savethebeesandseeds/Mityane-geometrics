#!/usr/bin/env node
// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
/** Pin the exact control implementation closure before compiling inside Linux. */
import { createHash } from 'node:crypto';
import { readFileSync, realpathSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

export const FILES = Object.freeze([
  'code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/control.h',
  'code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/artifact.h',
  'code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/sidecar.h',
  'code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/tensor_state.h',
  'code/operations/forecasting/evaluation/versions/v1/include/cuwacunu/evaluation/latent_dynamics/v1/baselines.h',
  'code/operations/forecasting/evaluation/versions/v1/src/baselines.cpp',
].sort());

export function identity(root) {
  root = realpathSync(root);
  const fields = [];
  for (const name of FILES) {
    const actual = realpathSync(path.join(root, name));
    const relative = path.relative(root, actual);
    if (path.isAbsolute(relative) || relative === '..' || relative.startsWith(`..${path.sep}`)) {
      throw new Error(`Implementation source escapes repository: ${name}`);
    }
    if (!statSync(actual).isFile()) throw new Error(`Not an implementation source: ${name}`);
    fields.push(name, createHash('sha256').update(readFileSync(actual)).digest('hex'));
  }
  const digest = createHash('sha256');
  // Keep the historical sorted paths, UTF-8 bytes and uint64 little-endian
  // length prefixes exactly: existing artifact source identities depend on it.
  for (const value of ['cuwacunu_artifact_record_v1', 'source_closure_v1', String(fields.length), ...fields]) {
    const raw = Buffer.from(value, 'utf8');
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(raw.length));
    digest.update(length).update(raw);
  }
  return digest.digest('hex');
}

export function main(args = process.argv.slice(2)) {
  const { values } = parseArgs({ args, options: {
    root: { type: 'string', default: '.' }, help: { type: 'boolean', short: 'h' },
  } });
  if (values.help) {
    console.log('Usage: source_identity.mjs [--root PATH]\nPin the exact control implementation closure before compiling inside Linux.');
    return;
  }
  console.log(identity(values.root));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
