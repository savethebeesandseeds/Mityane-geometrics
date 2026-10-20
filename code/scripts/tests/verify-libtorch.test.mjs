import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { fixtureCases, withFixture, checkParsing } from './verify-libtorch-fixtures.mjs';

for (const [name, run] of fixtureCases) test(name, () => withFixture(run));
test('proof JSON and SDK paths reject ambiguous or unsafe inputs', checkParsing);

test('CLI self-test exercises artificial fixtures and rejects mixed real-check mode', () => {
  const script = fileURLToPath(new URL('../verify-libtorch.mjs', import.meta.url));
  const result = spawnSync(process.execPath, [script, '--self-test'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.status, 'passed');
  assert.equal(report.fixture_cases, fixtureCases.length);
  assert.equal(report.real_SDK_reads, 0);
  assert.equal(report.real_SDK_mutations, 0);
  const mixed = spawnSync(process.execPath, [script, '--self-test', '--check'], { encoding: 'utf8' });
  assert.equal(mixed.status, 1);
  assert.match(mixed.stderr, /fixture mode cannot verify a real SDK/);
});
