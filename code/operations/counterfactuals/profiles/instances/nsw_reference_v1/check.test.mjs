import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {checkDefinition, checkBindings} from './check.mjs';
const read = () => JSON.parse(fs.readFileSync(new URL('./bindings.json', import.meta.url)));

test('binding refuses outcome/group leakage and an unreviewed fitted OLS dependency', () => {
  for (const mutate of [b => b.representation.input_fields.push('re78'),
    b => b.instances[0].matching.input_fields.push('train'), b => b.representation.synthetic_ols_dependency = true]) {
    const binding = read(); mutate(binding); assert.throws(() => checkDefinition(binding), /prohibited|matching|OLS/);
  }
});
test('binding refuses invented assignment probabilities, fresh-assessment claims and scientific admission', () => {
  for (const mutate of [b => b.instances[1].identification.allocation_probabilities = [185 / 445],
    b => b.partitions.protected_fresh_assessment = true, b => b.execution.admission = 'admitted']) {
    const binding = read(); mutate(binding); assert.throws(() => checkDefinition(binding), /randomization|protected|execution/);
  }
});
test('binding verification refuses changed source bytes and escaping artifact paths', () => {
  const changed = read(); changed.source.snapshot.sha256 = '0'.repeat(64);
  assert.throws(() => checkBindings(changed), /bytes changed|conflicting artifact/);
  const escaped = read(); escaped.source.snapshot.path = '../outside.csv';
  assert.throws(() => checkBindings(escaped), /path|relative|unsafe/);
});
