// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../LICENSE.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { reviewData, stableBest, safeRelative, OBLIGATIONS } from './review.mjs';
import { parseJSON, jsonBytes } from '../../../../shared/artifacts/node/v1/files.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../..');
const payload = path.join(root, 'code/embeddings/output/engineering/causal-fresh-v14-finalization/attempt-001/migration-snapshot-001/payload');
const parse = async file => parseJSON(await readFile(file), { duplicates: 'last', integers: 'bigint' });
async function fixture() {
  const card = await parse(path.join(payload, 'card-source-001/finalization-card.json'));
  const inputs = {}, byId = {};
  for (const row of card.inputs) {
    byId[row.id] = row;
    if (row.format === 'json') inputs[row.id] = await parse(safeRelative(path.join(payload, 'card-source-001'), row.staged_path));
  }
  const proposals = path.join(payload, 'proposals-001');
  const family = 'code/encoders/causal_patch_bottleneck_mae/versions/';
  return { card, inputs, byId, receipt: await parse(path.join(proposals, 'finalization-proposal.json')),
    ledger: await parse(path.join(proposals, 'doc/clean_causal_encoder_research_ledger.json')),
    current: await parse(path.join(proposals, family + 'current.json')),
    instance: await parse(path.join(proposals, family + 'instances/CPB-v14.market-01.json')) };
}

test('completed frozen DATA produces 3/22, no new leader, failed advancement and every obligation group', async () => {
  const result = reviewData(await fixture());
  assert.equal(result.numeric_passed, 3);
  assert.equal(result.numeric_total, 22);
  assert.equal(result.strict_new_quality_leaders, 0);
  assert.equal(result.overall_establishment_passed, false);
  assert.ok(result.checks > 500);
  assert.deepEqual(Object.keys(result.obligation_check_counts).sort(), OBLIGATIONS.map(([id]) => id).filter(id => id !== 'frozen-bindings').sort());
});

test('exact ties preserve earlier chronology in both directions', () => {
  const old = { method: 'earlier', score: 0.6 }, tied = { method: 'later', score: 0.6 };
  assert.equal(stableBest([old, tied]), old);
  assert.equal(stableBest([old, tied], 'min'), old);
  assert.throws(() => stableBest([{ score: NaN }]));
});

test('adjacent unsafe integer seeds remain distinct and round-trip exactly', () => {
  const left = parseJSON('{"seed":18446744073709551614}');
  const right = parseJSON('{"seed":18446744073709551615}');
  assert.equal(left.seed, 18446744073709551614n);
  assert.equal(right.seed, 18446744073709551615n);
  assert.notDeepEqual(left, right);
  assert.deepEqual(parseJSON(jsonBytes(right)), right);
});

test('path boundaries reject host and POSIX absolute paths and traversal', () => {
  for (const bad of ['/etc/passwd', 'C:\\Windows\\x', '../x', 'x/../y', 'x//y', './x', '']) assert.throws(() => safeRelative('/tmp/leaf', bad));
  assert.equal(safeRelative('/tmp/leaf', 'inputs/a.json'), '/tmp/leaf/inputs/a.json');
});

test('preserves every historical leader and refuses a fabricated broad V14 win', async () => {
  const data = await fixture();
  data.ledger.stage_leaders[0].method = 'CPB-v14.market-01';
  assert.throws(() => reviewData(data), /Strongest V3-V14 final leader/);
});

test('rejects retrospective protocol-floor edits even with unchanged outcome counts', async () => {
  const data = await fixture();
  const condition = data.inputs.comparison.numeric_conditions[0];
  const declared = data.inputs.experiment_card.advancement.criteria[0];
  const key = Object.keys(declared).find(key => typeof declared[key] === 'number');
  assert.ok(key, 'fixture must contain numeric prospective threshold');
  condition[key] += 0.0001;
  assert.throws(() => reviewData(data), /Exact prospective condition/);
});

test('rejects trained ancestry and full-native40 equality overclaims', async () => {
  const ancestry = await fixture();
  ancestry.instance.scientific_recipe_parent_is_source_only = false;
  assert.throws(() => reviewData(ancestry), /Fresh source-only recipe labels/);
  const pairing = await fixture();
  pairing.inputs.v13_establishment.whole_native40_initial_equality_claimed = true;
  assert.throws(() => reviewData(pairing), /partial20:40 semantics/);
});

test('refuses full-gradient claims beyond actual first/last local-adjoint scope', async () => {
  const data = await fixture();
  data.inputs.training_audit.temporal_pooling_saved_learning.temporal_moment_saved_learning.actual_same_forward_moment_witnesses[0].full_hidden_encoder_or_global_norm_reconstruction = true;
  assert.throws(() => reviewData(data), /Same-forward local-adjoint scope/);
});

test('saved build closure is a mapping with exactly eleven translation units', async () => {
  const original = await fixture();
  const closures = original.inputs.build.dependency_closures;
  assert.equal(Array.isArray(closures), false);
  assert.equal(Object.keys(closures).length, 11);
  for (const mutate of [
    value => { delete value[Object.keys(value)[0]]; },
    value => { value['invented-extra.d'] = value[Object.keys(value)[0]]; },
  ]) {
    const data = await fixture(); mutate(data.inputs.build.dependency_closures);
    assert.throws(() => reviewData(data), /Seven binaries and eleven translation units/);
  }
  original.inputs.build.dependency_closures = Object.values(closures);
  assert.throws(() => reviewData(original), /Seven binaries and eleven translation units/);
});
