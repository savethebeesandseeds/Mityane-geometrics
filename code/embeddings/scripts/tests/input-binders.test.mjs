// Copyright (c) 2026 Waajacu. Source-only fixtures; no payload/training execution.
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  MASTERS, PARENTS, TSV_COLUMNS, declaredTrainingRoles, safeTrainRole,
} from '../prepare-training-objective-inputs.mjs';
import {template, validateRoleRows} from '../prepare-v7-decoder-inputs.mjs';

test('TRAIN binder declares exactly 195 physical roles for fifteen frozen instances', () => {
  const roles = declaredTrainingRoles();
  assert.equal(roles.length, 195);
  assert.equal(new Set(roles.map(row => `${row.physicalParent}/${row.relative}`)).size, 195);
  assert.equal(new Set(roles.map(row => row.inputId)).size, 15);
  assert.deepEqual([...new Set(roles.map(row => row.master))], [...MASTERS]);
  assert.deepEqual([...new Set(roles.map(row => row.instanceParent))], PARENTS.map(parent => parent.id));
  for (const id of new Set(roles.map(row => row.inputId))) {
    const selected = roles.filter(row => row.inputId === id);
    assert.equal(selected.length, 13);
    assert.equal(new Set(selected.map(row => row.field)).size, 13);
    assert.equal(selected.filter(row => row.physicalParent === 'v8' && row.relative.startsWith('results/readouts/')).length, 6);
  }
  assert.equal(TSV_COLUMNS.length, 22);
});
test('TRAIN binder rejects held-out or ambiguous paths instead of broadening its allowlist', () => {
  for (const role of ['validation/data.pt', 'x/TEST/labels.pt', 'x/stress.pt', '../training.pt', '/training.pt', 'x//training.pt',
    'x\\training.pt', 'x\ttraining.pt']) assert.throws(() => safeTrainRole(role));
  assert.equal(safeTrainRole('reference/seed-4404-lag_sign/milestone-512/checkpoint.pt.training-raw.pt'),
    'reference/seed-4404-lag_sign/milestone-512/checkpoint.pt.training-raw.pt');
});
test('decoder binder keeps the exact 85 TRAIN/VALIDATION role matrix', () => {
  const expected = template(), rows = Object.entries(expected).map(([manifest_path, relative_path]) => ({manifest_path, relative_path}));
  assert.equal(rows.length, 85);
  assert.equal(rows.filter(row => row.manifest_path.startsWith('v4/')).length, 17);
  assert.equal(rows.filter(row => row.manifest_path.startsWith('v7/')).length, 68);
  const sorted = validateRoleRows([...rows].reverse());
  assert.deepEqual(sorted.map(row => row.manifest_path), Object.keys(expected).sort());
  assert.ok(rows.every(row => !/(^|[\W_])(test|stress)([\W_]|$)/i.test(row.manifest_path)));
  assert.throws(() => validateRoleRows(rows.slice(1)), /85-role/);
  assert.throws(() => validateRoleRows([...rows.slice(1), rows[1]]), /unique/);
  const wrong = rows.map(row => ({...row})); wrong[0].relative_path = 'output/runs/unrelated/checkpoint.pt';
  assert.throws(() => validateRoleRows(wrong), /closed role/);
});
