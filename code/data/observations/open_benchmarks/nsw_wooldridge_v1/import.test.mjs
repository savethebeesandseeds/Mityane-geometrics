import test from 'node:test';
import assert from 'node:assert/strict';
import {LOCK, parseCSV, validateMetadata, validateTable} from './import.mjs';
import {baselineRecords} from './materialize.mjs';

const expected = {...LOCK, rows: 2, assignment_counts: {'0': 1, '1': 1}};
const values = (assignment, re74 = '1.25', re78 = '2.5') => LOCK.fields.map(field =>
  ({train: assignment, re74, re78}[field] ?? '0'));
const csv = (rows, header = LOCK.fields) => Buffer.from([header.join(','), ...rows.map(row => row.join(','))].join('\n') + '\n');
const rows = [values('0'), values('1')];

test('artificial rows retain exact schema and only approved baseline role without estimating outcomes', () => {
  const result = validateTable(csv(rows), expected);
  assert.deepEqual(result.assignment_counts, {'0': 1, '1': 1});
  assert.deepEqual(result.approved_matching_fields, ['re74']);
  assert.equal(result.effect_estimated, false);
  assert.equal(result.outcome_values_aggregated, false);
  assert.deepEqual(parseCSV(Buffer.from('"a","b"\r\n"1","2"\r\n')), [['a', 'b'], ['1', '2']]);
});

test('reject changed headers, truncated/extra rows, ragged fields and wrong assignment populations', () => {
  assert.throws(() => validateTable(csv(rows, [...LOCK.fields].reverse()), expected), /schema/);
  assert.throws(() => validateTable(csv(rows.slice(1)), expected), /row count/);
  assert.throws(() => validateTable(csv([...rows, rows[0]]), expected), /row count/);
  assert.throws(() => validateTable(csv([rows[0].slice(1), rows[1]]), expected), /ragged/);
  assert.throws(() => validateTable(csv([values('1'), values('1')]), expected), /roster/);
  assert.throws(() => validateTable(csv([values('2'), values('1')]), expected), /binary/);
});

test('refuse nonfinite or missing approved input/outcome and undeclared covariates', () => {
  for (const baseline of ['NA', '', 'NaN', 'Infinity', '1e999', '-1']) {
    assert.throws(() => validateTable(csv([values('0', baseline), rows[1]]), expected));
  }
  assert.throws(() => validateTable(csv([values('0', '1', 'NA'), rows[1]]), expected), /observed finite/);
  assert.throws(() => validateTable(csv(rows), {...expected, approved_matching_fields: ['re74', 'age']}), /only re74/);
});

test('refuse malformed CSV, invalid encoding and oversized fields', () => {
  for (const source of ['a,b\n"1,2', 'a,b\n"1"x,2\n', 'a,b\n1"2,3\n', 'a,b\r1,2', 'a,b\n"one\ntwo",3']) {
    assert.throws(() => parseCSV(Buffer.from(source)));
  }
  assert.throws(() => parseCSV(Buffer.from([0xff])));
  assert.throws(() => parseCSV(Buffer.from('a,b\n' + '1'.repeat(1001) + ',2\n')), /bound/);
  assert.throws(() => parseCSV(Buffer.from('a,b\n1,\0\n')), /NUL/);
});

test('publisher metadata must bind exact package version, source revision, license and dataset', () => {
  const value = {Package: LOCK.package, Version: LOCK.package_version, RemoteSha: LOCK.source_commit,
    License: LOCK.publisher_license, _datasets: [{name: 'jtrain2', rows: LOCK.rows, fields: LOCK.fields}]};
  const check = item => validateMetadata(Buffer.from(JSON.stringify(item)));
  assert.equal(check(value).source_commit, LOCK.source_commit);
  for (const key of ['Package', 'Version', 'RemoteSha', 'License']) assert.throws(() => check({...value, [key]: 'changed'}), /differs/);
  assert.throws(() => check({...value, _datasets: [{...value._datasets[0], rows: 444}]}), /differs/);
});

test('baseline materialization excludes outcomes and all other source fields', () => {
  const original = baselineRecords(csv(rows), 'artificial-test-only', expected);
  const changed = rows.map(row => row.map((value, column) => ['train', 're74'].includes(LOCK.fields[column]) ? value : '987654'));
  assert.deepEqual(baselineRecords(csv(changed), 'artificial-test-only', expected), original);
  assert.deepEqual(original[0], {unit_id: 'artificial-test-only:source-row-0001', source_row_ordinal: 1,
    assignment: 0, values: [1.25]});
  assert.deepEqual(Object.keys(original[0]).sort(), ['assignment', 'source_row_ordinal', 'unit_id', 'values']);
  assert.notEqual(original[0].unit_id, original[1].unit_id);
});
