// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
import test from 'node:test';
import assert from 'node:assert/strict';
import {fullIdentity, identityKey, inventoryMembers, proposalPins, classifyFailures, requireCount, validatePreservationRoots} from './scope.mjs';
import {parseJSON, jsonBytes} from '../../../../shared/artifacts/node/v1/files.mjs';
const pin = (path = 'output/a.json', repository = 'cuwacunu_embedding') => ({repository, path, bytes: 17, sha256: 'a'.repeat(64)});

test('typed inventory selector selects host pins only and never member JSON children', () => {
  const host = pin(), hidden = pin('data/old-training-input.pt');
  const rows = inventoryMembers({inventory_field: 'mapped_files', entry_pin_field: 'host', prefix: ''},
    {mapped_files: [{host, container_path: '/old/a.json', historical_contents: {hidden}}]});
  assert.deepEqual(rows, [{pin: host, pointer: '/mapped_files/0/host'}]);
});
test('frozen copied-files destination selector retains external repository qualification', () => {
  const external = pin('output/reference.pt', 'cuwacunu_dev');
  assert.deepEqual(inventoryMembers({inventory_field: 'copied_files', entry_pin_field: 'destination'}, {copied_files: [{destination: external}]}),
    [{pin: external, pointer: '/copied_files/0/destination'}]);
});
test('unknown guessed selector, missing member pin and uninspected prefix are refused', () => {
  for (const descriptor of [{inventory_field: 'members', entry_pin_field: 'host'}, {inventory_field: 'files', entry_pin_field: 'source'},
    {inventory_field: 'files', entry_pin_field: 'host', prefix: '/old'}]) assert.throws(() => inventoryMembers(descriptor, {files: [{host: pin()}]}));
  assert.throws(() => inventoryMembers({inventory_field: 'files', entry_pin_field: 'host'}, {files: [{path: 'missing'}]}));
});
test('full tuple keeps same path distinct across repositories and byte generations', () => {
  assert.notEqual(identityKey(pin()), identityKey(pin('output/a.json', 'cuwacunu_dev')));
  assert.notEqual(identityKey(pin()), identityKey({...pin(), sha256: 'b'.repeat(64)}));
  assert.throws(() => fullIdentity({path: 'x', sha256: 'a'.repeat(64)}));
});
test('proposal-only exact pin traversal preserves unsafe integer seeds without following files', () => {
  const value = parseJSON('{"seed":18446744073709551614,"other_seed":18446744073709551615,"evidence":{"path":"x","bytes":1,"sha256":"' + 'a'.repeat(64) + '"}}');
  assert.notEqual(value.seed, value.other_seed);
  assert.equal(proposalPins(value).length, 1); assert.equal(proposalPins(value)[0].pointer, '/evidence');
  assert.deepEqual(parseJSON(jsonBytes(value)), value);
});
test('failure classification requires exact tuple membership and retains every original row', () => {
  const required = pin(), outside = pin('outside.json'), samePathWrongHash = {...required, sha256: 'b'.repeat(64)};
  const original = [required, outside, samePathWrongHash].map(identity => ({identity, manifest: {path: 'old.json'}, pointer: '/record'}));
  const result = classifyFailures(original, [{original: required}], {});
  assert.deepEqual(result.map(row => row.required_in_scope), [true, false, false]);
  assert.deepEqual(result.map(row => row.original_identity), original.map(row => row.identity));
  assert.ok(result.every(row => row.prior_failure_retained));
});
test('artifact-baseline repo field corrects record context but does not resolve old failure', () => {
  const wrong = pin('doc/dev.json'), correct = {...wrong, repository: 'cuwacunu_dev'};
  const failure = {identity: wrong, manifest: {path: 'baseline.json'}, pointer: '/entries/0'};
  const result = classifyFailures([failure], [{original: correct}], {entries: [{...pinFieldsForTest(wrong), repo: 'cuwacunu_dev'}]}, 'baseline.json')[0];
  assert.deepEqual(result.original_identity, wrong); assert.deepEqual(result.record_qualified_identity, correct);
  assert.equal(result.required_in_scope, true); assert.equal(result.disposition, 'record_schema_overinterpretation');
  assert.equal(Object.hasOwn(result, 'resolved'), false);
});
const pinFieldsForTest = ({path, bytes, sha256}) => ({path, bytes, sha256});
test('aggregate historical audit size is not reclassified as a file length', () => {
  const failure = {identity: pin('audit.json'), manifest: {path: 'baseline.json'}, pointer: '/passed_after_audit'};
  const row = classifyFailures([failure], [failure.identity], {current_artifacts_rehashed: false}, 'baseline.json')[0];
  assert.equal(row.record_qualified_identity, null); assert.equal(row.required_in_scope, false);
  assert.equal(row.disposition, 'record_schema_overinterpretation');
});
test('real historical inventory and ledger header pins remain opaque file tuples', () => {
  const failures = ['/complete_baseline', '/search_ledger'].map(pointer => ({identity: pin(pointer.slice(1)), manifest: {path: 'baseline.json'}, pointer}));
  const rows = classifyFailures(failures, [], {}, 'baseline.json');
  rows.forEach((row, index) => {
    assert.deepEqual(row.record_qualified_identity, failures[index].identity);
    assert.equal(row.disposition, 'outside_declared_traversal'); assert.equal(row.required_in_scope, false);
  });
});
test('declared coverage rejects missing or extra actual members', () => {
  assert.equal(requireCount('eleven closures', new Array(11), 11).length, 11);
  for (const count of [10, 12]) assert.throws(() => requireCount('eleven closures', new Array(count), 11));
});
test('migration historical inventory is V14 before, whose baseline-preservation pin separately binds V13 final', () => {
  const v13 = pin('doc/causal_fresh_v13/final-preservation.json'), checkpoints = Array.from({length: 119}, (_, index) => pin(`checkpoint-${index}.pt`));
  const before = {kind: 'cpb_v14_before_execution_file_preservation_v1', status: 'passed', baseline_preservation: v13, critical_checkpoint_pins: checkpoints};
  const final = {kind: 'causal_fresh_v13_final_preservation_v1', status: 'passed', critical_checkpoint_pins: checkpoints};
  validatePreservationRoots(before, final, v13);
  assert.throws(() => validatePreservationRoots(before, before, v13), /distinct actual V14 before and V13 final/);
  assert.throws(() => validatePreservationRoots(before, final, {...v13, sha256: 'b'.repeat(64)}), /declared V13 final full tuple/);
  assert.throws(() => validatePreservationRoots(before, {...final, critical_checkpoint_pins: checkpoints.slice(1)}, v13), /inventories exactly equal/);
});
