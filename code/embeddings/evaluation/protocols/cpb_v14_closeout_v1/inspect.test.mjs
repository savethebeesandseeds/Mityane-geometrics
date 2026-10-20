// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {inspectRecords, changedFields, inspect, TARGETS} from './inspect.mjs';
import {parseJSON, jsonBytes, sha256} from '../../../../shared/artifacts/node/v1/files.mjs';

const protocol = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1';
const family = 'code/embeddings/encoders/causal_patch_bottleneck_mae/versions';
const clone = value => parseJSON(jsonBytes(value));
const hash = 'f'.repeat(64);
const pointer = key => key.replaceAll('~', '~0').replaceAll('/', '~1');
function fixture() {
  const historical = {path: 'output/source.json', bytes: 123, sha256: hash};
  const catalogProposal = {kind: 'causal_encoder_live_catalog_v1', active_application: 'CPB-v3.market-01',
    application_selection_record: '../../cuwacunu_dev/selected.md', interpretation: 'Predecessor DEV selection.',
    clean_causal_training_baseline: {instance: 'CPB-v3.market-01'},
    fresh_research_candidate_history: [{instance: 'CPB-v4.market-01', numeric_passed: 19, saved_seed: 18446744073709551615n}],
    fresh_research_candidate: {instance: 'CPB-v14.market-01', numeric_passed: 3, numeric_total: 22, advancement_passed: false,
      design_file: 'CPB-v14.json', physical_version_root: 'v14', measured_instance_file: 'instances/CPB-v14.market-01.json',
      instance_record: 'old/instance.json', fresh_seed: 18446744073709551614n, validation_score: 0.7624567890123456, source: historical}};
  const instanceProposal = {kind: 'causal_encoder_measured_instance_v1', instance: 'CPB-v14.market-01',
    clean_baseline: 'CPB-v3.market-01', overall_winner: false, advancement_passed: false,
    application_selection: 'CPB-v3.market-01', application_selection_changed: false,
    version_root: 'code/encoders/causal_patch_bottleneck_mae/versions/v14', source_manifest: historical, source_capsule: [historical],
    quality: {numeric_passed: 3, numeric_total: 22, strict_new_scoped_quality_leaders: 0, failed: ['trend'], score: -0.00001, seed: 9007199254740993n},
    local_run_root: 'output/run', local_quality_root: 'output/run/quality', local_engineering_root: 'output/engineering', container_root: '/opt/old/run',
    source_only_parent: {version: 'v13', trained_ancestor: false}, unknown_original_field: {values: [0.1, -0, 1], exact_seed: 18446744073709551615n}};
  const aliases = [{original: {repository: 'cuwacunu_embedding', ...historical}, observed: {bytes: historical.bytes, sha256: historical.sha256}, current_path: 'code/embeddings/output/source.json'}];
  const bindings = Object.fromEntries(['admission', 'inputs', 'aliases', 'alias_verification', 'proposal_review', 'source_review'].map(name => [name, {path: `fixture/${name}.json`, bytes: 12, sha256: hash}]));
  const proposalPins = [{role: 'catalog_proposal', path: 'fixture/catalog.json', bytes: 15, sha256: hash}];
  const attemptRelative = '.temp/embeddings/closeout/cpb-v14/attempt-001';
  const archivedLedger = {path: '.archive/fixture-ledger.json', bytes: 125, sha256: hash};
  const predecessor = {repository: 'cuwacunu_embedding', catalog: {active_application: catalogProposal.active_application,
    interpretation: catalogProposal.interpretation, application_selection_record: catalogProposal.application_selection_record},
    instance: {application_selection: instanceProposal.application_selection, application_selection_changed: false},
    policy: 'These are preserved predecessor decisions. They do not establish a Mityane project-wide encoder selection.'};
  const selection = {instance: null, status: 'none_established_by_this_closeout'};
  const binding = {protocol, derivation: 'reviewed historical proposal DATA with explicit field adaptations', immutable_preapplication_inputs: bindings,
    historical_proposals: proposalPins, summary: TARGETS[2], research_execution_authorized: false};
  const locators = original => {
    const located = [];
    const walk = (value, at = '') => {
      if (!value || typeof value !== 'object') return;
      if (value.path && value.bytes && value.sha256) located.push({pointer: at, original_repository: 'cuwacunu_embedding', historical_pin: clone(value),
        current_path: aliases[0].current_path, observed_bytes: value.bytes, observed_sha256: value.sha256});
      Object.entries(value).forEach(([key, value]) => walk(value, `${at}/${pointer(key)}`));
    };
    walk(original);
    return {located, historical_pins_without_locator: [], policy: 'Historical path/bytes/hash tuples retain their original repository identity. Locators are separate navigation data; no relocated source substitutes for a frozen source capsule.'};
  };
  const catalog = clone(catalogProposal), instance = clone(instanceProposal);
  Object.assign(catalog, {active_application: null, application_selection_record: null,
    interpretation: 'V3 is the clean causal research baseline. V4 is the broad clean-causal breakthrough; later versions retain scoped improvements. Mityane has no project-wide application encoder selection. The latest measured candidate is V14, whose exact failed recipe is closed.',
    historical_predecessor_application: clone(predecessor), mityane_application_selection: clone(selection), metadata_closeout: clone(binding)});
  const old = catalogProposal.fresh_research_candidate;
  Object.assign(catalog.fresh_research_candidate, {design_file: `${family}/CPB-v14.json`, physical_version_root: `${family}/v14`, measured_instance_file: TARGETS[1], instance_record: TARGETS[1],
    navigation_path_base: 'Mityane repository root', evidence_locators: locators(old), historical_navigation: {original_repository: 'cuwacunu_embedding',
      fields: Object.fromEntries(['design_file', 'physical_version_root', 'measured_instance_file', 'instance_record'].map(key => [key, old[key]])),
      source_identity_policy: 'Frozen source identities remain bound to original paths and source-manifest bytes; maintained locations have separate identities.'}});
  Object.assign(instance, {application_selection: null, historical_predecessor_application: clone(predecessor), mityane_application_selection: clone(selection),
    historical_source_identity: {original_repository: 'cuwacunu_embedding', version_root: instanceProposal.version_root, source_manifest: clone(historical), source_capsule: [clone(historical)],
      policy: 'Historical execution used this frozen source identity. Maintained relocated or edited code is not substituted for it.'},
    version_root: `${family}/v14`, navigation_path_base: 'Mityane repository root', metadata_closeout: clone(binding), evidence_locators: locators(instanceProposal),
    current_locations: {design_file: `${family}/CPB-v14.json`, measured_instance_file: TARGETS[1], local_run_root: 'code/embeddings/output/run',
      local_quality_root: 'code/embeddings/output/run/quality', local_engineering_root: 'code/embeddings/output/engineering', historical_runtime_root: '/opt/old/run'}});
  const summary = {kind: 'cpb_v14_postmigration_metadata_closeout_v1', date: '2026-10-18', instance: 'CPB-v14.market-01', protocol,
    status: 'reviewed_metadata_derivation_application_and_preservation_recorded_separately', scientific_conclusion: {clean_research_baseline: 'CPB-v3.market-01',
      broad_clean_causal_breakthrough: 'CPB-v4.market-01', historical_balanced_leader: 'RPB-v24.market-01.alt-01 native36', measured_candidate_established: true,
      numeric_passed: 3, numeric_total: 22, strict_new_scoped_quality_leaders: 0, advancement_passed: false, exact_v14_recipe_closed: true, mityane_application_selection: null,
      fresh_weights_versions: 'CPB-v3 through CPB-v14; source-design branches do not imply accumulated trained weights'},
    immutable_preapplication_inputs: clone(bindings), historical_proposals: clone(proposalPins), preserved_archived_ledger: clone(archivedLedger),
    ledger_data_policy: 'The reviewed ledger proposal is immutable historical DATA. Its leaders, scores, histories and failures remain there; the archived prior ledger is unchanged. No parallel mutable progress ledger is created.',
    maintained_records: TARGETS.slice(0, 2), future_receipt_locations: {application: `${attemptRelative}/application.json`, preservation: `${attemptRelative}/preservation.json`},
    future_receipt_policy: 'Locations name future records and claim no result. Those records bind actual applied bytes; this summary does not hash itself or any later receipt.',
    scope: {metadata_only: true, new_training: false, model_or_head_replay: false, head_refits: false, python_executed_by_protocol: false, checkpoint_selection: false},
    next_scientific_work: {status: 'requires_separate_saved_TRAIN_only_protocol', question: 'Window overlap, native covariance and actual saved shared-position adjoints; two captured batches do not establish a complete training-trajectory explanation.', validation_tensor_diagnosis: false, v15_selected: false}};
  return {catalog, instance, summary, catalogProposal, instanceProposal, ledgerProposal: {baseline: 'CPB-v3.market-01'}, aliases, bindings, proposalPins, attemptRelative, archivedLedger};
}

test('independent inspection accepts only explicit adaptations and preserves all input DATA', () => {
  const input = fixture(), before = jsonBytes(input), result = inspectRecords(input);
  assert.ok(result.checks > 30); assert.equal(result.unchanged_scientific_data, true);
  assert.equal(result.mityane_selection, null); assert.equal(result.advancement_passed, false);
  assert.ok(jsonBytes(input).equals(before));
});

test('adjacent unsafe seed edits, score edits and arbitrary original fields cannot hide in adaptations', () => {
  for (const mutate of [
    value => { value.instance.quality.seed = 9007199254740992n; },
    value => { value.catalog.fresh_research_candidate.fresh_seed = 18446744073709551615n; },
    value => { value.instance.quality.score += 1e-15; },
    value => { delete value.instance.unknown_original_field; },
    value => { value.instance.unknown_original_field.values[1] = 0; },
    value => { value.instance.new_scientific_claim = true; },
  ]) { const input = fixture(); mutate(input); assert.throws(() => inspectRecords(input), /Unchanged complete/); }
});

test('whole prior history, baseline and source-only ancestry are immutable', () => {
  for (const mutate of [
    value => { value.catalog.fresh_research_candidate_history[0].numeric_passed = 18; },
    value => { value.catalog.clean_causal_training_baseline.instance = 'CPB-v14.market-01'; },
    value => { value.instance.source_only_parent.trained_ancestor = true; },
    value => { value.instance.source_manifest.sha256 = 'a'.repeat(64); },
  ]) { const input = fixture(); mutate(input); assert.throws(() => inspectRecords(input), /Unchanged complete/); }
});

test('neither unknown adapted fields nor hidden promotion/selection is accepted', () => {
  for (const mutate of [
    value => { value.catalog.mityane_application_selection.instance = 'CPB-v4.market-01'; },
    value => { value.instance.current_locations.promoted = true; },
    value => { value.catalog.fresh_research_candidate.historical_navigation.original_repository = 'another'; },
    value => { value.instance.historical_predecessor_application.catalog.active_application = null; },
    value => { value.summary.scientific_conclusion.advancement_passed = true; },
    value => { value.summary.scientific_conclusion.broad_clean_causal_breakthrough = 'CPB-v3.market-01'; },
  ]) { const input = fixture(); mutate(input); assert.throws(() => inspectRecords(input)); }
});

test('future receipt locations remain acyclic and cannot be turned into passed receipts', () => {
  for (const mutate of [
    value => { value.summary.preservation_sha256 = hash; },
    value => { value.summary.future_receipt_locations.application = value.summary.maintained_records[0]; },
    value => { value.summary.scope.new_training = true; },
    value => { value.instance.metadata_closeout.research_execution_authorized = true; },
  ]) { const input = fixture(); mutate(input); assert.throws(() => inspectRecords(input)); }
});

test('locators require exact repository/path/length/hash and cannot rewrite historical pins', () => {
  for (const mutate of [
    value => { value.aliases[0].original.repository = 'cuwacunu_dev'; },
    value => { value.aliases[0].observed.sha256 = '0'.repeat(64); },
    value => { value.aliases.push({...clone(value.aliases[0]), current_path: 'another/location.json'}); },
    value => { value.instance.evidence_locators.located[0].historical_pin.path = 'rewritten/source.json'; },
  ]) { const input = fixture(); mutate(input); assert.throws(() => inspectRecords(input)); }
});

test('independent field diff reports additions/removals and exact escaped integer changes', () => {
  const rows = changedFields({'a/b': {'~key': 9007199254740993n}, removed: false}, {'a/b': {'~key': 9007199254740994n}, added: {n: 2}});
  assert.deepEqual(rows.map(row => row.pointer), ['/a~1b/~0key', '/removed', '/added']);
  assert.equal(rows[0].before, 9007199254740993n); assert.equal(rows[0].after, 9007199254740994n);
  assert.equal(rows[1].operation, 'remove'); assert.equal(rows[2].operation, 'add');
  assert.deepEqual(rows[2].after, {n: 2});
});

test('failed source admission and an existing receipt stop inspection before any output', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cpb-v14-inspect-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const relative = '.temp/embeddings/closeout/cpb-v14/attempt-001', attempt = path.join(root, relative);
  fs.mkdirSync(attempt, {recursive: true});
  const sourceReviewPath = `${relative}/source-review.json`, outputPath = `${relative}/staged-review.json`;
  fs.writeFileSync(path.join(root, sourceReviewPath), jsonBytes({passed: false, files: []}));
  assert.throws(() => inspect({root, attempt, sourceReviewPath, outputPath}), /Passed independent source review/);
  assert.equal(fs.existsSync(path.join(root, outputPath)), false);
  const original = jsonBytes({status: 'failed', reason: 'preserve earlier attempt'});
  fs.writeFileSync(path.join(root, outputPath), original);
  assert.throws(() => inspect({root, attempt, sourceReviewPath, outputPath}), /New review receipt/);
  assert.equal(sha256(fs.readFileSync(path.join(root, outputPath))), sha256(original));
});
