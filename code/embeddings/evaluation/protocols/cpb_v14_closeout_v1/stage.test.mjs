// Copyright (c) 2026 Waajacu. See ../../../../../LICENSE.
// Engineering fixtures only: no saved scientific experiment is replayed.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {deriveRecords, fieldAdaptations, sameValue, stage, TARGETS, ORIGINAL_PROPOSALS} from './stage.mjs';
import {jsonBytes, parseJSON, sha256} from '../../../../shared/artifacts/node/v1/files.mjs';

const copy = value => parseJSON(jsonBytes(value));
const HASH = 'f'.repeat(64);
function fixture() {
  const priorCatalog = {
    kind: 'causal_encoder_live_catalog_v1', date: '2026-10-18',
    clean_causal_training_baseline: {instance: 'CPB-v3.market-01'},
    fresh_research_candidate: {instance: 'CPB-v13.market-01', saved_seed: 18446744073709551615n,
      quality_score: 0.7624567890123456, checkpoint: {path: 'output/v13.pt', bytes: 123, sha256: HASH}},
    fresh_research_candidate_history: [{instance: 'CPB-v4.market-01', numeric_passed: 19, numeric_total: 22}],
    active_application: 'CPB-v3.market-01', application_selection_record: '../../cuwacunu_dev/selection.md',
    interpretation: 'The predecessor selected V3 for its development application.',
  };
  const sourceManifest = {path: 'output/source-manifest.json', bytes: 9, sha256: HASH};
  const catalogProposal = {...copy(priorCatalog), fresh_research_candidate: {
    instance: 'CPB-v14.market-01', numeric_passed: 3, numeric_total: 22, advancement_passed: false,
    design_file: 'CPB-v14.json', physical_version_root: 'v14',
    measured_instance_file: 'instances/CPB-v14.market-01.json',
    instance_record: 'code/encoders/causal_patch_bottleneck_mae/versions/instances/CPB-v14.market-01.json',
    report: {path: 'doc/CAUSAL_FRESH_V14_RESULTS.md', bytes: 7, sha256: HASH},
  }, fresh_research_candidate_history: [...copy(priorCatalog.fresh_research_candidate_history),
    copy(priorCatalog.fresh_research_candidate)]};
  const instanceProposal = {
    kind: 'causal_encoder_measured_instance_v1', instance: 'CPB-v14.market-01',
    clean_baseline: 'CPB-v3.market-01', advancement_passed: false,
    application_selection: 'CPB-v3.market-01', application_selection_changed: false,
    quality: {numeric_passed: 3, numeric_total: 22, strict_new_scoped_quality_leaders: 0,
      scientific_seed: 18014398509481987n, scaled_readout: 0.7624567890123456,
      failures: ['past_trend:linear_r2', 'validation:standardized_MAE']},
    version_root: 'code/encoders/causal_patch_bottleneck_mae/versions/v14',
    source_manifest: sourceManifest, source_capsule: [sourceManifest],
    prospective_descriptor: {path: 'code/encoders/causal_patch_bottleneck_mae/versions/CPB-v14.json', bytes: 9, sha256: HASH},
    local_run_root: 'output/runs/v14', local_quality_root: 'output/runs/v14/quality-001',
    local_engineering_root: 'output/engineering/v14', container_root: '/opt/cuwacunu/research/v14',
    actual_witnesses: [{path: '/opt/cuwacunu/research/v14/witness.pt', bytes: 9, sha256: HASH}],
  };
  const result = {catalogProposal, instanceProposal, ledgerProposal: {
    kind: 'clean_causal_encoder_research_ledger_v1', baseline: 'CPB-v3.market-01',
    stage_leaders: [{task: 'activity', instance: 'CPB-v13.market-01', score: 0.7624567890123456}],
  }, priorCatalog, proposalPins: ORIGINAL_PROPOSALS, bindings: {
    proposal_review: {path: '.temp/embeddings/closeout/cpb-v14/attempt-001/proposal-review.json', bytes: 9, sha256: HASH},
  }, aliases: [{original_repository: 'cuwacunu_embedding', original_path: sourceManifest.path,
    expected_bytes: 9, expected_sha256: HASH,
    current_path: 'code/embeddings/output/source-manifest.json', observed_bytes: 9, observed_sha256: HASH}],
  attemptRelative: '.temp/embeddings/closeout/cpb-v14/attempt-001',
  archivedLedger: {path: '.archive/embeddings/pre-cleanup-2026-10-18/doc/clean_causal_encoder_research_ledger.json',
    bytes: 123, sha256: HASH}};
  const collect = value => {
    if (!value || typeof value !== 'object') return;
    if (value.path && value.bytes && value.sha256 && !result.aliases.some(row => row.original_path === value.path)) {
      result.aliases.push({original_repository: 'cuwacunu_embedding', original_path: value.path,
        expected_bytes: value.bytes, expected_sha256: value.sha256,
        current_path: `fixture/${result.aliases.length}.json`, observed_bytes: value.bytes, observed_sha256: value.sha256});
    }
    Object.values(value).forEach(collect);
  };
  collect(instanceProposal); collect(catalogProposal.fresh_research_candidate);
  return result;
}

test('derivation preserves exact unsafe seeds, quality values, source pins and proposal inputs', () => {
  const input = fixture(), before = jsonBytes(input);
  const result = deriveRecords(input);
  assert.ok(jsonBytes(input).equals(before));
  assert.equal(result.instance.quality.scientific_seed, 18014398509481987n);
  assert.ok(sameValue(result.instance.quality, input.instanceProposal.quality));
  assert.ok(sameValue(result.instance.source_manifest, input.instanceProposal.source_manifest));
  assert.ok(sameValue(result.instance.actual_witnesses, input.instanceProposal.actual_witnesses));
  assert.equal(parseJSON(jsonBytes(result.instance)).quality.scientific_seed, 18014398509481987n);
});

test('predecessor V3 selection is preserved as history while Mityane selection is null', () => {
  const input = fixture(), result = deriveRecords(input);
  assert.equal(result.catalog.active_application, null);
  assert.equal(result.catalog.application_selection_record, null);
  assert.equal(result.instance.application_selection, null);
  assert.equal(result.catalog.historical_predecessor_application.catalog.active_application, 'CPB-v3.market-01');
  assert.equal(result.instance.historical_predecessor_application.instance.application_selection, 'CPB-v3.market-01');
  assert.equal(result.catalog.clean_causal_training_baseline.instance, 'CPB-v3.market-01');
});

test('V14 navigation is current while historical source identity remains explicit', () => {
  const input = fixture(), result = deriveRecords(input);
  assert.equal(result.catalog.fresh_research_candidate.measured_instance_file, TARGETS[1]);
  assert.equal(result.catalog.fresh_research_candidate.instance_record, TARGETS[1]);
  assert.equal(result.catalog.fresh_research_candidate.design_file,
    'code/embeddings/encoders/causal_patch_bottleneck_mae/versions/CPB-v14.json');
  assert.equal(result.instance.version_root,
    'code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v14');
  assert.equal(result.instance.historical_source_identity.version_root, input.instanceProposal.version_root);
  assert.ok(sameValue(result.instance.prospective_descriptor, input.instanceProposal.prospective_descriptor));
  assert.equal(result.instance.evidence_locators.located[0].current_path, 'code/embeddings/output/source-manifest.json');
  assert.equal(result.instance.evidence_locators.located[0].historical_pin.path, 'output/source-manifest.json');
});

test('all prior candidate history and the V13 candidate remain intact', () => {
  const input = fixture(), result = deriveRecords(input);
  assert.ok(sameValue(result.catalog.fresh_research_candidate_history,
    input.catalogProposal.fresh_research_candidate_history));
  assert.equal(result.catalog.fresh_research_candidate_history.at(-1).saved_seed, 18446744073709551615n);
  input.catalogProposal.fresh_research_candidate_history[0].numeric_passed = 18;
  assert.throws(() => deriveRecords(input), /exact prefix/);
});

test('changed or missing prior catalog records and duplicate history are rejected', () => {
  for (const mutate of [
    value => { value.catalogProposal.fresh_research_candidate_history.pop(); },
    value => { value.catalogProposal.fresh_research_candidate_history.push({instance: 'invented'}); },
    value => { value.catalogProposal.fresh_research_candidate_history.at(-1).saved_seed = 18446744073709551614n; },
    value => { value.catalogProposal.date = '2026-10-19'; },
    value => { value.priorCatalog.fresh_research_candidate.instance = 'CPB-v12.market-01'; },
  ]) {
    const input = fixture(); mutate(input);
    assert.throws(() => deriveRecords(input));
  }
});

test('closeout rejects changed scientific conclusions or a changed baseline', () => {
  for (const mutate of [
    value => { value.instanceProposal.quality.numeric_passed = 4; },
    value => { value.instanceProposal.quality.strict_new_scoped_quality_leaders = 1; },
    value => { value.catalogProposal.fresh_research_candidate.advancement_passed = true; },
    value => { value.instanceProposal.clean_baseline = 'CPB-v14.market-01'; },
    value => { value.ledgerProposal.baseline = 'CPB-v14.market-01'; },
  ]) {
    const input = fixture(); mutate(input);
    assert.throws(() => deriveRecords(input));
  }
});

test('alias matching needs the exact historical generation and rejects conflicting locations', () => {
  const input = fixture(); input.aliases[0].expected_sha256 = 'a'.repeat(64);
  assert.throws(() => deriveRecords(input), /unresolved exact historical/);
  input.aliases[0].expected_sha256 = HASH;
  input.aliases.push({...input.aliases[0], current_path: 'code/embeddings/output/conflicting.json'});
  assert.throws(() => deriveRecords(input), /conflicting current locator/);
});

test('matching bytes from another repository cannot satisfy a historical alias', () => {
  const input = fixture(); input.aliases[0].original_repository = 'another_repository';
  assert.throws(() => deriveRecords(input), /unresolved exact historical/);
  input.instanceProposal.source_manifest.repository = 'another_repository';
  input.instanceProposal.source_capsule[0].repository = 'another_repository';
  assert.equal(deriveRecords(input).instance.evidence_locators.located[0].original_repository, 'another_repository');
});

test('manifest records every actual proposal field adaptation and no untouched quality field', () => {
  const input = fixture(), result = deriveRecords(input);
  for (const [index, name] of ['catalog', 'instance'].entries()) {
    const source = index === 0 ? input.catalogProposal : input.instanceProposal;
    assert.ok(sameValue(result.adaptations[index].fields, fieldAdaptations(source, result[name])));
    assert.equal(result.adaptations[index].fields.some(row => row.pointer.startsWith('/quality/')), false);
  }
  assert.ok(result.adaptations[1].fields.some(row => row.pointer === '/application_selection' &&
    row.before === 'CPB-v3.market-01' && row.after === null));
  assert.ok(result.adaptations[1].fields.some(row => row.pointer === '/historical_source_identity' &&
    row.operation === 'add'));
});

test('summary binds only preapplication inputs and names later receipts without hash cycles', () => {
  const input = fixture(), {summary} = deriveRecords(input);
  assert.equal(summary.future_receipt_locations.application, `${input.attemptRelative}/application.json`);
  assert.equal(summary.future_receipt_locations.preservation, `${input.attemptRelative}/preservation.json`);
  assert.equal(summary.scientific_conclusion.mityane_application_selection, null);
  assert.equal(summary.scope.new_training, false);
  assert.equal(summary.scope.model_or_head_replay, false);
  assert.equal(summary.next_scientific_work.v15_selected, false);
  assert.ok(sameValue(summary.historical_proposals, ORIGINAL_PROPOSALS));
  assert.equal(Object.hasOwn(summary, 'application_sha256'), false);
  assert.equal(Object.hasOwn(summary, 'preservation_sha256'), false);
});

test('field derivation handles removals, escaped pointers, exact integers and negative zero', () => {
  const fields = fieldAdaptations({'a/b': {'~key': 9007199254740993n}, removed: false, n: -0},
    {'a/b': {'~key': 9007199254740994n}, added: null, n: 0});
  assert.ok(fields.some(row => row.pointer === '/a~1b/~0key' && row.before === 9007199254740993n && row.after === 9007199254740994n));
  assert.ok(fields.some(row => row.pointer === '/removed' && row.operation === 'remove'));
  assert.ok(fields.some(row => row.pointer === '/added' && row.operation === 'add'));
  assert.ok(fields.some(row => row.pointer === '/n' && Object.is(row.before, -0) && Object.is(row.after, 0)));
});

// Deliberately fails before proposal processing or writes. This verifies the
// real staging boundary against changed live bytes and failed prerequisites.
function stagingBoundary(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'cpb-v14-stage-fixture-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const attemptRelative = '.temp/embeddings/closeout/cpb-v14/attempt-001';
  const attempt = path.join(root, attemptRelative);
  fs.mkdirSync(attempt, {recursive: true});
  const write = (relative, content) => {
    const bytes = Buffer.isBuffer(content) ? content : jsonBytes(content);
    const filename = path.join(root, relative);
    fs.mkdirSync(path.dirname(filename), {recursive: true}); fs.writeFileSync(filename, bytes);
    return {path: relative, bytes: bytes.length, sha256: sha256(bytes)};
  };
  const stageRelative = 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/stage.mjs';
  const sharedRelative = 'code/shared/artifacts/node/v1/files.mjs';
  const sourcePins = [
    write(stageRelative, fs.readFileSync(fileURLToPath(new URL('./stage.mjs', import.meta.url)))),
    write(sharedRelative, fs.readFileSync(fileURLToPath(new URL('../../../../shared/artifacts/node/v1/files.mjs', import.meta.url)))),
  ];
  const prior = write(TARGETS[0], {live: 'before'});
  const targets = [{...prior, exists: true}, ...TARGETS.slice(1).map(name => ({path: name, exists: false}))];
  const aliases = write(`${attemptRelative}/aliases.json`, {aliases: []});
  const aliasReview = {passed: true, status: 'passed', alias_manifest: aliases};
  const proposalReview = {passed: true, status: 'passed', alias_manifest: aliases};
  const admission = {kind: 'cpb_v14_postmigration_closeout_admission_v1', artifact_root: attemptRelative,
    scope: {compatible_reviewed_catalog_instance_summary_application: true,
      new_model_training: false, old_encoder_or_head_replay: false, head_refits: false}, target_observations: targets};
  const options = {root, attempt, admissionPath: `${attemptRelative}/admission.json`,
    inputsPath: `${attemptRelative}/inputs.json`, aliasManifestPath: aliases.path,
    aliasVerificationPath: `${attemptRelative}/alias-verification.json`,
    proposalReviewPath: `${attemptRelative}/proposal-review.json`, sourceReviewPath: `${attemptRelative}/source-review.json`};
  const refresh = () => {
    write(options.admissionPath, admission); write(options.inputsPath, {targets, proposals: []});
    const aliasVerificationPin = write(options.aliasVerificationPath, aliasReview);
    proposalReview.alias_verification = aliasVerificationPin;
    write(options.proposalReviewPath, proposalReview);
    write(options.sourceReviewPath, {passed: true, files: sourcePins});
  };
  refresh();
  return {root, attempt, options, aliasReview, proposalReview, refresh, before: prior,
    assertUntouched() {
      assert.equal(sha256(fs.readFileSync(path.join(root, TARGETS[0]))), prior.sha256);
      assert.equal(fs.existsSync(path.join(root, TARGETS[1])), false);
      assert.equal(fs.existsSync(path.join(root, TARGETS[2])), false);
      assert.equal(fs.existsSync(path.join(attempt, 'staged')), false);
      assert.equal(fs.existsSync(path.join(attempt, 'staged-change-manifest.json')), false);
    }};
}

test('failed alias or proposal review stops before any staged or live write', t => {
  const fixture = stagingBoundary(t);
  fixture.aliasReview.passed = false; fixture.refresh();
  assert.throws(() => stage(fixture.options), /passed independent proposal and alias reviews/);
  fixture.assertUntouched();
  fixture.aliasReview.passed = true; fixture.proposalReview.passed = false; fixture.refresh();
  assert.throws(() => stage(fixture.options), /passed independent proposal and alias reviews/);
  fixture.assertUntouched();
});

test('stale before pin stops staging and preserves concurrent live bytes', t => {
  const fixture = stagingBoundary(t);
  const changed = jsonBytes({live: 'concurrent edit'});
  fs.writeFileSync(path.join(fixture.root, TARGETS[0]), changed);
  assert.throws(() => stage(fixture.options), /target differs from current before pin/);
  assert.ok(fs.readFileSync(path.join(fixture.root, TARGETS[0])).equals(changed));
  assert.equal(fs.existsSync(path.join(fixture.attempt, 'staged')), false);
  assert.equal(fs.existsSync(path.join(fixture.root, TARGETS[1])), false);
  assert.equal(fs.existsSync(path.join(fixture.root, TARGETS[2])), false);
});

test('an expected absent target cannot be a dangling symlink', t => {
  const fixture = stagingBoundary(t);
  const target = path.join(fixture.root, TARGETS[1]);
  fs.mkdirSync(path.dirname(target), {recursive: true});
  fs.symlinkSync(path.join(fixture.root, 'missing-file'), target);
  assert.throws(() => stage(fixture.options), /expected absent target already exists/);
  assert.equal(fs.lstatSync(target).isSymbolicLink(), true);
  assert.equal(fs.existsSync(path.join(fixture.attempt, 'staged')), false);
});

test('stale source review and a mismatched alias manifest cannot stage records', t => {
  const fixture = stagingBoundary(t);
  const source = path.join(fixture.root, 'code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/stage.mjs');
  const bytes = fs.readFileSync(source); fs.appendFileSync(source, '\n// changed\n');
  assert.throws(() => stage(fixture.options), /source review stale/);
  fixture.assertUntouched();
  fs.writeFileSync(source, bytes);
  fixture.aliasReview.alias_manifest = {...fixture.aliasReview.alias_manifest, sha256: '0'.repeat(64)};
  fixture.refresh();
  assert.throws(() => stage(fixture.options), /alias verification does not bind/);
  fixture.assertUntouched();
});
