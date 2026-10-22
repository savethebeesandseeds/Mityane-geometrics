// Binding integrity only: no matching, estimator, model or outcome aggregation.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {admitFiles, readAdmitted, parseJSON, sha256, safeRelative, requireCondition as require}
  from '../../../../../shared/artifacts/node/v1/files.mjs';
import {baselineRecords} from '../../../../../data/observations/open_benchmarks/nsw_wooldridge_v1/materialize.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../../../../..');
const json = bytes => parseJSON(bytes, {duplicates: 'reject', integers: 'number', allowBOM: false});

export function checkDefinition(binding) {
  require(binding.schema_version === 'mityane.counterfactuals.profile-bindings/v1' && binding.revision === '1.0.0', 'unsupported binding version');
  require(binding.source.rows === 445 && binding.source.assignment_counts['1'] === 185 && binding.source.assignment_counts['0'] === 260,
    'actual source population differs');
  const r = binding.representation;
  require(r.provider_id === 'nsw-re74-identity/v1' && JSON.stringify(r.input_fields) === '["re74"]' &&
    JSON.stringify(r.native_shape) === '[1]' && r.geometry === 'absolute_distance/v1' && r.normalization === 'none',
  'prohibited representation input or incompatible geometry');
  require(r.readout === 'none' && r.fitted_preprocessing === 'not_fitted' && r.fit_population === 'not_fitted' &&
    r.selection_population === 'not_fitted' && r.synthetic_ols_dependency === false, 'unbound fitted or OLS dependency');
  require(binding.temporal.mode === 'retrospective' && binding.temporal.available_at === null &&
    binding.temporal.as_of_claim === false && binding.partitions.protected_fresh_assessment === false,
  'unjustified temporal or protected-assessment claim');
  require(binding.execution.implementation_registered === false && binding.execution.operation_runs.length === 0 &&
    binding.execution.scientific_verdict === 'not_evaluated' && binding.execution.admission === 'not_admitted',
  'binding must not invent execution or scientific evidence');
  for (const field of ['scientific_execution', 'matching', 'effect_estimation', 'training', 'external_embedding_api', 'row_data_publication'])
    require(binding.permissions[field] === false, 'permission exceeds source-binding scope');
  require(binding.instances.length === 2 && new Set(binding.instances.map(i => i.id)).size === 2, 'two distinct instances required');
  const [moc, iie] = binding.instances;
  require(moc.profile_id === 'OR06-MOC' && moc.profile_revision === '0.1.0' && moc.claim_level === 'observational_contrast' &&
    moc.requested_focal_units === 185 && moc.candidate_controls === 260 &&
    JSON.stringify(moc.matching.input_fields) === '["re74"]' && moc.matching.outcomes_consumed === false,
  'observational role, population or matching information differs');
  require(iie.profile_id === 'OR06-IIE' && iie.profile_revision === '0.1.0' && iie.claim_level === 'identified_intervention_effect' &&
    iie.requested_units === 445 && iie.matching === null && iie.assignment_field === 'train', 'intervention binding differs');
  require(iie.identification.allocation_probabilities === null && iie.identification.strata_or_cluster_ids === null &&
    iie.identification.observed_arm_fraction_is_known_assignment_probability === false &&
    iie.uncertainty.state === 'not_admitted' && iie.identification.untestable_assumptions.length >= 4,
  'invented randomization details or omitted identification limitations');
  for (const instance of binding.instances) require(instance.outcome.field === 're78' && instance.outcome.units === 'thousands of real dollars',
    'outcome role or units differ');
  return binding;
}

export function checkBindings(binding, root = ROOT) {
  checkDefinition(binding);
  const verified = new Map();
  const readPin = pin => {
    require(pin.path === safeRelative(pin.path) && !pin.path.includes('\\') && Number.isSafeInteger(pin.bytes), 'invalid artifact path or size');
    const previous = verified.get(pin.path);
    require(!previous || previous.sha256 === pin.sha256 && previous.bytes === pin.bytes, 'conflicting artifact pin');
    if (previous) return previous.content;
    const content = readAdmitted(admitFiles([path.resolve(root, pin.path)], {root})[0]);
    require(content.length === pin.bytes && sha256(content) === pin.sha256, `artifact bytes changed: ${pin.path}`);
    verified.set(pin.path, {...pin, content}); return content;
  };
  const visit = value => {
    if (!value || typeof value !== 'object') return;
    if (!Array.isArray(value) && Object.hasOwn(value, 'path') && Object.hasOwn(value, 'sha256') && Object.hasOwn(value, 'bytes')) readPin(value);
    for (const child of Object.values(value)) if (child && typeof child === 'object') visit(child);
  };
  visit(binding);
  const imported = json(readPin(binding.source.import_receipt));
  const materialized = json(readPin(binding.source.materialization_receipt));
  visit(imported.source_closure); visit(materialized.source_closure);
  require(imported.passed === true && materialized.passed === true && imported.dataset_id === binding.source.dataset_id &&
    imported.source.source_commit === binding.source.source_commit, 'source receipt identity differs');
  require(imported.matching_executed === false && imported.effect_estimated === false &&
    materialized.matching_executed === false && materialized.effect_estimated === false &&
    materialized.outcome_aggregates_computed === false, 'preparation contains unbound scientific execution');
  const actual = binding.representation.artifacts;
  require(JSON.stringify(actual) === JSON.stringify(materialized.artifacts), 'materialized artifact map differs');
  const provider = json(readPin(actual.provider)), space = json(readPin(actual.space_ref));
  const roster = json(readPin(actual.roster)), representation = json(readPin(actual.representations));
  require(provider.input_field === 're74' && provider.readout === 'none' && provider.geometry.id === 'absolute_distance/v1' &&
    space.provider_artifact.sha256 === actual.provider.sha256 && space.suitability_evidence === 'unassessed; descriptor binding only',
  'provider or SpaceRef binding differs');
  const expected = baselineRecords(readPin(binding.source.snapshot));
  require(roster.rows.length === 445 && representation.rows.length === 445 && expected.length === 445, 'retained roster length differs');
  expected.forEach((row, index) => {
    const observed = roster.rows[index], vector = representation.rows[index];
    require(observed.unit_id === row.unit_id && observed.source_row_ordinal === row.source_row_ordinal && observed.assignment === row.assignment &&
      vector.observation_ref.unit_id === row.unit_id && vector.observation_ref.source_snapshot_sha256 === binding.source.snapshot.sha256 &&
      JSON.stringify(vector.values) === JSON.stringify(row.values), 'row, baseline descriptor or source join differs');
  });
  return {schema_version: 'mityane.counterfactuals.profile-binding-check/v1', status: 'passed', profile_instances: 2,
    source_rows: expected.length, artifact_files_verified: verified.size, source_snapshot_sha256: binding.source.snapshot.sha256,
    scientific_execution: false, operation_implementations_registered: false, effect_estimated: false};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  require(process.platform === 'linux' && ROOT === '/mityane' && fs.existsSync('/.dockerenv'), 'use the managed Linux container');
  const bytes = readAdmitted(admitFiles([path.join(HERE, 'bindings.json')], {root: ROOT})[0]);
  process.stdout.write(JSON.stringify(checkBindings(json(bytes)), null, 2) + '\n');
}
