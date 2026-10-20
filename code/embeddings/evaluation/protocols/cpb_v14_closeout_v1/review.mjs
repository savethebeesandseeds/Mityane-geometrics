// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../LICENSE.
// Independent DATA review of already-generated V14 proposals. No proposer imports,
// model calls, tensor reads, numerical replay, or maintained-metadata writes.
import { readFile, lstat, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual as same } from 'node:util';
import { createIdentityResolver } from './aliases.mjs';
import { parseJSON, jsonBytes } from '../../../../shared/artifacts/node/v1/files.mjs';

const SELF = fileURLToPath(import.meta.url);
const ALIAS_SOURCE = fileURLToPath(new URL('./aliases.mjs', import.meta.url));
const JSON_SOURCE = fileURLToPath(new URL('../../../../shared/artifacts/node/v1/files.mjs', import.meta.url));
const FAMILY = 'code/encoders/causal_patch_bottleneck_mae/versions/';
const METHOD = 'CPB-v14.market-01';
const BASELINE = 'CPB-v3.market-01';
const VERSIONS = Array.from({ length: 12 }, (_, i) => i + 3);
const PAYLOAD = 'code/embeddings/output/engineering/causal-fresh-v14-finalization/attempt-001/migration-snapshot-001/payload';
const HISTORICAL_FINALIZATION = '/opt/cuwacunu/research/causal-fresh-v14-finalization/attempt-001';
const ORIGINAL_REVIEWER = { bytes: 46329, sha256: 'f323c74d3ccee0dc56cdf6f8db930ecfb3787b00173295027c872f8aab36caa9' };
const HISTORY_KEYS = ['candidates', 'stage_scorecards', 'reconstruction_stage_scorecards', 'leader_history',
  'displaced_leaders', 'candidate_status_history', 'direction_history', 'stage_leader_update_history', 'reconstruction_leader_history'];
const TEMPORAL = {
  temporal_pooling_policy: 'original-position-nested-trailing-local-global-half-mixture-mean4-regularized-stddev4-v1',
  temporal_pooling_geometry: [4, 16, 32, 8], temporal_pooling_trailing_lengths: [4, 8, 12, 16], temporal_pooling_mix: [0.5, 0.5],
  temporal_moment_policy: 'componentwise-centered-variance-two-sum-mean-live-attention-regularized-sqrt-v1',
  temporal_export_layout: 'per-trailing-window-eight-rows:first4-mean;last4-zero-based-regularized-stddev-v1',
  temporal_moment_partition: [4, 4],
  temporal_dispersion_transform_policy: 'sqrt-variance-plus-tau-minus-sqrt-tau-F32-add-sqrt-subtract-v1',
  temporal_dispersion_regularizer: 0.0009765625, empty_local_bin_policy: 'zero-whole-export-slice-after-bias-v1',
  registered_learned_tensors: 76, mandatory_identifiable_learned_tensors: 75,
  data_gradient_gauge_exemption: 'core.encoder.level.base.backbone.pool_score.bias',
};
const OBJECTIVE = { objective_policy: 'huber1-mean-plus-detached-error-gaussian-variance-v1', mean_huber_delta: 1,
  loss_coefficients: [1, 1], variance_error_detached: true, reporting_nll_backpropagated: false,
  direct_gradient_routes: 'mean->parent32+mean4;variance->uncertainty4;no-direct-cross-paths;shared-global-clip1' };
const GRADIENT = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1';
const PROOF_FLAGS = ['mean4_regularized_stddev4_export_layout', 'full_attention_dependent_regularized_moment_adjoints',
  'live_dispersion_score_gradient', 'raw_variance_root_dispersion_trace', 'zero_tiny_large_analytic_dispersion_adjoints',
  'supported_nonfinite_and_variance_overflow_refused', 'absent_local_huge_singleton_sanitized', 'captured_F32_residual_correction',
  'raw_root_absent_offset_exception', 'nested_trailing_supports_fixed_edge16', 'per_window_local_weights_captured',
  'longest_local_equals_global_two_sum', 'packed_padding_membership_invariance'];
const LOCAL_PARAMETERS = ['temporal_export_bias', 'temporal_export_weight', 'temporal_pool_positions'];

// Explicit obligation map to the frozen independent reviewer. Line groups cover
// every original assertion; environment/output transport is adapted separately.
export const OBLIGATIONS = [
  ['frozen-bindings', '69-100', 'Exact card,49 role pins,successful unapplied receipt,3 bounded output pins'],
  ['history-preservation', '101-131', 'V3 baseline,historical DEV selection,no winner,whole historical prefixes,prior leaders/catalog'],
  ['historical-controls', '132-164', 'Prospective V3-V13 controls,8 separate joins,partial20:40 versus full40 semantics'],
  ['stagecards', '165-194', 'Exact V14 initial/final market,timing,reconstruction scalars and counters'],
  ['leaders', '195-248', '16 final-score scopes,4 reconstruction scopes,stable ties,complete displaced records,no TRAIN-NLL gate'],
  ['trust-objective-geometry', '249-282', 'All card assertions,objective/gradient,nested geometry,75/76 gauge,shared clipping'],
  ['proof-and-learning', '283-332', '821 full-file/request proof contract,13 fixture flags,53/18/18/4 counts,8192 events,Hadamard scope'],
  ['protocol-outcome', '333-395', '22 exact prospective criteria,4ace8b protocol,producer pending,independent trust,13 pairings'],
  ['bounded-moment-audit', '396-486', '8 moment groups,2 same-forward witnesses,local adjoints,single clip,gauge descriptive,scope limits'],
  ['ancestry-bindings-counts', '487-527', 'Fresh source-only V13 ancestry,full moment record,evidence joins,goal flag,audit counts'],
  ['instance-source-closure', '528-578', 'Physical V14 roots,fresh counters,145/7/11 closure,source/binary bytes,freeze/preflight/mirror'],
  ['final-invariants', '579-594', 'Numeric/trust result,5 checkpoint identities,immutable descriptor,no replay/live writes'],
];

const finite = value => typeof value === 'number' && Number.isFinite(value);
const integer = value => Number.isInteger(value);
const pick = (record, keys) => Object.fromEntries(keys.map(key => [key, record[key]]));
const pinKeys = ['path', 'bytes', 'sha256'];
const bytesKeys = ['bytes', 'sha256'];
const scope = row => JSON.stringify(['dataset_id', 'task', 'input_policy', 'view', 'metric'].map(key => row[key] ?? null));
export function stableBest(rows, direction = 'max') {
  if (!rows.length || !rows.every(row => finite(row.score))) throw new Error('Finite nonempty score population required');
  return rows.reduce((best, row) => (direction === 'min' ? row.score < best.score : row.score > best.score) ? row : best);
}
export function safeRelative(root, relative) {
  if (typeof relative !== 'string' || !relative || path.posix.isAbsolute(relative) || path.win32.isAbsolute(relative)
      || relative.split(/[\\/]/).some(part => part === '..' || part === '' || part === '.')) throw new Error('Relative bounded metadata path required');
  const result = path.resolve(root, relative);
  if (!result.startsWith(path.resolve(root) + path.sep)) throw new Error('Metadata path escaped leaf');
  return result;
}
// Preserve Python's exact integer semantics (including uint64 seeds) and its
// last duplicate-key policy. No integer identity is rounded to a Number.
async function readJSON(file) { return parseJSON(await readFile(file), { duplicates: 'last', integers: 'bigint' }); }
async function rejectRedirectedAncestors(file) {
  for (let parent = path.dirname(path.resolve(file)); ; parent = path.dirname(parent)) {
    try {
      const stat = await lstat(parent);
      if (stat.isSymbolicLink() || !stat.isDirectory()) throw new Error(`Redirected output ancestor: ${parent}`);
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (parent === path.dirname(parent)) break;
  }
}
async function pin(file) {
  const stat = await lstat(file);
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`Existing unlinked metadata file required: ${file}`);
  const data = await readFile(file);
  return { path: file, bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') };
}

/** Pure DATA review, independently reconstructing conclusions from frozen inputs. */
export function reviewData({ card, inputs: I, byId: B, receipt, ledger, current, instance }) {
  let checks = 0;
  const groups = {};
  let group = 'history-preservation';
  const check = (ok, message) => { checks++; groups[group] = (groups[group] ?? 0) + 1; if (!ok) throw new Error(`${group}: ${message}`); };
  const eq = (actual, expected, message) => check(same(actual, expected), message);
  const fields = (actual, expected, message) => { for (const [key, value] of Object.entries(expected)) eq(actual[key], value, `${message}: ${key}`); };
  const at = (record, selectors) => {
    check(Array.isArray(selectors) && selectors.length > 0, 'Explicit actual receipt selector');
    for (const selector of selectors) { check(record !== null && Object.hasOwn(record, selector), 'Existing receipt selector'); record = record[selector]; }
    return record;
  };
  const old = I.ledger;
  check(ledger.baseline === BASELINE && old.baseline === BASELINE && current.active_application === BASELINE,
    'Preserve V3 baseline and immutable historical application field');
  eq(ledger.baseline_checkpoint, old.baseline_checkpoint, 'Complete baseline checkpoint retained');
  check(ledger.overall_winner === false && old.overall_winner === false, 'No overall winner');
  for (const key of HISTORY_KEYS) eq(ledger[key].slice(0, old[key].length), old[key], `Whole historical prefix ${key}`);
  check(ledger.candidates.length === old.candidates.length + 1 && ledger.candidates.filter(row => row.instance === METHOD).length === 1, 'Exactly one measured V14 candidate');
  for (const row of old.stage_leaders) check(ledger.leader_history.some(entry => same(entry, row) || same(entry.record, row)), 'Complete prior quality leader retained');
  check(ledger.reconstruction_leader_history.slice(old.reconstruction_leader_history.length).some(row => same(row.records, old.reconstruction_stage_leaders)), 'All four prior reconstruction records retained');
  const previous = I.catalog;
  eq(current.fresh_research_candidate_history.slice(0, -1), previous.fresh_research_candidate_history, 'Catalog history prefix');
  eq(current.fresh_research_candidate_history.at(-1), previous.fresh_research_candidate, 'Whole V13 current entry appended');
  for (const key of Object.keys(previous)) if (!['fresh_research_candidate', 'fresh_research_candidate_history', 'date'].includes(key)) eq(current[key], previous[key], `Other catalog field unchanged ${key}`);
  check(old.stage_scorecards.length === 22 && old.reconstruction_stage_scorecards.length === 22, '22 protected prior stagecards');
  check(ledger.stage_scorecards.length === 24 && ledger.reconstruction_stage_scorecards.length === 24, '24 initial/final stagecards');

  group = 'historical-controls';
  const controls = I.experiment_card.comparison_control;
  for (const v of VERSIONS.slice(0, -1)) eq(pick(B[`v${v}_table`], bytesKeys), pick(controls[`v${v}`], bytesKeys), `Exact protected V${v} table`);
  for (const v of [6, 7, 8, 9, 10, 11, 12, 13]) {
    const historical = I[`v${v}_establishment`];
    eq(pick(B[`v${v}_establishment`], bytesKeys), pick(controls[`v${v}_establishment`], bytesKeys), `Exact protected V${v} trust join`);
    check(I[`v${v}_table`].trust_protocol_admission_passed === false && historical.trust_admission_passed === true, `V${v} producer pending/separate trust`);
    eq(pick(historical.evidence.table_rows, bytesKeys), pick(B[`v${v}_table`], bytesKeys), `V${v} exact trust/table binding`);
    check(historical.initial_pairing_count === 13, 'Historical13 pairings');
    if ([8, 9, 11, 12, 13].includes(v)) fields(historical, { initial_partial_native20_40_pairings_passed: true,
      initial_pairing_native_coordinate_slice: [20, 40], whole_native40_initial_equality_claimed: false }, `V${v} partial20:40 semantics`);
    else check(historical.initial_full40_pairings_passed === true, `V${v} full40 semantics`);
  }

  group = 'stagecards';
  for (const [index, stage] of ['initial', 'final'].entries()) {
    const label = METHOD + (stage === 'initial' ? '.init' : '');
    const saved = I.v14_table;
    const qualityCard = ledger.stage_scorecards.at(-2 + index);
    const queryCard = ledger.reconstruction_stage_scorecards.at(-2 + index);
    for (const row of [qualityCard, queryCard]) fields(row, { instance: METHOD, checkpoint_stage: stage, source_updates: 0, causal_updates: stage === 'initial' ? 0 : 2048 }, 'Ordered V14 stagecard counters');
    eq(qualityCard.market_rows, saved.market_rows.filter(row => row.method === label), 'Four actual market rows');
    check(qualityCard.synthetic_panels.length === 4 && saved.synthetic.panels.length === 4, 'Four actual timing panels');
    for (const [n, panel] of saved.synthetic.panels.entries()) {
      const rows = panel.rows.filter(row => row.method === label);
      check(rows.length === 1, 'One actual timing row');
      eq(qualityCard.synthetic_panels[n].row, rows[0], 'Exact saved timing row');
      fields(qualityCard.synthetic_panels[n], Object.fromEntries(Object.entries(panel).filter(([key]) => key !== 'rows')), 'Exact panel metadata');
    }
    for (const split of ['train', 'validation']) {
      const savedQueries = saved.fixed_queries[stage][split];
      const keys = ['standardized_fixed_query_MAE', 'Gaussian_NLL_F64', 'valid_rows', 'target_cells',
        ...['mean_Huber_F64', 'variance_NLL_F64', 'joint_Gaussian_NLL_F64', 'total_objective_F64'].filter(key => Object.hasOwn(savedQueries, key))];
      eq(Object.keys(queryCard.queries[split]).sort(), keys.sort(), 'Exact fixed-query scalar schema');
      eq(queryCard.queries[split], pick(savedQueries, keys), 'Exact saved fixed-query scalars');
    }
  }

  group = 'leaders';
  const quality = (leader, version, stage) => {
    const table = I[`v${version}_table`];
    const label = `CPB-v${version}.market-01${stage === 'initial' ? '.init' : ''}`;
    let rows;
    if (leader.dataset_id === 'MKT-HIST-2') rows = table.market_rows.filter(row => row.method === label && row.task === leader.task);
    else {
      check(leader.dataset_id === 'TEMPO-5', 'Declared timing dataset');
      const panels = table.synthetic.panels.filter(panel => panel.input_policy === leader.input_policy && panel.view === leader.view);
      check(panels.length === 1, 'Unique timing panel');
      rows = panels[0].rows.filter(row => row.method === label);
    }
    check(rows.length === 1 && rows[0].checkpoint_stage === stage && finite(rows[0][leader.metric]), 'One finite exact saved stage score');
    return { method: label, score: rows[0][leader.metric], checkpoint_stage: stage };
  };
  const oldScopes = new Map(old.stage_leaders.map(row => [scope(row), row]));
  check(oldScopes.size === 16 && ledger.stage_leaders.length === 16, 'Sixteen quality scopes');
  eq(new Set(ledger.stage_leaders.map(scope)), new Set(oldScopes.keys()), 'Same exact quality scopes');
  const replacements = [];
  for (const leader of ledger.stage_leaders) {
    const original = oldScopes.get(scope(leader));
    const scores = VERSIONS.map(v => quality(leader, v, 'final'));
    const bestOld = stableBest(scores.slice(0, -1));
    const best = stableBest(scores);
    eq(pick(original, ['score', 'method']), pick(bestOld, ['score', 'method']), 'Old strongest V3-V13 final leader');
    eq(pick(leader, ['score', 'method']), pick(best, ['score', 'method']), 'Strongest V3-V14 final leader; chronological ties');
    eq(leader.stage_scores, [...original.stage_scores, quality(leader, 14, 'initial'), quality(leader, 14, 'final')], 'Whole prior plus two saved stage scores');
    eq(leader.all_final_scores, scores.map((row, i) => ({ ...row, table_source: pick(B[`v${VERSIONS[i]}_table`], pinKeys) })), 'Chronological scores and exact table pins');
    if (best.score > original.score) { check(best.method === METHOD, 'Only V14 strictly displaces leader'); replacements.push(scope(leader)); }
  }
  check(receipt.strict_new_quality_leaders.length === replacements.length, 'Dynamic new leader count');
  eq(new Set(receipt.strict_new_quality_leaders.map(scope)), new Set(replacements), 'Dynamic changed scopes');
  for (const change of receipt.strict_new_quality_leaders) {
    const original = oldScopes.get(scope(change));
    fields(change, { former_complete_record: original, former_method: original.method, former_score: original.score, new_method: METHOD, strictly_better: true }, 'Complete displaced record');
    check(change.new_score > original.score, 'Strict actual improvement');
  }
  fields(ledger.stage_leader_update, { strictly_improved_scopes: replacements.length, earlier_scopes_retained: 16 - replacements.length }, 'Dynamic leader summary');
  check(ledger.reconstruction_stage_leaders.length === 4, 'Four reconstruction scopes');
  for (const leader of ledger.reconstruction_stage_leaders) {
    check(['standardized_fixed_query_MAE', 'Gaussian_NLL_F64'].includes(leader.metric), 'Existing query metric');
    const scores = VERSIONS.map(v => ({ score: I[`v${v}_table`].fixed_queries.final[leader.split][leader.metric], method: `CPB-v${v}.market-01` }));
    eq(pick(leader, ['score', 'method']), stableBest(scores, 'min'), 'Lowest final reconstruction; chronological ties');
    check(leader.numeric_advancement_floor === !(leader.split === 'train' && leader.metric === 'Gaussian_NLL_F64'), 'TRAIN Gaussian NLL descriptive only');
  }

  group = 'trust-objective-geometry';
  const contract = card.evidence_contract;
  for (const [name, assertion] of Object.entries(contract.assertions)) eq(at(I[assertion.input], assertion.path), assertion.equals, `Actual mandatory trust ${name}`);
  for (const role of ['training', 'comparison', 'engineering', 'gradient', 'initial_proof', 'final_proof', 'initialization']) {
    fields(I[role], OBJECTIVE, `Unchanged objective ${role}`);
    check(role === 'initialization' ? !Object.hasOwn(I[role], 'gradient_policy') : I[role].gradient_policy === GRADIENT, `Gradient selector ${role}`);
  }
  for (const role of ['training', 'comparison', 'engineering', 'initial_proof', 'final_proof', 'initialization']) {
    check(!Object.hasOwn(I[role], 'temporal_pooling_partition'), `No legacy disjoint geometry ${role}`);
    fields(I[role], TEMPORAL, `Nested temporal/gauge contract ${role}`);
  }
  fields(I.training, { shared_global_clip: true, optimizer_independence_claimed: false, all_mandatory_identifiable_tensors_learned: true, mandatory_identifiable_tensor_value_capacity: 294751 }, 'Shared clip and tensor-event capacity');
  check(I.gradient.shared_clip_coupling_witness === true, 'Actual detached-error fixture');
  check(I.engineering.typed_negative_cases === 821 && I.experiment_card.engineering_typed_negative_cases === 821, '821 strict typed cases');

  group = 'proof-and-learning';
  const archiveContract = I.experiment_card.engineering_archive_contract;
  fields(archiveContract, { format: 'full_refusal_archives', all_full_negative_files_preserved: true }, 'Full refusal archives');
  for (const [key, value] of Object.entries({ typed_negative_full_archive_files: 817, typed_negative_direct_loader_requests: 4, typed_negative_compact_archive_files: 0, typed_negative_restored_target_controls: 0, typed_negative_legacy_conflict_files: 5 })) {
    check(archiveContract[key] === value && I.engineering[key] === value, `821=817+4 proof scope ${key}`);
  }
  for (const [key, value] of Object.entries(I.experiment_card.mechanism_proof_contract)) if (typeof value === 'boolean') eq(I.gradient[key], value, `Mechanism fixture ${key}`);
  fields(I.temporal_proof, I.experiment_card.temporal_pooling_proof_contract.required_proof_fields, 'Temporal fixture predicate');
  fields(I.temporal_proof, { direct_pool_kernel_calls: 53, F64_finite_difference_coordinates: 18, F32_finite_difference_coordinates: 18, isolated_autograd_grad_calls: 4, transform_derivative_upper_bound: 16 }, '53/18/18/4 fixture scope');
  for (const flag of PROOF_FLAGS) check(I.temporal_proof[flag] === true, `Thirteen temporal proof flags ${flag}`);
  const learning = I.temporal_learning;
  fields(learning, { status: 'passed', updates: 2048, group_rows: 8192, extra_encoder_calls: 0, weight_decay_counted_as_data_gradient: false }, 'Finer temporal learning');
  eq(learning.gauge.name, TEMPORAL.data_gradient_gauge_exemption, 'Softmax gauge');
  eq(pick(I.training.temporal_pooling_learning, bytesKeys), pick(B.temporal_learning, bytesKeys), 'Trajectory binds temporal learning');
  const between = learning.between_bin_positions;
  fields(between, { basis: 'hadamard-full16-signs-div4-v1', basis_signs: [[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]], full_position_divisor: 4, position_bin_width: 4, rows: 2048 }, 'Hadamard event basis');
  check(between.gradient_steps.length === 3 && between.movement_steps.length === 3 && [...between.gradient_steps, ...between.movement_steps].every(n => integer(n) && n > 0 && n <= 2048), 'Three positive gradient/movement EVENTS');
  check(between.final_contrast_changes.length === 3 && between.final_contrast_changes.every(finite) && finite(between.final_projection_change_l2) && between.final_projection_change_l2 > 0, 'Overall projection change; no per-coefficient gate');

  group = 'protocol-outcome';
  const conditions = I.comparison.numeric_conditions;
  check(conditions.length === 22 && new Set(conditions.map(row => row.id)).size === 22 && conditions.every(row => typeof row.passed === 'boolean'), '22 unique Boolean decisions');
  const declaredConditions = I.experiment_card.advancement.criteria;
  check(declaredConditions.length === 22, '22 prospective criteria');
  eq(conditions.map((row, i) => pick(row, Object.keys(declaredConditions[i]))), declaredConditions, 'Exact prospective condition identity/floors/source/rules');
  const goal = I.experiment_card.advancement.goal_protocol;
  check(goal.sha256 === '4ace8b79a496ff24440ef1a269b63b45b746624f51dbd734e1b9c013b8a93245', 'Frozen prospective protocol');
  eq(pick(I.comparison.goal_protocol, bytesKeys), pick(goal, bytesKeys), 'Comparison/protocol binding');
  const passed = conditions.filter(row => row.passed).length;
  const advancement = passed === 22 && I.comparison.full_coverage === true;
  fields(I.comparison, { numeric_passed: passed, full_coverage: true, trust_admission_passed: false, overall_establishment_passed: false, advancement_conditions_passed: false }, 'Frozen pending producer');
  eq(Object.keys(I.comparison.mandatory_postmeasurement_trust).sort(), ['initial_native20_40_and_statistics_pairings', 'independent_training_audit', 'independent_quality_audit'].sort(), 'Exactly three pending trust roles');
  check(Object.values(I.comparison.mandatory_postmeasurement_trust).every(value => value === 'pending'), 'Frozen pending fields remain pending');
  const established = I.establishment;
  fields(established, { trust_admission_passed: true, numeric_passed: passed, overall_establishment_passed: advancement, overall_winner: false, winner_selection_made: false }, 'Independent establishment outcome');
  for (const flag of PROOF_FLAGS) check(established[flag] === true, `Separate establishment temporal flag ${flag}`);
  fields(established, { temporal_pooling_direct_kernel_calls: 53, temporal_pooling_F64_finite_difference_coordinates: 18, temporal_pooling_F32_finite_difference_coordinates: 18, temporal_pooling_isolated_autograd_grad_calls: 4, engineering_typed_negative_cases: 821 }, 'Independent fixture counts');
  fields(established, { initial_partial_native20_40_pairings_passed: true, saved_initial_final_endpoint_proofs_bound: true, historical_saved_control_count: 11, historical_separate_trust_join_count: 8, comparison_method_count: 13, comparison_market_rows: 52, initial_pairing_count: 13, initial_pairing_feature_archives: 8, initial_pairing_statistic_archives: 5, initial_pairing_native_coordinate_slice: [20, 40], whole_native40_initial_equality_claimed: false }, 'Partial20:40 DATA / whole-statistic semantics');
  fields(established, { ...TEMPORAL, all_four_export_row_groups_and_position_contrasts_verified: true, all_three_between_bin_position_directions_verified: true, between_bin_overall_final_projection_verified: true, individual_final_coefficient_nonzero_gate: false }, 'Four local groups and three position events');

  group = 'bounded-moment-audit';
  const audit = I.training_audit;
  const moment = audit.temporal_pooling_saved_learning.temporal_moment_saved_learning;
  fields(moment, { status: 'passed', mismatches: [], moment_subgroup_rows_checked: 16384, whole_group_two_moment_partition_rows_checked: 8192, whole_group_partition_squared_norms_and_counts_verified: true, manual_F32_absolute_tolerance: 3e-5, manual_F32_relative_tolerance: 3e-5, per_scalar_gradient_proof_claimed: false, other2046_raw_gradients_reconstructed: false, pool_centered_variance_is_native_uncertainty: false }, 'Nested moment arithmetic scope');
  check(integer(moment.checks) && moment.checks > 0 && learning.moment_subgroup_rows === 16384, 'Positive moment checks/exact subgroup rows');
  check(I.training.all_eight_temporal_moment_subgroups_learned === true && moment.moment_subgroup_event_counts.length === 8 && learning.moment_subgroups.length === 8, 'Exactly eight moment subgroups');
  for (const [slot, row] of learning.moment_subgroups.entries()) {
    const bin = Math.floor(slot / 2), half = slot % 2;
    fields(row, { bin, moment: half ? 'regularized_stddev' : 'mean', export_row_begin: bin * 8 + half * 4, export_row_end: bin * 8 + (half + 1) * 4, final_weight_changed: true, final_bias_changed: true }, 'Exact moment subgroup rows/change');
    for (const key of ['weight_nonzero_data_gradient_steps', 'bias_nonzero_data_gradient_steps', 'weight_movement_steps', 'bias_movement_steps']) check(integer(row[key]) && row[key] > 0 && row[key] <= 2048 && row[key] === moment.moment_subgroup_event_counts[slot][key], `Exact subgroup event count ${key}`);
  }
  const witnesses = moment.actual_same_forward_moment_witnesses;
  eq(witnesses.map(row => row.update), [1, 2048], 'Only actual first/last witnesses');
  for (const [index, row] of witnesses.entries()) {
    fields(row, { same_forward_only: true, total_component_archive_fields: 60, new_temporal_trace_fields: 25, parameter_array_fields: 12, new_regularized_dispersion_trace_fields: 4, actual_local_weights_layout: 'same-forward-F32[R,4,T]-per-window-not-aggregate', local_weights_reconstructed_from_mixture_or_scores: false, regularized_sqrt_uses_actual_saved_root: true, attention_dispersion_score_and_position_routes_live: true, upstream_stage: 'PRECLIP-post-empty-export', local_export_and_positions_adjoint_verified: true, local_adjoint_absolute_tolerance: 1e-9, local_adjoint_relative_tolerance: 3e-4, postclip_export_nonvacuity_threshold: 1e-6, full_hidden_encoder_or_global_norm_reconstruction: false, matched_pre_step_decoder_adjoint_replay: false }, 'Same-forward local-adjoint scope');
    check(row.upstream_gradient_defined_by_route.length === 2 && row.upstream_gradient_defined_by_route.every(flag => typeof flag === 'boolean'), 'Two actual upstream-defined flags');
    check(finite(row.recorded_clip_factor_applied_once) && row.recorded_clip_factor_applied_once > 0 && row.recorded_clip_factor_applied_once <= 1 && finite(row.postclip_export_expected_maximum) && row.postclip_export_expected_maximum > 1e-6, 'Single clip and nonvacuous local export');
    eq(pick(row.artifact, bytesKeys), pick(audit.actual_pre_step_composite_objective.witnesses[index].witness, bytesKeys), 'Same witness composite alias');
    eq(pick(row.artifact, bytesKeys), pick(audit.temporal_pooling_saved_learning.actual_raw_array_witnesses[index].artifact, bytesKeys), 'Same witness temporal alias');
    const errors = row.local_parameter_gradient_error_summaries;
    eq(Object.keys(errors).sort(), LOCAL_PARAMETERS, 'Only three local parameter gates');
    for (const [name, values] of Object.entries(errors)) {
      eq(Object.keys(values).sort(), ['maximum_expected_absolute', 'maximum_actual_absolute', 'maximum_absolute_error', 'maximum_scaled_error'].sort(), `Error schema ${name}`);
      check(Object.values(values).every(value => finite(value) && value >= 0) && values.maximum_scaled_error <= 1, `Finite bounded local parity ${name}`);
    }
    const gauge = row.common_score_bias_gauge_descriptive;
    fields(gauge, { parity_gate: false, nonzero_gradient_required: false }, 'Gauge descriptive only');
    check(['actual', 'reference', 'absolute_error'].every(key => finite(gauge[key])), 'Finite descriptive gauge');
  }
  fields(established, { all_eight_temporal_moment_subgroups_verified: true, whole8_two4_squared_norms_and_element_counts_verified: true, actual_first_last_local_moment_adjoints_verified: true, actual_local_moment_adjoint_parameter_tensors: LOCAL_PARAMETERS, actual_local_moment_adjoint_absolute_tolerance: 1e-9, actual_local_moment_adjoint_relative_tolerance: 3e-4, each_actual_local_export_nonvacuity_threshold: 1e-6, common_score_bias_gauge_tight_parity_required: false, actual_component_archive_fields: 60, same_forward_temporal_fields: 25, new_regularized_dispersion_trace_fields: 4, regularized_sqrt_uses_actual_saved_root: true, attention_dispersion_score_and_position_routes_live: true, raw_root_absent_offset_exception: true, local_hidden_adjoint_array_available: false, score_to_hidden_route_reconstructed: false, hidden_reference_is_observed_gradient_parity: false, full_encoder_gradient_or_global_norm_reconstructed: false }, 'Separate bounded moment establishment');

  group = 'ancestry-bindings-counts';
  fields(I.experiment_card, { scientific_recipe_parent: 'CPB-v13 source only;fresh weights/optimizer/statistics,not trained ancestry', scoped_reference: 'CPB-v13.market-01' }, 'V13 source-only scientific parent');
  for (const record of [current.fresh_research_candidate, instance, receipt]) eq(record.temporal_moment_saved_learning, moment, 'Whole bounded moment record retained');
  for (const record of [current.fresh_research_candidate, instance]) fields(record, { all_eight_temporal_moment_subgroups_learned: true, scientific_recipe_parent: 'CPB-v13.market-01', scientific_recipe_parent_is_source_only: true, scoped_reference: 'CPB-v13.market-01' }, 'Fresh source-only recipe labels');
  const establishmentBindings = at(established, contract.establishment_bindings_path);
  for (const [role, key] of Object.entries(contract.establishment_binding_keys)) eq(pick(establishmentBindings[key], bytesKeys), pick(B[role], bytesKeys), `Exact completed evidence ${role}`);
  fields(ledger.next_direction, { goal_active: !advancement, goal_criteria_satisfied: advancement, later_training_authorized_by_this_finalization: false }, 'Goal outcome grants no later run');
  const core = at(audit, contract.training_audit_checks_path);
  const moduleCounts = Object.fromEntries(['numeric_statistics', 'saved_tensor_state', 'manual_saved_Gaussian', 'actual_pre_step_composite_objective', 'temporal_pooling_saved_learning'].map(name => [name, audit[name].checks]));
  const qualityCount = at(I.quality_audit, contract.quality_audit_checks_path);
  const totalChecks = core + Object.values(moduleCounts).reduce((a, b) => a + b, 0);
  fields(established, { training_audit_core_checks: core, training_audit_five_module_checks: moduleCounts, training_audit_total_checks: totalChecks, quality_audit_checks: qualityCount }, 'Core plus five modules; nested audit counted once');
  fields(instance, { independent_training_audit_core_checks: core, independent_training_audit_module_checks: moduleCounts, independent_training_audit_core_plus_modules_checks: totalChecks, independent_quality_audit_checks: qualityCount }, 'Instance audit counts');

  group = 'instance-source-closure';
  check(current.fresh_research_candidate.instance === METHOD && instance.instance === METHOD, 'Instance/catalog identity');
  fields(current.fresh_research_candidate, { physical_version_root: 'v14', design_file: 'CPB-v14.json', measured_instance_file: 'instances/CPB-v14.market-01.json' }, 'Own physical V14 catalog root');
  fields(instance, { version_root: FAMILY + 'v14', source_updates: 0, new_causal_updates: 2048, batch_size: 8 }, 'Own V14 root and fresh trajectory');
  check(integer(instance.attempts) && instance.attempts >= 2048 && instance.attempts <= 4096 && instance.attempts === I.training.attempted_batches && instance.attempts === audit.attempted_batches && audit.CSV_rows === instance.attempts && instance.successful_updates === 2048 && audit.successful_CSV_rows === 2048 && instance.skipped_ineligible_attempts === instance.attempts - 2048 && audit.skipped_ineligible_attempts === instance.attempts - 2048 && audit.checkpoint_attempt_counters['2048'] === instance.attempts, 'Actual successful/skipped/attempted counters');
  const counts = I.experiment_card.source_counts;
  fields(counts, { capsule_sources: 145, binaries: 7, translation_units: 11 }, '145/7/11 source closure');
  fields(instance, { physical_owned_family_files: counts.family_inside + counts.family_external_header, physical_owned_research_files: counts.research, frozen_source_capsule_files: counts.capsule_sources }, 'Physical source counts');
  for (const record of [current.fresh_research_candidate, instance]) {
    check(!Object.hasOwn(record, 'temporal_pooling_partition'), 'No legacy disjoint current field');
    fields(record, { ...TEMPORAL, shared_global_clip: true, optimizer_independence_claimed: false, initial_pairing_native_coordinate_slice: [20, 40], whole_native40_initial_equality_claimed: false }, 'Current geometry/clip/partial semantics');
    eq(pick(record.initial_native20_40_and_statistics_pairings, bytesKeys), pick(B.pairings, bytesKeys), 'Initial pairings exact binding');
  }
  fields(instance, { all_75_mandatory_identifiable_tensors_with_nonzero_CUDA_gradients_and_movement: true, mandatory_identifiable_tensor_value_capacity: 294751, data_learning_is_tensor_event_evidence_not_every_scalar: true, weight_decay_counted_as_data_learning: false }, 'Tensor-event/capacity wording');
  eq(instance.temporal_pooling_learning, pick(B.temporal_learning, pinKeys), 'Finer learning exact pin');
  eq(instance.temporal_pooling_proof, pick(B.temporal_proof, pinKeys), 'Temporal fixture exact pin');
  const admission = { source_files: counts.capsule_sources, owner_source_files: counts.family_inside + counts.family_external_header + counts.research, freeze_and_preflight_bound: true, normal_quality_all_stages_passed: true, scientific_model_head_or_training_replay: false };
  eq(established.normal_prospective_source_admission, admission, 'Normal source admission');
  for (const record of [current.fresh_research_candidate, instance, receipt]) {
    eq(record.normal_prospective_source_admission, admission, 'Whole source admission retained');
    eq(record.historical_V9_reporting_recovery, I.experiment_card.historical_V9_reporting_recovery, 'V9 reporting recovery opaque preservation');
    eq(record.historical_V9_reporting_recovery, established.historical_V9_reporting_recovery_preserved, 'V9 receipt summary matches');
  }
  const source = I.source_manifest;
  check(source.kind === 'causal_fresh_v14_source_capsule_v1' && source.card_sha256 === B.experiment_card.sha256 && source.files.length === 145 && new Set(source.files.map(row => row.path)).size === 145, 'Unique exact main145 source closure');
  // Python len() counted this filename-keyed mapping, not an array. Preserve
  // the actual saved schema and its exact eleven translation-unit closures.
  const closures = I.build.dependency_closures;
  check(Array.isArray(I.build.binaries) && I.build.binaries.length === 7 && closures !== null
    && typeof closures === 'object' && !Array.isArray(closures) && Object.keys(closures).length === 11,
  'Seven binaries and eleven translation units');
  eq(pick(I.build.source_manifest, bytesKeys), pick(B.source_manifest, bytesKeys), 'Build/source manifest exact binding');
  fields(I.mirror, { scope: 'Complete', source_files_verified: 145, all_mirrored_pt_ignored: true }, 'Complete normal ignored-PT mirror');
  check(I.training.raw_load_calls === 512 + instance.attempts + 36, 'Calibration/attempted/fixed-query raw loads');
  fields(I.prospective_freeze, { kind: 'causal_fresh_v14_prospective_freeze_v1', status: 'passed', owner_source_files: admission.owner_source_files, critical_checkpoints_verified: 119, latest_v13_sources_verified: 143 }, 'Freeze119/V13-143 ancestry closure');
  eq(pick(I.prospective_freeze.card, bytesKeys), pick(B.experiment_card, bytesKeys), 'Freeze/card pin');
  fields(I.protocol_preflight, { kind: 'causal_fresh_v14_source_protocol_preflight_v1', status: 'passed', card_sha256: B.experiment_card.sha256 }, 'Preflight/card closure');
  check(I.protocol_preflight.sources.length + 1 === 145, 'Actual preflight source closure');
  for (const record of [current.fresh_research_candidate, instance]) for (const role of ['prospective_freeze', 'protocol_preflight']) eq(pick(record[role], pinKeys), pick(B[role], pinKeys), `Prospective evidence exact ${role}`);

  group = 'final-invariants';
  for (const record of [current.fresh_research_candidate, instance.quality]) fields(record, { numeric_passed: passed, numeric_total: 22, advancement_passed: advancement, trust_admission_passed: true, full_coverage: true }, 'Actual numeric/trust/coverage result');
  check(instance.main_checkpoints.length === 5, 'Five checkpoint roles');
  eq(new Set(instance.main_checkpoints.map(row => path.posix.basename(row.path))), new Set(['inference-0.pt', 'inference-2048.pt', 'training-512.pt', 'training-1024.pt', 'training-2048.pt']), 'Exact checkpoint role names');
  check(instance.prospective_descriptor.sha256 === B.descriptor.sha256 && instance.prospective_descriptor_rewritten === false, 'Prospective descriptor immutable');
  fields(receipt, { model_calls: 0, head_fits: 0, training_updates: 0, live_files_written: 0 }, 'No replay/live application');
  return { checks, obligation_check_counts: groups, strict_new_quality_leaders: replacements.length,
    numeric_passed: passed, numeric_total: 22, overall_establishment_passed: advancement,
    training_audit_core_plus_modules_checks: totalChecks, quality_audit_checks: qualityCount };
}

async function reviewProposalsUnchecked({ root, aliases, aliasVerificationPath, cardPath, proposalsPath, output }) {
  if (process.platform !== 'linux') throw new Error('Managed Linux container only');
  await lstat('/.dockerenv');
  root = path.resolve(root);
  const localRoot = path.join(root, '.temp/embeddings/closeout/cpb-v14');
  if (!path.resolve(output).startsWith(localRoot + path.sep)) throw new Error('Exclusive local closeout receipt under .temp required');
  try { await lstat(output); throw new Error('Output already exists'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const manifest = await readJSON(aliases);
  const implementationPins = await Promise.all([SELF, ALIAS_SOURCE, JSON_SOURCE].map(file => pin(file)));
  const aliasVerification = await readJSON(aliasVerificationPath);
  const manifestPin = await pin(aliases);
  if (aliasVerification.status !== 'passed' || aliasVerification.passed !== true || !Array.isArray(aliasVerification.failures)
      || aliasVerification.failures.length !== 0 || !same(pick(aliasVerification.alias_manifest, bytesKeys), pick(manifestPin, bytesKeys))
      || path.resolve(aliasVerification.alias_manifest.path) !== path.resolve(aliases)) throw new Error('Complete passed alias verification bound to the actual manifest required');
  const resolver = createIdentityResolver({ root, manifest });
  const resolvedPins = [];
  const originalReviewer = path.join(root, PAYLOAD, 'source/review_proposals.py');
  const originalPin = await pin(originalReviewer);
  if (!same(pick(originalPin, bytesKeys), ORIGINAL_REVIEWER)) throw new Error('Frozen independent reviewer identity mismatch');
  const cardPin = await pin(cardPath);
  const card = await readJSON(cardPath);
  if (card.kind !== 'cpb_v14_metadata_only_finalization_card_v1' || card.inputs.length !== 49) throw new Error('Own admitted49-role metadata card required');
  const inputs = {}, byId = {};
  let bindingChecks = 0;
  const bound = async (file, expected) => {
    const actual = await pin(file);
    if (!same(pick(actual, bytesKeys), pick(expected, bytesKeys))) throw new Error(`Exact metadata binding failed: ${file}`);
    bindingChecks++;
    return actual;
  };
  const resolve = async binding => {
    const located = await resolver.resolve(binding);
    if (!located.absolute_path) throw new Error('Identity resolver must return absolute_path');
    await bound(located.absolute_path, binding);
    resolvedPins.push({ original: pick(binding, pinKeys), current_path: located.current_path, repository: located.repository });
    return located.absolute_path;
  };
  for (const row of card.inputs) {
    if (Object.hasOwn(byId, row.id)) throw new Error(`Unique input roles required: ${row.id}`);
    const staged = safeRelative(path.dirname(cardPath), row.staged_path);
    await bound(staged, row);
    await resolve(row);
    byId[row.id] = row;
    if (row.format === 'json') inputs[row.id] = await readJSON(staged);
  }
  const receiptPath = path.join(proposalsPath, 'finalization-proposal.json');
  const receiptPin = await pin(receiptPath);
  const receipt = await readJSON(receiptPath);
  if (receipt.kind !== 'cpb_v14_metadata_only_finalization_proposal_v1' || receipt.status !== 'passed_proposal_not_applied') throw new Error('Successful unapplied proposal required');
  if (!same(pick(receipt.card, bytesKeys), pick(cardPin, bytesKeys))) throw new Error('Actual generator card binding failed');
  if (!same(receipt.inputs, card.inputs.map(row => pick(row, pinKeys)))) throw new Error('Proposal receipt must bind exactly all49 card roles');
  await resolve({ ...receiptPin, path: `${HISTORICAL_FINALIZATION}/proposals-001/finalization-proposal.json` });
  const expectedOutputs = ['doc/clean_causal_encoder_research_ledger.json', FAMILY + 'current.json', FAMILY + 'instances/CPB-v14.market-01.json'];
  if (receipt.outputs.length !== 3 || !same(new Set(receipt.outputs.map(row => row.path)), new Set(expectedOutputs))) throw new Error('Exactly three metadata outputs required');
  const outputs = {};
  for (const row of receipt.outputs) {
    const file = safeRelative(proposalsPath, row.path);
    await bound(file, row);
    await resolve({ ...row, path: `${HISTORICAL_FINALIZATION}/proposals-001/${row.path}` });
    outputs[row.path] = await readJSON(file);
  }
  const ledger = outputs[expectedOutputs[0]], current = outputs[expectedOutputs[1]], instance = outputs[expectedOutputs[2]];
  const result = reviewData({ card, inputs, byId, receipt, ledger, current, instance });
  // The historic /opt tree no longer exists. Every old path is verified by the
  // full repository/path/size/hash alias tuple; no path-only root substitution.
  for (const row of inputs.source_manifest.files) {
    safeRelative('/unused-source-leaf', row.path);
    await resolve({ ...row, path: `${card.container_attempt}/source/${row.path}` });
  }
  for (const row of inputs.build.binaries) await resolve(row);
  // Extra binding checks close the immutable card/receipt/checkpoint references
  // which the original reviewer checked only indirectly or by role name.
  await resolve(receipt.card);
  for (const row of instance.main_checkpoints) await resolve(row);
  for (const role of ['prospective_freeze', 'protocol_preflight']) await resolve(byId[role]);
  for (const row of implementationPins) await bound(row.path, row);
  const review = {
    kind: 'mityane_cpb_v14_independent_metadata_proposal_review_v1', status: 'passed', passed: true,
    scope: 'Already-generated immutable proposal DATA only; no maintained metadata application',
    ...result, checks: result.checks + bindingChecks, byte_binding_checks: bindingChecks,
    obligations: OBLIGATIONS.map(([id, original_lines, coverage]) => ({ id, original_lines, coverage })),
    original_reviewer: { ...originalPin, current_path: path.relative(root, originalReviewer).split(path.sep).join('/') },
    transport_adaptations: [
      'Original Linux-only guard retained; historical /opt finalization output restriction replaced by exclusive current local closeout receipt.',
      'Original physical /opt source/binary lookups use full identity aliases; historical paths in proposal DATA remain unchanged.',
      'Original current.active_application=V3 is reviewed as historical DEV metadata only; it does not select a Mityane application.',
      'No frozen proposer or reviewer execution; this independent standard-library Node implementation has a new source identity.',
      'JSON integer identities use exact BigInt outside the safe Number range; last duplicate key matches the original Python loader.',
    ],
    historical_application_selection_only: true, project_application_selection_made: false,
    baseline_changed: false, descriptor_changed: false, producer_changed: false,
    model_calls: 0, head_fits: 0, training_updates: 0, live_files_written: 0,
    metadata_outputs: receipt.outputs, alias_manifest: manifestPin, alias_verification: await pin(aliasVerificationPath), resolved_pins: resolvedPins,
    inputs: [
      ...await Promise.all(expectedOutputs.map(async (relative, i) => {
        const actual = await pin(safeRelative(proposalsPath, relative));
        return { role: ['ledger_proposal', 'catalog_proposal', 'instance_proposal'][i], ...actual, path: path.relative(root, actual.path).split(path.sep).join('/') };
      })),
      { role: 'proposal_receipt', ...receiptPin, path: path.relative(root, receiptPath).split(path.sep).join('/') },
      { role: 'alias_manifest', ...await pin(aliases), path: path.relative(root, aliases).split(path.sep).join('/') },
      { role: 'alias_verification', ...await pin(aliasVerificationPath), path: path.relative(root, aliasVerificationPath).split(path.sep).join('/') },
    ],
    card: cardPin, generator_receipt: receiptPin, source: await pin(SELF),
    source_closure: implementationPins.map(row => ({ ...row, path: path.relative(root, row.path).split(path.sep).join('/') })),
  };
  await rejectRedirectedAncestors(output);
  await mkdir(path.dirname(output), { recursive: true });
  await rejectRedirectedAncestors(output);
  await writeFile(output, jsonBytes(review), { flag: 'wx' });
  return review;
}

export async function reviewProposals(options) {
  if (process.platform !== 'linux') throw new Error('Managed Linux container only');
  await lstat('/.dockerenv');
  const receiptRoot = path.resolve(options.root, '.temp/embeddings/closeout/cpb-v14');
  const output = path.resolve(options.output);
  if (!output.startsWith(receiptRoot + path.sep)) throw new Error('Bounded exclusive closeout output required');
  await rejectRedirectedAncestors(output);
  try { await lstat(output); throw new Error('Output already exists'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  try { return await reviewProposalsUnchecked(options); }
  catch (error) {
    // Preserve failed reviews too; a subsequent generation needs another leaf.
    const failed = { kind: 'mityane_cpb_v14_independent_metadata_proposal_review_v1', status: 'failed', passed: false,
      error: String(error.message), source: await pin(SELF), model_calls: 0, head_fits: 0, training_updates: 0, live_files_written: 0 };
    await rejectRedirectedAncestors(output);
    await mkdir(path.dirname(output), { recursive: true });
    await rejectRedirectedAncestors(output);
    await writeFile(output, jsonBytes(failed), { flag: 'wx' });
    throw error;
  }
}

async function cli() {
  const flags = {};
  for (let i = 2; i < process.argv.length; i += 2) {
    const key = process.argv[i];
    if (!['--root', '--aliases', '--alias-verification', '--card', '--proposals', '--output'].includes(key) || !process.argv[i + 1] || Object.hasOwn(flags, key)) throw new Error('Usage: review.mjs --root /mityane --aliases FILE --alias-verification FILE --output NEWFILE [--card FILE --proposals DIR]');
    flags[key] = process.argv[i + 1];
  }
  if (!flags['--root'] || !flags['--aliases'] || !flags['--alias-verification'] || !flags['--output']) throw new Error('root,aliases,passed alias verification,exclusive output required');
  const payload = path.join(flags['--root'], PAYLOAD);
  const result = await reviewProposals({ root: flags['--root'], aliases: flags['--aliases'], output: flags['--output'],
    aliasVerificationPath: flags['--alias-verification'],
    cardPath: flags['--card'] ?? path.join(payload, 'card-source-001/finalization-card.json'),
    proposalsPath: flags['--proposals'] ?? path.join(payload, 'proposals-001') });
  process.stdout.write(JSON.stringify({ status: result.status, checks: result.checks, numeric_passed: result.numeric_passed, strict_new_quality_leaders: result.strict_new_quality_leaders }) + '\n');
}
if (process.argv[1] && path.resolve(process.argv[1]) === SELF) cli().catch(error => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
