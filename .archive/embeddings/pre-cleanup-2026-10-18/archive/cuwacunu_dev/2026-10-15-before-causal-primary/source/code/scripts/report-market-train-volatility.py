#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Capture audited saved-TRAIN JSON without fitting or loading tensor contents."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path('/cuwacunu')
PROTOCOL = 'market-train-volatility-diagnosis-v1'
DURABLE = ROOT / 'doc/results/market_train_volatility_diagnosis_v1'
CARD = ROOT / 'doc/MARKET_TRAIN_VOLATILITY_DIAGNOSIS_CARD.md'
MANIFEST = DURABLE / 'input-manifest.json'
REPORT = ROOT / 'doc/MARKET_TRAIN_VOLATILITY_DIAGNOSIS_RESULTS.md'
CARD_HASH = '89a2005b87cf714a53da7aa4ed65a15da477712aca49a36ec159c75fd9de5a87'
MANIFEST_HASH = '4630c3f16a3ce6201105983733b31faa446bf22e03a123a79a8c38cbb664e8df'
TASKS = ['past_volatility', 'past_trend', 'past_activity', 'past_relation']
TASK_LABELS = ['Volatility', 'Trend', 'Activity', 'Relation']
MODELS = ['RPB-v22.market-02', 'RPB-v24.market-01.alt-01.init', 'RPB-v24.market-01.alt-01']
LEGEND = ('**MKT-HIST-2**: BTC/ETH/SOL January 2024 minute history; recent128 '
          'past-only targets; designed complexity **4/5**. Saved chronological TRAIN only, '
          'one source period; no newly scored VALIDATION or TEST.')
TEXT_SUFFIXES = {'.json', '.log', '.md', '.py', '.cpp', '.mk', '.sh', '.h', '.txt', '.sha256'}


def require(value, why):
    if not value:
        raise RuntimeError(why)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    require(path.is_file() and not path.is_symlink(), 'Missing file or symlink: ' + str(path))
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def write_new(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(contents)


def dump(path, value):
    write_new(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy_new(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    require(source.is_file() and not source.is_symlink(), 'Invalid snapshot source')
    with source.open('rb') as original, target.open('xb') as snapshot:
        while chunk := original.read(1024 * 1024):
            snapshot.write(chunk)
    require(record(source) == record(target), 'Snapshot mismatch: ' + str(source))


def fmt(value, digits=6):
    if value is None:
        return 'undefined'
    require(isinstance(value, (int, float)) and math.isfinite(value), 'Nonfinite report value')
    return f'{value:.{digits}g}'


def table(headers, rows):
    cell = lambda value: str(value).replace('|', r'\|')
    return '\n'.join(['| ' + ' | '.join(cell(value) for value in headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(cell(value) for value in row) + ' |' for row in rows])


def quality_table(methods, task_index, key):
    rows = []
    for method in methods:
        quality = method[key][task_index]
        rows.append([method['model'], method['native_width'], fmt(quality['ridge']['r2']),
                     fmt(quality['neural_mean_repetition_scores']['r2']),
                     fmt(100 * quality['ridge']['coverage'])])
    return table(['Method', 'Size', 'Linear head R²', 'Neural head R²', 'Coverage %'], rows)


def physical_table(methods, task_index):
    return table(['Method', 'Linear MAE', 'Linear RMSE', 'Mean neural MAE', 'Mean neural RMSE'],
                 [[method['model'], fmt(method['all_train_quality'][task_index]['ridge']['mae']),
                   fmt(method['all_train_quality'][task_index]['ridge']['rmse']),
                   fmt(method['all_train_quality'][task_index]['neural_mean_repetition_scores']['mae']),
                   fmt(method['all_train_quality'][task_index]['neural_mean_repetition_scores']['rmse'])]
                  for method in methods])


def objective_table(methods, task_index):
    rows = []
    for method in methods:
        item = method['task_diagnoses'][task_index]
        rows.append([method['model'], fmt(item['normalized_sse']),
                     fmt(item['ridge_weight_squared_norm']), fmt(item['ridge_objective_sse_plus_penalty']),
                     fmt(item['parent_padded_ridge_objective_sse_plus_penalty']),
                     fmt(item['ridge_objective_improvement_vs_parent_padded'])])
    return table(['Method', 'Standardized SSE', 'Weight penalty', 'Ridge objective',
                  'Padded-parent objective', 'Objective improvement'], rows)


def redistribution_table(methods, task_index, context='all_train_contributions'):
    rows = []
    for method in methods:
        item = method['task_diagnoses'][task_index]
        block = item[context]
        rows.append([method['model'], fmt(item['prefix_coefficient_difference_l2']),
                     fmt(item['prefix_coefficient_cosine']), fmt(item['change4_coefficient_l2']),
                     fmt(block['prefix_change_physical']['population_std']),
                     fmt(block['change4_contribution']['population_std']),
                     fmt(block['prefix_change_correlation']), fmt(block['cancellation_ratio'])])
    return table(['Method', 'Prefix weight shift L2', 'Prefix weight cosine', 'Change weight L2',
                  'Prefix-shift physical std', 'Change4 physical std', 'Prefix/change r',
                  'Cancellation ratio'], rows)


def validate_quality(methods):
    for index, method in enumerate(methods):
        require(method['model'] == MODELS[index] and method['native_width'] == (32 if index == 0 else 36),
                'Method identity or native width differs')
        require(method['saved_feature_valid_rows'] == 112 and method['ridge_three_repetitions_identical'],
                'Saved feature support or deterministic ridge equality failed')
        for key, rows in [('all_train_quality', 112), ('fixed_first_56_rows_quality', 56),
                          ('fixed_last_56_rows_quality', 56)]:
            require([item['task'] for item in method[key]] == TASKS, 'Task ordering differs')
            for quality in method[key]:
                score = quality['ridge']
                require(score['total'] == rows and score['valid'] == rows and score['coverage'] == 1,
                        'TRAIN ridge support differs')
                repetitions = quality['neural_each']
                require([item['repetition'] for item in repetitions] == [2701, 2802, 2903],
                        'Saved neural repetition policy differs')
                require(all(item['score']['valid'] == rows and item['score']['coverage'] == 1
                            for item in repetitions), 'Saved neural support differs')
        require([item['task'] for item in method['task_diagnoses']] == TASKS, 'Diagnosis task order differs')
        for item in method['task_diagnoses']:
            require(item['common_train_rows'] == 112, 'Common TRAIN support differs')
            for key in ['parent_padded_prediction_max_abs_normalized_error', 'kkt_relative_l2',
                        'prefix_redistribution_relative_l2',
                        'physical_decomposition_max_abs_target_standardized_error']:
                require(0 <= item[key] <= 1e-8, 'Admission tolerance failed: ' + key)


def render(result, verification, audit, preservation, run):
    methods = result['methods']
    lines = ['# Saved-TRAIN volatility diagnosis', '',
             'Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE.', '',
             '**This is new arithmetic on preserved TRAIN evidence.** It adds zero encoder calls, '
             'updates, scaler fits or head fits. The exact parent32 and its old linear predictor remain '
             'representable in native36; the audited saved heads measure how fitting used the appended coordinates.', '',
             LEGEND, '',
             'All112 existing TRAIN readout rows are from Jan1–21, one chronological source period. '
             'The original encoder runs used batch8 on CUDA:2048 parent updates, and2048 additional '
             'change-branch updates for the trained native36. Here analysis runs on CPU inside the existing '
             'GPU-enabled cuwacunu container. Initialized native36 already contains the trained parent; '
             'its change block has zero branch-training updates.', '',
             'Saved fixed heads are unchanged: ridge penalty1; neural16-tanh, Adam0.01,100 updates, '
             'repetitions2701/2802/2903. Four-output parameter counts are132/596 atD32 and148/660 atD36 '
             '(linear/neural). Three neural readouts share one encoder bank; they are not independent encoder runs. '
             'Neural table values average the three saved repetition scores. R² is a regression measure, '
             'not accuracy or a percentage. No post-encoder PCA, feature projection, new normalization fit '
             'or old encoder replay was used.', '']
    for task_index, label in enumerate(TASK_LABELS):
        unit = methods[0]['all_train_quality'][task_index]['unit']
        lines.extend(['## ' + label + ' — all112 TRAIN rows', '', LEGEND, '',
                      quality_table(methods, task_index, 'all_train_quality'), '',
                      'Physical error unit: `' + unit + '`; values describe the same112 TRAIN rows.', '', LEGEND, '',
                      physical_table(methods, task_index), '',
                      '### Existing ridge objective', '',
                      'Objective = sum of standardized TRAIN squared errors + squared coefficient norm; '
                      'the intercept is unpenalized. The comparison uses the already saved parent '
                      'predictor padded with four zeros, with no new fit.', '', LEGEND, '',
                      objective_table(methods, task_index), '',
                      '### Coefficient redistribution and contribution cancellation', '',
                      'Physical standard deviations use the target unit above. Prefix-shift contribution '
                      'uses the difference between saved new-prefix and parent coefficients. '
                      'Prefix/change correlation and cancellation use the new head’s full prefix contribution '
                      'and change4 contribution. Cancellation ratio = (var(prefix)+var(change))/var(total); '
                      'undefined denotes zero variance.', '', LEGEND, '', redistribution_table(methods, task_index), ''])
    lines.extend(['## Level/change correlation summaries', '',
                  'Correlations use the same112 TRAIN rows and saved normalizers. Counts at|r|≥0.9 '
                  'are descriptive flags, not a model-selection threshold. Change/change summaries exclude '
                  'the diagonal and count both matrix directions; level/change summaries include all128 pairs.', '', LEGEND, '',
                  table(['Method', 'Block pairs', 'Mean absolute r', 'Maximum absolute r', '|r| ≥0.9 pairs', 'Defined pairs'],
                        [[method['model'], label, fmt(method[key]['mean_absolute']),
                          fmt(method[key]['max_absolute']),
                          method[key]['absolute_correlation_at_least_0_9_count_descriptive_only'],
                          method[key]['count']]
                         for method in methods[1:]
                         for label, key in [('Prefix32/change4', 'prefix32_change4_cross_correlation_summary'),
                                            ('Change4/change4', 'change4_correlation_off_diagonal_summary')]]), '',
                  'Full32×4 and4×4 correlation/covariance matrices, every coordinate’s saved mean/scale, '
                  'target/residual correlations, coefficients, physical/standardized contribution RMS/std '
                  'and decomposition covariance matrices are retained in '
                  '[diagnosis.json](results/market_train_volatility_diagnosis_v1/receipts/reader/diagnosis.json). '
                  'The parent residual used for coordinate correlations is target minus parent prediction; '
                  'contribution error correlations use parent prediction minus target, as labelled in JSON.', '',
                  '## Change coordinates and target correlations', ''])
    for method in methods[1:]:
        lines.extend(['### ' + method['model'], '', LEGEND, '',
                      table(['Coordinate', 'Saved mean', 'Saved scale', 'Volatility r', 'Trend r', 'Activity r', 'Relation r'],
                            [[item['coordinate'], fmt(item['saved_feature_mean']), fmt(item['saved_feature_scale'])] +
                             [fmt(value) for value in item['target_correlations']]
                             for item in method['coordinates'] if item['block'] == 'change4']), '',
                      'Correlations with the existing parent residual, target minus prediction:', '', LEGEND, '',
                      table(['Coordinate', 'Volatility residual r', 'Trend residual r', 'Activity residual r', 'Relation residual r'],
                            [[item['coordinate']] + [fmt(value) for value in
                             item['parent_residual_target_minus_prediction_correlations']]
                             for item in method['coordinates'] if item['block'] == 'change4']), ''])
    lines.extend(['## Fixed chronology context', '',
                  'The first56 and last56 TRAIN rows were declared in advance. These descriptive blocks '
                  'are not additional data splits, independent confirmations or selected windows. '
                  'Each table’s neural value is the mean of the same three saved readouts.', ''])
    for label, quality, contributions in [('First56 TRAIN rows', 'fixed_first_56_rows_quality', 'fixed_first_56_rows_contributions'),
                                           ('Last56 TRAIN rows', 'fixed_last_56_rows_quality', 'fixed_last_56_rows_contributions')]:
        for task_index, task in enumerate(TASK_LABELS):
            lines.extend(['### ' + label + ' — ' + task, '', LEGEND + ' This table uses56 TRAIN rows.', '',
                          quality_table(methods, task_index, quality), '', LEGEND + ' This table uses56 TRAIN rows.', '',
                          redistribution_table(methods, task_index, contributions), ''])
    lines.extend(['## Arithmetic checks, provenance and timing', '',
                  'The reader opened exactly22 allowlisted TRAIN archives: one shared target bank, '
                  'three feature banks and18 saved fit/prediction archives. Exact prefix32, row/group '
                  'associations, support and stored normalizers passed. All deterministic ridge repetitions '
                  'match. Prediction reconstruction, padded-parent representation, normal equations, '
                  'coefficient redistribution, block decomposition and MSE identities passed the '
                  'prospective tolerances. No encoder checkpoint, raw candle, synthetic tensor or '
                  'held-out diagnostic tensor was opened.', '',
                  f"Independent standard-library arithmetic passed **{audit['checks']} checks**, maximum normalized "
                  f"difference **{fmt(audit['maximum_normalized_difference'])}**. Reader prediction reconstruction’s "
                  f"maximum target-standardized absolute difference was **{fmt(result['saved_ridge_reconstruction_max_abs_target_standardized_error'])}**.", '',
                  f"Reader load/verification: **{fmt(result['load_and_verification_seconds'])} seconds**; "
                  f"load/verification/diagnosis excluding final writes: **{fmt(result['load_verify_and_diagnose_seconds_excluding_write'])} seconds**; "
                  f"reader wall time including I/O: **{fmt(verification['reader_wall_seconds_including_io'])} seconds**. "
                  f"Independent audit: **{fmt(audit['seconds'])} seconds**. Reporting/capture time is separate in completion.json.", '',
                  f"Historical preservation verified **{preservation['files_unchanged']} files**, including all four "
                  'original handoff checkpoint hashes and five native36 checkpoint hashes. One initial '
                  'compile failure is preserved in build-01.log; its implementation correction and '
                  'successful build are preserved in build-02.log. No failed numerical reader run was '
                  'replaced or concealed.', '',
                  'Exact prospective card SHA256: `' + CARD_HASH + '`. Input manifest SHA256: `' + MANIFEST_HASH + '`. '
                  'Source snapshots, input/source/binary pins, actual opened-input ledger, independent audit, '
                  'preservation receipts and the full-precision tables are preserved under '
                  '[the evidence directory](results/market_train_volatility_diagnosis_v1/). '
                  'Binary inputs and the numerical witness containing raw native arrays remain in ignored '
                  'local output; their availability and hashes are pinned in asset_hashes.json. '
                  'No checkpoint, native tensor archive or raw numerical witness is copied into durable Git evidence.', '',
                  '## Interpretation and next decision', '',
                  'The literal parent-information-loss explanation is rejected for these saved banks: '
                  'all32 parent coordinates and normalizers are exact, and the old predictor padded with '
                  'four zero coefficients reproduces it exactly. The new jointly fitted ridge head '
                  'nevertheless changes its prefix coefficients. For volatility the shift L2 is0.602036 '
                  'against the parent coefficient L2 of1.275868 (about47.2%); coefficient cosine is0.881726. '
                  'The saved normal-equation redistribution identity passes, directly connecting the '
                  'changed prefix coefficients to the nonzero level/change cross-covariance term.', '',
                  'This is a different TRAIN fit: volatility linear R² rises0.767220→0.832289, '
                  'and its ridge objective falls27.699222→20.181915. It does not demonstrate an '
                  'optimization failure. Strong pairwise parent/change redundancy is not observed: '
                  'maximum|r| is0.320435, mean|r|0.093580 and none of128 pairs reaches the descriptive0.9 flag. '
                  'Destructive cancellation between the full volatility prefix and change predictions '
                  'is also not supported: their correlation is+0.276161 and cancellation ratio0.835096, '
                  'below1. The four change coordinates have moderate internal correlation '
                  '(maximum off-diagonal|r|0.631650), not near-duplicate pairs.', '',
                  'The benefit is uneven within the predeclared TRAIN chronology. Volatility linear R² '
                  'improves0.654690→0.772730 on the first56 rows, but falls0.932627→0.910444 on the last56. '
                  'The physical MSE delta is−3.65755e−8 in the first half and+3.60555e−9 in the last. '
                  'The change contribution correlates with parent prediction error at−0.553480 early '
                  'and−0.013355 late. These are descriptive results from the same112 fit rows; they '
                  'do not prove an unseen regime shift or the cause of the known validation failure.', '',
                  'Exact parent-prefix preservation and the padded-parent prediction check establish '
                  'that the old linear function remains representable. The measured coefficients and '
                  'TRAIN contributions describe how the unchanged ridge recipe redistributed its weights. '
                  'These TRAIN quantities cannot establish the cause of worse held-out generalization. '
                  'No new validation quality result, overall winner or promotion follows; RPB-v22 stays primary.', '',
                  'The preferred next hypothesis is to preserve change-energy information while the '
                  'branch learns signed trend: retain waveform reconstruction and add balanced '
                  'reconstruction of the existing signed-change/RMS moment bank from the same four '
                  'coordinates. It uses observed input structure, not market task labels. Generic '
                  'decorrelation is not established as the remedy by these results. See the '
                  '[next-direction proposal](MARKET_CHANGE_ENERGY_NEXT_DIRECTION.md) for the precise '
                  'scope and strict protection thresholds for the other market and TEMPO-5 gains. '
                  'No such objective is implemented or tested here; its complete recipe needs its '
                  'own instance/card before execution. Keep fixed heads, native36 and the2048/B8 '
                  'budget, and do not rescue a stopped recipe through width, rate, budget, head or '
                  'export-projection changes.', '',
                  'Local run: `' + str(run.relative_to(ROOT)) + '`.', ''])
    return '\n'.join(lines)


def main():
    started = time.monotonic()
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT,
            'Run only inside the existing managed cuwacunu container at /cuwacunu')
    require(len(sys.argv) == 2, 'Usage: report-market-train-volatility.py RUN_DIRECTORY')
    run = Path(sys.argv[1]).resolve()
    require(run.is_relative_to(ROOT / 'output/runs/market-train-volatility-diagnosis') and run.is_dir(),
            'Wrong local run directory')
    require(sha(CARD) == CARD_HASH and sha(MANIFEST) == MANIFEST_HASH, 'Frozen card/manifest differs')
    manifest = load(MANIFEST)
    require(manifest['protocol'] == PROTOCOL and manifest['card_sha256'] == CARD_HASH,
            'Manifest protocol/card mismatch')
    members = manifest['members']
    allowed = [member['path'] for member in members]
    require(len(allowed) == 22 and len(set(allowed)) == 22, 'Allowlist must contain exactly22 archives')
    for member in members:
        path = ROOT / member['path']
        require(path.resolve().is_relative_to(ROOT / 'output/runs') and
                'validation' not in member['path'].lower() and 'test' not in member['path'].lower(),
                'Non-TRAIN or unsafe allowlisted path')
        require(record(path) == {key: member[key] for key in ['sha256', 'bytes']},
                'Input hash/availability differs: ' + member['path'])

    verification = load(run / 'verification.json')
    audit = load(run / 'independent-audit.json')
    result = load(run / 'reader/diagnosis.json')
    opened = load(run / 'reader/opened-inputs.json')
    preservation = load(run / 'historical-preservation-after.json')
    provenance = load(run / 'provenance-before.json')
    historical = load(run / 'historical-preservation-before.json')
    require(verification['protocol'] == PROTOCOL and
            verification['status'] == 'reader_completed_preservation_passed_audit_pending',
            'Missing successful preserved reader verification')
    require(audit['protocol'] == PROTOCOL and audit['status'] == 'passed' and
            audit['actual_loaded_archives_match_allowlist'] and audit['loaded_archives'] == 22,
            'Independent audit must already have passed')
    require(result['protocol'] == PROTOCOL and result['dataset_id'] == 'MKT-HIST-2' and
            result['TRAIN_rows'] == 112 and result['ridge_penalty'] == 1,
            'Reader protocol/dataset/rows/recipe differs')
    require(result['parent_prefix32_exact'] and result['row_ids_validity_target_normalizers_verified'],
            'Parent prefix/normalizer association checks failed')
    require(len(opened) == 22 and len(set(opened)) == 22 and set(opened) == set(allowed) and
            opened == result['opened_input_paths'], 'Actual opened ledger differs from exact22 allowlist')
    for evidence, keys in [(verification, ['encoder_calls', 'encoder_updates', 'head_fits', 'scaler_fits',
                                           'held_out_tensor_archives']),
                           (audit, ['encoder_calls', 'encoder_updates', 'head_fits', 'held_out_tensors']),
                           (result, ['encoder_calls', 'encoder_updates', 'head_fits', 'normalizer_fits',
                                     'post_encoder_pca_fits'])]:
        require(all(evidence[key] == 0 for key in keys), 'Forbidden encoder/head/held-out work')
    require(verification['card_sha256'] == CARD_HASH and verification['input_manifest_sha256'] == MANIFEST_HASH,
            'Wrapper card/manifest binding differs')
    require(audit['diagnosis_sha256'] == sha(run / 'reader/diagnosis.json') and
            audit['witness_sha256'] == sha(run / 'reader/numerical-witness.json') and
            audit['auditor_sha256'] == sha(ROOT / 'code/scripts/audit-market-train-volatility.py'),
            'Audit source/result/witness binding differs')
    require(verification['provenance_sha256'] == sha(run / 'provenance-before.json'),
            'Provenance binding differs')
    for name, expected in provenance.items():
        require(record(Path(name)) == expected, 'Pinned source/binary changed: ' + name)
    require(preservation['status'] == 'passed' and preservation['files_unchanged'] == len(historical) and
            preservation['original_handoff_checkpoints_unchanged'] == 4 and
            preservation['native36_checkpoints_unchanged'] == 5 and
            preservation['baseline_sha256'] == sha(run / 'historical-preservation-before.json'),
            'Historical preservation receipt differs')
    # The wrapper already hashed this full historical inventory before/after the
    # reader. Its immutable receipt/hash binding is checked above; do not repeat
    # that same full pass while rendering JSON. Inputs and source pins are checked.
    validate_quality(result['methods'])

    new_outputs = [REPORT, DURABLE / 'receipts', DURABLE / 'sources', DURABLE / 'asset_hashes.json',
                   DURABLE / 'table_rows.json', DURABLE / 'completion.json', DURABLE / 'durable_files.json']
    require(all(not path.exists() for path in new_outputs), 'Report/capture destination already exists; preserve it')
    require((DURABLE / 'build-01.log').is_file() and (DURABLE / 'build-02.log').is_file(),
            'Both failed/successful build logs must be preserved')
    require((DURABLE / 'build-01.log').stat().st_size > 0 and (DURABLE / 'build-02.log').stat().st_size > 0,
            'Build log unexpectedly empty')
    source_paths = {Path(name) for name in provenance if Path(name).is_relative_to(ROOT) and
                    not Path(name).is_relative_to(ROOT / 'output')}
    source_paths.add(Path(__file__).resolve())
    source_paths.update([CARD, MANIFEST, ROOT / 'Makefile', ROOT / 'code/scripts/task.sh',
                         ROOT / 'code/evaluation/src/fixed_regression.cpp',
                         ROOT / 'doc/MARKET_CHANGE_ENERGY_NEXT_DIRECTION.md'])
    for source in sorted(source_paths):
        require(source.suffix in TEXT_SUFFIXES or source.name == 'Makefile',
                'Nontext source snapshot forbidden: ' + str(source))
        copy_new(source, DURABLE / 'sources' / source.relative_to(ROOT))
    run_assets = []
    for source in sorted(run.rglob('*')):
        require(not source.is_symlink(), 'Symlink in run tree')
        if not source.is_file():
            continue
        metadata = {'path': str(source.relative_to(run)), **record(source),
                    'copied_to_durable': False}
        if source.name != 'numerical-witness.json' and source.suffix in TEXT_SUFFIXES:
            copy_new(source, DURABLE / 'receipts' / source.relative_to(run))
            metadata['copied_to_durable'] = True
        run_assets.append(metadata)
    require(any(item['path'] == 'reader/numerical-witness.json' and not item['copied_to_durable']
                for item in run_assets), 'Raw numerical witness must remain local')
    source_pins = [{'path': str(source), **record(source)} for source in sorted(source_paths)]
    dump(DURABLE / 'asset_hashes.json', {
        'protocol': PROTOCOL, 'local_run_directory': str(run), 'card_sha256': CARD_HASH,
        'input_manifest_sha256': MANIFEST_HASH, 'diagnostic_input_archives': members,
        'actual_opened_input_paths': opened, 'run_assets': run_assets,
        'source_pins': source_pins, 'source_binary_provenance': provenance,
        'raw_numerical_witness_policy': 'ignored local only; hash and bytes retained, contents not copied',
    })
    panels = []
    for quality_key, label, rows in [('all_train_quality', 'all112 TRAIN rows', 112),
                                     ('fixed_first_56_rows_quality', 'fixed first56 TRAIN rows', 56),
                                     ('fixed_last_56_rows_quality', 'fixed last56 TRAIN rows', 56)]:
        for task_index, task in enumerate(TASKS):
            panels.append({'dataset_id': 'MKT-HIST-2', 'recipe_id': 'fixed_past_window_regression_v1',
                           'designed_complexity_level': 4,
                           'complexity_scale_max': 5, 'view': label, 'rows': rows, 'task': task,
                           'methods': [{'method': method['model'], 'size': method['native_width'],
                                        'linear_head_r2': method[quality_key][task_index]['ridge']['r2'],
                                        'neural_head_mean_r2': method[quality_key][task_index]['neural_mean_repetition_scores']['r2'],
                                        'coverage': method[quality_key][task_index]['ridge']['coverage'],
                                        'saved_scores': method[quality_key][task_index]}
                                       for method in result['methods']]})
    dump(DURABLE / 'table_rows.json', {'protocol': PROTOCOL, 'evidence_stage': 'saved TRAIN diagnosis',
                                     'new_encoder_updates': 0, 'new_head_fits': 0, 'panels': panels,
                                     'all_task_diagnoses': [{key: method[key] for key in
                                      ['model', 'task_diagnoses', 'coordinates']} for method in result['methods']]})
    write_new(REPORT, render(result, verification, audit, preservation, run))
    completion = {
        'protocol': PROTOCOL, 'status': 'complete', 'disposition': 'audited-saved-TRAIN-diagnosis-only',
        'run': str(run.relative_to(ROOT)), 'dataset_id': 'MKT-HIST-2', 'training_rows': 112,
        'diagnostic_input_archives': 22, 'new_encoder_runs': 0, 'encoder_calls': 0, 'encoder_updates': 0,
        'new_head_fits': 0, 'scaler_fits': 0, 'held_out_tensor_archives': 0, 'post_encoder_pca_fits': 0,
        'primary_changed': False, 'new_validation_quality': False, 'independent_audit_status': 'passed',
        'independent_audit_checks': audit['checks'], 'historical_files_unchanged': len(historical),
        'card_sha256': CARD_HASH, 'input_manifest_sha256': MANIFEST_HASH,
        'reader_binary_sha256': verification['binary_sha256'],
        'reader_wall_seconds_including_io': verification['reader_wall_seconds_including_io'],
        'reader_load_and_verification_seconds': result['load_and_verification_seconds'],
        'reader_load_verify_and_diagnose_seconds_excluding_write': result['load_verify_and_diagnose_seconds_excluding_write'],
        'independent_audit_seconds': audit['seconds'], 'reporting_capture_and_hash_seconds': time.monotonic() - started,
        'report_sha256': sha(REPORT), 'asset_hashes_sha256': sha(DURABLE / 'asset_hashes.json'),
        'table_rows_sha256': sha(DURABLE / 'table_rows.json'),
    }
    dump(DURABLE / 'completion.json', completion)
    receipts = [{'path': str(path.relative_to(DURABLE)), **record(path)}
                for path in sorted(DURABLE.rglob('*')) if path.is_file()]
    dump(DURABLE / 'durable_files.json', {
        'protocol': PROTOCOL, 'report': {'path': str(REPORT.relative_to(ROOT)), **record(REPORT)},
        'files': receipts, 'excludes_only_self': True,
    })
    print(json.dumps(completion, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
