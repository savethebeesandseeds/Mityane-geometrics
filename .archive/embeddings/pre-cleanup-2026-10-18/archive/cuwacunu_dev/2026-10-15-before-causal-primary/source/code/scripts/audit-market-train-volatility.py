#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent TRAIN arithmetic from the reader's local JSON witness; no fits."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path('/cuwacunu')
TASKS = ['past_volatility', 'past_trend', 'past_activity', 'past_relation']
CHECKS = 0
MAX_NORMALIZED_DIFFERENCE = 0.0


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual, expected, message, tolerance=1e-10):
    global CHECKS, MAX_NORMALIZED_DIFFERENCE
    CHECKS += 1
    if actual is None or expected is None:
        require(actual is None and expected is None, message + ': undefined differs')
        return
    error = abs(actual - expected) / (1 + abs(expected))
    require(math.isfinite(actual) and math.isfinite(expected) and error <= tolerance,
            f'{message}: {actual!r} != {expected!r}; normalized error {error!r}')
    MAX_NORMALIZED_DIFFERENCE = max(MAX_NORMALIZED_DIFFERENCE, error)


def mean(values):
    return math.fsum(values) / len(values)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def centered(a):
    avg = mean(a)
    return [value - avg for value in a]


def variance(a):
    return mean([value * value for value in centered(a)])


def corr(a, b):
    left, right = centered(a), centered(b)
    denominator = norm(left) * norm(right)
    return dot(left, right) / denominator if denominator > 0 else None


def column(matrix, j):
    return [row[j] for row in matrix]


def scores(prediction, target):
    errors = [p - y for p, y in zip(prediction, target)]
    squared = dot(errors, errors)
    total = dot(centered(target), centered(target))
    return {'r2': 1 - squared / total if total > 0 else None,
            'mae': mean([abs(error) for error in errors]),
            'rmse': math.sqrt(squared / len(errors))}


def main():
    require(Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT,
            'Run inside the managed cuwacunu container')
    require(len(sys.argv) == 2, 'Usage: audit-market-train-volatility.py RUN_DIRECTORY')
    started = time.monotonic()
    run = Path(sys.argv[1]).resolve()
    require(run.is_relative_to(ROOT / 'output/runs/market-train-volatility-diagnosis'), 'Wrong run path')
    output = run / 'independent-audit.json'
    require(not output.exists(), 'Audit already exists; preserve it')
    witness_path = run / 'reader/numerical-witness.json'
    result_path = run / 'reader/diagnosis.json'
    witness = json.loads(witness_path.read_text())
    result = json.loads(result_path.read_text())
    manifest = json.loads((ROOT / 'doc/results/market_train_volatility_diagnosis_v1/input-manifest.json').read_text())
    allowed = {member['path'] for member in manifest['members']}
    opened = result['opened_input_paths']
    require(len(opened) == 22 and len(set(opened)) == 22 and set(opened) == allowed,
            'Actual loader ledger differs from the exact22 TRAIN allowlist')
    require(witness['opened_input_paths'] == opened, 'Witness input ledger differs')
    targets = witness['targets']['values']
    require(len(targets) == 112 and all(len(row) == 4 for row in targets), 'Target geometry')
    require(all(all(row) for row in witness['targets']['valid']), 'Unsupported target rows')
    row_ids = witness['targets']['row_ids']
    require(all(a < b for a, b in zip(row_ids, row_ids[1:])), 'Chronological row IDs')
    methods = witness['methods']
    require(len(methods) == 3 and len(result['methods']) == 3, 'Method count')
    parent = methods[0]
    contexts = []
    for method_index, method in enumerate(methods):
        raw = method['raw_features']
        x = method['standardized_features']
        width = len(x[0])
        require(len(x) == 112 and width == (32 if method_index == 0 else 36), 'Feature geometry')
        require(method['row_ids'] == row_ids and all(method['feature_valid']), 'Feature support/association')
        for i, row in enumerate(raw):
            require(row[:32] == parent['raw_features'][i], 'Exact parent prefix differs')
        for j in range(width):
            values = column(raw, j)
            mu, scale = mean(values), max(math.sqrt(variance(values)), 1e-8)
            near(method['feature_mean'][j], mu, 'Stored TRAIN feature mean', 1e-12)
            near(method['feature_scale'][j], scale, 'Stored TRAIN feature scale', 1e-12)
            for i in range(112):
                near(x[i][j], (raw[i][j] - method['feature_mean'][j]) / method['feature_scale'][j],
                     'Feature normalization')
        w = method['ridge_weights']
        intercept = method['ridge_intercept']
        reconstructed = [[(dot(row, column(w, task)) + intercept[task]) * method['target_scale'][task]
                          + method['target_mean'][task] for task in range(4)] for row in x]
        for i in range(112):
            for task in range(4):
                near((reconstructed[i][task] - method['saved_ridge_predictions'][i][task])
                     / method['target_scale'][task], 0, 'Saved prediction reconstruction', 1e-8)
        task_context = []
        for task in range(4):
            y = column(targets, task)
            target_scale, target_mean = method['target_scale'][task], method['target_mean'][task]
            near(target_mean, mean(y), 'Target mean', 1e-12)
            near(target_scale, max(math.sqrt(variance(y)), 1e-8), 'Target scale', 1e-12)
            coefficients = column(w, task)
            standardized_y = [(value - target_mean) / target_scale for value in y]
            standardized_prediction = [dot(row, coefficients) + intercept[task] for row in x]
            residual = [p - yy for p, yy in zip(standardized_prediction, standardized_y)]
            xc = [centered(column(x, j)) for j in range(width)]
            yc = centered(standardized_y)
            kkt = [dot(col, residual) + coefficients[j] for j, col in enumerate(xc)]
            rhs = [dot(col,yc) for col in xc]
            centered_prediction = [dot(row,coefficients) for row in x]
            lhs = [dot(col,centered_prediction)+coefficients[j] for j,col in enumerate(xc)]
            denom = max(1.0,norm(lhs),norm(rhs))
            relative_kkt = norm(kkt) / denom
            require(relative_kkt <= 1e-8, 'Saved normal equations fail independent audit')
            detail = result['methods'][method_index]['task_diagnoses'][task]
            near(detail['normalized_sse'], dot(residual,residual), 'Archived ridge TRAIN SSE')
            near(detail['ridge_weight_squared_norm'], dot(coefficients,coefficients), 'Archived coefficient penalty')
            near(detail['ridge_objective_sse_plus_penalty'], dot(residual,residual)+dot(coefficients,coefficients),
                 'Archived ridge objective')
            near(detail['prefix_coefficient_difference_l2'],
                 norm([a-b for a,b in zip(coefficients[:32],column(parent['ridge_weights'],task))]),
                 'Prefix coefficient shift')
            for start, end, quality_key in [(0,112,'all_train_quality'),
                                           (0,56,'fixed_first_56_rows_quality'),
                                           (56,112,'fixed_last_56_rows_quality')]:
                reference = result['methods'][method_index][quality_key][task]
                require(reference['task'] == TASKS[task], 'Task association')
                ridge = scores(column(reconstructed, task)[start:end], y[start:end])
                for metric in ['r2', 'mae', 'rmse']:
                    near(reference['ridge'][metric], ridge[metric], 'TRAIN ridge score ' + metric)
                each = [scores(column(pred, task)[start:end], y[start:end])
                        for pred in method['saved_neural_predictions_each']]
                require(len(each) == 3, 'Three saved neural predictors required')
                for metric in ['r2', 'mae', 'rmse']:
                    near(reference['neural_mean_repetition_scores'][metric],
                         mean([record[metric] for record in each]), 'Mean saved neural score ' + metric)
            parent_prediction = column(parent['saved_ridge_predictions'], task)
            new_prediction = column(method['saved_ridge_predictions'], task)
            wp = column(parent['ridge_weights'], task)
            prefix_shift = [dot(row[:32], [a-b for a,b in zip(coefficients[:32], wp)]) * target_scale for row in x]
            change = [dot(row[32:], coefficients[32:]) * target_scale for row in x]
            intercept_shift = (intercept[task] - parent['ridge_intercept'][task]) * target_scale
            delta = [new-old for new,old in zip(new_prediction,parent_prediction)]
            for i in range(112):
                near((prefix_shift[i]+change[i]+intercept_shift-delta[i]) / target_scale,
                     0, 'Physical prediction delta decomposition', 1e-8)
            parent_error = [p-yy for p,yy in zip(parent_prediction,y)]
            mse_delta = mean([(p-yy)**2 for p,yy in zip(new_prediction,y)])-mean([e*e for e in parent_error])
            cross = 2*mean([e*d for e,d in zip(parent_error,delta)])
            square = mean([d*d for d in delta])
            near((mse_delta-cross-square)/(target_scale**2),0,'MSE delta identity',1e-8)
            context = detail['all_train_contributions']
            near(context['mse_new_minus_parent_physical'],mse_delta,'Physical MSE delta')
            near(context['twice_mean_parent_error_times_delta_physical'],cross,'Physical MSE cross term')
            near(context['mean_delta_squared_physical'],square,'Physical prediction delta square')
            near(context['change4_contribution']['population_std'],math.sqrt(variance(change)),
                 'Change block contribution standard deviation')
            near(context['prefix_change_physical']['population_std'],math.sqrt(variance(prefix_shift)),
                 'Prefix shift contribution standard deviation')
            level = [dot(row[:32],coefficients[:32])*target_scale for row in x]
            combined = [a+b for a,b in zip(level,change)]
            cancellation = (variance(level)+variance(change))/variance(combined) if variance(combined)>0 else None
            near(context['cancellation_ratio'],cancellation,'Prefix/change cancellation ratio')
            near(context['prefix_change_correlation'],corr(level,change),'Prefix/change contribution correlation')
            near(context['change4_parent_error_correlation'],corr(change,parent_error),
                 'Change contribution vs parent error correlation')
            blocks = [parent_prediction,prefix_shift,change,[intercept_shift]*112]
            for left_index,left in enumerate(blocks):
                for right_index,right in enumerate(blocks):
                    expected_covariance = dot(centered(left),centered(right))/112
                    near(context['delta_decomposition_covariance_physical'][left_index][right_index],
                         expected_covariance,'Physical block covariance')
            if width == 36:
                dw = [a-b for a,b in zip(coefficients[:32],wp)]
                left = [dot(col,[dot(row[:32],dw) for row in x])+dw[j]
                        for j,col in enumerate(xc[:32])]
                cross_vector = [dot(col,[dot(row[32:],coefficients[32:]) for row in x])
                                for col in xc[:32]]
                redistribution = norm([a+b for a,b in zip(left,cross_vector)]) / max(1,norm(left),norm(cross_vector))
                require(redistribution <= 1e-8,'Independent ridge redistribution identity')
                near(detail['prefix_redistribution_relative_l2'],redistribution,
                     'Reader redistribution residual',1e-8)
            task_context.append({'task':TASKS[task], 'independent_KKT_relative_l2':relative_kkt,
                                 'standardized_TRAIN_sse':dot(residual,residual),
                                 'ridge_weight_squared_norm':dot(coefficients,coefficients),
                                 'ridge_objective':dot(residual,residual)+dot(coefficients,coefficients),
                                 'prefix_coefficient_shift_l2':norm([a-b for a,b in zip(coefficients[:32],wp)]),
                                 'physical_prediction_delta_mse':mean([d*d for d in delta]),
                                 'physical_change_population_std':math.sqrt(variance(change)),
                                 'physical_prefix_shift_population_std':math.sqrt(variance(prefix_shift))})
        cross_correlations = [[corr(column(x,j),column(x,k)) for k in range(32,width)] for j in range(32)]
        if width == 36:
            for j in range(32):
                for k in range(4):
                    near(result['methods'][method_index]['prefix32_change4_cross_correlation'][j][k],
                         cross_correlations[j][k],'Level/change correlation')
            for j in range(4):
                for k in range(4):
                    near(result['methods'][method_index]['change4_correlation'][j][k],
                         corr(column(x,32+j),column(x,32+k)),'Change/change correlation')
        contexts.append({'model':method['model'],'tasks':task_context,
                         'independent_cross32x4_correlations':cross_correlations})
    output_data = {'protocol':manifest['protocol'],'status':'passed',
                   'scope':'independent Python standard-library arithmetic from local TRAIN witness; no tensors/models/fits',
                   'checks':CHECKS,'maximum_normalized_difference':MAX_NORMALIZED_DIFFERENCE,
                   'actual_loaded_archives_match_allowlist':True,'loaded_archives':22,
                   'encoder_calls':0,'encoder_updates':0,'head_fits':0,'held_out_tensors':0,
                   'witness_sha256':sha(witness_path),'diagnosis_sha256':sha(result_path),
                   'auditor_sha256':sha(Path(__file__)), 'seconds':time.monotonic()-started,
                   'independent_context':contexts}
    with output.open('x') as stream:
        json.dump(output_data,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:output_data[key] for key in ['status','checks','maximum_normalized_difference','seconds']}))


if __name__ == '__main__':
    main()
