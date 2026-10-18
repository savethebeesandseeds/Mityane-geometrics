#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Compare saved clean-v3/v4 results under the frozen22-condition card.

Pure JSON/hash/report arithmetic: no encoder, readout, fit or training execution.
"""
import argparse
import math
from pathlib import Path
import statistics

from report import (CARD_SHA, INPUT_SHA, GRADIENT_POLICY, TAG, INIT, TASKS,
                    MODES, VIEWS, REPS, MKT, T5, require, load, pin,
                    verify_pin, close, table, dump, write_new, sha)

BASE = 'CPB-v3.market-01'


def market_rows(result, method):
    rows = [row for row in result['market_rows'] if row['method'] == method]
    require(len(rows) == 4 and {r['task'] for r in rows} == set(TASKS), 'Exactly four selected market task rows')
    require(all(r['dataset_id'] == 'MKT-HIST-2' and r['size'] == 36 and r['total'] == 32 and
                r['designed_complexity_level'] == 4 for r in rows), 'Matched native36 market population')
    for row in rows:
        require(all(math.isfinite(row[key]) for key in ('linear_r2', 'neural_mean_r2', 'linear_mae',
                    'linear_rmse', 'neural_mean_mae', 'neural_mean_rmse')), 'Finite physical market measures')
        require(len(row['neural_r2_each']) == 3 and all(math.isfinite(v) for v in row['neural_r2_each']),
                'Three finite fixed neural repetitions')
    return {row['task']: row for row in rows}


def timing_rows(result, method):
    rows = {}
    for panel in result['synthetic']['panels']:
        key = (panel['input_policy'], panel['view'])
        require(key not in rows and key in {(m,v) for m in MODES for v in VIEWS}, 'Exact four timing panels')
        require(panel['dataset_id'] == 'TEMPO-5' and panel['task'] == 'lag_sign' and
                panel['designed_complexity_level'] == 5, 'Same complexity5/5 timing task')
        found = [r for r in panel['rows'] if r['method'] == method]
        require(len(found) == 1 and found[0]['size'] == 36, 'Exactly one selected native36 timing row')
        require(all(math.isfinite(found[0][k]) and 0 <= found[0][k] <= 1 for k in
                    ('linear_head_accuracy', 'neural_head_accuracy', 'coverage')), 'Finite timing fractions')
        rows[key] = found[0]
    require(len(rows) == 4, 'All four timing panels present')
    return rows


def proof_bindings(candidate, table_path):
    origin = table_path.parent
    receipt_path = verify_pin(candidate['quality_execution_receipt'], origin)
    receipt = load(receipt_path)
    require(receipt['status'] == 'passed' and receipt['model'] == TAG and
            receipt['protocol'] == 'causal-fresh-v4' and receipt['card_sha256'] == CARD_SHA,
            'Candidate quality receipt identity')
    build_path = verify_pin(receipt['build_admission'], receipt_path.parent)
    build = load(build_path)
    require(build['status'] == 'passed' and build['card_sha256'] == CARD_SHA and len(build['binaries']) == 6,
            'Six source-admitted candidate binaries')
    for binary in build['binaries']:
        verify_pin(binary, build_path.parent)
    gradient_path = verify_pin(receipt['variance_gradient_proof'], receipt_path.parent)
    require(gradient_path == verify_pin(candidate['variance_gradient_proof'], origin), 'Same admitted gradient proof')
    gradient = load(gradient_path)
    expected = {'status': 'passed', 'model': 'CPB-v4', 'gradient_policy': GRADIENT_POLICY,
                'actual_CUDA': True, 'native_variance_gradient_blocked': True,
                'mean_path_live': True, 'variance_head_learning': True,
                'forward_variance_input_dependence': True, 'manual_partial_gradient_parity': True,
                'nonvacuous_nonzero_fixture': True}
    require(all(gradient[k] == v for k,v in expected.items()), 'Actual CUDA mechanism proof')
    gradient_binary = verify_pin(gradient['binary'], gradient_path.parent)
    gradient_log = verify_pin(gradient['log'], gradient_path.parent)
    require(gradient_binary in {verify_pin(binary,build_path.parent) for binary in build['binaries']} and
            'PASS CPB-v4 variance_input_gradient_stop-v1:' in gradient_log.read_text(encoding='utf-8'),
            'Admitted mechanism binary and actual passing output')
    bound = {'quality_execution_receipt': pin(receipt_path), 'build_admission': pin(build_path),
             'variance_gradient_proof': pin(gradient_path), 'variance_test_binary': pin(gradient_binary),
             'variance_test_log': pin(gradient_log), 'saved_causal_proofs': {}}
    for stage, updates, receipt_key in (('initial',0,'initial_saved_proof'), ('final',2048,'post_training_proof')):
        checkpoint = verify_pin(receipt[stage+'_checkpoint'], receipt_path.parent)
        proof_path = verify_pin(receipt[receipt_key], receipt_path.parent)
        require(proof_path == verify_pin(candidate['proofs'][stage]['proof'], origin), 'Same scored checkpoint proof')
        proof = load(proof_path)
        require(proof['status'] == 'passed' and proof['model'] == 'CPB-v4' and
                proof['gradient_policy'] == GRADIENT_POLICY and
                proof['source_updates'] == 0 and proof['causal_updates'] == updates and
                proof['initialization_seed'] == 943001 and proof['pretrained_ancestor'] == 'none' and
                proof['checkpoint_sha256'] == sha(checkpoint) and
                Path(proof['checkpoint']).resolve(strict=True) == checkpoint and
                proof['all_cutoffs'] == 128 and proof['suffix_cases'] == 2350 and
                proof['internal_temporal_cases'] == 30 and proof['state_unchanged'] and
                proof['future_values_and_masks_invariant'], 'Saved initial/final causal proof and exact checkpoint')
        witness = verify_pin(candidate['proofs'][stage]['witness'], origin)
        detail = load(witness)
        require(detail['status'] == 'passed' and detail['all_cutoffs'] == 128 and
                detail['suffix_cases'] == 2350 and detail['internal_temporal_cases'] == 30 and
                detail['state_unchanged'], 'Bound detailed causal witness')
        bound['saved_causal_proofs'][stage] = {'checkpoint': pin(checkpoint), 'proof': pin(proof_path), 'witness': pin(witness)}
    for key in ('engineering_proof', 'fresh_initialization'):
        path = verify_pin(receipt[key], receipt_path.parent)
        value = load(path)
        require(value['status'] == 'passed', 'Retained candidate trust receipt: '+key)
        if key == 'engineering_proof':
            require(value['gradient_policy'] == GRADIENT_POLICY, 'Engineering proof uses selected gradient policy')
        else:
            summary_path = path.parent/'training-summary.json'
            summary = load(summary_path)
            require(summary == candidate['training'] and summary['gradient_policy'] == GRADIENT_POLICY and
                    summary['encoder_gradient_kind'] == 'partial-NLL-gradient;live-mean-residual;variance-input-stopped' and
                    summary['conditional_variance_head_trainable'] is True,
                    'Actual main trajectory uses selected partial-gradient rule and live variance head')
            bound['training_summary'] = pin(summary_path)
        bound[key] = pin(path)
    return bound


def evaluate(card, baseline, candidate, baseline_market, final_market, baseline_timing, final_timing):
    criteria = card['advancement']
    tolerance = criteria['numeric_tolerance']
    require(tolerance == 1e-12 and not card['criteria']['rescue_allowed'] and
            not card['criteria']['automatic_application_promotion'], 'Frozen tolerance and no rescue/promotion')
    gates = []

    def add(category, identity, actual, reference, rule, passed):
        gates.append({'id': identity, 'category': category, 'candidate': actual,
                      'dataset_id': 'TEMPO-5' if category == 'tempo5_retention' else 'MKT-HIST-2',
                      'baseline_or_floor': reference, 'rule': rule, 'passed': bool(passed)})

    require(len(criteria['market_retention']) == 8 and
            {(g['task'],g['metric']) for g in criteria['market_retention']} ==
            {(t,m) for t in TASKS for m in ('linear_r2','neural_mean_r2')}, 'Exactly eight market retention conditions')
    for spec in criteria['market_retention']:
        task, metric = spec['task'], spec['metric']
        require(spec['rule'] == 'candidate+1e-12>=baseline', 'Frozen market retention rule')
        reference, actual = baseline_market[task][metric], final_market[task][metric]
        close(reference, spec['baseline'], 'Card market floor equals immutable clean v3 result')
        add('market_retention', task+':'+metric, actual, reference, spec['rule'], actual+tolerance >= reference)
    require(len(criteria['tempo5_retention']) == 8 and
            {(s['input_policy'],s['view'],s['metric']) for s in criteria['tempo5_retention']} ==
            {(m,v,h) for m in MODES for v in VIEWS for h in ('linear_head_accuracy','neural_head_accuracy')},
            'Exactly eight timing retention conditions')
    for spec in criteria['tempo5_retention']:
        key, metric = (spec['input_policy'],spec['view']), spec['metric']
        require(spec['rule'] == 'candidate+1e-12>=baseline', 'Frozen timing retention rule')
        reference, actual = baseline_timing[key][metric], final_timing[key][metric]
        close(reference, spec['baseline'], 'Card timing floor equals immutable clean v3 result')
        add('tempo5_retention', ':'.join((*key,metric)), actual, reference, spec['rule'], actual+tolerance >= reference)
    require(len(criteria['fixed_query_MAE']) == 2 and {s['split'] for s in criteria['fixed_query_MAE']} ==
            {'train','validation'}, 'Exactly two fixed query MAE conditions')
    for spec in criteria['fixed_query_MAE']:
        split = spec['split']
        require(spec['rule'] == 'candidate<=baseline+1e-12', 'Frozen MAE rule')
        reference = baseline['fixed_queries']['final'][split]['standardized_fixed_query_MAE']
        actual = candidate['fixed_queries']['final'][split]['standardized_fixed_query_MAE']
        close(reference, spec['baseline'], 'Card query floor equals clean v3')
        add('fixed_query_MAE', split+':standardized_MAE', actual, reference, spec['rule'], actual <= reference+tolerance)
    spec = criteria['variance_quality']
    require(spec['metric'] == 'validation_Gaussian_NLL_F64' and spec['rule'] == 'candidate<baseline-1e-12', 'Strict NLL improvement rule')
    reference = baseline['fixed_queries']['final']['validation']['Gaussian_NLL_F64']
    actual = candidate['fixed_queries']['final']['validation']['Gaussian_NLL_F64']
    close(reference, spec['baseline'], 'Card NLL floor equals clean v3')
    add('variance_quality', 'known_validation:Gaussian_NLL_F64', actual, reference, spec['rule'], actual < reference-tolerance)
    require(criteria['neural_trend_robustness']['repetitions'] == list(REPS) and
            criteria['neural_trend_robustness']['rule'] == 'All3 final neural trend R2 strictly>0', 'Three strict positive trend fits')
    trend = final_market['past_trend']['neural_r2_each']
    add('neural_trend_robustness', 'past_trend:all3_neural_R2_positive', trend, 0.,
        criteria['neural_trend_robustness']['rule'], all(v > 0 for v in trend))
    require(len(criteria['relation_usefulness']) == 2 and
            {s['metric'] for s in criteria['relation_usefulness']} == {'linear_r2','neural_mean_r2'}, 'Two relation usefulness conditions')
    for spec in criteria['relation_usefulness']:
        require(spec['rule'] == 'candidate>0', 'Strict relation usefulness rule')
        actual = final_market['past_relation'][spec['metric']]
        add('relation_usefulness', 'past_relation:'+spec['metric']+':positive', actual, 0., spec['rule'], actual > 0.)
    require(len(gates) == 22 and len({g['id'] for g in gates}) == 22, 'Exactly22 unique prospective numeric conditions')
    return gates


def full_coverage(result, market_sets, timing_sets):
    saved = result['synthetic']['all_saved_repetitions_and_intervals']
    stages = ('final',) if result['model'] == BASE else ('initial','final')
    individual = True
    for stage in stages:
        require(len(saved[stage]) == 3, 'Three saved timing cohorts for coverage')
        for cohort in saved[stage]:
            require(len(cohort['methods']) == 2, 'Two saved scaler policies for coverage')
            for method in cohort['methods']:
                require(len(method['repetitions']) == 3, 'Three saved timing repetitions for coverage')
                for repetition in method['repetitions']:
                    for view in VIEWS:
                        population = repetition[view]['population']
                        individual &= (population['total_rows'] == population['valid_rows'] == 256 and
                                       population['total_source_groups'] == population['valid_source_groups'] == 128)
    return bool(result['full_coverage'] and individual and
                all(r['coverage'] == 1 and r['valid'] == r['total'] == 32 for rows in market_sets for r in rows.values()) and
                all(r['coverage'] == 1 for rows in timing_sets for r in rows.values()))


def display(value):
    if isinstance(value, list):
        return ', '.join(display(x) for x in value)
    if isinstance(value, float):
        return format(value,'.17g')
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('card','candidate-table','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--source',type=Path,help='Source/capsule root containing card-relative baseline control')
    args = parser.parse_args()
    require(Path('/.dockerenv').is_file(), 'Existing managed Linux container only')
    require(sha(args.card) == CARD_SHA, 'Frozen CPB-v4 prospective card')
    card = load(args.card)
    require(card['instance'] == TAG and card['baseline'] == BASE and card['baseline_remains_selected'] and
            not card['historical_reference_is_quality_gate'] and not card['quality']['historical_quality_floors'],
            'Only clean v3 baseline; historical pretrained encoders impose no floors')
    source = (args.source or args.card.resolve(strict=True).parent.parent).resolve(strict=True)
    baseline_path = verify_pin(card['comparison_control'],source)
    baseline = load(baseline_path)
    candidate_path = args.candidate_table.resolve(strict=True)
    candidate = load(candidate_path)
    require(baseline['kind'] == 'causal_fresh_v3_results_v1' and baseline['status'] == 'complete' and
            baseline['model'] == BASE and baseline['source_updates'] == 0 and baseline['trust_protocol_admission_passed'],
            'Immutable completed clean v3 control')
    require(candidate['kind'] == 'causal_fresh_v4_results_v1' and candidate['status'] == 'complete' and
            candidate['model'] == TAG and candidate['initial_label'] == INIT and candidate['source_updates'] == 0 and
            candidate['trust_protocol_admission_passed'] and candidate['clean_baseline'] == BASE and
            candidate['clean_baseline_unchanged'] and candidate['gradient_policy'] == GRADIENT_POLICY and
            not candidate['historical_encoder_quality_targets_used'] and not candidate['application_changed'],
            'Completed fresh candidate with clean baseline unchanged')
    require(verify_pin(candidate['card'],candidate_path.parent) == args.card.resolve(strict=True) and
            candidate['input_manifest']['sha256'] == INPUT_SHA, 'Candidate scored under this frozen card/input manifest')
    verify_pin(candidate['input_manifest'],candidate_path.parent)
    proofs = proof_bindings(candidate,candidate_path)
    baseline_market = market_rows(baseline,BASE)
    initial_market, final_market = market_rows(candidate,INIT), market_rows(candidate,TAG)
    baseline_timing = timing_rows(baseline,BASE)
    initial_timing, final_timing = timing_rows(candidate,INIT), timing_rows(candidate,TAG)
    for task in TASKS:
        row = baseline_market[task]
        require(all(row['unit'] == other[task]['unit'] and row['train_target_mean'] == other[task]['train_target_mean'] and
                    row['train_target_scale'] == other[task]['train_target_scale'] for other in (initial_market,final_market)),
                'Same saved target units and TRAIN scaling across clean comparison')
    gates = evaluate(card,baseline,candidate,baseline_market,final_market,baseline_timing,final_timing)
    categories = {}
    for gate in gates:
        bucket = categories.setdefault(gate['category'],{'passed':0,'total':0})
        bucket['total'] += 1; bucket['passed'] += int(gate['passed'])
    coverage = (full_coverage(candidate,(initial_market,final_market),(initial_timing,final_timing)) and
                full_coverage(baseline,(baseline_market,),(baseline_timing,)))
    passed = sum(g['passed'] for g in gates)
    market = [{**rows[task],'reused': label == BASE} for task in TASKS
              for label,rows in ((BASE,baseline_market),(INIT,initial_market),(TAG,final_market))]
    panels = [{'dataset_id':'TEMPO-5','task':'lag_sign','designed_complexity_level':5,'complexity_scale_max':5,
               'recipe_id':'tempo-5-slow-fast-nine-feature-v1',
               'head_fitting_dataset_id':'TEMPO-5','scoring_dataset_id':'TEMPO-5',
               'input_policy':mode,'view':view,
               'rows':[{**rows[(mode,view)],'reused':label==BASE} for label,rows in
                       ((BASE,baseline_timing),(INIT,initial_timing),(TAG,final_timing))]}
              for mode in MODES for view in VIEWS]
    improvements = []
    for task in TASKS:
        for metric in ('linear_r2','neural_mean_r2','linear_mae','linear_rmse','neural_mean_mae','neural_mean_rmse'):
            before,after = baseline_market[task][metric],final_market[task][metric]
            improvements.append({'task':task,'metric':metric,'v3_reused':before,'v4_final':after,
                                 'difference':after-before,'improved':after>before if metric.endswith('r2') else after<before})
    timing_changes = []
    for mode in MODES:
        for view in VIEWS:
            for metric in ('linear_head_accuracy','neural_head_accuracy'):
                before,after = baseline_timing[(mode,view)][metric],final_timing[(mode,view)][metric]
                timing_changes.append({'input_policy':mode,'view':view,'metric':metric,
                                       'v3_reused':before,'v4_final':after,'difference':after-before,
                                       'improved':after>before})
    report = {'kind':'causal_fresh_v4_clean_baseline_comparison_v1','status':'complete','baseline':BASE,
              'candidate':TAG,'initial_label':INIT,'baseline_unchanged':True,'baseline_scores_reused':True,
              'numeric_conditions':gates,'numeric_passed':passed,'numeric_total':22,'categories':categories,
              'trust_admission_passed':True,'full_coverage':coverage,'advancement_conditions_passed':passed==22 and coverage,
              'overall_winner_declared':False,'automatic_promotion':False,'application_changed':False,
              'historical_pretrained_quality_floors_used':False,'market_rows':market,'synthetic_panels':panels,
              'scoped_market_changes':improvements,'scoped_timing_changes':timing_changes,
              'card':pin(args.card),'baseline_control':pin(baseline_path),
              'candidate_table':pin(candidate_path),'candidate_trust_bindings':proofs,
              'fixed_queries':{BASE:baseline['fixed_queries']['final'],INIT:candidate['fixed_queries']['initial'],TAG:candidate['fixed_queries']['final']},
              'training_cost':{BASE:{'seconds':baseline['training']['GPU_step_seconds_synchronized'],'reused':True},
                               INIT:{'seconds':0,'reused':False},TAG:{'seconds':candidate['training']['GPU_step_seconds_synchronized'],'reused':False}},
              'counters':{'encoder_calls':0,'head_fits':0,'training_updates':0,'post_encoder_PCA_fits':0},
              'source':pin(__file__),'detailed_reporter_source':pin(Path(__file__).with_name('report.py'))}
    lines = ['# CPB-v4 compared with the preserved clean CPB-v3 baseline','',
             f'The candidate passes **{passed}/22** prospective numeric conditions. Trust admission: passed; full coverage: '+str(coverage).lower()+'.',
             '', 'CPB-v3 remains the selected clean baseline. These are known chronological development measurements; '
             'no overall winner, automatic application promotion or forecasting result is declared. V3 measurements are reused '
             'unchanged; both v4 initial and final embeddings have new unchanged-recipe heads. No pretrained v1/v2/RPB score is a floor.', '',
             'Only the variance input gradient is stopped at change4. The conditional variance head still learns and the forward '
             'Gaussian formula is unchanged. Encoder updates use the mean-path partial gradient, not the full joint-NLL gradient. '
             'One fresh 2048/B8 CUDA trajectory uses the same data, TRAIN-only calibration recipe, initialization/data/mask seeds '
             'and native36 fixed-head recipes. No head tuning, additional encoder training or v3 replay is performed by this comparison.', '']
    for task in TASKS:
        rows = [r for r in market if r['task']==task]
        lines.extend(['## '+task.removeprefix('past_').capitalize(),''])
        table(lines,MKT+' '+task+'; v3 reused, v4 initial/final newly measured.',
              ('Method','Size','Linear R²','Neural mean R²','Coverage %'),
              [(r['method'],36,display(r['linear_r2']),display(r['neural_mean_r2']),display(100*r['coverage'])) for r in rows])
        table(lines,MKT+' Physical errors in '+rows[0]['unit']+'.',
              ('Method','Linear MAE','Linear RMSE','Neural mean MAE','Neural mean RMSE'),
              [(r['method'],*(display(r[k]) for k in ('linear_mae','linear_rmse','neural_mean_mae','neural_mean_rmse'))) for r in rows])
        lines.extend([r['method']+' neural R² repetitions 2701/2802/2903: '+display(r['neural_r2_each'])+'.' for r in rows]+[''])
    for panel in panels:
        table(lines,T5+' '+panel['input_policy']+', '+panel['view']+'; v3 reused.',
              ('Method','Size','Linear head %','Neural head %','Coverage %'),
              [(r['method'],36,*(display(100*r[k]) for k in ('linear_head_accuracy','neural_head_accuracy','coverage'))) for r in panel['rows']])
    labels = (BASE,INIT,TAG)
    table(lines,MKT+' Immutable fixed-query standardized MAE. V3 cost is its reused historical 144.341116249s measurement, not a new run.',
          ('Encoder','Updates','Train error ↓','Validation error ↓','GPU training seconds'),
          [(label,0 if label==INIT else 2048,
            display(report['fixed_queries'][label]['train']['standardized_fixed_query_MAE']),
            display(report['fixed_queries'][label]['validation']['standardized_fixed_query_MAE']),
            display(report['training_cost'][label]['seconds'])) for label in labels])
    table(lines,MKT+' Same fixed queries; Gaussian NLL is separate and may be negative.',
          ('Encoder','Updates','TRAIN Gaussian NLL','VALIDATION Gaussian NLL'),
          [(label,0 if label==INIT else 2048,display(report['fixed_queries'][label]['train']['Gaussian_NLL_F64']),
            display(report['fixed_queries'][label]['validation']['Gaussian_NLL_F64'])) for label in labels])
    table(lines,MKT+' '+T5+' All 22 frozen numeric conditions; trust and coverage are separate.',
          ('Dataset','Condition','Candidate','Baseline/floor','Rule','Result'),
          [(g['dataset_id'],g['id'],display(g['candidate']),display(g['baseline_or_floor']),g['rule'],
            'PASS' if g['passed'] else 'FAIL') for g in gates])
    lines.extend(['Scoped gains and regressions are preserved at full precision in `comparison.json`. '
                  'Initial-to-final changes do not establish causation, and retention gates do not create an overall accuracy score. '
                  'Any further experiment needs a separate declaration; this report authorizes no rescue or promotion.',''])
    output = args.output.resolve()
    require(not output.exists(),'Exclusive new comparison output required')
    output.mkdir(parents=True)
    dump(output/'comparison.json',report)
    write_new(output/'report.md','\n'.join(lines)+'\n')
    print(f'Clean comparison complete: {passed}/22 numeric conditions; coverage={coverage}; baseline remains {BASE}')


if __name__ == '__main__':
    main()
