#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Compare saved clean V3/V4/V5 results under the prospectively stronger22-condition card.

Pure JSON/hash/report arithmetic: no encoder, readout, fit or training execution.
"""
import argparse
import math
from pathlib import Path
import statistics

from report import (CARD_SHA, INPUT_SHA, GRADIENT_POLICY, OBJECTIVE_POLICY,
                    ENCODER_GRADIENT_KIND, OBJECTIVE_FIELDS, POSTMEASUREMENT_TRUST_PENDING,
                    objective_contract, TAG, INIT, TASKS,
                    MODES, VIEWS, REPS, MKT, T5, require, load, pin,
                    verify_pin, close, table, dump, write_new, sha)

BASE = 'CPB-v3.market-01'
REF = 'CPB-v4.market-01'
PREVIOUS = 'CPB-v5.market-01'
WIDTHS = {BASE: 36, REF: 36, PREVIOUS: 40, INIT: 40, TAG: 40}


def market_rows(result, method):
    rows = [row for row in result['market_rows'] if row['method'] == method]
    require(len(rows) == 4 and {r['task'] for r in rows} == set(TASKS), 'Exactly four selected market task rows')
    require(all(r['dataset_id'] == 'MKT-HIST-2' and r['size'] == WIDTHS[method] and r['total'] == 32 and
                r['designed_complexity_level'] == 4 for r in rows), 'Declared native36/native40 market populations')
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
        require(len(found) == 1 and found[0]['size'] == WIDTHS[method], 'Exactly one selected dimension-declared timing row')
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
            receipt['protocol'] == 'causal-fresh-v6' and receipt['card_sha256'] == CARD_SHA,
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
    objective_contract(gradient, 'comparison-bound CUDA objective fixture')
    expected = {'status': 'passed', 'model': 'CPB-v6', 'gradient_policy': GRADIENT_POLICY,
                'actual_CUDA': True, 'mean_native_variance_gradient_blocked': True, 'uncertainty_native_variance_gradient_live': True, 'uncertainty_projection_learning': True,
                'mean_path_live': True, 'variance_head_learning': True,
                'forward_variance_input_dependence': True, 'manual_composite_gradient_parity': True,
                'nonvacuous_nonzero_fixture': True,
                'direct_NLL_mean_gradient_zero': True, 'huber_mean_gradient_live': True,
                'direct_Huber_uncertainty_gradient_zero': True, 'shared_clip_coupling_witness': True}
    require(all(gradient[k] == v for k,v in expected.items()), 'Actual CUDA mechanism proof')
    gradient_binary = verify_pin(gradient['binary'], gradient_path.parent)
    gradient_log = verify_pin(gradient['log'], gradient_path.parent)
    require(gradient_binary in {verify_pin(binary,build_path.parent) for binary in build['binaries']} and
            'PASS CPB-v6 huber1-mean-plus-detached-error-gaussian-variance-v1:' in gradient_log.read_text(encoding='utf-8'),
            'Admitted mechanism binary and actual passing output')
    bound = {'quality_execution_receipt': pin(receipt_path), 'build_admission': pin(build_path),
             'variance_gradient_proof': pin(gradient_path), 'variance_test_binary': pin(gradient_binary),
             'variance_test_log': pin(gradient_log), 'saved_causal_proofs': {}}
    for stage, updates, receipt_key in (('initial',0,'initial_saved_proof'), ('final',2048,'post_training_proof')):
        checkpoint = verify_pin(receipt[stage+'_checkpoint'], receipt_path.parent)
        proof_path = verify_pin(receipt[receipt_key], receipt_path.parent)
        require(proof_path == verify_pin(candidate['proofs'][stage]['proof'], origin), 'Same scored checkpoint proof')
        proof = load(proof_path)
        objective_contract(proof, 'comparison-bound saved causal proof')
        require(proof['status'] == 'passed' and proof['model'] == 'CPB-v6' and
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
            objective_contract(value, 'comparison-bound CUDA engineering proof')
            require(value['gradient_policy'] == GRADIENT_POLICY, 'Engineering proof uses selected gradient policy')
        else:
            summary_path = path.parent/'training-summary.json'
            summary = load(summary_path)
            objective_contract(summary, 'comparison-bound training summary')
            require(summary == candidate['training'] and summary['gradient_policy'] == GRADIENT_POLICY and
                    summary['encoder_gradient_kind'] == ENCODER_GRADIENT_KIND and
                    summary['conditional_variance_head_trainable'] is True,
                    'Actual main trajectory uses the selected composite Huber/detached-error variance rule and live variance head')
            bound['training_summary'] = pin(summary_path)
        bound[key] = pin(path)
    return bound


GOAL_SHA = '47fa165d325230ffca4f99c19bbec494d04f1daf0d99ae7e6bd197fdc47bec0d'

def evaluate(goal, candidate, market, timing):
    values = {task+':'+metric:market[task][metric] for task in TASKS for metric in ('linear_r2','neural_mean_r2')}
    values.update({mode+':'+view+':'+metric:timing[(mode,view)][metric] for mode in MODES for view in VIEWS for metric in ('linear_head_accuracy','neural_head_accuracy')})
    values.update({role+':standardized_MAE':candidate['fixed_queries']['final'][role]['standardized_fixed_query_MAE'] for role in ('train','validation')})
    values['known_validation:Gaussian_NLL_F64']=candidate['fixed_queries']['final']['validation']['Gaussian_NLL_F64']
    values['past_trend:all3_neural_R2_positive']=market['past_trend']['neural_r2_each']
    values['past_relation:linear_r2:positive']=market['past_relation']['linear_r2']
    values['past_relation:neural_mean_r2:positive']=market['past_relation']['neural_mean_r2']
    criteria=goal['criteria']
    require(len(criteria)==22 and len({c['id'] for c in criteria})==22 and set(values)=={c['id'] for c in criteria},'Exactly the22 authoritative goal identities')
    result=[]
    for c in criteria:
        value,floor,rule=values[c['id']],c['floor'],c['rule']
        if rule=='candidate + 1e-12 >= floor':
            passed=value+1e-12>=floor
        elif rule=='candidate <= floor + 1e-12':
            passed=value<=floor+1e-12
        elif rule=='candidate < floor - 1e-12':
            passed=value<floor-1e-12
        elif rule=='candidate > 0':
            require(floor==0,'Strict positivity floor');passed=value>0
        elif rule=='all three final neural trend R2 values > 0':
            require(floor==0 and len(value)==3,'Three frozen trend heads');passed=all(v>0 for v in value)
        else:
            raise RuntimeError('Unknown frozen goal rule: '+rule)
        require(all(math.isfinite(v) for v in value) if isinstance(value,list) else math.isfinite(value),'Finite candidate criterion')
        result.append({**c,'candidate':value,'passed':passed})
    return result

def full_coverage(result, markets, timings):
    return result['full_coverage'] and all(row['coverage']==1 and row['valid']==row['total']==32 for rows in markets for row in rows.values()) and all(row['coverage']==1 for rows in timings for row in rows.values())

def display(value):
    if isinstance(value,list):return ', '.join(format(v,'.9g') for v in value)
    if isinstance(value,float):return format(value,'.9g')
    return str(value)

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('card','candidate-table','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    require(Path('/.dockerenv').is_file(),'Existing managed container only')
    require(sha(args.card)==CARD_SHA,'Frozen v6 prospective card')
    card=load(args.card);source=(args.source or args.card.resolve(strict=True).parent.parent).resolve(strict=True)
    require(card['instance']==TAG and card['baseline']==BASE and card['scoped_references']==[REF,PREVIOUS] and card['baseline_remains_selected'] and not card['historical_reference_is_quality_gate'] and not card['quality']['historical_quality_floors'],'All three clean references protected; no historical oracle floors')
    goal_pin=card['advancement']['goal_protocol']
    require(goal_pin['sha256']==GOAL_SHA and card['advancement']['numeric_tolerance']==1e-12 and not card['criteria']['rescue_allowed'] and not card['criteria']['automatic_application_promotion'],'Frozen search goal and no rescue/promotion')
    goal_path=verify_pin({**goal_pin,'path':'doc/causal_fresh_v6/goal-protocol.json'},source);goal=load(goal_path)
    require(goal['kind']=='clean_causal_encoder_search_goal_v5_retention_protocol_v1' and goal['criterion_count']==22 and goal['criteria']==card['advancement']['criteria'] and goal['additional_train_NLL_gate'] is False,'Card criteria exactly preserve the separately reviewed V5-retention goal; no derived or extra floors')
    controls={};control_pins={}
    for key,label in (('v3',BASE),('v4',REF),('v5',PREVIOUS)):
        path=verify_pin(card['comparison_control'][key],source);r=load(path)
        require(r['kind']=='causal_fresh_'+key+'_results_v1' and r['status']=='complete' and r['model']==label and r['source_updates']==0 and r['trust_protocol_admission_passed'] and r['full_coverage'],'Preserved complete clean reference')
        controls[label]=r;control_pins[label]=pin(path)
    candidate_path=args.candidate_table.resolve(strict=True);candidate=load(candidate_path)
    objective_contract(candidate, 'measured candidate table')
    require(candidate['kind']=='causal_fresh_v6_results_v1' and candidate['status']=='complete' and candidate['model']==TAG and candidate['initial_label']==INIT and candidate['source_updates']==0 and candidate['runtime_trust_admission_passed'] and candidate['trust_protocol_admission_passed'] is False and candidate['mandatory_postmeasurement_trust']==POSTMEASUREMENT_TRUST_PENDING and candidate['overall_establishment_passed'] is False and candidate['clean_baseline']==BASE and candidate['clean_baseline_unchanged'] and candidate['gradient_policy']==GRADIENT_POLICY and not candidate['historical_encoder_quality_targets_used'] and not candidate['application_changed'],'Runtime-admitted fresh native40 candidate; mandatory independent audits/pairing still pending')
    require(verify_pin(candidate['card'],candidate_path.parent)==args.card.resolve(strict=True) and candidate['input_manifest']['sha256']==INPUT_SHA,'Candidate card/input identity');verify_pin(candidate['input_manifest'],candidate_path.parent)
    proofs=proof_bindings(candidate,candidate_path)
    labels=(BASE,REF,PREVIOUS,INIT,TAG)
    result_by_label={**controls,INIT:candidate,TAG:candidate}
    markets={label:market_rows(result_by_label[label],label) for label in labels}
    timings={label:timing_rows(result_by_label[label],label) for label in labels}
    for task in TASKS:
        reference=markets[BASE][task]
        require(all(reference['unit']==markets[label][task]['unit'] and reference['train_target_mean']==markets[label][task]['train_target_mean'] and reference['train_target_scale']==markets[label][task]['train_target_scale'] for label in labels),'Unchanged physical target units and TRAIN scaling')
    gates=evaluate(goal,candidate,markets[TAG],timings[TAG]);categories={}
    for gate in gates:
        bucket=categories.setdefault(gate['category'],{'passed':0,'total':0});bucket['total']+=1;bucket['passed']+=int(gate['passed'])
    require({k:v['total'] for k,v in categories.items()}=={'market_retention':8,'tempo5_retention':8,'fixed_query_MAE':2,'variance_quality':1,'neural_trend_robustness':1,'relation_usefulness':2},'Exact22 category cardinality')
    coverage=all(full_coverage(result_by_label[label],(markets[label],),(timings[label],)) for label in labels)
    passed=sum(g['passed'] for g in gates)
    market=[{**markets[label][task],'reused':label in controls} for task in TASKS for label in labels]
    panels=[{'dataset_id':'TEMPO-5','task':'lag_sign','designed_complexity_level':5,'complexity_scale_max':5,'recipe_id':'tempo-5-slow-fast-nine-feature-v1','head_fitting_dataset_id':'TEMPO-5','scoring_dataset_id':'TEMPO-5','input_policy':mode,'view':view,'rows':[{**timings[label][(mode,view)],'reused':label in controls} for label in labels]} for mode in MODES for view in VIEWS]
    changes=[{'reference':label,'task':task,'metric':metric,'reused_reference':markets[label][task][metric],'candidate':markets[TAG][task][metric],'difference':markets[TAG][task][metric]-markets[label][task][metric],'improved':markets[TAG][task][metric]>markets[label][task][metric] if metric.endswith('r2') else markets[TAG][task][metric]<markets[label][task][metric]} for label in controls for task in TASKS for metric in ('linear_r2','neural_mean_r2','linear_mae','linear_rmse','neural_mean_mae','neural_mean_rmse')]
    timing_changes=[{'reference':label,'input_policy':mode,'view':view,'metric':metric,'reused_reference':timings[label][(mode,view)][metric],'candidate':timings[TAG][(mode,view)][metric],'difference':timings[TAG][(mode,view)][metric]-timings[label][(mode,view)][metric]} for label in controls for mode in MODES for view in VIEWS for metric in ('linear_head_accuracy','neural_head_accuracy')]
    queries={label:result_by_label[label]['fixed_queries']['initial' if label==INIT else 'final'] for label in labels}
    cost={label:{'seconds':0 if label==INIT else result_by_label[label]['training']['GPU_step_seconds_synchronized'],'reused':label in controls} for label in labels}
    report={'kind':'causal_fresh_v6_clean_baseline_comparison_v1','status':'complete','baseline':BASE,'scoped_reference':REF,'scoped_references':[REF,PREVIOUS],**OBJECTIVE_FIELDS,'candidate':TAG,'initial_label':INIT,'baseline_unchanged':True,'baseline_scores_reused':True,'numeric_conditions':gates,'numeric_passed':passed,'numeric_total':22,'categories':categories,'runtime_trust_admission_passed':True,'trust_admission_passed':False,'mandatory_postmeasurement_trust':POSTMEASUREMENT_TRUST_PENDING,'full_coverage':coverage,'numeric_conditions_passed':passed==22,'numeric_and_coverage_conditions_passed':passed==22 and coverage,'advancement_conditions_passed':False,'overall_establishment_passed':False,'overall_winner_declared':False,'automatic_promotion':False,'application_changed':False,'historical_pretrained_quality_floors_used':False,'market_rows':market,'synthetic_panels':panels,'scoped_market_changes':changes,'scoped_timing_changes':timing_changes,'card':pin(args.card),'goal_protocol':pin(goal_path),'baseline_controls':control_pins,'candidate_table':pin(candidate_path),'candidate_trust_bindings':proofs,'fixed_queries':queries,'training_cost':cost,'capacity_confounds':'Native40 forward capacity,294752 learned values and D40 seeds/heads match V5. Relative to V3/V4 native36,1080 learned values and larger width-derived-seed readouts remain capacity confounds. One fresh trajectory and known development populations do not establish generalization or isolate every training cause.','counters':{'encoder_calls':0,'head_fits':0,'training_updates':0,'post_encoder_PCA_fits':0},'source':pin(__file__),'detailed_reporter_source':pin(Path(__file__).with_name('report.py'))}
    lines=['# CPB-v6 compared with all three preserved clean references','',f'Candidate passes **{passed}/22** prospectively strengthened V5-retention conditions. Runtime admission: passed; mandatory initial-pairing/training/quality audits: pending; overall advancement: pending. Full coverage: '+str(coverage).lower()+'.','', 'CPB-v3 remains selected; all V4/V5 scoped gains remain protected. V3/V4/V5 final rows and training costs are reused unchanged. V6 initial0/final2048 and their own fixed heads are new. Historical pass counts used their own frozen floors and are not recomputed or ranked as identical protocols. Numeric conditions and coverage are descriptive outcomes here; overall trust/advancement require separately bound initial pairing and independent training/quality audits. No overall winner, automatic promotion, forecasting or fresh-period confirmation follows.','',report['capacity_confounds'],'','The fixed Huber(delta1) mean objective is unweighted by variance; Gaussian variance calibration uses detached error, with coefficients1 each and a live uncertainty4 branch. Reported Gaussian NLL/MAE remain unchanged metrics. One global clip1/AdamW and future mean residuals retain coupling; this is not complete optimizer independence or the full joint-NLL mean gradient. All learned values/statistics start fresh; no old model execution or head tuning.','']
    for task in TASKS:
        rows=[markets[label][task] for label in labels];lines.extend(['## '+task.removeprefix('past_').capitalize(),''])
        table(lines,MKT+' '+task+'; V3/V4/V5 final rows reused, V6 initial/final newly measured.',('Method','Size','Linear R²','Neural mean R²','Coverage %'),[(r['method'],r['size'],display(r['linear_r2']),display(r['neural_mean_r2']),display(100*r['coverage'])) for r in rows])
        table(lines,MKT+' Physical errors in '+rows[0]['unit']+'.',('Method','Linear MAE','Linear RMSE','Neural mean MAE','Neural mean RMSE'),[(r['method'],*(display(r[k]) for k in ('linear_mae','linear_rmse','neural_mean_mae','neural_mean_rmse'))) for r in rows])
        lines.extend([r['method']+' neural R²2701/2802/2903: '+display(r['neural_r2_each'])+'.' for r in rows]+[''])
    for panel in panels:
        table(lines,T5+' '+panel['input_policy']+', '+panel['view']+'; V3/V4/V5 reused.',('Method','Size','Linear head %','Neural head %','Coverage %'),[(r['method'],r['size'],*(display(100*r[k]) for k in ('linear_head_accuracy','neural_head_accuracy','coverage'))) for r in panel['rows']])
    table(lines,MKT+' Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.',('Method','Updates','TRAIN MAE','VALIDATION MAE','GPU training seconds'),[(label,0 if label==INIT else 2048,display(queries[label]['train']['standardized_fixed_query_MAE']),display(queries[label]['validation']['standardized_fixed_query_MAE']),display(cost[label]['seconds'])+(' (reused)' if label in controls else '')) for label in labels])
    table(lines,MKT+' Same query distributions; NLL omits constant and can be negative.',('Method','TRAIN Gaussian NLL','VALIDATION Gaussian NLL'),[(label,display(queries[label]['train']['Gaussian_NLL_F64']),display(queries[label]['validation']['Gaussian_NLL_F64'])) for label in labels])
    table(lines,MKT+' '+T5+' All22 frozen numeric conditions; trust and coverage are separate.',('Dataset','Condition','Candidate','Frozen floor/ceiling','Source','Rule','Pass'),[(g['dataset_id'],g['id'],display(g['candidate']),display(g['floor']),g['floor_source'],g['rule'],'PASS' if g['passed'] else 'FAIL') for g in gates])
    lines.extend(['The exact declared trajectory is stopped at2048. Retain every failed condition and all three references and their scoped strengths. Further mechanisms require separate prospective declarations; no coefficient/seed/rate/budget/head rescue.',''])
    require(not args.output.exists(),'Preserve all comparison attempts; exclusive output required');args.output.mkdir(parents=True)
    dump(args.output/'comparison.json',report);write_new(args.output/'report.md','\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
