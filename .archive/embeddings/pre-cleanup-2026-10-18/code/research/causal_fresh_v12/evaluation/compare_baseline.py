#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Compare saved clean V3/V4/V5/V6/V7/V8/V9/V10/V11 results under the frozen22-condition card.

Pure JSON/hash/report arithmetic: no encoder, readout, fit or training execution.
"""
import argparse
import math
from pathlib import Path
import statistics
import report as detailed

from report import (CARD_SHA, INPUT_SHA, GRADIENT_POLICY, OBJECTIVE_POLICY,
                    ENCODER_GRADIENT_KIND, OBJECTIVE_FIELDS, POSTMEASUREMENT_TRUST_PENDING,
                    TEMPORAL_POLICY, TEMPORAL_PARTITION, TEMPORAL_FIELDS, LEARNING_FIELDS,
                    objective_contract, temporal_contract, learning_contract, TAG, INIT, TASKS,
                    MODES, VIEWS, REPS, MKT, T5, require, load, pin,
                    verify_pin, close, table, dump, write_new, sha)

BASE = 'CPB-v3.market-01'
REF = 'CPB-v4.market-01'
PREVIOUS = 'CPB-v5.market-01'
PARENT = 'CPB-v6.market-01'
LATEST = 'CPB-v7.market-01'
RECENT = 'CPB-v8.market-01'
RECIPE_PARENT = 'CPB-v9.market-01'
PRIOR = 'CPB-v10.market-01'
CLOSEST = 'CPB-v11.market-01'
WIDTHS = {BASE: 36, REF: 36, PREVIOUS: 40, PARENT: 40, LATEST: 40, RECENT: 40, RECIPE_PARENT: 40, PRIOR: 40, CLOSEST: 40, INIT: 40, TAG: 40}


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
            receipt['protocol'] == 'causal-fresh-v12' and receipt['card_sha256'] == CARD_SHA,
            'Candidate quality receipt identity')
    build_path = verify_pin(receipt['build_admission'], receipt_path.parent)
    build = load(build_path)
    require(build['status'] == 'passed' and build['card_sha256'] == CARD_SHA and len(build['binaries']) == 7,
            'Seven source-admitted candidate binaries')
    detailed.admitted_source(build, build_path, __file__)
    for binary in build['binaries']:
        verify_pin(binary, build_path.parent)
    temporal = detailed.temporal_proof(receipt['temporal_pooling_proof'], receipt_path.parent,
        {verify_pin(binary, build_path.parent) for binary in build['binaries']})
    require(verify_pin(candidate['temporal_pooling_proof'], origin) == verify_pin(temporal['proof'], origin) and
            candidate['temporal_mechanism_bindings'] == temporal,
            'Same separate admitted temporal mechanism proof and passing binary/log')
    gradient_path = verify_pin(receipt['variance_gradient_proof'], receipt_path.parent)
    require(gradient_path == verify_pin(candidate['variance_gradient_proof'], origin), 'Same admitted gradient proof')
    gradient = load(gradient_path)
    objective_contract(gradient, 'comparison-bound CUDA objective fixture')
    expected = {'status': 'passed', 'model': 'CPB-v12', 'gradient_policy': GRADIENT_POLICY,
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
            'PASS CPB-v12 huber1-mean-plus-detached-error-gaussian-variance-v1:' in gradient_log.read_text(encoding='utf-8'),
            'Admitted mechanism binary and actual passing output')
    bound = {'quality_execution_receipt': pin(receipt_path), 'build_admission': pin(build_path),
             'variance_gradient_proof': pin(gradient_path), 'variance_test_binary': pin(gradient_binary),
             'variance_test_log': pin(gradient_log), 'temporal_pooling_proof': temporal,
             'saved_causal_proofs': {}}
    for stage, updates, receipt_key in (('initial',0,'initial_saved_proof'), ('final',2048,'post_training_proof')):
        checkpoint = verify_pin(receipt[stage+'_checkpoint'], receipt_path.parent)
        proof_path = verify_pin(receipt[receipt_key], receipt_path.parent)
        require(proof_path == verify_pin(candidate['proofs'][stage]['proof'], origin), 'Same scored checkpoint proof')
        proof = load(proof_path)
        objective_contract(proof, 'comparison-bound saved causal proof')
        temporal_contract(proof, 'comparison-bound saved causal proof')
        require(proof['status'] == 'passed' and proof['model'] == 'CPB-v12' and
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
            temporal_contract(value, 'comparison-bound CUDA engineering proof')
            learning_contract(value, 'comparison-bound CUDA engineering proof')
            require(value['gradient_policy'] == GRADIENT_POLICY, 'Engineering proof uses selected gradient policy')
        else:
            summary_path = path.parent/'training-summary.json'
            summary = load(summary_path)
            objective_contract(summary, 'comparison-bound training summary')
            temporal_contract(summary, 'comparison-bound training summary')
            learning_contract(summary, 'comparison-bound training summary')
            require(summary == candidate['training'] and summary['gradient_policy'] == GRADIENT_POLICY and
                    summary['encoder_gradient_kind'] == ENCODER_GRADIENT_KIND and
                    summary['conditional_variance_head_trainable'] is True,
                    'Actual main trajectory uses the selected Huber plus detached-error variance rule and live variance head')
            bound['training_summary'] = pin(summary_path)
            learned = detailed.temporal_learning(summary, summary_path.parent)
            require(learned == candidate['temporal_learning'], 'Same actual temporal/gauge learning evidence')
            bound['temporal_learning'] = learned
        bound[key] = pin(path)
    return bound


GOAL_SHA = '866dec8a2d7fc3ab36383435295c785e71eea84ea701276cb6552ed4999a603b'

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
    global CARD_SHA, INPUT_SHA
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('card','candidate-table','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    require(Path('/.dockerenv').is_file(),'Existing managed container only')
    require(Path('/.dockerenv').is_file(),'Existing managed container only')
    card=detailed.configure_card(args.card);CARD_SHA,INPUT_SHA=detailed.CARD_SHA,detailed.INPUT_SHA
    source=(args.source or args.card.resolve(strict=True).parent.parent).resolve(strict=True)
    require(card['instance']==TAG and card['baseline']==BASE and card['scoped_references']==[REF,PREVIOUS,PARENT,LATEST,RECENT,RECIPE_PARENT,PRIOR,CLOSEST] and card['baseline_remains_selected'] and not card['historical_reference_is_quality_gate'] and not card['quality']['historical_quality_floors'],'All nine clean references protected; no historical oracle floors')
    goal_pin=card['advancement']['goal_protocol']
    require(goal_pin['sha256']==GOAL_SHA and card['advancement']['numeric_tolerance']==1e-12 and not card['criteria']['rescue_allowed'] and not card['criteria']['automatic_application_promotion'],'Frozen search goal and no rescue/promotion')
    goal_path=verify_pin({**goal_pin,'path':'doc/causal_fresh_v12/goal-protocol.json'},source);goal=load(goal_path)
    require(goal['kind']=='clean_causal_encoder_search_goal_v11_retention_protocol_v1' and goal['criterion_count']==22 and goal['criteria']==card['advancement']['criteria'] and goal['additional_train_NLL_gate'] is False,'Card criteria exactly preserve the reviewed V11-retention goal; no derived or extra floors')
    retention_review_path=verify_pin(card['retention_independent_review'],source)
    retention_admission_path=verify_pin(card['retention_application_admission'],source)
    retention_review,retention_admission=load(retention_review_path),load(retention_admission_path)
    require(sha(retention_review_path)=='9d15d8df155460787270df8fedc75f08ea31ad1381d6061a73e264343c5c78ee' and
            sha(retention_admission_path)=='e59f8eeaa4993ec116eca2751c2cbb9a957c6ae81b95047917b71e01485f3035' and
            retention_review['status']=='passed' and retention_review['checks']==723 and
            retention_review['protocol']['sha256']==GOAL_SHA and
            retention_admission['status']=='passed' and retention_admission['prospective_future_use_eligible'] is True and
            retention_admission['applied'] is True and
            retention_admission['independent_review']['sha256']==sha(retention_review_path) and
            any(binding['sha256']==GOAL_SHA and binding['bytes']==goal_path.stat().st_size for binding in retention_admission['files']) and
            retention_admission['additional_TRAIN_NLL_gate'] is False,
            'Separate reviewed/application admission permits the frozen future retention protocol')
    controls={};control_pins={};establishment_pins={}
    for key,label in (('v3',BASE),('v4',REF),('v5',PREVIOUS),('v6',PARENT),('v7',LATEST),('v8',RECENT),('v9',RECIPE_PARENT),('v10',PRIOR),('v11',CLOSEST)):
        path=verify_pin(card['comparison_control'][key],source);r=load(path)
        require(r['kind']=='causal_fresh_'+key+'_results_v1' and r['status']=='complete' and r['model']==label and r['source_updates']==0 and r['full_coverage'],'Preserved complete clean reference')
        if key in ('v6','v7','v8','v9','v10','v11'):
            establishment_path=verify_pin(card['comparison_control'][key+'_establishment'],source);establishment=load(establishment_path)
            require(establishment['status']=='passed' and establishment['candidate']==label and
                    establishment['measured_candidate_established'] and establishment['trust_admission_passed'] and
                    establishment['initial_partial_native20_40_pairings_passed' if key in ('v8','v9','v11') else 'initial_full40_pairings_passed'] and establishment['initial_pairing_count']==13 and
                    establishment['independent_training_audit_passed'] and establishment['independent_quality_audit_passed'] and
                    establishment['full_coverage'] and establishment['evidence']['table_rows']['sha256']==sha(path) and
                    r['runtime_trust_admission_passed'] and r['trust_protocol_admission_passed'] is False,
                    'Reference separately established full trust; original pending raw table preserved')
            establishment_pins[label]=pin(establishment_path)
            if key == 'v9':
                recovery=establishment['report_only_recovery']
                require(recovery['status']=='passed' and
                        recovery['card']['sha256']=='9c61e4517bc74f0e6a9327ba7f8ec11908b1c1c4f0f804bd7ac76a51631aa82d' and
                        recovery['receipt']['sha256']=='5f96dfab1b22a6c2df7f4bd8a0f55eeb0cb15ef072b05f80473707a31c63de72' and
                        recovery['science_stages_replayed']==0,
                        'V9 remains a separately established historical reporting-only recovery, with its failed original stages preserved')
        else:
            require(r['trust_protocol_admission_passed'],'Preserved established clean reference')
        controls[label]=r;control_pins[label]=pin(path)
    candidate_path=args.candidate_table.resolve(strict=True);candidate=load(candidate_path)
    objective_contract(candidate, 'measured candidate table')
    temporal_contract(candidate, 'measured candidate table')
    learning_contract(candidate, 'measured candidate table')
    require(candidate['kind']=='causal_fresh_v12_results_v1' and candidate['status']=='complete' and candidate['model']==TAG and candidate['initial_label']==INIT and candidate['source_updates']==0 and candidate['runtime_trust_admission_passed'] and candidate['trust_protocol_admission_passed'] is False and candidate['mandatory_postmeasurement_trust']==POSTMEASUREMENT_TRUST_PENDING and candidate['overall_establishment_passed'] is False and candidate['clean_baseline']==BASE and candidate['clean_baseline_unchanged'] and candidate['gradient_policy']==GRADIENT_POLICY and not candidate['historical_encoder_quality_targets_used'] and not candidate['application_changed'],'Runtime-admitted fresh native40 candidate; mandatory independent audits/pairing still pending')
    require(verify_pin(candidate['card'],candidate_path.parent)==args.card.resolve(strict=True) and candidate['input_manifest']['sha256']==INPUT_SHA,'Candidate card/input identity');verify_pin(candidate['input_manifest'],candidate_path.parent)
    proofs=proof_bindings(candidate,candidate_path)
    labels=(BASE,REF,PREVIOUS,PARENT,LATEST,RECENT,RECIPE_PARENT,PRIOR,CLOSEST,INIT,TAG)
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
    report={'kind':'causal_fresh_v12_clean_baseline_comparison_v1','status':'complete','baseline':BASE,'scoped_reference':CLOSEST,'scoped_reference_role':'latest measured historical control,not best/winner; source-only parent','scientific_recipe_parent':CLOSEST,'scientific_recipe_parent_is_source_only':True,'pairing_reference':'CPB-v11.market-01 saved initial DATA/statistics only','scoped_references':[REF,PREVIOUS,PARENT,LATEST,RECENT,RECIPE_PARENT,PRIOR,CLOSEST],**OBJECTIVE_FIELDS,**TEMPORAL_FIELDS,**LEARNING_FIELDS,'candidate':TAG,'initial_label':INIT,'baseline_unchanged':True,'baseline_scores_reused':True,'numeric_conditions':gates,'numeric_passed':passed,'numeric_total':22,'categories':categories,'runtime_trust_admission_passed':True,'trust_admission_passed':False,'mandatory_postmeasurement_trust':POSTMEASUREMENT_TRUST_PENDING,'full_coverage':coverage,'numeric_conditions_passed':passed==22,'numeric_and_coverage_conditions_passed':passed==22 and coverage,'advancement_conditions_passed':False,'overall_establishment_passed':False,'overall_winner_declared':False,'automatic_promotion':False,'application_changed':False,'historical_pretrained_quality_floors_used':False,'market_rows':market,'synthetic_panels':panels,'scoped_market_changes':changes,'scoped_timing_changes':timing_changes,'card':pin(args.card),'goal_protocol':pin(goal_path),'baseline_controls':control_pins,'baseline_establishment_controls':establishment_pins,'candidate_table':pin(candidate_path),'candidate_trust_bindings':proofs,'fixed_queries':queries,'training_cost':cost,'capacity_confounds':'Native40,294752 learned values,declared constructor draws,D40 readout seeds/recipes and2048/B8 remain fixed; actual constructor parity requires saved evidence. The V11 SOURCE shape/objective is freshly constructed. Attention mean4 is unchanged; dispersion4 uses fixed half-local/half-global uniform ORIGINAL support and its OWN live two-sum center. Its direct score/position derivative is zero; hidden and Transformer attention gradients remain live. These intermediate16 variance rows feed the unchangedglobal32-to-content20 map;they are not appended nativecoordinates or the decoder uncertainty4. Routing,scales and FLOPs change;no performance/cause claim is assumed. Initial DATA20:40 and five whole fresh-statistic archives pair against savedV11,while learnedcontent0:20 is allowed to change. All initial/finalheads are new. V11 is the latest measured control,source-only parent, not a winner. All other historicalcounts retain their owncards. No oldweights/model/headreplay,postPCA or tuning.','counters':{'encoder_calls':0,'head_fits':0,'training_updates':0,'post_encoder_PCA_fits':0},'source':pin(__file__),'detailed_reporter_source':pin(Path(__file__).with_name('report.py'))}
    lines=['# CPB-v12 compared with all nine preserved clean references','',f'Candidate passes **{passed}/22** prospectively strengthened V11-retention conditions. Runtime admission: passed; mandatory partial20:40/statistic-pairing/training/quality audits: pending; overall advancement: pending. Full coverage: '+str(coverage).lower()+'.','', 'CPB-v3 remains selected; all V4/V5/V6/V7/V8/V9/V10/V11 scoped gains remain protected. V3/V4/V5/V6/V7/V8/V9/V10/V11 final rows and training costs are reused unchanged. V12 initial0/final2048 and their own fixed heads are new. Historical pass counts used their own frozen floors and are not recomputed or ranked as identical protocols. Numeric conditions and coverage are descriptive outcomes here; overall trust/advancement require separately bound native20:40 initial DATA/whole-statistic pairing and independent training/quality audits. No overall winner, automatic promotion, forecasting or fresh-period confirmation follows.','',report['capacity_confounds'],'','V12 retains the V11 unit Huber1 plus detached-error variance-calibration objective, one globalclip1/singleAdamW. The summary keeps attention mean4 and uses own-center uniform dispersion4 per original bin;the global32-to-content20 map and existing nativeuncertainty4 remain. Emptylocalbins zero the whole8-row slice afterbias. No nativevariancecoordinates are appended. All learned values/statistics startfresh;all eight mean/variance subgroups require actual data-gradient/movement events and finalchanges,not every scalar/update. Relativebin/gauge/causal and resumeproofs remain mandatory. Only initial20:40 plus five whole TRAINstatistics pair versus savedV11. No scientific or quality replay,heldout cause,generalization or improvement is claimed. Complexity4/5 and5/5 are ordinal recipes,not observed difficulty.','']
    for task in TASKS:
        rows=[markets[label][task] for label in labels];lines.extend(['## '+task.removeprefix('past_').capitalize(),''])
        table(lines,MKT+' '+task+'; V3/V4/V5/V6/V7/V8/V9/V10/V11 final rows reused, V12 initial/final newly measured.',('Method','Size','Linear R²','Neural mean R²','Coverage %'),[(r['method'],r['size'],display(r['linear_r2']),display(r['neural_mean_r2']),display(100*r['coverage'])) for r in rows])
        table(lines,MKT+' Physical errors in '+rows[0]['unit']+'.',('Method','Linear MAE','Linear RMSE','Neural mean MAE','Neural mean RMSE'),[(r['method'],*(display(r[k]) for k in ('linear_mae','linear_rmse','neural_mean_mae','neural_mean_rmse'))) for r in rows])
        lines.extend([r['method']+' neural R²2701/2802/2903: '+display(r['neural_r2_each'])+'.' for r in rows]+[''])
    for panel in panels:
        table(lines,T5+' '+panel['input_policy']+', '+panel['view']+'; V3/V4/V5/V6/V7/V8/V9/V10/V11 reused.',('Method','Size','Linear head %','Neural head %','Coverage %'),[(r['method'],r['size'],*(display(100*r[k]) for k in ('linear_head_accuracy','neural_head_accuracy','coverage'))) for r in panel['rows']])
    table(lines,MKT+' Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.',('Method','Updates','TRAIN MAE','VALIDATION MAE','GPU training seconds'),[(label,0 if label==INIT else 2048,display(queries[label]['train']['standardized_fixed_query_MAE']),display(queries[label]['validation']['standardized_fixed_query_MAE']),display(cost[label]['seconds'])+(' (reused)' if label in controls else '')) for label in labels])
    table(lines,MKT+' Same query distributions; NLL omits constant and can be negative.',('Method','TRAIN Gaussian NLL','VALIDATION Gaussian NLL'),[(label,display(queries[label]['train']['Gaussian_NLL_F64']),display(queries[label]['validation']['Gaussian_NLL_F64'])) for label in labels])
    table(lines,MKT+' '+T5+' All22 frozen numeric conditions; trust and coverage are separate.',('Dataset','Condition','Candidate','Frozen floor/ceiling','Source','Rule','Pass'),[(g['dataset_id'],g['id'],display(g['candidate']),display(g['floor']),g['floor_source'],g['rule'],'PASS' if g['passed'] else 'FAIL') for g in gates])
    lines.extend(['The exact declared trajectory is stopped at2048. Retain every failed condition and all nine references and their scoped strengths. Further mechanisms require separate prospective declarations; no coefficient/seed/rate/budget/head rescue. V3\'s reported144s training cost is reused historical evidence,not a new matched benchmark.',''])
    require(not args.output.exists(),'Preserve all comparison attempts; exclusive output required');args.output.mkdir(parents=True)
    dump(args.output/'comparison.json',report);write_new(args.output/'report.md','\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
