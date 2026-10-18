#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent saved-scalar audit. Standard library only; never loads a model."""
# Owned derivative of doc/causal_fresh_v3/audit-quality.py. The v3 original
# and inert reader remain preserved; no v4 reporter/comparator code is imported.
import argparse
import datetime
import hashlib
import json
import math
import struct
from pathlib import Path
from archive_reader import Archive, Tensor, DTYPES

CARD_SHA = '114026b2e6d11d84aee1855a8af673d84e782ab057eb45d398a5d66fa94b1dce'
INPUT_SHA = 'a6bff90ffc9dd917c6a9c446659c52e0c76b5fb34a9f67201788eb3168948eb9'
TAG, INIT = 'CPB-v4.market-01', 'CPB-v4.market-01.init'
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
REPS, MASTERS = (2701, 2802, 2903), (941101, 941102, 941103)
STAGES, MODES = ('initial', 'final'), ('synthetic_train', 'market_scaler')
VIEWS, HEADS = ('validation_intact', 'validation_deleted'), ('ridge', 'tiny_secondary')
BASE = 'CPB-v3.market-01'
POLICY = 'variance_input_gradient_stop-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def mean(values):
    return math.fsum(values) / len(values)


def seed(rep):
    mask = (1 << 64) - 1
    x = (rep + 0x9e3779b97f4a7c15 * 36) & mask
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
    return x ^ (x >> 31)


class Audit:
    def __init__(self, maps):
        self.maps = sorted(maps, key=lambda pair: -len(pair[0]))
        self.checks, self.errors, self.pins = 0, [], {}

    def check(self, yes, why):
        self.checks += 1
        if not yes:
            self.errors.append(why)

    def close(self, a, b, why):
        self.check(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12, why)

    def path(self, name, origin=None):
        value = str(name)
        for old, new in self.maps:
            if value == old or value.startswith(old.rstrip('/') + '/'):
                value = new.rstrip('/') + value[len(old.rstrip('/')):]
                break
        path = Path(value)
        return path if path.is_absolute() else Path(origin or '.') / path

    def pin(self, path):
        path = Path(path).resolve(strict=True)
        out = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
        self.pins[str(path)] = out
        return out

    def bound(self, binding, origin=None):
        path = self.path(binding['path'], origin).resolve(strict=True)
        actual = self.pin(path)
        self.check(actual['bytes'] == binding['bytes'] and actual['sha256'] == binding['sha256'],
                   'Artifact bytes/hash: ' + str(path))
        return path


def proof(a, binding, checkpoint, updates, origin):
    path = a.bound(binding, origin)
    value = read(path)
    fields = {'kind': 'cpb_v4_saved_checkpoint_causal_proof_v1', 'status': 'passed',
              'gradient_policy': POLICY,
              'model': 'CPB-v4', 'source_updates': 0, 'causal_updates': updates,
              'initialization_seed': 943001, 'pretrained_ancestor': 'none',
              'new_training_updates': 0, 'all_cutoffs': 128, 'suffix_cases': 2350,
              'internal_temporal_cases': 30, 'state_unchanged': True,
              'future_values_and_masks_invariant': True, 'CPU_encoder_calls': 0, 'head_fits': 0}
    for key, expected in fields.items():
        a.check(value[key] == expected, 'Saved causal proof ' + key)
    a.check(a.path(value['checkpoint']).resolve(strict=True) == checkpoint and
            value['checkpoint_sha256'] == sha(checkpoint) and value['CUDA_encoder_calls'] > 0,
            'Proof covers the actual scored fresh checkpoint')
    detail = path.parent / (('initial0' if updates == 0 else 'final2048') + '-causality.json')
    a.pin(detail)
    witness = read(detail)
    a.check(witness['status'] == 'passed' and witness['all_cutoffs'] == 128 and
            witness['suffix_cases'] == 2350 and witness['internal_temporal_cases'] == 30 and
            witness['state_unchanged'], 'Saved detailed causal witness')
    return value


def tensor(archive, name, shape):
    value = archive.field(name)
    if not isinstance(value, Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError('Inert tensor shape/type differs: '+name)
    return value.values()


def regression(predicted, target, valid):
    selected = [(float(p),float(y)) for p,y,ok in zip(predicted,target,valid) if ok]
    if not selected:
        raise ValueError('Empty saved prediction support')
    center = mean([y for _,y in selected])
    errors = [p-y for p,y in selected]
    squared = math.fsum(e*e for e in errors)
    total = math.fsum((y-center)**2 for _,y in selected)
    return {'r2':1.-squared/total if total else None,
            'mae':mean([abs(e) for e in errors]),'rmse':math.sqrt(squared/len(errors)),
            'valid':len(selected),'total':len(valid),'coverage':len(selected)/len(valid)}


def prediction_audit(a,args,inputs):
    """Read saved arrays/coefficients as DATA. No neural or encoder forward."""
    output = []
    reference = {}
    for role,n in (('train',112),('validation',32)):
        entry = next(p for p in inputs['files'] if p['path'].endswith('/features/'+role+'-targets.pt'))
        path = a.bound({'path':str(args.data_root/entry['path']),'bytes':entry['bytes'],'sha256':entry['sha256']})
        archive = Archive(path)
        try:
            reference[role] = {'targets':tensor(archive,'targets',(n,4)),
                               'valid':tensor(archive,'valid',(n,4)),
                               'row_ids':tensor(archive,'row_ids',(n,)),
                               'group_ids':tensor(archive,'group_ids',(n,))}
        finally:
            archive.close()
    for stage in STAGES:
        native = {}
        for role,n in (('train',112),('validation',32)):
            path = args.run/'features'/stage/'market'/('cpb_v4-'+role+'.pt')
            a.pin(path); archive = Archive(path)
            try:
                native[role] = {'values':tensor(archive,'values',(n,36)),
                                'valid':tensor(archive,'valid',(n,)),
                                'row_ids':tensor(archive,'row_ids',(n,))}
            finally:
                archive.close()
            a.check(native[role]['row_ids'] == reference[role]['row_ids'], 'Saved feature/target exact row association '+stage+role)
            a.check(all(math.isfinite(v) for v in native[role]['values']), 'Finite saved native arrays '+stage+role)
        train = reference['train']
        feature_ids = [r for r in range(112) if native['train']['valid'][r]]
        feature_mean,feature_scale,target_mean,target_scale = [],[],[],[]
        for d in range(36):
            column = [native['train']['values'][r*36+d] for r in feature_ids]
            center = mean(column)
            feature_mean.append(center)
            feature_scale.append(max(1e-8,math.sqrt(mean([(v-center)**2 for v in column]))))
        for t in range(4):
            column = [train['targets'][r*4+t] for r in range(112) if train['valid'][r*4+t]]
            center = mean(column);target_mean.append(center)
            target_scale.append(max(1e-8,math.sqrt(mean([(v-center)**2 for v in column]))))
        reported = read(args.run/stage/'market-readouts/report.json')['methods'][0]['repetitions']
        for rep_index,rep in enumerate(REPS):
            folder = args.run/stage/'market-readouts/models/cpb_v4'/('rep-'+str(rep))
            fit_path = folder/'fit.pt';a.pin(fit_path);fit = Archive(fit_path)
            try:
                fitted = {name:tensor(fit,name,shape) for name,shape in (
                    ('feature_mean',(36,)),('feature_scale',(36,)),('target_mean',(4,)),('target_scale',(4,)),
                    ('ridge_weights',(36,4)),('ridge_intercept',(4,)),('supported_targets',(4,)),
                    ('target_training_counts',(4,)),('target_training_supported',(4,)))}
                actual_seed = tensor(fit,'actual_seed',())[0]
                a.check(actual_seed % (1<<64) == seed(rep), 'Saved fitted D36 seed signed-to-unsigned identity')
                a.check(tensor(fit,'training_feature_rows',()) == [len(feature_ids)] and
                        tensor(fit,'feature_training_supported',()) == [True], 'Saved TRAIN normalization support')
                for name,expected in (('feature_mean',feature_mean),('feature_scale',feature_scale),
                                      ('target_mean',target_mean),('target_scale',target_scale)):
                    for got,want in zip(fitted[name],expected):
                        a.close(got,want,'Independent TRAIN population statistic '+stage+name)
                a.check(all(fitted['supported_targets']) and all(fitted['target_training_supported']) and
                        fitted['target_training_counts'] == [112]*4, 'Saved target support/counts')
                for name,shape in (('neural_w1',(36,16)),('neural_b1',(16,)),('neural_w2',(16,4)),('neural_b2',(4,))):
                    a.check(all(math.isfinite(v) for v in tensor(fit,name,shape)), 'Finite saved neural coefficients; no forward '+name)
            finally:
                fit.close()
            for role,n in (('train',112),('validation',32)):
                path = folder/('predictions_'+role+'.pt');a.pin(path);saved = Archive(path)
                try:
                    prediction = {name:tensor(saved,name,(n,4)) for name in
                                  ('targets','target_valid','valid','ridge_predictions','neural_predictions')}
                    ids = tensor(saved,'row_ids',(n,));groups = tensor(saved,'group_ids',(n,))
                finally:
                    saved.close()
                a.check(ids == reference[role]['row_ids'] and groups == reference[role]['group_ids'] and
                        prediction['targets'] == reference[role]['targets'] and
                        prediction['target_valid'] == reference[role]['valid'], 'Saved predictions bind original targets/IDs '+stage+role)
                a.check(all(math.isfinite(v) for key in ('ridge_predictions','neural_predictions') for v in prediction[key]),
                        'All saved linear/neural predictions finite '+stage+role)
                for r in range(n):
                    for t in range(4):
                        ok = bool(native[role]['valid'][r] and reference[role]['valid'][r*4+t] and fitted['supported_targets'][t])
                        a.check(prediction['valid'][r*4+t] == ok, 'Exact saved prediction eligibility')
                        if not ok:continue
                        standardized = math.fsum((native[role]['values'][r*36+d]-fitted['feature_mean'][d])/
                            fitted['feature_scale'][d]*fitted['ridge_weights'][d*4+t] for d in range(36))+fitted['ridge_intercept'][t]
                        rebuilt = standardized*fitted['target_scale'][t]+fitted['target_mean'][t]
                        a.close(rebuilt,prediction['ridge_predictions'][r*4+t], 'Saved ridge coefficient arithmetic; no refit')
                for t,task in enumerate(TASKS):
                    scores = {head:regression(prediction[field][t::4],prediction['targets'][t::4],prediction['valid'][t::4])
                              for head,field in (('ridge','ridge_predictions'),('neural','neural_predictions'))}
                    if role == 'validation':
                        for head in ('ridge','neural'):
                            for metric in ('r2','mae','rmse','coverage'):
                                a.close(scores[head][metric],reported[rep_index]['targets'][t][head][metric],
                                        'Independent saved VALIDATION prediction metric '+stage+task+head+metric)
                    output.append({'stage':stage,'split':role,'repetition':rep,'task':task,'scores':scores})
    return output


def query_witness_audit(a,args,training):
    result = []
    for stage in STAGES:
        for role in ('train','validation'):
            path = args.training/'witnesses'/(stage+'-'+role+'-query.pt')
            a.pin(path);archive = Archive(path);shape = (8,3,128,9)
            try:
                values = {key:tensor(archive,key,shape) for key in ('mean','variance','targets','target_mask','visible')}
                ids = tensor(archive,'row_ids',(8,))
                stored_mae = tensor(archive,'row_mae',(8,));stored_nll = tensor(archive,'row_nll',(8,))
            finally:
                archive.close()
            a.check(all(not target or not visible for target,visible in zip(values['target_mask'],values['visible'])),
                    'Fixed query target/visible supports are disjoint')
            if role=='train':a.check(all(1704067259999<=row<=1705881599999 for row in ids),'Query witness TRAIN-only cutoff')
            expected = training['fixed_query_'+stage][role]
            for b in range(8):
                maes,nlls,counts = [],[],[]
                for c in range(3):
                    selected = [((b*3+c)*128+h)*9+f for h in range(128) for f in range(9)
                                if values['target_mask'][((b*3+c)*128+h)*9+f]]
                    counts.append(len(selected))
                    if not selected:continue
                    errors = [values['mean'][i]-values['targets'][i] for i in selected]
                    variances = [values['variance'][i] for i in selected]
                    a.check(all(math.isfinite(v) and v>0 for v in variances) and all(math.isfinite(e) for e in errors),
                            'Finite supported query residual/positive variance')
                    maes.append(mean([abs(e) for e in errors]))
                    nlls.append(mean([.5*(e*e/v+math.log(v)) for e,v in zip(errors,variances)]))
                for got,want in ((mean(maes),stored_mae[b]),(mean(nlls),stored_nll[b]),
                                 (mean(maes),expected['row_mae'][b]),(mean(nlls),expected['row_nll'][b])):
                    a.close(got,want,'Independent B8 fixed-query hierarchical MAE/NLL arithmetic')
                a.check(sum(counts)==expected['row_target_cells'][b],'Independent B8 original target-cell support')
            result.append({'stage':stage,'split':role,'witness_rows':8,'row_ids':ids,
                           'scope':'Only first8 saved masked-query examples; full112/32 row aggregation audited separately'})
    return result


def paired_initial_audit(a,args,declaration):
    pairs = []
    for entry in declaration['paired_initial_reference_inputs']:
        reference = a.bound({'path':entry['v3_container_path'],'bytes':entry['bytes'],'sha256':entry['sha256']})
        current = (args.run if entry['v4_root']=='quality' else args.training)/entry['v4_relative_path']
        reference_pin,current_pin = a.pin(reference),a.pin(current)
        old,new = Archive(reference),Archive(current)
        numeric,metadata = [],[]
        try:
            for field in entry['numeric_fields']:
                left,right = old.field(*field.split('.')),new.field(*field.split('.'))
                if not isinstance(left,Tensor) or not isinstance(right,Tensor):
                    raise ValueError('Pairing requires inert tensors: '+field)
                same_layout = left.shape==right.shape and left.storage.dtype==right.storage.dtype
                a.check(same_layout,'Paired initial logical tensor shape/dtype '+entry['id']+field)
                x,y = left.values(),right.values()
                same_bits = False
                if same_layout and len(x)==len(y):
                    fmt = DTYPES[left.storage.dtype][0]
                    same_bits = struct.pack('<'+str(len(x))+fmt,*x)==struct.pack('<'+str(len(y))+fmt,*y)
                a.check(same_bits,'Paired initial exact logical tensor bytes '+entry['id']+field)
                numeric.append({'field':field,'shape':list(left.shape),'dtype':left.storage.dtype,
                                'logical_values':len(x),'tensor_bytes_equal':same_bits})
            for field in entry.get('metadata_text_fields',[]):
                left,right = old.field(*field.split('.')),new.field(*field.split('.'))
                if left.storage.dtype!='ByteStorage' or right.storage.dtype!='ByteStorage':
                    raise ValueError('Expected text metadata in pairing: '+field)
                x,y = bytes(left.values()).decode('utf-8'),bytes(right.values()).decode('utf-8')
                metadata.append({'field':field,'v3':x,'v4':y,'equal':x==y,
                                 'interpretation':'Metadata identity is separate from fitted numeric statistics/embedding equality'})
        finally:
            old.close();new.close()
        pairs.append({'id':entry['id'],'v3_reference':reference_pin,'v4_initial_or_fresh_statistics':current_pin,
                      'whole_file_bytes_equal':reference_pin['bytes']==current_pin['bytes'] and reference_pin['sha256']==current_pin['sha256'],
                      'all_declared_numeric_tensor_values_exact':all(v['tensor_bytes_equal'] for v in numeric),
                      'numeric_tensors':numeric,'metadata_text':metadata})
    a.check(len(pairs)==13,'Exactly8 initial feature archives and5 fresh scaler/normalizer archives paired')
    return {'scope':'Pairing witness only, never quality selection; no V3/V4 encoder/head execution or fitting',
            'old_model_numeric_reads':0,'reference_data_archives':len(pairs),
            'all_declared_numeric_tensor_values_exact':all(p['all_declared_numeric_tensor_values_exact'] for p in pairs),
            'file_and_metadata_equality_reported_separately':True,'pairs':pairs}


def comparison_audit(a,args,card,tables):
    baseline_path = a.bound(card['comparison_control'],args.card.resolve().parent.parent)
    baseline = read(baseline_path)
    comparison = read(args.comparison/'comparison.json')
    a.check(baseline['model'] == BASE and baseline['source_updates'] == 0 and
            baseline['trust_protocol_admission_passed'] and baseline['full_coverage'], 'Preserved clean baseline control')
    a.check(comparison['baseline'] == BASE and comparison['candidate'] == TAG and comparison['baseline_unchanged'] and
            comparison['baseline_scores_reused'] and not comparison['historical_pretrained_quality_floors_used'] and
            not comparison['automatic_promotion'] and not comparison['application_changed'] and not comparison['overall_winner_declared'],
            'Comparison scope and no automatic promotion')
    a.check(a.bound(comparison['card']) == args.card.resolve() and
            a.bound(comparison['baseline_control']) == baseline_path and
            a.bound(comparison['candidate_table']) == (args.report/'table_rows.json').resolve(), 'Comparison exact bound inputs')
    market = lambda result,label:{r['task']:r for r in result['market_rows'] if r['method']==label}
    old,new = market(baseline,BASE),market(tables,TAG)
    panels = lambda result,label:{(p['input_policy'],p['view']):next(r for r in p['rows'] if r['method']==label)
                                  for p in result['synthetic']['panels']}
    old_timing,new_timing = panels(baseline,BASE),panels(tables,TAG)
    criteria = card['advancement'];epsilon = criteria['numeric_tolerance']
    a.check(epsilon == 1e-12, 'Exact prospective numeric tolerance')
    a.check(len(criteria['market_retention'])==8 and len(criteria['tempo5_retention'])==8 and
            len(criteria['fixed_query_MAE'])==2 and len(criteria['relation_usefulness'])==2,
            'Exact prospective8+8+2+1+1+2 condition layout')
    expected = {}
    def gate(identity,value,reference,passed):
        expected[identity] = {'candidate':value,'baseline_or_floor':reference,'passed':bool(passed)}
    for spec in criteria['market_retention']:
        a.check(spec['rule']=='candidate+1e-12>=baseline','Prospective market operator')
        t,m = spec['task'],spec['metric'];value,reference = new[t][m],old[t][m]
        a.close(reference,spec['baseline'],'Frozen clean market floor')
        gate(t+':'+m,value,reference,value+epsilon>=reference)
    for spec in criteria['tempo5_retention']:
        a.check(spec['rule']=='candidate+1e-12>=baseline','Prospective timing operator')
        key,m = (spec['input_policy'],spec['view']),spec['metric'];value,reference = new_timing[key][m],old_timing[key][m]
        a.close(reference,spec['baseline'],'Frozen clean timing floor')
        gate(':'.join((*key,m)),value,reference,value+epsilon>=reference)
    for spec in criteria['fixed_query_MAE']:
        a.check(spec['rule']=='candidate<=baseline+1e-12','Prospective query operator')
        split = spec['split'];value = tables['fixed_queries']['final'][split]['standardized_fixed_query_MAE']
        reference = baseline['fixed_queries']['final'][split]['standardized_fixed_query_MAE']
        a.close(reference,spec['baseline'],'Frozen clean query floor')
        gate(split+':standardized_MAE',value,reference,value<=reference+epsilon)
    value = tables['fixed_queries']['final']['validation']['Gaussian_NLL_F64']
    reference = baseline['fixed_queries']['final']['validation']['Gaussian_NLL_F64']
    a.close(reference,criteria['variance_quality']['baseline'],'Frozen strict clean NLL floor')
    a.check(criteria['variance_quality']['rule']=='candidate<baseline-1e-12','Prospective strict NLL operator')
    gate('known_validation:Gaussian_NLL_F64',value,reference,value<reference-epsilon)
    trend = new['past_trend']['neural_r2_each']
    a.check(criteria['neural_trend_robustness']['rule']=='All3 final neural trend R2 strictly>0' and
            criteria['neural_trend_robustness']['repetitions']==list(REPS),'Prospective strict three-fit trend rule')
    gate('past_trend:all3_neural_R2_positive',trend,0.,len(trend)==3 and all(v>0 for v in trend))
    for spec in criteria['relation_usefulness']:
        a.check(spec['rule']=='candidate>0','Prospective strict relation operator')
        m = spec['metric'];value = new['past_relation'][m]
        gate('past_relation:'+m+':positive',value,0.,value>0)
    a.check(len(expected)==22 and len(comparison['numeric_conditions'])==22, 'Exactly22 independently reduced conditions')
    a.check({g['id'] for g in comparison['numeric_conditions']}==set(expected),'Every prospective condition appears exactly once')
    for actual in comparison['numeric_conditions']:
        desired = expected[actual['id']]
        for field in ('candidate','baseline_or_floor','passed'):
            if isinstance(desired[field],list):a.check(actual[field]==desired[field],'All three trend repetitions retained')
            elif isinstance(desired[field],bool):a.check(actual[field] is desired[field],'Independent gate outcome '+actual['id'])
            else:a.close(actual[field],desired[field],'Independent gate numeric value '+actual['id'])
    passed = sum(g['passed'] for g in expected.values())
    a.check(comparison['numeric_passed']==passed and comparison['numeric_total']==22 and
            comparison['trust_admission_passed'] and comparison['full_coverage'] and
            comparison['advancement_conditions_passed'] == (passed==22), 'Independent final gate/coverage disposition')
    for item in comparison['market_rows']:
        source = old if item['method']==BASE else market(tables,item['method'])
        expected_row = dict(source[item['task']],reused=item['method']==BASE)
        a.check(item==expected_row,'Compared market row exact source copy')
    for panel in comparison['synthetic_panels']:
        a.check(panel['recipe_id']=='tempo-5-slow-fast-nine-feature-v1' and
                panel['head_fitting_dataset_id']==panel['scoring_dataset_id']=='TEMPO-5','Timing recipe/fit-score identity')
        key = (panel['input_policy'],panel['view'])
        for item in panel['rows']:
            original = old_timing[key] if item['method']==BASE else panels(tables,item['method'])[key]
            a.check(item==dict(original,reused=item['method']==BASE),'Compared timing row exact source copy')
    a.close(comparison['training_cost'][BASE]['seconds'],baseline['training']['GPU_step_seconds_synchronized'],'V3 historical training cost reused')
    a.check(comparison['training_cost'][BASE]['reused'] is True,'V3 training timer is not rerun')
    def bound_tree(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}.issubset(value):a.bound(value)
            else:
                for child in value.values():bound_tree(child)
        elif isinstance(value,list):
            for child in value:bound_tree(child)
    bound_tree(comparison['candidate_trust_bindings'])
    a.bound(comparison['source']);a.bound(comparison['detailed_reporter_source'])
    for path in (baseline_path,args.comparison/'comparison.json',args.comparison/'report.md'):a.pin(path)
    return {'numeric_passed':passed,'numeric_total':22,'independent_conditions':expected,'baseline':BASE,
            'baseline_unchanged':True,'trust_separate':True,'coverage_separate':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('card', 'input-manifest', 'raw-controls', 'training', 'run', 'report', 'quality-receipt',
                'comparison', 'audit-card', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/cuwacunu'))
    parser.add_argument('--map', action='append', default=[], metavar='ORIGINAL_PREFIX=LOCAL_PREFIX')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve existing audit: ' + str(args.output))
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('Execute this scalar auditor inside the existing managed container')
    a = Audit([item.split('=', 1) for item in args.map])
    declaration = read(args.audit_card)
    a.check(declaration['kind']=='causal_fresh_v4_saved_quality_audit_card_v1' and
            declaration['execution_pending'] and declaration['model_calls']==declaration['head_fits']==0,
            'Prospective saved-evidence audit declaration')
    for entry in declaration['auditor_sources']:
        path = Path(__file__).parent/entry['file']
        a.bound({'path':str(path),'bytes':entry['bytes'],'sha256':entry['sha256']})
    card, inputs, controls = read(args.card), read(args.input_manifest), read(args.raw_controls)
    a.check(sha(args.card) == CARD_SHA and sha(args.input_manifest) == INPUT_SHA, 'Frozen card/manifest')
    a.check(card['mechanism']['source_updates'] == 0 and not card['mechanism']['pretrained_model_inputs'] and
            not card['mechanism']['teacher_model_inputs'] and not card['mechanism']['checkpoint_import_for_initialization'],
            'Fresh training has no imported model/teacher')
    a.check(not card['historical_reference_is_quality_gate'] and not card['quality']['historical_quality_floors'] and
            not card['quality']['initial_quality_reused'], 'No inherited-quality oracle or initial score reuse')
    a.check(len(inputs['files']) == 20 and not inputs['old_fitted_scaler_loaded'] and
            not inputs['old_model_or_native_features_loaded'], 'Data-only input scope')
    for entry in inputs['files']:
        a.bound({'path': str(args.data_root / entry['path']), 'bytes': entry['bytes'], 'sha256': entry['sha256']})
    source = controls['source']
    a.check(source == card['quality']['raw_controls_reused'] and source in inputs['files'],
            'Raw source is exactly the declared data-only input pin')
    raw_source = a.bound({'path': str(args.data_root / source['path']), 'bytes': source['bytes'], 'sha256': source['sha256']})
    original_raw = [row for row in read(raw_source)['rows'] if row['source_method'] in ('raw_recent128', 'raw_full256')]
    a.check(controls['derived_without_new_fits'] and not controls['encoder_rows_included'] and
            not controls['quality_gates_included'] and len(original_raw) == len(controls['rows']) == 8,
            'Exactly eight source-pinned raw rows, with no encoder rows or gates')
    a.check(original_raw == controls['rows'], 'Raw rows exactly match the permitted source subset')
    quality = read(args.quality_receipt)
    a.check(quality['status'] == 'passed' and quality['model'] == TAG and quality['card_sha256'] == CARD_SHA and
            quality['old_encoder_calls'] == 0 and not quality['initial_quality_reused'] and
            quality['initial_and_final_heads_new'], 'Initial/final quality execution identity')
    checkpoints = {stage: a.bound(quality[stage + '_checkpoint'], args.quality_receipt.parent) for stage in STAGES}
    a.check(checkpoints['initial'] == (args.training/'checkpoints/inference-0.pt').resolve(strict=True) and
            checkpoints['final'] == (args.training/'checkpoints/inference-2048.pt').resolve(strict=True), 'Only initial0/final2048 scored')
    proof_values = {stage: proof(a, quality[key], checkpoints[stage], updates, args.quality_receipt.parent)
                    for stage, key, updates in (('initial', 'initial_saved_proof', 0), ('final', 'post_training_proof', 2048))}
    engineering = read(a.bound(quality['engineering_proof'], args.quality_receipt.parent))
    a.check(engineering['gradient_policy']==POLICY, 'Engineering selected gradient policy')
    a.check(engineering['status'] == 'passed' and engineering['source_updates'] == 0 and
            engineering['pretrained_ancestor'] == 'none' and engineering['old_checkpoints_opened'] == 0 and
            engineering['typed_resume_next_loss_model_moments_bitwise_exact'] and engineering['quality_updates'] == 0,
            'Actual fresh CUDA engineering and typed resume admission')
    a.check(engineering['initialization_seed'] == 943001 and engineering['all_initial_tensors_equal_seeded_constructor'] and
            engineering['first_typed_artifact_exact'] and engineering['scaler_and_bank_fitted_fixture_TRAIN_only'] and
            engineering['all_cutoffs_per_stage'] == 128 and engineering['suffix_cases_per_stage'] == 2350 and
            engineering['internal_temporal_cases_per_stage'] == 30 and engineering['typed_negative_cases'] >= 30 and
            all(v['nonzero_gradient_parameter_events'] > 0 and v['weight_movement_parameter_events'] > 0
                for v in engineering['actual_cuda_learning'].values()), 'Full fresh CUDA/trust matrix')
    build = read(a.bound(quality['build_admission'], args.quality_receipt.parent))
    a.check(build['status'] == 'passed' and build['card_sha256'] == CARD_SHA and len(build['binaries']) == 6,
            'Frozen build admission')
    for binary in build['binaries']:
        a.bound(binary)
    gradient_path = a.bound(quality['variance_gradient_proof'],args.quality_receipt.parent)
    gradient = read(gradient_path)
    for key,value in {'status':'passed','model':'CPB-v4','gradient_policy':POLICY,'actual_CUDA':True,
                      'native_variance_gradient_blocked':True,'mean_path_live':True,'variance_head_learning':True,
                      'forward_variance_input_dependence':True,'manual_partial_gradient_parity':True,
                      'nonvacuous_nonzero_fixture':True}.items():
        a.check(gradient[key]==value,'Nonvacuous actual CUDA mechanism proof '+key)
    binary_path = a.bound(gradient['binary'],gradient_path.parent)
    log_path = a.bound(gradient['log'],gradient_path.parent)
    a.check(binary_path in {a.path(p['path']).resolve(strict=True) for p in build['binaries']} and
            'PASS CPB-v4 variance_input_gradient_stop-v1:' in log_path.read_text(encoding='utf-8'),
            'Actual admitted mechanism binary and observed passing output')
    initialization = read(a.bound(quality['fresh_initialization'], args.quality_receipt.parent))
    for key, expected in {'status':'passed', 'initialization_seed':943001, 'pretrained_ancestor':'none',
            'source_updates':0, 'causal_updates':0, 'inherited_learned_parameters':0,
            'fresh_learned_parameter_values':293672, 'analytic_fixed_prior_values':3888,
            'old_model_artifacts_opened':0, 'old_normalization_artifacts_opened':0,
            'all_initial_parameters_equal_direct_seeded_constructor':True}.items():
        a.check(initialization[key] == expected, 'Main fresh initialization ' + key)
    a.check(initialization['direct_constructor_state_identity'] == initialization['initial_state_identity'],
            'Every initial tensor equals a direct fresh seeded constructor')
    a.check(a.bound(initialization['initial_checkpoint']) == checkpoints['initial'], 'Actual initialization checkpoint')
    calibration = read(args.training/'calibration/summary.json')
    for key, expected in {'status':'passed','TRAIN_anchors':4096,'history':128,'input_scaler_fits':1,
            'bank_normalizer_fits':1,'validation_rows_used':0,'test_rows_used':0,'encoder_calls':0,
            'old_normalization_artifacts_opened':0}.items():
        a.check(calibration[key] == expected, 'Fresh calibration ' + key)
    for item in calibration['artifacts']:
        a.bound({'path':item['file'],'bytes':item['bytes'],'sha256':item['sha256']}, args.training/'calibration')
    a.check(calibration['input_scaler_identity'] == initialization['fresh_input_scaler_identity'] and
            calibration['bank_identity'] == initialization['fresh_bank_identity'], 'Fresh fitted statistics feed actual initial state')
    training = read(args.training/'training-summary.json')
    a.check(training['gradient_policy']==POLICY and
            training['encoder_gradient_kind']=='partial-NLL-gradient;live-mean-residual;variance-input-stopped' and
            training['conditional_variance_head_trainable'] is True, 'Actual training selected partial gradient policy')
    a.check(training['status'] == 'passed' and training['source_updates'] == 0 and training['new_causal_updates'] == 2048 and
            training['batch_size'] == 8 and training['typed_resume_512_exact'] and training['parameters_with_recorded_updates'] == 293672 and
            training['fixed_prior_parameters'] == 3888 and training['new_scaler_fits'] == training['new_normalizer_fits'] == 1,
            'Complete fresh CUDA trajectory and protected statistics/prior')
    for key in ('old_model_artifacts_opened','old_normalization_artifacts_opened','CPU_encoder_calls','old_encoder_calls','head_fits','training_replays'):
        a.check(training[key] == 0, 'Training zero forbidden operation: ' + key)
    for stage in STAGES:
        q = read(args.training/('query-'+stage+'.json'))
        a.check(q == training['fixed_query_'+stage], 'Saved query association ' + stage)
        for role, n in (('train',112),('validation',32)):
            a.check(q[role]['valid_rows'] == len(q[role]['row_mae']) == len(q[role]['row_nll']) == n,
                    'Fixed query row support ' + stage + role)
            a.close(mean(q[role]['row_mae']), q[role]['standardized_fixed_query_MAE'], 'Fixed query MAE mean')
            a.close(mean(q[role]['row_nll']), q[role]['Gaussian_NLL_F64'], 'Fixed query separate NLL mean')
            a.check(sum(q[role]['row_target_cells']) == q[role]['target_cells'] == n*216, 'One-patch fixed query support')
    tables, completion = read(args.report/'table_rows.json'), read(args.report/'completion.json')
    a.check(tables['kind'] == 'causal_fresh_v4_results_v1' and tables['model'] == TAG and tables['source_updates'] == 0 and
            tables['new_market_rows'] == 8 and not tables['initial_quality_reused'] and not tables['historical_encoder_quality_targets_used'] and
            not tables['overall_winner_declared'] and not tables['application_changed'], 'Fresh descriptive report identity')
    a.check(tables['market_rows'][:8] == controls['rows'] and len(tables['market_rows']) == 16,
            'Eight raw and eight newly measured market rows only')
    a.check(tables['clean_baseline']=='CPB-v3.market-01' and tables['clean_baseline_unchanged'] and
            tables['gradient_policy']==POLICY, 'V3 clean baseline remains unchanged')
    saved_prediction_arithmetic = prediction_audit(a,args,inputs)
    query_witness_arithmetic = query_witness_audit(a,args,training)
    paired_initial = paired_initial_audit(a,args,declaration)
    clean_comparison = comparison_audit(a,args,card,tables)
    market_audit, synthetic_audit, synthetic_values = [], [], {}
    full_coverage = True
    for stage, label in zip(STAGES, (INIT,TAG)):
        report = read(args.run/stage/'market-readouts/report.json')
        a.check(report['dataset_id'] == 'MKT-HIST-2' and report['card_sha256'] == CARD_SHA and
                report['input_manifest_sha256'] == INPUT_SHA and (report['training_rows'],report['validation_rows']) == (112,32),
                'Market report population/card ' + stage)
        for key, value in {'ridge_penalty':1,'neural_hidden':16,'neural_learning_rate':.01,'neural_updates':100,'outputs':4}.items():
            a.check(report['heads'][key] == value, 'Market unchanged fixed head ' + key)
        a.check(len(report['methods']) == 1, 'One fresh market method per stage')
        method = report['methods'][0]
        a.check(method['name'] == 'cpb_v4' and method['size'] == 36 and method['ridge_parameters'] == 4*(36+1) and
                method['neural_parameters'] == 36*16+16+16*4+4 and method['ridge_fits'] == 1 and
                method['ridge_task_fits'] == 4 and method['neural_fits'] == 3 and method['training_feature_rows'] == 112,
                'Market D36 parameters and fit budgets')
        for key, role in (('training_path','train'),('validation_path','validation')):
            p = a.path(method[key]).resolve(strict=True)
            a.check(p == (args.run/'features'/stage/'market'/('cpb_v4-'+role+'.pt')).resolve(strict=True), 'Market feature stage association')
            a.pin(p)
        a.check(tuple(r['repetition'] for r in method['repetitions']) == REPS and
                tuple(r['actual_seed'] for r in method['repetitions']) == tuple(seed(x) for x in REPS), 'Exact market D36 stream seeds')
        rows = [r for r in tables['market_rows'] if r.get('method') == label]
        a.check(len(rows) == 4 and {r['task'] for r in rows} == set(TASKS), 'Four independent tasks per new checkpoint')
        for i, task in enumerate(TASKS):
            row = next(r for r in rows if r['task'] == task)
            reps = [r['targets'][i] for r in method['repetitions']]
            a.check(row['size'] == 36 and row['checkpoint_stage'] == stage and not row['reused'], 'New native market row provenance')
            for j, target in enumerate(reps):
                a.check(target['name'] == task and target['ridge'] == reps[0]['ridge'] and target['train_target_eligible'] == 112 and
                        target['train_target_supported'] and target['method_training_supported'], 'Market shared ridge/task support')
                a.check(method['repetitions'][j]['neural_updates_completed'] == 100 and
                        method['repetitions'][j]['ridge_fit_reused'] == (j != 0), 'Only TRAIN fixed market fitting')
                for head in ('ridge','neural'):
                    score = target[head]
                    a.check(score['valid'] == score['total'] == score['target_eligible'] == 32, 'Market complete support')
                    full_coverage &= score['valid'] == score['total'] == 32
                    for metric in ('r2','mae','rmse'):
                        a.check(math.isfinite(score[metric]), 'Finite market physical metric')
                    a.check(score['mae'] >= 0 and score['rmse'] >= score['mae']-1e-12, 'Physical MAE/RMSE bounds')
                    a.close(score['coverage'], score['valid']/score['total'], 'Market coverage arithmetic')
            for metric in ('r2','mae','rmse'):
                a.close(row['linear_'+metric], reps[0]['ridge'][metric], 'Reported market ridge metric')
                expected = [t['neural'][metric] for t in reps]
                a.check(row['neural_'+metric+'_each'] == expected, 'All three neural repetitions preserved')
                a.close(row['neural_mean_'+metric], mean(expected), 'Independent neural three-seed mean')
            a.check(row['valid'] == row['total'] == 32 and row['coverage'] == 1, 'Reported complete market coverage')
            market_audit.append({'stage':stage,'task':task,'linear_r2':row['linear_r2'],'neural_mean_r2':mean([t['neural']['r2'] for t in reps]),
                                 'linear_MAE':row['linear_mae'],'neural_mean_MAE':mean([t['neural']['mae'] for t in reps])})
        for master_index, master in enumerate(MASTERS):
            report = read(args.run/stage/'synthetic-readouts'/('master-'+str(master))/'report.json')
            a.check(report == tables['synthetic']['all_saved_repetitions_and_intervals'][stage][master_index],
                    'All original cohort repetitions and intervals retained exactly')
            a.check(report['protocol'] == 'fixed-feature-readouts-v1' and int(report['master_seed']) == master, 'TEMPO cohort identity')
            for key, expected in {'ridge_penalty':1,'tiny_hidden':16,'tiny_steps':100,'tiny_learning_rate':.01,
                    'validation_fits':0,'encoder_calls':0,'pca_fits':0}.items():
                a.check(report['recipe'][key] == expected, 'TEMPO unchanged fixed recipe ' + key)
            a.check(report['fit_counts'] == {'outer_train_normalizer_fits':2,'ridge_fits':6,'tiny_fits':6,
                    'validation_fits':0,'encoder_calls':0,'pca_fits':0}, 'TEMPO exact fit counts')
            a.check(len(report['methods']) == 2 and not report['pairs'] and {m['method'] for m in report['methods']} == set(MODES), 'Only declared two input policies')
            for method in report['methods']:
                mode = method['method']
                a.check(method['size'] == 36 and method['status'] == 'measured' and not method['inputs_train_prepared'] and
                        method['outer_train_normalizer_fits'] == 1 and len(method['repetitions']) == 3, 'Fresh native36 classification fitting')
                for j, rep in enumerate(method['repetitions']):
                    a.check(rep['id'] == 'rep-'+str(REPS[j]) and rep['status'] == 'measured' and
                            int(rep['actual_probe_seed_decimal']) == seed(REPS[j]) and rep['ridge_parameters'] == 2*(36+1) and
                            rep['neural_parameters'] == 36*16+16+16*2+2, 'TEMPO D36 parameters and exact stream seed')
                    for view in ('training',)+VIEWS:
                        score = rep[view];n,groups = (512,256) if view == 'training' else (256,128)
                        p = score['population']
                        a.check(p['total_rows'] == p['valid_rows'] == n and p['total_source_groups'] == p['valid_source_groups'] == groups,
                                'Complete source-group population')
                        full_coverage &= p['valid_rows'] == n and p['valid_source_groups'] == groups
                        a.close(p['coverage'],p['valid_rows']/n,'TEMPO population coverage')
                        for head in HEADS:
                            h = score[head]
                            a.check(h['total'] == h['valid'] == n and 0 <= h['correct'] <= n, 'TEMPO correct/valid counts')
                            a.close(h['accuracy'],h['correct']/n,'Independent TEMPO accuracy arithmetic')
                            a.close(h['coverage'],h['valid']/n,'TEMPO head coverage')
                            a.close(h['full_population_correctness'],h['correct']/n,'TEMPO full-population correctness')
                            synthetic_values.setdefault((stage,mode,view,head,master),[]).append(h['accuracy'])
                        if view in VIEWS:
                            for key,head in (('ridge_grouped_interval','ridge'),('tiny_grouped_interval','tiny_secondary')):
                                ci = score[key]
                                a.check(ci['replicates'] == 1000 and ci['confidence'] == .95 and ci['source_groups'] == groups and
                                        0 <= ci['lower'] <= ci['upper'] <= 1, 'Preserved conditional group bootstrap interval')
                                a.close(ci['estimate'],score[head]['accuracy'],'Interval point estimate')
        extraction = read(args.run/'features'/stage/'summary.json')
        a.check(extraction['source_updates'] == 0 and extraction['causal_updates'] == (0 if stage == 'initial' else 2048) and
                extraction['full_CUDA_encode_calls'] == 786 and extraction['synthetic_rows'] == 6144 and
                extraction['all_parameters_buffers_normalizer_scaler_unchanged'] and extraction['all_gradients_undefined'] and
                extraction['no_labels_in_encoder'] and a.path(extraction['checkpoint']).resolve(strict=True) == checkpoints[stage],
                'Exactly one frozen export per initial/final checkpoint')
        a.check(extraction['market']['train']['full_CUDA_encode_calls'] == 14 and extraction['market']['validation']['full_CUDA_encode_calls'] == 4 and
                extraction['market_scaler_identity'] == calibration['input_scaler_identity'] and extraction['bank_normalizer_identity'] == calibration['bank_identity'],
                'Fresh calibration and exact market extraction count')
        for key in ('encoder_updates','decoder_calls','old_encoder_calls','new_input_scaler_fits','new_bank_normalizer_fits'):
            a.check(extraction[key] == 0,'Frozen export zero operation ' + key)
    for stage in STAGES:
        for mode in MODES:
            for view in VIEWS:
                panel = next(p for p in tables['synthetic']['panels'] if p['input_policy'] == mode and p['view'] == view)
                row = next(r for r in panel['rows'] if r['checkpoint_stage'] == stage)
                a.check(row['size'] == 36 and not row['reused'] and row['method'] == (INIT if stage == 'initial' else TAG) and row['coverage'] == 1,
                        'Synthetic panel provenance/full coverage')
                for head,field in (('ridge','linear_head_accuracy'),('tiny_secondary','neural_head_accuracy')):
                    cohort_means = [mean(synthetic_values[(stage,mode,view,head,master)]) for master in MASTERS]
                    aggregate = tables['synthetic']['scores'][stage][mode][view][head]
                    a.check(len(aggregate['master_means']) == 3,'All three cohort means retained')
                    for got,want in zip(aggregate['master_means'],cohort_means):a.close(got,want,'Independent cohort repetition mean')
                    for key,want in (('mean',mean(cohort_means)),('minimum',min(cohort_means)),('maximum',max(cohort_means))):
                        a.close(aggregate[key],want,'Independent across-cohort summary')
                    a.close(row[field],mean(cohort_means),'Reported synthetic panel mean')
                    synthetic_audit.append({'stage':stage,'policy':mode,'view':view,'head':head,'cohort_means':cohort_means,'mean':mean(cohort_means)})
    exports = read(args.run/'features/extraction-summary.json')
    a.check(exports['status'] == 'passed' and exports['full_CUDA_encode_calls'] == 1572 and
            exports['new_synthetic_TRAIN_input_scaler_fits'] == 3 and exports['new_market_input_scaler_fits'] == exports['new_bank_normalizer_fits'] == 0 and
            exports['old_encoder_calls'] == exports['old_model_or_scaler_archives_opened'] == 0, 'Two new exports; only fresh synthetic TRAIN fits')
    for stage in STAGES:
        reader = read(args.run/stage/'synthetic-readouts/reader-summary.json')
        a.check(reader['model'] == TAG and reader['checkpoint_stage'] == stage and reader['native_width'] == 36 and
                all(reader[k] == 0 for k in ('encoder_calls','encoder_updates','old_encoder_calls','old_head_fits','post_encoder_pca_fits')),
                'Saved-stage CPU fixed-head reader provenance')
    for change in tables['descriptive_changes']['market']:
        rows = [next(r for r in market_audit if r['stage'] == stage and r['task'] == change['task']) for stage in STAGES]
        for metric, field in (('linear_r2','linear_r2'),('neural_mean_r2','neural_mean_r2'),('linear_mae','linear_MAE'),('neural_mean_mae','neural_mean_MAE')):
            a.close(change[metric]['initial'],rows[0][field],'Descriptive market initial value')
            a.close(change[metric]['final'],rows[1][field],'Descriptive market final value')
            a.close(change[metric]['difference'],rows[1][field]-rows[0][field],'Descriptive market difference')
    for change in tables['descriptive_changes']['synthetic']:
        rows = [next(r for r in synthetic_audit if r['stage'] == stage and r['policy'] == change['mode'] and
                     r['view'] == change['view'] and r['head'] == change['head']) for stage in STAGES]
        a.close(change['initial'],rows[0]['mean'],'Descriptive synthetic initial value')
        a.close(change['final'],rows[1]['mean'],'Descriptive synthetic final value')
        a.close(change['difference'],rows[1]['mean']-rows[0]['mean'],'Descriptive synthetic difference')
    for key in ('trust_protocol_admission_passed','full_coverage','clean_causal_candidate_measured'):
        a.check(tables[key] is True and completion[key] is True, 'Clean-baseline trust/completeness claim ' + key)
    a.check(full_coverage and not completion['historical_encoder_quality_targets_used'] and not completion['overall_winner_declared'],
            'Trust/completeness establish a measured candidate; numerical advancement is separately audited')
    assets = read(args.report/'asset_hashes.json')
    for binding in assets['run_assets']:a.bound(binding)
    for key in ('training_summary','initialization','calibration','typed_resume','query_initial','query_final','engineering_proof','build_admission','reporter','variance_gradient_proof'):
        a.bound(assets[key])
    for key in ('report','tables','assets'):a.bound(completion[key])
    for stage in STAGES:
        for key in ('proof','witness'):a.bound(assets['proofs'][stage][key])
    for path in (args.card,args.input_manifest,args.raw_controls,args.quality_receipt,args.training/'training-summary.json',
                 args.training/'fresh-initialization.json',args.training/'calibration/summary.json',Path(__file__),
                 Path(__file__).with_name('archive_reader.py'),args.audit_card):
        a.pin(path)
    result = {'kind':'causal_fresh_v4_quality_independent_audit_v1','status':'passed' if not a.errors else 'failed',
        'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':TAG,'checks':a.checks,'mismatches':a.errors,
        'scope':'Independent stdlib saved-scalar and inert saved-array arithmetic/hash bindings; no executable model loading, neural/encoder forwards, training or fitting.',
        'market_new_rows':8,'raw_control_rows':8,'synthetic_cohorts_per_stage':3,'input_policies':2,'head_repetitions':3,
        'export_calls':1572,'initial_and_final_quality_new':True,'historical_quality_targets_used':False,
        'candidate_measured_rule':'Fresh ancestry/trust/full coverage; independently audit22 numeric advancement conditions; clean V3 baseline unchanged.',
        'saved_prediction_arithmetic':saved_prediction_arithmetic,'clean_baseline_comparison':clean_comparison,
        'query_witness_arithmetic':query_witness_arithmetic,
        'paired_initial_equality':paired_initial,
        'market_arithmetic':market_audit,'synthetic_arithmetic':synthetic_audit,'full_coverage':full_coverage,
        'source_updates':0,'initialization_seed':943001,'proof_checks':{s:v['checks'] for s,v in proof_values.items()},
        'verified_artifacts':list(a.pins.values()),'path_maps':a.maps,
        'limitations':['All112TRAIN/32VALIDATION saved market predictions and physical metrics are inspected. Saved ridge outputs are reconstructed from coefficients; neural outputs are never recomputed.',
                       'TRAIN feature/target normalization moments are descriptive arithmetic checks of saved fitted statistics, not new fitted readouts/scalers.',
                       'TEMPO-5 correct/counts, means, groups, recipes/seeds and retained intervals are checked from saved scalar reports; individual neural classification outputs are not regenerated.',
                       'The recorded trust proofs supply actual CUDA learning, causal interventions and typed-resume evidence; this auditor binds their saved bytes.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'checks':a.checks,'mismatches':a.errors,'artifacts':len(a.pins),'output':str(args.output)}))
    return 0 if not a.errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
