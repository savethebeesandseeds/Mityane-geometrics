#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent saved-scalar audit. Standard library only; never loads a model."""
# Complete owned final V8 saved-quality auditor copy, independently adapted for V9.
# No producer/comparator/training-auditor implementation is imported.
import argparse
import datetime
import hashlib
import json
import math
import struct
from pathlib import Path
from archive_reader import Archive, Tensor, DTYPES

CARD_SHA = None  # Frozen audit card binds the measured prospective main card.
INPUT_SHA = None  # Bound by the separately admitted audit card, without a hash cycle.
TAG, INIT = 'CPB-v9.market-01', 'CPB-v9.market-01.init'
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
REPS, MASTERS = (2701, 2802, 2903), (941101, 941102, 941103)
STAGES, MODES = ('initial', 'final'), ('synthetic_train', 'market_scaler')
VIEWS, HEADS = ('validation_intact', 'validation_deleted'), ('ridge', 'tiny_secondary')
BASE = 'CPB-v3.market-01'
POLICY = 'huber1-mean-live;gaussian-variance-detached-error-live-uncertainty4;shared-global-clip1-v1'
OBJECTIVE = 'huber1-mean-plus-detached-error-gaussian-variance-v1'
ENCODER_KIND = 'Huber1-mean-gradient;detached-error-Gaussian-variance;live-uncertainty4;shared-global-clip1'
PENDING = {'initial_native20_40_and_statistics_pairings': 'pending', 'independent_training_audit': 'pending',
           'independent_quality_audit': 'pending'}
OBJECTIVE_FIELDS = {'objective_policy': OBJECTIVE, 'gradient_policy': POLICY,
                    'mean_huber_delta': 1., 'loss_coefficients': [1., 1.],
                    'variance_error_detached': True, 'reporting_nll_backpropagated': False}
TEMPORAL_POLICY = 'original-position-four-bin-shared-global-half-mixture-v1'
TEMPORAL_FIELDS = {'temporal_pooling_policy': TEMPORAL_POLICY,
                   'temporal_pooling_partition': [4,16,32,4,8],
                   'temporal_pooling_mix': [0.5,0.5],
                   'empty_local_bin_policy': 'zero-whole-export-slice-after-bias-v1'}
GAUGE = 'core.encoder.level.base.backbone.pool_score.bias'
LEARNING_FIELDS = {'registered_learned_tensors':76,
                   'mandatory_identifiable_learned_tensors':75,
                   'data_gradient_gauge_exemption':GAUGE}


def temporal_contract(a, value, origin):
    for key, expected in TEMPORAL_FIELDS.items():
        a.check(value.get(key) == expected, 'Actual temporal pooling contract '+origin+':'+key)


def learning_contract(a, value, origin):
    for key, expected in LEARNING_FIELDS.items():
        a.check(value.get(key) == expected, 'Explicit identifiable/gauge contract '+origin+':'+key)


def objective_contract(a, value, origin, require_gradient_policy=True):
    for key, expected in OBJECTIVE_FIELDS.items():
        if key == 'gradient_policy' and not require_gradient_policy:
            continue
        a.check(value.get(key) == expected, 'Actual V9 objective contract ' + origin + ':' + key)


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
    x = (rep + 0x9e3779b97f4a7c15 * 40) & mask
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
    objective_contract(a, value, 'saved causal proof')
    temporal_contract(a, value, 'saved causal proof')
    fields = {'kind': 'cpb_v9_saved_checkpoint_causal_proof_v1', 'status': 'passed',
              'gradient_policy': POLICY,
              'model': 'CPB-v9', 'source_updates': 0, 'causal_updates': updates,
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
            path = args.run/'features'/stage/'market'/('cpb_v9-'+role+'.pt')
            a.pin(path); archive = Archive(path)
            try:
                native[role] = {'values':tensor(archive,'values',(n,40)),
                                'valid':tensor(archive,'valid',(n,)),
                                'row_ids':tensor(archive,'row_ids',(n,))}
            finally:
                archive.close()
            a.check(native[role]['row_ids'] == reference[role]['row_ids'], 'Saved feature/target exact row association '+stage+role)
            a.check(all(math.isfinite(v) for v in native[role]['values']), 'Finite saved native arrays '+stage+role)
        train = reference['train']
        feature_ids = [r for r in range(112) if native['train']['valid'][r]]
        feature_mean,feature_scale,target_mean,target_scale = [],[],[],[]
        for d in range(40):
            column = [native['train']['values'][r*40+d] for r in feature_ids]
            center = mean(column)
            feature_mean.append(center)
            feature_scale.append(max(1e-8,math.sqrt(mean([(v-center)**2 for v in column]))))
        for t in range(4):
            column = [train['targets'][r*4+t] for r in range(112) if train['valid'][r*4+t]]
            center = mean(column);target_mean.append(center)
            target_scale.append(max(1e-8,math.sqrt(mean([(v-center)**2 for v in column]))))
        reported = read(args.run/stage/'market-readouts/report.json')['methods'][0]['repetitions']
        for rep_index,rep in enumerate(REPS):
            folder = args.run/stage/'market-readouts/models/cpb_v9'/('rep-'+str(rep))
            fit_path = folder/'fit.pt';a.pin(fit_path);fit = Archive(fit_path)
            try:
                fitted = {name:tensor(fit,name,shape) for name,shape in (
                    ('feature_mean',(40,)),('feature_scale',(40,)),('target_mean',(4,)),('target_scale',(4,)),
                    ('ridge_weights',(40,4)),('ridge_intercept',(4,)),('supported_targets',(4,)),
                    ('target_training_counts',(4,)),('target_training_supported',(4,)))}
                actual_seed = tensor(fit,'actual_seed',())[0]
                a.check(actual_seed % (1<<64) == seed(rep), 'Saved fitted D40 seed signed-to-unsigned identity')
                a.check(tensor(fit,'training_feature_rows',()) == [len(feature_ids)] and
                        tensor(fit,'feature_training_supported',()) == [True], 'Saved TRAIN normalization support')
                for name,expected in (('feature_mean',feature_mean),('feature_scale',feature_scale),
                                      ('target_mean',target_mean),('target_scale',target_scale)):
                    for got,want in zip(fitted[name],expected):
                        a.close(got,want,'Independent TRAIN population statistic '+stage+name)
                a.check(all(fitted['supported_targets']) and all(fitted['target_training_supported']) and
                        fitted['target_training_counts'] == [112]*4, 'Saved target support/counts')
                for name,shape in (('neural_w1',(40,16)),('neural_b1',(16,)),('neural_w2',(16,4)),('neural_b2',(4,))):
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
                        standardized = math.fsum((native[role]['values'][r*40+d]-fitted['feature_mean'][d])/
                            fitted['feature_scale'][d]*fitted['ridge_weights'][d*4+t] for d in range(40))+fitted['ridge_intercept'][t]
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


def temporal_evidence_audit(a,args,quality,build,training):
    """Bind new temporal fixture and actual-learning receipts; no graph replay."""
    path=a.bound(quality['temporal_pooling_proof'],args.quality_receipt.parent)
    value=read(path)
    temporal_contract(a,value,'dedicated temporal fixture')
    expected={'kind':'cpb_v9_temporal_pooling_mechanism_proof_v1','status':'passed','model':'CPB-v9',
              'actual_CUDA':True,'original_position_membership':True,'both_temporal_paths':True,
              'empty_bins_finite_zero':True,'true_per_bin_trace_weights':True,
              'manual_pooling_and_adjoint_parity':True,'four_export_slices_gradient_and_movement':True,
              'centered_position_contrasts_gradient_and_movement':True,'common_softmax_gauge_invariance':True,
              'shared_global_half_mixture':True,'relative_bin_global_sensitivity':True,
              'three_between_bin_contrasts_gradient_and_movement':True,
              'backbone_encode_calls':1,'direct_pool_kernel_calls':5,
              'public_encoder_calls':0,'head_fits':0,'synthetic_optimizer_steps':1,
              'manual_F32_absolute_tolerance':2e-5,'manual_F32_relative_tolerance':2e-5,
              'softmax_gauge_tolerance':2e-6,'bin_mass_tolerance':2e-6,'flat_trace_tolerance':8e-6,
              'empty_and_membership_zero':'exact'}
    for key,want in expected.items():a.check(value.get(key)==want,'Temporal fixture saved predicate '+key)
    binary,log=(a.bound(value[key],path.parent) for key in ('binary','log'))
    a.bound(value['execution'],path.parent)
    a.check(binary.name=='test_cpb_v9_temporal_pooling' and
            binary in {a.path(p['path']).resolve(strict=True) for p in build['binaries']} and
            all(token in log.read_text(encoding='utf-8') for token in
                ('PASS CPB-v9 original-position-four-bin-shared-global-half-mixture-v1:',
                 'manual original-position half-mixture pooling parity','four 8-row projection gradient and movement',
                 'three between-bin Hadamard contrast gradient and movement','relative-bin global sensitivity',
                 'empty-bin finite zero','common softmax gauge invariance','synthetic optimizer steps=1',
                 'backbone_encode_calls=1','public_encoder_calls=0','direct_pool_kernel_calls=5','head_fits=0')),
            'Exact admitted temporal binary and observed independent fixture PASS')
    receipt_path=a.bound(training['temporal_pooling_learning'],args.training)
    a.check(a.bound(quality['temporal_learning']['receipt'])==receipt_path,
            'Quality execution binds actual main temporal-learning receipt')
    observed=read(receipt_path)
    temporal_contract(a,observed,'actual temporal learning')
    for key,want in {'kind':'cpb_v9_actual_temporal_pooling_learning_v1','status':'passed','updates':2048,
                     'group_rows':8192,'extra_encoder_calls':0,'weight_decay_counted_as_data_gradient':False,
                     'movement_includes_AdamW_weight_decay':True}.items():
        a.check(observed.get(key)==want,'Actual temporal learning receipt '+key)
    a.check(len(observed['groups'])==4,'Four actual temporal groups')
    for index,group in enumerate(observed['groups']):
        a.check([group[k] for k in ('bin','export_row_begin','export_row_end','position_begin','position_end')]
                ==[index,8*index,8*(index+1),4*index,4*(index+1)],'Exact original-position/export-row groups')
        a.check(all(isinstance(group[k],int) and 0<group[k]<=2048 for k in
                    ('weight_nonzero_data_gradient_steps','bias_nonzero_data_gradient_steps',
                     'weight_movement_steps','bias_movement_steps','position_centered_data_gradient_steps',
                     'position_centered_movement_steps')) and group['final_weight_changed'] is True and
                group['final_bias_changed'] is True and math.isfinite(group['final_position_centered_change_l2'])
                and group['final_position_centered_change_l2']>0,
                'Each actual group has data gradients and movement; no every-update requirement')
    gauge=observed['gauge']
    a.check(gauge['name']==GAUGE and gauge['data_gradient_requirement_exempt'] is True and
            gauge['registered_in_same_AdamW'] is True and gauge['movement_is_not_data_learning_proof'] is True,
            'Observed registered scalar softmax gauge; decay is not data-gradient learning')
    artifact_paths=[a.bound(binding,receipt_path.parent) for binding in observed['artifacts']]
    between=observed['between_bin_positions']
    for key,want in {'basis':'hadamard-full16-signs-div4-v1',
                     'basis_signs':[[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],
                     'full_position_divisor':4,'position_bin_width':4,'rows':2048,
                     'gradient_definition':'observed-post-globalclip-data-gradient',
                     'movement_includes_AdamW_weight_decay':True,
                     'weight_decay_counted_as_data_learning':False,'extra_encoder_calls':0}.items():
        a.check(between.get(key)==want,'Fixed Hadamard between-bin learning contract '+key)
    a.check(len(between['gradient_steps'])==len(between['movement_steps'])==3 and
            all(isinstance(n,int) and 0<n<=2048 for n in between['gradient_steps']+between['movement_steps']) and
            len(between['final_contrast_changes'])==3 and
            all(math.isfinite(x) for x in between['final_contrast_changes']) and
            math.isfinite(between['final_projection_change_l2']) and between['final_projection_change_l2']>0,
            'Three observed direction data-gradient/movement events; overall final projection change only')
    a.check(math.isclose(math.fsum(x*x for x in between['final_contrast_changes']),
                         between['final_projection_change_l2']**2,rel_tol=1e-10,abs_tol=1e-12),
            'Saved three orthonormal coefficients preserve final full-position between projection norm')
    a.check(len(artifact_paths)==3 and {p.name for p in artifact_paths}=={
                'temporal-projection-learning.csv','temporal-pool-score-bias-learning.csv',
                'temporal-between-bin-position-learning.csv'},
            'Exact8192 group,2048 gauge and2048 between-bin evidence artifacts')
    with (args.training/'temporal-between-bin-position-learning.csv').open(encoding='utf-8') as stream:
        a.check(stream.readline().rstrip('\r\n').split(',')==[
                'update','between_bin_gradient_l2','between_bin_movement_l2',
                'contrast_0_gradient','contrast_1_gradient','contrast_2_gradient',
                'contrast_0_movement','contrast_1_movement','contrast_2_movement'],
                'Exact nine-column fixed-basis between-bin observation schema')
    for name,rows in (('temporal-projection-learning.csv',8192),('temporal-pool-score-bias-learning.csv',2048),
                      ('temporal-between-bin-position-learning.csv',2048)):
        with (args.training/name).open(encoding='utf-8') as stream:
            a.check(sum(1 for _ in stream)==rows+1,'Recorded row count '+name)
    a.check(training['all_four_temporal_export_row_groups_learned'] is True and
            training['all_four_pool_position_contrasts_learned'] is True and
            training['all_three_between_bin_position_directions_learned'] is True and
            training['final_between_bin_position_projection_changed'] is True and
            training['all_mandatory_identifiable_tensors_learned'] is True and
            training['mandatory_identifiable_tensor_value_capacity']==294751 and
            training['mandatory_identifiable_count_semantics']=='tensor numel capacity, not every-scalar gradient/movement proof' and
            training['per_scalar_nonzero_gradient_proof_claimed'] is False and
            training['recorded_movement_capacity_semantics']=='sum of numel of learned tensors with at least one recorded movement event; not a count of individually moved scalar values',
            'Main reports75 identifiable tensors and capacity; no every-scalar proof; gauge observed separately')
    parameter_path=args.training/'parameter-learning.json';a.pin(parameter_path)
    parameters=read(parameter_path)
    learned=[p for p in parameters if p['requires_grad']]
    mandatory=[p for p in learned if p['name']!=GAUGE]
    a.check(len(learned)==76 and len(mandatory)==75 and sum(p['values'] for p in learned)==294752 and
            sum(p['values'] for p in mandatory)==294751,'Registered capacity and exact mandatory scope')
    a.check(all(p['data_gradient_requirement_exempt']==(p['name']==GAUGE) for p in parameters) and
            all(p['nonzero_CUDA_gradient_steps']>0 and p['weight_movement_steps']>0 for p in mandatory),
            'Every identifiable tensor has recorded data-gradient/movement; only gauge exempt')
    return {'proof':a.pin(path),'binary':a.pin(binary),'log':a.pin(log),'learning_receipt':a.pin(receipt_path),
            'learning_artifacts':[a.pin(p) for p in artifact_paths],
            'scope':'Saved predicates/pins/row counts only; independent training auditor checks observed group arithmetic.',
            'registered_learned_tensors':76,'mandatory_identifiable_tensors':75,'gauge_parameter':GAUGE,
            'mandatory_identifiable_tensor_value_capacity':294751,
            'per_scalar_nonzero_gradient_proof_claimed':False,
            'weight_decay_counted_as_data_gradient':False}


def paired_initial_audit(a, args, declaration):
    """Compare native20:40 and whole saved V8 TRAIN statistics; no old models."""
    pairs = []
    feature_archives = statistic_archives = 0
    for entry in declaration['paired_initial_reference_inputs']:
        reference = a.bound(entry['reference_data_archive'])
        branch, relative = entry['candidate_relative_to_attempt'].split('/', 1)
        if branch not in ('quality-001', 'run-001'):
            raise ValueError('Unrecognized declared candidate evidence branch')
        current = (args.run if branch == 'quality-001' else args.training) / relative
        reference_pin, current_pin = a.pin(reference), a.pin(current)
        old, new = Archive(reference), Archive(current)
        numeric = []
        is_feature = branch == 'quality-001' and '/features/initial/' in ('/' + entry['candidate_relative_to_attempt'])
        feature_archives += int(is_feature)
        statistic_archives += int(not is_feature)
        shapes = {item['field']: item for item in entry['numeric_reference_shapes']}
        a.check(set(shapes) == set(entry['numeric_fields']), 'Exact declared pair field/shape allowlist')
        try:
            for field in entry['numeric_fields']:
                left, right = old.field(*field.split('.')), new.field(*field.split('.'))
                if not isinstance(left, Tensor) or not isinstance(right, Tensor):
                    raise ValueError('Pairing requires inert tensors: ' + field)
                shape, dtype = tuple(shapes[field]['shape']), shapes[field]['dtype']
                a.check(left.shape == right.shape == shape and left.storage.dtype == right.storage.dtype == dtype,
                        'V8/V9 declared numeric shape/dtype ' + entry['id'] + ':' + field)
                x, y = left.values(), right.values()
                is_feature_values = is_feature and field in ('values','training_values','validation_values','deleted_values')
                if is_feature_values:
                    a.check(len(shape) == 2 and shape[1] == 40 and
                            entry['feature_value_coordinate_slice'] == [20,40] and
                            all(math.isfinite(v) for v in y),
                            'Finite whole native40 array, exact comparison restricted to20:40')
                    # The new pooling changes content0:20. Never compare or gate
                    # those values against the old encoder; preserve their finiteness.
                    x = [v for row in range(shape[0]) for v in x[row*40+20:row*40+40]]
                    y = [v for row in range(shape[0]) for v in y[row*40+20:row*40+40]]
                fmt = DTYPES[dtype][0]
                exact = (len(x) == len(y) and
                         struct.pack('<' + str(len(x)) + fmt, *x) == struct.pack('<' + str(len(y)) + fmt, *y))
                a.check(exact, 'Native20:40 or whole support/statistic logical bytes exact ' + entry['id'] + ':' + field)
                numeric.append({'field': field, 'reference_shape': list(left.shape),
                                'candidate_shape': list(right.shape), 'dtype': dtype,
                                'compared_logical_values': len(x),
                                'comparison': 'native columns20:40 only' if is_feature_values else 'whole declared tensor',
                                'native_coordinate_slice': [20,40] if is_feature_values else None,
                                'tensor_bytes_equal': exact})
        finally:
            old.close()
            new.close()
        pairs.append({'id': entry['id'], 'v8_reference_data': reference_pin,
                      'v9_initial_or_fresh_statistics': current_pin,
                      'whole_file_bytes_equal': reference_pin['bytes'] == current_pin['bytes']
                                                and reference_pin['sha256'] == current_pin['sha256'],
                      'all_declared_numeric_tensor_values_exact': all(item['tensor_bytes_equal'] for item in numeric),
                      'numeric_tensors': numeric,
                      'metadata_interpretation': 'File bytes/hashes reported separately from logical tensor equality; version/source text is not required equal.'})
    a.check(feature_archives == 8 and statistic_archives == 5 and len(pairs) == 13,
            'Exactly8 partial-native20:40 initial DATA and5 whole fresh statistic archives')
    return {'scope': 'Native20:40 initial DATA/whole statistic pairing only; no V8/V9 model/head execution or fitting',
            'old_model_numeric_reads': 0, 'reference_data_archives': len(pairs),
            'feature_archives':feature_archives,'statistic_archives':statistic_archives,
            'feature_coordinate_slice':[20,40],'content0_20_equality_required':False,
            'whole_native40_equality_claimed':False,
            'all_declared_numeric_tensor_values_exact': all(p['all_declared_numeric_tensor_values_exact'] for p in pairs),
            'file_and_metadata_equality_reported_separately': True, 'pairs': pairs}


def comparison_audit(a,args,card,tables):
    """Independent max/min controls and22 operators; never imports reporter code."""
    origin = args.card.resolve().parent.parent
    reference_labels = {'v3':BASE,'v4':'CPB-v4.market-01','v5':'CPB-v5.market-01',
                        'v6':'CPB-v6.market-01','v7':'CPB-v7.market-01','v8':'CPB-v8.market-01'}
    controls,control_paths = {},{}
    establishment_paths = {}
    for version,label in reference_labels.items():
        p=a.bound(card['comparison_control'][version],origin)
        control_paths[label],controls[label]=p,read(p)
        r=controls[label]
        a.check(r['model']==label and r['source_updates']==0 and r['full_coverage'],
                'Complete preserved clean control '+label)
        if version in ('v6','v7','v8'):
            establishment_path=a.bound(card['comparison_control'][version+'_establishment'],origin)
            establishment_paths[label]=establishment_path
            established=read(establishment_path)
            a.check(established['status']=='passed' and established['candidate']==label and
                    established['measured_candidate_established'] and established['trust_admission_passed'] and
                    established['initial_partial_native20_40_pairings_passed' if version == 'v8' else 'initial_full40_pairings_passed'] and established['initial_pairing_count']==13 and
                    established['independent_training_audit_passed'] and established['independent_quality_audit_passed'] and
                    established['full_coverage'] and established['evidence']['table_rows']['sha256']==sha(p) and
                    r['runtime_trust_admission_passed'] and r['trust_protocol_admission_passed'] is False,
                    'Separate completed reference trust; raw producer pending bytes retained '+label)
        else:
            a.check(r['trust_protocol_admission_passed'],'Existing completed control trust '+label)
    goal_pin={**card['advancement']['goal_protocol'],'path':'doc/causal_fresh_v9/goal-protocol.json'}
    goal_path=a.bound(goal_pin,origin);goal=read(goal_path)
    a.check(sha(goal_path)=='e7920442fbc5528cbe7e67f71ada65101974a6a28ad4d39bc57dea461c477699'
            and goal['kind']=='clean_causal_encoder_search_goal_v8_retention_protocol_v1'
            and goal['criterion_count']==22 and goal['additional_train_NLL_gate'] is False
            and goal['criteria']==card['advancement']['criteria'],'Exact authoritative22-criterion goal; no TRAIN-NLL gate')
    retention_review_path=a.bound(card['retention_independent_review'],origin)
    retention_admission_path=a.bound(card['retention_application_admission'],origin)
    retention_review,retention_admission=read(retention_review_path),read(retention_admission_path)
    a.check(retention_review['status']=='passed' and retention_review['checks']==506 and
            retention_review['protocol']['sha256']==sha(goal_path) and
            retention_admission['status']=='passed' and retention_admission['prospective_future_use_eligible'] and
            retention_admission['independent_review']['sha256']==sha(retention_review_path) and
            retention_admission['additional_TRAIN_NLL_gate'] is False,
            'Separately independently reviewed and applied future V8 retention protocol')
    comparison=read(args.comparison/'comparison.json')
    a.check(comparison['baseline']==BASE and comparison['scoped_reference']=='CPB-v8.market-01'
            and comparison['scoped_references']==['CPB-v4.market-01','CPB-v5.market-01','CPB-v6.market-01','CPB-v7.market-01','CPB-v8.market-01']
            and comparison['candidate']==TAG and comparison['baseline_unchanged']
            and comparison['baseline_scores_reused'] and not comparison['historical_pretrained_quality_floors_used']
            and not comparison['automatic_promotion'] and not comparison['application_changed']
            and not comparison['overall_winner_declared'],'Comparison scope/no automatic promotion')
    a.check(a.bound(comparison['card'])==args.card.resolve()
            and a.bound(comparison['candidate_table'])==(args.report/'table_rows.json').resolve(),
            'Comparison bound prospective card and candidate')
    for label,p in control_paths.items():
        a.check(a.bound(comparison['baseline_controls'][label])==p,'Six exact reused clean controls '+label)
    a.check(set(comparison['baseline_establishment_controls'])==set(establishment_paths),
            'Exactly the separately established V6/V7/V8 references')
    for label,p in establishment_paths.items():
        a.check(a.bound(comparison['baseline_establishment_controls'][label])==p,
                'Same completed control establishment '+label)
    market=lambda result,label:{r['task']:r for r in result['market_rows'] if r['method']==label}
    timing=lambda result,label:{(p['input_policy'],p['view']):next(r for r in p['rows'] if r['method']==label)
                               for p in result['synthetic']['panels']}
    sources={**controls,INIT:tables,TAG:tables}
    a.check(len(controls)==6 and len(sources)==8 and
            set(comparison['baseline_controls'])==set(controls),
            'All six reused controls plus the two new stages are retained')
    mk={label:market(r,label) for label,r in sources.items()}
    tm={label:timing(r,label) for label,r in sources.items()}
    expected={}
    floor_sources={}
    for task in TASKS:
        for metric in ('linear_r2','neural_mean_r2'):
            value=mk[TAG][task][metric]
            best=max(controls, key=lambda label: mk[label][task][metric])
            floor=mk[best][task][metric]
            floor_sources[task+':'+metric]=best
            expected[task+':'+metric]=(value,floor,value+1e-12>=floor,'market_retention','candidate + 1e-12 >= floor')
    for mode in MODES:
        for view in VIEWS:
            for metric in ('linear_head_accuracy','neural_head_accuracy'):
                value=tm[TAG][(mode,view)][metric]
                best=max(controls, key=lambda label: tm[label][(mode,view)][metric])
                floor=tm[best][(mode,view)][metric]
                floor_sources[':'.join((mode,view,metric))]=best
                expected[':'.join((mode,view,metric))]=(value,floor,value+1e-12>=floor,'tempo5_retention','candidate + 1e-12 >= floor')
    for split in ('train','validation'):
        value=tables['fixed_queries']['final'][split]['standardized_fixed_query_MAE']
        best=min(controls, key=lambda label: controls[label]['fixed_queries']['final'][split]['standardized_fixed_query_MAE'])
        floor=controls[best]['fixed_queries']['final'][split]['standardized_fixed_query_MAE']
        floor_sources[split+':standardized_MAE']=best
        expected[split+':standardized_MAE']=(value,floor,value<=floor+1e-12,'fixed_query_MAE','candidate <= floor + 1e-12')
    value=tables['fixed_queries']['final']['validation']['Gaussian_NLL_F64']
    best=min(controls,key=lambda label:controls[label]['fixed_queries']['final']['validation']['Gaussian_NLL_F64'])
    floor=controls[best]['fixed_queries']['final']['validation']['Gaussian_NLL_F64']
    expected['known_validation:Gaussian_NLL_F64']=(value,floor,value<floor-1e-12,'variance_quality','candidate < floor - 1e-12')
    floor_sources['known_validation:Gaussian_NLL_F64']=best
    value=mk[TAG]['past_trend']['neural_r2_each']
    expected['past_trend:all3_neural_R2_positive']=(value,0,len(value)==3 and all(v>0 for v in value),
                                                  'neural_trend_robustness','all three final neural trend R2 values > 0')
    for metric in ('linear_r2','neural_mean_r2'):
        value=mk[TAG]['past_relation'][metric]
        expected['past_relation:'+metric+':positive']=(value,0,value>0,'relation_usefulness','candidate > 0')
    a.check(len(expected)==len(goal['criteria'])==len(comparison['numeric_conditions'])==22,
            'Independent exact22 conditions')
    a.check({g['id'] for g in goal['criteria']}=={g['id'] for g in comparison['numeric_conditions']}==set(expected),
            'All frozen identities exactly once')
    by_goal={g['id']:g for g in goal['criteria']}
    counts={}
    for actual in comparison['numeric_conditions']:
        identity=actual['id'];value,floor,passed,category,rule=expected[identity]
        frozen=by_goal[identity]
        a.close(frozen['floor'],floor,'Independent strongest-clean floor '+identity)
        a.check(frozen['floor_source']==floor_sources.get(identity,'strict positivity'), 'Independent best-source/tie policy '+identity)
        a.check(frozen['rule']==rule and frozen['category']==category,'Frozen operator/category '+identity)
        a.check({k:v for k,v in actual.items() if k not in ('candidate','passed')}==frozen,
                'Comparator copies entire frozen criterion without reinterpretation '+identity)
        if isinstance(value,list):a.check(actual['candidate']==value,'All3 actual trend repetitions')
        else:a.close(actual['candidate'],value,'Independent candidate value '+identity)
        a.check(actual['passed'] is bool(passed),'Independent gate outcome '+identity)
        bucket=counts.setdefault(category,{'passed':0,'total':0});bucket['total']+=1;bucket['passed']+=int(passed)
    passed=sum(row[2] for row in expected.values())
    a.check(comparison['numeric_passed']==passed and comparison['numeric_total']==22 and
            comparison['categories']==counts
            and comparison['full_coverage'] and comparison['runtime_trust_admission_passed']
            and comparison['trust_admission_passed'] is False
            and comparison['mandatory_postmeasurement_trust']==PENDING
            and comparison['numeric_conditions_passed']==(passed==22)
            and comparison['numeric_and_coverage_conditions_passed']==(passed==22)
            and comparison['advancement_conditions_passed'] is False
            and comparison['overall_establishment_passed'] is False,
            'Independent gate/category/coverage disposition')
    a.check(len(comparison['market_rows'])==len(sources)*len(TASKS) and
            {(r['method'],r['task']) for r in comparison['market_rows']}==
            {(label,task) for label in sources for task in TASKS},
            'Complete exact eight-way four-task comparison coverage')
    for item in comparison['market_rows']:
        original=mk[item['method']][item['task']]
        a.check(item==dict(original,reused=item['method'] in controls),'Exact eight-way market source row')
        a.check(item['size']==(36 if item['method'] in (BASE,'CPB-v4.market-01') else 40),'Explicit D36/D40 comparison size')
    a.check(len(comparison['synthetic_panels'])==4 and
            {(p['input_policy'],p['view']) for p in comparison['synthetic_panels']}==
            {(mode,view) for mode in MODES for view in VIEWS},
            'Complete two-policy/two-view comparison panels')
    for panel in comparison['synthetic_panels']:
        a.check(len(panel['rows'])==len(sources) and
                {r['method'] for r in panel['rows']}==set(sources),
                'Each TEMPO panel preserves all six controls and both new stages')
        a.check(panel['recipe_id']=='tempo-5-slow-fast-nine-feature-v1' and
                panel['head_fitting_dataset_id']==panel['scoring_dataset_id']=='TEMPO-5','Timing recipe/fit-score identity')
        key=(panel['input_policy'],panel['view'])
        for item in panel['rows']:
            a.check(item==dict(tm[item['method']][key],reused=item['method'] in controls),'Exact eight-way timing row')
    for label,r in sources.items():
        wanted=0 if label==INIT else r['training']['GPU_step_seconds_synchronized']
        a.close(comparison['training_cost'][label]['seconds'],wanted,'Actual versus reused training cost '+label)
        a.check(comparison['training_cost'][label]['reused']==(label in controls),'Training reuse label '+label)
        a.check(comparison['fixed_queries'][label]==r['fixed_queries']['initial' if label==INIT else 'final'],
                'Exact fixed-query initial/final/control association '+label)
    for item in comparison['scoped_market_changes']:
        ref,task,metric=item['reference'],item['task'],item['metric']
        before,after=mk[ref][task][metric],mk[TAG][task][metric]
        a.close(item['reused_reference'],before,'Scoped original market value')
        a.close(item['candidate'],after,'Scoped candidate market value')
        a.close(item['difference'],after-before,'Scoped market difference')
        a.check(item['improved']==(after>before if metric.endswith('r2') else after<before),'Scoped improvement direction')
    for item in comparison['scoped_timing_changes']:
        key=(item['input_policy'],item['view']);metric=item['metric']
        before,after=tm[item['reference']][key][metric],tm[TAG][key][metric]
        a.close(item['reused_reference'],before,'Scoped original timing value')
        a.close(item['candidate'],after,'Scoped candidate timing value')
        a.close(item['difference'],after-before,'Scoped timing difference')
    def bound_tree(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}.issubset(value):a.bound(value)
            else:
                for child in value.values():bound_tree(child)
        elif isinstance(value,list):
            for child in value:bound_tree(child)
    bound_tree(comparison['candidate_trust_bindings'])
    a.bound(comparison['source']);a.bound(comparison['detailed_reporter_source'])
    for p in (*control_paths.values(),goal_path,args.comparison/'comparison.json',args.comparison/'report.md'):a.pin(p)
    return {'numeric_passed':passed,'numeric_total':22,
            'independent_conditions':{k:{'candidate':v[0],'floor':v[1],'passed':v[2],'category':v[3],'rule':v[4]} for k,v in expected.items()},
            'baseline':BASE,'scoped_reference':'CPB-v8.market-01','baseline_unchanged':True,
            'trust_separate':True,'coverage_separate':True}

def reporting_recovery(a, declaration, args):
    """Audit explicit reporting-only recovery without treating failed quality as passed."""
    card_path = a.bound(declaration['reporting_recovery_card'])
    receipt_path = a.bound(declaration['reporting_recovery_receipt'])
    a.check(card_path == args.reporting_recovery_card.resolve(strict=True)
            and receipt_path == args.reporting_recovery_receipt.resolve(strict=True),
            'Explicit separately declared reporting-recovery inputs')
    card, receipt = read(card_path), read(receipt_path)
    identity = {'card_sha256':CARD_SHA, 'manifest_sha256':declaration['main_source_manifest_sha256'],
                'archive_sha256':'b54ffb00a1489dc4973efe7d93819196b4e5bcd445140851d977ba14732aa867',
                'source_files':118, 'binary_count':7, 'translation_units':11}
    a.check(card['kind'] == 'cpb_v9_report_only_recovery_card_v1' and card['root_reviewed'] is True
            and receipt['kind'] == 'cpb_v9_report_only_recovery_receipt_v1' and receipt['status'] == 'passed'
            and card['main_source_identity'] == receipt['main_source_identity'] == identity,
            'Same frozen scientific118 capsule with separate passed report-only recovery')
    a.check(a.bound(receipt['card']) == card_path, 'Recovery receipt binds exact separately reviewed card')
    a.check(card['failure_mirror']['sha256'] == 'e30f6cd71267be7b97337423b9fa2029922ee059685736bcff0fa53ac9cba532'
            and card['failure_mirror'] == receipt['failure_mirror'], 'Original failure mirror retained')
    a.bound(card['failure_mirror'])
    for name in ('failed_quality_stage', 'failed_report_stage'):
        a.check(card[name] == receipt[name], 'Original failed-stage identity: '+name)
        value = read(a.bound(card[name]))
        a.check(value['exit_code'] == 1 and (value['status'] == 'failed' if name == 'failed_quality_stage'
                else value['stage'] == 'report'), 'Original scientific wrapper/report stage remains failed')
    names = {'extraction','initial_market_heads','initial_synthetic_heads','final_market_heads','final_synthetic_heads'}
    a.check(set(card['scientific_stages']) == names and card['scientific_stages'] == receipt['scientific_stages'],
            'Five already-completed scientific stages, no replay')
    for name, binding in card['scientific_stages'].items():
        value = read(a.bound(binding))
        a.check(value['stage'] == name and value['exit_code'] == 0, 'Passed original scientific stage: '+name)
    a.check(receipt['scientific_inputs_before'] == receipt['scientific_inputs_after']
            and len(receipt['scientific_inputs_before']) == len(card['scientific_inputs']) == 531,
            'All531 scientific/failed input files unchanged by reporting')
    identities = lambda rows:{(item['path'],item['bytes'],item['sha256']) for item in rows}
    a.check(identities(receipt['scientific_inputs_before']) == identities(card['scientific_inputs']),
            'Exact531 before/after identities agree with prospective recovery declaration')
    for binding in receipt['scientific_inputs_after']:
        a.bound(binding)
    for key in ('report','comparison'):
        a.bound(receipt[key]['stage'])
        for binding in receipt[key]['output_pins']:
            a.bound(binding)
    a.check(a.bound(receipt['reporter']) == a.bound(card['reporter'])
            and a.bound(receipt['comparator']) == a.bound(card['comparator']),
            'Truthful external reporter and unchanged original comparator')
    a.check(receipt['original_quality_stage_status'] == 'failed'
            and receipt['completed_scientific_stages'] == 5
            and receipt['scientific_inputs_before_after_exact'] is True
            and receipt['frozen_scientific_source_unchanged'] is True
            and all(receipt[key] == 0 for key in ('encoder_calls','head_fits','training_updates','science_stages_replayed')),
            'Recovery runs aggregation only; no encoder/head/training/scientific replay')
    return {'status':'passed', 'card':a.pin(card_path), 'receipt':a.pin(receipt_path),
            'original_quality_stage_status':'failed', 'completed_scientific_stages':5,
            'science_stages_replayed':0, 'scientific_inputs_before_after_exact':True,
            'frozen_scientific_source_unchanged':True}


def main():
    global CARD_SHA, INPUT_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('card', 'input-manifest', 'raw-controls', 'training', 'run', 'report', 'quality-receipt',
                'comparison', 'audit-card', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/cuwacunu'))
    parser.add_argument('--reporting-recovery-card', type=Path, required=True)
    parser.add_argument('--reporting-recovery-receipt', type=Path, required=True)
    parser.add_argument('--map', action='append', default=[], metavar='ORIGINAL_PREFIX=LOCAL_PREFIX')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve existing audit: ' + str(args.output))
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('Execute this scalar auditor inside the existing managed container')
    a = Audit([item.split('=', 1) for item in args.map])
    declaration = read(args.audit_card)
    a.check(declaration['kind']=='causal_fresh_v9_saved_quality_audit_card_v1' and
            declaration['execution_pending'] and declaration['model_calls']==declaration['head_fits']==0,
            'Prospective saved-evidence audit declaration')
    CARD_SHA=declaration['main_card']['sha256']
    INPUT_SHA=declaration['input_manifest']['sha256']
    recovery_evidence = reporting_recovery(a, declaration, args)
    a.check(set(entry['file'] for entry in declaration['auditor_sources'])==
            {'audit.py','archive_reader.py','README.md','provenance.json'},
            'Four complete owned auditor sources; no producer implementation imports')
    for entry in declaration['auditor_sources']:
        path = Path(__file__).parent/entry['file']
        a.bound({'path':str(path),'bytes':entry['bytes'],'sha256':entry['sha256']})
    card, inputs, controls = read(args.card), read(args.input_manifest), read(args.raw_controls)
    a.check(sha(args.card) == CARD_SHA and args.card.stat().st_size==declaration['main_card']['bytes'] and
            sha(args.input_manifest) == INPUT_SHA and args.input_manifest.stat().st_size==declaration['input_manifest']['bytes'] and card['input_manifest']['sha256']==INPUT_SHA,
            'Frozen measured main card/manifest bound by separate prospective audit card')
    a.check(card['initial_pairing']['native_coordinate_slice']==[20,40] and
            card['initial_pairing']['content20_equality_required'] is False and
            card['initial_pairing']['native_coordinates_compared']==20,
            'Initial pairing is partial native20:40; not whole40')
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
    a.check(declaration['paired_initial_reference_inputs'] == card['initial_pairing']['references'],
            'Auditor partial20:40/statistic pairing allowlist equals frozen model declaration')
    quality = read(args.quality_receipt)
    objective_contract(a, quality, 'quality receipt')
    temporal_contract(a,quality,'quality receipt')
    learning_contract(a,quality,'quality receipt')
    a.check(quality['status'] == 'passed' and quality['model'] == TAG and quality['card_sha256'] == CARD_SHA and
            quality['old_encoder_calls'] == 0 and not quality['initial_quality_reused'] and
            quality['initial_and_final_heads_new'], 'Initial/final quality execution identity')
    checkpoints = {stage: a.bound(quality[stage + '_checkpoint'], args.quality_receipt.parent) for stage in STAGES}
    a.check(checkpoints['initial'] == (args.training/'checkpoints/inference-0.pt').resolve(strict=True) and
            checkpoints['final'] == (args.training/'checkpoints/inference-2048.pt').resolve(strict=True), 'Only initial0/final2048 scored')
    proof_values = {stage: proof(a, quality[key], checkpoints[stage], updates, args.quality_receipt.parent)
                    for stage, key, updates in (('initial', 'initial_saved_proof', 0), ('final', 'post_training_proof', 2048))}
    engineering = read(a.bound(quality['engineering_proof'], args.quality_receipt.parent))
    objective_contract(a, engineering, 'engineering')
    temporal_contract(a,engineering,'engineering')
    learning_contract(a,engineering,'engineering')
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
    a.check({f'temporal_export_bin_{i}' for i in range(4)} |
            {f'temporal_position_bin_{i}' for i in range(4)} |
            {f'temporal_between_direction_{i}' for i in range(3)} <= set(engineering['actual_cuda_learning']) and
            engineering['all_four_temporal_export_row_groups_learned'] is True and
            engineering['all_four_pool_position_contrasts_learned'] is True and
            engineering['all_three_between_bin_position_directions_learned'] is True and
            engineering['final_between_bin_position_projection_changed'] is True and
            engineering['weight_decay_counted_as_data_gradient'] is False,
            'Nonvacuous actual engineering temporal row/position learning groups')
    build = read(a.bound(quality['build_admission'], args.quality_receipt.parent))
    a.check(build['status'] == 'passed' and build['card_sha256'] == CARD_SHA and len(build['binaries']) == 7,
            'Frozen build admission')
    a.check(len(build['dependency_closures'])==11 and {Path(p['path']).name for p in build['binaries']}==
            {'test_causal_patch_bottleneck_mae_v9','test_cpb_v9_variance_gradient','test_cpb_v9_temporal_pooling',
             'causal_fresh_v9_train','causal_fresh_v9_extract','causal_fresh_v9_market_read','causal_fresh_v9_synthetic_read'},
            'Seven exact admitted binaries and eleven owned translation units')
    main_manifest = a.bound(build['source_manifest'])
    a.check(sha(main_manifest) == declaration['main_source_manifest_sha256']
            and len(read(main_manifest)['files']) == declaration['main_source_files'], 'Exact predeclared source capsule')
    for entry in read(main_manifest)['files']:
        a.bound(entry, main_manifest.parent)
    for binary in build['binaries']:
        a.bound(binary)
    gradient_path = a.bound(quality['variance_gradient_proof'],args.quality_receipt.parent)
    gradient = read(gradient_path)
    objective_contract(a, gradient, 'dedicated CUDA mechanism')
    for key,value in {'status':'passed','model':'CPB-v9','gradient_policy':POLICY,'actual_CUDA':True,
                      'mean_native_variance_gradient_blocked':True,'uncertainty_native_variance_gradient_live':True,'uncertainty_projection_learning':True,'mean_path_live':True,'variance_head_learning':True,
                      'forward_variance_input_dependence':True,'manual_composite_gradient_parity':True,
                      'direct_NLL_mean_gradient_zero':True,'huber_mean_gradient_live':True,
                      'direct_Huber_uncertainty_gradient_zero':True,'shared_clip_coupling_witness':True,
                      'nonvacuous_nonzero_fixture':True}.items():
        a.check(gradient[key]==value,'Nonvacuous actual CUDA mechanism proof '+key)
    binary_path = a.bound(gradient['binary'],gradient_path.parent)
    log_path = a.bound(gradient['log'],gradient_path.parent)
    a.check(binary_path in {a.path(p['path']).resolve(strict=True) for p in build['binaries']} and
            all(token in log_path.read_text(encoding='utf-8') for token in (
                'PASS CPB-v9 huber1-mean-plus-detached-error-gaussian-variance-v1:',
                'manual composite-objective native40 gradient parity', 'synthetic optimizer steps=1')),
            'Actual admitted mechanism binary and observed passing output')
    initialization = read(a.bound(quality['fresh_initialization'], args.quality_receipt.parent))
    objective_contract(a, initialization, 'fresh constructor', require_gradient_policy=False)
    temporal_contract(a,initialization,'fresh constructor')
    learning_contract(a,initialization,'fresh constructor')
    a.check('gradient_policy' not in initialization, 'Frozen fresh-constructor JSON omits gradient_policy; actual initial proof and checkpoint bind it')
    for key, expected in {'status':'passed', 'initialization_seed':943001, 'pretrained_ancestor':'none',
            'source_updates':0, 'causal_updates':0, 'inherited_learned_parameters':0,
            'fresh_learned_parameter_values':294752, 'analytic_fixed_prior_values':3888,
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
    objective_contract(a, training, 'actual training trajectory')
    temporal_contract(a,training,'actual training trajectory')
    learning_contract(a,training,'actual training trajectory')
    a.check(training['backward_tensor']=='objective.total_loss', 'Only explicit composite objective backpropagated')
    a.check(training['gradient_policy']==POLICY and
            training['encoder_gradient_kind']==ENCODER_KIND and
            training['conditional_variance_head_trainable'] is True, 'Actual training selected composite-objective gradient policy')
    a.check(training['status'] == 'passed' and training['source_updates'] == 0 and training['new_causal_updates'] == 2048 and
            training['batch_size'] == 8 and training['typed_resume_512_exact'] and 294751 <= training['learned_tensor_value_capacity_with_recorded_movement_events'] <= 294752 and
            training['fixed_prior_parameters'] == 3888 and training['new_scaler_fits'] == training['new_normalizer_fits'] == 1,
            'Complete fresh CUDA trajectory and protected statistics/prior')
    temporal_evidence=temporal_evidence_audit(a,args,quality,build,training)
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
            for scalar, rows in (('mean_Huber_F64','row_mean_huber'),('variance_NLL_F64','row_variance_nll'),
                                 ('total_objective_F64','row_total_objective')):
                a.check(len(q[role][rows]) == n and all(math.isfinite(v) for v in q[role][rows]),
                        'Complete finite query component rows')
                a.close(mean(q[role][rows]), q[role][scalar], 'Independent query component mean')
            a.check(all(v >= 0 for v in q[role]['row_mean_huber']), 'Huber query component nonnegative')
            for huber,variance,total,reporting in zip(q[role]['row_mean_huber'],q[role]['row_variance_nll'],
                                                     q[role]['row_total_objective'],q[role]['row_nll']):
                a.close(total,huber+variance,'Independent per-row composite sum')
                a.close(variance,reporting,'Detached variance and reporting NLL same forward value')
            a.check(sum(q[role]['row_target_cells']) == q[role]['target_cells'] == n*216, 'One-patch fixed query support')
    tables, completion = read(args.report/'table_rows.json'), read(args.report/'completion.json')
    a.check(tables['kind'] == 'causal_fresh_v9_results_v1' and tables['model'] == TAG and tables['source_updates'] == 0 and
            tables['new_market_rows'] == 8 and not tables['initial_quality_reused'] and not tables['historical_encoder_quality_targets_used'] and
            not tables['overall_winner_declared'] and not tables['application_changed'], 'Fresh descriptive report identity')
    a.check(tables['market_rows'][:8] == controls['rows'] and len(tables['market_rows']) == 16,
            'Eight raw and eight newly measured market rows only')
    objective_contract(a, tables, 'detailed tables')
    temporal_contract(a,tables,'detailed tables')
    learning_contract(a,tables,'detailed tables')
    a.check(a.bound(tables['temporal_pooling_proof'])==a.bound(quality['temporal_pooling_proof']),
            'Same separately bound temporal fixture in execution and primary tables')
    for key in ('proof','binary','log'):
        a.check(a.bound(tables['temporal_mechanism_bindings'][key])==a.path(temporal_evidence[key]['path']).resolve(),
                'Exact primary table temporal binding '+key)
    a.check(a.bound(tables['temporal_learning']['receipt'])==a.path(temporal_evidence['learning_receipt']['path']).resolve(),
            'Primary tables bind exact main temporal-learning receipt')
    a.check(tables['clean_baseline']=='CPB-v3.market-01' and tables['clean_baseline_unchanged'] and
            tables['gradient_policy']==POLICY and tables['scoped_reference']=='CPB-v8.market-01' and
            completion['scoped_reference']=='CPB-v8.market-01',
            'V3 clean baseline unchanged; direct architecture/initial-DATA comparator V8')
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
        a.check(method['name'] == 'cpb_v9' and method['size'] == 40 and method['ridge_parameters'] == 4*(40+1) and
                method['neural_parameters'] == 40*16+16+16*4+4 and method['ridge_fits'] == 1 and
                method['ridge_task_fits'] == 4 and method['neural_fits'] == 3 and method['training_feature_rows'] == 112,
                'Market D40 parameters and fit budgets')
        for key, role in (('training_path','train'),('validation_path','validation')):
            p = a.path(method[key]).resolve(strict=True)
            a.check(p == (args.run/'features'/stage/'market'/('cpb_v9-'+role+'.pt')).resolve(strict=True), 'Market feature stage association')
            a.pin(p)
        a.check(tuple(r['repetition'] for r in method['repetitions']) == REPS and
                tuple(r['actual_seed'] for r in method['repetitions']) == tuple(seed(x) for x in REPS), 'Exact market D40 stream seeds')
        rows = [r for r in tables['market_rows'] if r.get('method') == label]
        a.check(len(rows) == 4 and {r['task'] for r in rows} == set(TASKS), 'Four independent tasks per new checkpoint')
        for i, task in enumerate(TASKS):
            row = next(r for r in rows if r['task'] == task)
            reps = [r['targets'][i] for r in method['repetitions']]
            a.check(row['size'] == 40 and row['checkpoint_stage'] == stage and not row['reused'], 'New native market row provenance')
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
                a.check(method['size'] == 40 and method['status'] == 'measured' and not method['inputs_train_prepared'] and
                        method['outer_train_normalizer_fits'] == 1 and len(method['repetitions']) == 3, 'Fresh native40 classification fitting')
                for j, rep in enumerate(method['repetitions']):
                    a.check(rep['id'] == 'rep-'+str(REPS[j]) and rep['status'] == 'measured' and
                            int(rep['actual_probe_seed_decimal']) == seed(REPS[j]) and rep['ridge_parameters'] == 2*(40+1) and
                            rep['neural_parameters'] == 40*16+16+16*2+2, 'TEMPO D40 parameters and exact stream seed')
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
        temporal_contract(a,extraction,'saved '+stage+' extraction')
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
                a.check(row['size'] == 40 and not row['reused'] and row['method'] == (INIT if stage == 'initial' else TAG) and row['coverage'] == 1,
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
        a.check(reader['model'] == TAG and reader['checkpoint_stage'] == stage and reader['native_width'] == 40 and
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
    for key in ('runtime_trust_admission_passed','full_coverage','clean_causal_candidate_measured'):
        a.check(tables[key] is True and completion[key] is True, 'Runtime/completeness claim ' + key)
    for value in (tables, completion):
        a.check(value['trust_protocol_admission_passed'] is False
                and value['overall_establishment_passed'] is False
                and value['mandatory_postmeasurement_trust'] == PENDING,
                'Frozen producer retains full trust pending independent audits/pairings')
    a.check(full_coverage and not completion['historical_encoder_quality_targets_used'] and not completion['overall_winner_declared'],
            'Trust/completeness establish a measured candidate; numerical advancement is separately audited')
    assets = read(args.report/'asset_hashes.json')
    for binding in assets['run_assets']:a.bound(binding)
    for key in ('training_summary','initialization','calibration','typed_resume','query_initial','query_final','engineering_proof','build_admission','reporter','variance_gradient_proof'):
        a.bound(assets[key])
    for key in ('proof','binary','log'):a.bound(assets['temporal_pooling_proof'][key])
    a.bound(assets['temporal_learning']['receipt'])
    for binding in assets['temporal_learning']['artifacts']:a.bound(binding)
    for key in ('report','tables','assets'):a.bound(completion[key])
    for stage in STAGES:
        for key in ('proof','witness'):a.bound(assets['proofs'][stage][key])
    for path in (args.card,args.input_manifest,args.raw_controls,args.quality_receipt,args.training/'training-summary.json',
                 args.training/'fresh-initialization.json',args.training/'calibration/summary.json',Path(__file__),
                 Path(__file__).with_name('archive_reader.py'),args.audit_card):
        a.pin(path)
    result = {'kind':'causal_fresh_v9_quality_independent_audit_v1','status':'passed' if not a.errors else 'failed',
        'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':TAG,'checks':a.checks,'mismatches':a.errors,
        'scope':'Independent stdlib saved-scalar and inert saved-array arithmetic/hash bindings; no executable model loading, neural/encoder forwards, training or fitting.',
        'market_new_rows':8,'raw_control_rows':8,'synthetic_cohorts_per_stage':3,'input_policies':2,'head_repetitions':3,
        'export_calls':1572,'initial_and_final_quality_new':True,'historical_quality_targets_used':False,
        'candidate_measured_rule':'Audit runtime admission, coverage, partial native20:40/whole-statistic pairing and22 numeric conditions separately; producer full trust remains pending; clean V3 baseline unchanged.',
        'saved_prediction_arithmetic':saved_prediction_arithmetic,'clean_baseline_comparison':clean_comparison,
        'query_witness_arithmetic':query_witness_arithmetic,
        'paired_initial_equality':paired_initial,
        'temporal_mechanism_and_learning_bindings':temporal_evidence,
        'report_only_recovery':recovery_evidence,
        **TEMPORAL_FIELDS,**LEARNING_FIELDS,
        'objective_policy':OBJECTIVE,'gradient_policy':POLICY,
        'postmeasurement_quality_and_pairing_audit_passed':not a.errors,
        'overall_establishment_declared':False,
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

