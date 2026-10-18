#!/usr/bin/env python3
"""Blocked, metadata-only early-mixer report renderer.

No Torch/archive/model imports or tensor reads. Source-only artificial fixtures
are allowed in the managed container; completed metadata requires an explicit
release after independent review, complete inventory and successful audit.
"""
import argparse
import ast
import builtins
import symtable
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

AUTHORIZE_COMPLETED_METADATA = False
PROTOCOL = 'early-mixer-confirmation-v1'
IMPLEMENTATION_FIT = 'early-mixer-reliability-v1/lag_sign'
TAGS = ['RPB-v7.alt-04', 'RPB-v10.alt-04']
BUDGETS = [0, 512]
CARD = 'code/evaluation/cards/early_mixer_confirmation_v1.md'
CARD_SHA = '98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea'
TIMING = [64161, 65262, 66363, 67464, 68565]
AMPLITUDE = [69666, 70767, 71868, 72969, 74070]
REPS = ['rep-2701', 'rep-2802', 'rep-2903']
METHODS = [('raw', 576, 'Raw data'), ('mask_metadata', 288, 'Mask only'),
           ('pca_only', 32, 'PCA only — no encoder'), ('untrained_late', 32, 'Untrained late encoder'),
           ('untrained_early', 32, 'Untrained early encoder'), ('native_late', 32, 'RPB-v7.alt-04'),
           ('native_early', 32, 'RPB-v10.alt-04')]
VIEWS = ['validation_intact', 'validation_deleted']
REQUIRED_METADATA = ['results/report.json', 'results/complete.json', 'artifact-integrity.json',
                     'launch-plan.json', 'recipe-plan.json', 'admission/passed.json',
                     'admission/build-and-tests.log', 'source/' + CARD]


def require(ok, why):
    if not ok:
        raise AssertionError(why)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def regular(path):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and
            not path.is_symlink() and path.stat().st_nlink == 1, 'regular, unaliased explicit metadata path')
    for ancestor in path.parents:
        require(not ancestor.is_symlink(), 'metadata ancestor redirect')
    return path


def read_json(path):
    require(path.suffix == '.json', 'JSON metadata only; no archive payloads')
    def unique(items):
        value={}
        for key,item in items:
            require(key not in value,'unique metadata JSON keys');value[key]=item
        return value
    def invalid(value):
        raise AssertionError('nonfinite JSON constant '+value)
    return json.loads(regular(path).read_text(encoding='utf-8'),object_pairs_hook=unique,parse_constant=invalid)


def finite(value, why):
    require(type(value) in (int, float) and math.isfinite(value), why)
    return value


def near(left, right, why):
    require(left is not None and right is not None and
            abs(finite(left, why)-finite(right, why)) <= 2e-9+2e-9*abs(right), why)


def interval_matches(producer, audited):
    require(producer['replicates'] == audited['replicates'] == 1000 and
            producer['confidence'] == .95 and producer['source_groups'] == audited['source_groups'],
            'same conditional interval population and recipe')
    for key in ('estimate', 'lower', 'upper'):
        if audited[key] is None:
            require(producer[key] is None, 'undefined interval remains null')
        else:
            near(producer[key], audited[key], 'saved independent interval numerical association')


def score_fraction(score):
    require(type(score['total']) is int and type(score['valid']) is int and type(score['correct']) is int and
            0 <= score['correct'] <= score['valid'] <= score['total'] and score['total'] > 0, 'honest score counts')
    coverage = Fraction(score['valid'], score['total'])
    near(score['coverage'], float(coverage), 'coverage count arithmetic')
    if not score['valid']:
        require(score['accuracy'] is None, 'zero-support accuracy remains null')
        return None, coverage
    accuracy = Fraction(score['correct'], score['valid'])
    near(score['accuracy'], float(accuracy), 'conditional accuracy count arithmetic')
    return accuracy, coverage


def metric(method, view, head):
    if method['status'] != 'measured':
        require(method['status'] == 'unsupported_fit' and not method['repetitions'], 'unsupported method retained')
        return None
    require([r['repetition'] for r in method['repetitions']] == REPS, 'all three fixed repetitions')
    values = [score_fraction(r[view][head]) for r in method['repetitions']]
    accuracy = sum(a for a, _ in values) / 3 if all(a is not None for a, _ in values) else None
    return accuracy, sum(c for _, c in values) / 3


def rational(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'decimal': float(value)}


def aggregates(audit):
    panels = []
    for task in ('lag_sign', 'amplitude'):
        for view in VIEWS:
            records = []
            for name, size, label in METHODS:
                rows = []
                for cohort in audit['cohorts']:
                    data = next(t for t in cohort['tasks'] if t['task'] == task)
                    method = next(m for m in data['methods'] if m['method'] == name)
                    linear, neural = metric(method, view, 'ridge'), metric(method, view, 'tiny_secondary')
                    require(linear is None or neural is None or linear[1] == neural[1], 'common head coverage')
                    rows.append({'timing_master':cohort['timing_master'], 'data_master':data['data_master'],
                                 'status':'measured' if linear and neural and linear[0] is not None and neural[0] is not None else 'unsupported',
                                 'linear':rational(linear[0]) if linear and linear[0] is not None else None,
                                 'neural':rational(neural[0]) if neural and neural[0] is not None else None,
                                 'coverage':rational(linear[1]) if linear else None})
                item = {'method':name, 'display_label':label, 'size':size, 'per_master':rows,
                        'status':'measured' if all(r['status']=='measured' for r in rows) else 'unsupported'}
                for key in ('linear', 'neural', 'coverage'):
                    defined = all(r[key] is not None for r in rows)
                    values = [Fraction(r[key]['numerator'],r[key]['denominator']) for r in rows] if defined else []
                    item[key] = {'mean':rational(sum(values)/5), 'min':rational(min(values)), 'max':rational(max(values))} if values else None
                item['undefined_cohorts'] = {key:sum(r[key] is None for r in rows) for key in ('linear','neural','coverage')}
                records.append(item)
            effects = []
            for cohort in audit['cohorts']:
                data = next(t for t in cohort['tasks'] if t['task']==task)
                pairs = [p for p in data['pairs'] if p['view']==view]
                require(len(pairs)==3, 'all common-pop paired repetitions')
                effect = {'timing_master':cohort['timing_master'], 'data_master':data['data_master'], 'records':copy.deepcopy(pairs)}
                for head in ('ridge','tiny_secondary'):
                    effect[head+'_mean'] = math.fsum(p[head]['estimate'] for p in pairs)/3 if all(p['status']=='measured' and p[head]['estimate'] is not None for p in pairs) else None
                effects.append(effect)
            panels.append({'task':task,'budget':512,'view':view,'methods':records,'paired_effects':effects})
    return panels


def validate_pair(report, audit, complete):
    require(report['protocol'] == audit['protocol'] == complete['protocol'] == PROTOCOL and
            audit['status'] == 'passed' and complete['status'] == 'complete', 'complete measurement and passed independent audit')
    require(len(report['cohorts']) == len(audit['cohorts']) == 5, 'all five independent pairs')
    require(audit['source']['source_fingerprint'] == report['source_fingerprint'], 'same evaluated source')
    measured_pipelines = 0
    for p, a, master, amplitude in zip(report['cohorts'], audit['cohorts'], TIMING, AMPLITUDE):
        require(p['timing_master'] == a['timing_master'] == master and
                p['amplitude_data_master'] == a['amplitude_data_master'] == amplitude, 'fixed cohort association')
        metadata_matches(p['costs'], a['costs'])
        require(len(p['encoder_points']) == len(a['encoders']) == 2 and
                len(p['tasks']) == len(a['tasks']) == 2, 'complete ordered architecture/task matrix')
        for enc_p, enc_a, tag, placement in zip(p['encoder_points'], a['encoders'], ['RPB-v7.alt-04', 'RPB-v10.alt-04'], [0, 1]):
            require(enc_p['model_tag'] == enc_a['tag'] == tag and enc_p['placement'] == enc_a['placement'] == placement,
                    'same-run architecture/tag identity')
            prog = enc_p['encoder_progress']
            require(prog['completed'] == prog['attempted'] == enc_a['completed'] == enc_a['attempted'] == 512 and
                    prog['sampled_rows'] == enc_a['sampled_rows'] == 4096 and enc_a['role']==('early' if placement else 'late') and enc_a['budget']==512 and
                    prog['parameter_count']==prog['cuda_parameter_count']==225805 and len(prog['losses'])==512 and
                    prog['training_device'] in ('cuda','cuda:0') and all(prog[k] is True for k in ('last_input_cuda','last_loss_cuda','finite_gradients','weights_changed')), 'actual unskipped CUDA encoder prefix')
            for attempt,row in enumerate(prog['losses'],1):
                require(len(row)==5 and row[0]==row[1]==attempt and type(row[2]) is int and row[2]>0 and finite(row[3],'sampled Huber')>=0 and finite(row[4],'gradient norm')>=0, 'complete ordered objective trace')
            near(prog['training_seconds'], enc_a['training_seconds'], 'same cumulative training timer')
            for split in ('training', 'validation'):
                query = enc_a['fixed_query'][split]
                declared = enc_p[split + '_reconstruction']
                for key in ('standardized_mae', 'standardized_huber'):
                    if query[key] is None:
                        require(declared[key] is None, 'unsupported fixed query remains null')
                    else:
                        near(query[key], declared[key], 'same-run fixed query ' + key)
                for key in ('valid_examples', 'total_examples', 'valid_target_cells', 'requested_observed_target_cells'):
                    require(query[key] == declared[key], 'fixed query support association')
        for pt, at, task in zip(p['tasks'], a['tasks'], ['lag_sign', 'amplitude']):
            require(pt['task'] == at['task'] == task and pt['data_master'] == at['data_master'], 'separate task roles')
            readouts = pt['readouts']
            metadata_matches(readouts['fit_counts'], at['fit_counts'])
            metadata_matches(readouts['pairs'], at['pairs'])
            require([(v['id'],v['repetition'],v['view']) for v in at['pairs']] == [('native_early_minus_native_late',rep,view) for rep in REPS for view in VIEWS], 'ordered paired populations')
            task_pipelines = 3 * sum(m['status'] == 'measured' for m in at['methods'])
            require(at['fit_counts']['ridge_fits'] == at['fit_counts']['tiny_fits'] == task_pipelines,
                    'actual supported task/head fit counts')
            measured_pipelines += task_pipelines
            require([x['method'] for x in readouts['methods']] == [x[0] for x in METHODS] ==
                    [x['method'] for x in at['methods']], 'seven declared methods, including unsupported controls')
            for pm, am in zip(readouts['methods'], at['methods']):
                require(pm['size'] == am['size'] and pm['status'] == am['status'], 'method fit status and native size')
                require([r['id'] for r in pm['repetitions']] == REPS, 'all three producer head repetitions')
                if am['status'] != 'measured':
                    require(not am['repetitions'], 'unsupported audit never invents predictions')
                    continue
                require([r['repetition'] for r in am['repetitions']] == REPS, 'all three independent audit head repetitions')
                for pr, ar in zip(pm['repetitions'], am['repetitions']):
                    require(pr['id'] == ar['repetition'] and pr['actual_probe_seed_decimal'] == ar['actual_probe_seed_decimal'], 'head seed binding')
                    for view in ('training', *VIEWS):
                        for head in ('ridge', 'tiny_secondary'):
                            source_score, audit_score = pr[view][head], ar[view][head]
                            for key in ('total', 'valid', 'correct'):
                                require(source_score[key] == audit_score[key], 'saved count/score exact JSON association')
                            near(source_score['coverage'], audit_score['coverage'], 'same coverage arithmetic')
                            if audit_score['accuracy'] is None:
                                require(source_score['accuracy'] is None, 'unsupported accuracy remains null')
                            else:
                                near(source_score['accuracy'], audit_score['accuracy'], 'same saved conditional accuracy')
                            if view != 'training':
                                field = 'ridge_grouped_interval' if head == 'ridge' else 'tiny_grouped_interval'
                                interval_matches(pr[view][field], audit_score['grouped_interval'])
    require(complete['head_pipelines'] == audit['counts']['head_pipelines'] and
            complete['individual_heads'] == audit['counts']['individual_heads'] == 2 * complete['head_pipelines'] and
            measured_pipelines == complete['head_pipelines'], 'actual fitted budgets')
    require(complete['planned_pipelines'] == 210 and complete['planned_heads'] == 420 and
            complete['encoder_trajectories'] == 10 and complete['sampled_rows'] == 40960, 'fixed planned trajectory/head budgets')
    for obj in (report, complete):
        require(not obj['testing_accessed'] and not obj['stress_accessed'] and not obj['promotion'], 'no held-out TEST/stress or promotion')
    require(report['selection'] is False and complete['selection'] is False, 'no result-dependent selection')
    for key,value in {'encoder_updates_each':512,'attempt_limit':1024,'retained_points':20,'skipped_attempts':0,'unique_quality_native_exports':120,'initial_counterpart_exports':0,'full_native_export_calls':120,'query_evaluation_calls':20,'necessary_query_forwards':80,'driver_raw_outer_fits':10,'separate_initial_controls':2,'initial_shared_state_exact_before_training_and_heads':True,'extra_decoder_calibration_updates':0}.items():
        require(complete[key]==value,'explicit completion scope '+key)
    require(complete['helper_outer_train_fits']==audit['counts']['helper_outer_fits'] and 0<=complete['helper_outer_train_fits']<=50 and complete['decoder_update_scope']=='joint reconstruction during encoder updates; no extra decoder-only calibration','actual raw/head fitting and decoder scope')


def descriptive_guards(quality,training):
    flags,numeric={},{}
    for panel in quality:
        if panel['task']!='lag_sign': continue
        late=next(m for m in panel['methods'] if m['method']=='native_late')
        early=next(m for m in panel['methods'] if m['method']=='native_early')
        defined=late['status']==early['status']=='measured'
        for metric,agg in [('ridge_mean','mean'),('ridge_worst','min')]:
            a=late['linear'][agg]['decimal'] if defined else None
            b=early['linear'][agg]['decimal'] if defined else None
            flags[panel['view']+'_'+metric+'_no_worse']=defined and b>=a
            numeric[panel['view']+'_'+metric]={'late':a,'early':b}
        flags[panel['view']+'_equal_per_master_coverage']=defined and all(a['coverage']==b['coverage'] for a,b in zip(late['per_master'],early['per_master']))
    for split,key in [('training','train_standardized_mae'),('validation','validation_standardized_mae')]:
        a,b=[t[key]['mean'] if t[key] else None for t in training]
        flags[split+'_original_query_mae_mean_no_worse']=a is not None and b is not None and b<=a
        numeric[split+'_original_query_mae']={'late':a,'early':b}
    return {'guards':flags,'joint_direction_passed':all(flags.values()),'numeric':numeric,'automatic_promotion':False}


def summarize(report, audit, complete, bindings):
    validate_pair(report, audit, complete)
    training = []
    for placement, tag in [(0, 'RPB-v7.alt-04'), (1, 'RPB-v10.alt-04')]:
        rows = []
        for cohort in audit['cohorts']:
            enc = cohort['encoders'][placement]
            rows.append({'timing_master': cohort['timing_master'], 'updates': 512,
                         'train_standardized_mae': enc['fixed_query']['training']['standardized_mae'],
                         'validation_standardized_mae': enc['fixed_query']['validation']['standardized_mae'],
                         'GPU_update_loop_seconds': enc['training_seconds'],
                         'context_counts_requested_actual_restored': enc['context_counts']})
        entry = {'tag': tag, 'placement': placement, 'updates': 512, 'per_master': rows}
        for key in ('train_standardized_mae', 'validation_standardized_mae', 'GPU_update_loop_seconds'):
            values = [r[key] for r in rows]
            if any(v is None for v in values):
                entry[key] = None
            else:
                for value in values:
                    finite(value, 'finite training summary')
                entry[key] = {'mean': math.fsum(values) / 5, 'min': min(values), 'max': max(values)}
        training.append(entry)
    costs = {key: {'sum': math.fsum(c['costs'][key] for c in audit['cohorts']),
                   'mean': math.fsum(c['costs'][key] for c in audit['cohorts']) / 5}
             for key in audit['cohorts'][0]['costs']}
    retained_cohorts = copy.deepcopy(audit['cohorts'])
    for produced, retained in zip(report['cohorts'], retained_cohorts):
        for point, encoder in zip(produced['encoder_points'], retained['encoders']):
            encoder['encoder_progress'] = copy.deepcopy(point['encoder_progress'])
    quality=aggregates(audit);guards=descriptive_guards(quality,training)
    require(guards['guards']==audit['decision']['guards'] and guards['joint_direction_passed']==audit['decision']['joint_direction_passed'],'independent descriptive guards agree')
    return {'schema_version': 1, 'protocol': PROTOCOL, 'status': 'independently_audited',
            'promotion': False, 'selection': False, 'bindings': bindings,
            'recipe': {'encoder_runs': 10, 'parameter_count_each': 225805, 'native_width': 32,
                       'encoder_updates_each': 512, 'decoder_calibration_updates': 0, 'batch_size': 8,
                       'train_rows_per_task_cohort': 256, 'validation_rows_per_task_cohort': 128,
                       'TRAIN_source_groups': 128, 'VALIDATION_source_groups': 64, 'test_rows': 0,
                       'head_repetitions': [2701, 2802, 2903], 'ridge_penalty': 1,
                       'tiny_hidden': 16, 'tiny_Adam_learning_rate': .01, 'tiny_updates': 100,
                       'native_post_PCA': False, 'extra_validation_coordinate_deletion': .30,
                       'training_policy_id': 'rpb-training-context-deletion-015-v1',
                       'encoder_fit_namespace': PROTOCOL+'/lag_sign',
                       'implementation_fit_namespace': IMPLEMENTATION_FIT,
                       'checkpoint_roles_per_point': 5, 'confirmation_suffix': '.confirmation.pt',
                       'attempt_limit': 1024, 'log_every': 1, 'shared_driver_raw_outer_fits': 10,
                       'maximum_helper_outer_fits': 50, 'extra_decoder_calibration_updates': 0,
                       'late_architecture_id': 'aligned-mixer-after-temporal-v1',
                       'early_architecture_id': 'aligned-mixer-before-temporal-v1'},
            'counts': copy.deepcopy(audit['counts']), 'quality': quality, 'training': training, 'recomputed_guards':guards,
            'decision': copy.deepcopy(audit['decision']), 'completion': copy.deepcopy(complete),
            'mixed_wall_costs': costs, 'cohorts': retained_cohorts,
            'conditional_interval_recipe': {'replicates': 1000, 'confidence': .95,
                                            'scope': 'source groups within master, fixed checkpoint and head'},
            'independent_audit': {'checks': audit['checks'], 'archive_decodes': audit['archive_decodes'],
                                  'elapsed_seconds': audit['elapsed_seconds'], 'limits': copy.deepcopy(audit['limits'])},
            'interpretation_limits': ['Within-master source intervals condition on fixed checkpoints and heads; no across-encoder interval.',
                                     'Equal parameter counts do not imply equal compute; early mixing retains two temporal passes.',
                                     'No result proves an irreversible information loss or causal internal mechanism.',
                                     'Amplitude heads fit separate new TRAIN; timing encoder/scaler remain frozen.',
                                     'Unsupported fits and zero support stay explicit; no seed subset or averaged task score.',
                                     'No architecture promotion or change to preserved RPB-v7 instances.']}


def pct(value):
    return 'N/A' if value is None else f'{100 * value:.4f}'


def number(value, places=9):
    return 'N/A' if value is None else f'{value:.{places}f}'


def quality_value(method, key):
    return None if method[key] is None else method[key]['mean']['decimal']


def markdown(summary):
    lines = ['# Early mixer architecture confirmation', '',
             'Protocol: `early-mixer-confirmation-v1`. Completed known-VALIDATION comparison with an independent saved-arithmetic audit. No TEST or stress was generated or read.', '',
             'RPB-v7.alt-04 is a fresh late-mixer group using the preserved coordinate15 recipe. RPB-v10.alt-04 moves aligned W64 channel mixing before temporal encoding. Both have 225,805 parameters and serve native32; the early model retains a separate independent local temporal pass. Equal parameters do not mean equal compute.', '',
             'Five paired timing masters: 64161/65262/66363/67464/68565. Each trajectory used 256 TRAIN rows from 128 source groups, 128 VALIDATION rows from 64 disjoint groups, batch 8 and 512 unskipped CUDA AdamW updates. There are ten new encoder trajectories, not thirty runs from repeated heads. Each saw 4,096 sampled rows (16 equivalent presentations, not guaranteed epochs).', '',
             'Amplitude data masters 69666/70767/71868/72969/74070 provide separate TRAIN/VALIDATION head data. They never enter encoder or scaler fitting. Three fixed head repetitions use ridge penalty 1 and tanh 16 / Adam 0.01 / 100 updates. Native heads receive exact 32 without PCA; raw 576 includes 288 scaled values and 288 masks. PCA 32 is fitted only to raw TRAIN; mask-only has 288 inputs. Head parameter counts are 66/562 at 32, 578/4,658 at 288 and 1,154/9,266 at 576 (linear/neural).', '',
             'External cohort fit identity is early-mixer-confirmation-v1/lag_sign; the unchanged training implementation receives an explicit copy labelled early-mixer-reliability-v1/lag_sign. The old checkpoint and snapshot metadata keep that truthful implementation identity. A new fifth .confirmation.pt companion and separate confirmation snapshot audit bind the new cohort, original TRAIN/source order/scaler, counters, source scopes and four original parent bytes. No historical quality payload is an input. The new binding is admission metadata and changes no numerical training or serving function.', '',
             'Both untrained controls are retained because equal initial weights do not imply equal outputs for different computation order. Each fitted head is reused intact and with one fixed extra 30% coordinate deletion view. Coverage is valid / declared 128 VALIDATION rows; accuracy is conditional on that support. All fixed head and paired source intervals are retained in the durable JSON.', '']
    for panel in summary['quality']:
        task = 'Timing (lag sign)' if panel['task'] == 'lag_sign' else 'Amplitude transfer'
        view = 'intact VALIDATION' if panel['view'] == 'validation_intact' else 'extra 30% coordinate deletion VALIDATION'
        lines += [f'## {task}: {view}', '', '| Method | Size | Linear head % | Neural head % | Coverage % |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for method in panel['methods']:
            lines.append(f"| {method['display_label']} | {method['size']} | {pct(quality_value(method,'linear'))} | {pct(quality_value(method,'neural'))} | {pct(quality_value(method,'coverage'))} |")
        lines += ['', 'Equal-master means; neural repetitions are averaged within each master first. Full method/cohort ranges and unsupported statuses are retained in the linked JSON.', '',
                  '| Timing master / data master | RPB-v7.alt-04 linear / neural % | RPB-v10.alt-04 linear / neural % | Paired linear effect pp | Coverage late / early % |',
                  '| --- | ---: | ---: | ---: | ---: |']
        late = next(m for m in panel['methods'] if m['method'] == 'native_late')
        early = next(m for m in panel['methods'] if m['method'] == 'native_early')
        for a, b, effect in zip(late['per_master'], early['per_master'], panel['paired_effects']):
            if a['status'] != 'measured' or b['status'] != 'measured':
                lines.append(f"| {a['timing_master']} / {a['data_master']} | unsupported | unsupported | N/A | N/A |")
                continue
            delta = effect['ridge_mean']
            delta_text = 'N/A' if delta is None else f'{100*delta:+.4f}'
            lines.append(f"| {a['timing_master']} / {a['data_master']} | {pct(a['linear']['decimal'])} / {pct(a['neural']['decimal'])} | {pct(b['linear']['decimal'])} / {pct(b['neural']['decimal'])} | {delta_text} | {pct(a['coverage']['decimal'])} / {pct(b['coverage']['decimal'])} |")
        lines += ['', 'These score differences are descriptive across retrainings. Saved paired effects use the common supported population and conditional within-master 1,000 / 95% source bootstrap; interval bounds are not averaged into an encoder confidence interval.', '']
    lines += ['## Training and reconstruction', '',
              'Fixed ordinary whole-patch query standardized MAE uses the same original Q, targets, support and TRAIN scaler in each pair. Inference adds no training context deletion. Sampled optimization Huber and its complete 512-step traces are separate evidence; amplitude reconstruction was not measured.', '',
              '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |',
              '| --- | ---: | ---: | ---: | ---: |']
    for entry in summary['training']:
        lines.append(f"| {entry['tag']} | 512 | {number(entry['train_standardized_mae']['mean'] if entry['train_standardized_mae'] else None)} | {number(entry['validation_standardized_mae']['mean'] if entry['validation_standardized_mae'] else None)} | {number(entry['GPU_update_loop_seconds']['mean'] if entry['GPU_update_loop_seconds'] else None,6)} |")
    lines += ['', '| Timing master | RPB-v7.alt-04 TRAIN / VALIDATION MAE | RPB-v10.alt-04 TRAIN / VALIDATION MAE | GPU update seconds late / early |',
              '| --- | ---: | ---: | ---: |']
    for a, b in zip(summary['training'][0]['per_master'], summary['training'][1]['per_master']):
        lines.append(f"| {a['timing_master']} | {number(a['train_standardized_mae'])} / {number(a['validation_standardized_mae'])} | {number(b['train_standardized_mae'])} / {number(b['validation_standardized_mae'])} | {number(a['GPU_update_loop_seconds'],6)} / {number(b['GPU_update_loop_seconds'],6)} |")
    lines += ['', 'Synchronized cumulative update-loop times exclude head fitting, extraction, reconstruction queries and checkpoint writing. They are descriptive single-run measurements. Additional stages below include CPU transfer, parent verification, file hashing and I/O; pure CUDA kernel time is unmeasured.', '',
              '| Mixed wall-time scope | Total seconds | Mean per paired cohort seconds |', '| --- | ---: | ---: |']
    for key, value in summary['mixed_wall_costs'].items():
        lines.append(f"| `{key}` | {value['sum']:.6f} | {value['mean']:.6f} |")
    counts = summary['counts']; qa = summary['independent_audit']
    lines += ['', '## Descriptive guard checks', '', '| Check | Passed |', '| --- | --- |']
    for name,passed in summary['decision']['guards'].items():
        lines.append(f'| {name} | {str(passed).lower()} |')
    lines += ['', 'These guard flags describe timing mean/worst linear scores, equal per-master coverage and original-query mean MAE. They do not authorize automatic promotion or rescue by secondary amplitude/neural scores.', '', '## Evidence and limits', '',
              f"Actual supported budgets: {counts['head_pipelines']} pipelines / {counts['individual_heads']} heads (planned 210/420). The five paired cohorts retain 20 point-0/512 checkpoints, both initial controls, all ten complete traces and all original-query supports. {counts['native_full_export_calls']} full native export callbacks and {counts['fixed_query_writer_calls']} query-writer calls produced {counts['fixed_query_banks']} patch-query banks. Joint reconstruction updates still train the decoder; no extra decoder-only calibration occurred.", '',
              f"Independent audit PASS: {qa['checks']:,} checks, {qa['archive_decodes']:,} CPU archive decodes, {qa['elapsed_seconds']:.6f} seconds. It independently replays saved maps/logits/support/query reductions and source effects; it runs no model, CUDA forward, autodiff, optimizer, head fit or PCA/SVD fit. Ordinary CUDA checkpoint bodies remain byte-bound, with semantics established by source/engineering admission and CPU companions. Original sampler/Torch streams are source/admission-bound, not GPU reenacted.", '',
              'No numeric automatic promotion rule was declared. Assess timing intact/deleted linear accuracy, worst-cohort accuracy, fixed TRAIN/VALIDATION MAE, neural regressions and amplitude separately before any next change. This report does not replace original RPB-v7 weights or claim a causal internal mechanism.', '',
              '[Frozen prospective card](../../evaluation/cards/early_mixer_confirmation_v1.md). [Durable full summary](../../../doc/results/early_mixer_confirmation_v1.json).', '', 'Metadata/source bindings:', '']
    for key, value in sorted(summary['bindings'].items()):
        if isinstance(value, str):
            lines.append(f'- `{key}`: `{value}`')
    return '\n'.join(lines) + '\n'


def metadata_matches(left, right):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), 'exact metadata dictionary fields')
        for key in left:
            metadata_matches(left[key], right[key])
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), 'complete ordered metadata list')
        for a, b in zip(left, right):
            metadata_matches(a, b)
    elif type(left) is float:
        near(left, right, 'saved arithmetic numerical association')
    else:
        require(type(left) is type(right) and left == right, 'exact metadata identity/status/null/count')


def binding_paths(binding):
    require(set(binding)=={'protocol','root_approved_completed_metadata','capsule','audit_validation','audit_reader',
                          'source_fingerprint','expected_sha256','audit_checks','audit_archive_decodes'} and
            binding['protocol'] == PROTOCOL and binding['root_approved_completed_metadata'] is True,
            'explicit root-approved completed metadata record')
    require(all(type(binding[k]) is int and binding[k]>0 for k in ('audit_checks','audit_archive_decodes')), 'actual successful audit counts')
    capsule = Path(binding['capsule']); audit_path = Path(binding['audit_validation']); reader = Path(binding['audit_reader'])
    root = Path('/embedding/output/runs/rpb-early-mixer-confirmation')
    require(capsule.is_absolute() and capsule.parent == root and capsule.name.startswith('early-mixer-confirmation-'),
            'closed protocol capsule path')
    require(audit_path.is_absolute() and audit_path.is_relative_to(root/'audit-tools') and audit_path.name == 'validation.json' and
            reader.is_absolute() and reader.is_relative_to(root/'audit-tools') and reader.name == 'validate_early_mixer_confirmation.py',
            'explicit successful audit JSON and reader source paths')
    paths = [capsule / relative for relative in REQUIRED_METADATA] + [audit_path, reader]
    require(len(paths) == len(set(paths)) == 10 and set(binding['expected_sha256']) == set(str(p) for p in paths),
            'complete exact ten-role metadata/source matrix')
    require(all(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)
                for h in binding['expected_sha256'].values()), 'closed SHA256 values')
    require(isinstance(binding['source_fingerprint'],str) and len(binding['source_fingerprint'])==64 and
            all(c in '0123456789abcdef' for c in binding['source_fingerprint']), 'enclosing source SHA identity')
    return capsule,audit_path,reader,paths


def load_completed(binding_path):
    require(AUTHORIZE_COMPLETED_METADATA, 'blocked before any quality metadata access')
    binding = read_json(binding_path)
    capsule,audit_path,reader,paths = binding_paths(binding)
    # Admit the whole matrix, including alias/inode bounds, before hashing any body.
    for path in paths:
        regular(path)
    require(len({(p.stat().st_dev,p.stat().st_ino) for p in paths}) == 10, 'ten distinct regular metadata/source objects')
    for path in paths:
        require(sha_bytes(path.read_bytes()) == binding['expected_sha256'][str(path)], 'exact root-approved metadata/source bytes')
    report = read_json(capsule/'results/report.json'); complete = read_json(capsule/'results/complete.json'); audit = read_json(audit_path)
    inventory = read_json(capsule/'artifact-integrity.json')
    require(len(inventory['files']) == len({r['path'] for r in inventory['files']}), 'unique inventory paths')
    index = {r['path']:r for r in inventory['files']}
    for relative in REQUIRED_METADATA:
        if relative != 'artifact-integrity.json':
            require(index[relative]['sha256'] == binding['expected_sha256'][str(capsule/relative)] and
                    index[relative]['bytes'] == (capsule/relative).stat().st_size, 'completed inventory metadata association')
    source = audit['source']; launch = read_json(capsule/'launch-plan.json'); admission = read_json(capsule/'admission/passed.json')
    require(audit['status']=='passed' and complete['status']=='complete' and
            audit['checks']==binding['audit_checks'] and audit['archive_decodes']==binding['audit_archive_decodes'], 'actual successful audit work binding')
    require(source['source_fingerprint']==launch['source_fingerprint']==admission['source_fingerprint']==binding['source_fingerprint'], 'same enclosing source')
    require(source['human_card_sha256']==CARD_SHA==sha_bytes((capsule/'source'/CARD).read_bytes()), 'frozen prospective card')
    require(source['reader_sha256']==binding['expected_sha256'][str(reader)] and
            source['inventory_sha256']==binding['expected_sha256'][str(capsule/'artifact-integrity.json')], 'exact successful reader/inventory')
    require(source['admission_sha256']==launch['admission_passed_sha256']==binding['expected_sha256'][str(capsule/'admission/passed.json')] and
            source['admission_log_sha256']==launch['admission_log_sha256']==binding['expected_sha256'][str(capsule/'admission/build-and-tests.log')], 'actual admission binding')
    require(admission['status']=='passed' and admission['quality_generated'] is False, 'pre-quality actual CUDA admission')
    bindings = {'capsule':str(capsule),'audit_validation':str(audit_path),'audit_reader':str(reader),
                'source_fingerprint':binding['source_fingerprint'],'card_sha256':CARD_SHA,
                'audit_sha256':binding['expected_sha256'][str(audit_path)],'inventory_sha256':source['inventory_sha256'],
                'reader_sha256':source['reader_sha256'],'admission_sha256':source['admission_sha256'],
                'admission_log_sha256':source['admission_log_sha256'],'renderer_sha256':sha_bytes(Path(__file__).read_bytes()),
                'root_completed_metadata_record':str(binding_path),'root_completed_metadata_record_sha256':sha_bytes(binding_path.read_bytes()),
                'captured_compile_scopes':source['actual_compile_scopes'],'metadata_read_paths':[str(p) for p in paths],
                'checkpoint_artifacts':[copy.deepcopy(index[relative]) for master in TIMING for role in ('late','early')
                    for point in BUDGETS for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.confirmation.pt')
                    for relative in [f'results/seed-{master}-lag_sign/{role}/point-{point}/checkpoint.pt{suffix}']]}
    return summarize(report,audit,complete,bindings)


def artificial_metadata():
    # Synthetic JSON only; no filesystem or quality inputs needed.
    scores = {'total': 128, 'valid': 128, 'correct': 100, 'accuracy': 100/128, 'coverage': 1.}
    source = 'a' * 64
    report = {'protocol': PROTOCOL, 'source_fingerprint': source, 'cohorts': [],
              'testing_accessed': False, 'stress_accessed': False, 'selection': False, 'promotion': False}
    fixture = {'protocol': PROTOCOL, 'status': 'passed', 'source': {'source_fingerprint': source},
               'cohorts': [], 'checks': 123, 'archive_decodes': 7, 'elapsed_seconds': .2,
               'counts': {'head_pipelines': 210, 'individual_heads': 420, 'native_full_export_calls': 120,
                          'fixed_query_writer_calls': 20, 'fixed_query_banks': 80}, 'limits': {'saved_arithmetic_only': True}}
    complete = {'protocol': PROTOCOL, 'status': 'complete', 'head_pipelines': 210, 'individual_heads': 420,
                'planned_pipelines': 210, 'planned_heads': 420, 'encoder_trajectories': 10, 'sampled_rows': 40960,
                'testing_accessed': False, 'stress_accessed': False, 'promotion': False}
    for timing, amplitude in zip(TIMING, AMPLITUDE):
        costs = {key: .1 for key in ['generation_and_observation_io_seconds', 'binding_and_checkpoint_io_seconds',
                 'CUDA_query_transfer_verification_io_seconds', 'CUDA_native_transfer_verification_seconds',
                 'CPU_baseline_preparation_io_seconds', 'CPU_head_bootstrap_io_seconds']}
        cohort = {'timing_master': timing, 'amplitude_data_master': amplitude, 'tasks': [], 'encoders': [], 'costs': costs}
        produced = {'timing_master': timing, 'amplitude_data_master': amplitude, 'tasks': [], 'encoder_points': [], 'costs': dict(costs)}
        for placement, tag in [(0, 'RPB-v7.alt-04'), (1, 'RPB-v10.alt-04')]:
            query = {'standardized_mae': .1, 'standardized_huber': .05, 'valid_examples': 128,
                     'total_examples': 128, 'valid_target_cells': 1000, 'requested_observed_target_cells': 1000}
            cohort['encoders'].append({'tag': tag, 'placement': placement,'role':'early' if placement else 'late','budget':512, 'completed': 512, 'attempted': 512,
                                      'sampled_rows': 4096, 'training_seconds': 10., 'context_counts': [30, 28, 2],
                                      'fixed_query': {'training': dict(query), 'validation': dict(query)}})
            produced['encoder_points'].append({'model_tag': tag, 'placement': placement,
                 'encoder_progress': {'completed': 512, 'attempted': 512, 'sampled_rows': 4096, 'training_seconds': 10.,'parameter_count':225805,'cuda_parameter_count':225805,'training_device':'cuda','last_input_cuda':True,'last_loss_cuda':True,'finite_gradients':True,'weights_changed':True,'losses':[[i,i,120,.05,.01] for i in range(1,513)]},
                 'training_reconstruction': dict(query), 'validation_reconstruction': dict(query)})
        for task, master in [('lag_sign', timing), ('amplitude', amplitude)]:
            methods, produced_methods = [], []
            for name, width, _ in METHODS:
                reps, produced_reps = [], []
                for rep in REPS:
                    item = {'repetition': rep, 'actual_probe_seed_decimal': '123'}
                    produced_item = {'id': rep, 'actual_probe_seed_decimal': '123'}
                    for view in ('training', *VIEWS):
                        audited_scores = {head: dict(scores) for head in ('ridge', 'tiny_secondary')}
                        produced_scores = {head: dict(scores) for head in ('ridge', 'tiny_secondary')}
                        if view != 'training':
                            for head, key in [('ridge', 'ridge_grouped_interval'), ('tiny_secondary', 'tiny_grouped_interval')]:
                                interval = {'estimate': 100/128, 'lower': .7, 'upper': .9, 'source_groups': 64, 'replicates': 1000}
                                audited_scores[head]['grouped_interval'] = interval
                                produced_scores[key] = {**interval, 'confidence': .95}
                        item[view] = audited_scores; produced_item[view] = produced_scores
                    reps.append(item); produced_reps.append(produced_item)
                methods.append({'method': name, 'size': width, 'status': 'measured', 'repetitions': reps})
                produced_methods.append({'method': name, 'size': width, 'status': 'measured', 'repetitions': produced_reps})
            pairs = [{'id': 'native_early_minus_native_late', 'repetition': rep, 'view': view, 'status': 'measured',
                      **{head: {'estimate': 0., 'lower': -.1, 'upper': .1, 'source_groups': 64,
                                'replicates': 1000, 'confidence': .95} for head in ('ridge', 'tiny_secondary')}}
                     for rep in REPS for view in VIEWS]
            counts = {'ridge_fits': 21, 'tiny_fits': 21}
            cohort['tasks'].append({'task': task, 'data_master': master, 'methods': methods, 'pairs': copy.deepcopy(pairs), 'fit_counts': dict(counts)})
            produced['tasks'].append({'task': task, 'data_master': master,
                'readouts': {'methods': produced_methods, 'pairs': pairs, 'fit_counts': dict(counts)}})
        fixture['cohorts'].append(cohort)
        report['cohorts'].append(produced)
    complete.update(encoder_updates_each=512,attempt_limit=1024,retained_points=20,skipped_attempts=0,unique_quality_native_exports=120,initial_counterpart_exports=0,full_native_export_calls=120,query_evaluation_calls=20,necessary_query_forwards=80,driver_raw_outer_fits=10,separate_initial_controls=2,initial_shared_state_exact_before_training_and_heads=True,extra_decoder_calibration_updates=0,helper_outer_train_fits=50,selection=False,decoder_update_scope='joint reconstruction during encoder updates; no extra decoder-only calibration')
    fixture['counts']['helper_outer_fits']=50
    training=[{'train_standardized_mae':{'mean':.1},'validation_standardized_mae':{'mean':.1}} for _ in TAGS]
    fixture['decision']=descriptive_guards(aggregates(fixture),training)
    return report, fixture, complete


def self_test():
    scores = {'total': 128, 'valid': 128, 'correct': 100, 'accuracy': 100/128, 'coverage': 1.}
    require(score_fraction(scores)[0] == Fraction(25, 32), 'exact rational count fixture')
    zero = {'total': 128, 'valid': 0, 'correct': 0, 'accuracy': None, 'coverage': 0.}
    require(score_fraction(zero)[0] is None, 'unsupported support null fixture')
    report, fixture, complete = artificial_metadata()
    validate_pair(report, fixture, complete)
    # Independent arithmetic legitimately differs in the final floating bit.
    fixture['cohorts'][0]['tasks'][0]['methods'][0]['repetitions'][0]['validation_intact']['ridge']['grouped_interval']['lower'] += 1e-12
    validate_pair(report, fixture, complete)
    result = aggregates(fixture)
    require(len(result) == 4 and all(len(x['methods']) == 7 for x in result), 'two tasks/two views/seven methods')
    require(result[0]['methods'][0]['linear']['mean']['decimal'] == 100/128, 'equal-master rational mean')
    fixture['cohorts'][2]['tasks'][0]['methods'][2].update(status='unsupported_fit', repetitions=[])
    producer_method = report['cohorts'][2]['tasks'][0]['readouts']['methods'][2]
    producer_method.update(status='unsupported_fit', repetitions=[{'id': rep, 'status': 'unsupported_fit'} for rep in REPS])
    for task in [fixture['cohorts'][2]['tasks'][0], report['cohorts'][2]['tasks'][0]['readouts']]:
        task['fit_counts'] = {'ridge_fits': 18, 'tiny_fits': 18}
    fixture['counts']['head_pipelines'] = complete['head_pipelines'] = 207
    fixture['counts']['individual_heads'] = complete['individual_heads'] = 414
    validate_pair(report, fixture, complete)
    result = aggregates(fixture)
    require(result[0]['methods'][2]['status'] == 'unsupported' and result[0]['methods'][2]['linear'] is None, 'unsupported cohort never discarded or scored zero')
    negative = 0
    def reject(function):
        nonlocal negative
        try:
            function()
        except AssertionError:
            negative += 1
        else:
            raise AssertionError('invalid metadata fixture admitted')
    reject(lambda: score_fraction(dict(scores, correct=129)))
    bad = copy.deepcopy(report); bad['cohorts'][0]['tasks'][0]['readouts']['methods'][0]['repetitions'][0]['actual_probe_seed_decimal'] = '124'
    reject(lambda: validate_pair(bad, fixture, complete))
    bad = copy.deepcopy(report); bad['cohorts'][0]['encoder_points'].pop()
    reject(lambda: validate_pair(bad, fixture, complete))
    bad = copy.deepcopy(report); bad['cohorts'][0]['tasks'].pop()
    reject(lambda: validate_pair(bad, fixture, complete))
    bad = copy.deepcopy(report); bad['cohorts'][0]['tasks'][0]['readouts']['methods'][0]['repetitions'].pop()
    reject(lambda: validate_pair(bad, fixture, complete))
    rendered = markdown(summarize(report, fixture, complete, {}))
    require(rendered.count('| Method | Size | Linear head % | Neural head % | Coverage % |') == 4 and
            '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |' in rendered and
            'PCA only — no encoder' in rendered and 'Untrained late encoder' in rendered and 'Untrained early encoder' in rendered,
            'four standard tables/two untrained controls/standard training table')
    # Fitted TRAIN pipelines can have undefined per-view accuracy; preserve all
    # five statuses and actual zero coverage instead of filtering that cohort.
    np,na,nc=artificial_metadata()
    for pm,am in zip(np['cohorts'][0]['tasks'][0]['readouts']['methods'][-2:],na['cohorts'][0]['tasks'][0]['methods'][-2:]):
        for pr,ar in zip(pm['repetitions'],am['repetitions']):
            for head,key in [('ridge','ridge_grouped_interval'),('tiny_secondary','tiny_grouped_interval')]:
                for score in (pr['validation_deleted'][head],ar['validation_deleted'][head]):
                    score.update(valid=0,correct=0,accuracy=None,coverage=0.)
                pr['validation_deleted'][key].update(estimate=None,lower=None,upper=None,source_groups=0)
                ar['validation_deleted'][head]['grouped_interval'].update(estimate=None,lower=None,upper=None,source_groups=0)
    validate_pair(np,na,nc)
    null_quality=aggregates(na)
    item=next(m for m in null_quality[1]['methods'] if m['method']=='native_early')
    require(item['linear'] is None and item['coverage']['mean']['decimal']==.8 and len(item['per_master'])==5 and item['undefined_cohorts']['linear']==1,'null view remains declared with exact support')
    require(not descriptive_guards(null_quality,[{'train_standardized_mae':{'mean':.1},'validation_standardized_mae':{'mean':.1}} for _ in TAGS])['joint_direction_passed'],'undefined primary view fails descriptive guards')
    bad=copy.deepcopy(report);bad['cohorts'][0]['encoder_points'][0]['encoder_progress']['losses'].pop()
    reject(lambda:validate_pair(bad,fixture,complete))
    bad=copy.deepcopy(report);bad['cohorts'].pop()
    reject(lambda:validate_pair(bad,fixture,complete))
    root=Path('/embedding/output/runs/rpb-early-mixer-confirmation');capsule=root/'early-mixer-confirmation-artificial'
    paths=[capsule/n for n in REQUIRED_METADATA]+[root/'audit-tools/run-artificial/validation.json',root/'audit-tools/reader-artificial/validate_early_mixer_confirmation.py']
    bound={'protocol':PROTOCOL,'root_approved_completed_metadata':True,'capsule':str(capsule),'audit_validation':str(paths[-2]),'audit_reader':str(paths[-1]),'source_fingerprint':'a'*64,'expected_sha256':{str(p):'b'*64 for p in paths},'audit_checks':1,'audit_archive_decodes':1}
    require(len(binding_paths(bound)[3])==10,'one explicit closed later root binding')
    for mutation in ('hash','bounds','role','approval'):
        wrong=copy.deepcopy(bound)
        if mutation=='hash':wrong['expected_sha256'][str(paths[0])]='not-a-hash'
        elif mutation=='bounds':wrong['capsule']='/embedding/output/runs/old/early-mixer-confirmation-artificial'
        elif mutation=='role':wrong['expected_sha256'].pop(str(paths[0]))
        else:wrong['root_approved_completed_metadata']=False
        reject(lambda:binding_paths(wrong))
    near(1+3.5e-9,1,'additive tolerance inside sum/outside max')
    source=Path(__file__).read_text();ast.parse(source);table=symtable.symtable(source,str(__file__),'exec')
    known=set(table.get_identifiers())|set(dir(builtins))|{'__file__','__name__'};unresolved=[];scopes=references=0
    def scan(t):
        nonlocal scopes,references
        scopes+=1
        for symbol in t.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                references+=1
                if symbol.get_name() not in known:unresolved.append(symbol.get_name())
        for child in t.get_children():scan(child)
    scan(table);require(not unresolved,'all conditional global references resolved')
    require(not AUTHORIZE_COMPLETED_METADATA,'source-only completed metadata stays blocked')
    return {'scopes':scopes,'global_references':references,'unresolved_globals':unresolved,'source_sha256':sha_bytes(source.encode()),'status': 'passed', 'protocol': PROTOCOL, 'quality_metadata_reads': 0, 'tensor_reads': 0,
            'renderer_release': AUTHORIZE_COMPLETED_METADATA, 'artificial_only': True, 'negative_cases': negative}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--binding');parser.add_argument('--output-directory');parser.add_argument('--fixture-output')
    args=parser.parse_args();require(Path('/.dockerenv').is_file(),'managed container only')
    if args.self_test:
        require(not args.binding and not args.output_directory,'artificial mode cannot open a completed binding')
        result=self_test()
        if args.fixture_output:
            fixture=Path(args.fixture_output)
            require(fixture.is_absolute() and fixture.parent==Path('/embedding/output/runs/rpb-early-mixer-confirmation/report-tools') and
                    fixture.parent.resolve(strict=True)==fixture.parent and not fixture.exists(),'exclusive artificial fixture output')
            with fixture.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,sort_keys=True));return
    require(AUTHORIZE_COMPLETED_METADATA,'unreleased renderer refuses before any metadata read')
    require(args.binding and args.output_directory,'explicit completed binding and new output directory')
    output=Path(args.output_directory)
    require(output.is_absolute() and output.parent==Path('/embedding/output/runs/rpb-early-mixer-confirmation/report-tools') and
            output.parent.resolve(strict=True)==output.parent and not output.exists() and not output.is_symlink(), 'exclusive new ignored output')
    summary=load_completed(Path(args.binding));output.mkdir(exist_ok=False)
    body=json.dumps(summary,indent=2,allow_nan=False)+'\n'
    with (output/'early_mixer_confirmation_v1.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(body)
    with (output/'EARLY_MIXER_CONFIRMATION_DIAGNOSTIC.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write(markdown(summary))
    print(json.dumps({'status':'emitted','summary_sha256':sha_bytes(body.encode()),
                     'report_sha256':sha_bytes((output/'EARLY_MIXER_CONFIRMATION_DIAGNOSTIC.md').read_bytes()),'directory':str(output),'tensor_reads':0},sort_keys=True))


if __name__=='__main__':
    main()


