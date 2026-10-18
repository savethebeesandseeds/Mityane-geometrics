#!/usr/bin/env python3
"""Blocked, metadata-only pooled-context report renderer.

No Torch/archive/model imports or tensor reads. Source-only artificial fixtures
are allowed in the managed container; completed metadata requires an explicit
release after independent review, complete inventory and successful audit.
"""
import argparse
import ast
import builtins
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import symtable

AUTHORIZE_COMPLETED_METADATA = False
PROTOCOL = 'pooled-context-v1'
CARD = 'code/evaluation/cards/pooled_context_v1.md'
CARD_SHA = 'b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7'
TIMING = [53151, 54252, 55353, 56454, 57555]
AMPLITUDE = [58656, 59757, 60858, 61959, 63060]
BUDGETS = [0, 512]
TAGS = ['RPB-v10.alt-03', 'RPB-v12']
REPS = ['rep-2701', 'rep-2802', 'rep-2903']
METHODS = [('raw',576,'Raw data'), ('mask_metadata',288,'Mask only'), ('pca_only',32,'PCA only — no encoder'), ('untrained_compact',32,'Untrained compact encoder'), ('untrained_pooled',32,'Untrained pooled encoder'), ('native_compact',32,TAGS[0]), ('native_pooled',32,TAGS[1])]
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


def validate_pair(report, audit, complete):
    require(report['protocol'] == audit['protocol'] == complete['protocol'] == PROTOCOL and
            audit['status'] == 'passed' and complete['status'] == 'complete', 'completed audited fixed comparison')
    require(len(report['cohorts']) == len(audit['cohorts']) == 5, 'all five fresh independent pairs')
    require(report['source_fingerprint'] == audit['source']['source_fingerprint'], 'same enclosing source')
    actual = 0
    for p, a, master, amp in zip(report['cohorts'], audit['cohorts'], TIMING, AMPLITUDE):
        require(p['timing_master'] == a['timing_master'] == master and p['amplitude_data_master'] == a['amplitude_data_master'] == amp,
                'fixed cohort/data association')
        metadata_matches(p['costs'], a['costs'])
        require(len(p['encoder_points']) == 2 and len(a['encoders']) == 4 and len(p['tasks']) == len(a['tasks']) == 2,
                'complete role-major 0/512 state and task matrix')
        for index, role in enumerate(('control', 'candidate')):
            zero, positive = a['encoders'][index*2:index*2+2]
            require([zero['budget'], positive['budget']] == BUDGETS and zero['role'] == positive['role'] == role and
                    zero['tag'] == positive['tag'] == TAGS[index] and zero['placement'] == positive['placement'] == 1,
                    'same early placement and explicit input-source role')
            require(zero['global_pool_input_source'] == positive['global_pool_input_source'] == index and
                    zero['registered_parameter_values'] == positive['registered_parameter_values'] == (231949 if index else 225805) and
                    zero['shared_parameter_values'] == positive['shared_parameter_values'] == 219469 and
                    zero['inactive_projection_values'] == positive['inactive_projection_values'] == (2080 if index else 0) and
                    zero['loss_reachable_parameter_values'] == positive['loss_reachable_parameter_values'] == (229869 if index else 225805),
                    'complete registered/shared/inactive/loss-reachable parameter scope')
            require(zero['training_seconds'] == 0 and zero['attempted'] == zero['completed'] == zero['sampled_rows'] == 0 and
                    not zero['fixed_query'], 'untrained point is not a reconstruction measurement')
            produced = p['encoder_points'][index]; progress = produced['encoder_progress']
            require(produced['model_tag'] == positive['tag'] and produced['placement'] == 1 and
                    produced['global_pool_input_source'] == positive['global_pool_input_source'] == index and
                    progress['completed'] == progress['attempted'] == positive['completed'] == positive['attempted'] == 512 and
                    progress['sampled_rows'] == positive['sampled_rows'] == 4096 and
                    progress['parameter_count'] == progress['cuda_parameter_count'] == (231949 if index else 225805) and
                    len(progress['losses']) == 512, 'unskipped complete actual CUDA trajectory')
            near(progress['training_seconds'], positive['training_seconds'], 'same synchronized mixed loop timer')
            for attempt, row in enumerate(progress['losses'], 1):
                require(len(row) == 5 and row[0] == row[1] == attempt and type(row[2]) is int and row[2] > 0 and
                        finite(row[3], 'objective loss') >= 0 and finite(row[4], 'gradient norm') >= 0, 'full actual objective trace')
            for split in ('training', 'validation'):
                q, r = positive['fixed_query'][split], produced[split+'_reconstruction']
                for x, y in [('mae', 'standardized_mae'), ('huber', 'standardized_huber')]:
                    if q[x] is None:
                        require(r[y] is None, 'unsupported query remains null')
                    else:
                        near(q[x], r[y], 'same original query units')
                for key in ('valid_examples', 'total_examples', 'valid_target_cells', 'requested_observed_target_cells'):
                    require(q[key] == r[key], 'same original query support counts')
        for pt, at, task, data_master in zip(p['tasks'], a['tasks'], ('lag_sign', 'amplitude'), (master, amp)):
            require(pt['task'] == at['task'] == task and pt['data_master'] == at['data_master'] == data_master, 'two separate tasks')
            pr = pt['readouts']
            require([m['method'] for m in pr['methods']] == [m['method'] for m in at['methods']] == [m[0] for m in METHODS],
                    'all seven methods including unsupported fits')
            metadata_matches(pr['fit_counts'], at['fit_counts'])
            metadata_matches(pr['pairs'], at['pairs'])
            require([(pair['id'], pair['repetition'], pair['view']) for pair in at['pairs']] ==
                    [('native_pooled_minus_native_compact', rep, view) for rep in REPS for view in VIEWS], 'all paired populations and heads')
            for pm, am, (_, size, _) in zip(pr['methods'], at['methods'], METHODS):
                require(pm['size'] == am['size'] == size and pm['status'] == am['status'] and
                        [r['id'] for r in pm['repetitions']] == REPS, 'method status/width/repetition')
                if am['status'] != 'measured':
                    require(am['status'] == 'unsupported_fit' and not am['repetitions'], 'unsupported fit is retained without predictions')
                    continue
                require([r['repetition'] for r in am['repetitions']] == REPS, 'all independent fixed head records')
                actual += 3
                for pp, aa in zip(pm['repetitions'], am['repetitions']):
                    require(pp['id'] == aa['repetition'] and pp['actual_probe_seed_decimal'] == aa['actual_probe_seed_decimal'], 'exact head seed')
                    for view in ('training', *VIEWS):
                        for head in ('ridge', 'tiny_secondary'):
                            ps, qs = pp[view][head], aa[view][head]
                            for key in ('total', 'valid', 'correct'):
                                require(ps[key] == qs[key], 'actual saved prediction counts')
                            near(ps['coverage'], qs['coverage'], 'actual coverage')
                            if qs['accuracy'] is None:
                                require(ps['accuracy'] is None, 'zero support remains null')
                            else:
                                near(ps['accuracy'], qs['accuracy'], 'same saved conditional accuracy')
                            if view != 'training':
                                interval_matches(pp[view]['ridge_grouped_interval' if head == 'ridge' else 'tiny_grouped_interval'], qs['grouped_interval'])
    require(actual == complete['head_pipelines'] == audit['counts']['head_pipelines'] and
            complete['individual_heads'] == audit['counts']['individual_heads'] == 2*actual, 'actual supported fixed pipelines')
    for key, value in {'encoder_trajectories':10, 'encoder_updates_each':512, 'attempt_limit':1024, 'retained_points':20,
                       'sampled_rows':40960, 'skipped_attempts':0, 'planned_pipelines':210, 'planned_heads':420,
                       'unique_quality_native_exports':120, 'initial_counterpart_exports':0, 'full_native_export_calls':120,
                       'query_evaluation_calls':20, 'necessary_query_forwards':80, 'driver_raw_outer_fits':10,'separate_initial_controls':2,'initial_shared_state_exact_before_training_and_heads':True}.items():
        require(complete[key] == value, 'fixed complete work matrix '+key)
    for item in (report, complete):
        require(all(item[k] is False for k in ('testing_accessed', 'stress_accessed', 'selection', 'promotion')), 'no held-out/selection/promotion')


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


def arithmetic_guards(quality, training):
    guards, numeric = {}, {}
    for panel in quality:
        if panel['task'] != 'lag_sign':
            continue
        control = next(x for x in panel['methods'] if x['method']=='native_compact')
        candidate = next(x for x in panel['methods'] if x['method']=='native_pooled')
        view = panel['view']
        supported = control['status']==candidate['status']=='measured'
        for metric_name, agg in [('ridge_mean','mean'),('ridge_worst','min')]:
            c = control['linear'][agg]['decimal'] if supported else None
            g = candidate['linear'][agg]['decimal'] if supported else None
            numeric[view+'_'+metric_name] = {'control':c,'candidate':g}
            guards[view+'_'+metric_name+'_no_worse'] = supported and g >= c
        guards[view+'_equal_per_master_coverage'] = supported and all(a['coverage']==b['coverage'] for a,b in zip(control['per_master'],candidate['per_master']))
    for split in ('training','validation'):
        c,g = [row[split+'_standardized_mae']['mean'] if row[split+'_standardized_mae'] else None for row in training]
        numeric[split+'_original_query_mae']={'control':c,'candidate':g}
        guards[split+'_original_query_mae_mean_no_worse'] = c is not None and g is not None and g <= c
    return {'scope':'known development only; no promotion or selected budget','joint_direction_passed':all(guards.values()),'guards':guards,'numeric':numeric}


def summarize(report, audit, complete, bindings):
    validate_pair(report,audit,complete)
    quality = aggregates(audit); training = []
    cohorts = copy.deepcopy(audit['cohorts'])
    for cohort, produced in zip(cohorts,report['cohorts']):
        for index in range(2):
            cohort['encoders'][index*2+1]['encoder_progress']=copy.deepcopy(produced['encoder_points'][index]['encoder_progress'])
    for index,role in enumerate(('control','candidate')):
        rows=[]
        for cohort in cohorts:
            point=cohort['encoders'][index*2+1]
            rows.append({'timing_master':cohort['timing_master'],'updates':512,
                         'training_standardized_mae':point['fixed_query']['training']['mae'],
                         'validation_standardized_mae':point['fixed_query']['validation']['mae'],
                         'GPU_update_loop_seconds':point['training_seconds'], 'encoder_progress':copy.deepcopy(point['encoder_progress'])})
        item={'tag':TAGS[index],'role':role,'updates':512,'per_master':rows}
        for key in ('training_standardized_mae','validation_standardized_mae','GPU_update_loop_seconds'):
            values=[row[key] for row in rows]
            item[key]={'mean':math.fsum(values)/5,'min':min(values),'max':max(values)} if all(x is not None for x in values) else None
        training.append(item)
    guards=arithmetic_guards(quality,training)
    require(guards['guards']==audit['decision']['guards'] and guards['joint_direction_passed']==audit['decision']['joint_direction_passed'], 'independent literal joint guard agreement')
    costs={}
    for key in cohorts[0]['costs']:
        if key=='training_loop_scope':
            continue
        values=[c['costs'][key] for c in cohorts]
        costs[key]={'sum':math.fsum(values),'mean':math.fsum(values)/5,'per_master':values}
    return {'protocol':PROTOCOL,'status':'measured_unpromoted','description':'Projected D32 versus full pooled W64 summaries before the early global native32 bottleneck; original coordinate15 training unchanged.',
            'quality':quality,'training':training,'decision':copy.deepcopy(audit['decision']),'recomputed_guards':guards,
            'cohorts':cohorts,'mixed_wall_costs':costs,'counts':copy.deepcopy(audit['counts']),
            'independent_audit':{k:copy.deepcopy(audit[k]) for k in ('checks','archive_decodes','elapsed_seconds','limits')},
            'completion':copy.deepcopy(complete),'bindings':copy.deepcopy(bindings),'testing_accessed':False,'stress_accessed':False,'promotion':False}


def pct(value):
    return 'N/A' if value is None else f'{100*value:.4f}'


def number(value):
    return 'N/A' if value is None else f'{value:.9f}'


def markdown(summary):
    lines=['# Pooled temporal context diagnostic','',
           'Completed fixed `pooled-context-v1` development comparison with an independent saved-arithmetic audit. All five fresh paired cohorts and both separate initial controls are retained. No TEST or stress was generated or read, no point was selected, and no model is promoted.','',
           "RPB-v10.alt-03 is the projected-D32 early-mixer control. RPB-v12 retains each channel's learned W64 temporal summary until the global native32 bottleneck. Both use the original 0.15 context deletion, original TRAIN scaler, reconstruction targets, optimizer, masks and heads. The wider route also increases the first global layer capacity, so this comparison does not isolate projection loss as a cause. Both independent and contextual W summaries use semantic channel order and support bits; the exact served BD32 remains the decoder's only observation signal.",'',
           'The control registers 225,805 parameters; the candidate registers 231,949, including 6,144 additional first-layer values. Exactly 219,469 common same-name values and all buffers are copied from the saved zero-update control before AdamW. Only the wider first matrix retains its own initialization. Candidate diagnostic projection values number 2,080; they stay unchanged and have no AdamW moments. Its reconstruction graph reaches 229,869 registered values, 4,064 more than the compact control; this does not imply every scalar updates. Actual named optimizer activity remains in the JSON. CUDA inference still computes the declared diagnostic outputs. Initial native outputs are different functions and receive separate untrained controls.','',
           'Timing masters are 53151/54252/55353/56454/57555; separate amplitude-data masters are 58656/59757/60858/61959/63060. Each timing trajectory uses 256 TRAIN rows from 128 sources and 128 known VALIDATION rows from 64 disjoint sources. Exactly 512 joint encoder/decoder updates at batch 8 give 4,096 sampled row exposures per trajectory: 16 equivalent presentations, not guaranteed epochs. Ten trajectories give 5,120 updates and 40,960 exposures. There are zero extra decoder-only calibration updates; amplitude never fits an encoder or its scaler.','',
           'Raw data has 288 scaled values plus 288 masks. Mask only has 288 inputs. PCA only — no encoder — fits raw TRAIN and yields 32 coordinates; native 32 receives no PCA. Three fixed repetitions 2701/2802/2903 use Ridge penalty 1 and tanh 16/Adam 0.01 for 100 full-TRAIN updates. Width 32 has 66 linear/562 neural parameters; raw 576 has 1,154/9,266 and mask 288 has 578/4,658. Each TRAIN-fitted head is reused across both VALIDATION views.','']
    for panel in summary['quality']:
        task='Timing (lag sign)' if panel['task']=='lag_sign' else 'Frozen amplitude transfer'
        view='intact VALIDATION' if panel['view']=='validation_intact' else 'extra 30% coordinate deletion VALIDATION'
        lines += [f'## {task}: {view}','','| Method | Size | Linear head % | Neural head % | Coverage % |','| --- | ---: | ---: | ---: | ---: |']
        for m in panel['methods']:
            values=[m[key]['mean']['decimal'] if m[key] else None for key in ('linear','neural','coverage')]
            lines.append(f"| {m['display_label']} | {m['size']} | {pct(values[0])} | {pct(values[1])} | {pct(values[2])} |")
        lines += ['','Equal-master means average head repetitions within each master first. Unsupported cohorts remain present and make the corresponding aggregate undefined. Coverage is valid/declared rows; accuracy is conditional on that support.','',
                  '| Timing master / data master | RPB-v10.alt-03 linear / neural % | RPB-v12 linear / neural % | Paired linear effect pp | Coverage compact / pooled % |','| --- | ---: | ---: | ---: | ---: |']
        control=next(m for m in panel['methods'] if m['method']=='native_compact');candidate=next(m for m in panel['methods'] if m['method']=='native_pooled')
        for c,g,e in zip(control['per_master'],candidate['per_master'],panel['paired_effects']):
            delta='N/A' if e['ridge_mean'] is None else f"{100*e['ridge_mean']:+.4f}"
            left=f"{pct(c['linear']['decimal'])} / {pct(c['neural']['decimal'])}" if c['status']=='measured' else 'unsupported'
            right=f"{pct(g['linear']['decimal'])} / {pct(g['neural']['decimal'])}" if g['status']=='measured' else 'unsupported'
            cover=f"{pct(c['coverage']['decimal'] if c['coverage'] else None)} / {pct(g['coverage']['decimal'] if g['coverage'] else None)}"
            lines.append(f"| {c['timing_master']} / {c['data_master']} | {left} | {right} | {delta} | {cover} |")
        lines += ['','Full ranges, every marginal interval and all common-population paired source intervals remain in the durable JSON. Within-master 1,000-replicate/95% intervals are conditional on fixed encoders/readouts; bounds are never averaged into an encoder-seed confidence interval.','']
    lines += ['## Training and reconstruction','','| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |','| --- | ---: | ---: | ---: | ---: |']
    for t in summary['training']:
        values=[t[key]['mean'] if t[key] else None for key in ('training_standardized_mae','validation_standardized_mae','GPU_update_loop_seconds')]
        lines.append(f"| {t['tag']} | 512 | {number(values[0])} | {number(values[1])} | {number(values[2])} |")
    lines += ['','Errors are fixed original-query standardized MAE in common original-scaler units. Full sampled Huber objective/gradient/target-count traces remain separate. Synchronized CUDA-loop wall time includes CPU trace/live-state capture and is descriptive, not pure kernel cost. Checkpoint/continuation writing, query/native transfer and parent hashing, baseline/head/bootstrap and audit stages are separate.','',
              '| Timing master | RPB-v10.alt-03 TRAIN / VALIDATION MAE | RPB-v12 TRAIN / VALIDATION MAE | Loop seconds compact / pooled |','| --- | ---: | ---: | ---: |']
    for c,g in zip(summary['training'][0]['per_master'],summary['training'][1]['per_master']):
        lines.append(f"| {c['timing_master']} | {number(c['training_standardized_mae'])} / {number(c['validation_standardized_mae'])} | {number(g['training_standardized_mae'])} / {number(g['validation_standardized_mae'])} | {number(c['GPU_update_loop_seconds'])} / {number(g['GPU_update_loop_seconds'])} |")
    lines += ['','## Frozen joint direction','','| Guard | Passed |','| --- | --- |']
    for name,passed in summary['decision']['guards'].items():lines.append(f'| `{name}` | {str(passed).lower()} |')
    lines += ['',f"Joint direction passed: **{str(summary['decision']['joint_direction_passed']).lower()}**. Six numeric guards cover timing linear mean/worst scores in both views and mean original TRAIN/VALIDATION MAE; equal per-master coverage is checked separately. A failure stops this route without width/rate/budget/head rescue. A pass requires separately frozen fresh confirmation before promotion. All secondary neural/amplitude tradeoffs remain reported; strong initial amplitude controls limit learning-credit claims.",'',
              '| Mixed wall-time scope | Total seconds | Mean paired-cohort seconds |','| --- | ---: | ---: |']
    for key,value in summary['mixed_wall_costs'].items():lines.append(f"| `{key}` | {value['sum']:.6f} | {value['mean']:.6f} |")
    a=summary['independent_audit'];counts=summary['counts']
    lines += ['','## Evidence and limits','',f"Independent audit PASS: {a['checks']:,} checks/{a['archive_decodes']:,} CPU archive decodes/{a['elapsed_seconds']:.6f} seconds. Supported pipelines/heads: {counts['head_pipelines']}/{counts['individual_heads']} (planned 210/420). Actual full native callbacks: 120 distinct quality exports, with no counterpart reuse. Original timing queries use 20 writers/80 masked forwards. All 20 checkpoint points and five-role companions are retained.",'',
              'The CPU audit checks saved common/nonshared initialization, unchanged diagnostic projection and absent moments, original scaler/data/support, fixed query reductions, maps/logits/argmax and source intervals. It performs no CUDA/model/autodiff/optimizer/head fit or PCA/SVD fit. CUDA checkpoint bodies are byte-bound; actual named live CPU state/AdamW/scaler witnesses and captured CUDA admission establish association without independently reenacting the optimizer trajectory. No causal bottleneck claim or broad generalization claim follows from these synthetic tasks. Pure GPU kernel time is unmeasured. Historical quality payloads, TEST and stress are absent. Original RPB-v7 artifacts and the active v4 reference remain unchanged.','',
              '[Frozen prospective card](../../evaluation/cards/pooled_context_v1.md). [Full durable summary](../../../doc/results/pooled_context_v1.json).','','Metadata/source bindings:','']
    for key,value in sorted(summary['bindings'].items()):
        if isinstance(value,str):lines.append(f'- `{key}`: `{value}`')
    return '\n'.join(lines)+'\n'


def binding_paths(binding):
    require(set(binding)=={'protocol','root_approved_completed_metadata','capsule','audit_validation','audit_reader',
                          'source_fingerprint','expected_sha256','audit_checks','audit_archive_decodes'} and
            binding['protocol'] == PROTOCOL and binding['root_approved_completed_metadata'] is True,
            'explicit root-approved completed metadata record')
    require(all(type(binding[k]) is int and binding[k]>0 for k in ('audit_checks','audit_archive_decodes')), 'actual successful audit counts')
    capsule = Path(binding['capsule']); audit_path = Path(binding['audit_validation']); reader = Path(binding['audit_reader'])
    root = Path('/embedding/output/runs/rpb-pooled-context')
    require(capsule.is_absolute() and capsule.parent == root and capsule.name.startswith('pooled-context-'),
            'closed protocol capsule path')
    require(audit_path.is_absolute() and audit_path.is_relative_to(root/'audit-tools') and audit_path.name == 'validation.json' and
            reader.is_absolute() and reader.is_relative_to(root/'audit-tools') and reader.name == 'validate_pooled_context.py',
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
                'checkpoint_artifacts':[copy.deepcopy(index[relative]) for master in TIMING for role in ('control','candidate')
                    for point in BUDGETS for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.continuation.pt')
                    for relative in [f'results/seed-{master}-lag_sign/{role}/point-{point}/checkpoint.pt{suffix}']]}
    return summarize(report,audit,complete,bindings)


def artificial_metadata():
    # Only synthetic in-memory JSON: no filesystem, model, archive or quality inputs.
    scores = {'total':128,'valid':128,'correct':100,'accuracy':100/128,'coverage':1.}
    report = {'protocol':PROTOCOL,'source_fingerprint':'a'*64,'cohorts':[],
              'testing_accessed':False,'stress_accessed':False,'selection':False,'promotion':False}
    audit = {'protocol':PROTOCOL,'status':'passed','source':{'source_fingerprint':'a'*64},'cohorts':[],
             'checks':123,'archive_decodes':7,'elapsed_seconds':.2,
             'counts':{'head_pipelines':210,'individual_heads':420,'native_full_export_calls':120,'independent_initial_controls':2},'limits':{'saved_arithmetic_only':True}}
    complete = {'protocol':PROTOCOL,'status':'complete','head_pipelines':210,'individual_heads':420,
                'planned_pipelines':210,'planned_heads':420,'encoder_trajectories':10,'sampled_rows':40960,
                'encoder_updates_each':512,'attempt_limit':1024,'retained_points':20,'skipped_attempts':0,
                'unique_quality_native_exports':120,'initial_counterpart_exports':0,'full_native_export_calls':120,
                'query_evaluation_calls':20,'necessary_query_forwards':80,'driver_raw_outer_fits':10,'separate_initial_controls':2,'initial_shared_state_exact_before_training_and_heads':True,
                'testing_accessed':False,'stress_accessed':False,'selection':False,'promotion':False}
    for timing,amp in zip(TIMING,AMPLITUDE):
        costs={key:.1 for key in ('generation_and_observation_io_seconds','binding_and_checkpoint_io_seconds',
              'CUDA_query_transfer_verification_io_seconds','CUDA_unique_quality_native_transfer_verification_seconds',
              'CPU_baseline_preparation_io_seconds','CPU_head_bootstrap_io_seconds')}
        costs['training_loop_scope']='CUDA updates plus CPU trace/live-state evidence capture; not pure kernel time'
        a={'timing_master':timing,'amplitude_data_master':amp,'encoders':[],'tasks':[],'costs':copy.deepcopy(costs)}
        p={'timing_master':timing,'amplitude_data_master':amp,'encoder_points':[],'tasks':[],'costs':copy.deepcopy(costs)}
        for index,role in enumerate(('control','candidate')):
            for point in BUDGETS:
                query={'mae':.1,'huber':.05,'valid_examples':128,'total_examples':128,
                       'valid_target_cells':1000,'requested_observed_target_cells':1000}
                a['encoders'].append({'tag':TAGS[index],'design_tag':'RPB-v10' if index==0 else 'RPB-v12','role':role,
                    'placement':1,'global_pool_input_source':index,'registered_parameter_values':231949 if index else 225805,
                    'shared_parameter_values':219469,'inactive_projection_values':2080 if index else 0,
                    'loss_reachable_parameter_values':229869 if index else 225805,'budget':point,'completed':point,'attempted':point,'sampled_rows':point*8,
                    'training_seconds':point/50,'fixed_query':{s:copy.deepcopy(query) for s in ('training','validation')} if point else {}})
                if point:
                    produced_query={('standardized_mae' if k=='mae' else 'standardized_huber' if k=='huber' else k):v for k,v in query.items()}
                    p['encoder_points'].append({'model_tag':TAGS[index],'placement':1,'global_pool_input_source':index,
                        'encoder_progress':{'attempted':point,'completed':point,'sampled_rows':point*8,'training_seconds':point/50,
                            'parameter_count':231949 if index else 225805,'cuda_parameter_count':231949 if index else 225805,
                            'losses':[[i,i,64,.1/(i+1),.2] for i in range(1,point+1)]},
                        'training_reconstruction':copy.deepcopy(produced_query),'validation_reconstruction':copy.deepcopy(produced_query)})
        for task,master in (('lag_sign',timing),('amplitude',amp)):
            methods,produced_methods=[],[]
            for name,width,_ in METHODS:
                reps,produced_reps=[],[]
                for rep in REPS:
                    ar={'repetition':rep,'actual_probe_seed_decimal':'123'};pr={'id':rep,'actual_probe_seed_decimal':'123'}
                    for view in ('training',*VIEWS):
                        aa={head:dict(scores) for head in ('ridge','tiny_secondary')};pp=copy.deepcopy(aa)
                        if view!='training':
                            for head,key in (('ridge','ridge_grouped_interval'),('tiny_secondary','tiny_grouped_interval')):
                                interval={'estimate':100/128,'lower':.7,'upper':.9,'source_groups':64,'replicates':1000}
                                aa[head]['grouped_interval']=dict(interval);pp[key]={**interval,'confidence':.95}
                        ar[view]=aa;pr[view]=pp
                    reps.append(ar);produced_reps.append(pr)
                methods.append({'method':name,'size':width,'status':'measured','repetitions':reps})
                produced_methods.append({'method':name,'size':width,'status':'measured','repetitions':produced_reps})
            pairs=[{'id':'native_pooled_minus_native_compact','repetition':rep,'view':view,'status':'measured',
                    **{head:{'estimate':0.,'lower':-.1,'upper':.1,'source_groups':64,'replicates':1000,'confidence':.95}
                       for head in ('ridge','tiny_secondary')}} for rep in REPS for view in VIEWS]
            fit_counts={'ridge_fits':21,'tiny_fits':21}
            a['tasks'].append({'task':task,'data_master':master,'methods':methods,'pairs':copy.deepcopy(pairs),'fit_counts':dict(fit_counts)})
            p['tasks'].append({'task':task,'data_master':master,'readouts':{'methods':produced_methods,'pairs':pairs,'fit_counts':dict(fit_counts)}})
        audit['cohorts'].append(a);report['cohorts'].append(p)
    quality=aggregates(audit)
    training=[{s+'_standardized_mae':{'mean':.1} for s in ('training','validation')} for _ in TAGS]
    audit['decision']=arithmetic_guards(quality,training)
    return report,audit,complete


def self_test():
    near(1.+3.5e-9,1.,'declared additive tolerance includes both terms')
    report,audit,complete=artificial_metadata();summary=summarize(report,audit,complete,{})
    rendered=markdown(summary)
    require(rendered.count('| Method | Size | Linear head % | Neural head % | Coverage % |')==4 and
            '| Encoder | Updates | Train error ↓ | Validation error ↓ | GPU training seconds |' in rendered and
            'PCA only — no encoder' in rendered,'four exact standard panels and training header')
    require(len(summary['quality'])==4 and all(len(p['methods'])==7 for p in summary['quality']) and
            summary['quality'][0]['methods'][0]['linear']['mean']['decimal']==100/128,'exact rational master/head means')
    audit['cohorts'][0]['tasks'][0]['methods'][0]['repetitions'][0]['validation_intact']['ridge']['grouped_interval']['lower']+=1e-12
    validate_pair(report,audit,complete)
    # Missing fit stays unsupported; do not reduce the equal-master denominator.
    am=audit['cohorts'][2]['tasks'][0]['methods'][2];pm=report['cohorts'][2]['tasks'][0]['readouts']['methods'][2]
    am.update(status='unsupported_fit',repetitions=[])
    pm.update(status='unsupported_fit',repetitions=[{'id':rep,'status':'unsupported_fit'} for rep in REPS])
    for t in (audit['cohorts'][2]['tasks'][0],report['cohorts'][2]['tasks'][0]['readouts']):
        t['fit_counts']={'ridge_fits':18,'tiny_fits':18}
    audit['counts']['head_pipelines']=complete['head_pipelines']=207
    audit['counts']['individual_heads']=complete['individual_heads']=414
    summary=summarize(report,audit,complete,{})
    require(summary['quality'][0]['methods'][2]['linear'] is None and
            summary['quality'][0]['methods'][2]['undefined_cohorts']['linear']==1,'unsupported PCA cohort explicitly retained')
    # A fitted native head can have zero valid rows in one view.
    null_report,null_audit,null_complete=artificial_metadata()
    for method in (null_report['cohorts'][0]['tasks'][0]['readouts']['methods'][6],null_audit['cohorts'][0]['tasks'][0]['methods'][6]):
        for rep in method['repetitions']:
            for head in ('ridge','tiny_secondary'):
                rep['validation_deleted'][head].update(valid=0,correct=0,accuracy=None,coverage=0.)
                if 'grouped_interval' in rep['validation_deleted'][head]:
                    rep['validation_deleted'][head]['grouped_interval'].update(estimate=None,lower=None,upper=None,source_groups=0)
            for key in ('ridge_grouped_interval','tiny_grouped_interval'):
                if key in rep['validation_deleted']:
                    rep['validation_deleted'][key].update(estimate=None,lower=None,upper=None,source_groups=0)
    null_quality=aggregates(null_audit)
    null_audit['decision']=arithmetic_guards(null_quality,[{s+'_standardized_mae':{'mean':.1} for s in ('training','validation')} for _ in TAGS])
    result=summarize(null_report,null_audit,null_complete,{})
    native=result['quality'][1]['methods'][6]
    require(native['linear'] is None and native['coverage']['mean']['decimal']==.8 and
            native['undefined_cohorts']['linear']==1 and not result['decision']['joint_direction_passed'],
            'undefined view is null with actual coverage, all five cohorts and failed guards')
    negative=0
    def reject(action):
        nonlocal negative
        try: action()
        except (AssertionError,KeyError): negative+=1
        else: raise AssertionError('invalid artificial binding or schema admitted')
    for change in ('seed','point','cohort','head','pair','counter','status','width','input_source','registered','inactive','shared'):
        p,a,c=artificial_metadata()
        if change=='seed':p['cohorts'][0]['tasks'][0]['readouts']['methods'][0]['repetitions'][0]['actual_probe_seed_decimal']='124'
        elif change=='point':p['cohorts'][0]['encoder_points'].pop()
        elif change=='cohort':p['cohorts'].pop()
        elif change=='head':p['cohorts'][0]['tasks'][0]['readouts']['methods'][0]['repetitions'].pop()
        elif change=='pair':p['cohorts'][0]['tasks'][0]['readouts']['pairs'].pop()
        elif change=='counter':p['cohorts'][0]['encoder_points'][0]['encoder_progress']['attempted']=513
        elif change=='status':a['status']='failed'
        elif change=='width':p['cohorts'][0]['tasks'][0]['readouts']['methods'][0]['size']=32
        elif change=='input_source':p['cohorts'][0]['encoder_points'][1]['global_pool_input_source']=0
        elif change=='registered':p['cohorts'][0]['encoder_points'][1]['encoder_progress']['parameter_count']=225805
        elif change=='inactive':a['cohorts'][0]['encoders'][3]['inactive_projection_values']=2112
        elif change=='shared':a['cohorts'][0]['encoders'][3]['shared_parameter_values']=1
        reject(lambda:summarize(p,a,c,{}))
    root=Path('/embedding/output/runs/rpb-pooled-context');capsule=root/'pooled-context-fixture'
    paths=[capsule/r for r in REQUIRED_METADATA]+[root/'audit-tools/run-fixture/validation.json',root/'audit-tools/reader/validate_pooled_context.py']
    binding={'protocol':PROTOCOL,'root_approved_completed_metadata':True,'capsule':str(capsule),'audit_validation':str(paths[-2]),
             'audit_reader':str(paths[-1]),'source_fingerprint':'a'*64,'expected_sha256':{str(p):'b'*64 for p in paths},
             'audit_checks':123,'audit_archive_decodes':7}
    require(len(binding_paths(binding)[3])==10,'closed artificial metadata role binding')
    for change in ('hash','bounds','role','approval'):
        b=copy.deepcopy(binding)
        if change=='hash':b['expected_sha256'][str(paths[0])]='invalid'
        elif change=='bounds':b['capsule']='/embedding/output/runs/other/capsule'
        elif change=='role':b['expected_sha256'].pop(str(paths[0]))
        else:b['root_approved_completed_metadata']=False
        reject(lambda:binding_paths(b))
    require(not AUTHORIZE_COMPLETED_METADATA,'source fixture has no completed access')
    source=Path(__file__).read_text(encoding='utf-8');ast.parse(source)
    table=symtable.symtable(source,str(__file__),'exec');known=set(table.get_identifiers())|set(dir(builtins))|{'__file__','__name__'}
    stack=[table];scopes=references=0;unresolved=[]
    while stack:
        current=stack.pop();scopes+=1;stack.extend(current.get_children())
        for symbol in current.get_symbols():
            if symbol.is_referenced() and symbol.is_global():
                references+=1
                if symbol.get_name() not in known:unresolved.append(symbol.get_name())
    require(not unresolved,'source-only complete global names')
    return {'status':'passed','protocol':PROTOCOL,'artificial_only':True,'negative_cases':negative,
            'quality_metadata_reads':0,'tensor_reads':0,'renderer_release':False,'four_quality_panels':4,'all_cohorts':5,
            'scopes':scopes,'global_references':references,'unresolved_globals':unresolved,'source_sha256':sha_bytes(source.encode())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--binding');parser.add_argument('--output-directory');parser.add_argument('--fixture-output')
    args=parser.parse_args();require(Path('/.dockerenv').is_file(),'managed container only')
    if args.self_test:
        require(not args.binding and not args.output_directory,'artificial mode cannot open a completed binding')
        result=self_test()
        if args.fixture_output:
            fixture=Path(args.fixture_output)
            require(fixture.is_absolute() and fixture.parent==Path('/embedding/output/runs/rpb-pooled-context/report-tools') and
                    fixture.parent.resolve(strict=True)==fixture.parent and not fixture.exists(),'exclusive artificial fixture output')
            with fixture.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,sort_keys=True));return
    require(AUTHORIZE_COMPLETED_METADATA,'unreleased renderer refuses before any metadata read')
    require(args.binding and args.output_directory,'explicit completed binding and new output directory')
    output=Path(args.output_directory)
    require(output.is_absolute() and output.parent==Path('/embedding/output/runs/rpb-pooled-context/report-tools') and
            output.parent.resolve(strict=True)==output.parent and not output.exists() and not output.is_symlink(), 'exclusive new ignored output')
    summary=load_completed(Path(args.binding));output.mkdir(exist_ok=False)
    body=json.dumps(summary,indent=2,allow_nan=False)+'\n'
    with (output/'pooled_context_v1.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(body)
    with (output/'POOLED_CONTEXT_DIAGNOSTIC.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write(markdown(summary))
    print(json.dumps({'status':'emitted','summary_sha256':sha_bytes(body.encode()),
                     'report_sha256':sha_bytes((output/'POOLED_CONTEXT_DIAGNOSTIC.md').read_bytes()),'directory':str(output),'tensor_reads':0},sort_keys=True))


if __name__=='__main__':
    main()
