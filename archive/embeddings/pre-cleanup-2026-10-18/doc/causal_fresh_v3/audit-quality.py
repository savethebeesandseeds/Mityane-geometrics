#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent saved-scalar audit. Standard library only; never loads a model."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

CARD_SHA = 'f88fa9751ff9da0e808805da9975fb0bf3afd4433688f3cfb420686392e6396f'
INPUT_SHA = 'ce691b6ec2563cbf7a6b2a615c3912e3a9ec586f5198e1655450a22c9f9c9ffd'
TAG, INIT = 'CPB-v3.market-01', 'CPB-v3.market-01.init'
TASKS = ('past_volatility', 'past_trend', 'past_activity', 'past_relation')
REPS, MASTERS = (2701, 2802, 2903), (941101, 941102, 941103)
STAGES, MODES = ('initial', 'final'), ('synthetic_train', 'market_scaler')
VIEWS, HEADS = ('validation_intact', 'validation_deleted'), ('ridge', 'tiny_secondary')


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
    fields = {'kind': 'cpb_v3_saved_checkpoint_causal_proof_v1', 'status': 'passed',
              'model': 'CPB-v3', 'source_updates': 0, 'causal_updates': updates,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('card', 'input-manifest', 'raw-controls', 'training', 'run', 'report', 'quality-receipt', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/cuwacunu'))
    parser.add_argument('--map', action='append', default=[], metavar='ORIGINAL_PREFIX=LOCAL_PREFIX')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve existing audit: ' + str(args.output))
    if not Path('/.dockerenv').is_file():
        raise RuntimeError('Execute this scalar auditor inside the existing managed container')
    a = Audit([item.split('=', 1) for item in args.map])
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
    a.check(build['status'] == 'passed' and build['card_sha256'] == CARD_SHA and len(build['binaries']) == 5,
            'Frozen build admission')
    for binary in build['binaries']:
        a.bound(binary)
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
    a.check(tables['kind'] == 'causal_fresh_v3_results_v1' and tables['model'] == TAG and tables['source_updates'] == 0 and
            tables['new_market_rows'] == 8 and not tables['initial_quality_reused'] and not tables['historical_encoder_quality_targets_used'] and
            not tables['overall_winner_declared'] and not tables['application_changed'], 'Fresh descriptive report identity')
    a.check(tables['market_rows'][:8] == controls['rows'] and len(tables['market_rows']) == 16,
            'Eight raw and eight newly measured market rows only')
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
        a.check(method['name'] == 'cpb_v3' and method['size'] == 36 and method['ridge_parameters'] == 4*(36+1) and
                method['neural_parameters'] == 36*16+16+16*4+4 and method['ridge_fits'] == 1 and
                method['ridge_task_fits'] == 4 and method['neural_fits'] == 3 and method['training_feature_rows'] == 112,
                'Market D36 parameters and fit budgets')
        for key, role in (('training_path','train'),('validation_path','validation')):
            p = a.path(method[key]).resolve(strict=True)
            a.check(p == (args.run/'features'/stage/'market'/('cpb_v3-'+role+'.pt')).resolve(strict=True), 'Market feature stage association')
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
    for key in ('trust_protocol_admission_passed','full_coverage','clean_fresh_baseline_established'):
        a.check(tables[key] is True and completion[key] is True, 'Clean-baseline trust/completeness claim ' + key)
    a.check(full_coverage and not completion['historical_encoder_quality_targets_used'] and not completion['overall_winner_declared'],
            'Only trust and completeness establish a baseline; no quality-winner gate')
    assets = read(args.report/'asset_hashes.json')
    for binding in assets['run_assets']:a.bound(binding)
    for key in ('training_summary','initialization','calibration','typed_resume','query_initial','query_final','engineering_proof','build_admission','reporter'):
        a.bound(assets[key])
    for key in ('report','tables','assets'):a.bound(completion[key])
    for stage in STAGES:
        for key in ('proof','witness'):a.bound(assets['proofs'][stage][key])
    for path in (args.card,args.input_manifest,args.raw_controls,args.quality_receipt,args.training/'training-summary.json',
                 args.training/'fresh-initialization.json',args.training/'calibration/summary.json',Path(__file__)):
        a.pin(path)
    result = {'kind':'causal_fresh_v3_quality_independent_audit_v1','status':'passed' if not a.errors else 'failed',
        'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':TAG,'checks':a.checks,'mismatches':a.errors,
        'scope':'Independent standard-library saved-scalar arithmetic and hash bindings; no model/tensor loading, forwards, training or head fitting.',
        'market_new_rows':8,'raw_control_rows':8,'synthetic_cohorts_per_stage':3,'input_policies':2,'head_repetitions':3,
        'export_calls':1572,'initial_and_final_quality_new':True,'historical_quality_targets_used':False,
        'baseline_establishment_rule':'Fresh ancestry + trust admission + complete finite results and full declared coverage only; quality scores are descriptive.',
        'market_arithmetic':market_audit,'synthetic_arithmetic':synthetic_audit,'full_coverage':full_coverage,
        'source_updates':0,'initialization_seed':943001,'proof_checks':{s:v['checks'] for s,v in proof_values.items()},
        'verified_artifacts':list(a.pins.values()),'path_maps':a.maps,
        'limitations':['Reported scalar means/support/recipes and artifact bytes are independently checked; fitted predictions and model tensors are not recomputed.',
                       'The recorded trust proofs supply actual CUDA learning, causal interventions and typed-resume evidence; this auditor binds their saved bytes.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'checks':a.checks,'mismatches':a.errors,'artifacts':len(a.pins),'output':str(args.output)}))
    return 0 if not a.errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
