#!/usr/bin/env python3
"""Independent fresh early-mixer confirmation saved-arithmetic reader with release gates.

Unchanged pure arithmetic was copied from SHA-bound reader SOURCE only. No
historical reader/main is imported. Only the pinned stdlib Torch archive codec
is a runtime dependency. CUDA checkpoint bodies are byte-bound, never decoded.
Reader code, schema, fixtures and peer review are sealed before quality
generation with both flags false. A separate flags-only released copy requires
root authorization after the completed inventory.
"""
import argparse
import array
import ast
import builtins
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import symtable
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
sys.dont_write_bytecode = True
PROTOCOL = 'early-mixer-confirmation-v1'
IMPLEMENTATION_PROTOCOL = 'early-mixer-reliability-v1'
CARD = 'code/evaluation/cards/early_mixer_confirmation_v1.md'
CARD_SHA = '98ded5254e4b9bccb931fde491bf2657b7541f1561db3ff30fd1c4896f1433ea'
COHORT_ROOT = 'output/runs/rpb-early-mixer-confirmation'
BUDGETS = (0,512)
TIMING_MASTERS = (64161,65262,66363,67464,68565)
AMPLITUDE_MASTERS = (69666,70767,71868,72969,74070)
HEAD_REPETITIONS = (2701,2802,2903)
METHODS = {'raw':576,'mask_metadata':288,'pca_only':32,'untrained_late':32,'untrained_early':32,'native_late':32,'native_early':32}
VIEWS = ('training','validation-intact','validation-deleted')
FIT_PROTOCOL = IMPLEMENTATION_PROTOCOL+'/lag_sign'
EXTERNAL_FIT_PROTOCOL = PROTOCOL+'/lag_sign'
COUNTER_POLICY = 'splitmix64-counter-rows-masks-torch-attempt-v1'
POLICY = 'rpb-training-context-deletion-015-v1'
EARLY_ARCHITECTURE = 'aligned-mixer-before-temporal-v1'
VIEW_STREAMS = {'lag_sign':0x656d633174696d30,'amplitude':0x656d6331616d7030}
HELPER = 'output/runs/archive-controls/input-audit-20261006T191016Z-d7b99529/validate_phase1.py'
HELPER_SHA = '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d'
COPIED_ARITHMETIC_SOURCE_SHA256 = 'ed26cb7e10ec1c4219bc780dd9caa64f2c03438c48d22cd4169617a9a59dcc21'
COPIED_SOURCE_ADMISSION_SHA256 = 'bfcf36a1352f617a1c1d9acc65149454235e86d411de2ccd154520ad2342d0ca'
MASK64 = (1<<64)-1
ATOL = RTOL = 2e-9
F32_ATOL,F32_RTOL = 2e-6,2e-5
CHECKS = 0
ARCHIVES = 0
R = None
REVIEWED_SCHEMA = False
MEASURED_IMPLEMENTATION = False
ARCHITECTURES = ('aligned-mixer-after-temporal-v1', 'aligned-mixer-before-temporal-v1')
OUTPUT_SEMANTICS = ('local_observed_contextual_aligned_global_semantic_mlp_bottleneck_v1',
                    'local_observed_contextual_pretemporal_aligned_global_semantic_mlp_bottleneck_v1')
RECONSTRUCTION_SEMANTICS = ('exact_contextual_observed_global_semantic_mlp_export_v1',
                          'exact_pretemporal_contextual_observed_global_semantic_mlp_export_v1')
CONTEXT_LITERALS = {
    'training_policy_id': POLICY,
    'context_deletion_ratio': '0.15',
    'context_deletion_stream': '0x6374782d64726f70',
    'context_deletion_rng_policy': 'splitmix64-counter-base-plus-semantic-canonical-BCHF-ordinal;top53-u01;independent-ctx-drop-stream-v1',
    'context_deletion_repair_policy': 'eligible-channels-only;restore-earliest-erased-original-patch-first-original-visible-coordinate-until-two-groups-v1',
    'context_deletion_visibility_policy': 'V=(O&~A)&~E;E-subset-original-visible;repair-only-E;original-O-A-Q-eligibility-Huber-unchanged-v1',
    'context_deletion_count_policy': 'cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only',
    'context_deletion_resume_policy': 'fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API'
}


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def f32(value):
    result = struct.unpack('<f', struct.pack('<f', value))[0]
    check(math.isfinite(result), 'finite float32 representation')
    return result


def stream_seed(seed, stream):
    value = (seed + 0x9e3779b97f4a7c15 * stream) & MASK64
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK64
    return (value ^ (value >> 31)) & MASK64


def fnv64(text):
    result = 14695981039346656037
    for byte in text.encode('utf-8'):
        result = ((result ^ byte) * 1099511628211) & MASK64
    return result


def fnv(raw):
    value=14695981039346656037
    for byte in raw:value=((value^byte)*1099511628211)&MASK64
    return format(value,'016x')


class MT19937_64:
    """The fixed C++ engine; not Python's 32-bit random implementation."""
    def __init__(self, seed):
        self.values = [seed & MASK64]
        for i in range(1, 312):
            previous = self.values[-1]
            self.values.append((6364136223846793005 * (previous ^ (previous >> 62)) + i) & MASK64)
        self.index = 312

    def draw(self):
        if self.index == 312:
            for i in range(312):
                x = (self.values[i] & 0xffffffff80000000) | (self.values[(i + 1) % 312] & 0x7fffffff)
                self.values[i] = self.values[(i + 156) % 312] ^ (x >> 1) ^ (0xb5026f5aa96619e9 if x & 1 else 0)
            self.index = 0
        value = self.values[self.index]
        self.index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71d67fffeda60000
        value ^= (value << 37) & 0xfff7eee000000000
        value ^= value >> 43
        return value & MASK64


def deletion_view(base_values, base_mask, erasure):
    check(len(base_values) == len(base_mask) == len(erasure), 'coordinate support geometry')
    check(all(math.isfinite(x) for x in base_values), 'finite legal observations')
    check(all(bool(m) or value == 0. for value, m in zip(base_values, base_mask)), 'zero hidden base storage')
    mask = [bool(m) and not bool(e) for m, e in zip(base_mask, erasure)]
    return [value if valid else 0. for value, valid in zip(base_values, mask)], mask


def validate_pairs(labels, sources, masks):
    check(len(labels) == len(sources) == len(masks) and bool(sources), 'paired row geometry')
    groups = {}
    for index, source in enumerate(sources):
        check(labels[index] in (0, 1), 'binary labels')
        groups.setdefault(source, []).append(index)
    for indices in groups.values():
        check(len(indices) == 2 and {labels[i] for i in indices} == {0, 1}, 'complete opposite-label source pairs')
        check(masks[indices[0]] == masks[indices[1]], 'class-independent paired observation mask')
    return groups


def normalize(rows, valid, mean, scale):
    check(len(rows) == len(valid) and bool(mean) and len(mean) == len(scale) and
          all(math.isfinite(x) and x > 0 for x in scale), 'frozen map schema')
    output = []
    for row, supported in zip(rows, valid):
        check(len(row) == len(mean) and all(math.isfinite(x) for x in row), 'finite frozen feature rows')
        value = [(x - m) / s for x, m, s in zip(row, mean, scale)] if supported else [0.] * len(mean)
        check(all(math.isfinite(x) for x in value), 'finite frozen map arithmetic')
        output.append(value)
    return output


def score(predictions, labels, valid):
    check(len(predictions) == len(labels) == len(valid) and bool(labels), 'score row population')
    selected = [i for i, value in enumerate(valid) if value]
    correct = sum(predictions[i] == labels[i] for i in selected)
    return {'total': len(labels), 'valid': len(selected), 'correct': correct,
            'coverage': len(selected) / len(labels),
            'accuracy': correct / len(selected) if selected else None}


def paired_group_effect(left, right, labels, left_valid, right_valid, sources):
    check(len(left) == len(right) == len(labels) == len(left_valid) == len(right_valid) == len(sources),
          'paired effect row geometry')
    groups = {}
    for i, source in enumerate(sources):
        if left_valid[i] and right_valid[i]:
            group = groups.setdefault(source, [0, 0])
            group[0] += int(left[i] == labels[i]) - int(right[i] == labels[i])
            group[1] += 1
    numerator = sum(x[0] for x in groups.values())
    denominator = sum(x[1] for x in groups.values())
    return {'groups': [groups[key] for key in sorted(groups)], 'valid': denominator,
            'estimate': numerator / denominator if denominator else None}


def tensor(record, dtype, shape):
    check(record['dtype'] == dtype and record['shape'] == list(shape) and
          len(record['values']) == math.prod(shape), 'typed tensor shape/dtype')
    if dtype=='BoolStorage':check(all(x in (0,1) for x in record['values']),'canonical bool storage')
    return record['values']


def equal_tensor(left, right, message):
    check(left['dtype'] == right['dtype'] and left['shape'] == right['shape'] and
          list(left['values']) == list(right['values']), message)


def group(value):
    return value if isinstance(value,dict) else vars(value)


def scalar(archive,key,dtype='LongStorage'):
    return tensor(archive[key],dtype,[])[0]


def text(archive,key):
    return R.text(archive[key])


def load(path):
    global ARCHIVES
    check(Path(path).name!='checkpoint.pt','ordinary CUDA checkpoint body decode is forbidden; bytes and CPU companions only')
    ARCHIVES += 1
    return R.load(path)


def source_manifest(ids):
    return ''.join(str(len(x.encode('utf-8')))+':'+x for x in ids)


def mixed(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK64
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK64
    return x ^ (x >> 31)


def finite(values,message):
    check(all(math.isfinite(x) for x in values),message)


def bytes_of(record):
    if 'raw' in record:return record['raw']
    fmt={'FloatStorage':'f','DoubleStorage':'d','LongStorage':'q','BoolStorage':'B','ByteStorage':'B'}[record['dtype']]
    return struct.pack('<'+fmt*len(record['values']),*record['values'])


def exact(left,right,message):
    check(left['dtype']==right['dtype'] and left['shape']==right['shape'] and bytes_of(left)==bytes_of(right),message)


def scaler(archive):
    check(set(archive)=={'mean','scale','count','channel_ids','floor_applied','scale_floor'},'exact scaler schema')
    mean=tensor(archive['mean'],'DoubleStorage',[3,3]);scale=tensor(archive['scale'],'DoubleStorage',[3,3])
    count=tensor(archive['count'],'LongStorage',[3,3]);ids=tensor(archive['channel_ids'],'LongStorage',[3])
    floors=tensor(archive['floor_applied'],'BoolStorage',[3,3]);floor=scalar(archive,'scale_floor','DoubleStorage')
    finite(mean,'finite frozen scaler mean');check(ids==array.array('q',[0,1,2]) or list(ids)==[0,1,2],'semantic IDs')
    check(floor==1e-6 and all(math.isfinite(x) and x>=floor for x in scale) and all(x>0 for x in count),
          'supported frozen scaler')
    check(all(not floors[i] or scale[i]==floor for i in range(9)),'scale floor mask')
    raw=b''
    for key,enum in (('mean',7),('scale',7),('count',4),('channel_ids',4),('floor_applied',11)):
        value=archive[key];raw+=struct.pack('<q',enum)+b''.join(struct.pack('<q',d) for d in value['shape'])+bytes_of(value)
    identity='rpb-scaler-v1-fnv1a64-'+fnv(raw+struct.pack('<d',floor))
    return mean,scale,identity


def uniform_index(engine,count):
    check(type(count) is int and count>0, 'positive C++ uniform range')
    threshold = ((-count)&MASK64)%count
    while True:
        product = engine.draw()*count
        if (product&MASK64)>=threshold:
            return product>>64


def bootstrap_groups(groups,seed,replicates):
    check(replicates==1000, 'declared within-master source bootstrap budget')
    if not groups:return {'estimate':None,'lower':None,'upper':None,'source_groups':0,'replicates':replicates}
    if len(groups)==1:return {'estimate':groups[0][0]/groups[0][1],'lower':None,'upper':None,
                             'source_groups':1,'replicates':replicates}
    engine = MT19937_64(seed)
    samples = []
    for _ in range(replicates):
        draw = [groups[uniform_index(engine,len(groups))] for _ in groups]
        numerator = math.fsum(x[0] for x in draw)
        denominator = sum(x[1] for x in draw)
        check(denominator>0 and math.isfinite(numerator), 'finite whole-pair bootstrap sample')
        samples.append(numerator/denominator)
    samples.sort()
    return {'estimate':math.fsum(x[0] for x in groups)/sum(x[1] for x in groups),
            'lower':samples[int(.025*(replicates-1))],'upper':samples[int(.975*(replicates-1))],
            'source_groups':len(groups),'replicates':replicates}


def map_statistics(rows,valid,mean,scale,floor=1e-8):
    selected = [row for row,keep in zip(rows,valid) if keep]
    check(len(selected)>=2, 'TRAIN-only map support')
    for column in range(len(mean)):
        actual = math.fsum(row[column] for row in selected)/len(selected)
        std = math.sqrt(math.fsum((row[column]-actual)**2 for row in selected)/len(selected))
        close(actual,mean[column],'TRAIN-only map mean')
        close(max(floor,std),scale[column],'TRAIN-only population map scale')


def feature_archive(archive,rows,width,ids,labels,expected_dtype=None):
    check(set(archive)=={'features','valid','labels_scoring_only','source_ids_json','provenance'}, 'exact saved feature schema')
    dtype = archive['features']['dtype']
    check(dtype in ('FloatStorage','DoubleStorage') and (expected_dtype is None or dtype==expected_dtype), 'declared feature dtype')
    values = tensor(archive['features'],dtype,[rows,width])
    valid = list(tensor(archive['valid'],'BoolStorage',[rows]))
    finite(values,'finite saved features')
    check(json.loads(text(archive,'source_ids_json'))==ids and
          list(tensor(archive['labels_scoring_only'],'LongStorage',[rows]))==list(labels) and bool(text(archive,'provenance')),
          'explicit feature row/source/label/provenance association')
    check(all(valid[row] or all(x==0. for x in values[row*width:(row+1)*width]) for row in range(rows)),
          'invalid exported rows have zero payload without invented support')
    return [list(values[i*width:(i+1)*width]) for i in range(rows)],valid


def check_saved_score(actual,declared):
    for key in ('total','valid','correct'):
        check(actual[key]==declared[key], 'exact declared score denominator '+key)
    for key in ('accuracy','coverage'):
        if actual[key] is None:check(declared[key] is None,'undefined conditional score retained')
        else:close(actual[key],declared[key], 'declared saved score '+key)
    close(actual['correct']/actual['total'],declared['full_population_correctness'], 'full population correctness')


def check_interval(actual,declared):
    check(declared['replicates']==1000 and declared['confidence']==.95 and
          declared['source_groups']==actual['source_groups'], 'conditional interval population/recipe')
    for key in ('estimate','lower','upper'):
        if actual[key] is None:check(declared[key] is None,'undefined source interval retained')
        else:close(actual[key],declared[key], 'whole-source bootstrap '+key)


def infer_saved_fit(fit,features,valid,labels,ids,prediction,prepared,width,fitted_rows=256):
    check(set(prediction)=={'ridge','tiny_secondary','valid','probe_input_features','ridge_logits',
          'tiny_hidden_preactivation','tiny_logits','labels_scoring_only','source_ids_json'},
          'exact actual saved prediction schema')
    check(not any('pca' in key for key in fit), 'no post-encoder or hidden per-head PCA state')
    check(scalar(fit,'outer_normalizer_applied','BoolStorage')==int(not prepared) and
          scalar(fit,'outer_fitted_rows')==(0 if prepared else fitted_rows) and scalar(fit,'fitted_rows')==fitted_rows and
          scalar(fit,'ridge_penalty','DoubleStorage')==1 and scalar(fit,'tiny_hidden')==16 and
          scalar(fit,'tiny_steps')==100 and scalar(fit,'tiny_learning_rate','DoubleStorage')==.01,
          'fixed original TRAIN maps and probe budgets')
    outer_mean=tensor(fit['feature_mean'],'DoubleStorage',[width])
    outer_scale=tensor(fit['feature_scale'],'DoubleStorage',[width])
    if prepared:
        check(not any(outer_mean) and all(x==1 for x in outer_scale), 'explicit absent second outer normalizer')
    input_rows=normalize(features,valid,outer_mean,outer_scale)
    ridge_mean=tensor(fit['ridge_mean'],'DoubleStorage',[width]);ridge_scale=tensor(fit['ridge_scale'],'DoubleStorage',[width])
    tiny_mean=tensor(fit['tiny_mean'],'DoubleStorage',[width]);tiny_scale=tensor(fit['tiny_scale'],'DoubleStorage',[width])
    ri=normalize(input_rows,valid,ridge_mean,ridge_scale);ti=normalize(input_rows,valid,tiny_mean,tiny_scale)
    weights=tensor(fit['ridge_weights'],'DoubleStorage',[width,2]);intercept=tensor(fit['ridge_intercept'],'DoubleStorage',[2])
    w1=tensor(fit['tiny_w1'],'DoubleStorage',[width,16]);b1=tensor(fit['tiny_b1'],'DoubleStorage',[16])
    w2=tensor(fit['tiny_w2'],'DoubleStorage',[16,2]);b2=tensor(fit['tiny_b2'],'DoubleStorage',[2])
    stored_input=tensor(prediction['probe_input_features'],'DoubleStorage',[len(features),width])
    stored_ridge_logits=tensor(prediction['ridge_logits'],'DoubleStorage',[len(features),2])
    stored_hidden=tensor(prediction['tiny_hidden_preactivation'],'DoubleStorage',[len(features),16])
    stored_tiny_logits=tensor(prediction['tiny_logits'],'DoubleStorage',[len(features),2])
    ridge=list(tensor(prediction['ridge'],'LongStorage',[len(features)]))
    tiny=list(tensor(prediction['tiny_secondary'],'LongStorage',[len(features)]))
    check(list(tensor(prediction['valid'],'BoolStorage',[len(features)]))==valid and
          json.loads(text(prediction,'source_ids_json'))==ids and
          list(tensor(prediction['labels_scoring_only'],'LongStorage',[len(features)]))==list(labels),
          'saved prediction exact row/support/source association')
    for row in range(len(features)):
        for j in range(width):close(input_rows[row][j],stored_input[row*width+j],'saved frozen probe input')
        hidden=[]
        for j in range(16):
            h=math.fsum(ti[row][d]*w1[d*16+j] for d in range(width))+b1[j]
            close(h,stored_hidden[row*16+j],'saved tanh preactivation')
            hidden.append(math.tanh(h))
        for j in range(2):
            r=math.fsum(ri[row][d]*weights[d*2+j] for d in range(width))+intercept[j]
            t=math.fsum(hidden[d]*w2[d*2+j] for d in range(16))+b2[j]
            close(r,stored_ridge_logits[row*2+j],'saved Ridge logit arithmetic')
            close(t,stored_tiny_logits[row*2+j],'saved Tiny logit arithmetic')
        check(ridge[row]==int(stored_ridge_logits[row*2+1]>stored_ridge_logits[row*2]) and
              tiny[row]==int(stored_tiny_logits[row*2+1]>stored_tiny_logits[row*2]),
              'exact saved-logit argmax including first-class ties')
    return ridge,tiny,input_rows


def control_rows(root, training, validation, deleted):
    values, mask, ids, labels = training
    asset = load(root / 'raw-scaler.pt')
    check(set(asset) == {'mean', 'scale', 'counts'}, 'one task TRAIN raw ObservationScaler')
    mean = tensor(asset['mean'], 'DoubleStorage', [3, 3])
    scale = tensor(asset['scale'], 'DoubleStorage', [3, 3])
    counts = tensor(asset['counts'], 'DoubleStorage', [3, 3])
    for c in range(3):
        for f in range(3):
            selected = [values[(b * 3 + c) * 96 + h * 3 + f] for b in range(256) for h in range(32)
                        if mask[(b * 3 + c) * 96 + h * 3 + f]]
            d = c * 3 + f
            check(counts[d] == len(selected), 'ordinary raw observed TRAIN counts')
            avg = math.fsum(selected) / len(selected) if selected else 0.
            std = math.sqrt(math.fsum((x - avg) ** 2 for x in selected) / len(selected)) if selected else 0.
            close(mean[d], avg, 'ordinary raw observed TRAIN mean')
            close(scale[d], max(1e-8, std), 'ordinary raw population scale/floor')
    unprepared, masks, valids = [], [], []
    for split in (training, validation, deleted):
        data, support, _, _ = split
        rows, flags, valid = [], [], []
        for b in range(len(split[2])):
            observed = list(support[b * 288:(b + 1) * 288])
            rows.append([(data[b * 288 + j] - mean[(j // 96) * 3 + j % 3]) / scale[(j // 96) * 3 + j % 3]
                         if observed[j] else 0. for j in range(288)] + [float(x) for x in observed])
            flags.append([float(x) for x in observed]); valid.append(any(observed))
        unprepared.append(rows); masks.append(flags); valids.append(valid)
    outer = load(root / 'raw-outer-normalizer.pt')
    check(set(outer) == {'feature_mean', 'feature_scale', 'fitted_rows', 'fit_count', 'training_source_ids_json'},
          'one shared driver raw TRAIN outer map')
    fm = tensor(outer['feature_mean'], 'DoubleStorage', [576])
    fs = tensor(outer['feature_scale'], 'DoubleStorage', [576])
    check(scalar(outer, 'fitted_rows') == sum(valids[0]), 'one shared raw TRAIN fit population')
    check(scalar(outer, 'fit_count') == 1 and json.loads(text(outer, 'training_source_ids_json')) == ids, 'one exact TRAIN raw outer fit and source order')
    check(len(ids) == 256, 'one original TRAIN map from the bound 256-row source order; constructor/count is source-bound')
    map_statistics(unprepared[0], valids[0], fm, fs)
    prepared = [normalize(rows, keep, fm, fs) for rows, keep in zip(unprepared, valids)]
    result = {'raw': prepared, 'mask_metadata': masks, 'pca_only': [], '_raw_unprepared': unprepared}
    status = read_json(root / 'pca-status.json')
    check(status['width'] == 32 and not status['native_PCA'], 'raw standalone PCA only, no native PCA')
    if status['status'] == 'unsupported_fit':
        check(not (root / 'pca-preprocessing.pt').exists() and isinstance(status['reason'], str) and
              any(x in status['reason'] for x in ('PCA dimensions exceed numerical training rank',
                   'PCA dimensions exceed valid centered training-row bound', 'feature fit requires two valid training rows')),
              'explicit rank/support failure; no invented PCA map')
        result['pca_only'] = [[[0.] * 32 for _ in rows] for rows in prepared]
        return result, status
    check(status == {'status': 'measured', 'width': 32, 'native_PCA': False}, 'measured standalone PCA status')
    pc = load(root / 'pca-preprocessing.pt')
    check(set(pc) == {'feature_mean', 'feature_scale', 'fitted_rows', 'pca_mean', 'pca_components',
                     'pca_singular_values', 'pca_numerical_rank'}, 'PCA writer schema, no second raw normalizer')
    for key in ('feature_mean', 'feature_scale', 'fitted_rows'):
        exact(pc[key], outer[key], 'PCA shares byte-exact one driver raw map ' + key)
    n = sum(valids[0]); pm = tensor(pc['pca_mean'], 'DoubleStorage', [576])
    components = tensor(pc['pca_components'], 'DoubleStorage', [576, 32])
    singular = tensor(pc['pca_singular_values'], 'DoubleStorage', [min(n, 576)])
    finite(components, 'finite PCA columns'); finite(singular, 'finite PCA spectrum')
    check(all(singular[i] >= singular[i + 1] >= 0 for i in range(len(singular) - 1)) and singular[0] > 0,
          'ordered PCA spectrum')
    rank = sum(x > 576 * sys.float_info.epsilon * singular[0] for x in singular)
    check(scalar(pc, 'pca_numerical_rank') == rank and rank >= 32, 'supported explicit TRAIN numerical rank')
    for d in range(576):
        close(pm[d], math.fsum(row[d] for row, keep in zip(prepared[0], valids[0]) if keep) / n,
              'PCA mean uses shared prepared TRAIN only')
    for i in range(32):
        for j in range(i + 1):
            close(math.fsum(components[d * 32 + i] * components[d * 32 + j] for d in range(576)), float(i == j),
                  'PCA orthonormal columns')
    for rows, keep in zip(prepared, valids):
        result['pca_only'].append([[math.fsum((row[d] - pm[d]) * components[d * 32 + j] for d in range(576))
                                    for j in range(32)] if ok else [0.] * 32 for row, ok in zip(rows, keep)])
    pca_covariance_replay([row for row, ok in zip(unprepared[0], valids[0]) if ok], pc)
    return result, status


def accuracy_groups(predictions,labels,valid,ids):
    groups={}
    for i,source in enumerate(ids):
        if valid[i]:
            entry=groups.setdefault(source,[0,0]);entry[0]+=int(predictions[i]==labels[i]);entry[1]+=1
    return [groups[key] for key in sorted(groups)]


def fake(dtype,shape,values):
    fmt={'FloatStorage':'f','DoubleStorage':'d','LongStorage':'q','BoolStorage':'B','ByteStorage':'B'}[dtype]
    value=array.array(fmt,values)
    check(len(value)==math.prod(shape),'source fixture tensor shape')
    return {'dtype':dtype,'shape':list(shape),'values':value,'raw':value.tobytes()}


def fake_text(value):
    raw=value.encode('utf-8')
    return fake('ByteStorage',[len(raw)],raw)


def close(left,right,message,tolerance=None):
    # The prospectively frozen absolute+relative F64 arithmetic bound never
    # determines class decisions; saved logits own their exact tie argmax.
    check(math.isfinite(left) and math.isfinite(right) and abs(left-right)<=ATOL+RTOL*abs(right),message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path=Path(path)
    check(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve(strict=True)==path and
          path.stat().st_nlink==1,'direct regular metadata path before JSON content access')
    def pairs(items):
        value={}
        for key,item in items:
            check(key not in value,'unique JSON key')
            value[key]=item
        return value
    def invalid(value):
        raise AssertionError('nonfinite JSON literal '+value)
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=invalid)


def pca_covariance_replay(raw_rows,asset):
    fm=tensor(asset['feature_mean'],'DoubleStorage',[576]);fs=tensor(asset['feature_scale'],'DoubleStorage',[576])
    centered=normalize(raw_rows,[True]*len(raw_rows),fm,fs)
    pm=tensor(asset['pca_mean'],'DoubleStorage',[576]);components=tensor(asset['pca_components'],'DoubleStorage',[576,32])
    singular=tensor(asset['pca_singular_values'],'DoubleStorage',[min(len(raw_rows),576)])
    matrix=[[x-pm[d] for d,x in enumerate(row)] for row in centered]
    projected=[[math.fsum(row[d]*components[d*32+j] for d in range(576)) for j in range(32)] for row in matrix]
    for d in range(576):
        for j in range(32):
            lhs=math.fsum(row[d]*projection[j] for row,projection in zip(matrix,projected))
            rhs=singular[j]**2*components[d*32+j]
            close(lhs,rhs,'saved TRAIN PCA covariance eigenvector equation')
    return {'covariance_columns':32,'SVD_executed':False}


def population(valid,labels,ids):
    selected=[i for i,keep in enumerate(valid) if keep]
    return {'total_rows':len(ids),'valid_rows':len(selected),
            'class_valid_rows':[sum(labels[i]==j for i in selected) for j in (0,1)],
            'total_source_groups':len(set(ids)),'valid_source_groups':len({ids[i] for i in selected}),
            'coverage':len(selected)/len(ids)}


def fit_schema(fit,width,ids,rep,prepared,training,valid):
    expected={'feature_mean','feature_scale','outer_normalizer_applied','outer_fitted_rows','fitted_rows','ridge_mean','ridge_scale',
          'ridge_weights','ridge_intercept','tiny_mean','tiny_scale','tiny_w1','tiny_b1','tiny_w2','tiny_b2',
          'actual_probe_seed_decimal','training_source_ids_json','ridge_penalty','tiny_hidden','tiny_steps','tiny_learning_rate'}
    check(set(fit)==expected and text(fit,'actual_probe_seed_decimal')==str(stream_seed(rep,width))
          and json.loads(text(fit,'training_source_ids_json'))==ids,'exact new TRAIN-only head recipe and width-paired seeds')
    rows=sum(valid);check(scalar(fit,'fitted_rows')==rows,'exact fitting population')
    fm=tensor(fit['feature_mean'],'DoubleStorage',[width]);fs=tensor(fit['feature_scale'],'DoubleStorage',[width])
    if not prepared:map_statistics(training,valid,fm,fs)
    outer=normalize(training,valid,fm,fs)
    for name in ('ridge','tiny'):
        map_statistics(outer,valid,tensor(fit[name+'_mean'],'DoubleStorage',[width]),tensor(fit[name+'_scale'],'DoubleStorage',[width]))
    for key,value in fit.items():
        if value['dtype']=='DoubleStorage':finite(value['values'],'finite immutable fitted '+key)
    return rows


def finite_timer(value,scope):
    check(type(value) in (float,int) and math.isfinite(value) and value>=0,'finite nonnegative measured '+scope)
    return value


def logits(rows, weights, intercept):
    check(weights and all(len(row) == len(intercept) for row in weights), 'saved linear tensor geometry')
    check(all(len(row) == len(weights) for row in rows), 'saved feature width')
    return [[math.fsum(row[k] * weights[k][c] for k in range(len(weights))) + intercept[c]
             for c in range(len(intercept))] for row in rows]


def own_argmax(rows):
    check(all(len(x) == 2 and all(math.isfinite(v) for v in x) for x in rows), 'saved binary logits')
    return [0 if x[0] >= x[1] else 1 for x in rows]


def query_masks(observed, rows, channels=3, history=32, features=3, patch=8):
    check(history > 0 and history % patch == 0 and len(observed) == rows * channels * history * features, 'query geometry')
    trials = history // patch
    def index(b, c, h, f):
        return ((b * channels + c) * history + h) * features + f
    requested, visible, query, eligible = [], [], [], []
    for trial in range(trials):
        flags = []
        for b in range(rows):
            for c in range(channels):
                groups = [any(observed[index(b, c, h, f)] for h in range(p * patch, (p + 1) * patch)
                              for f in range(features)) for p in range(trials)]
                flags.append(groups[trial] and sum(groups) - int(groups[trial]) >= 2)
        eligible.extend(flags)
        for b in range(rows):
            for c in range(channels):
                for h in range(history):
                    for f in range(features):
                        original = bool(observed[index(b, c, h, f)])
                        requested.append(original and h // patch == trial)
                        hidden = h // patch == trial and flags[b * channels + c]
                        visible.append(original and not hidden)
                        query.append(original and hidden)
    return requested, visible, query, eligible


def reductions(prediction, target, query, rows, channels=3, history=32, features=3, patch=8):
    trials = history // patch
    check(len(prediction) == len(target) == len(query) == trials * rows * channels * history * features, 'saved query tensor geometry')
    sums = [0.] * (rows * channels)
    huber = [0.] * (rows * channels)
    counts = [0] * (rows * channels)
    for t in range(trials):
        for b in range(rows):
            for c in range(channels):
                for h in range(history):
                    for f in range(features):
                        i = ((((t * rows + b) * channels + c) * history + h) * features + f)
                        check(math.isfinite(prediction[i]) and math.isfinite(target[i]), 'finite query arrays')
                        if query[i]:
                            e = abs(prediction[i] - target[i])
                            sums[b * channels + c] += e
                            huber[b * channels + c] += .5 * e * e if e <= 1 else e - .5
                            counts[b * channels + c] += 1
                        else:
                            check(prediction[i] == target[i] == 0., 'zero unqueried saved values')
    cm = [s / n if n else 0. for s, n in zip(sums, counts)]
    ch = [s / n if n else 0. for s, n in zip(huber, counts)]
    em, eh, ev = [], [], []
    for b in range(rows):
        supported = [c for c in range(channels) if counts[b * channels + c]]
        ev.append(bool(supported))
        em.append(math.fsum(cm[b * channels + c] for c in supported) / len(supported) if supported else 0.)
        eh.append(math.fsum(ch[b * channels + c] for c in supported) / len(supported) if supported else 0.)
    n = sum(ev)
    return {'counts': counts, 'channel_mae': cm, 'channel_huber': ch, 'example_valid': ev,
            'example_mae': em, 'example_huber': eh,
            'mae': math.fsum(x for x, v in zip(em, ev) if v) / n if n else None,
            'huber': math.fsum(x for x, v in zip(eh, ev) if v) / n if n else None}


def admitted_relative_names(names):
    result = []
    for value in names:
        check(isinstance(value, str) and '\\' not in value and value, 'relative POSIX source/artifact name')
        path = PurePosixPath(value)
        check(not path.is_absolute() and '..' not in path.parts and str(path) == value and ':' not in value, 'bounded artifact path')
        result.append(value)
    check(len(set(result)) == len(result), 'unique complete artifact names')
    return result


def import_helper(repo):
    global R
    source = repo / HELPER
    check(source.is_file() and sha(source) == HELPER_SHA, 'exact pinned codec SOURCE before import')
    spec = importlib.util.spec_from_file_location('early_mixer_cpu_codec', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    R = module
    return module


def observations(value, rows, master, task, deleted=False):
    keys = {'observations', 'feature_mask', 'labels_scoring_only', 'source_ids_json'}
    if deleted:
        keys.add('requested_erasure')
    check(set(value) == keys, 'actual controlled observation literal key schema')
    data = tensor(value['observations'], 'DoubleStorage', [rows, 3, 32, 3])
    mask = tensor(value['feature_mask'], 'BoolStorage', [rows, 3, 32, 3])
    labels = list(tensor(value['labels_scoring_only'], 'LongStorage', [rows]))
    ids = json.loads(text(value, 'source_ids_json'))
    check(len(ids) == rows and len(set(ids)) == rows // 2 and
          all(isinstance(x, str) and re.fullmatch(re.escape(PROTOCOL) + f'/{task}/seed-{master}/{task}/source-[0-9]+', x) for x in ids),
          'new task/master/source namespace, not historical instances')
    finite(data, 'finite legal controlled values')
    check(all(m or x == 0. for x, m in zip(data, mask)), 'zero hidden observation storage')
    validate_pairs(labels, ids, [list(mask[i * 288:(i + 1) * 288]) for i in range(rows)])
    return data, mask, ids, labels


def coordinate_erasure(ids, task, master):
    namespace = PROTOCOL + '/' + task + '/validation-coordinate-dropout'
    actual = stream_seed(master, VIEW_STREAMS[task])
    result = []
    for source in ids:
        key = namespace + '/' + task + '/coordinate/' + str(len(source.encode('utf-8'))) + ':' + source
        engine = MT19937_64(stream_seed(actual, fnv64(key)))
        result.append([(engine.draw() >> 11) * (2.0 ** -53) < .30 for _ in range(288)])
    return result


def check_view(base, deleted, master, task, value):
    data, mask, ids, labels = base
    dv, dm, di, dl = deleted
    check(di == ids and dl == labels, 'deleted view preserves exact row/source/labels')
    requested = list(tensor(value['requested_erasure'], 'BoolStorage', [128, 3, 32, 3]))
    expected = [x for row in coordinate_erasure(ids, task, master) for x in row]
    check(requested == expected, 'one source-pair-shared MT64 erasure law, no mask repair')
    wanted, support = deletion_view(data, mask, requested)
    check(list(dv) == wanted and list(dm) == support, 'deletion-only values and support')
    return {'rate': .30, 'namespace': PROTOCOL + '/' + task + '/validation-coordinate-dropout',
            'actual_seed_decimal': str(stream_seed(master, VIEW_STREAMS[task])),
            'requested_coordinates': sum(requested), 'observed_coordinates_before': sum(mask),
            'observed_coordinates_after': sum(dm)}


def named_group(value, expected_total=None):
    value = group(value)
    count = scalar(value, 'count')
    check(count >= 0 and set(value) == {'count'} | {f'tensor_{i}' for i in range(count)}, 'complete numbered named tensor group')
    result = {}
    for i in range(count):
        item = group(value[f'tensor_{i}'])
        check(set(item) == {'parameter_name', 'value'}, 'named tensor item schema')
        name = text(item, 'parameter_name')
        tensor_value = item['value']
        check(name and name not in result and tensor_value['dtype'] in ('FloatStorage', 'DoubleStorage', 'LongStorage', 'BoolStorage'),
              'unique CPU named tensor type')
        check(len(tensor_value['values']) == math.prod(tensor_value['shape']), 'named tensor geometry')
        finite(tensor_value['values'], 'finite captured named tensor values')
        result[name] = tensor_value
    if expected_total is not None:
        check(sum(len(x['values']) for x in result.values()) == expected_total, 'fixed parameter total')
    return result


def same_named(left, right, message):
    check(set(left) == set(right), message + ': names')
    for name in left:
        exact(left[name], right[name], message + ': ' + name)


def initial_witness(asset):
    check(set(asset) == {'late_parameters', 'early_parameters', 'late_buffers', 'early_buffers', 'late_scaler', 'early_scaler'},
          'complete paired initialization writer schema')
    late = named_group(asset['late_parameters'], 225805)
    early = named_group(asset['early_parameters'], 225805)
    lb, eb = named_group(asset['late_buffers']), named_group(asset['early_buffers'])
    check(list(late)==list(early) and list(lb)==list(eb),'same paired parameter/buffer registration order')
    same_named(late, early, 'same registered initial parameters')
    same_named(lb, eb, 'same initial buffers')
    ls, es = group(asset['late_scaler']), group(asset['early_scaler'])
    check(set(ls) == set(es), 'paired initialization scaler schema')
    for key in ls:
        exact(ls[key], es[key], 'paired initialization scaler ' + key)
    _, _, identity = scaler(ls)
    return late, lb, ls, identity


def parse_settings(value, placement, master):
    result = {}
    for line in value.splitlines():
        check('=' in line, 'canonical settings line')
        key, val = line.split('=', 1)
        check(key not in result, 'unique resolved settings')
        result[key] = val
    expected = {'channel_count': 3, 'history_length': 32, 'input_width': 3, 'patch_length': 8,
                'encoder_width': 64, 'export_width': 32, 'num_layers': 3, 'num_heads': 4,
                'feedforward_width': 256, 'decoder_hidden_width': 128, 'channel_mixer_layers': 1,
                'global_bottleneck_mode': 2, 'dropout': 0, 'huber_delta': 1, 'sampling_interval': 1,
                'batch_size': 8, 'threads': 1, 'learning_rate': .001, 'weight_decay': .0001,
                'gradient_clip_norm': 1, 'steps': 512, 'attempt_limit': 1024, 'log_every': 1, 'seed': master,
                'layer_norm_epsilon': 1e-5, 'mask_ratio': .25, 'scale_floor': 1e-6}
    for key, val in expected.items():
        check(key in result and float(result[key]) == val, 'frozen resolved setting ' + key)
    check(result['device'] in ('cuda', 'cuda:0'), 'actual CUDA setting, no CPU model')
    check(result['channel_ids'] in ('', '0,1,2'), 'configured canonical channel IDs; empty resolves to range(C)')
    check(('channel_mixer_placement' not in result) if placement == 0 else result.get('channel_mixer_placement') == '1',
          'default placement omitted, explicit nondefault placement')
    check('global_pool_input_source' not in result,'unchanged projected-D global route; default field omitted')
    common = dict(result)
    common.pop('channel_mixer_placement', None)
    common['device'] = 'cuda'
    return common


def parent_audit(audit, master, placement, point, ids):
    check(text(audit, 'artifact_kind') == 'rpb_learning_curve_training_audit_v1' and
          text(audit, 'protocol_id') == FIT_PROTOCOL and text(audit, 'actual_training_seed') == str(master) and
          text(audit, 'initialization_seed') == str(mixed(master ^ 0x7270622d696e6974)) and
          text(audit, 'fit_source_manifest') == source_manifest(ids), 'exact label-free fit namespace/data/init seed')
    check(text(audit, 'model_tag') == ('RPB-v10' if placement else 'RPB-v7') and
          text(audit, 'architecture_id') == ARCHITECTURES[placement] and
          text(audit, 'channel_mixer_placement') == str(placement) and
          scalar(audit, 'channel_mixer_placement_value') == placement and
          text(audit, 'output_semantics') == OUTPUT_SEMANTICS[placement] and
          text(audit, 'reconstruction_export_semantics') == RECONSTRUCTION_SEMANTICS[placement], 'typed placement and architecture semantics')
    check(scalar(audit, 'attempted_steps') == scalar(audit, 'completed_steps') == point and
          scalar(audit, 'sampled_rows') == point * 8 and text(audit, 'model_weight_update_budget') == str(point),
          'absolute unskipped point counters')
    check(list(tensor(audit['channel_order'], 'LongStorage', [3])) == [0, 1, 2] and
          scalar(audit, 'sampling_interval', 'DoubleStorage') == 1 and scalar(audit, 'endpoint', 'DoubleStorage') == 31,
          'typed canonical semantic/time metadata')
    for key, expected in CONTEXT_LITERALS.items():
        check(text(audit, key) == expected, 'unchanged coordinate15 policy ' + key)
    check(scalar(audit, 'context_deletion_ratio_value', 'DoubleStorage') == .15 and
          scalar(audit, 'context_deletion_stream_value') == 0x6374782d64726f70, 'typed context recipe')
    check(not any(key.startswith('context_deletion_schedule') or key in ('context_ordinary_attempts', 'context_deletion_attempts') for key in audit),
          'no balanced or alternative recipe')
    check(not any(key.startswith('source_gain_') or key.startswith('pooled_') or key=='global_pool_input_source_value' for key in audit),
          'delegated ordinary coordinate15 audit has no gain or pooled-route additions')
    counts = [scalar(audit, key) for key in ('context_requested_deleted_coordinates', 'context_actual_deleted_coordinates', 'context_restored_coordinates')]
    check(0 <= counts[1] <= counts[0] <= point * 8 * 288 and counts[2] == counts[0] - counts[1], 'context counts requested/actual/restored')
    seconds = scalar(audit, 'training_seconds', 'DoubleStorage')
    finite_timer(seconds, 'CUDA update loop')
    check(scalar(audit, 'weights_changed', 'BoolStorage') == bool(point) and
          scalar(audit, 'finite_gradients', 'BoolStorage') == bool(point) and
          ((seconds == 0) if point == 0 else (seconds > 0)), 'actual training/point0 scalar witnesses')
    check(text(audit, 'rng_policy') == COUNTER_POLICY and
          text(audit, 'sampling_policy') == 'with_replacement_counter_rows;sampled_rows_includes_no_update_attempts' and
          text(audit, 'optimizer_policy') == 'one_continuous_AdamW_state;absolute_completed_update_budgets', 'unchanged absolute sampling/Torch/optimizer streams')
    common = parse_settings(text(audit, 'resolved_settings'), placement, master)
    return {'counts': counts, 'training_seconds': seconds, 'common_settings': common,
            'dataset_id': text(audit, 'training_dataset_id'), 'scaler_id': text(audit, 'preprocessing_id'),
            'core_source': text(audit, 'core_writer_source_fingerprint'),
            'training_source': text(audit, 'training_producer_source_fingerprint'),'parameter_count':225805}


def progress(value, audit_info):
    check(value['attempted'] == value['completed'] == 512 and value['sampled_rows'] == 4096 and
          value['parameter_count'] == value['cuda_parameter_count'] == 225805 and
          value['training_device'] in ('cuda', 'cuda:0') and
          all(value[k] is True for k in ('last_input_cuda', 'last_loss_cuda', 'finite_gradients', 'weights_changed')),
          'complete actual CUDA encoder trajectory')
    close(value['training_seconds'], audit_info['training_seconds'], 'same saved update-loop timer')
    check(value['training_dataset_id'] == audit_info['dataset_id'] and value['preprocessing_id'] == audit_info['scaler_id'], 'progress fit/scaler association')
    trace = value['losses']
    check(len(trace) == 512, 'all 512 trace entries, not selected rows')
    for i, row in enumerate(trace):
        check(len(row) == 5 and row[0] == row[1] == i + 1 and type(row[2]) is int and row[2] > 0 and
              math.isfinite(row[3]) and row[3] >= 0 and math.isfinite(row[4]) and row[4] >= 0, 'ordered exact trace prefix/targets/finite loss and gradient')
    return trace


def query_archive(asset, split, frozen_scaler):
    data, mask, ids, labels = split
    rows = len(ids)
    expected = {'standardized_prediction', 'standardized_target', 'target_mask', 'requested_observed_target_mask',
                'visible_mask', 'trial_channel_eligible', 'channel_target_counts', 'channel_valid',
                'channel_standardized_mae', 'channel_standardized_huber', 'example_valid', 'example_standardized_mae',
                'example_standardized_huber', 'source_ids_json'}
    check(set(asset) == expected, 'unchanged fixed query archive schema')
    shape = [4, rows, 3, 32, 3]
    p = tensor(asset['standardized_prediction'], 'DoubleStorage', shape)
    t = tensor(asset['standardized_target'], 'DoubleStorage', shape)
    q = list(tensor(asset['target_mask'], 'BoolStorage', shape))
    rq = list(tensor(asset['requested_observed_target_mask'], 'BoolStorage', shape))
    v = list(tensor(asset['visible_mask'], 'BoolStorage', shape))
    e = list(tensor(asset['trial_channel_eligible'], 'BoolStorage', [4, rows, 3]))
    wanted_rq, wanted_v, wanted_q, wanted_e = query_masks(list(mask), rows)
    check(rq == wanted_rq and v == wanted_v and q == wanted_q and e == wanted_e, 'ordinary original-Q visibility/eligibility/support, no E augmentation')
    check(json.loads(text(asset, 'source_ids_json')) == ids, 'fixed query exact source order')
    mean, scale, _ = scaler(frozen_scaler)
    for i, keep in enumerate(q):
        if keep:
            original = i % (rows * 288)
            channel = (original // 96) % 3
            feature = original % 3
            want = f32((data[original] - mean[channel * 3 + feature]) / scale[channel * 3 + feature])
            check(abs(t[i] - want) <= F32_ATOL + F32_RTOL * abs(want), 'frozen TRAIN scaler/F32 target arithmetic')
    result = reductions(p, t, q, rows)
    shapes = {'channel_target_counts': ('LongStorage', [rows, 3], result['counts']),
              'channel_valid': ('BoolStorage', [rows, 3], [bool(x) for x in result['counts']]),
              'channel_standardized_mae': ('DoubleStorage', [rows, 3], result['channel_mae']),
              'channel_standardized_huber': ('DoubleStorage', [rows, 3], result['channel_huber']),
              'example_valid': ('BoolStorage', [rows], result['example_valid']),
              'example_standardized_mae': ('DoubleStorage', [rows], result['example_mae']),
              'example_standardized_huber': ('DoubleStorage', [rows], result['example_huber'])}
    for key, (dtype, dims, values) in shapes.items():
        saved = tensor(asset[key], dtype, dims)
        if dtype in ('BoolStorage', 'LongStorage'):
            check(list(saved) == values, 'exact fixed query counts/support ' + key)
        else:
            for actual, wanted in zip(saved, values):
                close(actual, wanted, 'hierarchical query reduction ' + key)
    result['requested_observed_target_cells'] = sum(rq)
    result['valid_target_cells'] = sum(result['counts'])
    result['valid_examples'] = sum(result['example_valid'])
    result['total_examples'] = rows
    return result


def audit_readouts(directory, data, controls, pca_status, provider_records, master, task):
    train, val, deleted = data; root = directory / 'readouts'; report = read_json(root / 'report.json')
    methods = methods_for_task(task)
    check(report['protocol'] == 'fixed-feature-readouts-v1' and report['master_seed'] == str(master), 'actual fixed readout writer')
    check(report['recipe'] == {'ridge_penalty': 1, 'tiny_hidden': 16, 'tiny_steps': 100, 'tiny_learning_rate': .01,
          'probe_seed_policy': 'stream_seed(repetition,width);same_equal_width_methods', 'validation_fits': 0,
          'encoder_calls': 0, 'pca_fits': 0}, 'unchanged fixed heads, no view or encoder fitting')
    check([x['method'] for x in report['methods']] == list(methods), 'all unique task methods, controls fit once')
    stored = {}; summaries = []; fits = outer_fits = 0
    for item in report['methods']:
        name = item['method']; width = methods[name]; prepared = name in ('raw', 'pca_only'); path = root / name
        check(item['size'] == width and item['inputs_train_prepared'] is prepared,
              'prepared raw/PCA bypass only; native outer stage unchanged')
        surfaces = []
        for view, split, popkey in zip(VIEWS, data, ('training_population', 'validation_intact_population', 'validation_deleted_population')):
            feature = load(path / (view + '-features.pt'))
            rows, valid = feature_archive(feature, len(split[2]), width, split[2], split[3],
                                         'DoubleStorage' if name in controls else 'FloatStorage')
            support = [any(split[1][b * 288:(b + 1) * 288]) for b in range(len(split[2]))]
            if name in controls:
                check(valid == support, 'raw/PCA/mask observation-any support independent of native')
                for a, b in zip(rows, controls[name][len(surfaces)]):
                    for x, y in zip(a, b): close(x, y, 'saved prepared legal control ' + name)
            else:
                check(all(not ok or support[b] for b, ok in enumerate(valid)), 'native support subset of original observations')
                check(text(feature, 'provenance') == provider_records[name]['extractor_provenance'], 'same retained immutable CUDA surface')
            check(item[popkey] == population(valid, split[3], split[2]), 'explicit per-method view support')
            surfaces.append((rows, valid))
        rows, valid = surfaces[0]
        supported = sum(valid) >= 2 and {train[3][i] for i, ok in enumerate(valid) if ok} == {0, 1}
        if name == 'pca_only' and pca_status['status'] == 'unsupported_fit': supported = False
        check(item['status'] == ('measured' if supported else 'unsupported_fit'), 'independent ordinary TRAIN fit support')
        count = {'outer_train_normalizer_fits': int(supported and not prepared), 'ridge_fits': 3 if supported else 0,
                 'tiny_fits': 3 if supported else 0, 'validation_fits': 0}
        check(read_json(path / 'fit-counts.json') == count and item['outer_train_normalizer_fits'] == count['outer_train_normalizer_fits'],
              'one native/mask outer map; driver prepared raw/PCA not refitted')
        outer_fits += count['outer_train_normalizer_fits']
        method_summary = {'method': name, 'size': width, 'status': item['status'], 'repetitions': []}
        check(len(item['repetitions']) == 3, 'three fixed head repetitions including unsupported')
        for rep, item_rep in zip(HEAD_REPETITIONS, item['repetitions']):
            repid = f'rep-{rep}'; repdir = path / repid
            check(item_rep['id'] == repid and item_rep['status'] == item['status'] and
                  item_rep['actual_probe_seed_decimal'] == str(stream_seed(rep, width)) and
                  item_rep['ridge_parameters'] == 2 * width + 2 and item_rep['neural_parameters'] == 16 * (width + 3) + 2,
                  'actual width-paired seed and fixed head parameter counts')
            if not supported:
                check(not repdir.exists() and not any(k in item_rep for k in ('fit_artifact', 'training', 'validation_intact', 'validation_deleted')),
                      'unsupported fit invents no asset/predictions')
                continue
            fits += 1; fit = load(repdir / 'fit.pt'); fitted = fit_schema(fit, width, train[2], rep, prepared, rows, valid)
            check(item_rep['fit_artifact'] == f'{name}/{repid}/fit.pt', 'one ordinary TRAIN asset per unique pipeline')
            predictions = []; summary = {'repetition': repid, 'actual_probe_seed_decimal': str(stream_seed(rep, width))}
            for view, split, surface, key in zip(VIEWS, data, surfaces, ('training', 'validation_intact', 'validation_deleted')):
                feature_rows, keep = surface; prediction = load(repdir / (view + '-predictions.pt'))
                ridge, tiny, _ = infer_saved_fit(fit, feature_rows, keep, split[3], split[2], prediction, prepared, width, fitted)
                declared = item_rep[key]
                check(declared['population'] == population(keep, split[3], split[2]), 'saved prediction support/source denominator')
                scores = {'ridge': score(ridge, split[3], keep), 'tiny_secondary': score(tiny, split[3], keep)}
                for head in scores: check_saved_score(scores[head], declared[head])
                if view != 'training':
                    seed = stream_seed(master, fnv64(name + '/' + repid + '/' + view))
                    check(declared['bootstrap_seed_decimal'] == str(seed), 'unchanged per-method interval seed')
                    for head, preds, field in [('ridge', ridge, 'ridge_grouped_interval'), ('tiny_secondary', tiny, 'tiny_grouped_interval')]:
                        ci = bootstrap_groups(accuracy_groups(preds, split[3], keep, split[2]), seed, 1000)
                        check_interval(ci, declared[field]); scores[head]['grouped_interval'] = ci
                summary[key] = scores; predictions.append((ridge, tiny, keep))
            stored[name, rep] = predictions; method_summary['repetitions'].append(summary)
        summaries.append(method_summary)
    check(report['fit_counts'] == {'outer_train_normalizer_fits': outer_fits, 'ridge_fits': fits, 'tiny_fits': fits,
                                  'validation_fits': 0, 'encoder_calls': 0, 'pca_fits': 0}, 'actual supported unique fit counts')
    budgets = (512,)
    expected_ids = {'native_early_minus_native_late'}
    check(len(report['pairs']) == len(budgets) * 6, 'one declared comparison per budget/rep/view')
    pairs = []
    for record in report['pairs']:
        check(record['id'] in expected_ids and record['repetition'] in [f'rep-{r}' for r in HEAD_REPETITIONS] and
              record['view'] in ('validation_intact', 'validation_deleted'), 'explicit budget-specific vector comparison')
        candidate = 'native_early'; reference = 'native_late'
        rep = int(record['repetition'][4:]); view = 1 if record['view'] == 'validation_intact' else 2
        if (candidate, rep) not in stored or (reference, rep) not in stored:
            check(record['status'] == 'unsupported_fit', 'unsupported pair remains explicit'); pairs.append(record); continue
        a, b = stored[candidate, rep][view], stored[reference, rep][view]
        common = [bool(x and y) for x, y in zip(a[2], b[2])]
        check(record['status'] == ('measured' if any(common) else 'unsupported_zero_common') and
              record['common_population'] == population(common, val[3], val[2]), 'per-budget common supported population')
        seed = stream_seed(master, fnv64(record['id'] + '/' + record['repetition'] + '/' + record['view']))
        check(record['bootstrap_seed_decimal'] == str(seed), 'generic unchanged pair bootstrap seed')
        for head, index in [('ridge', 0), ('tiny_secondary', 1)]:
            effect = paired_group_effect(a[index], b[index], val[3], a[2], b[2], val[2])
            check_interval(bootstrap_groups(effect['groups'], seed, 1000), record[head])
        pairs.append(record)
    check(len({(p['id'], p['repetition'], p['view']) for p in pairs}) == len(budgets) * 6,
          'all vector comparison records unique, no selected points/heads')
    return {'data_master': master, 'methods': summaries, 'pairs': pairs, 'fit_counts': report['fit_counts'],
            'driver_raw_outer_fits': 1}, report


def replay_parent(audit, raw, scaler_asset, controlled, master, placement, point, scopes):
    data, mask, ids, labels = controlled
    info = parent_audit(audit, master, placement, point, ids)
    check(info['core_source'] == scopes['core_writer'] and info['training_source'] == scopes['curve_training'],
          'actual captured core/continuous-trainer compile scopes')
    check(text(audit, 'source_fingerprint_scope') ==
          'ordinary_checkpoint_field=core_writer;companion_audit_training_producer=evaluation_source', 'parent source scope literal')
    check(text(raw, 'artifact_kind') == 'rpb_raw_uniform_history_v1' and scalar(raw, 'format_version') == 1,
          'typed legal timing raw companion')
    exact(raw['data'], fake('DoubleStorage', [256,3,32,3], data), 'raw fitting data exactly controlled timing TRAIN')
    exact(raw['observed'], fake('BoolStorage', [256,3,32,3], mask), 'raw fitting mask exactly controlled timing TRAIN')
    check(list(tensor(raw['channel_ids'], 'LongStorage', [3])) == [0,1,2] and
          list(tensor(raw['endpoints'], 'DoubleStorage', [256])) == [31.] * 256 and
          scalar(raw, 'sampling_interval', 'DoubleStorage') == 1., 'raw semantic/time metadata')
    check(text(raw,'feature_units') == text(audit,'feature_units') == 'unitless,unitless,unitless', 'same ordinary units')
    check(text(scaler_asset,'artifact_kind') == 'rpb_frozen_training_scaler_v1' and
          scalar(scaler_asset,'format_version') == 1, 'typed scaler companion')
    frozen = group(scaler_asset['scaler'])
    mean, scale, identity = scaler(frozen)
    check(identity == info['scaler_id'] == text(scaler_asset,'preprocessing_id'), 'exact frozen scaler identity')
    check(text(raw,'schema_id') == text(scaler_asset,'schema_id') and
          text(raw,'dataset_id') == text(scaler_asset,'fit_dataset_id') == info['dataset_id'] == text(audit,'scaler_fit_dataset_id'),
          'timing-only fitting schema/data association')
    count = tensor(frozen['count'],'LongStorage',[3,3]); floors = tensor(frozen['floor_applied'],'BoolStorage',[3,3])
    for c in range(3):
        for f in range(3):
            values = [data[b*288+c*96+h*3+f] for b in range(256) for h in range(32) if mask[b*288+c*96+h*3+f]]
            i = c*3+f
            check(len(values) == count[i] and values, 'ordinary observed TRAIN scaler count')
            wanted_mean = math.fsum(values)/len(values)
            sd = math.sqrt(math.fsum((x-wanted_mean)**2 for x in values)/len(values))
            close(mean[i],wanted_mean,'ordinary TRAIN scaler mean')
            close(scale[i],max(1e-6,sd),'ordinary TRAIN population scale')
            check(bool(floors[i]) == (sd<1e-6), 'frozen scaler floor condition')
    info.update({'schema_id':text(raw,'schema_id'),'frozen_scaler':frozen})
    return info


def snapshot_binding(asset, audit, info, cp, placement, point, scopes, initial):
    check(text(asset,'artifact_kind') == 'rpb_early_mixer_cuda_snapshot_v1', 'strict new immutable CUDA snapshot kind')
    expected = {
        'protocol_id':IMPLEMENTATION_PROTOCOL, 'original_training_protocol_id':FIT_PROTOCOL,
        'parent_checkpoint_path':str(cp), 'parent_writer_source_fingerprint':scopes['core_writer'],
        'parent_training_producer_source_fingerprint':scopes['curve_training'],
        'snapshot_loader_source_fingerprint':scopes['early_adapter'],
        'source_fingerprint_scope':'parent=core_writer_and_training_producer;loader=new_early_mixer_adapter',
        'original_encoder_attempted':str(point), 'original_encoder_completed':str(point),
        'training_schema_id':info['schema_id'], 'training_dataset_id':info['dataset_id'],
        'preprocessing_id':info['scaler_id'], 'parameter_count':'225805',
        'encoder_updates':'0','decoder_updates':'0','head_refits':'0','no_optimizer_created':'true',
        'snapshot_policy':'independent_CUDA_model;eval_no_grad;frozen_timing_TRAIN_scaler;ordinary_query_no_context_erasure;no_refit'
    }
    for key, value in expected.items():
        check(text(asset,key) == value, 'snapshot lineage/immutability '+key)
    check(text(asset,'inference_device') in ('cuda','cuda:0'), 'snapshot actual CUDA inference')
    for key in ('channel_mixer_placement_value','original_encoder_attempted_value','original_encoder_completed_value'):
        check(scalar(asset,key) == (placement if key.startswith('channel') else point), 'typed original snapshot '+key)
    overrides = set(expected) | {'artifact_kind'}
    for key,value in audit.items():
        if value.get('dtype') == 'ByteStorage' and key not in overrides:
            check(text(asset,key) == text(audit,key), 'snapshot preserves original typed audit text '+key)
    for key, value in zip(('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'),info['counts']):
        check(text(asset,key) == str(value), 'snapshot saved point context count '+key)
    for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt'):
        # Whole CUDA checkpoint bytes are hashed, never CPU-decoded.
        check(text(asset,'parent_content_id'+suffix) == 'fnv1a64-runtime-content-v1-'+fnv(Path(str(cp)+suffix).read_bytes()), 'same original point byte/FNV binding '+suffix)
    saved_scaler = group(asset['scaler'])
    check(set(saved_scaler) == set(info['frozen_scaler']), 'snapshot scaler schema')
    for key in saved_scaler: exact(saved_scaler[key],info['frozen_scaler'][key],'snapshot frozen TRAIN scaler '+key)
    params = named_group(asset['model_parameters'],225805); buffers = named_group(asset['model_buffers'])
    check(list(params) == sorted(params) and list(buffers) == sorted(buffers), 'snapshot saved named tensors are lexicographic')
    base_params, base_buffers, base_scaler, base_id = initial
    check(set(params) == set(base_params) and set(buffers) == set(base_buffers), 'unchanged registered modules/dimensions')
    for name in params:
        check(params[name]['dtype'] == base_params[name]['dtype'] and params[name]['shape'] == base_params[name]['shape'], 'same parameter architecture '+name)
    if point == 0:
        same_named(params,base_params,'snapshot exact paired initial parameters')
        same_named(buffers,base_buffers,'snapshot exact initial buffers')
    else:
        check(any(bytes_of(params[name]) != bytes_of(base_params[name]) for name in params), 'trained snapshot actually changes weights')
    for key in base_scaler: exact(saved_scaler[key],base_scaler[key],'all retained points keep ordinary TRAIN scaler '+key)
    base = ('RPB-v10' if placement else 'RPB-v7') + '; immutable timing-protocol checkpoint; exact contextual-global32 CUDA serving; ordinary original-Q decoder'
    return {'extractor_provenance':base+'; '+base,'placement':placement,'point':point,
            'architecture_id':ARCHITECTURES[placement],'scaler_identity':info['scaler_id'],
            'parent_checkpoint_sha256':sha(cp),'checkpoint_body_decoded':False,
            'actual_encoder_execution':'source/admission-bound; no CPU model rerun'}


def query_summary(saved, replay, asset, artifact):
    n = replay['total_examples']; valid = replay['valid_examples']; targets = replay['valid_target_cells']; requested = replay['requested_observed_target_cells']
    expected = {'status':'measured' if valid else 'unsupported_zero_support','units':'frozen training-scaler standardized',
                'total_examples':n,'valid_examples':valid,'valid_channels':sum(bool(x) for x in replay['counts']),
                'valid_target_cells':targets,'requested_observed_target_cells':requested,'artifact':artifact,
                'query_checksum_fnv1a64':fnv(bytes_of(asset['target_mask']))}
    for key,value in expected.items(): check(saved[key] == value,'saved query report '+key)
    for key,value in [('example_coverage',valid/n),('target_coverage',targets/requested if requested else None),
                      ('standardized_mae',replay['mae']),('standardized_huber',replay['huber'])]:
        if value is None: check(saved[key] is None,'unsupported query null '+key)
        else: close(saved[key],value,'reported hierarchical query '+key)


def fixed_plan(plan):
    expected={'protocol':PROTOCOL,'timing_master_seeds':list(TIMING_MASTERS),'amplitude_data_master_seeds':list(AMPLITUDE_MASTERS),
        'tags':['RPB-v7.alt-04','RPB-v10.alt-04'],'encoder_trajectories':10,'encoder_updates_each':512,
        'attempt_limit':1024,'log_every':1,'skipped_attempts_permitted':False,'retained_points':20,
        'decoder_updates':0,'extra_decoder_calibration_updates':0,
        'decoder_update_scope':'joint reconstruction during encoder updates; no extra decoder-only calibration',
        'batch_size':8,'sampled_rows':40960,'train_pairs':128,'validation_pairs':64,'test_pairs':0,'shape':[3,32,3],
        'native_width':32,'parameter_count_each':225805,'channel_mixer_placements':[0,1],'global_pool_input_sources':[0,0],
        'checkpoint_roles_per_point':5,'external_fit_protocol':EXTERNAL_FIT_PROTOCOL,'implementation_fit_protocol':FIT_PROTOCOL,
        'point_binding_suffix':'.confirmation.pt','head_repetitions':list(HEAD_REPETITIONS),'planned_pipelines':210,'planned_heads':420,
        'deletion_rate':.30,'methods_each_task':7,'separate_initial_controls':2,'planned_unique_quality_native_exports':120,
        'planned_initial_counterpart_exports':0,'planned_native_export_calls':120,'planned_query_writer_calls':20,
        'planned_query_forwards':80,'driver_raw_outer_fits':10,'helper_outer_train_fits':50,'raw_outer_fit_shared_with_PCA':True,
        'quality_gain':False,'encoder_device':'CUDA','prior_quality_input_roles':[],'quality_generated':False,
        'testing_accessed':False,'stress_accessed':False,'selection':False,'promotion':False,'human_card_sha256':CARD_SHA}
    for key,value in expected.items():check(plan[key]==value,'fixed confirmation recipe '+key)
    check(re.fullmatch('[a-f0-9]{64}',plan['source_fingerprint']) is not None,'actual captured source identity')


def admit_file_matrix(records, base, hasher=None):
    admitted_relative_names([x['path'] for x in records])
    paths=[]; inodes=set()
    for row in records:
        path=base/row['path']
        check(path.is_file() and not path.is_symlink() and path.resolve(strict=True)==path and path.is_relative_to(base), 'bounded alias-free archive/source path')
        st=path.stat();inode=(st.st_dev,st.st_ino)
        check(st.st_nlink==1 and inode not in inodes,'no hardlink/alias in entire matrix')
        check(type(row['bytes']) is int and row['bytes']>0 and row['bytes']==st.st_size and
              bool(re.fullmatch('[a-f0-9]{64}',row['sha256'])),'whole matrix size/hash declaration')
        inodes.add(inode);paths.append(path)
    if hasher:
        for row,path in zip(records,paths): check(hasher(path)==row['sha256'],'inventoried immutable file bytes')
    return {x['path']:x for x in records}


def capsule_binding(capsule):
    check(capsule.is_absolute() and capsule.is_dir() and capsule.resolve(strict=True)==capsule and
          capsule.parent.as_posix().endswith('/'+COHORT_ROOT) and capsule.name.startswith('early-mixer-confirmation-'), 'exclusive declared new capsule')
    inv=read_json(capsule/'artifact-integrity.json')
    check(inv['protocol']==PROTOCOL and inv['inventory_excludes_itself'] and inv['checksum_algorithm']=='sha256-file-bytes' and
          inv['file_count']==len(inv['files']) and inv['total_bytes']==sum(x['bytes'] for x in inv['files']), 'complete declared new inventory')
    index=admit_file_matrix(inv['files'],capsule)
    actual={x.relative_to(capsule).as_posix() for x in capsule.rglob('*') if x.is_file() or x.is_symlink()}
    check(actual==set(index)|{'artifact-integrity.json'},'no untracked capsule files before any payload hashes')
    for name,row in index.items(): check(sha(capsule/name)==row['sha256'],'all new inventoried bytes')
    return index


def compile_scope(log, filename, macro, object_name=None):
    values = set()
    matching = 0
    for line in log.splitlines():
        if filename not in line or ' -c ' not in line:
            continue
        if object_name is not None and not re.search(r' -o [^\s]*' + re.escape(object_name) + r'(?:\s|$)', line):
            continue
        found = re.search(r'-D' + re.escape(macro) + r'=\\?"([0-9a-f]{64})\\?"', line)
        if found:
            matching += 1
            values.add(found.group(1))
    if object_name is not None:
        check(matching == 1, 'one unambiguous exact compile command for target ' + object_name)
    check(len(values) == 1, 'one actual compile scope for source/object/macro ' + filename + '/' + str(object_name) + '/' + macro)
    return next(iter(values))


def flags_only_source_binding(runtime, prequality):
    normalized=runtime
    for name in (b'REVIEWED_SCHEMA',b'MEASURED_IMPLEMENTATION'):
        true=rb'(?m)^'+name+rb' = True(?=\r?$)'
        false=rb'(?m)^'+name+rb' = False(?=\r?$)'
        check(len(re.findall(true,normalized))==1 and not re.findall(false,normalized), 'runtime release has exactly one enabled '+name.decode())
        check(len(re.findall(false,prequality))==1 and not re.findall(true,prequality), 'captured prequality has exactly one disabled '+name.decode())
        normalized=re.sub(true,name+b' = False',normalized)
    check(normalized==prequality,'only the two authorized release flag lines differ from prequality source bytes')
    return {'prequality_source_sha256':hashlib.sha256(prequality).hexdigest(),
            'released_source_sha256':hashlib.sha256(runtime).hexdigest(),'two_flag_reverse_byte_proof':True}


def source_binding(capsule,index):
    manifest=read_json(capsule/'source-manifest.json');records=manifest['sources']
    check(manifest['protocol']==PROTOCOL and manifest['human_card']==CARD, 'whole source closure protocol/card')
    names=admit_source_roles(records,index)
    canonical=''.join(x['sha256']+'  '+x['path']+'\n' for x in records).encode()
    source=hashlib.sha256(canonical).hexdigest()
    check(source==manifest['source_fingerprint']==(capsule/'source-fingerprint.txt').read_text().strip() and
          (capsule/'source-inputs.sha256').read_bytes()==canonical and
          (capsule/'admission/source-inputs.sha256').read_bytes()==canonical and
          read_json(capsule/'admission/source-manifest.json')['sources']==records,'canonical measured/admission compiled closure')
    check(index['source/'+CARD]['sha256']==CARD_SHA,'prospectively frozen card')
    plan=read_json(capsule/'recipe-plan.json');fixed_plan(plan);check(plan['source_fingerprint']==source,'plan actual source')
    launch=read_json(capsule/'launch-plan.json');admission=read_json(capsule/'admission/passed.json')
    check(launch['protocol']==PROTOCOL and launch['status']=='frozen_before_quality_generation' and
          launch['source_fingerprint']==source and launch['human_card_sha256']==CARD_SHA and
          launch['historical_payload_roles']==0 and launch['encoder_trajectories']==10 and launch['new_heads_planned']==420,'durable launch before generation')
    check(admission['protocol']==PROTOCOL and admission['status']=='passed' and admission['source_fingerprint']==source and
          admission['human_card_sha256']==CARD_SHA and admission['source_preserved'] and
          not admission['quality_generated'] and admission['historical_payload_roles']==0,'actual pre-data CUDA admission')
    check(sha(capsule/'admission/passed.json')==launch['admission_passed_sha256'] and
          sha(capsule/'admission/build-and-tests.log')==launch['admission_log_sha256']==admission['log_sha256'] and
          sha(capsule/'recipe-plan.json')==launch['recipe_sha256'] and launch['compiled_binary']==admission['compiled_binary'],'exact admission/launch/binary/recipe binding')
    log=(capsule/'admission/build-and-tests.log').read_text()
    for marker in ('Early mixer model CUDA admission passed','Early mixer CUDA adapter admission passed','Early mixer confirmation CUDA admission passed','Fixed feature readout tests passed',
                   'Frozen role guard checks passed','Container SDK proof:',source):
        check(marker in log,'actual source-only admission marker '+marker)
    check('/embedding/.external/libtorch' not in log and 'not found' not in log,'pinned internal SDK resolution')
    scopes={'core_writer':compile_scope(log,'code/encoders/raw_patch_bottleneck_mae/src/workflow.cpp','RPB_SOURCE_ID'),
            'curve_training':compile_scope(log,'code/encoders/raw_patch_bottleneck_mae/src/learning_curve_adapter.cpp','EVALUATION_SOURCE_ID'),
            'early_adapter':compile_scope(log,'code/encoders/raw_patch_bottleneck_mae/src/early_mixer_adapter.cpp','EARLY_MIXER_ADAPTER_SOURCE_ID','early_mixer_adapter.o'),
            'confirmation_adapter':compile_scope(log,'code/encoders/raw_patch_bottleneck_mae/src/early_mixer_confirmation_adapter.cpp','EARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID','early_mixer_confirmation_adapter.o')}
    check(compile_scope(log,'code/evaluation/src/early_mixer_confirmation_main.cpp','EVALUATION_SOURCE_ID')==source and scopes['confirmation_adapter']==source,'enclosing main/loader scope')
    check(len(launch['reader'])==3,'three presealed reader source/fixture/seal records')
    allowed={'independent-reader/source.py','independent-reader/source-fixtures.json','independent-reader/reader-seal.json'}
    captured=[]
    for row in launch['reader']:
        name=Path(row['captured']).relative_to(capsule).as_posix();captured.append(name)
        check(name in allowed and index[name]['sha256']==row['sha256'] and index[name]['bytes']==row['bytes'],'exact pre-data captured reader')
    check(set(captured)==allowed,'complete prequality source/fixture/seal capture')
    release=flags_only_source_binding(Path(__file__).read_bytes(),(capsule/'independent-reader/source.py').read_bytes())
    seal=read_json(capsule/'independent-reader/reader-seal.json');fixtures=read_json(capsule/'independent-reader/source-fixtures.json')
    check(seal['status']=='passed' and seal['source_sha256']==release['prequality_source_sha256'] and
          seal['fixtures_sha256']==sha(capsule/'independent-reader/source-fixtures.json') and fixtures['status']=='passed' and
          fixtures['archive_payload_reads']==0,'source-only fixture/review/seal record')
    for row in fixtures['writer_source_schema'] + fixtures['sdk_source_schema']:
        check(index['source/'+row['path']]['sha256']==row['sha256'],'source-schema fixture bound to the actual captured writer '+row['path'])
    roles=read_json(capsule/'input-role-plan.json')
    check(roles['protocol']==PROTOCOL and roles['historical_payload_roles']==0 and roles['prior_quality_input_roles']==[],'zero historical payload roles')
    preserved=read_json(capsule/'source-preserved-after.json')
    check(preserved['protocol']==PROTOCOL and preserved['source_fingerprint']==source and preserved['source_preserved'] and preserved['reader_preserved'],'sources/reader preserved after run')
    fixed_plan(read_json(capsule/'results/recipe-plan.json'))
    card=read_json(capsule/'results/card.json')
    check(card=={'protocol':PROTOCOL,'human_card_sha256':CARD_SHA,'source_fingerprint':source,'historical_payload_roles':0,'testing_accessed':False,'stress_accessed':False},'truthful instantiated card')
    return {'source_fingerprint':source,'human_card_sha256':CARD_SHA,'actual_compile_scopes':scopes,
            'admission_sha256':sha(capsule/'admission/passed.json'),'admission_log_sha256':admission['log_sha256'],
            'reader_sha256':release['released_source_sha256'],'prequality_reader_sha256':release['prequality_source_sha256'],
            'reader_release':release,'reader_fixtures_sha256':seal['fixtures_sha256'],
            'captured_source_files':len(records),'inventory_sha256':sha(capsule/'artifact-integrity.json')}


def output_archive_matrix(results, report):
    check(len(report['cohorts']) == 5,'complete five-cohort report before archive role discovery')
    expected = set()
    for cohort,master in zip(report['cohorts'],TIMING_MASTERS):
        prefix = f'seed-{master}-lag_sign/'
        expected.add(prefix+'initial-pair.pt')
        for role in ('late','early'):
            for point in BUDGETS:
                base = prefix+f'{role}/point-{point}/'
                expected.update(base+'checkpoint.pt'+suffix for suffix in
                                ('','.audit.pt','.scaler.pt','.training-raw.pt','.confirmation.pt'))
                expected.add(base+'snapshot-assets/early-mixer-snapshot-audit.pt')
                expected.add(base+'snapshot-assets/confirmation-snapshot-audit.pt')
                if point: expected.update(base+split+'-reconstruction.pt' for split in ('training','validation'))
        check(len(cohort['tasks']) == 2,'both task reports before archive role discovery')
        for task,item in zip(('lag_sign','amplitude'),cohort['tasks']):
            observed = prefix if task=='lag_sign' else prefix+'amplitude/'
            expected.update(observed+'controlled-'+view+'.pt' for view in ('training','validation','validation-deleted'))
            root = prefix+task+'/'
            expected.update(root+view for view in ('raw-scaler.pt','raw-outer-normalizer.pt'))
            # Two separately fitted initial controls; no shared counterpart assets.
            status = read_json(results/root/'pca-status.json')
            if status['status']=='measured': expected.add(root+'pca-preprocessing.pt')
            check([m['method'] for m in item['readouts']['methods']] == list(METHODS),
                  'complete fixed task methods before archive decode')
            for method in item['readouts']['methods']:
                base = root+'readouts/'+method['method']+'/'
                expected.update(base+view+'-features.pt' for view in VIEWS)
                if method['status']=='measured':
                    for rep in HEAD_REPETITIONS:
                        path=base+f'rep-{rep}/'
                        expected.add(path+'fit.pt')
                        expected.update(path+view+'-predictions.pt' for view in VIEWS)
    actual = {p.relative_to(results).as_posix() for p in results.rglob('*.pt')}
    check(actual == expected,'closed all-point/control/live-state/head/query archive matrix')
    return expected


def run(capsule, output):
    started=time.monotonic()
    index=capsule_binding(capsule);source=source_binding(capsule,index)
    scopes=source['actual_compile_scopes'];repo=capsule.parents[3];import_helper(repo)
    results=capsule/'results';report=read_json(results/'report.json');complete=read_json(results/'complete.json')
    check(report['protocol']==PROTOCOL and report['source_fingerprint']==source['source_fingerprint'] and
          len(report['cohorts'])==5 and all(report[k] is False for k in ('testing_accessed','stress_accessed','selection','promotion')),
          'fresh five-cohort fixed report, no testing/stress/selection/promotion')
    archive_matrix=output_archive_matrix(results,report)
    audited=[];all_sources=set();pipelines=0;paired_records=0
    for cohort,master,amplitude_master in zip(report['cohorts'],TIMING_MASTERS,AMPLITUDE_MASTERS):
        check(cohort['timing_master']==master and cohort['amplitude_data_master']==amplitude_master,'five frozen independent cohort mappings')
        check(len(cohort['encoder_points'])==2 and len(cohort['tasks'])==2,'complete two-architecture/two-task report matrix')
        root=results/f'seed-{master}-lag_sign'
        timing_files=[load(root/('controlled-'+name+'.pt')) for name in ('training','validation','validation-deleted')]
        timing=[observations(value,rows,master,'lag_sign',i==2) for i,(value,rows) in enumerate(zip(timing_files,(256,128,128)))]
        timing_view=check_view(timing[1],timing[2],master,'lag_sign',timing_files[2])
        initial=initial_witness(load(root/'initial-pair.pt'))
        initialization_binding(read_json(root/'initialization-audit.json'),master,scopes)
        check(read_json(root/'initial-pair.json')=={'full_named_parameters_exact':True,'buffers_exact':True,'scaler_exact':True,
              'initial_features_assumed_equal':False,'initial_control_sets':2,'per_step_trace_counts_exact':True,
              'initial_shared_state_exact_before_training_and_heads':True},'full initialization and stream witnesses')
        providers={};infos={};trace_by_version={};query_assets={};encoders=[]
        for placement,version in enumerate(('late','early')):
            for point in (0,512):
                cp=root/version/f'point-{point}/checkpoint.pt'
                audit=load(Path(str(cp)+'.audit.pt'))
                info=replay_parent(audit,load(Path(str(cp)+'.training-raw.pt')),load(Path(str(cp)+'.scaler.pt')),
                                   timing[0],master,placement,point,scopes)
                check(info['scaler_id']==initial[3],'all four points same initial frozen timing scaler')
                saved=load(cp.parent/'snapshot-assets/early-mixer-snapshot-audit.pt')
                method=('native_' if point else 'untrained_')+version
                side=load(Path(str(cp)+'.confirmation.pt'))
                supplement=load(cp.parent/'snapshot-assets/confirmation-snapshot-audit.pt')
                confirmation_fields=confirmation_binding(side,audit,info,cp,placement,point,timing[0][2],scopes)
                providers[method]=snapshot_with_confirmation(saved,side,supplement,audit,info,cp,placement,point,timing[0][2],scopes,initial)
                infos[version,point]=info
                if point:
                    actual=read_json(root/version/'encoder-progress.json')
                    trace_by_version[version]=progress(actual,info)
                    declared=cohort['encoder_points'][placement]
                    check(declared['model_tag']==('RPB-v10.alt-04' if placement else 'RPB-v7.alt-04') and declared['placement']==placement and
                          declared['encoder_progress']==actual,'report actual encoder point and complete saved trace')
                    static=read_json(root/version/'trainer-audit.json')
                    for key,value in static.items():
                        if key=='snapshot_adapter_source_fingerprint':check(value==scopes['early_adapter'],'new live snapshot adapter scope')
                        elif key=='snapshot_policy':check(value=='new_protocol_bound_CUDA_only;historical_CPU_snapshot_not_called','no CPU model snapshot')
                        elif key in confirmation_fields:check(value==confirmation_fields[key],'new static external/implementation binding '+key)
                        else:check(value==text(audit,key),'static delegated trainer association '+key)
                    queries={}
                    for name,split in [('training',timing[0]),('validation',timing[1])]:
                        asset=load(cp.parent/(name+'-reconstruction.pt'))
                        replay=query_archive(asset,split,info['frozen_scaler']);query_assets[version,name]=asset
                        query_summary(declared[name+'_reconstruction'],replay,asset,name+'-reconstruction.pt')
                        queries[name]={'standardized_mae':replay['mae'],'standardized_huber':replay['huber'],
                                       'valid_examples':replay['valid_examples'],'total_examples':replay['total_examples'],
                                       'valid_target_cells':replay['valid_target_cells'],'requested_observed_target_cells':replay['requested_observed_target_cells']}
                    encoders.append({'tag':declared['model_tag'],'role':version,'budget':512,'placement':placement,'architecture_id':ARCHITECTURES[placement],
                                     'training_seconds':info['training_seconds'],'attempted':512,'completed':512,'sampled_rows':4096,
                                     'context_counts':info['counts'],'fixed_query':queries})
        check(infos['late',512]['common_settings']==infos['early',512]['common_settings'] and
              infos['late',512]['counts']==infos['early',512]['counts'],'identical original configuration/augmentation streams, sole mixer placement')
        for a,b in zip(trace_by_version['late'],trace_by_version['early']):check(a[:3]==b[:3],'exact paired absolute trace/counter/target prefix')
        for name in ('training','validation'):
            for key in ('standardized_target','target_mask','requested_observed_target_mask','visible_mask','trial_channel_eligible',
                        'channel_target_counts','channel_valid','example_valid','source_ids_json'):
                exact(query_assets['late',name][key],query_assets['early',name][key],'same paired original query '+key)
        tasks=[]
        for task,task_master,declared in zip(('lag_sign','amplitude'),(master,amplitude_master),cohort['tasks']):
            check(declared['task']==task and declared['data_master']==task_master,'separate timing/amplitude task and data seed')
            path=root/task
            if task=='lag_sign':data=timing;view=timing_view
            else:
                files=[load(path/('controlled-'+name+'.pt')) for name in ('training','validation','validation-deleted')]
                data=[observations(value,rows,task_master,task,i==2) for i,(value,rows) in enumerate(zip(files,(256,128,128)))]
                view=check_view(data[1],data[2],task_master,task,files[2])
            check(set(data[0][2]).isdisjoint(data[1][2]),'TRAIN and intact VALIDATION independent source populations')
            for split in data[:2]:
                ids=set(split[2]);check(all_sources.isdisjoint(ids),'all five cohorts and two tasks distinct source groups');all_sources.update(ids)
            controls,status=control_rows(path,*data)
            # Shared raw/PCA arithmetic and covariance were already verified once by control_rows.
            summary,original=audit_readouts(path,data,controls,status,providers,task_master,task)
            check(original==declared['readouts'],'embedded report exactly retains actual per-task scores/statuses/intervals')
            check(read_json(path/'preprocessing-fit-counts.json')=={
                'ObservationScaler_fits':1,'driver_raw_outer_fits':1,'driver_raw_outer_shared_with_PCA':True,
                'helper_outer_train_fits':original['fit_counts']['outer_train_normalizer_fits'],
                'planned_helper_outer_train_fits':5,'validation_fits':0,'native_PCA_fits':0},
                'one shared driver TRAIN raw/PCA fit and all actual supported helper fits')
            pipelines+=original['fit_counts']['ridge_fits'];paired_records+=len(original['pairs'])
            summary.update({'task':task,'validation_deletion':view,'PCA_status':status});tasks.append(summary)
        expected_costs={'generation_and_observation_io_seconds','binding_and_checkpoint_io_seconds','CUDA_query_transfer_verification_io_seconds',
                        'CUDA_native_transfer_verification_seconds','CPU_baseline_preparation_io_seconds','CPU_head_bootstrap_io_seconds'}
        check(set(cohort['costs'])==expected_costs,'declared mixed wall cost scopes, no invented pure kernel costs')
        for key,value in cohort['costs'].items():finite_timer(value,key)
        audited.append({'timing_master':master,'amplitude_data_master':amplitude_master,'encoders':encoders,'tasks':tasks,'costs':cohort['costs']})
        print('Audited early/late paired cohort '+str(master),flush=True)
    helper_outer=sum(t['fit_counts']['outer_train_normalizer_fits'] for c in audited for t in c['tasks'])
    check(0<=helper_outer<=50,'actual helper fits preserve unsupported cohorts')
    expected_complete={'protocol':PROTOCOL,'status':'complete','cohorts':5,'tasks_each':2,'encoder_trajectories':10,'encoder_updates_each':512,
                       'attempt_limit':1024,'retained_points':20,'skipped_attempts':0,
                       'sampled_rows':40960,'head_pipelines':pipelines,'individual_heads':2*pipelines,'planned_pipelines':210,'planned_heads':420,
                       'unique_quality_native_exports':120,'initial_counterpart_exports':0,
                       'full_native_export_calls':120,'query_evaluation_calls':20,'necessary_query_forwards':80,
                       'driver_raw_outer_fits':10,'helper_outer_train_fits':helper_outer,
                       'initial_shared_state_exact_before_training_and_heads':True,'separate_initial_controls':2,
                       'extra_decoder_calibration_updates':0,
                       'decoder_update_scope':'joint reconstruction during encoder updates; no extra decoder-only calibration',
                       'selection':False,'CPU_encoder_training':False,'CPU_encoder_forward':False,
                       'testing_accessed':False,'stress_accessed':False,'promotion':False}
    check(complete==expected_complete and paired_records==60 and len(all_sources)==1920,'actual complete populations/head/export/query counts')
    # Recheck every captured byte after arithmetic; no encoder/checkpoint reconstruction.
    for name,row in index.items():check(sha(capsule/name)==row['sha256'],'immutable capsule after independent arithmetic')
    return {'protocol':PROTOCOL,'status':'passed','checks':CHECKS,'archive_decodes':ARCHIVES,'elapsed_seconds':time.monotonic()-started,
            'source':source,'cohorts':audited,'decision':decision_summary(audited),'counts':{'unique_source_groups':1920,'encoder_runs':10,'head_pipelines':pipelines,
                    'individual_heads':2*pipelines,'paired_records':60,'head_effects':120,'native_full_export_calls':120,
                    'fixed_query_writer_calls':20,'fixed_query_banks':80,'retained_points':20,'checkpoint_roles_per_point':5,
                    'driver_raw_outer_fits':10,'helper_outer_fits':sum(t['fit_counts']['outer_train_normalizer_fits'] for c in audited for t in c['tasks']),'new_archive_roles':len(archive_matrix)},
            'limits':{'model_forward_or_reconstruction':False,'CUDA_execution':False,'autodiff_or_optimizer_execution':False,
                     'encoder_or_head_fits':0,'PCA_or_SVD_fits':0,'ordinary_CUDA_checkpoint_body_decodes':0,
                     'historical_payload_reads':0,'TEST_or_stress_reads':0,'across_encoder_seed_interval':False,
                     'saved_arithmetic_only':True,'checkpoint_feature_association':'frozen source/admission and saved snapshot witness bound; no encoder rerun',
                     'original_sampling_and_Torch_streams':'source/admission, exact trace target counts and cumulative context counts bound; not GPU reenacted',
                     'costs':'synchronized update-loop timers and separately named mixed inference/transfer/verification/I/O wall scopes; not pure GPU kernel comparisons',
                     'interpretation':'equal parameter counts do not imply equal compute; no causal internal timing-loss proof, no task averaging or promotion'}}


def fixture_parent(master, placement, point, ids):
    settings={'channel_count':3,'history_length':32,'input_width':3,'patch_length':8,'encoder_width':64,
              'export_width':32,'num_layers':3,'num_heads':4,'feedforward_width':256,'decoder_hidden_width':128,
              'channel_mixer_layers':1,'global_bottleneck_mode':2,'steps':512,'batch_size':8,'threads':1,
              'log_every':1,'attempt_limit':1024,'seed':master,'dropout':0.,'layer_norm_epsilon':1e-5,
              'mask_ratio':.25,'huber_delta':1.,'scale_floor':1e-6,'sampling_interval':1.,'learning_rate':.001,
              'weight_decay':.0001,'gradient_clip_norm':1.,'device':'cuda','channel_ids':''}
    if placement:settings['channel_mixer_placement']=1
    fields={'artifact_kind':'rpb_learning_curve_training_audit_v1','protocol_id':FIT_PROTOCOL,
            'actual_training_seed':str(master),'initialization_seed':str(mixed(master ^ 0x7270622d696e6974)),
            'fit_source_manifest':source_manifest(ids),'rng_policy':COUNTER_POLICY,'model_weight_update_budget':str(point),
            'training_device':'cuda','training_producer_source_fingerprint':'2'*64,'core_writer_source_fingerprint':'1'*64,
            'resolved_settings':'\n'.join(key+'='+str(value) for key,value in settings.items()),
            'training_dataset_id':'fixture-dataset','scaler_fit_dataset_id':'fixture-dataset',
            'parameter_count':'225805','cuda_parameter_count':'225805','preprocessing_id':'pending',
            'feature_units':'unitless,unitless,unitless','sampling_policy':'with_replacement_counter_rows;sampled_rows_includes_no_update_attempts',
            'optimizer_policy':'one_continuous_AdamW_state;absolute_completed_update_budgets',
            'source_fingerprint_scope':'ordinary_checkpoint_field=core_writer;companion_audit_training_producer=evaluation_source',
            'channel_mixer_placement':str(placement),'architecture_id':ARCHITECTURES[placement],
            'output_semantics':OUTPUT_SEMANTICS[placement],'reconstruction_export_semantics':RECONSTRUCTION_SEMANTICS[placement],
            'model_tag':'RPB-v10' if placement else 'RPB-v7',**CONTEXT_LITERALS}
    audit={key:fake_text(value) for key,value in fields.items()}
    for key,value in {'attempted_steps':point,'completed_steps':point,'sampled_rows':point*8,
                      'channel_mixer_placement_value':placement,'context_deletion_stream_value':0x6374782d64726f70,
                      'context_requested_deleted_coordinates':0,'context_actual_deleted_coordinates':0,'context_restored_coordinates':0}.items():
        audit[key]=fake('LongStorage',[],[value])
    audit['channel_order']=fake('LongStorage',[3],[0,1,2])
    for key,value in {'sampling_interval':1.,'endpoint':31.,'training_seconds':1. if point else 0.,'context_deletion_ratio_value':.15}.items():
        audit[key]=fake('DoubleStorage',[],[value])
    for key in ('weights_changed','finite_gradients'):audit[key]=fake('BoolStorage',[],[bool(point)])
    return audit


def schema_source_fixtures(repo):
    paths=['code/evaluation/src/early_mixer_confirmation_main.cpp','code/shared/src/fixed_feature_readouts.cpp',
        'code/shared/src/paired_pooling.cpp','code/encoders/raw_patch_bottleneck_mae/src/early_mixer_adapter.cpp',
        'code/encoders/raw_patch_bottleneck_mae/src/learning_curve_adapter.cpp',
        'code/encoders/raw_patch_bottleneck_mae/src/early_mixer_confirmation_adapter.cpp',
        'code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/early_mixer_confirmation_adapter.h',
        'code/encoders/raw_patch_bottleneck_mae/tests/early_mixer_confirmation_adapter_test.cpp']
    records=[]
    for name in paths:
        path=repo/name
        check(path.is_file() and path.resolve(strict=True)==path and not path.is_symlink() and path.stat().st_nlink==1,'bounded writer SOURCE fixture')
        records.append({'path':name,'sha256':sha(path)})
    main=(repo/paths[0]).read_text();body=main[main.index('void save_observations('):main.index('void initial_pair(')]
    check(set(re.findall(r'a\.write\("([^\"]+)"',body))=={'observations','feature_mask','labels_scoring_only','source_ids_json','requested_erasure'},'exact controlled writer keys')
    query=(repo/paths[2]).read_text();body=query[query.index('std::string reconstruction('):query.index('void check_progress(')]
    check(set(re.findall(r'output\.write\("([^\"]+)"',body))=={'standardized_prediction','standardized_target','target_mask','requested_observed_target_mask',
        'visible_mask','trial_channel_eligible','channel_target_counts','channel_valid','channel_standardized_mae','channel_standardized_huber',
        'example_valid','example_standardized_mae','example_standardized_huber','source_ids_json'},'unchanged original-query writer keys')
    check('out.provenance+"; "+snapshot.features.provenance' in query,'actual native surface provenance is extracted base then provider base/suffix')
    original=(repo/paths[3]).read_text();wrapper=(repo/paths[5]).read_text();header=(repo/paths[6]).read_text()
    for literal in ('fnv1a64-runtime-content-v1-','independent_CUDA_model;eval_no_grad;frozen_timing_TRAIN_scaler;ordinary_query_no_context_erasure;no_refit',
                    'parent=core_writer_and_training_producer;loader=new_early_mixer_adapter','cpu(z)','no_optimizer_created'):
        check(literal in original,'truthful literal delegated snapshot SOURCE '+literal)
    for literal in ('content_checkpoint','content_audit','content_scaler','content_training_raw','content_confirmation',
        'confirmation_protocol_id','external_fit_protocol_id','training_implementation_fit_protocol_id','confirmation_scope',
        'parent_snapshot_adapter_source_fingerprint','original_training_source_manifest_id','parameter_count_value',
        'new-confirmation-binding;unchanged-reliability-CUDA-serving;no-old-quality-input',
        'side.keys().size()==record.fields.size()+record.values.size()', 'original.features.extract(batch)', 'original.reconstruct(batch,mask)'):
        check(literal in wrapper,'closed new wrapper/source delegation literal '+literal)
    check('rpb_early_mixer_confirmation_binding_v1' in header and 'rpb_early_mixer_confirmation_cuda_snapshot_v1' in header and
        'confirmation-snapshot-audit.pt' in header and '.confirmation.pt' in header,'new fifth role and separate snapshot literal kinds')
    check('Early mixer confirmation CUDA admission passed' in (repo/paths[7]).read_text(),'new artificial CUDA wrapper admission test SOURCE bound')
    for literal in ('fit_count','training_source_ids_json','raw-outer-normalizer.pt','outer.write("feature_mean"','outer.write("feature_scale"'):
        check(literal in main,'actual shared raw outer writer '+literal)
    readouts=(repo/paths[1]).read_text()
    for literal in ('ridge_logits','tiny_hidden_preactivation','tiny_logits','actual_probe_seed_decimal','training_source_ids_json',
                    'ridge_penalty','tiny_hidden','tiny_steps','tiny_learning_rate','comparison_reference','comparison_candidate'):
        check(literal in readouts,'unchanged saved head SOURCE '+literal)
    learner=(repo/paths[4]).read_text()
    for key in CONTEXT_LITERALS:check(key in learner,'unchanged learner field '+key)
    return records


def _base_self_test(repo):
    start_checks=CHECKS;start_archives=ARCHIVES;negative=0
    def reject(call):
        nonlocal negative
        try:call()
        except (AssertionError,RuntimeError,ValueError,KeyError):negative+=1
        else:raise AssertionError('negative artificial fixture accepted')
    import_helper(repo);writer_sources=schema_source_fixtures(repo)
    check(MT19937_64(5489).draw()==14514284786278117030,'MT19937_64 reference word')
    ids=[f'{PROTOCOL}/lag_sign/seed-{TIMING_MASTERS[0]}/lag_sign/source-{i//2}' for i in range(256)]
    labels=[i%2 for i in range(256)]
    data=[b%2+h*.01 for b in range(256) for c in range(3) for h in range(32) for f in range(3)];mask=[True]*len(data)
    controlled=(data,mask,ids,labels)
    observed={'observations':fake('DoubleStorage',[256,3,32,3],data),'feature_mask':fake('BoolStorage',[256,3,32,3],mask),
              'labels_scoring_only':fake('LongStorage',[256],labels),'source_ids_json':fake_text(json.dumps(ids))}
    observations(observed,256,TIMING_MASTERS[0],'lag_sign')
    wrong=dict(observed);wrong['observed']=wrong.pop('observations');reject(lambda:observations(wrong,256,TIMING_MASTERS[0],'lag_sign'))
    reject(lambda:validate_pairs([0,0],['a','a'],[[True],[True]]))
    reject(lambda:validate_pairs([0,1],['a','a'],[[True],[False]]))
    check(deletion_view([1.,0.,2.],[True,False,True],[False,False,True])==([1.,0.,0.],[True,False,False]),'deletion-only original visible values')
    reject(lambda:deletion_view([1.,9.],[True,False],[False,False]))
    erased=coordinate_erasure(ids[:2],'lag_sign',TIMING_MASTERS[0]);check(erased[0]==erased[1],'source-pair-shared all-coordinate erasure')
    check(erased[0]!=coordinate_erasure(ids[:1],'amplitude',AMPLITUDE_MASTERS[0])[0],'independent task namespace/stream')
    selected=[b%2+h*.01 for b in range(256) for h in range(32)]
    mean=math.fsum(selected)/len(selected);sd=math.sqrt(math.fsum((x-mean)**2 for x in selected)/len(selected))
    frozen={'mean':fake('DoubleStorage',[3,3],[mean]*9),'scale':fake('DoubleStorage',[3,3],[sd]*9),
            'count':fake('LongStorage',[3,3],[8192]*9),'channel_ids':fake('LongStorage',[3],[0,1,2]),
            'floor_applied':fake('BoolStorage',[3,3],[False]*9),'scale_floor':fake('DoubleStorage',[],[1e-6])}
    identity=scaler(frozen)[2]
    raw={key:fake_text(value) for key,value in {'artifact_kind':'rpb_raw_uniform_history_v1','schema_id':'fixture-schema',
         'dataset_id':'fixture-dataset','feature_units':'unitless,unitless,unitless'}.items()}
    raw.update({'format_version':fake('LongStorage',[],[1]),'data':observed['observations'],'observed':observed['feature_mask'],
                'channel_ids':fake('LongStorage',[3],[0,1,2]),'endpoints':fake('DoubleStorage',[256],[31.]*256),
                'sampling_interval':fake('DoubleStorage',[],[1.])})
    scaler_asset={key:fake_text(value) for key,value in {'artifact_kind':'rpb_frozen_training_scaler_v1','schema_id':'fixture-schema',
                  'fit_dataset_id':'fixture-dataset','preprocessing_id':identity}.items()}
    scaler_asset.update({'format_version':fake('LongStorage',[],[1]),'scaler':frozen})
    scopes={'core_writer':'1'*64,'curve_training':'2'*64,'early_adapter':'3'*64}
    initial_params={'a.weight':fake('FloatStorage',[225804],[0.]*225804),'b.weight':fake('FloatStorage',[1],[0.])}
    empty={'count':fake('LongStorage',[],[0])}
    def named(values):
        return {'count':fake('LongStorage',[],[len(values)]),**{f'tensor_{i}':{'parameter_name':fake_text(name),'value':value} for i,(name,value) in enumerate(values.items())}}
    witness={name+'_parameters':named(initial_params) for name in ('late','early')}
    witness.update({name+'_buffers':empty for name in ('late','early')});witness.update({name+'_scaler':frozen for name in ('late','early')})
    initial=initial_witness(witness)
    bad=dict(witness);bad['early_parameters']=named({'different.name':fake('FloatStorage',[225805],[0.]*225805)});reject(lambda:initial_witness(bad))
    bad=dict(witness);bad['early_parameters']=named(dict(reversed(list(initial_params.items()))));reject(lambda:initial_witness(bad))
    with tempfile.TemporaryDirectory(prefix='early-reader-snapshot-source-') as temp:
        cp=Path(temp)/'checkpoint.pt'
        for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.confirmation.pt'):Path(str(cp)+suffix).write_text('artificial source fixture '+suffix)
        for placement in (0,1):
            for point in (0,512):
                audit=fixture_parent(TIMING_MASTERS[0],placement,point,ids);audit['preprocessing_id']=fake_text(identity)
                audit['fit_source_manifest_id']=fake_text('manifest-fixture')
                info=replay_parent(audit,raw,scaler_asset,controlled,TIMING_MASTERS[0],placement,point,scopes)
                for key,value in [('channel_mixer_placement',str(1-placement)),('architecture_id',ARCHITECTURES[1-placement]),
                                  ('output_semantics','wrong'),('training_policy_id','ordinary'),('protocol_id',PROTOCOL+'/amplitude')]:
                    bad=dict(audit);bad[key]=fake_text(value);reject(lambda:parent_audit(bad,TIMING_MASTERS[0],placement,point,ids))
                bad=dict(audit);bad['completed_steps']=fake('LongStorage',[],[point+1]);reject(lambda:parent_audit(bad,TIMING_MASTERS[0],placement,point,ids))
                asset={k:v for k,v in audit.items() if v.get('dtype')=='ByteStorage'}
                values={'artifact_kind':'rpb_early_mixer_cuda_snapshot_v1','protocol_id':IMPLEMENTATION_PROTOCOL,'original_training_protocol_id':FIT_PROTOCOL,
                        'parent_checkpoint_path':str(cp),'parent_writer_source_fingerprint':'1'*64,'parent_training_producer_source_fingerprint':'2'*64,
                        'snapshot_loader_source_fingerprint':'3'*64,'source_fingerprint_scope':'parent=core_writer_and_training_producer;loader=new_early_mixer_adapter',
                        'original_encoder_attempted':str(point),'original_encoder_completed':str(point),'training_schema_id':'fixture-schema',
                        'training_dataset_id':'fixture-dataset','preprocessing_id':identity,'parameter_count':'225805','encoder_updates':'0',
                        'decoder_updates':'0','head_refits':'0','no_optimizer_created':'true','inference_device':'cuda',
                        'snapshot_policy':'independent_CUDA_model;eval_no_grad;frozen_timing_TRAIN_scaler;ordinary_query_no_context_erasure;no_refit'}
                for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt'):values['parent_content_id'+suffix]='fnv1a64-runtime-content-v1-'+fnv(Path(str(cp)+suffix).read_bytes())
                for key,value in values.items():asset[key]=fake_text(value)
                for key in ('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'):asset[key]=fake_text('0')
                for key,value in {'channel_mixer_placement_value':placement,'original_encoder_attempted_value':point,'original_encoder_completed_value':point}.items():asset[key]=fake('LongStorage',[],[value])
                params=dict(initial_params)
                if point:params['a.weight']=fake('FloatStorage',[225804],[1.]+[0.]*225803)
                asset.update({'model_parameters':named(params),'model_buffers':empty,'scaler':frozen})
                result=snapshot_binding(asset,audit,info,cp,placement,point,scopes,initial)
                check(result['checkpoint_body_decoded'] is False,'snapshot evidence never decodes CUDA CP')
                dual_scopes={**scopes,'confirmation_adapter':'4'*64}
                bindings=[]
                for is_snapshot in (False,True):
                    fields,typed=confirmation_expected(audit,info,cp,placement,point,ids,dual_scopes,is_snapshot)
                    bindings.append({**{k:fake_text(v) for k,v in fields.items()},**{k:fake('LongStorage',[],[v]) for k,v in typed.items()}})
                combined=snapshot_with_confirmation(asset,*bindings,audit,info,cp,placement,point,ids,dual_scopes,initial)
                base=('RPB-v10' if placement else 'RPB-v7')+'; immutable timing-protocol checkpoint; exact contextual-global32 CUDA serving; ordinary original-Q decoder'
                check(combined['extractor_provenance']==base+'; '+base+';external-confirmation-cohort='+EXTERNAL_FIT_PROTOCOL+';training-implementation='+FIT_PROTOCOL,
                      'exact global_surface ordering: original extracted base then complete provider base and confirmation suffix')
                bad=dict(bindings[1]);bad['content_confirmation']=fake_text('corrupt')
                reject(lambda:snapshot_with_confirmation(asset,bindings[0],bad,audit,info,cp,placement,point,ids,dual_scopes,initial))
                bad=dict(asset);bad['no_optimizer_created']=fake_text('false');reject(lambda:snapshot_binding(bad,audit,info,cp,placement,point,scopes,initial))
                bad=dict(asset);bad['original_encoder_completed_value']=fake('LongStorage',[],[point+1]);reject(lambda:snapshot_binding(bad,audit,info,cp,placement,point,scopes,initial))
    reject(lambda:load(Path('/artificial/checkpoint.pt')));check(ARCHIVES==start_archives,'CP rejection precedes codec/read counter')
    # Real fixed query branch, including exact dtype, original eligibility and corrupt-target rejection.
    small=(data[:576],mask[:576],ids[:2],labels[:2]);rq,visible,q,eligible=query_masks(small[1],2)
    target=[f32((small[0][i%576]-mean)/sd) if keep else 0. for i,keep in enumerate(q)]
    prediction=[x+.25 if keep else 0. for x,keep in zip(target,q)];replay=reductions(prediction,target,q,2)
    asset={'standardized_prediction':fake('DoubleStorage',[4,2,3,32,3],prediction),'standardized_target':fake('DoubleStorage',[4,2,3,32,3],target),
           'target_mask':fake('BoolStorage',[4,2,3,32,3],q),'requested_observed_target_mask':fake('BoolStorage',[4,2,3,32,3],rq),
           'visible_mask':fake('BoolStorage',[4,2,3,32,3],visible),'trial_channel_eligible':fake('BoolStorage',[4,2,3],eligible),
           'channel_target_counts':fake('LongStorage',[2,3],replay['counts']),'channel_valid':fake('BoolStorage',[2,3],[bool(x) for x in replay['counts']]),
           'channel_standardized_mae':fake('DoubleStorage',[2,3],replay['channel_mae']),'channel_standardized_huber':fake('DoubleStorage',[2,3],replay['channel_huber']),
           'example_valid':fake('BoolStorage',[2],replay['example_valid']),'example_standardized_mae':fake('DoubleStorage',[2],replay['example_mae']),
           'example_standardized_huber':fake('DoubleStorage',[2],replay['example_huber']),'source_ids_json':fake_text(json.dumps(small[2]))}
    actual=query_archive(asset,small,frozen);close(actual['mae'],.25,'actual cell/channel/example reduction')
    bad=dict(asset);bad['standardized_target']=fake('DoubleStorage',[4,2,3,32,3],[x+1. if keep else 0. for x,keep in zip(target,q)])
    reject(lambda:query_archive(bad,small,frozen))
    bad=dict(asset);bad['channel_standardized_mae']=fake('DoubleStorage',[2,3],[.3]*6);reject(lambda:query_archive(bad,small,frozen))
    bad=dict(asset);bad['standardized_prediction']=fake('FloatStorage',[4,2,3,32,3],prediction);reject(lambda:query_archive(bad,small,frozen))
    # The actual saved classifier equation branch, own-argmax ties and explicit native no-PCA.
    width=2;rows=[[0.,0.],[0.,0.]];valid=[True,False];lab=[0,1];sources=['a','b']
    fit={key:fake('DoubleStorage',[width],[0.,0.]) for key in ('feature_mean','ridge_mean','tiny_mean')}
    fit.update({key:fake('DoubleStorage',[width],[1.,1.]) for key in ('feature_scale','ridge_scale','tiny_scale')})
    fit['outer_normalizer_applied']=fake('BoolStorage',[],[True])
    for key,value in {'outer_fitted_rows':2,'fitted_rows':2,'tiny_hidden':16,'tiny_steps':100}.items():fit[key]=fake('LongStorage',[],[value])
    fit.update({'ridge_penalty':fake('DoubleStorage',[],[1.]),'tiny_learning_rate':fake('DoubleStorage',[],[.01]),
                'ridge_weights':fake('DoubleStorage',[2,2],[0.]*4),'ridge_intercept':fake('DoubleStorage',[2],[0.,0.]),
                'tiny_w1':fake('DoubleStorage',[2,16],[0.]*32),'tiny_b1':fake('DoubleStorage',[16],[0.]*16),
                'tiny_w2':fake('DoubleStorage',[16,2],[0.]*32),'tiny_b2':fake('DoubleStorage',[2],[0.,0.])})
    pred={'ridge':fake('LongStorage',[2],[0,0]),'tiny_secondary':fake('LongStorage',[2],[0,0]),'valid':fake('BoolStorage',[2],valid),
          'probe_input_features':fake('DoubleStorage',[2,2],[0.]*4),'ridge_logits':fake('DoubleStorage',[2,2],[0.]*4),
          'tiny_hidden_preactivation':fake('DoubleStorage',[2,16],[0.]*32),'tiny_logits':fake('DoubleStorage',[2,2],[0.]*4),
          'labels_scoring_only':fake('LongStorage',[2],lab),'source_ids_json':fake_text(json.dumps(sources))}
    infer_saved_fit(fit,rows,valid,lab,sources,pred,False,width,2)
    bad=dict(pred);bad['ridge']=fake('LongStorage',[2],[1,0]);reject(lambda:infer_saved_fit(fit,rows,valid,lab,sources,bad,False,width,2))
    bad=dict(fit);bad['pca_components']=fake('DoubleStorage',[2,2],[0.]*4);reject(lambda:infer_saved_fit(bad,rows,valid,lab,sources,pred,False,width,2))
    bad=dict(pred);bad['ridge_logits']=fake('DoubleStorage',[2,2],[.01,0.,0.,0.]);reject(lambda:infer_saved_fit(fit,rows,valid,lab,sources,bad,False,width,2))
    pair=paired_group_effect([1,1,0,0],[0,1,0,1],[0,1,0,1],[True]*4,[True]*4,['a','a','b','b'])
    check(pair['valid']==4 and pair['estimate']==-.5,'complete-source candidate-reference correctness arithmetic')
    check(bootstrap_groups([],55,1000)['lower'] is None and bootstrap_groups([[1,2]],55,1000)['upper'] is None,'unsupported bootstrap remains null')
    interval=bootstrap_groups([[1,2],[0,2]],55,1000);check_interval(interval,{'replicates':1000,'confidence':.95,**interval})
    with tempfile.TemporaryDirectory(prefix='early-reader-path-source-') as temp:
        base=Path(temp).resolve();records=[];hashes=[]
        for i in range(3):
            path=base/f'file-{i}';path.write_text('artificial '+str(i));records.append({'path':path.name,'bytes':path.stat().st_size,'sha256':sha(path)})
        admit_file_matrix(records,base,lambda path:hashes.append(path) or sha(path));check(len(hashes)==3,'whole-source/artifact positive branch')
        bad=[dict(x) for x in records];bad[-1]['path']='../escape';hashes.clear();reject(lambda:admit_file_matrix(bad,base,lambda path:hashes.append(path) or sha(path)))
        check(not hashes,'late invalid name before any hash')
        late=base/records[-1]['path'];late.unlink();os.link(base/records[0]['path'],late);hashes.clear()
        reject(lambda:admit_file_matrix(records,base,lambda path:hashes.append(path) or sha(path)));check(not hashes,'late hardlink before any hash')
        metadata=base/'metadata.json';metadata.write_text('{"scope":"artificial"}')
        check(read_json(metadata)=={'scope':'artificial'},'metadata positive path branch')
        redirect=base/'redirect.json';redirect.symlink_to(metadata);reject(lambda:read_json(redirect))
    for macro in ('RPB_SOURCE_ID','EVALUATION_SOURCE_ID','EARLY_MIXER_ADAPTER_SOURCE_ID'):
        check(compile_scope('g++ -D'+macro+'=\\"'+'a'*64+'\\" -c code/example.cpp -o result.o','code/example.cpp',macro)=='a'*64,'actual escaped compile-scope branch')
    prequality=b'REVIEWED_SCHEMA = False\nMEASURED_IMPLEMENTATION = False\nuntouched = 512\n'
    released=prequality.replace(b' = False',b' = True')
    check(flags_only_source_binding(released,prequality)['two_flag_reverse_byte_proof'],'actual prequality-to-release byte branch')
    reject(lambda:flags_only_source_binding(released.replace(b'512',b'2048'),prequality))
    reject(lambda:flags_only_source_binding(prequality,prequality))
    reject(lambda:flags_only_source_binding(released+b'REVIEWED_SCHEMA = True\n',prequality))
    module=symtable.symtable(Path(__file__).read_text(),str(__file__),'exec');scopes_seen=[];unresolved=set();references=0
    def walk(scope):
        nonlocal references
        scopes_seen.append(scope)
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global():
                references+=1
                if symbol.get_name() not in globals() and symbol.get_name() not in vars(builtins):unresolved.add(symbol.get_name())
        for child in scope.get_children():walk(child)
    walk(module);check(not unresolved,'all measured/conditional globals resolved '+str(sorted(unresolved)))
    check(ARCHIVES==start_archives,'source fixtures decode zero archive payloads')
    return {'status':'passed','protocol':PROTOCOL,'checks':CHECKS-start_checks,'negative_cases':negative,
            'global_scopes':len(scopes_seen),'global_references':references,'unresolved_globals':[],
            'archive_payload_reads':0,'quality_payload_hashes':0,'writer_source_schema':writer_sources,
            'measured_execution_enabled':bool(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION)}


def requirements():
    return {'protocol':PROTOCOL,'status':'released' if REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION else 'blocked-prospective',
        'measured_execution_enabled':bool(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION),'human_card_sha256':CARD_SHA,
        'external_fit_protocol':EXTERNAL_FIT_PROTOCOL,'implementation_fit_protocol':FIT_PROTOCOL,
        'checkpoint_roles_per_point':5,'continuation_state_roles':0,'planned_head_pipelines':210,'planned_individual_heads':420,
        'encoder_trajectories':10,'encoder_updates_each':512,'native_full_exports':120,'query_writers':20,'query_forwards':80,
        'historical_payload_roles':0,'ordinary_CUDA_checkpoint_body_decodes':0,'independent_model_forwards':0,'head_or_PCA_fits':0,
        'float64_tolerance':[ATOL,RTOL],'float32_tolerance':[F32_ATOL,F32_RTOL],
        'release_protocol':'immutable FALSE triple before generation; separate two-lines-only copy after completed inventory authorization',
        'interpretation':'descriptive fresh fixed-budget confirmation; no selection or numeric promotion'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--requirements',action='store_true');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--capsule');parser.add_argument('--output');parser.add_argument('--fixture-output');args=parser.parse_args()
    if args.requirements:print(json.dumps(requirements(),sort_keys=True));return
    if args.self_test:
        check(Path('/.dockerenv').is_file(),'source fixtures run inside managed container')
        result=self_test(Path('/embedding'))
        if args.fixture_output:
            path=Path(args.fixture_output);check(not path.exists(),'exclusive new source fixture record');path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,sort_keys=True));return
    check(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION,'unreleased reader refuses before any measured path/hash/archive access')
    check(Path('/.dockerenv').is_file() and args.capsule and args.output,'managed container and explicit capsule/new output required')
    capsule=Path(args.capsule);output=Path(args.output)
    check(capsule.is_absolute() and output.is_absolute() and capsule.resolve(strict=True)==capsule,'explicit absolute measured scope')
    repo=capsule.parents[3];allowed=repo/COHORT_ROOT/'audit-tools'
    check(output.is_relative_to(allowed) and output.resolve(strict=False)==output and not output.exists() and not output.is_symlink() and
          not output.is_relative_to(capsule),'exclusive audit-tools output outside all measured evidence')
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:result=run(capsule,output)
    except Exception as error:
        result={'protocol':PROTOCOL,'status':'failed','checks':CHECKS,'archive_decodes':ARCHIVES,'elapsed_seconds':time.monotonic()-start,
                'reader_sha256':sha(Path(__file__)),'error':str(error),'preserved_measured_capsule':str(capsule)}
        (output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True);raise
    path=output/'validation.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'checks':result['checks'],'archive_decodes':result['archive_decodes'],
                      'elapsed_seconds':result['elapsed_seconds'],'validation':str(path),'sha256':sha(path)}),flush=True)




def admit_source_roles(records, index):
    names = [row['path'] for row in records]
    admitted_relative_names(names)
    check(names == sorted(names) and records, 'whole sorted unique captured source matrix')
    check(required_source_roles().issubset(names), 'all executed writers and both exact SDK installation sources captured')
    # Admit every source name before accessing even the first source-record binding.
    for name in names:
        check(name.startswith('code/') or name in ('Makefile', 'dependencies.lock', 'setup.sh'),
              'source-only closure names; exact root setup.sh permission')
    for row in records:
        check(type(row['bytes']) is int and row['bytes'] > 0 and
              bool(re.fullmatch('[a-f0-9]{64}', row['sha256'])), 'declared source size/hash schema')
        for prefix in ('source/', 'admission/source/'):
            check(index[prefix + row['path']]['sha256'] == row['sha256'] and
                  index[prefix + row['path']]['bytes'] == row['bytes'], 'whole admitted source record binding')
    return names

def methods_for_task(task):
    check(task in ('lag_sign', 'amplitude'), 'two separately declared tasks')
    return METHODS

def decision_summary(cohorts):
    means = {}; flags = {}; per_master = []
    for view in ('validation_intact','validation_deleted'):
        rows = []
        for cohort in cohorts:
            task = next(t for t in cohort['tasks'] if t['task']=='lag_sign')
            row = {'timing_master':cohort['timing_master']}
            for method in ('native_late','native_early'):
                item = next(m for m in task['methods'] if m['method']==method)
                if item['status'] != 'measured' or len(item['repetitions']) != 3:
                    row[method] = None; continue
                scores = [r[view]['ridge'] for r in item['repetitions']]
                if any(s['accuracy'] is None for s in scores):
                    row[method] = None; continue
                row[method] = {'accuracy':math.fsum(s['accuracy'] for s in scores)/3,
                               'coverage':math.fsum(s['coverage'] for s in scores)/3}
            rows.append(row)
        defined = all(row['native_late'] is not None and row['native_early'] is not None for row in rows)
        if defined:
            means[view] = {method:{'ridge_mean':math.fsum(row[method]['accuracy'] for row in rows)/5,
                                  'ridge_worst_cohort':min(row[method]['accuracy'] for row in rows),
                                  'coverage_mean':math.fsum(row[method]['coverage'] for row in rows)/5}
                           for method in ('native_late','native_early')}
            flags[view+'_ridge_mean_no_worse'] = means[view]['native_early']['ridge_mean'] >= means[view]['native_late']['ridge_mean']
            flags[view+'_ridge_worst_no_worse'] = means[view]['native_early']['ridge_worst_cohort'] >= means[view]['native_late']['ridge_worst_cohort']
            flags[view+'_equal_per_master_coverage'] = all(row['native_early']['coverage']==row['native_late']['coverage'] for row in rows)
        else:
            means[view] = None
            for suffix in ('ridge_mean_no_worse','ridge_worst_no_worse','equal_per_master_coverage'): flags[view+'_'+suffix] = False
        per_master.append({'view':view,'rows':rows})
    reconstruction = {}
    for split in ('training','validation'):
        reconstruction[split]={}
        for role in ('late','early'):
            values=[next(e for e in c['encoders'] if e['role']==role and e['budget']==512)['fixed_query'][split]['standardized_mae'] for c in cohorts]
            reconstruction[split][role]=math.fsum(values)/5 if all(value is not None for value in values) else None
        flags[split+'_original_query_mae_mean_no_worse'] = all(value is not None for value in reconstruction[split].values()) and reconstruction[split]['early'] <= reconstruction[split]['late']
    return {'scope':'known development only; no promotion or selected budget','joint_direction_passed':all(flags.values()),
            'guards':flags,'timing_ridge':means,'original_query_mae':reconstruction,'per_master_primary':per_master,
            'amplitude_secondary':True,'intervals_are_within_master_conditional':True}



def required_source_roles():
    encoder='code/encoders/raw_patch_bottleneck_mae/'
    return {CARD,'Makefile','dependencies.lock','setup.sh','code/scripts/install-libtorch.py',
        'code/evaluation/src/early_mixer_confirmation_main.cpp','code/evaluation/include/frozen_role_guard.h',
        'code/shared/src/fixed_feature_readouts.cpp','code/shared/include/embedding/shared/fixed_feature_readouts.h',
        'code/shared/tests/fixed_feature_readouts_test.cpp','code/shared/src/paired_pooling.cpp',
        encoder+'src/workflow.cpp',encoder+'src/learning_curve_adapter.cpp',encoder+'src/early_mixer_adapter.cpp',
        encoder+'src/early_mixer_confirmation_adapter.cpp',encoder+'src/training_source_gain.cpp',
        encoder+'include/embedding/encoders/raw_patch_bottleneck_mae/early_mixer_adapter.h',
        encoder+'include/embedding/encoders/raw_patch_bottleneck_mae/early_mixer_confirmation_adapter.h',
        encoder+'include/embedding/encoders/raw_patch_bottleneck_mae/training_source_gain.h',
        encoder+'tests/early_mixer_model_test.cpp',encoder+'tests/early_mixer_adapter_test.cpp',
        encoder+'tests/early_mixer_confirmation_adapter_test.cpp',
        'code/scripts/prepare-fresh-decoder-replication.py','code/scripts/prepare-early-mixer-confirmation.py',
        'code/scripts/check-early-mixer-confirmation.sh','code/scripts/evaluate-early-mixer-confirmation.sh'}

def confirmation_expected(audit,info,cp,placement,point,ids,scopes,snapshot=False):
    fields={
        'artifact_kind':'rpb_early_mixer_confirmation_cuda_snapshot_v1' if snapshot else 'rpb_early_mixer_confirmation_binding_v1',
        'confirmation_protocol_id':PROTOCOL,'external_fit_protocol_id':EXTERNAL_FIT_PROTOCOL,
        'training_implementation_fit_protocol_id':FIT_PROTOCOL,'confirmation_scope':'quality',
        'confirmation_role':'early' if placement else 'late',
        'confirmation_instance_group':'RPB-v10.alt-04' if placement else 'RPB-v7.alt-04',
        'confirmation_adapter_source_fingerprint':scopes['confirmation_adapter'],
        'parent_core_writer_source_fingerprint':scopes['core_writer'],
        'parent_training_producer_source_fingerprint':scopes['curve_training'],
        'parent_snapshot_adapter_source_fingerprint':scopes['early_adapter'],
        'training_dataset_id':info['dataset_id'],'training_schema_id':info['schema_id'],'preprocessing_id':info['scaler_id'],
        'original_training_source_manifest':source_manifest(ids),
        'original_training_source_manifest_id':text(audit,'fit_source_manifest_id'),
        'initialization_seed':text(audit,'initialization_seed'),'actual_training_seed':text(audit,'actual_training_seed'),
        'training_policy_id':POLICY,'architecture_id':ARCHITECTURES[placement],'feature_units':'unitless,unitless,unitless',
        'snapshot_policy':'new-confirmation-binding;unchanged-reliability-CUDA-serving;no-old-quality-input'}
    for key,suffix in zip(('content_checkpoint','content_audit','content_scaler','content_training_raw'),
                          ('','.audit.pt','.scaler.pt','.training-raw.pt')):
        fields[key]='fnv1a64-runtime-content-v1-'+fnv(Path(str(cp)+suffix).read_bytes())
    if snapshot:fields['content_confirmation']='fnv1a64-runtime-content-v1-'+fnv(Path(str(cp)+'.confirmation.pt').read_bytes())
    values={'channel_mixer_placement_value':placement,'attempted_steps':point,'completed_steps':point,
            'sampled_rows':point*8,'parameter_count_value':225805}
    values.update(dict(zip(('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'),info['counts'])))
    return fields,values

def confirmation_binding(asset,audit,info,cp,placement,point,ids,scopes,snapshot=False):
    fields,values=confirmation_expected(audit,info,cp,placement,point,ids,scopes,snapshot)
    check(set(asset)==set(fields)|set(values),'closed separate confirmation binding schema; no fabricated continuation')
    for key,wanted in fields.items():check(text(asset,key)==wanted,'exact confirmation versus implementation identity '+key)
    for key,wanted in values.items():check(scalar(asset,key)==wanted,'typed confirmation point/counter '+key)
    check(text(audit,'protocol_id')==FIT_PROTOCOL and FIT_PROTOCOL!=EXTERNAL_FIT_PROTOCOL,
          'truthful original implementation protocol retained, not retagged')
    check(all(x.startswith(EXTERNAL_FIT_PROTOCOL+'/') for x in ids),'external cohort source namespace bound separately')
    return fields

def snapshot_with_confirmation(original,side,supplement,audit,info,cp,placement,point,ids,scopes,initial):
    record=snapshot_binding(original,audit,info,cp,placement,point,scopes,initial)
    confirmation_binding(side,audit,info,cp,placement,point,ids,scopes)
    confirmation_binding(supplement,audit,info,cp,placement,point,ids,scopes,True)
    base=('RPB-v10' if placement else 'RPB-v7')+'; immutable timing-protocol checkpoint; exact contextual-global32 CUDA serving; ordinary original-Q decoder'
    check(record['extractor_provenance']==base+'; '+base,'unchanged delegated provider and extracted numerical provenance')
    record['extractor_provenance']=base+'; '+base+';external-confirmation-cohort='+EXTERNAL_FIT_PROTOCOL+';training-implementation='+FIT_PROTOCOL
    record.update({'external_fit_protocol':EXTERNAL_FIT_PROTOCOL,'implementation_fit_protocol':FIT_PROTOCOL,
                   'confirmation_sidecar_sha256':sha(Path(str(cp)+'.confirmation.pt')),
                   'confirmation_adapter_source_fingerprint':scopes['confirmation_adapter']})
    return record

def initialization_binding(value,master,scopes):
    expected={'common_parameters_exact':'true','common_buffers_exact':'true','scaler_exact':'true','training_dataset_exact':'true',
        'counter_streams_exact':'true','parameter_count':'225805','late_architecture_id':ARCHITECTURES[0],
        'early_architecture_id':ARCHITECTURES[1],'initialization_seed':str(mixed(master^0x7270622d696e6974)),
        'initial_features_equality_required':'false','snapshot_loader_source_fingerprint':scopes['early_adapter'],
        'confirmation_protocol_id':PROTOCOL,'external_fit_protocol_id':EXTERNAL_FIT_PROTOCOL,
        'training_implementation_fit_protocol_id':FIT_PROTOCOL,'confirmation_adapter_source_fingerprint':scopes['confirmation_adapter'],
        'historical_quality_input_roles':'0'}
    check(value==expected,'complete new confirmation paired initialization and dual source scopes')

def sdk_source_fixtures(repo):
    result=[]
    for name in ('setup.sh','code/scripts/install-libtorch.py'):
        path=repo/name
        check(path.is_file() and path.resolve(strict=True)==path and not path.is_symlink() and path.stat().st_nlink==1,'exact internal SDK SOURCE role')
        result.append({'path':name,'sha256':sha(path)})
    check('/opt/cuwacunu_embedding/libtorch' in (repo/'setup.sh').read_text(),'internal SDK setup literal')
    return result

def self_test(repo):
    baseline=_base_self_test(repo);negative=baseline['negative_cases'];start_archives=ARCHIVES
    def reject(call):
        nonlocal negative
        try:call()
        except (AssertionError,RuntimeError,ValueError,KeyError):negative+=1
        else:raise AssertionError('new confirmation negative accepted')
    ids=[f'{PROTOCOL}/lag_sign/seed-{TIMING_MASTERS[0]}/lag_sign/source-{i//2}' for i in range(256)]
    scopes={'core_writer':'1'*64,'curve_training':'2'*64,'early_adapter':'3'*64,'confirmation_adapter':'4'*64}
    with tempfile.TemporaryDirectory(prefix='confirmation-source-only-') as tmp:
        cp=Path(tmp)/'checkpoint.pt'
        for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.confirmation.pt'):Path(str(cp)+suffix).write_bytes(('artificial '+suffix).encode())
        for placement in (0,1):
            for point in (0,512):
                audit=fixture_parent(TIMING_MASTERS[0],placement,point,ids);audit['fit_source_manifest_id']=fake_text('manifest-fixture')
                info=parent_audit(audit,TIMING_MASTERS[0],placement,point,ids);info['schema_id']='schema-fixture'
                for snapshot in (False,True):
                    fields,values=confirmation_expected(audit,info,cp,placement,point,ids,scopes,snapshot)
                    asset={k:fake_text(v) for k,v in fields.items()};asset.update({k:fake('LongStorage',[],[v]) for k,v in values.items()})
                    confirmation_binding(asset,audit,info,cp,placement,point,ids,scopes,snapshot)
                    for key,value in [('external_fit_protocol_id',FIT_PROTOCOL),('training_implementation_fit_protocol_id',EXTERNAL_FIT_PROTOCOL),
                                      ('confirmation_role','late' if placement else 'early'),('parent_snapshot_adapter_source_fingerprint','4'*64),
                                      ('confirmation_instance_group','RPB-v7'),('content_checkpoint','corrupt')]:
                        bad=dict(asset);bad[key]=fake_text(value);reject(lambda:confirmation_binding(bad,audit,info,cp,placement,point,ids,scopes,snapshot))
                    bad=dict(asset);bad['completed_steps']=fake('LongStorage',[],[point+1]);reject(lambda:confirmation_binding(bad,audit,info,cp,placement,point,ids,scopes,snapshot))
                    bad=dict(asset);bad['unexpected']=fake_text('bad');reject(lambda:confirmation_binding(bad,audit,info,cp,placement,point,ids,scopes,snapshot))
    # Execute exact source-name and object-specific macro admission branches.
    names=sorted(required_source_roles());records=[{'path':n,'bytes':1,'sha256':'a'*64} for n in names]
    index={prefix+n:{'bytes':1,'sha256':'a'*64} for n in names for prefix in ('source/','admission/source/')}
    admit_source_roles(records,index)
    reject(lambda:admit_source_roles(records+[{'path':'unexpected-root.sh','bytes':1,'sha256':'a'*64}],index))
    reject(lambda:admit_source_roles([r for r in records if r['path']!='setup.sh'],index))
    reject(lambda:admit_source_roles([r for r in records if r['path']!='code/scripts/install-libtorch.py'],index))
    log='g++ -DEARLY_MIXER_ADAPTER_SOURCE_ID=\\"'+'a'*64+'\\" -c code/example.cpp -o early_mixer_adapter.o\n'
    log+='g++ -DEARLY_MIXER_ADAPTER_SOURCE_ID=\\"'+'b'*64+'\\" -c code/example.cpp -o early_mixer_curve_adapter.o\n'
    check(compile_scope(log,'code/example.cpp','EARLY_MIXER_ADAPTER_SOURCE_ID','early_mixer_adapter.o')=='a'*64,'object-specific truthful delegate scope')
    reject(lambda:compile_scope(log+log.splitlines()[0]+'\n','code/example.cpp','EARLY_MIXER_ADAPTER_SOURCE_ID','early_mixer_adapter.o'))
    initial_json={'common_parameters_exact':'true','common_buffers_exact':'true','scaler_exact':'true','training_dataset_exact':'true',
        'counter_streams_exact':'true','parameter_count':'225805','late_architecture_id':ARCHITECTURES[0],
        'early_architecture_id':ARCHITECTURES[1],'initialization_seed':str(mixed(TIMING_MASTERS[0]^0x7270622d696e6974)),
        'initial_features_equality_required':'false','snapshot_loader_source_fingerprint':scopes['early_adapter'],
        'confirmation_protocol_id':PROTOCOL,'external_fit_protocol_id':EXTERNAL_FIT_PROTOCOL,'training_implementation_fit_protocol_id':FIT_PROTOCOL,
        'confirmation_adapter_source_fingerprint':scopes['confirmation_adapter'],'historical_quality_input_roles':'0'}
    initialization_binding(initial_json,TIMING_MASTERS[0],scopes)
    bad=dict(initial_json);bad['training_implementation_fit_protocol_id']=EXTERNAL_FIT_PROTOCOL
    reject(lambda:initialization_binding(bad,TIMING_MASTERS[0],scopes))
    synthetic=[]
    for master in TIMING_MASTERS:
        methods=[]
        for name in ('native_late','native_early'):
            scores={'accuracy':.75,'coverage':1.}
            repetitions=[{'validation_intact':{'ridge':dict(scores)},'validation_deleted':{'ridge':dict(scores)}} for _ in range(3)]
            methods.append({'method':name,'status':'measured','repetitions':repetitions})
        synthetic.append({'timing_master':master,'tasks':[{'task':'lag_sign','methods':methods}],
            'encoders':[{'role':role,'budget':512,'fixed_query':{split:{'standardized_mae':.1} for split in ('training','validation')}} for role in ('late','early')]})
    check(decision_summary(synthetic)['joint_direction_passed'],'defined equal descriptive directions')
    synthetic[-1]['tasks'][0]['methods'][1]['repetitions'][0]['validation_deleted']['ridge']['accuracy']=None
    result=decision_summary(synthetic)
    check(result['timing_ridge']['validation_deleted'] is None and not result['joint_direction_passed'],'zero-support view retained as null, never discarded or rescued')
    synthetic[-1]['encoders'][1]['fixed_query']['validation']['standardized_mae']=None
    check(decision_summary(synthetic)['original_query_mae']['validation']['early'] is None,'unsupported original-query MAE retained as null')
    check(ARCHIVES==start_archives,'new confirmation fixtures have no archive decodes')
    baseline.update({'checks':CHECKS,'negative_cases':negative,'sdk_source_schema':sdk_source_fixtures(repo),
                     'source_sha256':sha(Path(__file__)),'measured_execution_enabled':bool(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION)})
    module=symtable.symtable(Path(__file__).read_text(),str(__file__),'exec');unresolved=[];scopes_seen=[];refs=0
    def walk(scope):
        nonlocal refs
        scopes_seen.append(scope)
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global():
                refs+=1
                if symbol.get_name() not in globals() and symbol.get_name() not in vars(builtins):unresolved.append(symbol.get_name())
        for child in scope.get_children():walk(child)
    walk(module);check(not unresolved,'all confirmation measured/schema globals resolved')
    baseline.update({'global_scopes':len(scopes_seen),'global_references':refs,'unresolved_globals':unresolved,'checks':CHECKS})
    return baseline

if __name__ == "__main__": main()
