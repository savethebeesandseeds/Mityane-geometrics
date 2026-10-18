#!/usr/bin/env python3
"""Independent pooled temporal context saved-arithmetic reader with release gates.

Only pinned archive-codec SOURCE is imported. Pure arithmetic was copied from
the immutable reviewed gain/curve-reader SOURCE, never its historical main. The code,
schema, artificial fixtures and peer review are sealed before quality with both
flags false. A NEW flags-only copy needs explicit completed-inventory release.
CUDA checkpoint bodies stay byte-bound and are never CPU-decoded.
"""
import argparse
import array
import ast
import builtins
import collections
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
PROTOCOL = 'pooled-context-v1'
CARD = 'code/evaluation/cards/pooled_context_v1.md'
CARD_SHA = 'b9f3f92e69cb55295dafa2e9f5d0d776b0b8c8bcde2762c241dfb225dbaf4fa7'
COHORT_ROOT = 'output/runs/rpb-pooled-context'
TIMING_MASTERS = (53151,54252,55353,56454,57555)
AMPLITUDE_MASTERS = (58656,59757,60858,61959,63060)
HEAD_REPETITIONS = (2701,2802,2903)
BUDGETS = (0,512)
METHODS = {'raw':576,'mask_metadata':288,'pca_only':32,'untrained_compact':32,'untrained_pooled':32,'native_compact':32,'native_pooled':32}
VIEWS = ('training','validation-intact','validation-deleted')
FIT_PROTOCOL = PROTOCOL+'/lag_sign'
COUNTER_POLICY = 'splitmix64-counter-rows-masks-torch-attempt-v1'
POLICY = 'rpb-training-context-deletion-015-v1'
PARAMETER_COUNTS = (225805,231949)
SHARED_PARAMETERS = 219469
NONSHARED_PARAMETER = 'global_pool_first.weight'
INACTIVE_PARAMETERS = ('export_projection.weight','export_projection.bias')
ARCHITECTURES = ('aligned-mixer-before-temporal-v1','aligned-mixer-before-temporal-pooled-context-width-v1')
OUTPUT_SEMANTICS = ('local_observed_contextual_pretemporal_aligned_global_semantic_mlp_bottleneck_v1',
                    'local_diagnostic_observed_contextual_pretemporal_aligned_pooled_W_global_semantic_mlp_bottleneck_v1')
RECONSTRUCTION_SEMANTICS = ('exact_pretemporal_contextual_observed_global_semantic_mlp_export_v1',
                          'exact_pretemporal_contextual_observed_pooled_W_global_semantic_mlp_export_v1')
INPUT_SEMANTICS = ('semantic_ordered_projected_D_channel_summaries_v1',
                   'semantic_ordered_temporally_pooled_W_before_diagnostic_D_projection_v1')
VIEW_STREAMS = {'lag_sign':0x7063763174696d30,'amplitude':0x70637631616d7030}
HELPER = 'output/runs/archive-controls/input-audit-20261006T191016Z-d7b99529/validate_phase1.py'
HELPER_SHA = '4eb501222fb1d9205ae13c5bc0bf1b5fc96ebd2faef3ed247dd7809fb86a453d'
COPIED_READER_SOURCE_SHA256 = 'ee43b13788935692536268d84e5daeb899bf5e86f24652c16afad35b6320f8ab'
MASK64 = (1<<64)-1
ATOL = RTOL = 2e-9
F32_ATOL,F32_RTOL = 2e-6,2e-5
CHECKS = 0
ARCHIVES = 0
R = None
REVIEWED_SCHEMA = False
MEASURED_IMPLEMENTATION = False
CONTEXT_LITERALS = {
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
    check(set(outer) == {'feature_mean', 'feature_scale', 'fitted_rows'},
          'one shared driver raw TRAIN outer map')
    fm = tensor(outer['feature_mean'], 'DoubleStorage', [576])
    fs = tensor(outer['feature_scale'], 'DoubleStorage', [576])
    check(scalar(outer, 'fitted_rows') == sum(valids[0]), 'one shared raw TRAIN fit population')
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


def progress(value, audit_info, point=None):
    if point is None:
        point = value['completed']
    check(point in BUDGETS and value['attempted'] == value['completed'] == point and
          value['sampled_rows'] == point * 8 and value['parameter_count'] ==
          value['cuda_parameter_count'] == audit_info['parameter_count'] and value['training_device'] in ('cuda', 'cuda:0'),
          'exact continuous CUDA point/cumulative counters')
    for key in ('last_input_cuda', 'last_loss_cuda', 'finite_gradients', 'weights_changed'):
        check(value[key] is bool(point), 'actual point0/positive CUDA flag ' + key)
    close(value['training_seconds'], audit_info['training_seconds'], 'same cumulative point timer')
    check(value['training_dataset_id'] == audit_info['dataset_id'] and
          value['preprocessing_id'] == audit_info['scaler_id'], 'point fit/scaler association')
    trace = value['losses']
    check(len(trace) == point, 'complete absolute trace; no sparse log subset')
    for i, row in enumerate(trace, 1):
        check(len(row) == 5 and row[0] == row[1] == i and type(row[2]) is int and row[2] > 0 and
              math.isfinite(row[3]) and row[3] >= 0 and math.isfinite(row[4]) and row[4] >= 0,
              'ordered unskipped absolute attempt/target/loss/gradient trace')
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


def original_point_prefix(zero,positive):
    for key in ('parameters','buffers'):
        check(set(zero[key]) == set(positive[key]),'same registered state at points0/512')
    check(any(bytes_of(zero['parameters'][name]) != bytes_of(positive['parameters'][name]) for name in zero['parameters']),
          'actual positive model update witness')
    check(zero['trace'] == [] and positive['trace'][0][0:2] == [1,1] and len(positive['trace']) == 512,
          'full absolute continuous prefix without skipping')


def decision_summary(cohorts):
    means = {}; flags = {}; per_master = []
    for view in ('validation_intact','validation_deleted'):
        rows = []
        for cohort in cohorts:
            task = next(t for t in cohort['tasks'] if t['task']=='lag_sign')
            row = {'timing_master':cohort['timing_master']}
            for method in ('native_compact','native_pooled'):
                item = next(m for m in task['methods'] if m['method']==method)
                if item['status'] != 'measured' or len(item['repetitions']) != 3:
                    row[method] = None; continue
                scores = [r[view]['ridge'] for r in item['repetitions']]
                if any(s['accuracy'] is None for s in scores):
                    row[method] = None; continue
                row[method] = {'accuracy':math.fsum(s['accuracy'] for s in scores)/3,
                               'coverage':math.fsum(s['coverage'] for s in scores)/3}
            rows.append(row)
        defined = all(row['native_compact'] is not None and row['native_pooled'] is not None for row in rows)
        if defined:
            means[view] = {method:{'ridge_mean':math.fsum(row[method]['accuracy'] for row in rows)/5,
                                  'ridge_worst_cohort':min(row[method]['accuracy'] for row in rows),
                                  'coverage_mean':math.fsum(row[method]['coverage'] for row in rows)/5}
                           for method in ('native_compact','native_pooled')}
            flags[view+'_ridge_mean_no_worse'] = means[view]['native_pooled']['ridge_mean'] >= means[view]['native_compact']['ridge_mean']
            flags[view+'_ridge_worst_no_worse'] = means[view]['native_pooled']['ridge_worst_cohort'] >= means[view]['native_compact']['ridge_worst_cohort']
            flags[view+'_equal_per_master_coverage'] = all(row['native_pooled']['coverage']==row['native_compact']['coverage'] for row in rows)
        else:
            means[view] = None
            for suffix in ('ridge_mean_no_worse','ridge_worst_no_worse','equal_per_master_coverage'): flags[view+'_'+suffix] = False
        per_master.append({'view':view,'rows':rows})
    reconstruction = {}
    for split in ('training','validation'):
        reconstruction[split] = {role:math.fsum(next(e for e in c['encoders'] if e['role']==role and e['budget']==512)['fixed_query'][split]['mae']
                                              for c in cohorts)/5 for role in ('control','candidate')}
        flags[split+'_original_query_mae_mean_no_worse'] = reconstruction[split]['candidate'] <= reconstruction[split]['control']
    return {'scope':'known development only; no promotion or selected budget','joint_direction_passed':all(flags.values()),
            'guards':flags,'timing_ridge':means,'original_query_mae':reconstruction,'per_master_primary':per_master,
            'amplitude_secondary':True,'intervals_are_within_master_conditional':True}


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
    expected_ids = {'native_pooled_minus_native_compact'}
    check(len(report['pairs']) == len(budgets) * 6, 'one declared comparison per budget/rep/view')
    pairs = []
    for record in report['pairs']:
        check(record['id'] in expected_ids and record['repetition'] in [f'rep-{r}' for r in HEAD_REPETITIONS] and
              record['view'] in ('validation_intact', 'validation_deleted'), 'explicit budget-specific vector comparison')
        candidate = 'native_pooled'; reference = 'native_compact'
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


def capsule_binding(capsule):
    check(capsule.is_absolute() and capsule.is_dir() and capsule.resolve(strict=True)==capsule and
          capsule.parent.as_posix().endswith('/'+COHORT_ROOT) and capsule.name.startswith('pooled-context-'), 'exclusive declared new capsule')
    inv=read_json(capsule/'artifact-integrity.json')
    check(inv['protocol']==PROTOCOL and inv['inventory_excludes_itself'] and inv['checksum_algorithm']=='sha256-file-bytes' and
          inv['file_count']==len(inv['files']) and inv['total_bytes']==sum(x['bytes'] for x in inv['files']), 'complete declared new inventory')
    index=admit_file_matrix(inv['files'],capsule)
    actual={x.relative_to(capsule).as_posix() for x in capsule.rglob('*') if x.is_file() or x.is_symlink()}
    check(actual==set(index)|{'artifact-integrity.json'},'no untracked capsule files before any payload hashes')
    for name,row in index.items(): check(sha(capsule/name)==row['sha256'],'all new inventoried bytes')
    return index


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
    for marker in ('Early mixer model CUDA admission passed','Early mixer CUDA adapter admission passed','Matched target gain CUDA admission passed',
                   'Pooled context model CUDA admission passed','Pooled context adapter CUDA admission passed','Fixed feature readout tests passed',
                   'Frozen role guard checks passed','Container SDK proof:',source):
        check(marker in log,'actual source-only admission marker '+marker)
    check('/embedding/.external/libtorch' not in log and 'not found' not in log,'pinned internal SDK resolution')
    scopes={'core_writer':compile_scope(log,'code/encoders/raw_patch_bottleneck_mae/src/workflow.cpp','RPB_SOURCE_ID'),
            'pooled_training':compile_scope(log,'code/encoders/raw_patch_bottleneck_mae/src/learning_curve_adapter.cpp','EVALUATION_SOURCE_ID'),
            'pooled_adapter':compile_scope(log,'code/encoders/raw_patch_bottleneck_mae/src/pooled_context_adapter.cpp','POOLED_CONTEXT_ADAPTER_SOURCE_ID','pooled_context_adapter.o')}
    check(compile_scope(log,'code/evaluation/src/pooled_context_main.cpp','EVALUATION_SOURCE_ID')==source and scopes['pooled_adapter']==source,'enclosing main/loader scope')
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
        for role in ('control','candidate'):
            for point in BUDGETS:
                base = prefix+f'{role}/point-{point}/'
                expected.update(base+'checkpoint.pt'+suffix for suffix in
                                ('','.audit.pt','.scaler.pt','.training-raw.pt','.continuation.pt'))
                expected.add(base+'snapshot-assets/pooled-context-snapshot-audit.pt')
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


def fixture_scaler():
    return {'mean':fake('DoubleStorage',[3,3],[.25]*9),'scale':fake('DoubleStorage',[3,3],[2.]*9),
            'count':fake('LongStorage',[3,3],[4096]*9),'channel_ids':fake('LongStorage',[3],[0,1,2]),
            'floor_applied':fake('BoolStorage',[3,3],[False]*9),'scale_floor':fake('DoubleStorage',[],[1e-6])}


def fixture_named(values):
    return {'count':fake('LongStorage',[],[len(values)]),
            **{f'tensor_{i}':{'parameter_name':fake_text(name),'value':value}
               for i,(name,value) in enumerate(values.items())}}


def fixture_query(data,mask,ids,frozen):
    rows=len(ids);requested,visible,query,eligible=query_masks(mask,rows);mean,scale,_=scaler(frozen)
    target=[f32((data[i%(rows*288)]-mean[((i//96)%3)*3+i%3])/scale[((i//96)%3)*3+i%3]) if keep else 0.
            for i,keep in enumerate(query)]
    prediction=[value+.25 if keep else 0. for value,keep in zip(target,query)]
    reduction=reductions(prediction,target,query,rows)
    asset={'standardized_prediction':fake('DoubleStorage',[4,rows,3,32,3],prediction),
           'standardized_target':fake('DoubleStorage',[4,rows,3,32,3],target),
           'target_mask':fake('BoolStorage',[4,rows,3,32,3],query),
           'requested_observed_target_mask':fake('BoolStorage',[4,rows,3,32,3],requested),
           'visible_mask':fake('BoolStorage',[4,rows,3,32,3],visible),
           'trial_channel_eligible':fake('BoolStorage',[4,rows,3],eligible),'source_ids_json':fake_text(json.dumps(ids))}
    for key,dtype,shape,values in (
        ('channel_target_counts','LongStorage',[rows,3],reduction['counts']),
        ('channel_valid','BoolStorage',[rows,3],[bool(x) for x in reduction['counts']]),
        ('channel_standardized_mae','DoubleStorage',[rows,3],reduction['channel_mae']),
        ('channel_standardized_huber','DoubleStorage',[rows,3],reduction['channel_huber']),
        ('example_valid','BoolStorage',[rows],reduction['example_valid']),
        ('example_standardized_mae','DoubleStorage',[rows],reduction['example_mae']),
        ('example_standardized_huber','DoubleStorage',[rows],reduction['example_huber'])):
        asset[key]=fake(dtype,shape,values)
    return asset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements',action='store_true');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--capsule');parser.add_argument('--output');parser.add_argument('--fixture-output');args=parser.parse_args()
    if args.requirements:print(json.dumps(requirements(),sort_keys=True));return
    if args.self_test:
        check(Path('/.dockerenv').is_file(),'SOURCE fixtures run in managed container')
        result=self_test(Path('/embedding'))
        if args.fixture_output:
            path=Path(args.fixture_output);check(path.is_absolute() and not path.exists(),'exclusive source fixture output')
            path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,sort_keys=True));return
    check(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION,'unreleased reader rejects before measured paths/hashes/codec')
    check(Path('/.dockerenv').is_file() and args.capsule and args.output,'managed container and explicit scope')
    capsule=Path(args.capsule);output=Path(args.output)
    check(capsule.is_absolute() and output.is_absolute() and capsule.resolve(strict=True)==capsule,'explicit absolute completed capsule')
    allowed=capsule.parents[3]/COHORT_ROOT/'audit-tools'
    check(output.is_relative_to(allowed) and output.resolve(strict=False)==output and not output.exists() and not output.is_symlink() and
          not output.is_relative_to(capsule),'exclusive audit-tools output outside measured evidence')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:result=run(capsule,output)
    except Exception as error:
        result={'protocol':PROTOCOL,'status':'failed','checks':CHECKS,'archive_decodes':ARCHIVES,'elapsed_seconds':time.monotonic()-started,
                'reader_sha256':sha(Path(__file__)),'error':str(error),'preserved_measured_capsule':str(capsule)}
        (output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True);raise
    path=output/'validation.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'checks':result['checks'],'archive_decodes':result['archive_decodes'],
                      'elapsed_seconds':result['elapsed_seconds'],'validation':str(path),'sha256':sha(path)}),flush=True)

def parse_settings(value, placement, master, candidate=False):
    result={}
    for line in value.splitlines():
        check('=' in line,'canonical settings line');key,val=line.split('=',1)
        check(key not in result,'unique setting');result[key]=val
    expected={'channel_count':3,'history_length':32,'input_width':3,'patch_length':8,'encoder_width':64,
              'export_width':32,'num_layers':3,'num_heads':4,'feedforward_width':256,'decoder_hidden_width':128,
              'channel_mixer_layers':1,'global_bottleneck_mode':2,'channel_mixer_placement':1,'dropout':0,
              'huber_delta':1,'sampling_interval':1,'batch_size':8,'threads':1,'learning_rate':.001,
              'weight_decay':.0001,'gradient_clip_norm':1,'steps':512,'attempt_limit':1024,'log_every':1,
              'seed':master,'layer_norm_epsilon':1e-5,'mask_ratio':.25,'scale_floor':1e-6}
    check(placement==1,'fixed full early mixer')
    for key,val in expected.items():check(key in result and float(result[key])==val,'frozen setting '+key)
    if candidate:check(result.get('global_pool_input_source')=='1','candidate declared W input')
    else:check('global_pool_input_source' not in result,'ordinary source0 text omission preserved')
    check(result['device'] in ('cuda','cuda:0') and result['channel_ids'] in ('','0,1,2'),'CUDA canonical semantic channels')
    result['device']='cuda';result.pop('global_pool_input_source',None)
    return result


def pooled_metadata(audit,candidate):
    expected={'global_pool_input_source':str(int(candidate)),
              'global_pool_input_semantics':INPUT_SEMANTICS[int(candidate)],
              'pooled_shared_parameter_values':str(SHARED_PARAMETERS),
              'pooled_copied_parameter_values':str(SHARED_PARAMETERS if candidate else 0),
              'pooled_inactive_projection_values':str(2080 if candidate else 0),
              'pooled_loss_reachable_parameter_values':str(229869 if candidate else 225805),
              'pooled_nonshared_parameter_name':NONSHARED_PARAMETER,
              'pooled_initialization_policy':('copy-all-shared-named-parameters-and-buffers-from-compact-point0-before-AdamW;except-global_pool_first.weight' if candidate else
                                              'compact-control-independent-initialization;no-copy'),
              'continuation_state_artifact_kind':'rpb_pooled_context_continuation_state_v1',
              'continuation_state_suffix':'.continuation.pt',
              'continuation_state_policy':'live_named_CPU_model_buffers_scaler_AdamW_and_complete_trace_v1',
              'curve_skip_policy':'abort_ineligible_attempt;no_skipped_update_prefix_permitted'}
    for key,wanted in expected.items():check(text(audit,key)==wanted,'closed pooled metadata '+key)
    check(scalar(audit,'global_pool_input_source_value')==int(candidate),'typed global input source')
    for k,v in {'shared_parameter_values':SHARED_PARAMETERS,'copied_parameter_values':SHARED_PARAMETERS if candidate else 0,
                'inactive_projection_values':2080 if candidate else 0}.items():check(scalar(audit,k)==v,'typed initialization counts '+k)
    paired=json.loads(text(audit,'paired_initialization_json'))
    check(paired=={'input_source':int(candidate),'copied_before_AdamW':True,'shared_parameter_values':SHARED_PARAMETERS,
                   'copied_parameter_values':SHARED_PARAMETERS if candidate else 0,'inactive_projection_values':2080 if candidate else 0,
                   'nonshared_parameter_name':NONSHARED_PARAMETER,'reference_point0_path':text(audit,'pooled_reference_point0_path')},
          'complete typed paired initialization JSON')
    check(not any(key.startswith('source_gain') or key.startswith('gain_view') or key=='gain_manifest_id' for key in audit),
          'no magnitude recipe or new loss')


def parent_audit(audit,master,candidate,point,ids):
    role=int(candidate)
    check(text(audit,'artifact_kind')=='rpb_learning_curve_training_audit_v1' and
          text(audit,'protocol_id')==FIT_PROTOCOL and text(audit,'actual_training_seed')==str(master) and
          text(audit,'initialization_seed')==str(mixed(master^0x7270622d696e6974)) and
          text(audit,'fit_source_manifest')==source_manifest(ids),'label-free original TRAIN/init association')
    check(text(audit,'model_tag')==('RPB-v12' if candidate else 'RPB-v10') and
          text(audit,'architecture_id')==ARCHITECTURES[role] and text(audit,'channel_mixer_placement')=='1' and
          scalar(audit,'channel_mixer_placement_value')==1 and text(audit,'output_semantics')==OUTPUT_SEMANTICS[role] and
          text(audit,'reconstruction_export_semantics')==RECONSTRUCTION_SEMANTICS[role],'typed architecture/global decoder route')
    check(scalar(audit,'attempted_steps')==scalar(audit,'completed_steps')==point and scalar(audit,'sampled_rows')==point*8 and
          text(audit,'model_weight_update_budget')==str(point),'exact no-skip absolute counters')
    check(list(tensor(audit['channel_order'],'LongStorage',[3]))==[0,1,2] and
          scalar(audit,'sampling_interval','DoubleStorage')==1 and scalar(audit,'endpoint','DoubleStorage')==31,'typed semantic/time metadata')
    for key,wanted in CONTEXT_LITERALS.items():check(text(audit,key)==wanted,'original coordinate15 '+key)
    check(scalar(audit,'context_deletion_ratio_value','DoubleStorage')==.15 and
          scalar(audit,'context_deletion_stream_value')==0x6374782d64726f70,'typed context rate/stream')
    check(not any(k.startswith('context_deletion_schedule') or k in ('context_ordinary_attempts','context_deletion_attempts') for k in audit),
          'no balanced policy')
    counts=[scalar(audit,k) for k in ('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates')]
    check(0<=counts[1]<=counts[0]<=point*8*288 and counts[2]==counts[0]-counts[1],'exact context count law')
    seconds=scalar(audit,'training_seconds','DoubleStorage');finite_timer(seconds,'CUDA loop plus evidence')
    check(scalar(audit,'weights_changed','BoolStorage')==bool(point) and scalar(audit,'finite_gradients','BoolStorage')==bool(point) and
          (seconds>0 if point else seconds==0),'point0/positive training evidence')
    check(text(audit,'rng_policy')==COUNTER_POLICY and
          text(audit,'sampling_policy')=='with_replacement_counter_rows;sampled_rows_includes_no_update_attempts' and
          text(audit,'optimizer_policy')=='one_continuous_AdamW_state;absolute_completed_update_budgets' and
          text(audit,'training_policy_id')==POLICY,'original row/mask/Torch/AdamW/loss policy')
    pooled_metadata(audit,candidate)
    common=parse_settings(text(audit,'resolved_settings'),1,master,candidate)
    return {'counts':counts,'training_seconds':seconds,'common_settings':common,'dataset_id':text(audit,'training_dataset_id'),
            'scaler_id':text(audit,'preprocessing_id'),'core_source':text(audit,'core_writer_source_fingerprint'),
            'training_source':text(audit,'training_producer_source_fingerprint'),'parameter_count':PARAMETER_COUNTS[role]}


def replay_parent(audit,raw,scaler_asset,controlled,master,candidate,point,scopes):
    data,mask,ids,labels=controlled;info=parent_audit(audit,master,candidate,point,ids)
    check(info['core_source']==scopes['core_writer'] and info['training_source']==scopes['pooled_training'],'actual captured writer/learner scopes')
    check(text(audit,'source_fingerprint_scope')=='ordinary_checkpoint_field=core_writer;companion_audit_training_producer=evaluation_source','parent scope literal')
    check(text(raw,'artifact_kind')=='rpb_raw_uniform_history_v1' and scalar(raw,'format_version')==1,'ordinary raw kind')
    exact(raw['data'],fake('DoubleStorage',[256,3,32,3],data),'original TRAIN values')
    exact(raw['observed'],fake('BoolStorage',[256,3,32,3],mask),'original TRAIN support')
    check(list(tensor(raw['channel_ids'],'LongStorage',[3]))==[0,1,2] and list(tensor(raw['endpoints'],'DoubleStorage',[256]))==[31.]*256 and
          scalar(raw,'sampling_interval','DoubleStorage')==1.,'raw time/semantic IDs')
    check(text(raw,'feature_units')==text(audit,'feature_units')=='unitless,unitless,unitless','original units')
    check(text(scaler_asset,'artifact_kind')=='rpb_frozen_training_scaler_v1' and scalar(scaler_asset,'format_version')==1,'frozen scaler kind')
    frozen=group(scaler_asset['scaler']);mean,scale,identity=scaler(frozen)
    check(identity==info['scaler_id']==text(scaler_asset,'preprocessing_id'),'frozen original scaler identity')
    check(text(raw,'schema_id')==text(scaler_asset,'schema_id') and text(raw,'dataset_id')==text(scaler_asset,'fit_dataset_id')==
          info['dataset_id']==text(audit,'scaler_fit_dataset_id'),'ordinary data/schema binding')
    count=tensor(frozen['count'],'LongStorage',[3,3]);floors=tensor(frozen['floor_applied'],'BoolStorage',[3,3])
    for c in range(3):
        for f in range(3):
            values=[data[b*288+c*96+h*3+f] for b in range(256) for h in range(32) if mask[b*288+c*96+h*3+f]];i=c*3+f
            check(len(values)==count[i] and values,'original observed scaler count')
            mu=math.fsum(values)/len(values);sd=math.sqrt(math.fsum((x-mu)**2 for x in values)/len(values))
            close(mean[i],mu,'original mean');close(scale[i],max(1e-6,sd),'original population scale')
            check(bool(floors[i])==(sd<1e-6),'scaler floor exact')
    info.update(schema_id=text(raw,'schema_id'),frozen_scaler=frozen)
    return info


def initial_witness(asset):
    expected={v+'_'+k for v in ('control','candidate') for k in ('parameters','buffers','scaler')}
    check(set(asset)==expected,'complete two-architecture initial state schema')
    params=[named_group(asset[v+'_parameters'],PARAMETER_COUNTS[i]) for i,v in enumerate(('control','candidate'))]
    buffers=[named_group(asset[v+'_buffers']) for v in ('control','candidate')]
    check(list(params[0])==list(params[1]) and list(buffers[0])==list(buffers[1]),'identical registration names/order')
    shared=0
    for name in params[0]:
        if name==NONSHARED_PARAMETER:
            tensor(params[0][name],'FloatStorage',[64,99]);tensor(params[1][name],'FloatStorage',[64,195])
        else:exact(params[0][name],params[1][name],'copied same-name/shape initial '+name);shared+=len(params[0][name]['values'])
    check(shared==SHARED_PARAMETERS,'exact common initialized scalar count')
    same_named(buffers[0],buffers[1],'all common buffers copied before AdamW')
    scalers=[group(asset[v+'_scaler']) for v in ('control','candidate')]
    check(set(scalers[0])==set(scalers[1]),'paired scaler schema')
    for k in scalers[0]:exact(scalers[0][k],scalers[1][k],'same original TRAIN scaler '+k)
    identity=scaler(scalers[0])[2]
    return {role:(params[i],buffers[i],scalers[i],identity) for i,role in enumerate(('control','candidate'))}


def continuation_state(asset,audit,info,cp,point,placement,initial,candidate=False):
    check(text(asset,'artifact_kind')=='rpb_pooled_context_continuation_state_v1','pooled LIVE continuation kind')
    check(text(asset,'checkpoint_path')==str(cp) and text(asset,'state_capture_policy')==
          'live_named_CUDA_parameter_state_to_CPU;no_model_forward;no_optimizer_reload_or_step','actual live CUDA state copy')
    audit_text={key for key,value in audit.items() if value.get('dtype')=='ByteStorage'}-{'artifact_kind','model_weight_update_budget'}
    for k in audit_text:check(text(asset,k)==text(audit,k),'exact continuation inherited text '+k)
    integers={'attempted_steps':point,'completed_steps':point,'sampled_rows':point*8,'channel_mixer_placement_value':1,
              'global_pool_input_source_value':int(candidate),'shared_parameter_values':SHARED_PARAMETERS,
              'copied_parameter_values':SHARED_PARAMETERS if candidate else 0,'inactive_projection_values':2080 if candidate else 0}
    for k,n in zip(('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'),info['counts']):integers[k]=n
    for k,n in integers.items():check(scalar(asset,k)==n,'LIVE typed counter/config '+k)
    check(scalar(asset,'training_seconds','DoubleStorage')==info['training_seconds'],'exact update timer copy')
    counters=list(tensor(asset['loss_trace_counters'],'LongStorage',[point,3]));values=list(tensor(asset['loss_trace_values'],'DoubleStorage',[point,2]));trace=[]
    for i in range(point):
        a,b,n=counters[3*i:3*i+3];loss,grad=values[2*i:2*i+2]
        check(a==b==i+1 and n>0 and math.isfinite(loss) and loss>=0 and math.isfinite(grad) and grad>=0,'complete successful absolute trace')
        trace.append([a,b,n,loss,grad])
    params=named_group(asset['model_parameters'],PARAMETER_COUNTS[int(candidate)]);buffers=named_group(asset['model_buffers'])
    check(list(params)==sorted(params) and list(buffers)==sorted(buffers),'lexical LIVE state')
    base_params,base_buffers,base_scaler,identity=initial
    check(set(params)==set(base_params) and set(buffers)==set(base_buffers),'same registered architecture at retained points')
    for name in params:check(params[name]['dtype']==base_params[name]['dtype'] and params[name]['shape']==base_params[name]['shape'],'fixed own architecture '+name)
    saved_initial=named_group(asset['initial_model_parameters'],PARAMETER_COUNTS[int(candidate)])
    saved_initial_buffers=named_group(asset['initial_model_buffers'])
    same_named(saved_initial,base_params,'LIVE original initialized parameter witness')
    same_named(saved_initial_buffers,base_buffers,'LIVE original initialized buffer witness')
    if not point:same_named(params,base_params,'point0 initialized values');same_named(buffers,base_buffers,'point0 buffers')
    if candidate:
        exact(asset['candidate_first_before_copy'],base_params[NONSHARED_PARAMETER],'wider first retains its own fresh fan-in initialization')
        inactive=0
        for name in INACTIVE_PARAMETERS:exact(params[name],base_params[name],'unchanged diagnostic projection '+name);inactive+=len(params[name]['values'])
        check(inactive==2080,'exact inactive diagnostic registered values')
    frozen=group(asset['scaler']);check(set(frozen)==set(base_scaler),'LIVE frozen scaler keys')
    for k in frozen:exact(frozen[k],base_scaler[k],'original scaler unchanged '+k)
    check(scaler(frozen)[2]==identity==info['scaler_id'],'LIVE scaler identity')
    optimizer=group(asset['optimizer_state']);n=scalar(optimizer,'parameter_count');active=scalar(optimizer,'active_state_count')
    check(n==len(params) and 0<=active<=n and set(optimizer)=={'parameter_count','active_state_count'}|{f'parameter_{i}' for i in range(n)},'complete named optimizer matrix')
    states={}
    for i,name in enumerate(params):
        entry=group(optimizer[f'parameter_{i}']);check(text(entry,'parameter_name')==name and
              list(tensor(entry['parameter_shape'],'LongStorage',[len(params[name]['shape'])]))==params[name]['shape'],'named optimizer shape association')
        present=scalar(entry,'has_state','BoolStorage');expected={'parameter_name','parameter_shape','has_state'}
        if candidate and name in INACTIVE_PARAMETERS:check(not present,'inactive diagnostic projection has no fabricated AdamW state')
        if present:
            expected|={'step','exp_avg','exp_avg_sq'};step=scalar(entry,'step');check(0<step<=point,'active moment absolute step')
            first=tensor(entry['exp_avg'],params[name]['dtype'],params[name]['shape']);second=tensor(entry['exp_avg_sq'],params[name]['dtype'],params[name]['shape'])
            finite(first,'finite first moment');finite(second,'finite second moment');check(all(x>=0 for x in second),'nonnegative second moment')
            states[name]={'step':step,'first':entry['exp_avg'],'second':entry['exp_avg_sq']}
        check(set(entry)==expected,'exact active/inactive moment schema')
    check(len(states)==active and (active>0 if point else active==0),'initial empty optimizer, actual positive named states')
    typed=set(integers)|{'training_seconds','loss_trace_counters','loss_trace_values','model_parameters','model_buffers',
                        'initial_model_parameters','initial_model_buffers','scaler','optimizer_state'}
    if candidate:typed.add('candidate_first_before_copy')
    check(set(asset)==audit_text|typed|{'artifact_kind','checkpoint_path','state_capture_policy'},'complete continuation fields')
    return {'trace':trace,'parameters':params,'buffers':buffers,'optimizer':states,'trace_counters':asset['loss_trace_counters'],
            'trace_values':asset['loss_trace_values'],'optimizer_parameter_count':n,'optimizer_active_state_count':active,
            'registered_parameter_values':PARAMETER_COUNTS[int(candidate)],'inactive_projection_values':2080 if candidate else 0,
            'optimizer_replayed':False,'state_capture':'actual live CUDA state copied to CPU'}


def snapshot_binding(asset,audit,info,cp,candidate,point,scopes,initial):
    role=int(candidate);check(text(asset,'artifact_kind')=='rpb_pooled_context_cuda_snapshot_v1','pooled immutable snapshot kind')
    expected={'protocol_id':PROTOCOL,'original_training_protocol_id':FIT_PROTOCOL,'parent_checkpoint_path':str(cp),
              'parent_writer_source_fingerprint':scopes['core_writer'],'parent_training_producer_source_fingerprint':scopes['pooled_training'],
              'snapshot_loader_source_fingerprint':scopes['pooled_adapter'],
              'source_fingerprint_scope':'parent=core_writer_and_training_producer;loader=new_pooled_context_adapter',
              'original_encoder_attempted':str(point),'original_encoder_completed':str(point),'training_schema_id':info['schema_id'],
              'training_dataset_id':info['dataset_id'],'preprocessing_id':info['scaler_id'],'parameter_count':str(PARAMETER_COUNTS[role]),
              'encoder_updates':'0','decoder_updates':'0','head_refits':'0','no_optimizer_created':'true',
              'snapshot_policy':'independent_CUDA_model;eval_no_grad;frozen_timing_TRAIN_scaler;ordinary_query_no_context_erasure;no_refit'}
    for k,v in expected.items():check(text(asset,k)==v,'snapshot immutable association '+k)
    check(text(asset,'inference_device') in ('cuda','cuda:0'),'actual CUDA inference source/admission bound')
    for k,v in {'channel_mixer_placement_value':1,'global_pool_input_source_value':role,'original_encoder_attempted_value':point,'original_encoder_completed_value':point}.items():
        check(scalar(asset,k)==v,'snapshot typed original state '+k)
    for k,v in audit.items():
        if v.get('dtype')=='ByteStorage' and k not in set(expected)|{'artifact_kind'}:check(text(asset,k)==text(audit,k),'snapshot retained audit text '+k)
    for k,v in zip(('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'),info['counts']):check(text(asset,k)==str(v),'snapshot context counts')
    for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.continuation.pt'):
        check(text(asset,'parent_content_id'+suffix)=='fnv1a64-runtime-content-v1-'+fnv(Path(str(cp)+suffix).read_bytes()),'five byte-bound parent roles '+suffix)
    saved_scaler=group(asset['scaler']);check(set(saved_scaler)==set(info['frozen_scaler']),'snapshot scaler keys')
    for k in saved_scaler:exact(saved_scaler[k],info['frozen_scaler'][k],'snapshot ordinary scaler '+k)
    params=named_group(asset['model_parameters'],PARAMETER_COUNTS[role]);buffers=named_group(asset['model_buffers'])
    check(list(params)==sorted(params) and list(buffers)==sorted(buffers),'snapshot lexical named state')
    base_params,base_buffers,base_scaler,base_id=initial
    check(set(params)==set(base_params) and set(buffers)==set(base_buffers),'snapshot registered names')
    for name in params:check(params[name]['dtype']==base_params[name]['dtype'] and params[name]['shape']==base_params[name]['shape'],'snapshot own tensor shapes '+name)
    if not point:same_named(params,base_params,'snapshot own untrained parameters');same_named(buffers,base_buffers,'snapshot own untrained buffers')
    else:check(any(bytes_of(params[n])!=bytes_of(base_params[n]) for n in params),'positive actual model change')
    if candidate:
        for name in INACTIVE_PARAMETERS:exact(params[name],base_params[name],'snapshot inactive diagnostic unchanged '+name)
    for k in base_scaler:exact(saved_scaler[k],base_scaler[k],'all retained points use original scaler '+k)
    base=('RPB-v12' if candidate else 'RPB-v10')+'; immutable timing-protocol checkpoint; exact contextual-global32 CUDA serving; ordinary original-Q decoder'
    return {'extractor_provenance':base+'; '+base,'placement':1,'global_pool_input_source':role,'point':point,
            'architecture_id':ARCHITECTURES[role],'scaler_identity':info['scaler_id'],'parent_checkpoint_sha256':sha(cp),
            'checkpoint_body_decoded':False,'actual_encoder_execution':'source/admission-bound; no CPU model rerun'}


def fixed_plan(plan):
    expected={'protocol':PROTOCOL,'timing_master_seeds':list(TIMING_MASTERS),'amplitude_data_master_seeds':list(AMPLITUDE_MASTERS),
              'tags':['RPB-v10.alt-03','RPB-v12'],'encoder_trajectories':10,'encoder_updates_each':512,'decoder_updates':0,
              'extra_decoder_calibration_updates':0,'decoder_update_scope':'joint reconstruction during encoder updates; no extra decoder-only calibration',
              'batch_size':8,'sampled_rows':40960,'train_pairs':128,'validation_pairs':64,'test_pairs':0,'shape':[3,32,3],
              'native_width':32,'parameter_counts':[225805,231949],'global_pool_input_sources':[0,1],'shared_parameter_values':219469,
              'candidate_inactive_projection_values':2080,'candidate_loss_reachable_parameter_values':229869,'checkpoint_roles_per_point':5,
              'head_repetitions':list(HEAD_REPETITIONS),'planned_pipelines':210,'planned_heads':420,'attempt_limit':1024,
              'log_every':1,'methods_each_task':7,'retained_points':20,'separate_initial_controls':2,'planned_native_export_calls':120,
              'planned_unique_quality_native_exports':120,'planned_initial_counterpart_exports':0,'quality_gain':False,
              'planned_query_writer_calls':20,'planned_query_forwards':80,'driver_raw_outer_fits':10,'helper_outer_train_fits':50,
              'raw_outer_fit_shared_with_PCA':True,'skipped_attempts_permitted':False,'deletion_rate':.30,'encoder_device':'CUDA',
              'prior_quality_input_roles':[],'quality_generated':False,'testing_accessed':False,'stress_accessed':False,'selection':False,
              'promotion':False,'human_card_sha256':CARD_SHA}
    check(set(plan)==set(expected)|{'source_fingerprint'},'closed prospective pooled recipe')
    for k,v in expected.items():check(plan[k]==v,'declared fixed plan '+k)
    check(re.fullmatch('[a-f0-9]{64}',plan['source_fingerprint']) is not None,'actual enclosing fingerprint')


def required_source_roles():
    return {CARD,'Makefile','dependencies.lock','setup.sh','code/scripts/install-libtorch.py',
            'code/evaluation/src/pooled_context_main.cpp','code/encoders/raw_patch_bottleneck_mae/src/workflow.cpp',
            'code/encoders/raw_patch_bottleneck_mae/src/learning_curve_adapter.cpp',
            'code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h',
            'code/encoders/raw_patch_bottleneck_mae/src/pooled_context_adapter.cpp',
            'code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/pooled_context_adapter.h',
            'code/encoders/raw_patch_bottleneck_mae/tests/pooled_context_adapter_test.cpp',
            'code/encoders/raw_patch_bottleneck_mae/tests/pooled_context_model_test.cpp',
            'code/encoders/raw_patch_bottleneck_mae/tests/pooled_context_old_model_reference.h',
            'code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/model.h',
            'code/shared/src/paired_pooling.cpp','code/shared/src/fixed_feature_readouts.cpp',
            'code/scripts/prepare-pooled-context.py','code/scripts/check-pooled-context.sh','code/scripts/evaluate-pooled-context.sh'}


def run(capsule,output):
    started=time.monotonic();index=capsule_binding(capsule);source=source_binding(capsule,index)
    scopes=source['actual_compile_scopes'];import_helper(capsule.parents[3])
    results=capsule/'results';report=read_json(results/'report.json');complete=read_json(results/'complete.json')
    check(set(report)=={'protocol','source_fingerprint','cohorts','testing_accessed','stress_accessed','selection','promotion'} and
          report['protocol']==PROTOCOL and report['source_fingerprint']==source['source_fingerprint'] and len(report['cohorts'])==5 and
          all(report[k] is False for k in ('testing_accessed','stress_accessed','selection','promotion')),'complete fresh fixed five cohorts')
    archives=output_archive_matrix(results,report);audited=[];all_sources=set();pipelines=pairs=helper_fits=0
    for cohort,master,amp_master in zip(report['cohorts'],TIMING_MASTERS,AMPLITUDE_MASTERS):
        check(cohort['timing_master']==master and cohort['amplitude_data_master']==amp_master and
              len(cohort['encoder_points'])==2 and len(cohort['tasks'])==2,'fixed timing/amplitude pair association')
        root=results/f'seed-{master}-lag_sign';files=[load(root/('controlled-'+name+'.pt')) for name in ('training','validation','validation-deleted')]
        timing=[observations(v,rows,master,'lag_sign',i==2) for i,(v,rows) in enumerate(zip(files,(256,128,128)))]
        timing_view=check_view(timing[1],timing[2],master,'lag_sign',files[2]);initial=initial_witness(load(root/'initial-pair.pt'))
        check(read_json(root/'initial-pair.json')=={'common_named_parameters_exact':True,'shared_parameter_values':219469,
              'nonshared_parameter_name':NONSHARED_PARAMETER,'buffers_exact':True,'scaler_exact':True,'copied_before_AdamW':True,
              'initial_features_equality_required':False,'initial_control_sets':2,'per_step_trace_counts_exact':True,'sampled_rows_exact':True},
              'initial state/source stream declaration with independent untrained controls')
        init=read_json(root/'initialization-audit.json')
        check(init=={'common_parameters_exact':'true','common_buffers_exact':'true','scaler_exact':'true','training_dataset_exact':'true',
                     'counter_streams_exact':'true','shared_parameter_values':'219469','control_registered_parameter_values':'225805',
                     'candidate_registered_parameter_values':'231949','candidate_inactive_projection_values':'2080',
                     'candidate_loss_reachable_parameter_values':'229869','initial_features_equality_required':'false',
                     'nonshared_parameter_name':NONSHARED_PARAMETER,'copied_before_AdamW':'true','snapshot_loader_source_fingerprint':scopes['pooled_adapter']},
              'exact paired initialization gate; initial features are separate controls')
        providers={};infos={};states={};query_assets={};encoders=[]
        reference=root/'control/point-0/checkpoint.pt'
        for candidate,role in ((False,'control'),(True,'candidate')):
            role_states={}
            for point in BUDGETS:
                cp=root/role/f'point-{point}/checkpoint.pt';audit=load(Path(str(cp)+'.audit.pt'))
                reference_binding(audit,reference,candidate)
                info=replay_parent(audit,load(Path(str(cp)+'.training-raw.pt')),load(Path(str(cp)+'.scaler.pt')),timing[0],master,candidate,point,scopes)
                check(info['scaler_id']==initial[role][3],'original common timing scaler at every retained point')
                state=continuation_state(load(Path(str(cp)+'.continuation.pt')),audit,info,cp,point,1,initial[role],candidate)
                role_states[point]=state;infos[role,point]=info;states[role,point]=state
                if point:
                    actual=read_json(root/role/'encoder-progress.json');check(progress(actual,info,point)==state['trace'],'full JSON trace matches LIVE saved trace')
                else:actual=None
                snapshot=load(cp.parent/'snapshot-assets/pooled-context-snapshot-audit.pt')
                method=('native_pooled' if candidate else 'native_compact') if point else ('untrained_pooled' if candidate else 'untrained_compact')
                providers[method]=snapshot_binding(snapshot,audit,info,cp,candidate,point,scopes,initial[role])
                same_named(named_group(snapshot['model_parameters']),state['parameters'],'snapshot equals saved LIVE model')
                same_named(named_group(snapshot['model_buffers']),state['buffers'],'snapshot equals LIVE buffers')
                static=read_json(root/role/'trainer-audit.json')
                for key,value in static.items():
                    if key=='snapshot_adapter_source_fingerprint':check(value==scopes['pooled_adapter'],'new adapter static source')
                    elif key=='snapshot_policy':check(value=='new_pooled_protocol_bound_CUDA_only;historical_CPU_snapshot_not_called','no historical CPU encoder path')
                    else:check(value==text(audit,key),'static original TRAIN association '+key)
                queries={}
                if point:
                    declared=cohort['encoder_points'][int(candidate)]
                    check(declared['model_tag']==('RPB-v12' if candidate else 'RPB-v10.alt-03') and declared['placement']==1 and
                          declared['global_pool_input_source']==int(candidate) and declared['encoder_progress']==actual,
                          'all positive roles reported in fixed order')
                    for split,data in (('training',timing[0]),('validation',timing[1])):
                        asset=load(cp.parent/(split+'-reconstruction.pt'));replay=query_archive(asset,data,info['frozen_scaler']);query_assets[role,split]=asset
                        query_summary(declared[split+'_reconstruction'],replay,asset,split+'-reconstruction.pt')
                        queries[split]={k:replay[k] for k in ('mae','huber','valid_examples','total_examples','valid_target_cells','requested_observed_target_cells')}
                encoders.append({'tag':'RPB-v12' if candidate else 'RPB-v10.alt-03','design_tag':'RPB-v12' if candidate else 'RPB-v10',
                                 'role':role,'placement':1,'global_pool_input_source':int(candidate),'budget':point,'architecture_id':ARCHITECTURES[int(candidate)],
                                 'registered_parameter_values':PARAMETER_COUNTS[int(candidate)],'shared_parameter_values':SHARED_PARAMETERS,
                                 'inactive_projection_values':2080 if candidate else 0,'loss_reachable_parameter_values':229869 if candidate else 225805,
                                 'training_seconds':info['training_seconds'],'attempted':point,'completed':point,'sampled_rows':point*8,
                                 'context_counts':info['counts'],'optimizer_active_state_count':state['optimizer_active_state_count'],'fixed_query':queries})
            original_point_prefix(role_states[0],role_states[512])
        for point in BUDGETS:
            check(infos['control',point]['common_settings']==infos['candidate',point]['common_settings'] and
                  infos['control',point]['counts']==infos['candidate',point]['counts'],'same original recipe/settings excluding explicit input source')
            exact(states['control',point]['trace_counters'],states['candidate',point]['trace_counters'],'same unskipped original A/Q target support counts')
        for split in ('training','validation'):
            for key in ('standardized_target','target_mask','requested_observed_target_mask','visible_mask','trial_channel_eligible',
                        'channel_target_counts','channel_valid','example_valid','source_ids_json'):
                exact(query_assets['control',split][key],query_assets['candidate',split][key],'paired original query universe '+key)
        tasks=[]
        for task,task_master,declared in zip(('lag_sign','amplitude'),(master,amp_master),cohort['tasks']):
            check(declared['task']==task and declared['data_master']==task_master,'two declared task populations');path=root/task
            if task=='lag_sign':data,view=timing,timing_view
            else:
                tf=[load(path/('controlled-'+n+'.pt')) for n in ('training','validation','validation-deleted')]
                data=[observations(v,r,task_master,task,i==2) for i,(v,r) in enumerate(zip(tf,(256,128,128)))]
                view=check_view(data[1],data[2],task_master,task,tf[2])
            check(set(data[0][2]).isdisjoint(data[1][2]),'TRAIN and VALIDATION source separation')
            for split in data[:2]:
                ids=set(split[2]);check(all_sources.isdisjoint(ids),'all task/cohort source groups distinct');all_sources.update(ids)
            controls,status=control_rows(path,*data);summary,original=audit_readouts(path,data,controls,status,providers,task_master,task)
            check(original==declared['readouts'],'embedded exact saved readout report')
            fits=original['fit_counts'];pipelines+=fits['ridge_fits'];helper_fits+=fits['outer_train_normalizer_fits'];pairs+=len(original['pairs'])
            summary.update(task=task,validation_deletion=view,PCA_status=status,independent_initial_controls=2);tasks.append(summary)
        cost_numeric={'generation_and_observation_io_seconds','binding_and_checkpoint_io_seconds','CUDA_query_transfer_verification_io_seconds',
                      'CUDA_unique_quality_native_transfer_verification_seconds','CPU_baseline_preparation_io_seconds','CPU_head_bootstrap_io_seconds'}
        check(set(cohort['costs'])==cost_numeric|{'training_loop_scope'},'truthful mixed cost stages')
        for key in cost_numeric:finite_timer(cohort['costs'][key],key)
        check(cohort['costs']['training_loop_scope']=='CUDA updates plus CPU trace/live-state evidence capture; not pure kernel time','training evidence overhead disclosed')
        audited.append({'timing_master':master,'amplitude_data_master':amp_master,'encoders':encoders,'tasks':tasks,'costs':cohort['costs']})
        print('Audited pooled-context cohort '+str(master),flush=True)
    expected={'protocol':PROTOCOL,'status':'complete','cohorts':5,'tasks_each':2,'encoder_trajectories':10,'encoder_updates_each':512,
              'attempt_limit':1024,'retained_points':20,'sampled_rows':40960,'skipped_attempts':0,'head_pipelines':pipelines,'individual_heads':2*pipelines,
              'planned_pipelines':210,'planned_heads':420,'unique_quality_native_exports':120,'initial_counterpart_exports':0,
              'full_native_export_calls':120,'query_evaluation_calls':20,'necessary_query_forwards':80,
              'driver_raw_outer_fits':10,'helper_outer_train_fits':helper_fits,'separate_initial_controls':2,'initial_shared_state_exact_before_training_and_heads':True,
              'CPU_encoder_training':False,'CPU_encoder_forward':False,'testing_accessed':False,'stress_accessed':False,'selection':False,'promotion':False}
    check(complete==expected and pairs==60 and len(all_sources)==1920,'exact complete fixed work/paired matrix')
    for name,row in index.items():check(sha(capsule/name)==row['sha256'],'capsule unchanged after saved arithmetic')
    return {'protocol':PROTOCOL,'status':'passed','checks':CHECKS,'archive_decodes':ARCHIVES,'elapsed_seconds':time.monotonic()-started,
            'source':source,'cohorts':audited,'decision':decision_summary(audited),
            'counts':{'unique_source_groups':1920,'encoder_runs':10,'retained_points':20,'head_pipelines':pipelines,'individual_heads':2*pipelines,
                      'paired_records':60,'head_effects':120,'native_full_export_calls':120,'fixed_query_writer_calls':20,'fixed_query_banks':80,
                      'driver_raw_outer_fits':10,'helper_outer_train_fits':helper_fits,'new_archive_roles':len(archives),'independent_initial_controls':2},
            'limits':{'saved_arithmetic_only':True,'CUDA_execution':False,'model_forward_or_reconstruction':False,'autodiff_or_optimizer_execution':False,
                      'encoder_or_head_fits':0,'PCA_or_SVD_fits':0,'ordinary_CUDA_checkpoint_body_decodes':0,'historical_payload_reads':0,
                      'TEST_or_stress_reads':0,'across_encoder_seed_interval':False,'float32_target_tolerance':[F32_ATOL,F32_RTOL],
                      'float64_saved_math_tolerance':[ATOL,RTOL],'checkpoint_and_training':'actual named live CPU state/AdamW/traces plus captured CUDA admission; no trajectory rerun',
                      'inference':'captured CUDA source/admission and own-role saved snapshot arrays; no independent encoder execution',
                      'costs':'synchronized update loop includes CPU trace/evidence; inference/transfer/verification/I/O and CPU stages are mixed wall times',
                      'interpretation':'route and first-layer capacity change together; no causal projection-loss claim, selected cohort/head or promotion'}}


def reference_binding(audit,reference,candidate):
    check(text(audit,'pooled_reference_point0_path')==(str(reference) if candidate else ''),'only exact same-cohort control point0 is referenced')
    suffixes=('','.audit.pt','.scaler.pt','.training-raw.pt','.continuation.pt')
    keys={k for k in audit if k.startswith('pooled_reference_content_id')}
    check(keys==({'pooled_reference_content_id'+s for s in suffixes} if candidate else set()),'complete five-reference IDs or no control copy claim')
    if candidate:
        for s in suffixes:check(text(audit,'pooled_reference_content_id'+s)=='fnv1a64-runtime-content-v1-'+fnv(Path(str(reference)+s).read_bytes()),'exact compact point0 reference bytes '+s)


def schema_source_fixtures(repo):
    paths=['code/evaluation/src/pooled_context_main.cpp','code/shared/src/fixed_feature_readouts.cpp','code/shared/src/paired_pooling.cpp',
           'code/encoders/raw_patch_bottleneck_mae/src/learning_curve_adapter.cpp','code/encoders/raw_patch_bottleneck_mae/src/pooled_context_adapter.cpp',
           'code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/pooled_context_adapter.h',
           'code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/model.h',
           'code/encoders/raw_patch_bottleneck_mae/src/workflow.cpp',
           'code/encoders/raw_patch_bottleneck_mae/include/embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h',
           'code/scripts/prepare-pooled-context.py']
    for name in paths:
        p=repo/name;check(p.is_file() and p.resolve(strict=True)==p and not p.is_symlink() and p.stat().st_nlink==1,'all writer SOURCE admitted before hash')
    records=[{'path':p,'sha256':sha(repo/p)} for p in paths];main_source=(repo/paths[0]).read_text()
    body=main_source[main_source.index('void save_observations('):main_source.index('void initial_pair(')]
    check(set(re.findall(r'a\.write\("([^\"]+)"',body))=={'observations','feature_mask','labels_scoring_only','source_ids_json','requested_erasure'},'actual controlled keys')
    check(set(re.findall(r'raw_outer\.write\("([^\"]+)"',main_source))=={'feature_mean','feature_scale','fitted_rows'},'prepared raw map schema')
    for s in ('untrained_compact','untrained_pooled','native_compact','native_pooled','initial-pair.pt','initialization-audit.json','pooled-context-v1'):
        check(s in main_source,'declared matrix/init writer '+s)
    learner=(repo/paths[3]).read_text();state=learner[learner.index('void save_continuation_state('):learner.index('void save_gain_view(')]
    for k in ('global_pool_input_source_value','shared_parameter_values','copied_parameter_values','inactive_projection_values',
              'initial_model_parameters','initial_model_buffers','candidate_first_before_copy','loss_trace_counters','loss_trace_values','optimizer_state'):
        check('"'+k+'"' in state,'actual live state literal '+k)
    check(learner.index('prepare_pooled_initialization(state);')<learner.index('state->optimizer ='),'copy occurs before optimizer creation')
    for s in ('copy-all-shared-named-parameters-and-buffers-from-compact-point0-before-AdamW;except-global_pool_first.weight',
              'compact-control-independent-initialization;no-copy','paired_initialization_json'):
        check(s in learner,'actual paired copy policy '+s)
    adapter=(repo/paths[4]).read_text();header=(repo/paths[5]).read_text()
    for s in ('rpb_pooled_context_cuda_snapshot_v1','pooled-context-snapshot-audit.pt','rpb_pooled_context_continuation_state_v1'):
        check(s in header or s in learner or s in (repo/paths[8]).read_text(),'new exact artifact '+s)
    for s in ('parent=core_writer_and_training_producer;loader=new_pooled_context_adapter','candidate_first_before_copy',
              'independent_CUDA_model;eval_no_grad;frozen_timing_TRAIN_scaler;ordinary_query_no_context_erasure;no_refit','cpu(z)'):
        check(s in adapter,'CUDA source/array lineage '+s)
    query=(repo/paths[2]).read_text();body=query[query.index('std::string reconstruction('):query.index('void check_progress(')]
    check(set(re.findall(r'output\.write\("([^\"]+)"',body))=={'standardized_prediction','standardized_target','target_mask',
          'requested_observed_target_mask','visible_mask','trial_channel_eligible','channel_target_counts','channel_valid',
          'channel_standardized_mae','channel_standardized_huber','example_valid','example_standardized_mae','example_standardized_huber','source_ids_json'},'original query literal schema')
    shared=(repo/paths[1]).read_text()
    for s in ('ridge_logits','tiny_hidden_preactivation','tiny_logits','inputs_train_prepared','actual_probe_seed_decimal'):check(s in shared,'saved head witness '+s)
    helper_tree=ast.parse((repo/paths[9]).read_text());plan_node=next(n for n in helper_tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixed_plan')
    check(len(plan_node.body)==1 and isinstance(plan_node.body[0],ast.Return) and isinstance(plan_node.body[0].value,ast.Dict),
          'bounded literal helper recipe fixture; no helper import or measured branch')
    namespace={'PROTOCOL':PROTOCOL,'TIMING':list(TIMING_MASTERS),'AMPLITUDE':list(AMPLITUDE_MASTERS)}
    exec(compile(ast.Module(body=[plan_node],type_ignores=[]),'<artificial-recipe-SOURCE>','exec'),namespace)
    fixed_plan({**namespace['fixed_plan'](),'human_card_sha256':CARD_SHA,'source_fingerprint':'a'*64})
    return records


def self_test(repo):
    started=time.monotonic();start=CHECKS;archive_start=ARCHIVES;import_helper(repo);negatives=0
    def reject(action):
        nonlocal negatives
        try:action()
        except (AssertionError,RuntimeError,ValueError,KeyError,TypeError,OverflowError):negatives+=1
        else:raise AssertionError('negative SOURCE fixture accepted')
    ids=[f'artificial-source-{i}' for i in range(128) for _ in range(2)];master=TIMING_MASTERS[0];frozen=fixture_scaler()
    pair=fixture_initial();initial=initial_witness(pair)
    bad=dict(pair);wrong=dict(initial['candidate'][0]);wrong['global_pool_first.bias']=fake('FloatStorage',[64],[.01]*64)
    bad['candidate_parameters']=fixture_named(wrong);reject(lambda:initial_witness(bad))
    bad=dict(pair);bad['candidate_parameters']=fixture_named(dict(reversed(list(initial['candidate'][0].items()))));reject(lambda:initial_witness(bad))
    bad=dict(pair);wrong=dict(initial['candidate'][0]);wrong[NONSHARED_PARAMETER]=initial['control'][0][NONSHARED_PARAMETER]
    bad['candidate_parameters']=fixture_named(wrong);reject(lambda:initial_witness(bad))
    check(MT19937_64(5489).draw()==14514284786278117030,'exact MT64 primitive')
    check(own_argmax([[1.,1.],[0.,1.]])==[0,1],'saved own-logit tie classes')
    close(1.+3.5e-9,1.,'both additive absolute and relative tolerance terms')
    reject(lambda:close(1.+1e-7,1.,'corrupt arithmetic'))
    rejected_count=ARCHIVES;reject(lambda:load(Path('/synthetic/checkpoint.pt')));check(ARCHIVES==rejected_count,'checkpoint body rejection precedes codec/count')
    branch_count=0
    with tempfile.TemporaryDirectory(prefix='pooled-reader-state-SOURCE-') as directory:
        root=Path(directory).resolve();reference=root/'control/point-0/checkpoint.pt';reference.parent.mkdir(parents=True)
        for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.continuation.pt'):Path(str(reference)+suffix).write_bytes(('artificial'+suffix).encode())
        scopes={'core_writer':'1'*64,'pooled_training':'2'*64,'pooled_adapter':'3'*64}
        for candidate,role in ((False,'control'),(True,'candidate')):
            recorded=[]
            for point in BUDGETS:
                cp=root/role/f'point-{point}/checkpoint.pt';cp.parent.mkdir(parents=True,exist_ok=True)
                for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.continuation.pt'):
                    p=Path(str(cp)+suffix)
                    if not p.exists():p.write_bytes(('artificial'+role+suffix).encode())
                audit=fixture_parent(master,candidate,point,ids,reference);reference_binding(audit,reference,candidate)
                info=parent_audit(audit,master,candidate,point,ids);info.update(schema_id='fixture-schema',frozen_scaler=frozen)
                # Actual raw/scaler admission branch, with absent zeros and a
                # known nonzero affine mean; no generated quality data.
                mask_values=[h%2==0 for b in range(256) for c in range(3) for h in range(32) for f in range(3)]
                raw_values=[(.25+(-2. if b%2 else 2.)) if h%2==0 else 0. for b in range(256) for c in range(3) for h in range(32) for f in range(3)]
                raw_asset={'artifact_kind':fake_text('rpb_raw_uniform_history_v1'),'format_version':fake('LongStorage',[],[1]),
                           'data':fake('DoubleStorage',[256,3,32,3],raw_values),'observed':fake('BoolStorage',[256,3,32,3],mask_values),
                           'channel_ids':fake('LongStorage',[3],[0,1,2]),'endpoints':fake('DoubleStorage',[256],[31.]*256),
                           'sampling_interval':fake('DoubleStorage',[],[1.]),'feature_units':fake_text('unitless,unitless,unitless'),
                           'schema_id':fake_text('fixture-schema'),'dataset_id':fake_text('fixture-dataset')}
                scaler_asset={'artifact_kind':fake_text('rpb_frozen_training_scaler_v1'),'format_version':fake('LongStorage',[],[1]),
                              'scaler':frozen,'preprocessing_id':fake_text(info['scaler_id']),'schema_id':fake_text('fixture-schema'),
                              'fit_dataset_id':fake_text('fixture-dataset')}
                replay_parent(audit,raw_asset,scaler_asset,(raw_values,mask_values,ids,[0,1]*128),master,candidate,point,scopes)
                bad=dict(raw_asset);bad['dataset_id']=fake_text('wrong');reject(lambda bad=bad:replay_parent(audit,bad,scaler_asset,
                       (raw_values,mask_values,ids,[0,1]*128),master,candidate,point,scopes))
                state=fixture_continuation(audit,info,cp,point,initial[role],candidate)
                result=continuation_state(state,audit,info,cp,point,1,initial[role],candidate);recorded.append(result);branch_count+=1
                snapshot=fixture_snapshot(audit,info,cp,point,candidate,scopes,initial[role],state)
                snapshot_binding(snapshot,audit,info,cp,candidate,point,scopes,initial[role]);branch_count+=1
                bad=dict(audit);bad['pooled_shared_parameter_values']=fake_text('219468');reject(lambda bad=bad:parent_audit(bad,master,candidate,point,ids))
                bad=dict(audit);bad['global_pool_input_source_value']=fake('LongStorage',[],[1-int(candidate)])
                reject(lambda bad=bad:parent_audit(bad,master,candidate,point,ids))
                bad=dict(snapshot);bad['snapshot_loader_source_fingerprint']=fake_text('4'*64)
                reject(lambda bad=bad:snapshot_binding(bad,audit,info,cp,candidate,point,scopes,initial[role]))
                if point:
                    bad=dict(state);bad['loss_trace_counters']=fake('LongStorage',[0,3],[])
                    reject(lambda bad=bad:continuation_state(bad,audit,info,cp,point,1,initial[role],candidate))
                    bad=dict(state);opt=dict(state['optimizer_state']);entry=dict(opt['parameter_0']);entry['step']=fake('LongStorage',[],[point+1]);opt['parameter_0']=entry;bad['optimizer_state']=opt
                    reject(lambda bad=bad:continuation_state(bad,audit,info,cp,point,1,initial[role],candidate))
                if candidate:
                    bad=dict(state);values=list(state['candidate_first_before_copy']['values']);values[-1]+=.01
                    bad['candidate_first_before_copy']=fake('FloatStorage',[64,195],values)
                    reject(lambda bad=bad:continuation_state(bad,audit,info,cp,point,1,initial[role],candidate))
                    bad=dict(state);params=dict(result['parameters']);params['export_projection.bias']=fake('FloatStorage',[32],[.01]*32);bad['model_parameters']=fixture_named(params)
                    reject(lambda bad=bad:continuation_state(bad,audit,info,cp,point,1,initial[role],candidate))
                    bad=dict(audit);bad['pooled_reference_point0_path']=fake_text('/unrelated/checkpoint.pt');reject(lambda bad=bad:reference_binding(bad,reference,candidate))
                    bad=dict(state);opt=dict(state['optimizer_state']);j=list(result['parameters']).index('export_projection.bias');entry=dict(opt[f'parameter_{j}'])
                    entry.update(has_state=fake('BoolStorage',[],[True]),step=fake('LongStorage',[],[1]),exp_avg=fake('FloatStorage',[32],[0.]*32),exp_avg_sq=fake('FloatStorage',[32],[0.]*32))
                    opt[f'parameter_{j}']=entry;bad['optimizer_state']=opt
                    reject(lambda bad=bad:continuation_state(bad,audit,info,cp,point,1,initial[role],candidate))
            original_point_prefix(*recorded)
    raw=[1. if j%5 else 0. for _ in range(256) for j in range(288)];observed=[j%5!=0 for _ in range(256) for j in range(288)]
    for task,task_master in (('lag_sign',master),('amplitude',AMPLITUDE_MASTERS[0])):
        source_ids=[f'{PROTOCOL}/{task}/seed-{task_master}/{task}/source-{i}' for i in range(128) for _ in range(2)]
        asset={'observations':fake('DoubleStorage',[256,3,32,3],raw),'feature_mask':fake('BoolStorage',[256,3,32,3],observed),
               'labels_scoring_only':fake('LongStorage',[256],[0,1]*128),'source_ids_json':fake_text(json.dumps(source_ids))}
        observations(asset,256,task_master,task)
        bad=dict(asset);bad['observed']=bad.pop('observations');reject(lambda bad=bad:observations(bad,256,task_master,task))
        vi=[f'{PROTOCOL}/{task}/seed-{task_master}/{task}/source-{i}' for i in range(128,192) for _ in range(2)]
        base=(raw[:128*288],observed[:128*288],vi,[0,1]*64);erasure=[x for row in coordinate_erasure(vi,task,task_master) for x in row]
        dv,dm=deletion_view(base[0],base[1],erasure);saved={'requested_erasure':fake('BoolStorage',[128,3,32,3],erasure)}
        check_view(base,(dv,dm,vi,base[3]),task_master,task,saved)
        wrong=list(erasure);wrong[-1]=not wrong[-1];reject(lambda:check_view(base,(dv,dm,vi,base[3]),task_master,task,{'requested_erasure':fake('BoolStorage',[128,3,32,3],wrong)}))
    qs=(raw[:576],observed[:576],ids[:2],[0,1]);qa=fixture_query(*qs[:3],frozen)
    check(query_archive(qa,qs,frozen)['mae']==.25,'actual F64 query hierarchy')
    bad=dict(qa);bad['channel_target_counts']=fake('LongStorage',[2,3],[0]*6);reject(lambda:query_archive(bad,qs,frozen))
    fixture_saved_head(ids,reject)
    cohorts=[]
    for timing_master in TIMING_MASTERS:
        methods=[{'method':m,'status':'measured','repetitions':[{'validation_intact':{'ridge':{'accuracy':.75,'coverage':1.}},
                 'validation_deleted':{'ridge':{'accuracy':.75,'coverage':1.}}} for _ in HEAD_REPETITIONS]} for m in ('native_compact','native_pooled')]
        cohorts.append({'timing_master':timing_master,'tasks':[{'task':'lag_sign','methods':methods}],
                        'encoders':[{'role':r,'budget':512,'fixed_query':{'training':{'mae':.1},'validation':{'mae':.1}}} for r in ('control','candidate')]})
    check(decision_summary(cohorts)['joint_direction_passed'],'exact equality joint guard fixture')
    bad=json.loads(json.dumps(cohorts));bad[0]['tasks'][0]['methods'][1]['repetitions'][0]['validation_deleted']['ridge'].update(accuracy=None,coverage=0.)
    decision=decision_summary(bad);check(not decision['joint_direction_passed'] and decision['timing_ridge']['validation_deleted'] is None,'null support retained and joint guard false')
    names=sorted(required_source_roles());records=[{'path':n,'bytes':1,'sha256':'a'*64} for n in names]
    index={prefix+n:{'bytes':1,'sha256':'a'*64} for prefix in ('source/','admission/source/') for n in names};check(admit_source_roles(records,index)==names,'closed writer/SDK source admission')
    touched=[]
    class Spy(dict):
        def __getitem__(self,key):touched.append(key);return super().__getitem__(key)
    for invalid in ('README.md','../setup.sh','setup-extra.sh','code/../escape','/absolute'):
        bad=sorted(records+[{'path':invalid,'bytes':1,'sha256':'a'*64}],key=lambda x:x['path']);touched.clear()
        reject(lambda bad=bad:admit_source_roles(bad,Spy(index)));check(not touched,'all source names admitted before bindings')
    for missing in ('setup.sh','code/scripts/install-libtorch.py','code/encoders/raw_patch_bottleneck_mae/src/pooled_context_adapter.cpp'):
        reject(lambda missing=missing:admit_source_roles([r for r in records if r['path']!=missing],index))
    with tempfile.TemporaryDirectory(prefix='pooled-matrix-SOURCE-') as directory:
        root=Path(directory).resolve();a=root/'a';z=root/'z';a.write_bytes(b'artificial');os.link(a,z)
        rows=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in (a,z)];hashed=[]
        reject(lambda:admit_file_matrix(rows,root,lambda p:hashed.append(p)));check(not hashed,'all inodes admitted before payload hashes')
    original=Path(__file__).read_bytes();enabled=original.replace(b'REVIEWED_SCHEMA = False\n',b'REVIEWED_SCHEMA = True\n',1).replace(b'MEASURED_IMPLEMENTATION = False\n',b'MEASURED_IMPLEMENTATION = True\n',1)
    flags_only_source_binding(enabled,original)
    known=set(globals())|set(dir(builtins));scopes=[];unresolved=set()
    def scan(table):
        scopes.append(table)
        for symbol in table.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known:unresolved.add(symbol.get_name())
        for child in table.get_children():scan(child)
    scan(symtable.symtable(Path(__file__).read_text(),str(Path(__file__)),'exec'));check(not unresolved,'all measured globals resolve before release')
    writers=schema_source_fixtures(repo);sdk=[{'path':n,'sha256':sha(repo/n)} for n in ('setup.sh','code/scripts/install-libtorch.py')]
    check(ARCHIVES==archive_start,'zero archive and quality reads in SOURCE fixtures')
    return {'protocol':PROTOCOL,'status':'passed','checks':CHECKS-start,'negative_cases':negatives,'scopes':len(scopes),'unresolved_globals':sorted(unresolved),
            'writer_source_schema':writers,'sdk_source_schema':sdk,'archive_payload_reads':0,'quality_payload_hashes':0,'elapsed_seconds':time.monotonic()-started,
            'live_state_and_snapshot_positive_branches':branch_count,'measured_execution_enabled':bool(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION),
            'human_card_sha256':CARD_SHA,'source_sha256':sha(Path(__file__))}


def requirements():
    return {'protocol':PROTOCOL,'status':'released-completed-capsule-only' if REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION else 'blocked-source-only',
            'measured_execution_enabled':bool(REVIEWED_SCHEMA and MEASURED_IMPLEMENTATION),'human_card_sha256':CARD_SHA,
            'timing_masters':TIMING_MASTERS,'amplitude_masters':AMPLITUDE_MASTERS,'methods':METHODS,'saved_math_tolerance':[ATOL,RTOL],
            'planned_pipelines':210,'planned_individual_heads':420,'registered_parameters':PARAMETER_COUNTS,
            'shared_parameter_values':SHARED_PARAMETERS,'inactive_projection_values':2080,'historical_payload_roles':0,
            'ordinary_CUDA_checkpoint_body_decodes':0,'encoder_or_head_or_SVD_fits':0,'attempts':512,'batch':8,
            'release_protocol':'FALSE code/schema/fixtures sealed before generation; NEW two-line release only after completed inventory authorization'}


def fixture_initial():
    pair={};frozen=fixture_scaler()
    for candidate,role in ((False,'control'),(True,'candidate')):
        params={'export_projection.bias':fake('FloatStorage',[32],[0.]*32),'export_projection.weight':fake('FloatStorage',[32,64],[0.]*2048),
                'global_pool_first.bias':fake('FloatStorage',[64],[0.]*64),
                'global_pool_first.weight':fake('FloatStorage',[64,195 if candidate else 99],[0.]*(12480 if candidate else 6336)),
                'global_pool_second.bias':fake('FloatStorage',[32],[0.]*32),'global_pool_second.weight':fake('FloatStorage',[32,64],[0.]*2048),
                'z_artificial_remaining':fake('FloatStorage',[215245],[0.]*215245)}
        pair.update({role+'_parameters':fixture_named(params),role+'_buffers':fixture_named({'buffer':fake('LongStorage',[],[0])}),role+'_scaler':frozen})
    return pair


def fixture_parent(master,candidate,point,ids,reference):
    settings={'channel_count':3,'history_length':32,'input_width':3,'patch_length':8,'encoder_width':64,'export_width':32,
              'num_layers':3,'num_heads':4,'feedforward_width':256,'decoder_hidden_width':128,'channel_mixer_layers':1,
              'global_bottleneck_mode':2,'channel_mixer_placement':1,'steps':512,'batch_size':8,'threads':1,'log_every':1,
              'attempt_limit':1024,'seed':master,'dropout':0.,'layer_norm_epsilon':1e-5,'mask_ratio':.25,'huber_delta':1.,
              'scale_floor':1e-6,'sampling_interval':1.,'learning_rate':.001,'weight_decay':.0001,'gradient_clip_norm':1.,'device':'cuda','channel_ids':''}
    if candidate:settings['global_pool_input_source']=1
    fields={'artifact_kind':'rpb_learning_curve_training_audit_v1','protocol_id':FIT_PROTOCOL,'actual_training_seed':str(master),
            'initialization_seed':str(mixed(master^0x7270622d696e6974)),'fit_source_manifest':source_manifest(ids),'rng_policy':COUNTER_POLICY,
            'model_weight_update_budget':str(point),'training_device':'cuda','training_producer_source_fingerprint':'2'*64,
            'core_writer_source_fingerprint':'1'*64,'resolved_settings':'\n'.join(k+'='+str(v) for k,v in settings.items()),
            'training_dataset_id':'fixture-dataset','scaler_fit_dataset_id':'fixture-dataset','parameter_count':str(PARAMETER_COUNTS[int(candidate)]),
            'cuda_parameter_count':str(PARAMETER_COUNTS[int(candidate)]),'preprocessing_id':scaler(fixture_scaler())[2],
            'feature_units':'unitless,unitless,unitless','sampling_policy':'with_replacement_counter_rows;sampled_rows_includes_no_update_attempts',
            'optimizer_policy':'one_continuous_AdamW_state;absolute_completed_update_budgets',
            'source_fingerprint_scope':'ordinary_checkpoint_field=core_writer;companion_audit_training_producer=evaluation_source',
            'channel_mixer_placement':'1','architecture_id':ARCHITECTURES[int(candidate)],'output_semantics':OUTPUT_SEMANTICS[int(candidate)],
            'reconstruction_export_semantics':RECONSTRUCTION_SEMANTICS[int(candidate)],'model_tag':'RPB-v12' if candidate else 'RPB-v10',
            'training_policy_id':POLICY,**CONTEXT_LITERALS,'global_pool_input_source':str(int(candidate)),
            'global_pool_input_semantics':INPUT_SEMANTICS[int(candidate)],'pooled_initialization_policy':
            'copy-all-shared-named-parameters-and-buffers-from-compact-point0-before-AdamW;except-global_pool_first.weight' if candidate else
            'compact-control-independent-initialization;no-copy','pooled_shared_parameter_values':'219469','pooled_copied_parameter_values':'219469' if candidate else '0',
            'pooled_inactive_projection_values':'2080' if candidate else '0','pooled_loss_reachable_parameter_values':'229869' if candidate else '225805',
            'pooled_nonshared_parameter_name':NONSHARED_PARAMETER,'pooled_reference_point0_path':str(reference) if candidate else '',
            'continuation_state_artifact_kind':'rpb_pooled_context_continuation_state_v1','continuation_state_suffix':'.continuation.pt',
            'continuation_state_policy':'live_named_CPU_model_buffers_scaler_AdamW_and_complete_trace_v1',
            'curve_skip_policy':'abort_ineligible_attempt;no_skipped_update_prefix_permitted'}
    fields['paired_initialization_json']=json.dumps({'input_source':int(candidate),'copied_before_AdamW':True,'shared_parameter_values':219469,
            'copied_parameter_values':219469 if candidate else 0,'inactive_projection_values':2080 if candidate else 0,
            'nonshared_parameter_name':NONSHARED_PARAMETER,'reference_point0_path':str(reference) if candidate else ''},separators=(',',':'))
    if candidate:
        for suffix in ('','.audit.pt','.scaler.pt','.training-raw.pt','.continuation.pt'):
            fields['pooled_reference_content_id'+suffix]='fnv1a64-runtime-content-v1-'+fnv(Path(str(reference)+suffix).read_bytes())
    audit={k:fake_text(v) for k,v in fields.items()}
    for k,v in {'attempted_steps':point,'completed_steps':point,'sampled_rows':point*8,'channel_mixer_placement_value':1,
                'global_pool_input_source_value':int(candidate),'shared_parameter_values':219469,'copied_parameter_values':219469 if candidate else 0,
                'inactive_projection_values':2080 if candidate else 0,'context_deletion_stream_value':0x6374782d64726f70,
                'context_requested_deleted_coordinates':0,'context_actual_deleted_coordinates':0,'context_restored_coordinates':0}.items():audit[k]=fake('LongStorage',[],[v])
    audit['channel_order']=fake('LongStorage',[3],[0,1,2])
    for k,v in {'sampling_interval':1.,'endpoint':31.,'training_seconds':1. if point else 0.,'context_deletion_ratio_value':.15}.items():audit[k]=fake('DoubleStorage',[],[v])
    for k in ('weights_changed','finite_gradients'):audit[k]=fake('BoolStorage',[],[bool(point)])
    return audit


def fixture_continuation(audit,info,cp,point,initial,candidate):
    base_params,buffers,frozen,identity=initial;params=dict(base_params)
    if point:params[NONSHARED_PARAMETER]=fake('FloatStorage',base_params[NONSHARED_PARAMETER]['shape'],[.001]*len(base_params[NONSHARED_PARAMETER]['values']))
    optimizer={'parameter_count':fake('LongStorage',[],[len(params)]),'active_state_count':fake('LongStorage',[],[int(bool(point))])}
    for i,(name,value) in enumerate(params.items()):
        active=bool(point) and name==NONSHARED_PARAMETER
        entry={'parameter_name':fake_text(name),'parameter_shape':fake('LongStorage',[len(value['shape'])],value['shape']),'has_state':fake('BoolStorage',[],[active])}
        if active:entry.update(step=fake('LongStorage',[],[point]),exp_avg=fake('FloatStorage',value['shape'],[.001]*len(value['values'])),
                               exp_avg_sq=fake('FloatStorage',value['shape'],[.001]*len(value['values'])))
        optimizer[f'parameter_{i}']=entry
    asset={k:v for k,v in audit.items() if v.get('dtype')=='ByteStorage' and k not in ('artifact_kind','model_weight_update_budget')}
    asset.update(artifact_kind=fake_text('rpb_pooled_context_continuation_state_v1'),checkpoint_path=fake_text(str(cp)),
                 state_capture_policy=fake_text('live_named_CUDA_parameter_state_to_CPU;no_model_forward;no_optimizer_reload_or_step'),
                 model_parameters=fixture_named(params),model_buffers=fixture_named(buffers),initial_model_parameters=fixture_named(base_params),
                 initial_model_buffers=fixture_named(buffers),scaler=frozen,optimizer_state=optimizer,
                 training_seconds=fake('DoubleStorage',[],[info['training_seconds']]),
                 loss_trace_counters=fake('LongStorage',[point,3],[v for i in range(point) for v in (i+1,i+1,72)]),
                 loss_trace_values=fake('DoubleStorage',[point,2],[v for i in range(point) for v in (.5,.1)]))
    for k in ('attempted_steps','completed_steps','sampled_rows','channel_mixer_placement_value','global_pool_input_source_value','shared_parameter_values',
              'copied_parameter_values','inactive_projection_values','context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'):asset[k]=audit[k]
    if candidate:asset['candidate_first_before_copy']=base_params[NONSHARED_PARAMETER]
    return asset


def fixture_snapshot(audit,info,cp,point,candidate,scopes,initial,state):
    asset={k:v for k,v in audit.items() if v.get('dtype')=='ByteStorage'}
    fields={'artifact_kind':'rpb_pooled_context_cuda_snapshot_v1','protocol_id':PROTOCOL,'original_training_protocol_id':FIT_PROTOCOL,
            'parent_checkpoint_path':str(cp),'parent_writer_source_fingerprint':scopes['core_writer'],'parent_training_producer_source_fingerprint':scopes['pooled_training'],
            'snapshot_loader_source_fingerprint':scopes['pooled_adapter'],'source_fingerprint_scope':'parent=core_writer_and_training_producer;loader=new_pooled_context_adapter',
            'original_encoder_attempted':str(point),'original_encoder_completed':str(point),'training_schema_id':info['schema_id'],
            'training_dataset_id':info['dataset_id'],'preprocessing_id':info['scaler_id'],'parameter_count':str(PARAMETER_COUNTS[int(candidate)]),
            'encoder_updates':'0','decoder_updates':'0','head_refits':'0','no_optimizer_created':'true','inference_device':'cuda',
            'snapshot_policy':'independent_CUDA_model;eval_no_grad;frozen_timing_TRAIN_scaler;ordinary_query_no_context_erasure;no_refit'}
    for k,v in fields.items():asset[k]=fake_text(v)
    for k in ('context_requested_deleted_coordinates','context_actual_deleted_coordinates','context_restored_coordinates'):asset[k]=fake_text(str(scalar(audit,k)))
    for s in ('','.audit.pt','.scaler.pt','.training-raw.pt','.continuation.pt'):asset['parent_content_id'+s]=fake_text('fnv1a64-runtime-content-v1-'+fnv(Path(str(cp)+s).read_bytes()))
    for k,v in {'channel_mixer_placement_value':1,'global_pool_input_source_value':int(candidate),'original_encoder_attempted_value':point,'original_encoder_completed_value':point}.items():asset[k]=fake('LongStorage',[],[v])
    asset.update(model_parameters=state['model_parameters'],model_buffers=state['model_buffers'],scaler=initial[2])
    return asset


def fixture_saved_head(ids,reject):
    fit={k:fake('DoubleStorage',[2],[0.,0.] if k.endswith('mean') else [1.,1.]) for k in ('feature_mean','feature_scale','ridge_mean','ridge_scale','tiny_mean','tiny_scale')}
    fit.update(outer_normalizer_applied=fake('BoolStorage',[],[True]),outer_fitted_rows=fake('LongStorage',[],[2]),fitted_rows=fake('LongStorage',[],[2]),
               ridge_penalty=fake('DoubleStorage',[],[1.]),tiny_hidden=fake('LongStorage',[],[16]),tiny_steps=fake('LongStorage',[],[100]),tiny_learning_rate=fake('DoubleStorage',[],[.01]),
               ridge_weights=fake('DoubleStorage',[2,2],[1.,-1.,0.,0.]),ridge_intercept=fake('DoubleStorage',[2],[0.,0.]),tiny_w1=fake('DoubleStorage',[2,16],[0.]*32),
               tiny_b1=fake('DoubleStorage',[16],[0.]*16),tiny_w2=fake('DoubleStorage',[16,2],[0.]*32),tiny_b2=fake('DoubleStorage',[2],[0.,0.]),
               actual_probe_seed_decimal=fake_text(str(stream_seed(2701,2))),training_source_ids_json=fake_text(json.dumps(ids[:2])))
    fit_schema(fit,2,ids[:2],2701,False,[[-1.,-1.],[1.,1.]],[True,True])
    prediction={'ridge':fake('LongStorage',[2],[1,0]),'tiny_secondary':fake('LongStorage',[2],[0,0]),'valid':fake('BoolStorage',[2],[True,True]),
                'probe_input_features':fake('DoubleStorage',[2,2],[0.,0.,1.,-1.]),'ridge_logits':fake('DoubleStorage',[2,2],[0.,1e-12,1.,-1.]),
                'tiny_hidden_preactivation':fake('DoubleStorage',[2,16],[0.]*32),'tiny_logits':fake('DoubleStorage',[2,2],[0.]*4),
                'labels_scoring_only':fake('LongStorage',[2],[0,1]),'source_ids_json':fake_text(json.dumps(ids[:2]))}
    ridge,_,_=infer_saved_fit(fit,[[0.,0.],[1.,-1.]],[True,True],[0,1],ids[:2],prediction,False,2,2);check(ridge[0]==1,'tolerance never rewrites saved own-logit class')
    bad=dict(prediction);bad['ridge']=fake('LongStorage',[2],[0,0]);reject(lambda:infer_saved_fit(fit,[[0.,0.],[1.,-1.]],[True,True],[0,1],ids[:2],bad,False,2,2))
    bad=dict(fit);bad['post_encoder_pca']=fake_text('forbidden');reject(lambda:infer_saved_fit(bad,[[0.,0.],[1.,-1.]],[True,True],[0,1],ids[:2],prediction,False,2,2))
    bad=dict(fit);bad['actual_probe_seed_decimal']=fake_text('1');reject(lambda:fit_schema(bad,2,ids[:2],2701,False,[[-1.,-1.],[1.,1.]],[True,True]))


if __name__=='__main__':main()
