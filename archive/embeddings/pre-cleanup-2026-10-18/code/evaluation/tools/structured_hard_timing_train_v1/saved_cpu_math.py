"""Pure saved CPU arithmetic, extracted byte-for-byte from reviewed B114 SOURCE.
No path resolution, imports of historical orchestration, archive reads, fits or models.
The new protocol's entry point supplies an independently admitted pinned codec.
"""
import array
import json
import math
import struct
MASK64 = (1 << 64) - 1
ATOL = RTOL = 2e-9
CHECKS = 0
R = None

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


