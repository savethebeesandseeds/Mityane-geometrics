# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Owned scalar kernels for a prospective saved-TRAIN audit; no model code.

Only call after the independent allowlisted card is frozen and admitted inside
the existing container. Scalar equality does not prove detach or live routing.
"""
import math

CLIP_PAIRS = {'v6': (('pre_clip_global_norm','clip_factor'),),
              'v8': (('pre_clip_global_norm','clip_factor'),)}
TRAJECTORY_COLUMNS = {version: ('mean_Huber_F64','variance_NLL_F64','total_objective_F64','Gaussian_NLL_F64')+
                      ('pre_clip_global_norm','clip_factor')
                      for version in ('v6','v8')}


def describe_trajectory(records, version):
    """Fixed all2048/four512 summaries of already recorded TRAIN scalars."""
    if len(records) != 2048 or [row['update'] for row in records] != list(range(1, 2049)):
        raise ValueError('Exactly2048 successful updates in recorded order')
    output = {}
    for name, begin, end in (('overall', 0, 2048), ('updates1_512', 0, 512),
                             ('updates513_1024', 512, 1024), ('updates1025_1536', 1024, 1536),
                             ('updates1537_2048', 1536, 2048)):
        chosen = records[begin:end]
        statistics = {}
        for column in TRAJECTORY_COLUMNS[version]:
            values = [float(row[column]) for row in chosen]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Finite recorded trajectory scalar required')
            ordered = sorted(values)
            n = len(ordered)
            statistics[column] = {'minimum': ordered[0],
                                  'mean': math.fsum(values) / n,
                                  'median': (ordered[(n - 1) // 2] + ordered[n // 2]) / 2,
                                  'maximum': ordered[-1]}
        clips = {}
        for _, factor in CLIP_PAIRS[version]:
            count = sum(row[factor] < 1. for row in chosen)
            clips[factor] = {'factor_below1_updates': count, 'frequency': count/len(chosen)}
        output[name] = {'first_update': begin + 1, 'last_update': end,
                        'updates': len(chosen), 'observed_clipping': clips,
                        'scalars': statistics}
    return output


def describe_temporal_query(query):
    """Independent first8 observed/raw and masked-output descriptive bins.

    Original raw values are read as F64. Supported losses promote individual
    stored operands before subtraction. Each bin has its own eligibility; this
    function never treats its losses as additive pieces of the full objective.
    """
    def mean(values):
        return math.fsum(values)/len(values) if values else None

    def moments(values):
        center=mean(values)
        return math.sqrt(mean([(value-center)**2 for value in values])) if values else None

    def quantiles(values):
        if not values:
            return None
        values=sorted(values)
        result={}
        for q in (0.,.01,.05,.25,.5,.75,.95,.99,1.):
            position=q*(len(values)-1)
            left,right=math.floor(position),math.ceil(position)
            result[str(q)]=values[left]+(position-left)*(values[right]-values[left])
        return result

    def index(b,c,h,f):
        return ((b*3+c)*128+h)*9+f

    def metrics(indices):
        errors=[float(query['mean'][i])-float(query['targets'][i]) for i in indices]
        variances=[float(query['variance'][i]) for i in indices]
        return dict(target_cells=len(indices),mean_absolute_error=mean([abs(e) for e in errors]),
                    mean_Huber_delta1=mean([.5*e*e if abs(e)<=1. else abs(e)-.5 for e in errors]),
                    Gaussian_NLL=mean([.5*(e*e/v+math.log(v)) for e,v in zip(errors,variances)]),
                    mean_variance=mean(variances),absolute_error_quantiles=quantiles([abs(e) for e in errors]))

    bins=[]
    for lower,upper in ((0,32),(32,64),(64,96),(96,128)):
        selected={(b,c):[index(b,c,h,f) for h in range(lower,upper) for f in range(9)
                         if query['target_mask'][index(b,c,h,f)]] for b in range(8) for c in range(3)}
        rows=[]
        for b,identity in enumerate(query['row_ids']):
            channels=[metrics(selected[b,c]) for c in range(3) if selected[b,c]]
            if channels:
                rows.append(dict(row_id=identity,eligible_channels=len(channels),
                    **{key:mean([channel[key] for channel in channels]) for key in
                       ('mean_absolute_error','mean_Huber_delta1','Gaussian_NLL')}))
        channel_features=[]
        raw=[]
        for c in range(3):
            for f in range(9):
                indices=[index(b,c,h,f) for b in range(8) for h in range(lower,upper)
                         if query['target_mask'][index(b,c,h,f)]]
                channel_features.append(dict(channel_id=c,feature=f,**metrics(indices)))
                for b,identity in enumerate(query['row_ids']):
                    values=[query['raw'][index(b,c,h,f)] for h in range(lower,upper)
                            if query['observed'][index(b,c,h,f)]]
                    differences=[abs(query['raw'][index(b,c,h,f)]-query['raw'][index(b,c,h-1,f)])
                                 for h in range(lower+1,upper) if query['observed'][index(b,c,h,f)]
                                 and query['observed'][index(b,c,h-1,f)]]
                    raw.append(dict(row_id=identity,channel_id=c,feature=f,observed_cells=len(values),
                                    raw_mean=mean(values),raw_population_std=moments(values),
                                    within_bin_observed_adjacent_pairs=len(differences),
                                    raw_mean_absolute_first_difference=mean(differences)))
        pooled=[i for indices in selected.values() for i in indices]
        bins.append(dict(history_begin_inclusive=lower,history_end_exclusive=upper,
                         age_order='oldest-to-most-recent',pooled_supported_cells=metrics(pooled),
                         eligible_examples=len(rows),hierarchical_rows=rows,
                         eligible_example_channel_cell_mean={key:mean([row[key] for row in rows]) for key in
                             ('mean_absolute_error','mean_Huber_delta1','Gaussian_NLL')},
                         channel_features=channel_features,observed_raw_by_row_channel_feature=raw))
    if sum(item['pooled_supported_cells']['target_cells'] for item in bins)!=sum(query['target_mask']):
        raise ValueError('Four fixed bins must partition masked target support')
    return dict(population=8,history=128,bins=bins)


def reduce_components(target, prediction, variance, supported, batch=8,
                      channels=3, history=128, features=9):
    size = batch * channels * history * features
    if not all(len(values) == size for values in
               (target, prediction, variance, supported)):
        raise ValueError('Exact declared supported-cell population required')
    rows = []
    channel_rows = []
    for b in range(batch):
        row = []
        for c in range(channels):
            losses = []
            for h in range(history):
                for f in range(features):
                    i = ((b * channels + c) * history + h) * features + f
                    if not supported[i]:
                        continue
                    # Promote each stored operand before subtraction. No F32
                    # tensor subtraction or unsupported loss algebra occurs.
                    mean, truth, v = (float(prediction[i]), float(target[i]),
                                      float(variance[i]))
                    if not all(math.isfinite(x) for x in (mean, truth, v)) or v <= 0:
                        raise ValueError('Finite supported operands and positive variance')
                    error = mean - truth
                    absolute = abs(error)
                    huber = (.5 * error * error if absolute <= 1
                             else absolute - .5)
                    gaussian = .5 * (error * error / v + math.log(v))
                    losses.append((absolute, gaussian, huber, huber + gaussian))
            if losses:
                item = {'row': b, 'channel': c, 'cells': len(losses),
                        'means': [math.fsum(x[k] for x in losses) / len(losses)
                                  for k in range(4)]}
                row.append(item)
                channel_rows.append(item)
        rows.append({'eligible_channels': len(row),
                     'target_cells': sum(x['cells'] for x in row),
                     'means': [math.fsum(x['means'][k] for x in row) / len(row)
                               if row else 0. for k in range(4)]})
    eligible = [row for row in rows if row['eligible_channels']]
    if not eligible:
        raise ValueError('At least one eligible saved TRAIN example required')
    return {'eligible_examples': len(eligible), 'rows': rows,
            'per_example_channel': channel_rows,
            'scalar': [math.fsum(row['means'][k] for row in eligible) / len(eligible)
                       for k in range(4)],
            'order': ['MAE', 'Gaussian_NLL', 'Huber_delta1', 'composite_Huber_plus_Gaussian'],
            'hierarchy': 'supported cells within channel, eligible channels within example, eligible examples'}


def describe_temporal_groups(records):
    """Independent fixed summaries of recorded post-globalclip V8 observations.

    This does not reconstruct missing upstream gradients or any V6 subgroup.
    The caller admits precisely8192 update-major records before calling it.
    """
    fields=('weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2',
            'pool_position_centered_gradient_l2','pool_position_centered_movement_l2')
    events=('weight_nonzero_data_gradient_steps','bias_nonzero_data_gradient_steps',
            'weight_movement_steps','bias_movement_steps','position_centered_data_gradient_steps',
            'position_centered_movement_steps')
    if len(records)!=8192:
        raise ValueError('Exactly8192 declared temporal records')
    output=[]
    for group in range(4):
        chosen=[row for row in records if row['bin']==group]
        if len(chosen)!=2048 or [row['update'] for row in chosen]!=list(range(1,2049)):
            raise ValueError('Exact per-bin successful update association')
        counts={event:sum(record[field]!=0. for record in chosen)
                for event,field in zip(events,fields)}
        descriptions={}
        for label,begin,end in (('all2048',0,2048),('updates1_512',0,512),
                               ('updates513_1024',512,1024),('updates1025_1536',1024,1536),
                               ('updates1537_2048',1536,2048)):
            statistics={}
            for field in fields:
                values=[row[field] for row in chosen[begin:end]]
                ordered=sorted(values);count=len(ordered)
                statistics[field]=dict(minimum=ordered[0],maximum=ordered[-1],
                    mean=math.fsum(values)/count,
                    median=(ordered[(count-1)//2]+ordered[count//2])/2.,
                    positive_event_updates=sum(value>0. for value in values))
            descriptions[label]=dict(first_update=begin+1,last_update=end,rows=end-begin,
                                     observed_norms=statistics)
        output.append(dict(bin=group,export_rows=[8*group,8*(group+1)],
            original_patch_positions=[4*group,4*(group+1)],event_counts=counts,
            fixed_overall_and512_blocks=descriptions))
    return output


def temporal_array_group(arrays, group):
    """F64 scalar norms/counts from one actual saved12-array witness.

    Tensor unpacking promotes each saved F32 operand to Python float BEFORE
    subtraction. Observed clipped gradients need no upstream reconstruction.
    """
    if group not in range(4):
        raise ValueError('One declared original-position bin')
    def length(values):
        return math.sqrt(math.fsum(value*value for value in values))
    def slice_for(stem,size):
        return {name:[float(value) for value in arrays[stem][name][group*size:(group+1)*size]]
                for name in ('before','after','clipped_gradient')}
    weight=slice_for('temporal_export_weight',512)
    bias=slice_for('temporal_export_bias',8)
    position=slice_for('temporal_pool_positions',4)
    movement=lambda block:[float(end)-float(start) for start,end in zip(block['before'],block['after'])]
    centered=lambda values:[value-math.fsum(values)/len(values) for value in values]
    vectors=(weight['clipped_gradient'],bias['clipped_gradient'],movement(weight),movement(bias),
             centered(position['clipped_gradient']),centered(movement(position)))
    keys=('weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2',
          'pool_position_centered_gradient_l2','pool_position_centered_movement_l2')
    norms={name:length(values) for name,values in zip(keys,vectors)}
    counts=[sum(value!=0. for value in values) for values in
            (vectors[0]+vectors[1],vectors[2]+vectors[3],vectors[4],vectors[5])]
    return dict(bin=group,norms=norms,nonzero_and_movement_counts=counts)
