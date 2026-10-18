# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Independent saved scalar kernels for V11 descriptive TRAIN observations.

No gradients, local adjoints, hidden states, model/head modules, old V9
reductions, finite differences, fits or optimizer counterfactuals are executed.
"""
import math

QUANTILES=(0.,.01,.05,.25,.5,.75,.95,.99,1.)
MOMENT_COLUMNS=('update','bin','moment','export_row_begin','export_row_end',
 'weight_gradient_l2','bias_gradient_l2','weight_movement_l2','bias_movement_l2',
 'nonzero_export_gradient_values','moved_export_values')


def average(values):
    return math.fsum(values)/len(values) if values else None


def quantiles(values):
    if not values:return None
    if not all(math.isfinite(value) for value in values):raise ValueError('Finite descriptive scalar values required')
    ordered=sorted(values);answer={}
    for probability in QUANTILES:
        coordinate=(len(ordered)-1)*probability
        lower,upper=math.floor(coordinate),math.ceil(coordinate)
        answer[str(probability)]=ordered[lower]+(coordinate-lower)*(ordered[upper]-ordered[lower])
    return answer


def describe_moment_csv(records):
    """All eight groups and fixed blocks; no learning/event/partition gate replay."""
    if len(records)!=16384:raise ValueError('The declared full16384-row descriptive population is required')
    answer=[]
    for name,first,last in (('overall',1,2048),('updates1_512',1,512),
                            ('updates513_1024',513,1024),('updates1025_1536',1025,1536),
                            ('updates1537_2048',1537,2048)):
        groups=[]
        for bin_index in range(4):
            for branch,label in enumerate(('mean','centered_variance')):
                selected=[row for row in records if row['bin']==bin_index and row['moment']==label
                          and first<=row['update']<=last]
                if len(selected)!=last-first+1:raise ValueError('Fixed complete subgroup/update population')
                begin=8*bin_index+4*branch
                groups.append(dict(bin=bin_index,moment=label,export_row_begin=begin,export_row_end=begin+4,
                    records=len(selected),scalars={column:dict(mean=average([row[column] for row in selected]),
                        quantiles=quantiles([row[column] for row in selected])) for column in MOMENT_COLUMNS[5:]}))
        answer.append(dict(name=name,first_update=first,last_update=last,updates=last-first+1,subgroups=groups))
    return dict(overall_and_fixed512=answer,CSV_columns=list(MOMENT_COLUMNS),recorded_rows=len(records))


def describe_moment_witness(update,attempt,clip_factor,positions,valid,rows,weight,bias,routes,packed_shape=None):
    """Describe saved moments and explicitly contracted before-step export rows.

    Every packed row is contracted then an absent local bin forces its output
    slice to zero after bias. Summary statistics use present rows only. Mixture
    concentration includes every valid packed token, including global context.
    No hidden states, derivatives, alternative forward or noncaptured decoder.
    """
    R=len(rows)
    if packed_shape is None:
        if not R or len(positions)%R:raise ValueError('Saved packed geometry')
        T=len(positions)//R
    else:
        recorded_R,T=packed_shape
        if recorded_R!=R:raise ValueError('Saved packed row association')
    if len(positions)!=R*T or len(valid)!=len(positions):raise ValueError('Saved packed logical sizes')
    if len(weight)!=32*64 or len(bias)!=32:raise ValueError('Before-step32x64 export geometry')
    observed=[]
    for name,route in routes:
        present=route['bin_valid'];means=route['bin_mean'];variances=route['bin_variance']
        mixture=route['mixture_weights'];upstream=route['upstream_pre_clip']
        if not (len(present)==R*4 and len(means)==len(variances)==R*4*64
                and len(mixture)==R*4*T and len(upstream)==R*32):
            raise ValueError('Saved moments/mix/upstream logical sizes')
        bins=[]
        for bin_index in range(4):
            members=[sum(bool(valid[r*T+t]) and 4*bin_index<=positions[r*T+t]<4*(bin_index+1)
                         for t in range(T)) for r in range(R)]
            active=[r for r in range(R) if present[r*4+bin_index]]
            mean_values=[float(means[(r*4+bin_index)*64+d]) for r in active for d in range(64)]
            variance_values=[float(variances[(r*4+bin_index)*64+d]) for r in active for d in range(64)]
            concentration=[];effective=[];maximum=[]
            for r in active:
                weights=[float(mixture[(r*4+bin_index)*T+t]) for t in range(T) if valid[r*T+t]]
                squared=math.fsum(value*value for value in weights)
                if squared<=0 or not weights:raise ValueError('Present saved mixture needspositive valid-token sum of squares')
                concentration.append(squared);effective.append(1./squared)
                maximum.append(max(weights))
            outputs=[]
            for branch,moments in enumerate((means,variances)):
                values=[]
                for r in range(R):
                    for coordinate in range(4):
                        export_row=8*bin_index+4*branch+coordinate
                        # Operands arrive as Python scalars; the before-step
                        # affine is disclosed arithmetic, not module inference.
                        value=math.fsum(float(moments[(r*4+bin_index)*64+d])*
                                        float(weight[export_row*64+d]) for d in range(64))+float(bias[export_row])
                        values.append(value if present[r*4+bin_index] else 0.)
                outputs.append(values)
            upstream_mean=[float(upstream[r*32+8*bin_index+d]) for r in active for d in range(4)]
            upstream_variance=[float(upstream[r*32+8*bin_index+4+d]) for r in active for d in range(4)]
            bins.append(dict(bin=bin_index,present_local_rows=len(active),local_member_counts=members,
                mean64_quantiles=quantiles(mean_values),centered_variance64_quantiles=quantiles(variance_values),
                mixture_sum_squares_quantiles=quantiles(concentration),
                mixture_effective_weight_count_quantiles=quantiles(effective),
                mixture_max_weight_quantiles=quantiles(maximum),manual_export_mean4_values=outputs[0],
                manual_export_variance4_values=outputs[1],
                manual_export_mean4_quantiles=quantiles([outputs[0][r*4+d] for r in active for d in range(4)]),
                manual_export_variance4_quantiles=quantiles([outputs[1][r*4+d] for r in active for d in range(4)]),
                PRECLIP_upstream_mean4_quantiles=quantiles(upstream_mean),
                PRECLIP_upstream_variance4_quantiles=quantiles(upstream_variance)))
        observed.append(dict(route=name,upstream_defined=bool(route['upstream_defined']),bins=bins))
    return dict(update=update,attempt=attempt,pre_step_completed_updates=update-1,
        clip_factor=clip_factor,packed_rows=rows,packed_positions=positions,packed_valid=valid,
        geometry=dict(R=R,T=T),routes=observed)

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

