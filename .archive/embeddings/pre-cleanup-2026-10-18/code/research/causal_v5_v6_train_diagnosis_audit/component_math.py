# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Owned scalar kernels for a prospective saved-TRAIN audit; no model code.

Only call after the independent allowlisted card is frozen and admitted inside
the existing container. Scalar equality does not prove detach or live routing.
"""
import math

TRAJECTORY_COLUMNS = ('mean_Huber_F64', 'variance_NLL_F64', 'total_objective_F64',
                      'Gaussian_NLL_F64', 'pre_clip_global_norm', 'clip_factor')


def describe_trajectory(records):
    """Fixed all2048/four512 summaries of already recorded TRAIN scalars."""
    if len(records) != 2048 or [row['update'] for row in records] != list(range(1, 2049)):
        raise ValueError('Exactly2048 successful updates in recorded order')
    output = {}
    for name, begin, end in (('overall', 0, 2048), ('updates1_512', 0, 512),
                             ('updates513_1024', 512, 1024), ('updates1025_1536', 1024, 1536),
                             ('updates1537_2048', 1536, 2048)):
        chosen = records[begin:end]
        statistics = {}
        for column in TRAJECTORY_COLUMNS:
            values = [float(row[column]) for row in chosen]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Finite recorded trajectory scalar required')
            ordered = sorted(values)
            n = len(ordered)
            statistics[column] = {'minimum': ordered[0],
                                  'mean': math.fsum(values) / n,
                                  'median': (ordered[(n - 1) // 2] + ordered[n // 2]) / 2,
                                  'maximum': ordered[-1]}
        clips = sum(row['clip_factor'] < 1. for row in chosen)
        output[name] = {'first_update': begin + 1, 'last_update': end,
                        'rows': len(chosen), 'clip_applied_rows': clips,
                        'clip_applied_fraction': clips / len(chosen),
                        'statistics': statistics}
    return output


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
