#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Read-only admission inventory; never captures embeddings or fits a model.

Run only in the managed Linux container. Whole files are hashed, while market
records are decoded only through 28 January 2024. The preserved later period
has no assigned evidence role and its prices, labels and outcomes are unopened.
"""
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import struct


ROOT = Path('/cuwacunu')
TICK = 60_000
DAY = 24 * 60 * TICK
JAN = 1_704_067_200_000
LAST_INSPECTED = JAN + 28 * DAY - 1
CACHE = struct.Struct('<q5dq4d')
EXPECTED_ARCHIVE = '50e1d6514076ab34dcfa1194d8f0d97296f176e76174a78efadc97c5c0d1c34e'
EXPECTED_SIDECAR = '05ce98a37370b362cf718da998ccd18e1345026daf58ae258ed091fb220e98f7'
EXPECTED_MODEL = '2d61978bbf601e9050f97bb22387e688acba523d772ea0c540715fb93a260f09'
VERIFIED_CONTAINER_ID = '82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e'


def utc(milliseconds):
    return dt.datetime.fromtimestamp(milliseconds / 1000, dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def inspect_source(symbol, semantic_id):
    source = ROOT / f'data/mkt-hist-1/csv/{symbol}-1m-2024-01.csv'
    cache = source.with_suffix('.cache.bin')
    initial = [(p.stat().st_size, p.stat().st_mtime_ns, sha(p)) for p in (source, cache)]
    assert CACHE.size == 88 and cache.stat().st_size % CACHE.size == 0
    rows = {}
    with source.open(newline='') as stream:
        for columns in csv.reader(stream):
            opened = int(columns[0])
            # Stop before interpreting preserved later prices or targets.
            if opened > LAST_INSPECTED:
                break
            assert len(columns) == 12
            closed = int(columns[6])
            assert closed <= LAST_INSPECTED
            assert closed not in rows
            row = (opened, *(float(x) for x in columns[1:6]), closed,
                   float(columns[7]), float(int(columns[8])),
                   float(columns[9]), float(columns[10]))
            rows[closed] = row
    cache_rows = {}
    with cache.open('rb') as stream:
        # The admitted prefix is fixed independently of later market values.
        for _ in range(28 * 1440):
            raw = stream.read(CACHE.size)
            assert len(raw) == CACHE.size
            row = CACHE.unpack(raw)
            assert row[6] <= LAST_INSPECTED and row[6] not in cache_rows
            cache_rows[row[6]] = row
    expected = [JAN + (i + 1) * TICK - 1 for i in range(28 * 1440)]
    assert list(rows) == list(cache_rows) == expected, 'Source/cache prefix must be a complete ordered minute grid'
    invalid = []
    for closed in expected:
        left, right = rows[closed], cache_rows[closed]
        assert left == right, f'CSV/CACHE88 original F64 mismatch at {closed}'
        valid = right[0] != -(2 ** 63)
        features = (*right[1:6], *right[7:])
        finite = all(math.isfinite(value) for value in features)
        if not valid or not finite or right[4] <= 0:
            invalid.append(closed)
        assert right[0] == closed - TICK + 1
    for p, before in zip((source, cache), initial):
        assert (p.stat().st_size, p.stat().st_mtime_ns, sha(p)) == before, 'Source changed during inventory'
    return {
        'semanticId': semantic_id, 'instrument': symbol,
        'csv': {**pin(source), 'bytes': initial[0][0]},
        'cache': {**pin(cache), 'bytes': initial[1][0], 'recordBytes': 88,
                  'physicalRecordCount': initial[1][0] // CACHE.size,
                  'openedReadOnly': True, 'freshRelativeToCsv': initial[1][1] >= initial[0][1]},
        'inspectedDevelopmentPrefix': {
            'firstCloseMs': expected[0], 'lastCloseMs': expected[-1],
            'firstCloseUtc': utc(expected[0]), 'lastCloseUtc': utc(expected[-1]),
            'rows': len(expected), 'validFinitePositiveCloseRows': len(expected) - len(invalid),
            'invalidRows': len(invalid), 'missingRows': 0, 'duplicateRows': 0,
            'csvCacheF64FieldMismatches': 0,
        },
        'invalidCloseKeys': invalid,
    }


def main():
    assert Path('/.dockerenv').exists() and Path.cwd() == ROOT, 'Use the existing managed container at /cuwacunu'
    assert VERIFIED_CONTAINER_ID.startswith(os.environ['HOSTNAME'])
    sources = [inspect_source(symbol, i) for i, symbol in enumerate(('BTCUSDT', 'ETHUSDT', 'SOLUSDT'))]
    checkpoint = ROOT / 'output/baselines/CPB-v3.market-01/inference.pt'
    sidecar = Path(str(checkpoint) + '.manifest.json')
    assert sha(checkpoint) == EXPECTED_ARCHIVE and sha(sidecar) == EXPECTED_SIDECAR
    manifest = json.loads(sidecar.read_text())
    assert manifest['archive_kind'] == 'inference' and manifest['archive_sha256'] == EXPECTED_ARCHIVE
    assert len(manifest['components']) == 1 and manifest['components'][0]['model_ref'] == EXPECTED_MODEL
    invalid = set().union(*(set(s.pop('invalidCloseKeys')) for s in sources))
    stages = []
    for name, first_day, last_day, minimum, cutoff_parent in (
        ('representation_train', 22, 23, 512, JAN + 21 * DAY - 1),
        ('later_risk_head_train', 24, 25, 512, JAN + 23 * DAY - 1),
        ('development_evaluation', 26, 28, 1024, JAN + 25 * DAY - 1),
    ):
        first = JAN + (first_day - 1) * DAY + TICK - 1
        last = JAN + last_day * DAY - 1
        first_origin = first + 130 * TICK
        last_origin = last - 128 * TICK
        origins = list(range(first_origin, last_origin + 1, TICK))
        rejected = [origin for origin in origins if any(origin - 130 * TICK <= bad <= origin + 128 * TICK for bad in invalid)]
        eligible = len(origins) - len(rejected)
        assert cutoff_parent < first - TICK + 1 and eligible >= minimum
        assert origins[-1] + 128 * TICK <= last
        stages.append({
            'stage': name, 'rawRoleFirstCloseMs': first, 'rawRoleLastCloseMs': last,
            'rawRoleFirstCloseUtc': utc(first), 'rawRoleLastCloseUtc': utc(last),
            'rawMinuteTicks': (last - first) // TICK + 1,
            'firstEligibleOriginMs': first_origin, 'lastEligibleOriginMs': last_origin,
            'firstEligibleOriginUtc': utc(first_origin), 'lastEligibleOriginUtc': utc(last_origin),
            'candidateOriginsAfterEnvelopePurge': len(origins), 'eligibleOrigins': eligible,
            'invalidOrMissingFullSupportOrigins': len(rejected),
            'minimumLegalOrigins': minimum, 'minimumSatisfied': eligible >= minimum,
            'leadingPurgedOrigins': 130, 'trailingPurgedOrigins': 128,
            'lastFeatureModelFitCutoffMs': cutoff_parent, 'stageFitOrScoringCutoffMs': last,
            'mature128MinuteLabelOriginsByStageCutoff': eligible,
            'assetLabelAssociationsPerRiskTarget': eligible * 3,
            'allHorizonPairAssociations': eligible * 3,
            'perHorizonPairAssociations': {str(horizon): eligible for horizon in (1, 16, 128)},
        })
    assert all(s['cache']['freshRelativeToCsv'] for s in sources)
    pinned = [Path(__file__), ROOT / 'AGENTS.md', ROOT / 'config/encoder.conf', ROOT / 'config/encoder_runtime.conf',
              ROOT / 'doc/CAUSAL_PRIMARY_INTEGRATION.md', ROOT / 'code/data/README.md',
              ROOT / 'code/data/include/cuwacunu/data/record_types.h', ROOT / 'code/data/src/record_types.cpp',
              ROOT / 'code/integration/include/cuwacunu/encoder/encoder.h',
              ROOT / 'code/latent_market/versions/v1/include/cuwacunu/latent_market/timeline.h',
              ROOT / 'code/latent_market/versions/v1/include/cuwacunu/latent_market/risk_targets.h', checkpoint, sidecar]
    pinned.extend(ROOT / source[form]['path'] for source in sources for form in ('csv', 'cache'))
    pinned.extend(p for p in sorted((ROOT / 'code/encoders/causal_patch_bottleneck_mae/versions/v3').rglob('*')) if p.is_file())
    receipt = {
        'schema': 'latent_state_control_campaign_admission_v1',
        'date': '2026-10-18', 'datasetId': 'MKT-CAUSAL-STATE-CONTROL-V1-DEV',
        'scope': 'Prospective conditional development source/support admission; no market fit, embedding capture or quality measurement',
        'runtime': {'containerName': 'cuwacunu', 'containerId': VERIFIED_CONTAINER_ID,
                    'observedContainerHostname': os.environ['HOSTNAME'],
                    'launcher': 'container.ps1', 'python': os.sys.version.split()[0]},
        'availability': {
            'id': 'ASSUMED-CANDLE-CLOSE-ZERO-LATENCY-V1', 'measuredReceiptLatency': False,
            'closeKey': 'Exchange candle close_time in epoch milliseconds',
            'assumedReceiptTime': 'close_time', 'decisionTime': 'origin close_time',
            'latencyMs': 0, 'legalObservationRule': 'close_time <= decision_time',
            'limitations': 'Historical close keys do not establish actual feed arrival or execution latency; this is conditional development evidence, with no live-use or live-causality claim',
            'missingStalePolicy': 'Require every exact one-minute tick across all BTC/ETH/SOL channels; never forward-fill stale rows; preserve invalid/unavailable associations and reject unsupported origins',
        },
        'encoderNomination': {
            'outcome': 'eligible', 'instance': 'CPB-v3.market-01', 'family': 'causal_patch_bottleneck_mae', 'version': 'v3',
            'geometry': {'channels': 3, 'history': 128, 'features': 9, 'patch': 8, 'nativeWidth': 36, 'cadenceMs': TICK},
            'checkpoint': {**pin(checkpoint), 'bytes': checkpoint.stat().st_size}, 'sidecar': pin(sidecar),
            'modelRef': EXPECTED_MODEL, 'artifactComponent': manifest['components'][0],
            'fitCutoffMs': JAN + 21 * DAY - 1, 'fitCutoffUtc': utc(JAN + 21 * DAY - 1),
            'recordedProvenance': {'sourceUpdates': 0, 'causalUpdates': 2048, 'seed': 943001,
                                   'inputScaler': 'rpb-scaler-v1-fnv1a64-9d33c2019209d321',
                                   'bankNormalizer': 'rpb-relative-bank-v1:rpb-bank-standardizer-v1-fnv1a64-4d6d0b5761eaad8f'},
            'rationale': 'Disclosed convenience baseline: reuse the exact selected clean causal native36 interface and saved TRAIN statistics to bound downstream implementation and comparison cost. No new candidate search or measured retention claim on future latent-state/risk targets is made.',
            'reusedEvidence': 'doc/CAUSAL_PRIMARY_INTEGRATION.md cites existing v3 results/review: strong volatility/linear activity and TEMPO-5 timing; weak neural trend/relation, neural activity decline, improved MAE but worse known-VALIDATION NLL. These mixed findings do not establish downstream usefulness or an overall winner.',
            'cost': 'Recorded 293672 learned values, native36 exports, H128 inference; no new throughput or compute benchmark. The existing candidate avoids a new Encoder fit; saved captures, once separately admitted, are reusable.',
            'frozenContract': 'Weights/scaler/bank remain frozen; genuine raw CPU F64, Boolean masks, semantic channel IDs 0/1/2, actual close grids and explicit per-row cutoff; no PCA, statistic fit, fallback or Encoder update.',
            'separateDecision': 'Campaign eligibility is a disclosed baseline nomination, distinct from application selection. Current-chain engineering and final campaign closure remain separate prerequisites for capture/fitting.',
            'replacementPolicy': 'A changed Encoder requires a new admitted campaign, compatible capture/state/readout artifacts and common-observable comparisons under the same periods/budgets; latent losses cannot rank different coordinate spaces.',
        },
        'sources': sources,
        'supportRecipe': {'history': 128, 'historySteps': 4, 'spacingTicks': 1,
                          'forecastHorizonsTicks': [1, 16, 128], 'futureRiskHorizonTicks': 128,
                          'fullEnvelopeTicks': [-130, 128], 'fullEnvelopeInclusiveTicks': 259,
                          'rolePolicy': 'The complete raw envelope must lie within the stage; purge 130 leading and 128 trailing origin ticks per stage',
                          'riskSupportOnly': 'No future-risk values or forecast errors are computed. Support requires all 128 future minute closes, source-valid finite F64 features and positive closes on every channel.'},
        'stages': stages,
        'chronologyAdmission': {'outcome': 'eligible',
                               'scope': 'Feasible fixed-stage development chronology at registered minimum legal origin counts 512/512/1024; conditional on the assumed availability policy',
                               'stateAndControlFitEndsMs': stages[0]['stageFitOrScoringCutoffMs'],
                               'riskHeadFitEndsMs': stages[1]['stageFitOrScoringCutoffMs'],
                               'guarantee': 'All retained history/target envelopes lie within disjoint stages; Encoder fit precedes representation history, representation/control fitting precedes later forecast issuance/risk fitting, and risk fitting precedes development evaluation. No cross-fitting is needed for this declared feasibility assessment.',
                               'limits': 'Counts establish support and chronology, not statistical power, component retention, forecast skill or risk utility; future learned methods must still meet numeric protocol and frozen-output gates.'},
        'exposure': {'2024-01-01/2024-01-21': 'Encoder fitting/statistics exposure; no independent evidence',
                     '2024-01-22/2024-01-28': 'Previously development-exposed; conditional development reuse only',
                     '2024-01-29/2024-01-31': 'Preserved, unassigned; no price/label/outcome decoding or TEST designation in this inventory. Whole-file hashes include these bytes only as immutable source identity.'},
        'execution': {'encoderCalls': 0, 'encoderUpdates': 0, 'statisticFits': 0, 'modelFits': 0,
                      'captureCalls': 0, 'qualityMetricsComputed': 0, 'sourceOrCacheWrites': 0},
        'sourceFiles': [pin(p) for p in sorted(set(pinned))],
    }
    print(json.dumps(receipt, indent=2) + '\n', end='')


if __name__ == '__main__':
    main()
