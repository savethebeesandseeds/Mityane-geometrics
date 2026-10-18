#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Declare a CPU future-risk correctness comparison for synthetic study outputs."""
import hashlib
from pathlib import Path
import re
import sys

if not Path('/.dockerenv').exists():
    raise SystemExit('Run only inside the managed Linux container.')
if len(sys.argv) != 4:
    raise SystemExit('usage: make_utility_fixture.py SYNTHETIC_INPUT_LEAF STUDY_OUTPUT_LEAF NEW_SETTINGS_FILE')
inputs, study, destination = (Path(p).resolve() for p in sys.argv[1:])
build = Path('/opt/cuwacunu/build')
if any(not p.is_relative_to(build) for p in (inputs, study, destination)):
    raise SystemExit('Synthetic fixtures must remain under /opt/cuwacunu/build.')

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''): h.update(part)
    return h.hexdigest()

settings = (inputs/'study.conf').read_text()
def integer(key):
    return int(re.search(r'\b'+key+r'\s*=\s*([0-9]+)\s*;', settings).group(1))
card = destination.with_suffix('.card.txt')
with card.open('x') as stream:
    stream.write('SYNTHETIC FUTURE-RISK UTILITY V1 ENGINEERING CARD\n'
                 'Seven fixed arms: current_e/history_e/current_z/history_z/e_history+forecast_e/'
                 'e_history+forecast_z/z_history+forecast_z. Every declared horizon; same K4 history.\n'
                 'Ridge penalty .01, std floor 1e-6. Identical complete mature TRAIN rows and heldout intersection.\n'
                 'Per-asset centered population future one-minute log-return dispersion and origin-relative downside.\n'
                 'TRAIN predictor features are retrospective input-only; VALIDATION uses issued forecasts.\n'
                 'No market evidence, quality threshold, statistical independence or portfolio claim.\n')
files = {'CARD': card, 'SOURCE_CLOSURE': inputs/'source-closure.sha256',
         'TIMELINE': inputs/'timeline.bin', 'PAIRS': inputs/'pairs.bin', 'FUTURE_RISK': inputs/'future-risk.bin',
         'STATE_BANK': study/'state-bank.pt', 'STATE_PREDICTIONS': study/'state-predictions.pt',
         'E_PREDICTIONS': study/'e-controls/ridge-k4.pt', 'E_MODEL': study/'e-controls/baseline-k4.pt'}
with destination.open('x') as stream:
    stream.write('FUTURE_RISK_UTILITY_V1 {\n')
    for key, path in files.items():
        stream.write(f'  {key} = {path};\n  {key}_SHA256 = {digest(path)};\n')
    fit = integer('FIT_CUTOFF_MS')
    stream.write(f'  FIT_CUTOFF_MS = {fit};\n  REPRESENTATION_FIT_CUTOFF_MS = {fit};\n'
                 f'  SCORING_CUTOFF_MS = {integer("SCORING_CUTOFF_MS")};\n'
                 '  RIDGE_PENALTY = 0.01;\n  STD_FLOOR = 1e-6;\n}\n')
print(f'Synthetic utility settings: {destination}')
