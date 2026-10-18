#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Artificial shared-study control publication; no market or Encoder execution."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

if not Path('/.dockerenv').is_file():
    raise SystemExit('Use the managed Linux container.')
build = Path(sys.argv[1]).resolve()
if not build.is_relative_to(Path('/opt/cuwacunu/build')):
    raise SystemExit('Engineering artifacts must stay in the container build tree.')
parent = Path(tempfile.mkdtemp(prefix='embedding-controls-', dir=build/'tmp'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields_digest(domain, fields):
    result = hashlib.sha256()
    for value in ['cuwacunu_artifact_record_v1', domain, str(len(fields)), *fields]:
        raw = value.encode()
        result.update(len(raw).to_bytes(8, 'little'))
        result.update(raw)
    return result.hexdigest()


def run(arguments, succeeds=True):
    result = subprocess.run([str(v) for v in arguments], text=True, capture_output=True)
    with (parent/'pipeline.log').open('a') as stream:
        stream.write('$ '+' '.join(str(v) for v in arguments)+'\n'+result.stdout+result.stderr)
    assert (result.returncode == 0) == succeeds, result.stdout+result.stderr
    return result.stdout+result.stderr


fixture = parent/'fixture'
run([build/'latent_market_v1_pipeline_fixture', fixture])
original = (fixture/'study.conf').read_text()


def setting(key):
    return re.search(r'\b'+key+r'\s*=\s*([^;]+);', original).group(1).strip()


card = parent/'synthetic-control-card.txt'
card.write_text('SYNTHETIC DIRECT EMBEDDING CONTROL ENGINEERING CARD V1\n'
                'Artificial saved native D4,K4,horizons1/4 from existing pipeline fixture.\n'
                'Independent tanh8 residual models; hidden Xavier gain1, zero output and biases.\n'
                'TRAIN-weighted input transform and RMS residual scales, floor1e-8.\n'
                'AdamW lr.001,beta.9/.999,eps1e-8,decay.0001,clip1;3updates/horizon,batch4.\n'
                'Repetition seeds17,29; local initialization/sampling seed=seed+horizon.\n'
                'Publish complete checked inference/training artifacts and pinned predictions.\n'
                'No actual Encoder, market fitting, quality/point-skill or utility claim.\n')
protocol = fields_digest('latent-state-control-study-protocol-v1', [digest(card)])
admission = parent/'synthetic-fit-admission.txt'
bindings = {'scope': 'synthetic-engineering', 'protocol_sha256': protocol,
            'timeline_sha256': digest(fixture/'timeline.bin'), 'pairs_sha256': digest(fixture/'pairs.bin'),
            'source_closure_sha256': digest(fixture/'source-closure.sha256'),
            'encoder_model_ref': setting('ENCODER_MODEL_REF'), 'fit_cutoff_ms': setting('FIT_CUTOFF_MS'),
            'current_chain_and_control_witnesses': 'synthetic-fixture-only'}
admission.write_text('latent-market-control-fit-admission-v1\n'+''.join(f'{k} {v}\n' for k,v in bindings.items()))
settings = parent/'controls.conf'
paths = {'CARD': card, 'SOURCE_CLOSURE': fixture/'source-closure.sha256', 'TIMELINE': fixture/'timeline.bin',
         'PAIRS': fixture/'pairs.bin', 'CAPTURE_RECEIPT': fixture/'preparation.txt', 'FIT_ADMISSION': admission}
text = 'DIRECT_EMBEDDING_CONTROL_STUDY_V1 {\n  SCOPE = synthetic-engineering;\n'
for key,path in paths.items():
    text += f'  {key} = {path};\n  {key}_SHA256 = {digest(path)};\n'
text += (f'  ENCODER_CHECKPOINT = {fixture/"fixture-checkpoint.txt"};\n'
         f'  ENCODER_SIDECAR_SHA256 = {setting("ENCODER_SIDECAR_SHA256")};\n'
         f'  ENCODER_MODEL_REF = {setting("ENCODER_MODEL_REF")};\n'
         f'  FIT_CUTOFF_MS = {setting("FIT_CUTOFF_MS")};\n'
         '  HIDDEN_WIDTH = 8;\n  UPDATES = 3;\n  MICROBATCH = 4;\n  SEEDS = 17,29;\n}\n')
settings.write_text(text)
output = parent/'published'
run([build/'latent_market_v1_embedding_controls', settings, output])
comparison = json.loads((output/'comparison.json').read_text())
assert len(comparison['controls']) == 2
assert comparison['protocol_sha256'] == protocol
assert len(comparison['bundle_ref']) == 64
for selected in comparison['controls']:
    directory = output/selected['role']
    inference = json.loads((directory/'inference.pt.manifest.json').read_text())
    training = json.loads((directory/'training.pt.manifest.json').read_text())
    assert inference['archive_kind'] == 'inference' and training['archive_kind'] == 'training'
    assert digest(directory/'inference.pt') == inference['archive_sha256']
    assert digest(directory/'predictions.pt') == selected['output_sha256']
    assert digest(directory/'inference.pt.manifest.json') == selected['sidecar_sha256']
receipt = (output/'study-controls.txt').read_text()
assert 'encoder_calls 0\nencoder_updates 0\nTEST_opened false\npoint_skill_decision open' in receipt

# A wrong upstream ModelRef fails before constructing an output leaf.
wrong = parent/'wrong-parent.conf'
wrong.write_text(text.replace('ENCODER_MODEL_REF = '+setting('ENCODER_MODEL_REF'), 'ENCODER_MODEL_REF = '+'0'*64))
run([build/'latent_market_v1_embedding_controls', wrong, parent/'wrong-parent-output'], succeeds=False)
assert not (parent/'wrong-parent-output').exists()

# Changed admitted source bytes do not get silently consumed.
stale = parent/'stale-pairs.conf'
stale.write_text(text.replace('PAIRS_SHA256 = '+digest(fixture/'pairs.bin'), 'PAIRS_SHA256 = '+'0'*64))
run([build/'latent_market_v1_embedding_controls', stale, parent/'stale-output'], succeeds=False)
assert not (parent/'stale-output').exists()

# Prospective registration does not authorize market fitting or a caller-written pass.
market = parent/'market-scope.conf'
market.write_text(text.replace('SCOPE = synthetic-engineering', 'SCOPE = admitted-development'))
run([build/'latent_market_v1_embedding_controls', market, parent/'market-output'], succeeds=False)
assert not (parent/'market-output').exists()

# The same output cannot be overwritten by another run.
before = digest(output/'comparison.json')
run([build/'latent_market_v1_embedding_controls', settings, output], succeeds=False)
assert digest(output/'comparison.json') == before
print('PASS artificial shared-study controls: all repetitions, complete typed artifacts, checked comparison pins, stale-parent/input rejection, exclusive publication; no market claim')
print('Evidence:', parent)
