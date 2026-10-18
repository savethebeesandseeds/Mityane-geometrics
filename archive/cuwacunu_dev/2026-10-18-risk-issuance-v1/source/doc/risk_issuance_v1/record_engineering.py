#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Record separately observed synthetic head-stage checks and compiler closure."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path('/cuwacunu')
SESSION = Path('/opt/cuwacunu/build/risk-issuance-final-20261018')
DIRECTORY = ROOT / 'doc/risk_issuance_v1'
COMMAND = ['bash', 'code/scripts/task.sh', SESSION.name, '-f', 'Makefile', '-f',
           'code/latent_market/risk_issuance/versions/v1/v1.mk', '-j2',
           'test-risk-issuance-v1', 'test-risk-issuance-pipeline']


def require(value, message):
    if not value:
        raise SystemExit(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def pin(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def main():
    require(Path('/.dockerenv').exists() and Path.cwd() == ROOT, 'Use the managed container at /cuwacunu.')
    destination = DIRECTORY / 'engineering.json'
    require(not destination.exists(), 'Preserve the existing receipt; use a new attempt record.')
    completion_path = DIRECTORY / 'final-command-result.json'
    completion = json.loads(completion_path.read_text())
    log = DIRECTORY / 'build-test.log'
    require(completion.get('schema') == 'observed-managed-container-command-result-v1' and
            completion.get('session') == SESSION.name and completion.get('exitCode') == 0 and
            completion.get('command') == COMMAND and completion.get('logSha256') == digest(log),
            'A separately observed successful final command must bind this exact log and session.')
    snapshot_path = DIRECTORY / 'build-source-verification.json'
    snapshot = json.loads(snapshot_path.read_text())
    snapshot_inputs = json.loads((DIRECTORY / 'build-source-inputs.json').read_text())
    snapshot_root = Path(snapshot['buildSourceRoot']).resolve(strict=True)
    require(snapshot['schema'] == 'verified-container-build-source-snapshot-v1' and
            snapshot['sourceRoot'] == str(ROOT) and snapshot_root.is_relative_to(Path('/opt/cuwacunu/build')) and
            digest(DIRECTORY / 'build-source-inputs.json') == snapshot['manifestSha256'] and
            snapshot['files'] == snapshot_inputs['files'] and
            completion.get('snapshotCompilationExitCode') == 0 and
            completion.get('snapshotCompilationLogSha256') == digest(DIRECTORY / 'snapshot-build.log'),
            'Separately observed snapshot compilation and verified source inventory are required.')
    for entry in snapshot['files']:
        require(digest(ROOT / entry['path']) == entry['sha256'] == digest(snapshot_root / entry['path']),
                'Live and compiled snapshot bytes differ: ' + entry['path'])
    snapshot_pins = {entry['path']: entry['sha256'] for entry in snapshot['files']}
    require(len(snapshot_pins) == len(snapshot['files']), 'Duplicate snapshot source paths.')
    text = log.read_text(encoding='utf-8-sig')
    counts = re.findall(r'(\d+) risk issuance v1 synthetic checks passed', text)
    require(counts and int(counts[-1]) > 0 and 'PASS staged risk-head pipeline:' in text and
            'make: ***' not in text, 'Required final unit and integrated pipeline passes are missing.')
    matches = re.findall(r'Evidence preserved at (/opt/cuwacunu/build/[^\r\n]+)', text)
    require(matches, 'Preserved final synthetic pipeline evidence is missing.')
    evidence = Path(matches[-1].strip()).resolve(strict=True)
    require(evidence.is_relative_to((SESSION / 'tmp').resolve(strict=True)), 'Unexpected evidence directory.')
    leaf = evidence / 'complete'
    source = ROOT / 'code/latent_market/risk_issuance/versions/v1/tests/pipeline_test.py'
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('observed_risk_stage_checks', source)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    manifest, panel = checks.check_publication(leaf)
    require(manifest['policy_sha256'] == digest(DIRECTORY / 'policy.json'), 'The exact prospective stage policy differs.')
    require(panel == json.loads((evidence / 'summary.json').read_text()), 'Retained panel counts differ.')
    source_tool = ROOT / 'code/latent_market/risk_issuance/versions/v1/source_identity.py'
    source_spec = importlib.util.spec_from_file_location('risk_implementation_identity', source_tool)
    identity = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(identity)
    registry = json.loads((ROOT / identity.REGISTRY).read_text())
    require(manifest['source_closure_sha256'] == identity.identity(ROOT, registry), 'Compiled implementation closure differs.')
    # Verify that this additive task did not change either earlier implementation.
    previous = []
    for name in ['doc/forecast_issuance_v1/engineering.json', 'doc/latent_state_control_v1/engineering.json']:
        record = json.loads((ROOT / name).read_text())
        for entry in record['sourceFiles']:
            require(digest(ROOT / entry['path']) == entry['sha256'], 'Protected earlier source changed: ' + entry['path'])
        previous.append(pin(ROOT / name))
    sources = {Path('Makefile'), Path('code/scripts/task.sh'), Path('container.ps1'),
               Path('environment-image.json'), Path('doc/risk_issuance_v1/policy.json'),
               Path('doc/risk_issuance_v1/record_engineering.py'), completion_path.relative_to(ROOT), log.relative_to(ROOT)}
    sources.update(Path(name) for name in registry)
    sources.update(path.relative_to(ROOT) for path in [snapshot_path, DIRECTORY / 'build-source-inputs.json',
        DIRECTORY / 'snapshot-build.log', DIRECTORY / 'prepare-build-source.ps1', DIRECTORY / 'prepare_build_source.py'])
    for path in (ROOT / 'code/latent_market/risk_issuance/versions/v1').rglob('*'):
        if path.is_file() and not path.is_symlink():
            sources.add(path.relative_to(ROOT))
    dependencies = []
    expected = ['risk_issuance_v1_test.d', 'risk_issuance_v1_pipeline.d',
                'latent_dynamics_v1_baselines.d', 'latent_dynamics_v1_risk_readouts.d']
    require(all((SESSION / name).is_file() for name in expected), 'Every actual compiler dependency file is required.')
    for path in sorted(SESSION.glob('*.d')):
        lines = path.read_text().replace('\\\n', '').splitlines()
        require(lines and ':' in lines[0], 'Empty or malformed compiler dependencies.')
        local = []
        for name in shlex.split(lines[0].split(':', 1)[1]):
            actual = Path(name)
            if not actual.is_absolute():
                actual = ROOT / actual
            actual = actual.resolve(strict=True)
            if actual.is_relative_to(ROOT):
                relative = actual.relative_to(ROOT)
                require(relative.as_posix() in snapshot_pins and digest(actual) == snapshot_pins[relative.as_posix()],
                        'Actual compiler dependency is absent from the verified snapshot: ' + str(relative))
                sources.add(relative)
                local.append(relative.as_posix())
        require(local, 'Compiler dependencies contain no local sources.')
        dependencies.append({'path': str(path), 'sha256': digest(path), 'projectSources': local})
    binaries = []
    for name in ['risk_issuance_v1_test', 'risk_issuance_v1_pipeline']:
        path = SESSION / name
        require(path.is_file() and path.stat().st_size > 0, 'A required final executable is missing.')
        binaries.append({'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size})
    artifacts = [{'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
                 for path in sorted(evidence.rglob('*')) if path.is_file()]
    require(artifacts and (leaf / 'published/readouts.pt').is_file() and
            (leaf / 'published/predictions.pt').is_file(), 'Complete synthetic output artifacts are required.')
    receipt = {
        'schema': 'cuwacunu-staged-risk-head-engineering-v1', 'date': '2026-10-18',
        'milestone': 'risk-chronological-issuance',
        'scope': 'Additive head-side chronology and immutable saved-array serving, with a seven-arm artificial engineering panel only',
        'container': {'name': 'cuwacunu', 'id': '82d102f2ef3f559b38e99622d771dd6b6c1f10dedecce1c42c498afe9903440e', 'session': SESSION.name},
        'compilationSourceSnapshot': {'root': str(snapshot_root), 'verification': pin(snapshot_path),
                                      'inputFiles': len(snapshot['files']), 'reason': 'Avoid Windows-mounted header-lookup latency; final tests execute against live /cuwacunu source.'},
        'command': COMMAND, 'exitCode': 0, 'syntheticUnitChecks': int(counts[-1]),
        'completedCommand': pin(completion_path), 'policy': pin(DIRECTORY / 'policy.json'), 'log': pin(log),
        'headModelRef': manifest['head_model_ref'], 'bankIdentity': manifest['bank_identity'],
        'implementationSourceSha256': manifest['source_closure_sha256'], 'panel': panel,
        'previousReceiptsPreserved': previous, 'binaries': binaries, 'compilerDependencies': dependencies,
        'evidenceDirectory': str(evidence), 'artifacts': artifacts,
        'checks': ['Checked issued later TRAIN features with exact Encoder/state/target/Dynamics and direct-control ancestry',
                   'TRAIN-only numerical fitting views with mature, stage-contained per-observable common labels',
                   'Strict origin-side upstream and head cutoff checks, including equality and delayed decisions',
                   'Evaluation and excluded-label independence of fitted parameters and frozen normalizers',
                   'Label-free prediction, retained invalid associations and compatible common-observable scoring',
                   'Private immutable heads and authenticated exact-byte head/prediction/ledger roundtrips',
                   'Corruption, parent, coordinate, stage, retrospective, overwrite and market-scope rejection',
                   'No Encoder calls or upstream updates during head fitting and serving'],
        'sourceFiles': [pin(ROOT / path) for path in sorted(sources)],
        'limits': ['All fitted tensors and observed scores belong to artificial engineering fixtures; no market capture or fitting is performed.',
                   'Synthetic upstream initialization is separate from the head phase; the head phase consumes frozen checked producers.',
                   'The registered market panel still requires distinct ridge and residual-MLP forecast arms plus conventional/capacity/placebo controls.',
                   'No state eligibility, latent point skill, actual risk utility, independent confirmation, affine campaign-folding or portfolio decision is closed.',
                   'Cross-fitting and pooling independently fitted coordinate vectors remain unadmitted; one head belongs to one exact bank.',
                   'ObservableTargets is caller-authenticated label/formula/source authority; application adapters must authenticate raw label archives and original associations. The synthetic adapter verifies its pinned future-risk archive.',
                   'The historical retrospective utility diagnostic and all earlier source/evidence identities remain unchanged.']}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print('Engineering receipt:', destination)
    print('SHA256:', digest(destination))


if __name__ == '__main__':
    main()
