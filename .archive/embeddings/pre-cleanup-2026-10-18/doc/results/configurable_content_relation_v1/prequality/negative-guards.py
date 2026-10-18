"""Actual CLI rejection witnesses; no successful candidate or optimizer call."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('capsule', type=Path)
args = parser.parse_args()
root = args.capsule
check(Path('/.dockerenv').is_file(), 'managed container required')
check(root.is_absolute() and root.resolve(strict=True) == root and not root.is_symlink()
      and root.parent == Path('/embedding/output/runs/rpb-v22-content-relation')
      and root.name.startswith('admission-'), 'exact new candidate namespace')
receipt = json.loads((root / 'freeze/receipt.json').read_text())
admission = json.loads((root / 'quality-admission.json').read_text())
check(admission['status'] == 'admitted', 'completed actual CUDA admission required')
check(not (root / 'candidates').exists() and not (root / 'screen').exists(), 'before quality')
build = Path(receipt['build_directory'])
check(build == Path('/opt/cuwacunu_embedding/build/rpb-v22-content-relation'), 'named build')
binary = build / 'embedding_configurable_content_relation'
out = root / 'negative-admission-guards'
out.mkdir()
temporary = root / 'candidates'
temporary.mkdir()
check(temporary.resolve(strict=True).parent == root, 'owned empty candidate directory')
base = [str(binary), '--data-root', receipt['original_capsule'] + '/quality-data',
        '--inputs-sha256', str(root / 'freeze/inputs.sha256'), '--original-capsule',
        receipt['original_capsule'], '--output', str(temporary / 'master-941101'),
        '--profile', 'A', '--master', '941101', '--config', str(root / 'freeze/config.conf'),
        '--card', str(root / 'freeze/card.md'), '--card-sha256', receipt['card_sha256'],
        '--source-id', receipt['source_fingerprint']]
records = []
try:
    for name, mutations, expected in (
        ('wrong-source', {'--source-id': '0' * 64}, 'compiled source and exact prospective card binding'),
        ('wrong-card', {'--card-sha256': '0' * 64}, 'compiled source and exact prospective card binding'),
        ('early-third', {'--master': '941103', '--output': str(temporary / 'master-941103')},
         'third-admission.json')):
        command = base.copy()
        for key, value in mutations.items():
            command[command.index(key) + 1] = value
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, check=False)
        log = out / (name + '.log')
        with log.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(result.stdout)
        check(result.returncode != 0 and expected in result.stdout,
              'specific rejection: ' + name)
        check(not list(temporary.iterdir()), 'rejection creates no candidate output')
        records.append(dict(name=name, argv=command, exit_code=result.returncode,
                            log_sha256=sha(log), candidate_outputs=0))
finally:
    # This is only the exact empty directory created above, never recursive removal.
    check(temporary.resolve(strict=True).parent == root and not list(temporary.iterdir()),
          'owned directory remains empty')
    temporary.rmdir()
value = dict(status='passed', protocol=receipt['protocol'],
             source_fingerprint=receipt['source_fingerprint'], card_sha256=receipt['card_sha256'],
             producer_binary_sha256=sha(binary), script_sha256=sha(Path(__file__)),
             probes=records, candidate_outputs=0)
with (out / 'receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(value))
