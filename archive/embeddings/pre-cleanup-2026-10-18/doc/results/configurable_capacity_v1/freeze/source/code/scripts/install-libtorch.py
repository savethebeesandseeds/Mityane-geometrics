#!/usr/bin/env python3
"""Copy or verify the existing pinned Linux LibTorch SDK inside the container.

No downloads, package installs, container lifecycle, build, SDK execution, or
SDK deletions occur here.  A partial/conflicting destination is preserved and
rejected.  Completion evidence is outside the SDK.  --self-test uses temporary
artificial bundles only, never either real SDK path.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import uuid

sys.dont_write_bytecode = True
VERSION = '2.6.0+cu124'
ABI = 1
SOURCE = Path('/embedding/.external/libtorch')
TARGET = Path('/opt/cuwacunu_embedding/libtorch')
SETUP = Path('/opt/cuwacunu_embedding/setup')
POINTER = 'libtorch-installed.json'
PROFILE = Path('/etc/profile.d/embedding.sh')
CHUNK = 8 * 1024 * 1024
KIND = 'cuwacunu_libtorch_installation_v1'
OLD_RUNTIME = b'/embedding/.external/libtorch/lib'
NEW_RUNTIME = b'/opt/cuwacunu_embedding/libtorch/lib'
OLD_PROFILE = b'''export CUDA_VERSION=12.4
export CUDNN_VERSION=9
case ":$PATH:" in
  *:/usr/local/cuda-12.4/bin:*) ;;
  *) export PATH="/usr/local/cuda-12.4/bin:$PATH" ;;
esac
case "${LD_LIBRARY_PATH:-}" in
  /embedding/.external/libtorch/lib:/usr/local/cuda-12.4/lib64*) ;;
  *) export LD_LIBRARY_PATH="/embedding/.external/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
'''
NEW_PROFILE = OLD_PROFILE.replace(OLD_RUNTIME, NEW_RUNTIME)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def exists(path):
    return os.path.lexists(path)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate proof JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise AssertionError('nonfinite proof JSON: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def canonical_directory(path):
    require(path.is_absolute() and path.is_dir() and not path.is_symlink() and
            path.resolve(strict=True) == path, 'canonical existing directory: ' + str(path))
    for ancestor in path.parents:
        require(not ancestor.is_symlink(), 'redirected directory ancestor: ' + str(ancestor))


def canonical_file(path):
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and
            path.resolve(strict=True) == path and path.stat().st_nlink == 1,
            'canonical regular evidence file: ' + str(path))
    canonical_directory(path.parent)


def safe_relative(value):
    path = Path(value)
    require(value and '\\' not in value and '\n' not in value and '\r' not in value and
            not path.is_absolute() and all(part not in ('', '.', '..') for part in path.parts) and
            path.as_posix() == value, 'safe SDK relative path')
    return path


def tree(root):
    """Admit the entire lexical tree without following symlink directories."""
    canonical_directory(root)
    rows = []
    def visit(directory):
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            relative = path.relative_to(root).as_posix()
            safe_relative(relative)
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            require(not mode & (stat.S_ISUID | stat.S_ISGID), 'privileged SDK mode rejected')
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                require(target and not Path(target).is_absolute(), 'SDK symlink must be relative and internal')
                resolved = path.resolve(strict=True)
                require(resolved.is_relative_to(root) and resolved != root,
                        'SDK symlink escape or invalid root target')
                rows.append({'path': relative, 'kind': 'symlink', 'target': target})
            elif stat.S_ISDIR(info.st_mode):
                require(path.resolve(strict=True) == path, 'SDK directory redirect')
                rows.append({'path': relative, 'kind': 'directory', 'mode': mode})
                visit(path)
            else:
                require(stat.S_ISREG(info.st_mode), 'non-regular SDK member rejected')
                rows.append({'path': relative, 'kind': 'file', 'bytes': info.st_size, 'mode': mode,
                             '_signature': (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)})
    visit(root)
    require(rows and len({r['path'] for r in rows}) == len(rows), 'nonempty unique SDK tree')
    return sorted(rows, key=lambda row: row['path'])


def open_regular(path, signature=None):
    # Prevent a file changed to a symlink between tree admission and opening.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    info = os.fstat(descriptor)
    actual = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if not stat.S_ISREG(info.st_mode) or (signature is not None and actual != signature):
        os.close(descriptor)
        raise AssertionError('SDK file changed after path admission: ' + str(path))
    return os.fdopen(descriptor, 'rb'), actual


def hash_file(path, signature=None):
    stream, before = open_regular(path, signature)
    value = hashlib.sha256()
    size = 0
    with stream:
        while chunk := stream.read(CHUNK):
            value.update(chunk); size += len(chunk)
        info = os.fstat(stream.fileno())
        require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == before,
                'SDK file changed while hashing')
    return size, value.hexdigest()


def bundle_identity(root):
    # This reads only small metadata/header bytes; it never executes SDK code.
    paths = [root/'build-version', root/'share/cmake/Torch/TorchConfig.cmake',
             root/'share/cmake/Torch/TorchConfigVersion.cmake',
             root/'include/torch/csrc/api/include/torch/torch.h']
    require(all(p.is_file() for p in paths), 'complete LibTorch headers and CMake metadata required')
    require(paths[0].read_text().strip() == VERSION, 'expected LibTorch ' + VERSION)
    config = paths[1].read_text()
    abis = re.findall(r'set\(TORCH_CXX_FLAGS\s+"-D_GLIBCXX_USE_CXX11_ABI=([01])"\)', config)
    require(abis == ['1'], 'exact Linux C++11 ABI=1 CMake declaration')
    require(re.findall(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', paths[2].read_text()) == ['2.6.0'],
            'matching Torch CMake package version')
    for name in ('libtorch.so', 'libtorch_cpu.so', 'libtorch_cuda.so', 'libc10.so', 'libc10_cuda.so'):
        path = root/'lib'/name
        require(path.is_file() and path.resolve(strict=True).is_relative_to(root),
                'internal Linux library required: ' + name)
        with path.open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', 'Linux ELF library required: ' + name)
    return {'build_version': VERSION, 'cxx11_abi': ABI,
            'metadata_sha256': {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in paths[:3]}}


def hashed_tree(root, rows=None):
    rows = tree(root) if rows is None else rows
    result = []
    for row in rows:
        record = {k: v for k, v in row.items() if not k.startswith('_')}
        if row['kind'] == 'file':
            size, checksum = hash_file(root/safe_relative(row['path']), row['_signature'])
            require(size == row['bytes'], 'SDK member size')
            record['sha256'] = checksum
        result.append(record)
    return result


def durable_new(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest(raw)


def profile_plan(profile):
    if profile is None:
        return {'state': 'not_requested'}
    canonical_directory(profile.parent)
    if not exists(profile):
        return {'state': 'absent', 'path': str(profile)}
    canonical_file(profile)
    raw = profile.read_bytes()
    require(raw in (OLD_PROFILE, NEW_PROFILE), 'unknown/custom runtime profile preserved without changes')
    return {'state': 'legacy' if raw == OLD_PROFILE else 'internal',
            'path': str(profile), 'sha256': digest(raw), 'mode': stat.S_IMODE(profile.stat().st_mode)}


def configure_profile(plan, evidence):
    if plan['state'] != 'legacy':
        return dict(plan)
    profile = Path(plan['path'])
    canonical_file(profile)
    require(profile.read_bytes() == OLD_PROFILE and digest(OLD_PROFILE) == plan['sha256'],
            'known runtime profile unchanged after admission')
    require(OLD_PROFILE.count(OLD_RUNTIME) == 2 and NEW_PROFILE.count(NEW_RUNTIME) == 2 and
            NEW_PROFILE.replace(NEW_RUNTIME, OLD_RUNTIME) == OLD_PROFILE,
            'exact two-prefix profile reverse-byte proof')
    backup = evidence/'embedding-profile.before.sh'
    with backup.open('xb') as stream:
        stream.write(OLD_PROFILE); stream.flush(); os.fsync(stream.fileno())
    temporary = profile.parent/('.embedding-sdk-' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(NEW_PROFILE); stream.flush(); os.fsync(stream.fileno())
    os.chmod(temporary, plan['mode'])
    canonical_file(profile)
    require(profile.read_bytes() == OLD_PROFILE, 'runtime profile unchanged before atomic replacement')
    os.replace(temporary, profile)
    descriptor = os.open(profile.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(profile.read_bytes() == NEW_PROFILE and backup.read_bytes() == OLD_PROFILE,
            'profile update and exact preserved backup')
    return {'state': 'updated_to_internal', 'path': str(profile),
            'before_sha256': digest(OLD_PROFILE), 'after_sha256': digest(NEW_PROFILE),
            'backup_path': str(backup), 'backup_sha256': digest(OLD_PROFILE),
            'two_prefix_reverse_byte_proof': True, 'CUDA_PATH_and_CUDNN_lines_unchanged': True}


def verify_profile(profile):
    plan = profile_plan(profile)
    require(plan['state'] != 'legacy', 'legacy SDK runtime profile remains active; invoke installer to migrate it')
    return plan


def ensure_parent(path):
    # Never traverse or replace a preexisting symlink/conflicting directory.
    if exists(path):
        canonical_directory(path)
    else:
        canonical_directory(path.parent)
        path.mkdir()
        canonical_directory(path)


def verify_complete(source, target, setup, compare_source=True, profile=None):
    canonical_directory(target); canonical_directory(setup)
    pointer = setup/POINTER
    require(exists(pointer), 'unproven/partial SDK target preserved; no completion pointer')
    canonical_file(pointer)
    reference = parse(pointer.read_bytes())
    require(reference['artifact_kind'] == KIND and reference['target'] == str(target), 'SDK pointer identity')
    proof_path = Path(reference['proof_path'])
    require(proof_path.name == 'proof.json' and proof_path.parent.parent == setup and
            proof_path.parent.name.startswith('sdk-'), 'proof outside SDK in an exclusive setup record')
    canonical_file(proof_path)
    proof_raw = proof_path.read_bytes()
    require(digest(proof_raw) == reference['proof_sha256'], 'exact completed SDK proof')
    proof = parse(proof_raw)
    require(proof['artifact_kind'] == KIND and proof['status'] == 'complete' and
            proof['target'] == str(target) and proof['source'] == str(source) and
            proof['build_version'] == VERSION and proof['cxx11_abi'] == ABI,
            'pinned completed SDK identity')
    inventory_path = proof_path.parent/'inventory.json'
    canonical_file(inventory_path); raw_inventory = inventory_path.read_bytes()
    require(digest(raw_inventory) == proof['inventory_sha256'], 'exact completed file inventory')
    inventory = parse(raw_inventory)
    require(inventory['artifact_kind'] == KIND and inventory['source'] == str(source) and
            inventory['target'] == str(target), 'inventory source and target')
    expected = inventory['entries']
    # Re-admit all destination members before reading any SDK file bytes.
    target_rows = tree(target)
    require([{k:v for k,v in r.items() if k != '_signature'} for r in target_rows] ==
            [{k:v for k,v in r.items() if k != 'sha256'} for r in expected], 'complete SDK paths/types/sizes/modes')
    require(hashed_tree(target, target_rows) == expected, 'every installed SDK member SHA unchanged')
    require(bundle_identity(target) == proof['bundle_identity'], 'installed SDK ABI/version metadata')
    if compare_source and exists(source):
        source_rows = tree(source)
        require(bundle_identity(source) == proof['bundle_identity'] and
                hashed_tree(source, source_rows) == expected, 'existing local source remains identical to installed SDK')
    profile_info = verify_profile(profile)
    return {'status': 'verified', 'artifact_kind': KIND, 'target': str(target),
            'proof_path': str(proof_path), 'proof_sha256': reference['proof_sha256'],
            'inventory_sha256': proof['inventory_sha256'], 'regular_files': proof['regular_files'],
            'regular_file_bytes': proof['regular_file_bytes'], 'sdk_mutations': 0,
            'runtime_profile': profile_info}


def install(source, target, setup, installer_sha, profile=None):
    # Reject an unfamiliar profile before creating/copying any SDK destination.
    runtime = profile_plan(profile)
    if exists(target):
        result = verify_complete(source, target, setup)
        if runtime['state'] == 'legacy':
            evidence = setup/('sdk-profile-' + uuid.uuid4().hex)
            evidence.mkdir()
            info = configure_profile(runtime, evidence)
            record = {'artifact_kind': KIND, 'status': 'complete', 'SDK_proof_sha256': result['proof_sha256'],
                      'installer_sha256': installer_sha, 'runtime_profile': info, 'SDK_mutations': 0}
            profile_proof = durable_new(evidence/'proof.json', record)
            result['runtime_profile_proof_sha256'] = profile_proof
        result['runtime_profile'] = verify_profile(profile)
        return result
    require(not exists(setup/POINTER), 'orphan SDK completion pointer preserved; target is absent')
    # Whole source tree and safe symlinks are admitted before metadata reads or target creation.
    rows = tree(source)
    identity = bundle_identity(source)
    canonical_directory(target.parent)
    ensure_parent(setup)
    leaf = setup/('sdk-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12])
    leaf.mkdir()
    durable_new(leaf/'attempt.json', {'artifact_kind': KIND, 'status': 'started',
                'source': str(source), 'target': str(target), 'installer_sha256': installer_sha,
                'build_version': VERSION, 'cxx11_abi': ABI})
    # mkdir is exclusive even if another process changed the target after the check.
    target.mkdir()
    try:
        # Make real directories first. Symlinks are created only after file copies.
        for row in rows:
            if row['kind'] == 'directory':
                (target/safe_relative(row['path'])).mkdir()
        expected = []
        for row in rows:
            record = {k:v for k,v in row.items() if not k.startswith('_')}
            destination = target/safe_relative(row['path'])
            if row['kind'] == 'file':
                incoming, signature = open_regular(source/safe_relative(row['path']), row['_signature'])
                checksum = hashlib.sha256(); size = 0
                with incoming, destination.open('xb') as outgoing:
                    while chunk := incoming.read(CHUNK):
                        checksum.update(chunk); size += len(chunk); outgoing.write(chunk)
                    info = os.fstat(incoming.fileno())
                    require((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns) == signature,
                            'source changed while copying SDK member')
                    outgoing.flush(); os.fsync(outgoing.fileno())
                os.chmod(destination, row['mode'])
                target_size, target_hash = hash_file(destination)
                require(size == target_size == row['bytes'] and target_hash == checksum.hexdigest(),
                        'streamed source and independently read destination SHA match')
                record['sha256'] = target_hash
            elif row['kind'] == 'symlink':
                os.symlink(row['target'], destination)
            expected.append(record)
        for row in reversed(rows):
            if row['kind'] == 'directory':
                os.chmod(target/safe_relative(row['path']), row['mode'])
        require(hashed_tree(source) == expected, 'all source SDK bytes unchanged after copy')
        require(hashed_tree(target) == expected and bundle_identity(target) == identity,
                'complete installed SDK exact inventory and identity')
        runtime_info = configure_profile(runtime, leaf)
        inventory = {'artifact_kind': KIND, 'source': str(source), 'target': str(target), 'entries': expected}
        inventory_sha = durable_new(leaf/'inventory.json', inventory)
        files = [r for r in expected if r['kind'] == 'file']
        proof = {'artifact_kind': KIND, 'status': 'complete', 'source': str(source), 'target': str(target),
                 'build_version': VERSION, 'cxx11_abi': ABI, 'bundle_identity': identity,
                 'inventory_sha256': inventory_sha, 'regular_files': len(files),
                 'regular_file_bytes': sum(r['bytes'] for r in files),
                 'symlinks': sum(r['kind'] == 'symlink' for r in expected),
                 'directories': sum(r['kind'] == 'directory' for r in expected),
                 'installer_sha256': installer_sha, 'source_and_destination_every_file_SHA_verified': True,
                 'source_unchanged_after_copy': True, 'SDK_execution': False,
                 'runtime_profile': runtime_info,
                 'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        proof_sha = durable_new(leaf/'proof.json', proof)
        durable_new(setup/POINTER, {'artifact_kind': KIND, 'target': str(target),
                    'proof_path': str(leaf/'proof.json'), 'proof_sha256': proof_sha})
        return verify_complete(source, target, setup, compare_source=False, profile=profile)
    except BaseException as error:
        # Preserve partial SDK and all attempt evidence; never clean or replace it.
        durable_new(leaf/'failure.json', {'artifact_kind': KIND, 'status': 'failed_preserved',
                    'source': str(source), 'target': str(target), 'error': type(error).__name__ + ': ' + str(error)})
        raise


def mount_for(path):
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        mount = Path(fields[4].replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\'))
        if path == mount or path.is_relative_to(mount):
            mounts.append(mount)
    require(mounts, 'container filesystem mount identity')
    return max(mounts, key=lambda p: len(p.parts))


def container_environment():
    require(sys.platform.startswith('linux') and Path('/.dockerenv').is_file(), 'managed Linux container only')
    require(os.geteuid() == 0, 'dependency setup requires root inside the managed container')
    require(Path(__file__).resolve() == Path('/embedding/code/scripts/install-libtorch.py') and
            Path('/embedding/container.ps1').is_file(), 'authoritative project container entry only')
    release = Path('/etc/os-release').read_text()
    require(re.search(r'^ID=debian$', release, re.M) and re.search(r'^VERSION_ID="?12"?$', release, re.M),
            'approved Debian 12 container')
    require(mount_for(TARGET) == Path('/') and mount_for(SETUP) == Path('/'),
            'SDK and evidence must be in the container filesystem, not a workspace bind or other mount')


def fixture_bundle(root):
    root.mkdir()
    files = {'build-version': VERSION+'\n',
             'share/cmake/Torch/TorchConfig.cmake': 'set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")\n',
             'share/cmake/Torch/TorchConfigVersion.cmake': 'set(PACKAGE_VERSION "2.6.0")\n',
             'include/torch/csrc/api/include/torch/torch.h': '// artificial header\n'}
    for name, text in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    (root/'lib').mkdir()
    for name in ('libtorch.so','libtorch_cpu.so','libtorch_cuda.so','libc10.so','libc10_cuda.so'):
        (root/'lib'/name).write_bytes(b'\x7fELF artificial fixture '+name.encode())
    os.symlink('libtorch.so',root/'lib/internal.so')


def self_test():
    # Only this disposable fixture root is mutated. Neither real SDK path is consulted.
    require(sys.platform.startswith('linux') and Path('/.dockerenv').is_file(), 'fixture execution inside container only')
    negatives = 0
    def reject(call):
        nonlocal negatives
        try:
            call()
        except (AssertionError, FileExistsError, FileNotFoundError, RuntimeError):
            negatives += 1
        else:
            raise AssertionError('invalid artificial SDK state accepted')
    with tempfile.TemporaryDirectory(prefix='cuwacunu-libtorch-fixture-') as temporary:
        base=Path(temporary);source=base/'source';target=base/'installed';setup=base/'setup'
        fixture_bundle(source)
        profile=base/'embedding.sh';profile.write_bytes(OLD_PROFILE)
        before=hashed_tree(source)
        installed=install(source,target,setup,'a'*64,profile)
        require(installed['status']=='verified' and before==hashed_tree(source), 'verified fixture copy preserves source')
        require(profile.read_bytes()==NEW_PROFILE, 'only known generated runtime prefix migrated')
        profile_backup=Path(installed['proof_path']).parent/'embedding-profile.before.sh'
        require(profile_backup.read_bytes()==OLD_PROFILE, 'exact old runtime profile backup retained outside SDK')
        proof_path=installed['proof_path'];reused=install(source,target,setup,'b'*64)
        require(reused['proof_path']==proof_path and reused['sdk_mutations']==0, 'idempotent complete SDK reuse')
        header=source/'include/torch/csrc/api/include/torch/torch.h'
        header.write_text('// changed artificial header\n')
        reject(lambda: install(source,target,setup,'b'*64))
        header.write_text('// artificial header\n')
        (target/'lib/libtorch_cpu.so').write_bytes(b'changed')
        reject(lambda: install(source,target,setup,'b'*64))
        partial=base/'partial';partial.mkdir();(partial/'retained').write_text('preserve')
        reject(lambda: install(source,partial,base/'uncreated-setup','a'*64))
        require((partial/'retained').read_text()=='preserve' and not exists(base/'uncreated-setup'), 'partial target preserved')
        conflict=base/'conflict';conflict.write_text('preserve')
        reject(lambda: install(source,conflict,setup,'a'*64))
        require(conflict.read_text()=='preserve', 'conflicting file preserved')
        escaped=base/'escaped-source';fixture_bundle(escaped);os.symlink('../../outside',escaped/'lib/escape')
        (base/'outside').write_text('outside')
        reject(lambda: install(escaped,base/'no-escape-target',base/'escape-setup','a'*64))
        require(not exists(base/'no-escape-target'), 'escaping symlink fails before target creation')
        absolute=base/'absolute-source';fixture_bundle(absolute);os.symlink(str(absolute/'lib/libtorch.so'),absolute/'lib/absolute')
        reject(lambda: tree(absolute))
        broken=base/'broken-source';fixture_bundle(broken);os.symlink('absent',broken/'lib/broken')
        reject(lambda: tree(broken))
        wrong=base/'wrong-source';fixture_bundle(wrong);(wrong/'build-version').write_text('2.7.0\n')
        reject(lambda: install(wrong,base/'no-wrong-target',base/'wrong-setup','a'*64))
        require(not exists(base/'no-wrong-target'), 'wrong version fails before target creation')
        abi=base/'abi-source';fixture_bundle(abi);(abi/'share/cmake/Torch/TorchConfig.cmake').write_text('set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")\n')
        reject(lambda: install(abi,base/'no-abi-target',base/'abi-setup','a'*64))
        proof = Path(proof_path)
        proof.write_bytes(proof.read_bytes()+b' ')
        reject(lambda: verify_complete(source,target,setup))
        reject(lambda: parse('{"a":1,"a":2}'))
        custom=base/'custom-profile';custom.write_text('# keep custom profile\n')
        reject(lambda: install(source,base/'no-custom-target',setup,'a'*64,custom))
        require(custom.read_text()=='# keep custom profile\n' and not exists(base/'no-custom-target'),
                'unknown custom profile preserved before any SDK target creation')
        profile.write_bytes(OLD_PROFILE+b'# changed\n')
        reject(lambda: profile_plan(profile))
    return {'status':'passed','negative_cases':negatives,'real_SDK_reads':0,'real_SDK_mutations':0,
            'downloads':0,'build_or_SDK_execution':False,'fixture_only':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--check',action='store_true',help='Verify complete internal SDK and its immutable proof; never copy.')
    args=parser.parse_args()
    if args.self_test:
        require(not args.check,'fixture mode cannot verify a real SDK')
        print(json.dumps(self_test(),sort_keys=True));return
    container_environment()
    if args.check:
        result=verify_complete(SOURCE,TARGET,SETUP,profile=PROFILE)
    else:
        result=install(SOURCE,TARGET,SETUP,digest(Path(__file__).read_bytes()),PROFILE)
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
