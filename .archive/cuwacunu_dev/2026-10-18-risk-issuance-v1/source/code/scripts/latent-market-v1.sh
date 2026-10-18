#!/usr/bin/env bash
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Use the managed cuwacunu container.' >&2; exit 1; }
source /etc/profile.d/embedding.sh
[[ $# -ge 1 ]] || { echo 'Usage: latent-market-v1.sh check|source-pins|prepare|study|utility|fixture [arguments]' >&2; exit 1; }
operation="$1"; shift
session=latent-market-v1
build="/opt/cuwacunu/build/$session"
case "$operation" in
  check)
    exec bash code/scripts/task.sh "$session" -j2 build-latent-market-v1 test-latent-market-v1
    ;;
  prepare|study|utility)
    [[ $# == 2 || ( "$operation" == study && $# == 4 ) ]] || { echo 'Expected SETTINGS NEW_OUTPUT_LEAF [TRAINING_CHECKPOINT SIDECAR_SHA256 for study only]' >&2; exit 1; }
    [[ "$operation" == study || $# == 2 ]] || exit 1
    bash code/scripts/task.sh "$session" -j2 build-latent-market-v1
    exec 9>"$build/.task.lock"
    flock -n 9 || { echo 'Latent market session is busy.' >&2; exit 1; }
    export TMPDIR="$build/tmp"
    exec "$build/latent_market_v1_$operation" "$@"
    ;;
  fixture)
    [[ $# == 1 ]] || { echo 'Expected NEW_FIXTURE_LEAF' >&2; exit 1; }
    bash code/scripts/task.sh "$session" build-latent-market-v1-fixture
    exec "$build/latent_market_v1_pipeline_fixture" "$@"
    ;;
  source-pins)
    [[ $# == 1 ]] || { echo 'Expected NEW_SOURCE_MANIFEST_FILE' >&2; exit 1; }
    bash code/scripts/task.sh "$session" -j2 build-latent-market-v1
    python3 - "$build" "$1" <<'PY'
import hashlib, os, pathlib, shlex, sys, tempfile
build, destination = map(pathlib.Path, sys.argv[1:])
root = pathlib.Path.cwd()
members = {root/'Makefile', root/'code/scripts/task.sh', root/'code/scripts/latent-market-v1.sh'}
members.update(root/p for p in ['code/latent_market/versions/v1/v1.mk', 'code/networks/predictive_training/versions/v1/v1.mk', 'code/evaluation/latent_dynamics/versions/v1/baselines.mk'])
for name in ['latent_market_v1_prepare.d', 'latent_market_v1_study.d', 'latent_market_v1_utility.d', 'record_types.d', 'latent_dynamics_v1_baselines.d', 'latent_dynamics_v1_risk_readouts.d']:
    first = (build/name).read_text().replace('\\\n', '').splitlines()[0]
    for raw in shlex.split(first.split(':', 1)[1]):
        path = pathlib.Path(raw).resolve()
        if path.is_relative_to(root): members.add(path)
destination.parent.mkdir(parents=True, exist_ok=True)
fd, staging = tempfile.mkstemp(prefix='.latent-source-', dir=destination.parent)
try:
    with os.fdopen(fd, 'w', newline='\n') as stream:
        for path in sorted(members):
            stream.write(hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()+'  '+str(path)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.link(staging, destination)
finally:
    os.unlink(staging)
print(f'Pinned {len(members)} source/build files in {destination}')
PY
    ;;
  *) echo "Unknown latent market operation: $operation" >&2; exit 1;;
esac
