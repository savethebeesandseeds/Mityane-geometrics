#!/usr/bin/env bash
# One frozen challenge matrix and all nine predetermined CUDA trajectories.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
[[ -f /.dockerenv && "$PWD" == /mityane && $# == 1 ]] || exit 1
capsule=$1
[[ "$capsule" == /mityane/.temp/embeddings/runs/rpb-v20-capacity/admission-* ]]
[[ $(realpath -e -- "$capsule") == "$capsule" && ! -L "$capsule" ]]
[[ -f "$capsule/admission.json" && ! -e "$capsule/quality-data" && ! -e "$capsule/profiles" ]]
protocol=code/embeddings/evaluation/protocols/configurable_capacity_v1
build=/opt/mityane-geometrics/embeddings/build/rpb-v20-capacity
exec 9>"$build/.task.lock"
flock -n 9 || { printf '%s\n' 'Capacity build session is busy.' >&2; exit 1; }
bash code/scripts/node.sh "$protocol/capsule.mjs" verify-admission "$capsule" >"$capsule/admission-before-quality.log"
sha256sum -c "$capsule/freeze/sources.sha256" >"$capsule/source-before-quality.log"
sha256sum -c "$capsule/freeze/binaries.sha256" >"$capsule/binary-before-quality.log"
source_id=$(sha256sum "$capsule/freeze/sources.sha256" | cut -d ' ' -f1)
card_sha=$(sha256sum "$capsule/freeze/card.md" | cut -d ' ' -f1)
"$build/embedding_configurable_capacity_data" --quality "$capsule/quality-data" >"$capsule/data-quality.log" 2>&1
python3 -B "$protocol/capsule.py" bind "$capsule" >"$capsule/input-freeze.log"
mkdir -- "$capsule/profiles"
for profile in A B C; do
  mkdir -- "$capsule/profiles/$profile"
  for master in 941101 941102 941103; do
    printf 'CUDA_TRAINING profile=%s master=%s\n' "$profile" "$master"
    "$build/embedding_configurable_capacity" --data-root "$capsule/quality-data" \
      --inputs-sha256 "$capsule/freeze/inputs.sha256" --output "$capsule/profiles/$profile/master-$master" \
      --profile "$profile" --master "$master" --config "$capsule/freeze/config.conf" \
      --card "$capsule/freeze/card.md" --card-sha256 "$card_sha" --source-id "$source_id" \
      >"$capsule/profile-$profile-master-$master.log" 2>&1
  done
done
"$build/embedding_configurable_capacity_readout" --data-root "$capsule/quality-data" \
  --producer-root "$capsule" --output "$capsule/readouts" \
  --inputs-sha256 "$capsule/freeze/inputs.sha256" --sources-sha256 "$capsule/freeze/sources.sha256" \
  --card-sha256 "$card_sha" >"$capsule/readouts.log" 2>&1
python3 -B "$protocol/publish.py" --capsule "$capsule" >"$capsule/publish.log" 2>&1
sha256sum -c "$capsule/freeze/sources.sha256" >"$capsule/source-after-quality.log"
sha256sum -c "$capsule/freeze/binaries.sha256" >"$capsule/binary-after-quality.log"
(cd -- "$capsule/quality-data"; sha256sum -c "$capsule/freeze/inputs.sha256") >"$capsule/inputs-after-quality.log"
sha256sum -c "$capsule/freeze/information.sha256" >"$capsule/information-after-quality.log"
bash code/scripts/node.sh "$protocol/capsule.mjs" finish "$capsule"
