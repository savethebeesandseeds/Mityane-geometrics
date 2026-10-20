#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run frozen admission inside the managed container.' >&2; exit 1; }
root="$PWD/.temp/embeddings/runs/rpb-v7-decoder-calibration"
session='rpb-paired-pooling'
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs check-card v7-decoder-calibration-v1 "$PWD"
mkdir -p -- "$root/admission"
admission="$(mktemp -d "$root/admission/admission-XXXXXX")"
printf 'Decoder admission: %s\n' "$admission"
mapfile -t sources < <(make -s print-v7-decoder-calibration-sources)
[[ ${#sources[@]} -gt 0 ]]
mkdir -- "$admission/source"
cp --parents -- "${sources[@]}" "$admission/source/"
sha256sum -- "${sources[@]}" > "$admission/source-inputs.sha256"
source_id="$(sha256sum "$admission/source-inputs.sha256" | cut -d ' ' -f 1)"
printf '%s\n' "$source_id" > "$admission/source-fingerprint.txt"
bash code/scripts/task.sh "$session" -j2 v7-decoder-calibration test-frozen-role-guard test-rpb-decoder-calibration \
  2>&1 | tee "$admission/build-and-tests.log"
binary="/opt/mityane-geometrics/embeddings/build/$session/embedding_v7_decoder_calibration"
[[ "$("$binary" --source-id)" == "$source_id" ]]
sha256sum -c "$admission/source-inputs.sha256" > "$admission/source-preserved-after.txt"
(cd -- "$admission/source"; sha256sum -c "$admission/source-inputs.sha256") > "$admission/copied-source-preserved-after.txt"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs pass-admission v7-decoder-calibration-v1 "$admission" "$source_id"
printf '%s\n' "$admission" > "$root/admission-approved.path"
printf 'Decoder admission passed: %s\n' "$admission"
