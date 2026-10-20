#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
[[ -f /.dockerenv && $# -eq 0 ]] || { echo 'Run frozen measurement inside the managed container.' >&2; exit 1; }
binary="${V7_DECODER_CALIBRATION_BIN:-/opt/mityane-geometrics/embeddings/build/rpb-paired-pooling/embedding_v7_decoder_calibration}"
root="$PWD/.temp/embeddings/runs/rpb-v7-decoder-calibration"
[[ -n "${V7_DECODER_CALIBRATION_SOURCE_INPUTS:-}" ]] || { echo 'Use the Make target to bind sources.' >&2; exit 1; }
mkdir -p -- "$root"
capsule="$(mktemp -d "$root/decoder-calibration-XXXXXX")"
printf 'Decoder calibration capsule: %s\n' "$capsule"
read -r -a sources <<< "$V7_DECODER_CALIBRATION_SOURCE_INPUTS"
mkdir -- "$capsule/source"
cp --parents -- "${sources[@]}" "$capsule/source/"
sha256sum -- "${sources[@]}" > "$capsule/source-inputs.sha256"
source_id="$(sha256sum "$capsule/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]]
printf '%s\n' "$source_id" > "$capsule/source-fingerprint.txt"
[[ -f "$root/admission-approved.path" ]] || { echo 'Complete actual decoder CUDA admission first.' >&2; exit 1; }
admission="$(cat "$root/admission-approved.path")"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs verify-admission v7-decoder-calibration-v1 "$admission" "$root/admission" "$source_id"
cp -a -- "$admission" "$capsule/admission"
"$binary" --parent-roles > "$capsule/parent-role-plan.json"
bash code/scripts/node.sh code/embeddings/scripts/prepare-v7-decoder-inputs.mjs --repo-root "$PWD" --capsule "$capsule" --historical-parent-root "${MITYANE_HISTORICAL_PARENT_ROOT:?Set the explicit retained parent root}"
"$binary" --output "$capsule/results" --parent-input-manifest "$capsule/parent-inputs.sha256" \
  --admission-log "$capsule/admission/build-and-tests.log" \
  --admission-sha256 "$(sha256sum "$capsule/admission/build-and-tests.log" | cut -d ' ' -f 1)" \
  2>&1 | tee "$capsule/command.log"
bash code/scripts/node.sh code/embeddings/scripts/prepare-v7-decoder-inputs.mjs --repo-root "$PWD" --capsule "$capsule" --historical-parent-root "${MITYANE_HISTORICAL_PARENT_ROOT:?Set the explicit retained parent root}" --verify > "$capsule/inputs-preserved-after.txt"
sha256sum -c "$capsule/source-inputs.sha256" > "$capsule/source-preserved-after.txt"
(cd -- "$capsule/source"; sha256sum -c "$capsule/source-inputs.sha256") > "$capsule/copied-source-preserved-after.txt"
bash code/scripts/node.sh code/embeddings/scripts/runner-metadata.mjs inventory v7-decoder-calibration-v1 "$capsule"
printf 'Decoder calibration complete: %s\n' "$capsule"
