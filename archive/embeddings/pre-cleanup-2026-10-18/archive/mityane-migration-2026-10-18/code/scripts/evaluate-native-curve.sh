#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv ]] || { echo 'Run evaluation inside the managed development container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'native-curve-v1 has a fixed declared recipe; no runner overrides.' >&2; exit 1; }
binary="${NATIVE_CURVE_BIN:-/opt/cuwacunu_embedding/build/rpb-native-curve/embedding_native_curve}"
run_root="${EMBEDDING_RUN_ROOT:-$PWD/output/runs/rpb-native-curve}"
mkdir -p -- "$run_root"
run_directory="$(mktemp -d "$run_root/native-curve-XXXXXX")"
printf 'Native curve capsule: %s\n' "$run_directory"
[[ -n "${NATIVE_CURVE_SOURCE_INPUTS:-}" ]] || { echo 'Run through the native-curve Make target to freeze exact source inputs.' >&2; exit 1; }
read -r -a source_inputs <<< "$NATIVE_CURVE_SOURCE_INPUTS"
mkdir -- "$run_directory/source"
cp --parents -- "${source_inputs[@]}" "$run_directory/source/"
sha256sum -- "${source_inputs[@]}" > "$run_directory/source-inputs.sha256"
source_id="$(sha256sum "$run_directory/source-inputs.sha256" | cut -d ' ' -f 1)"
[[ "$source_id" == "$("$binary" --source-id)" ]] || { echo 'Compiled source fingerprint differs from frozen source snapshot.' >&2; exit 1; }
printf '%s\n' "$source_id" > "$run_directory/source-fingerprint.txt"
# The CLI saves the launch plan, passes its actual CUDA/parity gate, then
# creates the fresh development experiment; all TEST generation follows selection.
"$binary" --output "$run_directory/results"
printf 'Native curve artifacts: %s\n' "$run_directory/results"
