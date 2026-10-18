#!/usr/bin/env bash
# Future engineering admission only; no quality generation or source mutation.
set -euo pipefail

# Keep these five compile records inside the unchanged checker's own log even
# when the named build already contains every object. Argument boundaries are
# retained; -W changes Make's scheduling, not any source file timestamp.
make() {
  command make \
    -W code/encoders/raw_patch_bottleneck_mae/src/workflow.cpp \
    -W code/encoders/raw_patch_bottleneck_mae/src/learning_curve_adapter.cpp \
    -W code/encoders/raw_patch_bottleneck_mae/src/early_mixer_adapter.cpp \
    -W code/encoders/raw_patch_bottleneck_mae/src/early_mixer_confirmation_adapter.cpp \
    -W code/evaluation/src/early_mixer_confirmation_main.cpp \
    "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  [[ -f /.dockerenv && $# -eq 0 ]] || {
    echo 'Use this future admission wrapper without arguments inside the managed container.' >&2
    exit 1
  }
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
  export -f make
  exec bash code/scripts/check-early-mixer-confirmation.sh
fi
