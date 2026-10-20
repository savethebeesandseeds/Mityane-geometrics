#!/usr/bin/env bash
# Copyright (c) 2026 Waajacu. See LICENSE.
# A model-owned entry through the shared managed-container runtime.
set -euo pipefail
test -f /.dockerenv
cd /mityane
case "${1:-}" in
  build)
    shift
    if (($# == 0)); then set -- build-cpb-v15-experiment; fi
    exec bash code/scripts/task.sh cpb-v15-objective-001 "$@"
    ;;
  prepare)
    shift
    exec bash code/scripts/node.sh code/embeddings/evaluation/protocols/cpb_v15_objective_v1/prepare.mjs "$@"
    ;;
  run)
    shift
    exec bash code/scripts/node.sh code/embeddings/evaluation/protocols/cpb_v15_objective_v1/run.mjs "$@"
    ;;
  *) printf '%s\n' 'usage: cpb-v15-objective.sh build [make targets...] | prepare --inputs CARD | run --inputs CARD --phase PHASE [--seed N --arm ARM]' >&2; exit 2 ;;
esac
