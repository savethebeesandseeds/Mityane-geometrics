#!/usr/bin/env bash
# Copyright (c) 2026 Waajacu. See LICENSE.
set -euo pipefail
test -f /.dockerenv
cd /mityane
case "${1:-}" in
  build)
    shift
    if (($# == 0)); then set -- build-cpb-v16-experiment; fi
    exec bash code/scripts/task.sh cpb-v16-temporal-001 "$@"
    ;;
  prepare|run|report)
    entry=$1; shift
    exec bash code/scripts/node.sh "code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/$entry.mjs" "$@"
    ;;
  *) printf '%s\n' 'usage: cpb-v16-temporal.sh build [targets...] | prepare|run|report --inputs CARD [phase flags...]'; exit 2 ;;
esac
