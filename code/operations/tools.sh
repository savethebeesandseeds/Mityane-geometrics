#!/usr/bin/env bash
# Common command entry; project execution and builds remain in the managed container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && "$PWD" == /mityane ]] || {
  echo 'Use code/operations/tools.ps1 in the managed mityane-geometrics container.' >&2; exit 2;
}
case "${1:-}" in
  check|run)
    bash code/scripts/task.sh operation-tools-v1 build-operation-tools >&2
    ;;
esac
exec bash code/scripts/node.sh code/operations/tools.mjs "$@"
