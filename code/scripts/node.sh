#!/usr/bin/env bash
# Execute maintained JavaScript tooling with the existing pinned Linux runtime.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && $(uname -s) == Linux && $(pwd -P) == /mityane ]] || {
  echo 'Run Node tooling through container.ps1 in mityane-geometrics.' >&2; exit 1;
}
source code/scripts/node-runtime.sh
[[ -x $MITYANE_NODE_HOME/bin/node && -f $MITYANE_NODE_HOME/LICENSE ]] || {
  echo 'Missing pinned Node runtime; run bash setup.sh --no-download in the managed container.' >&2; exit 1;
}
[[ $("$MITYANE_NODE_HOME/bin/node" --version) == "$MITYANE_NODE_VERSION" ]] || {
  echo 'Existing Node version differs from the pinned runtime; preserved for inspection.' >&2; exit 1;
}
exec "$MITYANE_NODE_HOME/bin/node" "$@"
