#!/usr/bin/env bash
# Dependencies only: verify/reuse the exact cached Node binary and license.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ $# == 0 && -f /.dockerenv && $(uname -s) == Linux && $(pwd -P) == /mityane ]] || {
  echo 'Run dependency setup through container.ps1 in mityane-geometrics.' >&2; exit 1;
}
source code/scripts/node-runtime.sh
setup_state=/opt/mityane-geometrics/setup
mkdir -p "$setup_state"
exec 9>"$setup_state/node-runtime.lock"
flock -n 9 || { echo 'Another pinned Node dependency check is running.' >&2; exit 1; }
[[ -f $MITYANE_NODE_ARCHIVE ]] || {
  echo "Missing cached Linux Node archive: $MITYANE_NODE_ARCHIVE" >&2
  echo 'Restore the exact archive described in doc/ENVIRONMENT.md; no download or host fallback is performed.' >&2
  exit 1
}
printf '%s  %s\n' "$MITYANE_NODE_SHA256" "$MITYANE_NODE_ARCHIVE" | sha256sum --check --status
if [[ ! -e $MITYANE_NODE_HOME && ! -L $MITYANE_NODE_HOME ]]; then
  mkdir -p "$MITYANE_NODE_HOME"
  tar -xJf "$MITYANE_NODE_ARCHIVE" --strip-components=1 -C "$MITYANE_NODE_HOME" \
    node-v22.23.3-linux-x64/bin/node node-v22.23.3-linux-x64/LICENSE
fi
[[ -d $MITYANE_NODE_HOME && ! -L $MITYANE_NODE_HOME && -x $MITYANE_NODE_HOME/bin/node && -f $MITYANE_NODE_HOME/LICENSE ]] || {
  echo 'Existing Node runtime is incomplete or indirect; preserved for inspection.' >&2; exit 1;
}
for member in bin/node LICENSE; do
  cmp -s <(tar -xOJf "$MITYANE_NODE_ARCHIVE" "node-v22.23.3-linux-x64/$member") "$MITYANE_NODE_HOME/$member" || {
    echo "Existing Node $member differs from the pinned archive; preserved for inspection." >&2; exit 1;
  }
done
[[ $("$MITYANE_NODE_HOME/bin/node" --version) == "$MITYANE_NODE_VERSION" ]] || {
  echo 'Unexpected Node version; preserved for inspection.' >&2; exit 1;
}
printf 'Verified cached Linux Node %s at %s; no downloads or system package changes.\n' "$MITYANE_NODE_VERSION" "$MITYANE_NODE_HOME"
