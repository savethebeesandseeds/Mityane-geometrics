#!/usr/bin/env bash
# Dependencies only. No downloads, APT changes, builds, tests or services.
set -euo pipefail
[[ $# == 0 && -f /.dockerenv && $(uname -s) == Linux && -d /mityane/web/dashboard ]] || {
  echo 'Run this dependency setup through container.ps1 in mityane-geometrics.' >&2; exit 1;
}
runtime=/opt/mityane-geometrics/web/tools/node-v22.23.3-linux-x64
archive=/opt/mityane-geometrics/web/downloads/node-v22.23.3-linux-x64.tar.xz
digest=df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de
[[ -f $archive ]] || {
  echo "Missing cached Linux Node 22.23.3 archive: $archive" >&2
  echo 'Restore the exact archive described in web/dashboard/README.md; no download or host fallback is performed.' >&2
  exit 1
}
printf '%s  %s\n' "$digest" "$archive" | sha256sum --check --status
if [[ ! -e $runtime ]]; then
  mkdir -p "$runtime"
  tar -xJf "$archive" --strip-components=1 -C "$runtime" \
    node-v22.23.3-linux-x64/bin/node node-v22.23.3-linux-x64/LICENSE
fi
[[ -x $runtime/bin/node && -f $runtime/LICENSE ]] || { echo 'Existing runtime is incomplete; preserved for inspection.' >&2; exit 1; }
cmp -s <(tar -xOJf "$archive" node-v22.23.3-linux-x64/bin/node) "$runtime/bin/node" || {
  echo 'Existing Node binary differs from the pinned archive; preserved.' >&2; exit 1;
}
[[ $("$runtime/bin/node" --version) == v22.23.3 ]] || { echo 'Unexpected Node version; preserved.' >&2; exit 1; }
printf 'Verified cached Linux Node v22.23.3 at %s; no downloads or system package changes.\n' "$runtime"
