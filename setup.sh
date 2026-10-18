#!/usr/bin/env bash
# Dependency-only setup for the local snapshot of the verified embedding
# environment. Container lifecycle and project operations stay elsewhere.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ $# -eq 0 || ( $# -eq 1 && "${1:-}" == --no-download ) ]] || {
  echo 'Usage: bash setup.sh [--no-download] (dependency verification and cached missing-package installation only)' >&2; exit 1;
}
[[ -f /.dockerenv && "$(pwd -P)" == /mityane ]] || {
  echo 'Run setup.sh in the managed mityane-geometrics container at /mityane.' >&2; exit 1;
}
source /etc/os-release
[[ "$ID" == debian && "$VERSION_ID" == 12 ]] || {
  echo 'setup.sh requires the copied Debian 12 environment.' >&2; exit 1;
}
[[ $EUID -eq 0 ]] || { echo 'Run setup.sh as root inside the container.' >&2; exit 1; }

torch_root=/opt/cuwacunu_embedding/libtorch
setup_state=/opt/mityane-geometrics/setup
export DEBIAN_FRONTEND=noninteractive
missing=()
matched=0
verify_locked_packages() {
  local allow_missing="$1" entry package required_version installed actual_version
  missing=()
  matched=0
  while IFS= read -r entry || [[ -n "$entry" ]]; do
    [[ -n "$entry" && "$entry" != \#* ]] || continue
    [[ "$entry" =~ ^([a-z0-9][a-z0-9+.-]*)=(.+)$ ]] || {
      echo "Malformed dependency lock entry: $entry" >&2; return 1;
    }
    package="${BASH_REMATCH[1]}"
    required_version="${BASH_REMATCH[2]}"
    installed="$(dpkg-query -W -f='${Status}\t${Version}' "$package" 2>/dev/null || true)"
    if [[ "$installed" == $'install ok installed\t'* ]]; then
      actual_version="${installed#*$'\t'}"
      [[ "$actual_version" == "$required_version" ]] || {
        echo "Installed $package=$actual_version differs from locked $required_version; preserved. Restore the recorded snapshot or authorize an explicit dependency repair." >&2
        return 1
      }
      matched=$((matched + 1))
    elif [[ "$allow_missing" == true ]]; then
      missing+=("$entry")
    else
      echo "Locked package $entry is still missing after cached dependency installation; preserved for inspection." >&2
      return 1
    fi
  done < dependencies.lock
}
verify_locked_packages true

# A matching copied image takes this route with no APT update, installation or
# download. Missing packages can use an already cached exact package only.
cached_requested=${#missing[@]}
if (( cached_requested )); then
  printf 'Missing pinned packages (cached installation only):\n'
  printf '  %s\n' "${missing[@]}"
  apt-get --no-download --no-remove install -y --no-install-recommends "${missing[@]}" || {
    echo 'Cached dependency installation failed. No download was requested; preserve and inspect this environment, or restore the recorded snapshot.' >&2
    exit 1
  }
  # APT success alone is not proof that every locked package still matches.
  # Reject missing packages or version drift before reporting a ready SDK.
  verify_locked_packages false
fi
printf 'Pinned package verification: %s exact installed versions; %s cached packages requested.\n' "$matched" "$cached_requested"

# Reuse the copied SDK and its complete immutable proof/inventory. This never
# reads the old sibling workspace, downloads a bundle or copies a second SDK.
python3 -B code/scripts/install-libtorch.py --check
source /etc/profile.d/embedding.sh
[[ "${LANG:-}" == en_US.UTF-8 ]] || export LANG=en_US.UTF-8
mkdir -p "$setup_state"
dpkg-query -W -f='${Package}=${Version}\n' > "$setup_state/debian-packages.txt"
for library in libtorch.so libtorch_cpu.so libtorch_cuda.so; do
  dependencies="$(ldd "$torch_root/lib/$library")"
  if grep -q 'not found' <<< "$dependencies"; then
    printf '%s\n' "$dependencies" >&2
    echo 'Missing LibTorch runtime dependency; copied environment preserved for inspection.' >&2
    exit 1
  fi
done
nvcc --version
dpkg-query -W cuda-toolkit-12-4 cudnn9-cuda-12
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf 'Dependencies ready: GCC %s; LibTorch %s; SDK reused without mutation.\n' "$(g++ -dumpfullversion)" "$(cat "$torch_root/build-version")"
