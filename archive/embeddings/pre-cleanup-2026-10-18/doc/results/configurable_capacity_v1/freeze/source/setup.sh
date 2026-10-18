#!/usr/bin/env bash
# Dependency installation adapted from cuwacunu_torch/setup.sh:
# C++ build/debug tools, NVIDIA Debian 12 CUDA 12.4 and cuDNN 9 packages.
# Build/run and container lifecycle stay in Makefile and container.ps1.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /etc/os-release
[[ "$ID" == debian && "$VERSION_ID" == 12 ]] || {
  echo 'setup.sh requires Debian 12.' >&2; exit 1;
}
[[ $EUID -eq 0 ]] || { echo 'Run setup.sh as root inside the container.' >&2; exit 1; }
[[ $# -eq 0 ]] || { echo 'Usage: bash setup.sh (dependency installation only)' >&2; exit 1; }
[[ -f /.dockerenv ]] || { echo 'Run setup.sh in the managed container.' >&2; exit 1; }
torch_root=/opt/cuwacunu_embedding/libtorch
export DEBIAN_FRONTEND=noninteractive
apt=(apt-get -o Acquire::Retries=3 -o Acquire::https::Timeout=30 -o Acquire::http::Timeout=30)
setup_state=/opt/cuwacunu_embedding/setup
mkdir -p "$setup_state"
"${apt[@]}" update
mapfile -t bootstrap < <(grep -E '^(ca-certificates|curl)=' dependencies.lock)
"${apt[@]}" install -y --no-install-recommends "${bootstrap[@]}"

# Use the same keyring and repository as the original project's installer.
keyring="$setup_state/cuda-keyring_1.1-1_all.deb"
if [[ "$(dpkg-query -W -f='${Version}' cuda-keyring 2>/dev/null || true)" != '1.1-1' ]]; then
  curl -fsSL -o "$keyring" https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
  printf '%s  %s\n' e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d "$keyring" | sha256sum -c -
  dpkg -i "$keyring"
fi
"${apt[@]}" update
# Exact versions captured from the approved Debian 12 environment. Fail if a
# repository no longer supplies them; never silently substitute other versions.
mapfile -t packages < dependencies.lock
"${apt[@]}" install -y --no-install-recommends "${packages[@]}"

# Copy the existing pinned Linux input into the container filesystem. The
# installer verifies every source/destination file and preserves any conflict.
python3 -B code/scripts/install-libtorch.py

sed -i 's/^[[:space:]]*#[[:space:]]*en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
locale-gen en_US.UTF-8
update-locale --reset LANG=en_US.UTF-8

# Keep all bundled cuDNN components together. The system library directory must
# not precede LibTorch's own runtime, and CUDA's stubs are never runtime paths.
cat > /etc/profile.d/embedding.sh <<'EOF'
export CUDA_VERSION=12.4
export CUDNN_VERSION=9
case ":$PATH:" in
  *:/usr/local/cuda-12.4/bin:*) ;;
  *) export PATH="/usr/local/cuda-12.4/bin:$PATH" ;;
esac
case "${LD_LIBRARY_PATH:-}" in
  /opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64*) ;;
  *) export LD_LIBRARY_PATH="/opt/cuwacunu_embedding/libtorch/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
EOF
profile_line='source /etc/profile.d/embedding.sh'
grep -Fqx "$profile_line" /root/.bashrc || printf '\n%s\n' "$profile_line" >> /root/.bashrc
source /etc/profile.d/embedding.sh
dpkg-query -W -f='${Package}=${Version}\n' > "$setup_state/debian-packages.txt"
for library in libtorch.so libtorch_cpu.so libtorch_cuda.so; do
  dependencies="$(ldd "$torch_root/lib/$library")"
  if grep -q 'not found' <<< "$dependencies"; then
    printf '%s\n' "$dependencies" >&2
    echo 'Missing LibTorch runtime dependency; environment preserved for inspection.' >&2
    exit 1
  fi
done
nvcc --version
dpkg-query -W cuda-toolkit-12-4 cudnn9-cuda-12
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf 'Dependencies ready: %s; %s\n' "$(g++ -dumpfullversion)" "$(cat "$torch_root/build-version")"
