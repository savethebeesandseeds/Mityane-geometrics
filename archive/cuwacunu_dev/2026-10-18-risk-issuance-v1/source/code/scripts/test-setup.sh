#!/usr/bin/env bash
# Exercise the real setup entry point with artificial package state only.
# No APT operation, real package change, SDK read or SDK mutation is performed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && "$(pwd -P)" == /cuwacunu ]] || {
  echo 'Run this fixture in the managed cuwacunu container at /cuwacunu.' >&2; exit 1;
}
project_root="$(pwd -P)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/cuwacunu-setup-fixture.XXXXXX")"
trap 'rm -rf -- "$fixture_root"' EXIT
mkdir -p "$fixture_root/bin"

# A copied setup script runs against its own tiny lock/state in a temporary
# directory. Only its entry-point pwd check is substituted; all execution is
# still inside the real managed Linux container. Python stops at the SDK boundary.
cat > "$fixture_root/bash-env" <<'EOF'
pwd() { printf '/cuwacunu\n'; }
EOF
cat > "$fixture_root/bin/dpkg-query" <<'EOF'
#!/bin/bash
set -euo pipefail
[[ $# -eq 3 && "$1" == -W && "$2" == '-f=${Status}\t${Version}' ]] || exit 45
package="$3"
[[ -f "$SETUP_FIXTURE_CASE/state/$package" ]] || exit 1
printf 'install ok installed\t%s' "$(cat "$SETUP_FIXTURE_CASE/state/$package")"
EOF
cat > "$fixture_root/bin/apt-get" <<'EOF'
#!/bin/bash
set -euo pipefail
printf '%s\n' "$@" > "$SETUP_FIXTURE_CASE/apt-arguments"
[[ "$*" == '--no-download --no-remove install -y --no-install-recommends fixture-b=1' ]] || exit 46
case "$SETUP_FIXTURE_MODE" in
  repair) printf '1\n' > "$SETUP_FIXTURE_CASE/state/fixture-b" ;;
  drift)
    printf '1\n' > "$SETUP_FIXTURE_CASE/state/fixture-b"
    printf '2\n' > "$SETUP_FIXTURE_CASE/state/fixture-a"
    ;;
  leave_missing) : ;;
  failure) exit 42 ;;
  *) exit 47 ;;
esac
EOF
cat > "$fixture_root/bin/python3" <<'EOF'
#!/bin/bash
set -euo pipefail
[[ "$*" == '-B code/scripts/install-libtorch.py --check' ]] || exit 48
printf 'reached-sdk-boundary\n' > "$SETUP_FIXTURE_CASE/sdk-boundary"
exit 73
EOF
chmod +x "$fixture_root/bin/dpkg-query" "$fixture_root/bin/apt-get" "$fixture_root/bin/python3"

passed=0
run_case() {
  local mode="$1" expected_exit="$2" expected_message="$3" apt_expected="$4"
  local case_root="$fixture_root/$mode" status
  mkdir -p "$case_root/state"
  cp -- "$project_root/setup.sh" "$case_root/setup.sh"
  printf 'fixture-a=1\nfixture-b=1\n' > "$case_root/dependencies.lock"
  printf '1\n' > "$case_root/state/fixture-a"
  if [[ "$mode" == matched || "$mode" == mismatch ]]; then
    printf '1\n' > "$case_root/state/fixture-b"
  fi
  if [[ "$mode" == mismatch ]]; then
    printf '2\n' > "$case_root/state/fixture-a"
  elif [[ "$mode" == malformed ]]; then
    printf 'invalid lock entry\n' > "$case_root/dependencies.lock"
  fi
  if env BASH_ENV="$fixture_root/bash-env" PATH="$fixture_root/bin:$PATH" \
      SETUP_FIXTURE_CASE="$case_root" SETUP_FIXTURE_MODE="$mode" \
      bash "$case_root/setup.sh" --no-download > "$case_root/log" 2>&1; then
    status=0
  else
    status=$?
  fi
  if [[ "$status" != "$expected_exit" ]] || ! grep -Fq -- "$expected_message" "$case_root/log"; then
    cat "$case_root/log" >&2
    echo "Setup fixture '$mode' failed: exit $status, expected $expected_exit." >&2
    exit 1
  fi
  if [[ "$apt_expected" == true ]]; then
    [[ -f "$case_root/apt-arguments" ]] || { echo "Missing cached-APT witness: $mode" >&2; exit 1; }
  else
    [[ ! -e "$case_root/apt-arguments" ]] || { echo "Unexpected APT call: $mode" >&2; exit 1; }
  fi
  if [[ "$expected_exit" == 73 ]]; then
    [[ -f "$case_root/sdk-boundary" ]] || { echo "Missing SDK-boundary witness: $mode" >&2; exit 1; }
  else
    [[ ! -e "$case_root/sdk-boundary" ]] || { echo "Invalid package state reached SDK: $mode" >&2; exit 1; }
    ! grep -Fq 'Pinned package verification:' "$case_root/log" || {
      echo "Invalid package state was reported verified: $mode" >&2; exit 1;
    }
  fi
  passed=$((passed + 1))
}

run_case matched 73 '2 exact installed versions; 0 cached packages requested.' false
run_case repair 73 '2 exact installed versions; 1 cached packages requested.' true
run_case drift 1 'Installed fixture-a=2 differs from locked 1' true
run_case leave_missing 1 'fixture-b=1 is still missing after cached dependency installation' true
run_case failure 1 'Cached dependency installation failed.' true
run_case mismatch 1 'Installed fixture-a=2 differs from locked 1' false
run_case malformed 1 'Malformed dependency lock entry:' false
printf 'Setup fixture: %s cases pass; real APT operations, package changes and SDK reads/mutations: 0.\n' "$passed"
