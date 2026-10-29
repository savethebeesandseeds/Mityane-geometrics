#!/usr/bin/env bash
# Same checks as tests/demo_workflow.cmake. The managed image has no CMake.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: demo_workflow.sh DEMO WORK_DIR' >&2
  exit 1
fi
demo=$1
work=$2
rm -rf -- "$work"
mkdir -p -- "$work"

run_demo() {
  "$demo" "$@"
}

reject_demo() {
  if "$demo" "$@"; then
    echo "Demo accepted invalid arguments: $*" >&2
    exit 1
  fi
}

run_demo --steps 12 --seed 127 --checkpoint "$work/full.pt" --report "$work/full.json"
run_demo --steps 5 --seed 127 --save-every 2 --checkpoint "$work/split/model.pt"
run_demo --resume "$work/split/model.pt" --steps 7 \
  --checkpoint "$work/split/model.pt" --report "$work/resumed.json"
run_demo --resume "$work/split/model.pt" --evaluate-only --report "$work/evaluated.json"

python3 - "$work" <<'PY'
import json
import pathlib
import sys
work = pathlib.Path(sys.argv[1])
documents = {}
for name in ("full", "resumed", "evaluated"):
    documents[name] = json.loads((work / f"{name}.json").read_text())
    completed = documents[name]["completed_steps"]
    if completed != 12:
        raise SystemExit(f"Wrong completed step count in {name}: {completed}")
if documents["full"]["final"] != documents["resumed"]["final"] or documents["full"]["final"] != documents["evaluated"]["final"]:
    raise SystemExit("Resumed/evaluation-only reports must equal uninterrupted CPU training")
if documents["resumed"]["initial_steps"] != 5 or documents["evaluated"]["initial_steps"] != 12:
    raise SystemExit("Resume/evaluation did not preserve progress")
PY

before=$(sha256sum "$work/split/model.pt" | awk 'NR==1 { print $1 }')
reject_demo --resume "$work/split/model.pt" --seed 42 --steps 1
reject_demo --resume "$work/split/model.pt" --evaluate-only --steps 1
reject_demo --resume "$work/split/model.pt" --evaluate-only --report "$work/split/model.pt"
reject_demo --steps 1 --steps 2
after=$(sha256sum "$work/split/model.pt" | awk 'NR==1 { print $1 }')
if [[ "$before" != "$after" ]]; then
  echo 'Invalid command altered the input checkpoint' >&2
  exit 1
fi

printf '%s\n' probe > "$work/case-probe"
if [[ -e "$work/CASE-PROBE" ]]; then
  case_id=$(python3 -c 'import secrets; print(secrets.token_hex(6))')
  reject_demo --steps 1 --checkpoint "$work/case-${case_id}.pt" \
    --report "$work/CASE-${case_id}.PT"
  if [[ -e "$work/case-${case_id}.pt" ]]; then
    run_demo --resume "$work/case-${case_id}.pt" --evaluate-only
  fi
fi
echo 'Demo workflow: uninterrupted = resumed = evaluation-only; invalid requests preserve checkpoint'
