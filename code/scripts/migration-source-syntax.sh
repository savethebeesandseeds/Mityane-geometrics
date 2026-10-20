#!/usr/bin/env bash
# Parse maintained Bash/Node source without importing modules, producing bytecode, or
# executing research, setup or runner commands. Always enter through root make.
set -euo pipefail
[[ -f /.dockerenv && "$PWD" == /mityane ]] || {
  echo 'Use container.ps1 and the root migration syntax target.' >&2; exit 1;
}
shell_count=0
node_count=0
while IFS= read -r -d '' source; do
  case "$source" in
    *.sh) bash -n -- "$source"; shell_count=$((shell_count + 1));;
    *.mjs) bash code/scripts/node.sh --check -- "$source"; node_count=$((node_count + 1));;
  esac
done < <(
  printf '%s\0' setup.sh
  find code web -type d \( -name node_modules -o -name .build -o -name .external \
    -o -name output -o -name public -o -name dist \) -prune -o \
    -type f \( -name '*.sh' -o -name '*.mjs' \) -print0
)
printf 'Syntax only: %s Bash and %s Node source files passed.\n' \
  "$shell_count" "$node_count"
