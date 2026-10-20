#!/usr/bin/env bash
# Build/admit new infrastructure, then generate/freeze the complete fresh data.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
[[ -f /.dockerenv && "$PWD" == /mityane ]] || exit 1
protocol_root=code/embeddings/evaluation/protocols/fixed_prior_fresh_confirmation_v1
session=rpb-fixed-prior-fresh-confirmation
run_root=/mityane/.temp/embeddings/runs/$session
build_root=/opt/mityane-geometrics/embeddings/build/$session
mkdir -p "$run_root"
admission=$(mktemp -d "$run_root/admission-XXXXXX")
printf '%s\n' "$admission"
bash code/scripts/task.sh "$session" -f "$protocol_root/Makefile" \
  print-fixed-prior-fresh-confirmation-sources >"$admission/source-list.log"
sed -n '/^Makefile$/,$p' "$admission/source-list.log" >"$admission/source-list.txt"
bash code/scripts/node.sh "$protocol_root/prepare_confirmation.mjs" \
  --source-list "$admission/source-list.txt" --output "$admission/source-freeze" >"$admission/source-freeze.log"
sha256sum -c "$admission/source-freeze/sources.sha256" >"$admission/source-check-before.log"
bash code/scripts/task.sh "$session" -f "$protocol_root/Makefile" -j2 \
  fixed-prior-fresh-confirmation fixed-prior-fresh-data >"$admission/build.log" 2>&1
exec 9>"$build_root/.task.lock"
flock -n 9 || { printf '%s\n' 'The named session is busy.' >&2; exit 1; }
"$build_root/embedding_write_fresh_tempo3_data" --engineering "$admission/data-engineering" \
  >"$admission/engineering.log" 2>&1
# Model primitive is unchanged and already admitted. This new protocol verifies
# actual CUDA training/load plus its new typed fresh-source loader.
"$build_root/embedding_fixed_prior_fresh_confirmation" --engineering "$admission/CUDA-engineering" \
  >>"$admission/engineering.log" 2>&1
card_sha=$(sha256sum "$admission/source-freeze/card.md" | cut -d ' ' -f1)
"$build_root/embedding_write_fresh_tempo3_data" \
  --output "$admission/fresh-data" --card "$admission/source-freeze/card.md" \
  --card-sha256 "$card_sha" --masters 80787,81888,82989,84090,85191 \
  >"$admission/data-generation.log" 2>&1
sha256sum -c "$admission/source-freeze/sources.sha256" >"$admission/source-check-after.log"
bash code/scripts/node.sh "$protocol_root/prepare_confirmation.mjs" \
  --source-list "$admission/source-list.txt" --input-root "$admission/fresh-data" \
  --output "$admission/freeze" >"$admission/freeze.log"
cmp "$admission/source-freeze/sources.sha256" "$admission/freeze/sources.sha256"
printf 'ADMISSION_PASSED %s\n' "$admission"
