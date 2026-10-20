#!/usr/bin/env bash
# Freeze/build/admit one separate design in the existing managed container.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
[[ -f /.dockerenv && "$PWD" == /mityane ]] || exit 1
run_root=/mityane/.temp/embeddings/runs/rpb-fixed-prior-relation-screen
mkdir -p "$run_root"
admission=$(mktemp -d "$run_root/admission-XXXXXX")
printf '%s\n' "$admission"
bash code/scripts/task.sh rpb-fixed-prior-relation-screen \
  -f code/embeddings/evaluation/protocols/fixed_prior_relation_screen_v1/Makefile \
  print-fixed-prior-relation-screen-sources >"$admission/source-list.log"
sed -n '/^Makefile$/,$p' "$admission/source-list.log" >"$admission/source-list.txt"
bash code/scripts/node.sh code/embeddings/evaluation/protocols/fixed_prior_relation_screen_v1/prepare_screen.mjs \
  --source-list "$admission/source-list.txt" \
  --input-root /mityane/code/embeddings/output/runs/rpb-structured-hard-timing/structured-hard-timing-xrZMAS \
  --masters 75272,76373 --output "$admission/freeze" >"$admission/freeze.log"
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-check-before.log"
bash code/scripts/task.sh rpb-fixed-prior-relation-screen \
  -f code/embeddings/evaluation/protocols/fixed_prior_relation_screen_v1/Makefile -j2 \
  fixed-prior-relation-screen \
  /opt/mityane-geometrics/embeddings/build/rpb-fixed-prior-relation-screen/code/protocols/fixed_prior_relation_screen_v1/fixed_prior_temporal_relation_test \
  >"$admission/build-tests.log" 2>&1
# Parallel CPU compilation above; CUDA admission programs execute sequentially.
/opt/mityane-geometrics/embeddings/build/rpb-fixed-prior-relation-screen/code/protocols/fixed_prior_relation_screen_v1/fixed_prior_temporal_relation_test \
  >>"$admission/build-tests.log" 2>&1
/opt/mityane-geometrics/embeddings/build/rpb-fixed-prior-relation-screen/embedding_fixed_prior_relation_screen \
  --engineering "$admission/engineering" >>"$admission/build-tests.log" 2>&1
sha256sum -c "$admission/freeze/sources.sha256" >"$admission/source-check-after.log"
printf 'ADMISSION_PASSED %s\n' "$admission"
