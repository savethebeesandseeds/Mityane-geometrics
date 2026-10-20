# Bounded language-migration fixtures and optional native candidates.
# These opt-in engineering targets never resume paused research runs.
PYTHON_MIGRATION_BUILD := $(if $(BUILD_DIR),$(BUILD_DIR)/python-migration,/opt/mityane-geometrics/migration/candidates)
PYTHON_MIGRATION_TMP := $(PYTHON_MIGRATION_BUILD)/tmp
SAVED_TENSOR_V1_BUILD ?= $(PYTHON_MIGRATION_BUILD)/saved-tensor-v1
INDEPENDENT_CPU_V1_BUILD ?= $(PYTHON_MIGRATION_BUILD)/independent-cpu-v1

include code/shared/artifacts/saved_tensor/versions/v1/v1.mk
include code/embeddings/evaluation/independent_cpu/versions/v1/v1.mk

.PHONY: test-python-migration-node test-python-migration-candidates test-language-migration test-historical-parent-paths test-migration-source-syntax test-migration-protocol-cli test-migration-source-closures test-migration-reader-source
test-migration-source-closures:
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || exit 1
	@set -eu; task_tmp=$$(realpath -m -- "$(PYTHON_MIGRATION_TMP)"); \
	case "$$task_tmp" in /opt/mityane-geometrics/*) ;; *) exit 1;; esac; \
	mkdir -p -- "$$task_tmp"; test "$$(realpath -e -- "$$task_tmp")" = "$$task_tmp"; \
	TMPDIR="$$task_tmp" bash code/scripts/node.sh --test code/scripts/tests/migration-source-closure.test.mjs

test-migration-reader-source:
	@bash code/scripts/migration-reader-source-fixtures.sh "$(PYTHON_MIGRATION_TMP)"

test-migration-source-syntax:
	@bash code/scripts/migration-source-syntax.sh

test-migration-protocol-cli:
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }
	@set -eu; task_tmp=$$(realpath -m -- "$(PYTHON_MIGRATION_TMP)"); \
	case "$$task_tmp" in /opt/mityane-geometrics/*) ;; *) exit 1;; esac; \
	mkdir -p -- "$$task_tmp"; test "$$(realpath -e -- "$$task_tmp")" = "$$task_tmp"; \
	TMPDIR="$$task_tmp" bash code/scripts/node.sh --test code/embeddings/evaluation/protocols/node/v1/cli_integration_test.mjs

test-python-migration-node:
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }
	@set -eu; task_tmp=$$(realpath -m -- "$(PYTHON_MIGRATION_TMP)"); \
	case "$$task_tmp" in /opt/mityane-geometrics/*) ;; *) echo 'Migration fixture directory must stay under /opt/mityane-geometrics.' >&2; exit 1;; esac; \
	mkdir -p -- "$$task_tmp"; test "$$(realpath -e -- "$$task_tmp")" = "$$task_tmp"; \
	TMPDIR="$$task_tmp" bash code/scripts/node.sh --test \
	  code/shared/artifacts/node/v1/tests/*.test.mjs \
	  code/embeddings/scripts/tests/*.test.mjs \
	  code/embeddings/tests/preparation-migration.test.mjs \
	  code/embeddings/evaluation/protocols/node/v1/*_test.mjs \
	  code/embeddings/evaluation/protocols/configurable_capacity_v1/capsule_test.mjs \
	  code/embeddings/evaluation/protocols/configurable_learned_relation_v1/gate_test.mjs

test-python-migration-candidates: test-python-migration-node check-saved-tensor-v1 check-independent-cpu-v1

# Complete language-boundary regression: metadata ports plus retained scientific
# artificial fixtures. This target never invokes a research runner or saved run.
test-language-migration: test-migration-source-syntax test-python-migration-node test-historical-parent-paths test-migration-source-closures test-migration-reader-source
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }
	@set -eu; task_tmp=$$(realpath -m -- "$(PYTHON_MIGRATION_TMP)"); \
	case "$$task_tmp" in /opt/mityane-geometrics/*) ;; *) exit 1;; esac; \
	mkdir -p -- "$$task_tmp"; export TMPDIR="$$task_tmp"; \
	for fixture in \
	  code/embeddings/evaluation/protocols/configurable_branch_v1/audit_test.py \
	  code/embeddings/evaluation/protocols/configurable_capacity_v1/evaluation_audit_test.py \
	  code/embeddings/evaluation/protocols/configurable_learned_relation_v1/gate_test.py \
	  code/embeddings/evaluation/protocols/configurable_learned_relation_v1/publish_test.py \
	  code/embeddings/evaluation/protocols/configurable_content_relation_v1/gate_test.py \
	  code/embeddings/evaluation/protocols/configurable_content_relation_v1/publish_test.py \
	  code/embeddings/evaluation/protocols/learned_train_v1/test.py; do \
	  python3 -B "$$fixture"; python3 -B -O "$$fixture"; \
	done

HISTORICAL_PATHS_TEST := $(PYTHON_MIGRATION_BUILD)/historical-paths/test
$(HISTORICAL_PATHS_TEST): code/embeddings/evaluation/tests/historical_parent_paths_test.cpp code/embeddings/evaluation/include/historical_parent_paths.h code/scripts/python-migration.mk
	@set -eu; test -f /.dockerenv && test "$(CURDIR)" = /mityane || exit 1; \
	build=$$(realpath -m -- '$(dir $(HISTORICAL_PATHS_TEST))'); \
	case "$$build" in /opt/mityane-geometrics/*) ;; *) exit 1;; esac; \
	task_tmp=$$(realpath -m -- "$$build/tmp"); \
	case "$$task_tmp" in "$$build"/*) ;; *) exit 1;; esac; \
	mkdir -p -- "$$task_tmp"; export TMPDIR="$$task_tmp"; \
	exec 9>"$$build/.migration.lock"; flock -n 9 || exit 1; \
	temporary=$$(mktemp "$$build/test.XXXXXX"); trap 'rm -f -- "$$temporary"' EXIT HUP INT TERM; \
	$(CXX) -std=c++17 -O1 -Wall -Wextra $< -o "$$temporary"; \
	mv -f -- "$$temporary" "$$build/test"; trap - EXIT HUP INT TERM

test-historical-parent-paths: $(HISTORICAL_PATHS_TEST)
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || exit 1
	"$(HISTORICAL_PATHS_TEST)"
