# Candidate migration targets; the root exposes them only through explicit goals.
# Run only in the managed Linux container once runtime work is authorized again.
SAVED_TENSOR_V1_ROOT := code/shared/artifacts/saved_tensor/versions/v1
SAVED_TENSOR_V1_BUILD ?= /opt/mityane-geometrics/migration/saved-tensor-v1
SAVED_TENSOR_V1_FLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -fno-fast-math -ffp-contract=off

.PHONY: build-saved-tensor-v1 check-saved-tensor-v1
build-saved-tensor-v1: $(SAVED_TENSOR_V1_BUILD)/reader-test

$(SAVED_TENSOR_V1_BUILD)/reader-test: $(SAVED_TENSOR_V1_ROOT)/src/reader.cpp $(SAVED_TENSOR_V1_ROOT)/tests/reader_test.cpp $(SAVED_TENSOR_V1_ROOT)/include/mityane/saved_tensor/v1/reader.h $(SAVED_TENSOR_V1_ROOT)/v1.mk
	@set -eu; \
	test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }; \
	build=$$(realpath -m -- '$(SAVED_TENSOR_V1_BUILD)'); \
	case "$$build" in /opt/mityane-geometrics/*) ;; *) echo 'Saved-tensor build must stay under /opt/mityane-geometrics.' >&2; exit 1;; esac; \
	tempdir=$$(realpath -m -- "$$build/tmp"); \
	case "$$tempdir" in "$$build"/*) ;; *) echo 'Compiler TMPDIR escapes named build directory.' >&2; exit 1;; esac; \
	mkdir -p "$$tempdir"; export TMPDIR="$$tempdir"; \
	exec 9>"$$build/.migration.lock"; flock -n 9 || { echo 'Saved-tensor build directory is busy.' >&2; exit 1; }; \
	tmp=$$(mktemp "$$build/reader-test.XXXXXX"); trap 'rm -f -- "$$tmp"' EXIT HUP INT TERM; \
	$(CXX) $(SAVED_TENSOR_V1_FLAGS) -I$(SAVED_TENSOR_V1_ROOT)/include $(filter %.cpp,$^) -o "$$tmp"; \
	mv -f -- "$$tmp" "$$build/reader-test"; trap - EXIT HUP INT TERM

check-saved-tensor-v1: build-saved-tensor-v1
	@set -eu; \
	test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }; \
	build=$$(realpath -m -- '$(SAVED_TENSOR_V1_BUILD)'); \
	case "$$build" in /opt/mityane-geometrics/*) ;; *) echo 'Saved-tensor tests must stay under /opt/mityane-geometrics.' >&2; exit 1;; esac; \
	tempdir=$$(realpath -m -- "$$build/tmp"); \
	case "$$tempdir" in "$$build"/*) ;; *) echo 'Test TMPDIR escapes named build directory.' >&2; exit 1;; esac; \
	mkdir -p "$$tempdir"; export TMPDIR="$$tempdir"; \
	exec 9>"$$build/.migration.lock"; flock -n 9 || { echo 'Saved-tensor build directory is busy.' >&2; exit 1; }; \
	fixtures=$$(realpath -m -- "$$build/fixtures"); \
	case "$$fixtures" in "$$build"/*) ;; *) echo 'Fixture path escapes named build directory.' >&2; exit 1;; esac; \
	"$$build/reader-test" "$$fixtures"
