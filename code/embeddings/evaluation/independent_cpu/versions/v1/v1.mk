# Candidate migration targets; the root exposes them only through explicit goals.
# No Torch link flags, model code, producer math, Python or new packages.
INDEPENDENT_CPU_V1_ROOT := code/embeddings/evaluation/independent_cpu/versions/v1
INDEPENDENT_CPU_V1_BUILD ?= /opt/mityane-geometrics/migration/independent-cpu-v1
INDEPENDENT_CPU_V1_FLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -fno-fast-math -ffp-contract=off

.PHONY: build-independent-cpu-v1 check-independent-cpu-v1
build-independent-cpu-v1: $(INDEPENDENT_CPU_V1_BUILD)/math-test

$(INDEPENDENT_CPU_V1_BUILD)/math-test: $(INDEPENDENT_CPU_V1_ROOT)/src/math.cpp $(INDEPENDENT_CPU_V1_ROOT)/tests/math_test.cpp $(INDEPENDENT_CPU_V1_ROOT)/include/mityane/independent_cpu/v1/math.h $(INDEPENDENT_CPU_V1_ROOT)/v1.mk
	@set -eu; \
	test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }; \
	build=$$(realpath -m -- '$(INDEPENDENT_CPU_V1_BUILD)'); \
	case "$$build" in /opt/mityane-geometrics/*) ;; *) echo 'Independent CPU build must stay under /opt/mityane-geometrics.' >&2; exit 1;; esac; \
	tempdir=$$(realpath -m -- "$$build/tmp"); \
	case "$$tempdir" in "$$build"/*) ;; *) echo 'Compiler TMPDIR escapes named build directory.' >&2; exit 1;; esac; \
	mkdir -p "$$tempdir"; export TMPDIR="$$tempdir"; \
	exec 9>"$$build/.migration.lock"; flock -n 9 || { echo 'Independent CPU build directory is busy.' >&2; exit 1; }; \
	tmp=$$(mktemp "$$build/math-test.XXXXXX"); trap 'rm -f -- "$$tmp"' EXIT HUP INT TERM; \
	$(CXX) $(INDEPENDENT_CPU_V1_FLAGS) -I$(INDEPENDENT_CPU_V1_ROOT)/include $(filter %.cpp,$^) -o "$$tmp"; \
	mv -f -- "$$tmp" "$$build/math-test"; trap - EXIT HUP INT TERM

check-independent-cpu-v1: build-independent-cpu-v1
	@set -eu; \
	test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }; \
	build=$$(realpath -m -- '$(INDEPENDENT_CPU_V1_BUILD)'); \
	case "$$build" in /opt/mityane-geometrics/*) ;; *) echo 'Independent CPU tests must stay under /opt/mityane-geometrics.' >&2; exit 1;; esac; \
	tempdir=$$(realpath -m -- "$$build/tmp"); \
	case "$$tempdir" in "$$build"/*) ;; *) echo 'Test TMPDIR escapes named build directory.' >&2; exit 1;; esac; \
	mkdir -p "$$tempdir"; export TMPDIR="$$tempdir"; \
	exec 9>"$$build/.migration.lock"; flock -n 9 || { echo 'Independent CPU build directory is busy.' >&2; exit 1; }; \
	"$$build/math-test"
