# Included by the root Makefile for both isolated maintenance and mixed builds.
.PHONY: test-maintenance check-layout
MAINTENANCE_TEST_TMP := $(if $(filter /opt/mityane-geometrics/%,$(TMPDIR)),$(TMPDIR),/opt/mityane-geometrics/maintenance/tests/tmp)
test-maintenance:
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }
	@mkdir -p "$(MAINTENANCE_TEST_TMP)"
	TMPDIR="$(MAINTENANCE_TEST_TMP)" bash code/scripts/node.sh --test code/scripts/tests/*.test.mjs

check-layout:
	bash code/scripts/node.sh code/scripts/check-layout.mjs

include web/dashboard/scripts/branding.mk

.PHONY: test-cpb-v14-closeout
CPB_V14_CLOSEOUT_TEST_FILES ?= $(wildcard code/embeddings/evaluation/protocols/cpb_v14_closeout_v1/*.test.mjs)
test-cpb-v14-closeout:
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }
	@mkdir -p "$(MAINTENANCE_TEST_TMP)"
	TMPDIR="$(MAINTENANCE_TEST_TMP)" bash code/scripts/node.sh --test $(CPB_V14_CLOSEOUT_TEST_FILES)

# Opt-in candidate checks only. Not prerequisites of maintained tools or runs.
include code/scripts/python-migration.mk
