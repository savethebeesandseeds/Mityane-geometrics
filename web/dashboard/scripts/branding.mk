# SPDX-License-Identifier: AGPL-3.0-only
# Included by the root Makefile. Native products stay in the container.
DASHBOARD_BRANDING_BIN := /opt/mityane-geometrics/web/build/prepare-branding-native
DASHBOARD_TMP := $(if $(filter /opt/mityane-geometrics/%,$(TMPDIR)),$(TMPDIR),/opt/mityane-geometrics/web/tmp)

.PHONY: dashboard-branding-native test-dashboard-branding
dashboard-branding-native: $(DASHBOARD_BRANDING_BIN)

$(DASHBOARD_BRANDING_BIN): web/dashboard/scripts/prepare-branding-native.cpp web/dashboard/scripts/branding.mk
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Build only through container.ps1 in mityane-geometrics.' >&2; exit 1; }
	mkdir -p "$(@D)" "$(DASHBOARD_TMP)"
	@set -e; exec 9>"$@.lock"; flock 9; \
	  temporary="$@.tmp.$$$$"; trap 'rm -f -- "$$temporary"' EXIT; \
	  TMPDIR="$(DASHBOARD_TMP)" $(CXX) -std=c++20 -O2 -Wall -Wextra $< -ldl -o "$$temporary"; \
	  mv -f -- "$$temporary" "$@"

test-dashboard-branding: dashboard-branding-native
	mkdir -p "$(DASHBOARD_TMP)"
	TMPDIR="$(DASHBOARD_TMP)" bash code/scripts/node.sh --test web/dashboard/tests/prepare-branding.test.mjs
