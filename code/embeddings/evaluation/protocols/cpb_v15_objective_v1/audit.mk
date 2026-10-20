# Copyright (c) 2026 Waajacu. See LICENSE.
# Native saved-array arithmetic, isolated from producer implementation headers.
CPB_V15_AUDIT_ROOT := code/embeddings/evaluation/protocols/cpb_v15_objective_v1
CPB_V15_AUDIT_FLAGS := -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
.PHONY: build-cpb-v15-audit test-cpb-v15-audit
build-cpb-v15-audit: $(BUILD_DIR)/cpb_v15_independent_audit $(BUILD_DIR)/cpb_v15_audit_fixture
$(BUILD_DIR)/cpb_v15_independent_audit.o: $(CPB_V15_AUDIT_ROOT)/audit.cpp
	test -f /.dockerenv
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_AUDIT_FLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_independent_audit: $(BUILD_DIR)/cpb_v15_independent_audit.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_audit_fixture.o: $(CPB_V15_AUDIT_ROOT)/audit_fixture.cpp
	test -f /.dockerenv
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_AUDIT_FLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_audit_fixture: $(BUILD_DIR)/cpb_v15_audit_fixture.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-cpb-v15-audit: build-cpb-v15-audit
	CPB_V15_AUDIT_BINARY="$(BUILD_DIR)/cpb_v15_independent_audit" CPB_V15_AUDIT_FIXTURE_BINARY="$(BUILD_DIR)/cpb_v15_audit_fixture" bash code/scripts/node.sh --test $(CPB_V15_AUDIT_ROOT)/audit_test.mjs $(CPB_V15_AUDIT_ROOT)/audit-run.test.mjs
-include $(BUILD_DIR)/cpb_v15_independent_audit.d $(BUILD_DIR)/cpb_v15_audit_fixture.d
