# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../LICENSE.
# Family screen. Width and update count are checkpoint data, not copied harness constants.
CPB_SCREEN_PROTOCOL := code/embeddings/evaluation/protocols/cpb_family_screen_v1
CPB_SCREEN_CPPFLAGS := $(CPB_FAMILY_CPPFLAGS) -Icode/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/include
CPB_SCREEN_SELF_TEST_OUTPUT ?= $(RUN_ROOT)/cpb-family-screen-audit-self-test
.PHONY: build-cpb-family-screen test-cpb-family-screen
build-cpb-family-screen: $(BUILD_DIR)/cpb_family_screen_readout $(BUILD_DIR)/cpb_family_screen_audit
test-cpb-family-screen: build-cpb-family-screen
	bash code/scripts/node.sh --test $(CPB_SCREEN_PROTOCOL)/decision.test.mjs
	rm -rf "$(CPB_SCREEN_SELF_TEST_OUTPUT)" "$(CPB_SCREEN_SELF_TEST_OUTPUT).scratch"
	mkdir -p "$(dir $(CPB_SCREEN_SELF_TEST_OUTPUT))"
	$(BUILD_DIR)/cpb_family_screen_audit --self-test --output "$(CPB_SCREEN_SELF_TEST_OUTPUT)"
$(BUILD_DIR)/cpb_family_screen_readout.o: $(CPB_SCREEN_PROTOCOL)/readout.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_family_screen_fixed_regression.o: code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/src/fixed_regression.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_family_screen_readout: $(BUILD_DIR)/cpb_family_screen_readout.o $(BUILD_DIR)/cpb_family_screen_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_family_screen_audit.o: $(CPB_SCREEN_PROTOCOL)/audit.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_family_screen_audit: $(BUILD_DIR)/cpb_family_screen_audit.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_family_screen_readout.d $(BUILD_DIR)/cpb_family_screen_fixed_regression.d $(BUILD_DIR)/cpb_family_screen_audit.d
