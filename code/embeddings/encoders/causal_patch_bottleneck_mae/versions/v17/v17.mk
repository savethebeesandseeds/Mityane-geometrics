# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../LICENSE.
# Uses only the authoritative root native toolchain and inherited SDK settings.
CPB_V17_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v17
CPB_V17_PROTOCOL := code/embeddings/evaluation/protocols/cpb_v17_balance_v1
CPB_V17_CPPFLAGS := -I$(CPB_V17_ROOT)/include $(CPB_FAMILY_CPPFLAGS) -Icode/shared/artifacts/versions/v1/include -Icode/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/include
CPB_V17_OUTPUT ?= $(RUN_ROOT)/engineering-001
.PHONY: build-cpb-v17 test-cpb-v17 build-cpb-v17-trainer build-cpb-v17-windows build-cpb-v17-diagnostic build-cpb-v17-readout build-cpb-v17-audit
build-cpb-v17: $(BUILD_DIR)/test_cpb_v17
test-cpb-v17: build-cpb-v17
	$(BUILD_DIR)/test_cpb_v17 --settings $(CPB_V17_ROOT)/config/research.conf --alternative-settings $(CPB_FAMILY_ROOT)/config/paired.conf --output "$(CPB_V17_OUTPUT)"
build-cpb-v17-trainer: $(BUILD_DIR)/cpb_v17_train
build-cpb-v17-windows: $(BUILD_DIR)/cpb_v17_windows
build-cpb-v17-diagnostic: $(BUILD_DIR)/cpb_v17_diagnostic
build-cpb-v17-readout: $(BUILD_DIR)/cpb_v17_readout
build-cpb-v17-audit: $(BUILD_DIR)/cpb_v17_audit
$(BUILD_DIR)/cpb_v17_engineering.o: $(CPB_V17_ROOT)/tests/engineering_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V17_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_cpb_v17: $(BUILD_DIR)/cpb_v17_engineering.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v17_train.o: code/embeddings/scripts/cpb_v17_train.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V17_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v17_train: $(BUILD_DIR)/cpb_v17_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v17_windows.o: code/data/market/experiments/cpb_v17_balance_v1/windows.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V17_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v17_windows: $(BUILD_DIR)/cpb_v17_windows.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v17_diagnostic.o: $(CPB_V17_PROTOCOL)/diagnostic.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V17_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v17_diagnostic: $(BUILD_DIR)/cpb_v17_diagnostic.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v17_readout.o: $(CPB_V17_PROTOCOL)/readout.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V17_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v17_fixed_regression.o: code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/src/fixed_regression.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V17_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v17_readout: $(BUILD_DIR)/cpb_v17_readout.o $(BUILD_DIR)/cpb_v17_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v17_audit.o: $(CPB_V17_PROTOCOL)/audit.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V17_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v17_audit: $(BUILD_DIR)/cpb_v17_audit.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v17_engineering.d $(BUILD_DIR)/cpb_v17_train.d $(BUILD_DIR)/cpb_v17_windows.d $(BUILD_DIR)/cpb_v17_diagnostic.d $(BUILD_DIR)/cpb_v17_readout.d $(BUILD_DIR)/cpb_v17_fixed_regression.d $(BUILD_DIR)/cpb_v17_audit.d
