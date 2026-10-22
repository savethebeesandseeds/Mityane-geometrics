# Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
# Uses only the root native toolchain/SDK; no standalone environment.
CPB_V18_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v18
CPB_V18_PROTOCOL := code/embeddings/evaluation/protocols/cpb_v18_span_v1
CPB_V18_CPPFLAGS := -I$(CPB_V18_ROOT)/include -I$(CPB_V17_ROOT)/include $(CPB_FAMILY_CPPFLAGS) -Icode/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/include
CPB_V18_OUTPUT ?= $(RUN_ROOT)/engineering-001
.PHONY: build-cpb-v18 test-cpb-v18 build-cpb-v18-trainer build-cpb-v18-readout
build-cpb-v18: $(BUILD_DIR)/test_cpb_v18
test-cpb-v18: build-cpb-v18
	$(BUILD_DIR)/test_cpb_v18 --settings $(CPB_V18_ROOT)/config/research.conf --alternative-settings $(CPB_FAMILY_ROOT)/config/paired.conf --output "$(CPB_V18_OUTPUT)"
build-cpb-v18-trainer: $(BUILD_DIR)/cpb_v18_train
build-cpb-v18-readout: $(BUILD_DIR)/cpb_v18_readout
$(BUILD_DIR)/cpb_v18_engineering.o: $(CPB_V18_ROOT)/tests/engineering_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V18_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_cpb_v18: $(BUILD_DIR)/cpb_v18_engineering.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v18_train.o: code/embeddings/scripts/cpb_v18_train.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V18_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v18_train: $(BUILD_DIR)/cpb_v18_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v18_readout.o: $(CPB_V18_PROTOCOL)/readout.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V18_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v18_fixed_regression.o: code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/src/fixed_regression.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V18_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v18_readout: $(BUILD_DIR)/cpb_v18_readout.o $(BUILD_DIR)/cpb_v18_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v18_engineering.d $(BUILD_DIR)/cpb_v18_train.d $(BUILD_DIR)/cpb_v18_readout.d $(BUILD_DIR)/cpb_v18_fixed_regression.d
