# Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
# Uses the authoritative root toolchain and existing LibTorch SDK.
CPB_V19_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v19
CPB_V19_PROTOCOL := code/embeddings/evaluation/protocols/cpb_v19_teacher_v1
CPB_V19_CPPFLAGS := -I$(CPB_V19_ROOT)/include $(CPB_FAMILY_CPPFLAGS) -Icode/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/include
CPB_V19_OUTPUT ?= $(RUN_ROOT)/engineering-001
.PHONY: build-cpb-v19 test-cpb-v19 build-cpb-v19-trainer build-cpb-v19-readout
build-cpb-v19: $(BUILD_DIR)/test_cpb_v19
test-cpb-v19: build-cpb-v19
	$(BUILD_DIR)/test_cpb_v19 --settings $(CPB_V19_ROOT)/config/research.conf --alternative-settings $(CPB_FAMILY_ROOT)/config/paired.conf --output "$(CPB_V19_OUTPUT)"
build-cpb-v19-trainer: $(BUILD_DIR)/cpb_v19_train
build-cpb-v19-readout: $(BUILD_DIR)/cpb_v19_readout
$(BUILD_DIR)/cpb_v19_engineering.o: $(CPB_V19_ROOT)/tests/engineering_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V19_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_cpb_v19: $(BUILD_DIR)/cpb_v19_engineering.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v19_train.o: code/embeddings/scripts/cpb_v19_train.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V19_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v19_train: $(BUILD_DIR)/cpb_v19_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v19_readout.o: $(CPB_V19_PROTOCOL)/readout.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V19_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v19_fixed_regression.o: code/embeddings/evaluation/protocols/cpb_v16_temporal_v1/deps/src/fixed_regression.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V19_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v19_readout: $(BUILD_DIR)/cpb_v19_readout.o $(BUILD_DIR)/cpb_v19_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v19_engineering.d $(BUILD_DIR)/cpb_v19_train.d $(BUILD_DIR)/cpb_v19_readout.d $(BUILD_DIR)/cpb_v19_fixed_regression.d
