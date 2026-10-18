# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separate application/evaluation adapters; no raw-family encoder dependency.
CPB_V1_APP_ROOT := code/evaluation/causal_patch_bottleneck_mae_v1
CPB_V1_APP_CPPFLAGS := -Icode/encoders/causal_patch_bottleneck_mae/versions/v1/include -Icode/data/include \
  -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V1_EXTRACT := $(BUILD_DIR)/causal_patch_bottleneck_mae_v1_extract
CPB_V1_READ := $(BUILD_DIR)/causal_patch_bottleneck_mae_v1_read
CPB_V1_HEAD_DEPS := code/evaluation/market_synthetic_transfer_v1/deps
CPB_V1_HEAD_OBJECT_ROOT := /opt/cuwacunu/build/market-location-scale-transfer
CPB_V1_HEAD_OBJECTS := $(CPB_V1_HEAD_OBJECT_ROOT)/market_location_scale_synthetic_transfer_data.o \
  $(CPB_V1_HEAD_OBJECT_ROOT)/market_location_scale_synthetic_transfer_feature_harness.o \
  $(CPB_V1_HEAD_OBJECT_ROOT)/market_location_scale_synthetic_transfer_fixed_feature_readouts.o
.PHONY: build-cpb-v1-extract build-cpb-v1-read
build-cpb-v1-extract: $(CPB_V1_EXTRACT)
build-cpb-v1-read: $(CPB_V1_READ)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v1_extract.o: $(CPB_V1_APP_ROOT)/extract_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V1_APP_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V1_EXTRACT): $(BUILD_DIR)/causal_patch_bottleneck_mae_v1_extract.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/causal_patch_bottleneck_mae_v1_read.o: $(CPB_V1_APP_ROOT)/read_main.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(CPB_V1_HEAD_DEPS)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1 $(CXXFLAGS) -c $< -o $@
$(CPB_V1_READ): $(BUILD_DIR)/causal_patch_bottleneck_mae_v1_read.o $(CPB_V1_HEAD_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v1_extract.d $(BUILD_DIR)/causal_patch_bottleneck_mae_v1_read.d
