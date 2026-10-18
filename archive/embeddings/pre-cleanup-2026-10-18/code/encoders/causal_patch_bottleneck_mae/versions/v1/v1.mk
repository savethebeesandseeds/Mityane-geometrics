# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V1_ROOT := code/encoders/causal_patch_bottleneck_mae/versions/v1
CPB_V1_CPPFLAGS := -I$(CPB_V1_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V1_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v1
.PHONY: build-cpb-v1-engineering test-cpb-v1
build-cpb-v1-engineering: $(CPB_V1_TEST)
test-cpb-v1: build-cpb-v1-engineering
	$(CPB_V1_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v1_test.o: $(CPB_V1_ROOT)/tests/v1_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V1_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v1_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v1_test.d
