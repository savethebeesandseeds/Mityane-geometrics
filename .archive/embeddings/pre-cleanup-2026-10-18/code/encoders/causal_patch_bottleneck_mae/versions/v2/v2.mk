# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V2_ROOT := code/encoders/causal_patch_bottleneck_mae/versions/v2
CPB_V2_CPPFLAGS := -I$(CPB_V2_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V2_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v2
.PHONY: build-cpb-v2-engineering test-cpb-v2
build-cpb-v2-engineering: $(CPB_V2_TEST)
test-cpb-v2: build-cpb-v2-engineering
	$(CPB_V2_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v2_test.o: $(CPB_V2_ROOT)/tests/v2_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V2_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V2_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v2_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v2_test.d
