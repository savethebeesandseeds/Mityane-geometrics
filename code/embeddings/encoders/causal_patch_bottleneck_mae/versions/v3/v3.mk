# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V3_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v3
CPB_V3_CPPFLAGS := -I$(CPB_V3_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V3_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v3
.PHONY: build-cpb-v3-engineering test-cpb-v3
build-cpb-v3-engineering: $(CPB_V3_TEST)
test-cpb-v3: build-cpb-v3-engineering
	$(CPB_V3_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v3_test.o: $(CPB_V3_ROOT)/tests/v3_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V3_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V3_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v3_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v3_test.d
