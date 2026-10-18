# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V7_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v7
CPB_V7_CPPFLAGS := -I$(CPB_V7_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V7_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v7
CPB_V7_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v7_variance_gradient
.PHONY: build-cpb-v7-engineering test-cpb-v7 build-cpb-v7-variance-gradient test-cpb-v7-variance-gradient
build-cpb-v7-engineering: $(CPB_V7_TEST) $(CPB_V7_VARIANCE_GRADIENT_TEST)
test-cpb-v7: build-cpb-v7-engineering
	$(CPB_V7_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v7_test.o: $(CPB_V7_ROOT)/tests/v7_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V7_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V7_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v7_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v7_test.d

build-cpb-v7-variance-gradient: $(CPB_V7_VARIANCE_GRADIENT_TEST)
test-cpb-v7-variance-gradient: build-cpb-v7-variance-gradient
	$(CPB_V7_VARIANCE_GRADIENT_TEST) --settings $(CPB_V7_ROOT)/config/v7.conf
$(BUILD_DIR)/cpb_v7_variance_gradient_test.o: $(CPB_V7_ROOT)/tests/v7_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V7_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V7_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v7_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v7_variance_gradient_test.d
