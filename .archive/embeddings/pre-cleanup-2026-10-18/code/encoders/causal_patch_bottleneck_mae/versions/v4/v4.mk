# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V4_ROOT := code/encoders/causal_patch_bottleneck_mae/versions/v4
CPB_V4_CPPFLAGS := -I$(CPB_V4_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V4_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v4
CPB_V4_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v4_variance_gradient
.PHONY: build-cpb-v4-engineering test-cpb-v4 build-cpb-v4-variance-gradient test-cpb-v4-variance-gradient
build-cpb-v4-engineering: $(CPB_V4_TEST) $(CPB_V4_VARIANCE_GRADIENT_TEST)
test-cpb-v4: build-cpb-v4-engineering
	$(CPB_V4_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v4_test.o: $(CPB_V4_ROOT)/tests/v4_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V4_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V4_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v4_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v4_test.d

build-cpb-v4-variance-gradient: $(CPB_V4_VARIANCE_GRADIENT_TEST)
test-cpb-v4-variance-gradient: build-cpb-v4-variance-gradient
	$(CPB_V4_VARIANCE_GRADIENT_TEST) --settings $(CPB_V4_ROOT)/config/v4.conf
$(BUILD_DIR)/cpb_v4_variance_gradient_test.o: $(CPB_V4_ROOT)/tests/v4_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V4_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V4_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v4_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v4_variance_gradient_test.d
