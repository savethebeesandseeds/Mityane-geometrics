# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V5_ROOT := code/encoders/causal_patch_bottleneck_mae/versions/v5
CPB_V5_CPPFLAGS := -I$(CPB_V5_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V5_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v5
CPB_V5_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v5_variance_gradient
.PHONY: build-cpb-v5-engineering test-cpb-v5 build-cpb-v5-variance-gradient test-cpb-v5-variance-gradient
build-cpb-v5-engineering: $(CPB_V5_TEST) $(CPB_V5_VARIANCE_GRADIENT_TEST)
test-cpb-v5: build-cpb-v5-engineering
	$(CPB_V5_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v5_test.o: $(CPB_V5_ROOT)/tests/v5_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V5_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V5_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v5_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v5_test.d

build-cpb-v5-variance-gradient: $(CPB_V5_VARIANCE_GRADIENT_TEST)
test-cpb-v5-variance-gradient: build-cpb-v5-variance-gradient
	$(CPB_V5_VARIANCE_GRADIENT_TEST) --settings $(CPB_V5_ROOT)/config/v5.conf
$(BUILD_DIR)/cpb_v5_variance_gradient_test.o: $(CPB_V5_ROOT)/tests/v5_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V5_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V5_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v5_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v5_variance_gradient_test.d
