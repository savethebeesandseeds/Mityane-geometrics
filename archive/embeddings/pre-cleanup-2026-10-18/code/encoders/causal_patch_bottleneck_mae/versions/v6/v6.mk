# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V6_ROOT := code/encoders/causal_patch_bottleneck_mae/versions/v6
CPB_V6_CPPFLAGS := -I$(CPB_V6_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V6_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v6
CPB_V6_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v6_variance_gradient
.PHONY: build-cpb-v6-engineering test-cpb-v6 build-cpb-v6-variance-gradient test-cpb-v6-variance-gradient
build-cpb-v6-engineering: $(CPB_V6_TEST) $(CPB_V6_VARIANCE_GRADIENT_TEST)
test-cpb-v6: build-cpb-v6-engineering
	$(CPB_V6_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v6_test.o: $(CPB_V6_ROOT)/tests/v6_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V6_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V6_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v6_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v6_test.d

build-cpb-v6-variance-gradient: $(CPB_V6_VARIANCE_GRADIENT_TEST)
test-cpb-v6-variance-gradient: build-cpb-v6-variance-gradient
	$(CPB_V6_VARIANCE_GRADIENT_TEST) --settings $(CPB_V6_ROOT)/config/v6.conf
$(BUILD_DIR)/cpb_v6_variance_gradient_test.o: $(CPB_V6_ROOT)/tests/v6_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V6_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V6_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v6_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v6_variance_gradient_test.d
