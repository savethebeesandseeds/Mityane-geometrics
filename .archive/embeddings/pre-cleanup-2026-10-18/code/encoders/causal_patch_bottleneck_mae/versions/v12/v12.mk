# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V12_ROOT := code/encoders/causal_patch_bottleneck_mae/versions/v12
CPB_V12_CPPFLAGS := -I$(CPB_V12_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V12_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v12
CPB_V12_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v12_variance_gradient
CPB_V12_TEMPORAL_POOLING_TEST := $(BUILD_DIR)/test_cpb_v12_temporal_pooling
.PHONY: build-cpb-v12-engineering test-cpb-v12 build-cpb-v12-variance-gradient test-cpb-v12-variance-gradient
build-cpb-v12-engineering: $(CPB_V12_TEST) $(CPB_V12_VARIANCE_GRADIENT_TEST) $(CPB_V12_TEMPORAL_POOLING_TEST)
test-cpb-v12: build-cpb-v12-engineering
	$(CPB_V12_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v12_test.o: $(CPB_V12_ROOT)/tests/v12_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V12_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V12_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v12_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v12_test.d

build-cpb-v12-variance-gradient: $(CPB_V12_VARIANCE_GRADIENT_TEST)
test-cpb-v12-variance-gradient: build-cpb-v12-variance-gradient
	$(CPB_V12_VARIANCE_GRADIENT_TEST) --settings $(CPB_V12_ROOT)/config/v12.conf
$(BUILD_DIR)/cpb_v12_variance_gradient_test.o: $(CPB_V12_ROOT)/tests/v12_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V12_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V12_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v12_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v12_variance_gradient_test.d

.PHONY: build-cpb-v12-temporal-pooling test-cpb-v12-temporal-pooling
build-cpb-v12-temporal-pooling: $(CPB_V12_TEMPORAL_POOLING_TEST)
test-cpb-v12-temporal-pooling: build-cpb-v12-temporal-pooling
	$(CPB_V12_TEMPORAL_POOLING_TEST) --settings $(CPB_V12_ROOT)/config/v12.conf
$(BUILD_DIR)/cpb_v12_temporal_pooling_test.o: $(CPB_V12_ROOT)/tests/v12_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V12_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V12_TEMPORAL_POOLING_TEST): $(BUILD_DIR)/cpb_v12_temporal_pooling_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v12_temporal_pooling_test.d
