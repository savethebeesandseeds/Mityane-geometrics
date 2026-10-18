# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V9_ROOT := code/encoders/causal_patch_bottleneck_mae/versions/v9
CPB_V9_CPPFLAGS := -I$(CPB_V9_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V9_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v9
CPB_V9_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v9_variance_gradient
CPB_V9_TEMPORAL_POOLING_TEST := $(BUILD_DIR)/test_cpb_v9_temporal_pooling
.PHONY: build-cpb-v9-engineering test-cpb-v9 build-cpb-v9-variance-gradient test-cpb-v9-variance-gradient
build-cpb-v9-engineering: $(CPB_V9_TEST) $(CPB_V9_VARIANCE_GRADIENT_TEST) $(CPB_V9_TEMPORAL_POOLING_TEST)
test-cpb-v9: build-cpb-v9-engineering
	$(CPB_V9_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v9_test.o: $(CPB_V9_ROOT)/tests/v9_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V9_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V9_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v9_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v9_test.d

build-cpb-v9-variance-gradient: $(CPB_V9_VARIANCE_GRADIENT_TEST)
test-cpb-v9-variance-gradient: build-cpb-v9-variance-gradient
	$(CPB_V9_VARIANCE_GRADIENT_TEST) --settings $(CPB_V9_ROOT)/config/v9.conf
$(BUILD_DIR)/cpb_v9_variance_gradient_test.o: $(CPB_V9_ROOT)/tests/v9_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V9_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V9_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v9_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v9_variance_gradient_test.d

.PHONY: build-cpb-v9-temporal-pooling test-cpb-v9-temporal-pooling
build-cpb-v9-temporal-pooling: $(CPB_V9_TEMPORAL_POOLING_TEST)
test-cpb-v9-temporal-pooling: build-cpb-v9-temporal-pooling
	$(CPB_V9_TEMPORAL_POOLING_TEST) --settings $(CPB_V9_ROOT)/config/v9.conf
$(BUILD_DIR)/cpb_v9_temporal_pooling_test.o: $(CPB_V9_ROOT)/tests/v9_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V9_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V9_TEMPORAL_POOLING_TEST): $(BUILD_DIR)/cpb_v9_temporal_pooling_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v9_temporal_pooling_test.d
