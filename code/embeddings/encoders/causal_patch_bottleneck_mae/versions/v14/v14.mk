# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V14_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v14
CPB_V14_CPPFLAGS := -I$(CPB_V14_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V14_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v14
CPB_V14_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v14_variance_gradient
CPB_V14_TEMPORAL_POOLING_TEST := $(BUILD_DIR)/test_cpb_v14_temporal_pooling
.PHONY: build-cpb-v14-engineering test-cpb-v14 build-cpb-v14-variance-gradient test-cpb-v14-variance-gradient
build-cpb-v14-engineering: $(CPB_V14_TEST) $(CPB_V14_VARIANCE_GRADIENT_TEST) $(CPB_V14_TEMPORAL_POOLING_TEST)
test-cpb-v14: build-cpb-v14-engineering
	$(CPB_V14_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v14_test.o: $(CPB_V14_ROOT)/tests/v14_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V14_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V14_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v14_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v14_test.d

build-cpb-v14-variance-gradient: $(CPB_V14_VARIANCE_GRADIENT_TEST)
test-cpb-v14-variance-gradient: build-cpb-v14-variance-gradient
	$(CPB_V14_VARIANCE_GRADIENT_TEST) --settings $(CPB_V14_ROOT)/config/v14.conf
$(BUILD_DIR)/cpb_v14_variance_gradient_test.o: $(CPB_V14_ROOT)/tests/v14_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V14_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V14_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v14_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v14_variance_gradient_test.d

.PHONY: build-cpb-v14-temporal-pooling test-cpb-v14-temporal-pooling
build-cpb-v14-temporal-pooling: $(CPB_V14_TEMPORAL_POOLING_TEST)
test-cpb-v14-temporal-pooling: build-cpb-v14-temporal-pooling
	$(CPB_V14_TEMPORAL_POOLING_TEST) --settings $(CPB_V14_ROOT)/config/v14.conf
$(BUILD_DIR)/cpb_v14_temporal_pooling_test.o: $(CPB_V14_ROOT)/tests/v14_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V14_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V14_TEMPORAL_POOLING_TEST): $(BUILD_DIR)/cpb_v14_temporal_pooling_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v14_temporal_pooling_test.d
