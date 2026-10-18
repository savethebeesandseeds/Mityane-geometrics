# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V8_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v8
CPB_V8_CPPFLAGS := -I$(CPB_V8_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V8_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v8
CPB_V8_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v8_variance_gradient
CPB_V8_TEMPORAL_POOLING_TEST := $(BUILD_DIR)/test_cpb_v8_temporal_pooling
.PHONY: build-cpb-v8-engineering test-cpb-v8 build-cpb-v8-variance-gradient test-cpb-v8-variance-gradient
build-cpb-v8-engineering: $(CPB_V8_TEST) $(CPB_V8_VARIANCE_GRADIENT_TEST) $(CPB_V8_TEMPORAL_POOLING_TEST)
test-cpb-v8: build-cpb-v8-engineering
	$(CPB_V8_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v8_test.o: $(CPB_V8_ROOT)/tests/v8_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V8_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V8_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v8_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v8_test.d

build-cpb-v8-variance-gradient: $(CPB_V8_VARIANCE_GRADIENT_TEST)
test-cpb-v8-variance-gradient: build-cpb-v8-variance-gradient
	$(CPB_V8_VARIANCE_GRADIENT_TEST) --settings $(CPB_V8_ROOT)/config/v8.conf
$(BUILD_DIR)/cpb_v8_variance_gradient_test.o: $(CPB_V8_ROOT)/tests/v8_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V8_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V8_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v8_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v8_variance_gradient_test.d

.PHONY: build-cpb-v8-temporal-pooling test-cpb-v8-temporal-pooling
build-cpb-v8-temporal-pooling: $(CPB_V8_TEMPORAL_POOLING_TEST)
test-cpb-v8-temporal-pooling: build-cpb-v8-temporal-pooling
	$(CPB_V8_TEMPORAL_POOLING_TEST) --settings $(CPB_V8_ROOT)/config/v8.conf
$(BUILD_DIR)/cpb_v8_temporal_pooling_test.o: $(CPB_V8_ROOT)/tests/v8_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V8_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V8_TEMPORAL_POOLING_TEST): $(BUILD_DIR)/cpb_v8_temporal_pooling_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v8_temporal_pooling_test.d
