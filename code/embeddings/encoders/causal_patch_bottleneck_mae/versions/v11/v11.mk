# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V11_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v11
CPB_V11_CPPFLAGS := -I$(CPB_V11_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V11_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v11
CPB_V11_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v11_variance_gradient
CPB_V11_TEMPORAL_POOLING_TEST := $(BUILD_DIR)/test_cpb_v11_temporal_pooling
.PHONY: build-cpb-v11-engineering test-cpb-v11 build-cpb-v11-variance-gradient test-cpb-v11-variance-gradient
build-cpb-v11-engineering: $(CPB_V11_TEST) $(CPB_V11_VARIANCE_GRADIENT_TEST) $(CPB_V11_TEMPORAL_POOLING_TEST)
test-cpb-v11: build-cpb-v11-engineering
	$(CPB_V11_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v11_test.o: $(CPB_V11_ROOT)/tests/v11_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V11_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V11_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v11_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v11_test.d

build-cpb-v11-variance-gradient: $(CPB_V11_VARIANCE_GRADIENT_TEST)
test-cpb-v11-variance-gradient: build-cpb-v11-variance-gradient
	$(CPB_V11_VARIANCE_GRADIENT_TEST) --settings $(CPB_V11_ROOT)/config/v11.conf
$(BUILD_DIR)/cpb_v11_variance_gradient_test.o: $(CPB_V11_ROOT)/tests/v11_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V11_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V11_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v11_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v11_variance_gradient_test.d

.PHONY: build-cpb-v11-temporal-pooling test-cpb-v11-temporal-pooling
build-cpb-v11-temporal-pooling: $(CPB_V11_TEMPORAL_POOLING_TEST)
test-cpb-v11-temporal-pooling: build-cpb-v11-temporal-pooling
	$(CPB_V11_TEMPORAL_POOLING_TEST) --settings $(CPB_V11_ROOT)/config/v11.conf
$(BUILD_DIR)/cpb_v11_temporal_pooling_test.o: $(CPB_V11_ROOT)/tests/v11_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V11_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V11_TEMPORAL_POOLING_TEST): $(BUILD_DIR)/cpb_v11_temporal_pooling_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v11_temporal_pooling_test.d
