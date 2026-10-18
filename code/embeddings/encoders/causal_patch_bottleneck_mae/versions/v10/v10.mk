# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V10_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v10
CPB_V10_CPPFLAGS := -I$(CPB_V10_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V10_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v10
CPB_V10_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v10_variance_gradient
CPB_V10_TEMPORAL_POOLING_TEST := $(BUILD_DIR)/test_cpb_v10_temporal_pooling
.PHONY: build-cpb-v10-engineering test-cpb-v10 build-cpb-v10-variance-gradient test-cpb-v10-variance-gradient
build-cpb-v10-engineering: $(CPB_V10_TEST) $(CPB_V10_VARIANCE_GRADIENT_TEST) $(CPB_V10_TEMPORAL_POOLING_TEST)
test-cpb-v10: build-cpb-v10-engineering
	$(CPB_V10_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v10_test.o: $(CPB_V10_ROOT)/tests/v10_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V10_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V10_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v10_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v10_test.d

build-cpb-v10-variance-gradient: $(CPB_V10_VARIANCE_GRADIENT_TEST)
test-cpb-v10-variance-gradient: build-cpb-v10-variance-gradient
	$(CPB_V10_VARIANCE_GRADIENT_TEST) --settings $(CPB_V10_ROOT)/config/v10.conf
$(BUILD_DIR)/cpb_v10_variance_gradient_test.o: $(CPB_V10_ROOT)/tests/v10_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V10_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V10_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v10_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v10_variance_gradient_test.d

.PHONY: build-cpb-v10-temporal-pooling test-cpb-v10-temporal-pooling
build-cpb-v10-temporal-pooling: $(CPB_V10_TEMPORAL_POOLING_TEST)
test-cpb-v10-temporal-pooling: build-cpb-v10-temporal-pooling
	$(CPB_V10_TEMPORAL_POOLING_TEST) --settings $(CPB_V10_ROOT)/config/v10.conf
$(BUILD_DIR)/cpb_v10_temporal_pooling_test.o: $(CPB_V10_ROOT)/tests/v10_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V10_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V10_TEMPORAL_POOLING_TEST): $(BUILD_DIR)/cpb_v10_temporal_pooling_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v10_temporal_pooling_test.d
