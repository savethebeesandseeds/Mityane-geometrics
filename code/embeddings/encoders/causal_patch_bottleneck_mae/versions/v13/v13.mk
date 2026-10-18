# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V13_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v13
CPB_V13_CPPFLAGS := -I$(CPB_V13_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V13_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v13
CPB_V13_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v13_variance_gradient
CPB_V13_TEMPORAL_POOLING_TEST := $(BUILD_DIR)/test_cpb_v13_temporal_pooling
.PHONY: build-cpb-v13-engineering test-cpb-v13 build-cpb-v13-variance-gradient test-cpb-v13-variance-gradient
build-cpb-v13-engineering: $(CPB_V13_TEST) $(CPB_V13_VARIANCE_GRADIENT_TEST) $(CPB_V13_TEMPORAL_POOLING_TEST)
test-cpb-v13: build-cpb-v13-engineering
	$(CPB_V13_TEST)
$(BUILD_DIR)/causal_patch_bottleneck_mae_v13_test.o: $(CPB_V13_ROOT)/tests/v13_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V13_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V13_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v13_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v13_test.d

build-cpb-v13-variance-gradient: $(CPB_V13_VARIANCE_GRADIENT_TEST)
test-cpb-v13-variance-gradient: build-cpb-v13-variance-gradient
	$(CPB_V13_VARIANCE_GRADIENT_TEST) --settings $(CPB_V13_ROOT)/config/v13.conf
$(BUILD_DIR)/cpb_v13_variance_gradient_test.o: $(CPB_V13_ROOT)/tests/v13_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V13_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V13_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v13_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v13_variance_gradient_test.d

.PHONY: build-cpb-v13-temporal-pooling test-cpb-v13-temporal-pooling
build-cpb-v13-temporal-pooling: $(CPB_V13_TEMPORAL_POOLING_TEST)
test-cpb-v13-temporal-pooling: build-cpb-v13-temporal-pooling
	$(CPB_V13_TEMPORAL_POOLING_TEST) --settings $(CPB_V13_ROOT)/config/v13.conf
$(BUILD_DIR)/cpb_v13_temporal_pooling_test.o: $(CPB_V13_ROOT)/tests/v13_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V13_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V13_TEMPORAL_POOLING_TEST): $(BUILD_DIR)/cpb_v13_temporal_pooling_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v13_temporal_pooling_test.d
