# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V16_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v16
CPB_V16_CPPFLAGS := -I$(CPB_V16_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V16_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v16
CPB_V16_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v16_variance_gradient
CPB_V16_TEMPORAL_OBJECTIVE_TEST := $(BUILD_DIR)/test_cpb_v16_temporal_objective
CPB_V16_CONTROL_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v4
CPB_V16_ENGINEERING_OUTPUT ?=
CPB_V16_ENGINEERING_SEED ?=
.PHONY: build-cpb-v16-engineering test-cpb-v16 build-cpb-v16-variance-gradient test-cpb-v16-variance-gradient build-cpb-v16-temporal-objective test-cpb-v16-temporal-objective
build-cpb-v16-engineering: $(CPB_V16_TEST) $(CPB_V16_VARIANCE_GRADIENT_TEST) $(CPB_V16_TEMPORAL_OBJECTIVE_TEST)
test-cpb-v16: build-cpb-v16-engineering
	@test -n "$(CPB_V16_ENGINEERING_OUTPUT)" -a -n "$(CPB_V16_ENGINEERING_SEED)"
	$(CPB_V16_TEST) --settings $(CPB_V16_ROOT)/config/v16.conf --output "$(CPB_V16_ENGINEERING_OUTPUT)" --seed "$(CPB_V16_ENGINEERING_SEED)"
$(BUILD_DIR)/causal_patch_bottleneck_mae_v16_test.o: $(CPB_V16_ROOT)/tests/v16_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V16_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V16_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v16_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v16_test.d

build-cpb-v16-variance-gradient: $(CPB_V16_VARIANCE_GRADIENT_TEST)
test-cpb-v16-variance-gradient: build-cpb-v16-variance-gradient
	$(CPB_V16_VARIANCE_GRADIENT_TEST) --settings $(CPB_V16_ROOT)/config/v16.conf
$(BUILD_DIR)/cpb_v16_variance_gradient_test.o: $(CPB_V16_ROOT)/tests/v16_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V16_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V16_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v16_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v16_variance_gradient_test.d

build-cpb-v16-temporal-objective: $(CPB_V16_TEMPORAL_OBJECTIVE_TEST)
test-cpb-v16-temporal-objective: build-cpb-v16-temporal-objective
	@test -n "$(CPB_V16_ENGINEERING_SEED)"
	$(CPB_V16_TEMPORAL_OBJECTIVE_TEST) --settings $(CPB_V16_ROOT)/config/v16.conf --control-settings $(CPB_V16_CONTROL_ROOT)/config/v4.conf --seed "$(CPB_V16_ENGINEERING_SEED)"
$(BUILD_DIR)/cpb_v16_temporal_objective_test.o: $(CPB_V16_ROOT)/tests/v16_temporal_objective_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V16_CPPFLAGS) -I$(CPB_V16_CONTROL_ROOT)/include $(CXXFLAGS) -c $< -o $@
$(CPB_V16_TEMPORAL_OBJECTIVE_TEST): $(BUILD_DIR)/cpb_v16_temporal_objective_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v16_temporal_objective_test.d
