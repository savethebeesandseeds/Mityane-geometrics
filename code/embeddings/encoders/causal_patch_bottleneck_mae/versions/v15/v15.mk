# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Standalone first-party closure: only this new family, plus Torch/STL.
CPB_V15_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v15
CPB_V15_CPPFLAGS := -I$(CPB_V15_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V15_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v15
CPB_V15_VARIANCE_GRADIENT_TEST := $(BUILD_DIR)/test_cpb_v15_variance_gradient
CPB_V15_OBJECTIVE_TEST := $(BUILD_DIR)/test_cpb_v15_objective
CPB_V15_CONTROL_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v4
CPB_V15_ENGINEERING_OUTPUT ?=
CPB_V15_ENGINEERING_SEED ?=
.PHONY: build-cpb-v15-engineering test-cpb-v15 build-cpb-v15-variance-gradient test-cpb-v15-variance-gradient build-cpb-v15-objective test-cpb-v15-objective
build-cpb-v15-engineering: $(CPB_V15_TEST) $(CPB_V15_VARIANCE_GRADIENT_TEST) $(CPB_V15_OBJECTIVE_TEST)
test-cpb-v15: build-cpb-v15-engineering
	@test -n "$(CPB_V15_ENGINEERING_OUTPUT)" -a -n "$(CPB_V15_ENGINEERING_SEED)"
	$(CPB_V15_TEST) --settings $(CPB_V15_ROOT)/config/v15.conf --output "$(CPB_V15_ENGINEERING_OUTPUT)" --seed "$(CPB_V15_ENGINEERING_SEED)"
$(BUILD_DIR)/causal_patch_bottleneck_mae_v15_test.o: $(CPB_V15_ROOT)/tests/v15_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V15_TEST): $(BUILD_DIR)/causal_patch_bottleneck_mae_v15_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_patch_bottleneck_mae_v15_test.d

build-cpb-v15-variance-gradient: $(CPB_V15_VARIANCE_GRADIENT_TEST)
test-cpb-v15-variance-gradient: build-cpb-v15-variance-gradient
	$(CPB_V15_VARIANCE_GRADIENT_TEST) --settings $(CPB_V15_ROOT)/config/v15.conf
$(BUILD_DIR)/cpb_v15_variance_gradient_test.o: $(CPB_V15_ROOT)/tests/v15_variance_gradient_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V15_VARIANCE_GRADIENT_TEST): $(BUILD_DIR)/cpb_v15_variance_gradient_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v15_variance_gradient_test.d

build-cpb-v15-objective: $(CPB_V15_OBJECTIVE_TEST)
test-cpb-v15-objective: build-cpb-v15-objective
	$(CPB_V15_OBJECTIVE_TEST) --settings $(CPB_V15_ROOT)/config/v15.conf --control-settings $(CPB_V15_CONTROL_ROOT)/config/v4.conf --seed 945101
$(BUILD_DIR)/cpb_v15_objective_test.o: $(CPB_V15_ROOT)/tests/v15_objective_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_CPPFLAGS) -I$(CPB_V15_CONTROL_ROOT)/include $(CXXFLAGS) -c $< -o $@
$(CPB_V15_OBJECTIVE_TEST): $(BUILD_DIR)/cpb_v15_objective_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v15_objective_test.d
