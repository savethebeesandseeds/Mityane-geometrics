# Copyright (c) 2026 Waajacu. See LICENSE.
# Included by the authoritative repository-root Makefile. No container lifecycle.
CPB_V15_OBJECTIVE_ROOT := code/embeddings/evaluation/protocols/cpb_v15_objective_v1
CPB_V15_OBJECTIVE_CPPFLAGS := -I$(CPB_V15_OBJECTIVE_ROOT)/deps/include -I$(CPB_V15_OBJECTIVE_ROOT) -I$(CPB_V15_OBJECTIVE_ROOT)/include -Icode/embeddings/encoders/causal_patch_bottleneck_mae/versions/v4/include -Icode/embeddings/encoders/causal_patch_bottleneck_mae/versions/v15/include $(COMMON_CPPFLAGS)
CPB_V15_OBJECTIVE_BINARIES := $(BUILD_DIR)/cpb_v15_objective_control_train $(BUILD_DIR)/cpb_v15_objective_candidate_train $(BUILD_DIR)/cpb_v15_objective_control_extract $(BUILD_DIR)/cpb_v15_objective_candidate_extract $(BUILD_DIR)/cpb_v15_objective_control_confirmation_extract $(BUILD_DIR)/cpb_v15_objective_candidate_confirmation_extract $(BUILD_DIR)/cpb_v15_objective_market_read $(BUILD_DIR)/cpb_v15_objective_synthetic_read $(BUILD_DIR)/cpb_v15_objective_target_prepare $(BUILD_DIR)/cpb_v15_objective_confirmation_read
CPB_V15_OBJECTIVE_HEAD_OBJECTS := $(BUILD_DIR)/cpb_v15_objective_head_data.o $(BUILD_DIR)/cpb_v15_objective_head_feature_harness.o $(BUILD_DIR)/cpb_v15_objective_head_fixed_feature_readouts.o
.PHONY: build-cpb-v15-experiment build-cpb-v15-objective-preparation test-cpb-v15-objective-preparation
build-cpb-v15-experiment: $(CPB_V15_OBJECTIVE_BINARIES)
build-cpb-v15-objective-preparation: $(BUILD_DIR)/cpb_v15_objective_target_prepare
test-cpb-v15-objective-preparation:
	bash code/scripts/node.sh --test $(CPB_V15_OBJECTIVE_ROOT)/prepare.test.mjs $(CPB_V15_OBJECTIVE_ROOT)/run.test.mjs
define CPB_V15_OBJECTIVE_GUARD
	test -f /.dockerenv && test "$(abspath $(BUILD_DIR))" = "/opt/mityane-geometrics/embeddings/build/cpb-v15-objective-001"
endef
$(BUILD_DIR)/cpb_v15_objective_control_train.o: $(CPB_V15_OBJECTIVE_ROOT)/train_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_control_train: $(BUILD_DIR)/cpb_v15_objective_control_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_candidate_train.o: $(CPB_V15_OBJECTIVE_ROOT)/train_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) -DCPB_OBJECTIVE_CANDIDATE=1 $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_candidate_train: $(BUILD_DIR)/cpb_v15_objective_candidate_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_control_extract.o: $(CPB_V15_OBJECTIVE_ROOT)/extract_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_control_extract: $(BUILD_DIR)/cpb_v15_objective_control_extract.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_candidate_extract.o: $(CPB_V15_OBJECTIVE_ROOT)/extract_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) -DCPB_OBJECTIVE_CANDIDATE=1 $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_candidate_extract: $(BUILD_DIR)/cpb_v15_objective_candidate_extract.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_control_confirmation_extract.o: $(CPB_V15_OBJECTIVE_ROOT)/confirmation_extract_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_control_confirmation_extract: $(BUILD_DIR)/cpb_v15_objective_control_confirmation_extract.o $(BUILD_DIR)/cpb_v15_objective_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_candidate_confirmation_extract.o: $(CPB_V15_OBJECTIVE_ROOT)/confirmation_extract_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) -DCPB_OBJECTIVE_CANDIDATE=1 $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_candidate_confirmation_extract: $(BUILD_DIR)/cpb_v15_objective_candidate_confirmation_extract.o $(BUILD_DIR)/cpb_v15_objective_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_market_read.o: $(CPB_V15_OBJECTIVE_ROOT)/market_read_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_market_read: $(BUILD_DIR)/cpb_v15_objective_market_read.o $(BUILD_DIR)/cpb_v15_objective_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_synthetic_read.o: $(CPB_V15_OBJECTIVE_ROOT)/synthetic_read_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_synthetic_read: $(BUILD_DIR)/cpb_v15_objective_synthetic_read.o $(CPB_V15_OBJECTIVE_HEAD_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_target_prepare.o: $(CPB_V15_OBJECTIVE_ROOT)/target_prepare_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_target_prepare: $(BUILD_DIR)/cpb_v15_objective_target_prepare.o $(BUILD_DIR)/cpb_v15_objective_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_confirmation_read.o: $(CPB_V15_OBJECTIVE_ROOT)/confirmation_read_main.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_confirmation_read: $(BUILD_DIR)/cpb_v15_objective_confirmation_read.o $(BUILD_DIR)/cpb_v15_objective_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_v15_objective_fixed_regression.o: $(CPB_V15_OBJECTIVE_ROOT)/deps/src/fixed_regression.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v15_objective_head_%.o: $(CPB_V15_OBJECTIVE_ROOT)/deps/src/%.cpp
	$(CPB_V15_OBJECTIVE_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V15_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
-include $(wildcard $(BUILD_DIR)/cpb_v15_objective_*.d)
