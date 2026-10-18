# Additive lightweight screen; no historical target or source is modified.
include Makefile
.DEFAULT_GOAL := architecture-screen
ARCH_SCREEN_ROOT := code/evaluation/protocols/architecture_screen_v1
ARCH_SCREEN_OBJECT_DIR := $(OBJECT_DIR)/protocols/architecture_screen_v1
ARCH_SCREEN_BIN := $(BUILD_DIR)/embedding_architecture_screen
ARCH_SCREEN_RPB_HEADERS := $(addprefix $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/,config.h types.h encoder.h tokenization.h masking.h objectives.h model.h visible_difference.h preprocessing.h training_utils.h context_deletion.h temporal_relation_bank.h temporal_dynamics_objective.h)
ARCH_SCREEN_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
ARCH_SCREEN_SHARED_SOURCES := $(addprefix code/shared/src/,$(addsuffix .cpp,$(ARCH_SCREEN_SHARED_NAMES)))
ARCH_SCREEN_SHARED_HEADERS := $(wildcard code/shared/include/embedding/shared/*.h)
ARCH_SCREEN_INPUTS := $(sort Makefile $(ARCH_SCREEN_ROOT)/Makefile $(ARCH_SCREEN_ROOT)/architecture_screen_main.cpp $(ARCH_SCREEN_ROOT)/retained_input.h $(ARCH_SCREEN_ROOT)/prepare_screen.py $(ARCH_SCREEN_ROOT)/check_saved.py code/evaluation/cards/architecture_screen_v1.md code/evaluation/include/frozen_role_guard.h $(ARCH_SCREEN_RPB_HEADERS) $(wildcard code/shared/include/embedding/shared/*.h) $(ARCH_SCREEN_SHARED_SOURCES) $(RPB_ROOT)/tests/temporal_relation_bank_test.cpp $(RPB_ROOT)/tests/temporal_dynamics_objective_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h code/shared/tests/shared_test_support.h $(SDK_PROVENANCE_INPUTS))
ARCH_SCREEN_SOURCE_ID := $(shell sha256sum $(ARCH_SCREEN_INPUTS) | sha256sum | cut -d ' ' -f 1)
ARCH_SCREEN_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests -Icode/evaluation/include -DARCHITECTURE_SCREEN_SOURCE_ID=\"$(ARCH_SCREEN_SOURCE_ID)\" -DRPB_SOURCE_ID=\"$(ARCH_SCREEN_SOURCE_ID)\"
ARCH_SCREEN_SHARED_OBJECTS := $(addprefix $(ARCH_SCREEN_OBJECT_DIR)/shared/,$(addsuffix .o,$(ARCH_SCREEN_SHARED_NAMES)))

.PHONY: architecture-screen print-architecture-screen-sources test-architecture-screen-v14 test-architecture-screen-v15 test-architecture-screen-runner
architecture-screen: $(ARCH_SCREEN_BIN)
print-architecture-screen-sources:
	@printf '%s\n' $(ARCH_SCREEN_INPUTS)
$(ARCH_SCREEN_OBJECT_DIR)/architecture_screen_main.o: $(ARCH_SCREEN_ROOT)/architecture_screen_main.cpp $(ARCH_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(ARCH_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(ARCH_SCREEN_OBJECT_DIR)/shared/%.o: code/shared/src/%.cpp $(ARCH_SCREEN_SHARED_HEADERS)
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(ARCH_SCREEN_BIN): $(ARCH_SCREEN_OBJECT_DIR)/architecture_screen_main.o $(ARCH_SCREEN_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(ARCH_SCREEN_OBJECT_DIR)/temporal_relation_bank_test.o: $(RPB_ROOT)/tests/temporal_relation_bank_test.cpp $(ARCH_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(ARCH_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(ARCH_SCREEN_OBJECT_DIR)/temporal_relation_bank_test: $(ARCH_SCREEN_OBJECT_DIR)/temporal_relation_bank_test.o $(ARCH_SCREEN_OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-architecture-screen-v14: $(ARCH_SCREEN_OBJECT_DIR)/temporal_relation_bank_test
	$(ARCH_SCREEN_OBJECT_DIR)/temporal_relation_bank_test
$(ARCH_SCREEN_OBJECT_DIR)/temporal_dynamics_objective_test.o: $(RPB_ROOT)/tests/temporal_dynamics_objective_test.cpp $(ARCH_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(ARCH_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(ARCH_SCREEN_OBJECT_DIR)/temporal_dynamics_objective_test: $(ARCH_SCREEN_OBJECT_DIR)/temporal_dynamics_objective_test.o $(ARCH_SCREEN_OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-architecture-screen-v15: $(ARCH_SCREEN_OBJECT_DIR)/temporal_dynamics_objective_test
	$(ARCH_SCREEN_OBJECT_DIR)/temporal_dynamics_objective_test
test-architecture-screen-runner: $(ARCH_SCREEN_BIN)
	@test -n "$(ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set ENGINEERING_OUTPUT to an absent exclusive fixture directory.' >&2; exit 1; }
	$(ARCH_SCREEN_BIN) --engineering "$(ENGINEERING_OUTPUT)"
-include $(wildcard $(ARCH_SCREEN_OBJECT_DIR)/*.d $(ARCH_SCREEN_OBJECT_DIR)/shared/*.d)
