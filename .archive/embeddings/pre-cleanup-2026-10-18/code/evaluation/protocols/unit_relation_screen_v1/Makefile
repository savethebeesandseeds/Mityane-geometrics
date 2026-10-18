# Separate unit-relation v18 screen; every prior source and model stays frozen.
BUILD_DIR ?= /opt/cuwacunu_embedding/build/rpb-unit-relation-screen
include Makefile
.DEFAULT_GOAL := unit-relation-screen
UNIT_SCREEN_ROOT := code/evaluation/protocols/unit_relation_screen_v1
UNIT_SCREEN_OBJECT_DIR := $(OBJECT_DIR)/protocols/unit_relation_screen_v1
UNIT_SCREEN_BIN := $(BUILD_DIR)/embedding_unit_relation_screen
UNIT_SCREEN_RPB_HEADERS := $(addprefix $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/,config.h types.h encoder.h tokenization.h masking.h objectives.h model.h visible_difference.h preprocessing.h training_utils.h context_deletion.h temporal_relation_bank.h partitioned_temporal_relation.h fixed_prior_temporal_relation.h unit_temporal_relation.h)
UNIT_SCREEN_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
UNIT_SCREEN_SHARED_SOURCES := $(addprefix code/shared/src/,$(addsuffix .cpp,$(UNIT_SCREEN_SHARED_NAMES)))
UNIT_SCREEN_INPUTS := $(sort Makefile $(UNIT_SCREEN_ROOT)/Makefile $(UNIT_SCREEN_ROOT)/unit_relation_screen_main.cpp $(UNIT_SCREEN_ROOT)/prepare_screen.py $(UNIT_SCREEN_ROOT)/check_saved_stdlib.py $(UNIT_SCREEN_ROOT)/retained_input.h code/evaluation/protocols/architecture_screen_v1/Makefile code/evaluation/cards/unit_relation_screen_v1.md code/evaluation/include/frozen_role_guard.h $(UNIT_SCREEN_RPB_HEADERS) $(wildcard code/shared/include/embedding/shared/*.h) $(UNIT_SCREEN_SHARED_SOURCES) $(RPB_ROOT)/tests/unit_temporal_relation_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h code/shared/tests/shared_test_support.h code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/evaluation/protocols/visible_difference_v1/reader-v2/saved_cpu_math.py code/evaluation/protocols/visible_difference_v1/reader-v2/legacy_implementation.py $(SDK_PROVENANCE_INPUTS))
UNIT_SCREEN_SOURCE_ID := $(shell sha256sum $(UNIT_SCREEN_INPUTS) | sha256sum | cut -d ' ' -f 1)
UNIT_SCREEN_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests -Icode/evaluation/include -DUNIT_RELATION_SCREEN_SOURCE_ID=\"$(UNIT_SCREEN_SOURCE_ID)\" -DARCHITECTURE_SCREEN_SOURCE_ID=\"$(UNIT_SCREEN_SOURCE_ID)\" -DRPB_SOURCE_ID=\"$(UNIT_SCREEN_SOURCE_ID)\"
# These nine objects were compiled with the same root COMMON_CPPFLAGS/CXXFLAGS,
# without RPB/architecture macros. Their actual source and headers are above.
# An absent artifact fails here; this protocol never rebuilds the old objects.
UNIT_SCREEN_SHARED_DIR := /opt/cuwacunu_embedding/build/rpb-architecture-screen/code/protocols/architecture_screen_v1/shared
UNIT_SCREEN_SHARED_OBJECTS := $(addprefix $(UNIT_SCREEN_SHARED_DIR)/,$(addsuffix .o,$(UNIT_SCREEN_SHARED_NAMES)))

.PHONY: unit-relation-screen print-unit-relation-screen-sources test-unit-relation-screen-v18 test-unit-relation-screen-runner
unit-relation-screen: $(UNIT_SCREEN_BIN)
print-unit-relation-screen-sources:
	@printf '%s\n' $(UNIT_SCREEN_INPUTS)
$(UNIT_SCREEN_OBJECT_DIR)/unit_relation_screen_main.o: $(UNIT_SCREEN_ROOT)/unit_relation_screen_main.cpp $(UNIT_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(UNIT_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(UNIT_SCREEN_SHARED_OBJECTS):
	@printf '%s\n' 'Required frozen generic shared object is absent: $@' >&2; exit 1
$(UNIT_SCREEN_BIN): $(UNIT_SCREEN_OBJECT_DIR)/unit_relation_screen_main.o $(UNIT_SCREEN_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(UNIT_SCREEN_OBJECT_DIR)/unit_temporal_relation_test.o: $(RPB_ROOT)/tests/unit_temporal_relation_test.cpp $(UNIT_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(UNIT_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(UNIT_SCREEN_OBJECT_DIR)/unit_temporal_relation_test: $(UNIT_SCREEN_OBJECT_DIR)/unit_temporal_relation_test.o $(UNIT_SCREEN_SHARED_DIR)/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-unit-relation-screen-v18: $(UNIT_SCREEN_OBJECT_DIR)/unit_temporal_relation_test
	$(UNIT_SCREEN_OBJECT_DIR)/unit_temporal_relation_test
test-unit-relation-screen-runner: $(UNIT_SCREEN_BIN)
	@test -n "$(ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set ENGINEERING_OUTPUT to an absent exclusive fixture directory.' >&2; exit 1; }
	$(UNIT_SCREEN_BIN) --engineering "$(ENGINEERING_OUTPUT)"
-include $(wildcard $(UNIT_SCREEN_OBJECT_DIR)/*.d)
