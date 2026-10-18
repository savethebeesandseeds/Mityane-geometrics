# Separate fixed v16 screen; frozen v14/v15 producer files are not modified.
BUILD_DIR ?= /opt/cuwacunu_embedding/build/rpb-partitioned-relation-screen
include Makefile
.DEFAULT_GOAL := partitioned-relation-screen
PART_SCREEN_ROOT := code/evaluation/protocols/partitioned_relation_screen_v1
PART_SCREEN_OBJECT_DIR := $(OBJECT_DIR)/protocols/partitioned_relation_screen_v1
PART_SCREEN_BIN := $(BUILD_DIR)/embedding_partitioned_relation_screen
PART_SCREEN_RPB_HEADERS := $(addprefix $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/,config.h types.h encoder.h tokenization.h masking.h objectives.h model.h visible_difference.h preprocessing.h training_utils.h context_deletion.h temporal_relation_bank.h partitioned_temporal_relation.h)
PART_SCREEN_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
PART_SCREEN_SHARED_SOURCES := $(addprefix code/shared/src/,$(addsuffix .cpp,$(PART_SCREEN_SHARED_NAMES)))
PART_SCREEN_INPUTS := $(sort Makefile $(PART_SCREEN_ROOT)/Makefile $(PART_SCREEN_ROOT)/partitioned_relation_screen_main.cpp $(PART_SCREEN_ROOT)/prepare_screen.py $(PART_SCREEN_ROOT)/check_saved_stdlib.py code/evaluation/protocols/architecture_screen_v1/retained_input.h code/evaluation/protocols/architecture_screen_v1/Makefile code/evaluation/cards/partitioned_relation_screen_v1.md code/evaluation/include/frozen_role_guard.h $(PART_SCREEN_RPB_HEADERS) $(wildcard code/shared/include/embedding/shared/*.h) $(PART_SCREEN_SHARED_SOURCES) $(RPB_ROOT)/tests/partitioned_temporal_relation_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h code/shared/tests/shared_test_support.h code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/evaluation/protocols/visible_difference_v1/reader-v2/saved_cpu_math.py code/evaluation/protocols/visible_difference_v1/reader-v2/legacy_implementation.py $(SDK_PROVENANCE_INPUTS))
PART_SCREEN_SOURCE_ID := $(shell sha256sum $(PART_SCREEN_INPUTS) | sha256sum | cut -d ' ' -f 1)
PART_SCREEN_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests -Icode/evaluation/include -DARCHITECTURE_SCREEN_SOURCE_ID=\"$(PART_SCREEN_SOURCE_ID)\" -DRPB_SOURCE_ID=\"$(PART_SCREEN_SOURCE_ID)\"
# These nine objects were compiled with the same root COMMON_CPPFLAGS/CXXFLAGS,
# without RPB/architecture macros. Their actual source and headers are above.
# An absent artifact fails here; this protocol never rebuilds the old objects.
PART_SCREEN_SHARED_DIR := /opt/cuwacunu_embedding/build/rpb-architecture-screen/code/protocols/architecture_screen_v1/shared
PART_SCREEN_SHARED_OBJECTS := $(addprefix $(PART_SCREEN_SHARED_DIR)/,$(addsuffix .o,$(PART_SCREEN_SHARED_NAMES)))

.PHONY: partitioned-relation-screen print-partitioned-relation-screen-sources test-partitioned-relation-screen-v16 test-partitioned-relation-screen-runner
partitioned-relation-screen: $(PART_SCREEN_BIN)
print-partitioned-relation-screen-sources:
	@printf '%s\n' $(PART_SCREEN_INPUTS)
$(PART_SCREEN_OBJECT_DIR)/partitioned_relation_screen_main.o: $(PART_SCREEN_ROOT)/partitioned_relation_screen_main.cpp $(PART_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(PART_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(PART_SCREEN_SHARED_OBJECTS):
	@printf '%s\n' 'Required frozen generic shared object is absent: $@' >&2; exit 1
$(PART_SCREEN_BIN): $(PART_SCREEN_OBJECT_DIR)/partitioned_relation_screen_main.o $(PART_SCREEN_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(PART_SCREEN_OBJECT_DIR)/partitioned_temporal_relation_test.o: $(RPB_ROOT)/tests/partitioned_temporal_relation_test.cpp $(PART_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(PART_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(PART_SCREEN_OBJECT_DIR)/partitioned_temporal_relation_test: $(PART_SCREEN_OBJECT_DIR)/partitioned_temporal_relation_test.o $(PART_SCREEN_SHARED_DIR)/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-partitioned-relation-screen-v16: $(PART_SCREEN_OBJECT_DIR)/partitioned_temporal_relation_test
	$(PART_SCREEN_OBJECT_DIR)/partitioned_temporal_relation_test
test-partitioned-relation-screen-runner: $(PART_SCREEN_BIN)
	@test -n "$(ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set ENGINEERING_OUTPUT to an absent exclusive fixture directory.' >&2; exit 1; }
	$(PART_SCREEN_BIN) --engineering "$(ENGINEERING_OUTPUT)"
-include $(wildcard $(PART_SCREEN_OBJECT_DIR)/*.d)
