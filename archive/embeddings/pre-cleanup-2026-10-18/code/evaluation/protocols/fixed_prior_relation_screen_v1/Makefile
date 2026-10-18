# Separate fixed-prior v17 screen; frozen v14/v15/v16 files are not modified.
BUILD_DIR ?= /opt/cuwacunu_embedding/build/rpb-fixed-prior-relation-screen
include Makefile
.DEFAULT_GOAL := fixed-prior-relation-screen
FIXED_SCREEN_ROOT := code/evaluation/protocols/fixed_prior_relation_screen_v1
FIXED_SCREEN_OBJECT_DIR := $(OBJECT_DIR)/protocols/fixed_prior_relation_screen_v1
FIXED_SCREEN_BIN := $(BUILD_DIR)/embedding_fixed_prior_relation_screen
FIXED_SCREEN_RPB_HEADERS := $(addprefix $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/,config.h types.h encoder.h tokenization.h masking.h objectives.h model.h visible_difference.h preprocessing.h training_utils.h context_deletion.h temporal_relation_bank.h partitioned_temporal_relation.h fixed_prior_temporal_relation.h)
FIXED_SCREEN_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
FIXED_SCREEN_SHARED_SOURCES := $(addprefix code/shared/src/,$(addsuffix .cpp,$(FIXED_SCREEN_SHARED_NAMES)))
FIXED_SCREEN_INPUTS := $(sort Makefile $(FIXED_SCREEN_ROOT)/Makefile $(FIXED_SCREEN_ROOT)/fixed_prior_relation_screen_main.cpp $(FIXED_SCREEN_ROOT)/prepare_screen.py $(FIXED_SCREEN_ROOT)/check_saved_stdlib.py code/evaluation/protocols/architecture_screen_v1/retained_input.h code/evaluation/protocols/architecture_screen_v1/Makefile code/evaluation/cards/fixed_prior_relation_screen_v1.md code/evaluation/include/frozen_role_guard.h $(FIXED_SCREEN_RPB_HEADERS) $(wildcard code/shared/include/embedding/shared/*.h) $(FIXED_SCREEN_SHARED_SOURCES) $(RPB_ROOT)/tests/fixed_prior_temporal_relation_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h code/shared/tests/shared_test_support.h code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/evaluation/protocols/visible_difference_v1/reader-v2/saved_cpu_math.py code/evaluation/protocols/visible_difference_v1/reader-v2/legacy_implementation.py $(SDK_PROVENANCE_INPUTS))
FIXED_SCREEN_SOURCE_ID := $(shell sha256sum $(FIXED_SCREEN_INPUTS) | sha256sum | cut -d ' ' -f 1)
FIXED_SCREEN_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests -Icode/evaluation/include -DFIXED_PRIOR_RELATION_SCREEN_SOURCE_ID=\"$(FIXED_SCREEN_SOURCE_ID)\" -DARCHITECTURE_SCREEN_SOURCE_ID=\"$(FIXED_SCREEN_SOURCE_ID)\" -DRPB_SOURCE_ID=\"$(FIXED_SCREEN_SOURCE_ID)\"
# These nine objects were compiled with the same root COMMON_CPPFLAGS/CXXFLAGS,
# without RPB/architecture macros. Their actual source and headers are above.
# An absent artifact fails here; this protocol never rebuilds the old objects.
FIXED_SCREEN_SHARED_DIR := /opt/cuwacunu_embedding/build/rpb-architecture-screen/code/protocols/architecture_screen_v1/shared
FIXED_SCREEN_SHARED_OBJECTS := $(addprefix $(FIXED_SCREEN_SHARED_DIR)/,$(addsuffix .o,$(FIXED_SCREEN_SHARED_NAMES)))

.PHONY: fixed-prior-relation-screen print-fixed-prior-relation-screen-sources test-fixed-prior-relation-screen-v17 test-fixed-prior-relation-screen-runner
fixed-prior-relation-screen: $(FIXED_SCREEN_BIN)
print-fixed-prior-relation-screen-sources:
	@printf '%s\n' $(FIXED_SCREEN_INPUTS)
$(FIXED_SCREEN_OBJECT_DIR)/fixed_prior_relation_screen_main.o: $(FIXED_SCREEN_ROOT)/fixed_prior_relation_screen_main.cpp $(FIXED_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FIXED_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(FIXED_SCREEN_SHARED_OBJECTS):
	@printf '%s\n' 'Required frozen generic shared object is absent: $@' >&2; exit 1
$(FIXED_SCREEN_BIN): $(FIXED_SCREEN_OBJECT_DIR)/fixed_prior_relation_screen_main.o $(FIXED_SCREEN_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(FIXED_SCREEN_OBJECT_DIR)/fixed_prior_temporal_relation_test.o: $(RPB_ROOT)/tests/fixed_prior_temporal_relation_test.cpp $(FIXED_SCREEN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FIXED_SCREEN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(FIXED_SCREEN_OBJECT_DIR)/fixed_prior_temporal_relation_test: $(FIXED_SCREEN_OBJECT_DIR)/fixed_prior_temporal_relation_test.o $(FIXED_SCREEN_SHARED_DIR)/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-fixed-prior-relation-screen-v17: $(FIXED_SCREEN_OBJECT_DIR)/fixed_prior_temporal_relation_test
	$(FIXED_SCREEN_OBJECT_DIR)/fixed_prior_temporal_relation_test
test-fixed-prior-relation-screen-runner: $(FIXED_SCREEN_BIN)
	@test -n "$(ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set ENGINEERING_OUTPUT to an absent exclusive fixture directory.' >&2; exit 1; }
	$(FIXED_SCREEN_BIN) --engineering "$(ENGINEERING_OUTPUT)"
-include $(wildcard $(FIXED_SCREEN_OBJECT_DIR)/*.d)
