# Fresh v17 instance confirmation; every earlier screen remains frozen.
BUILD_DIR ?= /opt/cuwacunu_embedding/build/rpb-fixed-prior-fresh-confirmation
include Makefile
.DEFAULT_GOAL := fixed-prior-fresh-confirmation
FRESH_CONFIRM_ROOT := code/evaluation/protocols/fixed_prior_fresh_confirmation_v1
FRESH_CONFIRM_OBJECT_DIR := $(OBJECT_DIR)/protocols/fixed_prior_fresh_confirmation_v1
FRESH_CONFIRM_BIN := $(BUILD_DIR)/embedding_fixed_prior_fresh_confirmation
FRESH_CONFIRM_DATA_BIN := $(BUILD_DIR)/embedding_write_fresh_tempo3_data
FRESH_CONFIRM_RPB_HEADERS := $(addprefix $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/,config.h types.h encoder.h tokenization.h masking.h objectives.h model.h visible_difference.h preprocessing.h training_utils.h context_deletion.h temporal_relation_bank.h partitioned_temporal_relation.h fixed_prior_temporal_relation.h)
FRESH_CONFIRM_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
FRESH_CONFIRM_SHARED_SOURCES := $(addprefix code/shared/src/,$(addsuffix .cpp,$(FRESH_CONFIRM_SHARED_NAMES)))
FRESH_CONFIRM_INPUTS := $(sort Makefile $(FRESH_CONFIRM_ROOT)/Makefile $(FRESH_CONFIRM_ROOT)/fresh_confirmation_main.cpp $(FRESH_CONFIRM_ROOT)/prepare_confirmation.py $(FRESH_CONFIRM_ROOT)/check_saved_stdlib.py $(FRESH_CONFIRM_ROOT)/retained_input.h $(FRESH_CONFIRM_ROOT)/fresh_tempo3_data.h $(FRESH_CONFIRM_ROOT)/write_fresh_tempo3_data.cpp code/evaluation/benchmarks/structured_hard_timing/structured_hard_timing.h code/evaluation/benchmarks/structured_hard_timing/structured_hard_timing.cpp code/evaluation/protocols/architecture_screen_v1/Makefile code/evaluation/cards/fixed_prior_fresh_confirmation_v1.md code/evaluation/include/frozen_role_guard.h $(FRESH_CONFIRM_RPB_HEADERS) $(wildcard code/shared/include/embedding/shared/*.h) $(FRESH_CONFIRM_SHARED_SOURCES) $(RPB_ROOT)/tests/fixed_prior_temporal_relation_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h code/shared/tests/shared_test_support.h code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/evaluation/protocols/visible_difference_v1/reader-v2/saved_cpu_math.py code/evaluation/protocols/visible_difference_v1/reader-v2/legacy_implementation.py $(SDK_PROVENANCE_INPUTS))
FRESH_CONFIRM_SOURCE_ID := $(shell sha256sum $(FRESH_CONFIRM_INPUTS) | sha256sum | cut -d ' ' -f 1)
FRESH_CONFIRM_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests -Icode/evaluation/include -Icode/evaluation/benchmarks/structured_hard_timing -DFIXED_PRIOR_FRESH_CONFIRMATION_SOURCE_ID=\"$(FRESH_CONFIRM_SOURCE_ID)\" -DFIXED_PRIOR_RELATION_SCREEN_SOURCE_ID=\"$(FRESH_CONFIRM_SOURCE_ID)\" -DARCHITECTURE_SCREEN_SOURCE_ID=\"$(FRESH_CONFIRM_SOURCE_ID)\" -DRPB_SOURCE_ID=\"$(FRESH_CONFIRM_SOURCE_ID)\"
# These nine objects were compiled with the same root COMMON_CPPFLAGS/CXXFLAGS,
# without RPB/architecture macros. Their actual source and headers are above.
# An absent artifact fails here; this protocol never rebuilds the old objects.
FRESH_CONFIRM_SHARED_DIR := /opt/cuwacunu_embedding/build/rpb-architecture-screen/code/protocols/architecture_screen_v1/shared
FRESH_CONFIRM_SHARED_OBJECTS := $(addprefix $(FRESH_CONFIRM_SHARED_DIR)/,$(addsuffix .o,$(FRESH_CONFIRM_SHARED_NAMES)))

.PHONY: fixed-prior-fresh-data test-fixed-prior-fresh-data-engineering fixed-prior-fresh-confirmation print-fixed-prior-fresh-confirmation-sources test-fixed-prior-fresh-confirmation-v17 test-fixed-prior-fresh-confirmation-runner
fixed-prior-fresh-confirmation: $(FRESH_CONFIRM_BIN)
fixed-prior-fresh-data: $(FRESH_CONFIRM_DATA_BIN)
print-fixed-prior-fresh-confirmation-sources:
	@printf '%s\n' $(FRESH_CONFIRM_INPUTS)
$(FRESH_CONFIRM_OBJECT_DIR)/fresh_confirmation_main.o: $(FRESH_CONFIRM_ROOT)/fresh_confirmation_main.cpp $(FRESH_CONFIRM_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FRESH_CONFIRM_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(FRESH_CONFIRM_OBJECT_DIR)/write_fresh_tempo3_data.o: $(FRESH_CONFIRM_ROOT)/write_fresh_tempo3_data.cpp $(FRESH_CONFIRM_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FRESH_CONFIRM_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(FRESH_CONFIRM_OBJECT_DIR)/structured_hard_timing.o: code/evaluation/benchmarks/structured_hard_timing/structured_hard_timing.cpp $(FRESH_CONFIRM_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FRESH_CONFIRM_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(FRESH_CONFIRM_DATA_BIN): $(FRESH_CONFIRM_OBJECT_DIR)/write_fresh_tempo3_data.o $(FRESH_CONFIRM_OBJECT_DIR)/structured_hard_timing.o $(FRESH_CONFIRM_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-fixed-prior-fresh-data-engineering: $(FRESH_CONFIRM_DATA_BIN)
	@test -n "$(DATA_ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set DATA_ENGINEERING_OUTPUT to an absent exclusive fixture directory.' >&2; exit 1; }
	$(FRESH_CONFIRM_DATA_BIN) --engineering "$(DATA_ENGINEERING_OUTPUT)"
$(FRESH_CONFIRM_SHARED_OBJECTS):
	@printf '%s\n' 'Required frozen generic shared object is absent: $@' >&2; exit 1
$(FRESH_CONFIRM_BIN): $(FRESH_CONFIRM_OBJECT_DIR)/fresh_confirmation_main.o $(FRESH_CONFIRM_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(FRESH_CONFIRM_OBJECT_DIR)/fixed_prior_temporal_relation_test.o: $(RPB_ROOT)/tests/fixed_prior_temporal_relation_test.cpp $(FRESH_CONFIRM_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FRESH_CONFIRM_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(FRESH_CONFIRM_OBJECT_DIR)/fixed_prior_temporal_relation_test: $(FRESH_CONFIRM_OBJECT_DIR)/fixed_prior_temporal_relation_test.o $(FRESH_CONFIRM_SHARED_DIR)/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-fixed-prior-fresh-confirmation-v17: $(FRESH_CONFIRM_OBJECT_DIR)/fixed_prior_temporal_relation_test
	$(FRESH_CONFIRM_OBJECT_DIR)/fixed_prior_temporal_relation_test
test-fixed-prior-fresh-confirmation-runner: $(FRESH_CONFIRM_BIN)
	@test -n "$(ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set ENGINEERING_OUTPUT to an absent exclusive fixture directory.' >&2; exit 1; }
	$(FRESH_CONFIRM_BIN) --engineering "$(ENGINEERING_OUTPUT)"
-include $(wildcard $(FRESH_CONFIRM_OBJECT_DIR)/*.d)
