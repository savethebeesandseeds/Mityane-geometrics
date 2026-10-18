# Separate two-task multiband comparison; prior producers stay frozen.
BUILD_DIR ?= /opt/cuwacunu_embedding/build/rpb-multiband-screen
include Makefile
.DEFAULT_GOAL := multiband-screen
MB_ROOT := code/evaluation/protocols/multiband_screen_v1
MB_DATA := code/evaluation/benchmarks/two_component
MB_OBJECT_DIR := $(OBJECT_DIR)/protocols/multiband_screen_v1
MB_BIN := $(BUILD_DIR)/embedding_multiband_screen
MB_WRITER := $(BUILD_DIR)/embedding_multiband_quality_data
MB_CARD := code/evaluation/cards/multiband_screen_v1.md
MB_RPB_HEADERS := $(addprefix $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/,config.h types.h encoder.h tokenization.h masking.h objectives.h model.h visible_difference.h preprocessing.h training_utils.h context_deletion.h temporal_relation_bank.h partitioned_temporal_relation.h fixed_prior_temporal_relation.h unit_temporal_relation.h spectral_relation_bank.h spectral_temporal_relation.h)
MB_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
MB_SHARED_SOURCES := $(addprefix code/shared/src/,$(addsuffix .cpp,$(MB_SHARED_NAMES)))
MB_INPUTS := $(sort Makefile $(MB_ROOT)/Makefile $(MB_ROOT)/producer_main.cpp $(MB_ROOT)/retained_input.h $(MB_ROOT)/write_quality_data.cpp $(MB_ROOT)/prepare_screen.py $(MB_ROOT)/check_saved_stdlib.py $(MB_ROOT)/admit.sh $(MB_ROOT)/run_screen.sh code/scripts/task.sh code/evaluation/protocols/architecture_screen_v1/Makefile $(MB_CARD) code/evaluation/include/frozen_role_guard.h $(MB_RPB_HEADERS) $(wildcard code/shared/include/embedding/shared/*.h) $(MB_SHARED_SOURCES) $(RPB_ROOT)/tests/unit_temporal_relation_test.cpp $(RPB_ROOT)/tests/spectral_temporal_relation_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h code/shared/tests/shared_test_support.h $(MB_DATA)/two_component.h $(MB_DATA)/two_component.cpp $(MB_DATA)/observed_information.h $(MB_DATA)/observed_information.cpp code/evaluation/cards/two_component_information_v1.md doc/results/two_component_information_v1.json code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/evaluation/protocols/visible_difference_v1/reader-v2/saved_cpu_math.py code/evaluation/protocols/visible_difference_v1/reader-v2/legacy_implementation.py $(SDK_PROVENANCE_INPUTS))
MB_SOURCE_ID := $(shell sha256sum $(MB_INPUTS) | sha256sum | cut -d ' ' -f 1)
MB_CARD_SHA := $(shell sha256sum $(MB_CARD) | cut -d ' ' -f 1)
MB_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests -I$(MB_DATA) -Icode/evaluation/include -DMULTIBAND_SCREEN_SOURCE_ID=\"$(MB_SOURCE_ID)\" -DMULTIBAND_SCREEN_CARD_SHA256=\"$(MB_CARD_SHA)\" -DUNIT_RELATION_SCREEN_SOURCE_ID=\"$(MB_SOURCE_ID)\" -DARCHITECTURE_SCREEN_SOURCE_ID=\"$(MB_SOURCE_ID)\" -DRPB_SOURCE_ID=\"$(MB_SOURCE_ID)\"
# Generic objects have no encoder/protocol macros and use the same pinned SDK.
# Reuse their artifacts; absent objects reject instead of rebuilding old code.
MB_SHARED_DIR := /opt/cuwacunu_embedding/build/rpb-architecture-screen/code/protocols/architecture_screen_v1/shared
MB_SHARED_OBJECTS := $(addprefix $(MB_SHARED_DIR)/,$(addsuffix .o,$(MB_SHARED_NAMES)))

.PHONY: multiband-screen multiband-quality-data print-multiband-screen-sources test-multiband-screen-v18 test-multiband-screen-v19 test-multiband-screen-runner test-multiband-quality-data
multiband-screen: $(MB_BIN)
multiband-quality-data: $(MB_WRITER)
print-multiband-screen-sources:
	@printf '%s\n' $(MB_INPUTS)
$(MB_OBJECT_DIR)/producer_main.o: $(MB_ROOT)/producer_main.cpp $(MB_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MB_OBJECT_DIR)/write_quality_data.o: $(MB_ROOT)/write_quality_data.cpp $(MB_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MB_OBJECT_DIR)/two_component.o: $(MB_DATA)/two_component.cpp $(MB_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MB_SHARED_OBJECTS):
	@printf '%s\n' 'Required frozen generic shared object is absent: $@' >&2; exit 1
$(MB_BIN): $(MB_OBJECT_DIR)/producer_main.o $(MB_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(MB_WRITER): $(MB_OBJECT_DIR)/write_quality_data.o $(MB_OBJECT_DIR)/two_component.o $(MB_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(MB_OBJECT_DIR)/unit_temporal_relation_test.o: $(RPB_ROOT)/tests/unit_temporal_relation_test.cpp $(MB_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MB_OBJECT_DIR)/spectral_temporal_relation_test.o: $(RPB_ROOT)/tests/spectral_temporal_relation_test.cpp $(MB_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MB_OBJECT_DIR)/unit_temporal_relation_test: $(MB_OBJECT_DIR)/unit_temporal_relation_test.o $(MB_SHARED_DIR)/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(MB_OBJECT_DIR)/spectral_temporal_relation_test: $(MB_OBJECT_DIR)/spectral_temporal_relation_test.o $(MB_SHARED_DIR)/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-multiband-screen-v18: $(MB_OBJECT_DIR)/unit_temporal_relation_test
	$(MB_OBJECT_DIR)/unit_temporal_relation_test
test-multiband-screen-v19: $(MB_OBJECT_DIR)/spectral_temporal_relation_test
	$(MB_OBJECT_DIR)/spectral_temporal_relation_test
test-multiband-screen-runner: $(MB_BIN)
	@test -n "$(ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set ENGINEERING_OUTPUT to an absent fixture leaf.' >&2; exit 1; }
	$(MB_BIN) --engineering "$(ENGINEERING_OUTPUT)"
test-multiband-quality-data: $(MB_WRITER)
	@test -n "$(DATA_ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set DATA_ENGINEERING_OUTPUT to an absent data-only fixture leaf.' >&2; exit 1; }
	$(MB_WRITER) --engineering "$(DATA_ENGINEERING_OUTPUT)"
-include $(wildcard $(MB_OBJECT_DIR)/*.d)
