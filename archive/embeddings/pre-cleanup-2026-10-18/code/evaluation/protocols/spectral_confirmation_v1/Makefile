# Separate unchanged-v19 fresh-source confirmation; earlier producers stay frozen.
BUILD_DIR ?= /opt/cuwacunu_embedding/build/rpb-spectral-confirmation
include Makefile
.DEFAULT_GOAL := spectral-confirmation
SC_ROOT := code/evaluation/protocols/spectral_confirmation_v1
SC_DATA := code/evaluation/benchmarks/two_component
SC_OBJECT_DIR := $(OBJECT_DIR)/protocols/spectral_confirmation_v1
SC_BIN := $(BUILD_DIR)/embedding_spectral_confirmation
SC_WRITER := $(BUILD_DIR)/embedding_spectral_confirmation_quality_data
SC_CARD := code/evaluation/cards/spectral_confirmation_v1.md
SC_RPB_HEADERS := $(addprefix $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/,config.h types.h encoder.h tokenization.h masking.h objectives.h model.h visible_difference.h preprocessing.h training_utils.h context_deletion.h temporal_relation_bank.h partitioned_temporal_relation.h fixed_prior_temporal_relation.h unit_temporal_relation.h spectral_relation_bank.h spectral_temporal_relation.h)
SC_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
SC_SHARED_SOURCES := $(addprefix code/shared/src/,$(addsuffix .cpp,$(SC_SHARED_NAMES)))
SC_INPUTS := $(sort Makefile $(SC_ROOT)/Makefile $(SC_ROOT)/producer_main.cpp $(SC_ROOT)/retained_input.h $(SC_ROOT)/write_quality_data.cpp $(SC_ROOT)/prepare_confirmation.py $(SC_ROOT)/check_saved_stdlib.py $(SC_ROOT)/admit.sh $(SC_ROOT)/run_confirmation.sh $(SC_ROOT)/publish.py code/scripts/task.sh code/evaluation/protocols/architecture_screen_v1/Makefile $(SC_CARD) code/evaluation/cards/multiband_screen_v1.md code/evaluation/include/frozen_role_guard.h $(SC_RPB_HEADERS) $(wildcard code/shared/include/embedding/shared/*.h) $(SC_SHARED_SOURCES) $(RPB_ROOT)/tests/spectral_temporal_relation_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h code/shared/tests/shared_test_support.h $(SC_DATA)/two_component.h $(SC_DATA)/two_component.cpp $(SC_DATA)/observed_information.h $(SC_DATA)/observed_information.cpp code/evaluation/cards/two_component_information_v1.md doc/results/two_component_information_v1.json code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/evaluation/protocols/visible_difference_v1/reader-v2/saved_cpu_math.py code/evaluation/protocols/visible_difference_v1/reader-v2/legacy_implementation.py $(SDK_PROVENANCE_INPUTS))
SC_SOURCE_ID := $(shell sha256sum $(SC_INPUTS) | sha256sum | cut -d ' ' -f 1)
SC_CARD_SHA := $(shell sha256sum $(SC_CARD) | cut -d ' ' -f 1)
SC_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests -I$(SC_DATA) -Icode/evaluation/include -DSPECTRAL_CONFIRMATION_SOURCE_ID=\"$(SC_SOURCE_ID)\" -DSPECTRAL_CONFIRMATION_CARD_SHA256=\"$(SC_CARD_SHA)\" -DMULTIBAND_SCREEN_SOURCE_ID=\"$(SC_SOURCE_ID)\" -DRPB_SOURCE_ID=\"$(SC_SOURCE_ID)\"
# These generic objects carry no encoder/protocol macros and use the pinned SDK.
# Missing artifacts reject; this protocol never rebuilds or changes their producer.
SC_SHARED_DIR := /opt/cuwacunu_embedding/build/rpb-architecture-screen/code/protocols/architecture_screen_v1/shared
SC_SHARED_OBJECTS := $(addprefix $(SC_SHARED_DIR)/,$(addsuffix .o,$(SC_SHARED_NAMES)))

.PHONY: spectral-confirmation spectral-confirmation-quality-data print-spectral-confirmation-sources test-spectral-confirmation-v19 test-spectral-confirmation-runner test-spectral-confirmation-quality-data
spectral-confirmation: $(SC_BIN)
spectral-confirmation-quality-data: $(SC_WRITER)
print-spectral-confirmation-sources:
	@printf '%s\n' $(SC_INPUTS)
$(SC_OBJECT_DIR)/producer_main.o: $(SC_ROOT)/producer_main.cpp $(SC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(SC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(SC_OBJECT_DIR)/write_quality_data.o: $(SC_ROOT)/write_quality_data.cpp $(SC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(SC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(SC_OBJECT_DIR)/two_component.o: $(SC_DATA)/two_component.cpp $(SC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(SC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(SC_SHARED_OBJECTS):
	@printf '%s\n' 'Required frozen generic shared object is absent: $@' >&2; exit 1
$(SC_BIN): $(SC_OBJECT_DIR)/producer_main.o $(SC_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(SC_WRITER): $(SC_OBJECT_DIR)/write_quality_data.o $(SC_OBJECT_DIR)/two_component.o $(SC_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(SC_OBJECT_DIR)/spectral_temporal_relation_test.o: $(RPB_ROOT)/tests/spectral_temporal_relation_test.cpp $(SC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(SC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(SC_OBJECT_DIR)/spectral_temporal_relation_test: $(SC_OBJECT_DIR)/spectral_temporal_relation_test.o $(SC_SHARED_DIR)/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-spectral-confirmation-v19: $(SC_OBJECT_DIR)/spectral_temporal_relation_test
	$(SC_OBJECT_DIR)/spectral_temporal_relation_test
test-spectral-confirmation-runner: $(SC_BIN)
	@test -n "$(ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set ENGINEERING_OUTPUT to an absent fixture leaf.' >&2; exit 1; }
	$(SC_BIN) --engineering "$(ENGINEERING_OUTPUT)"
test-spectral-confirmation-quality-data: $(SC_WRITER)
	@test -n "$(DATA_ENGINEERING_OUTPUT)" || { printf '%s\n' 'Set DATA_ENGINEERING_OUTPUT to an absent data-only fixture leaf.' >&2; exit 1; }
	$(SC_WRITER) --engineering "$(DATA_ENGINEERING_OUTPUT)"
-include $(wildcard $(SC_OBJECT_DIR)/*.d)
