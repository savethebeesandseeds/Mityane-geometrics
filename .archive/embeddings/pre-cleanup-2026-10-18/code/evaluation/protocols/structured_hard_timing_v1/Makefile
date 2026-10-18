# Additive protocol build. Run from the repository root with -f this file.
# Historical Makefile bytes, source formulas and object recipes remain literal.
include Makefile

.DEFAULT_GOAL := structured-hard-timing
STRUCTURED_HARD_TIMING_ROOT := code/evaluation/protocols/structured_hard_timing_v1
STRUCTURED_HARD_TIMING_GENERATOR_ROOT := code/evaluation/benchmarks/structured_hard_timing
STRUCTURED_HARD_TIMING_OBJECT_DIR := $(OBJECT_DIR)/protocols/structured_hard_timing_v1
STRUCTURED_HARD_TIMING_BIN := $(BUILD_DIR)/embedding_structured_hard_timing
STRUCTURED_HARD_TIMING_ADAPTER_OBJECT := $(STRUCTURED_HARD_TIMING_OBJECT_DIR)/structured_hard_timing_adapter.o
STRUCTURED_HARD_TIMING_RUN_OBJECT := $(STRUCTURED_HARD_TIMING_OBJECT_DIR)/paired_timing_run.o
STRUCTURED_HARD_TIMING_GENERATOR_OBJECT := $(STRUCTURED_HARD_TIMING_OBJECT_DIR)/structured_hard_timing_generator.o
STRUCTURED_HARD_TIMING_SOLVABILITY_OBJECT := $(STRUCTURED_HARD_TIMING_OBJECT_DIR)/cross_feature_solvability.o
STRUCTURED_HARD_TIMING_TEST_BIN := $(STRUCTURED_HARD_TIMING_OBJECT_DIR)/structured_hard_timing_adapter_test
STRUCTURED_HARD_TIMING_INPUTS := $(sort $(EARLY_MIXER_RELIABILITY_INPUTS) $(SDK_PROVENANCE_INPUTS) $(STRUCTURED_HARD_TIMING_ROOT)/Makefile $(STRUCTURED_HARD_TIMING_ROOT)/structured_hard_timing_adapter.h $(STRUCTURED_HARD_TIMING_ROOT)/structured_hard_timing_adapter.cpp $(STRUCTURED_HARD_TIMING_ROOT)/paired_timing_run.h $(STRUCTURED_HARD_TIMING_ROOT)/analytic_support.h $(STRUCTURED_HARD_TIMING_ROOT)/paired_timing_run.cpp $(STRUCTURED_HARD_TIMING_ROOT)/structured_hard_timing_main.cpp $(STRUCTURED_HARD_TIMING_ROOT)/structured_hard_timing_adapter_test.cpp $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/structured_hard_timing.h $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/structured_hard_timing.cpp $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/generator_test.cpp $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/cross_feature_solvability.h $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/cross_feature_solvability.cpp $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/cross_feature_test.cpp $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/Makefile code/evaluation/cards/structured_hard_timing_comparison_v2.md)
STRUCTURED_HARD_TIMING_INPUTS := $(sort $(STRUCTURED_HARD_TIMING_INPUTS) $(STRUCTURED_HARD_TIMING_ROOT)/prepare_saved_run.py $(STRUCTURED_HARD_TIMING_ROOT)/check-structured-hard-timing.sh $(STRUCTURED_HARD_TIMING_ROOT)/evaluate-structured-hard-timing.sh doc/dataset_registry.json)
STRUCTURED_HARD_TIMING_SOURCE_ID := $(shell sha256sum $(STRUCTURED_HARD_TIMING_INPUTS) | sha256sum | cut -d ' ' -f 1)
STRUCTURED_HARD_TIMING_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -I$(STRUCTURED_HARD_TIMING_ROOT) -I$(STRUCTURED_HARD_TIMING_GENERATOR_ROOT) -DEVALUATION_SOURCE_ID=\"$(STRUCTURED_HARD_TIMING_SOURCE_ID)\" -DSTRUCTURED_HARD_TIMING_ADAPTER_SOURCE_ID=\"$(STRUCTURED_HARD_TIMING_SOURCE_ID)\"
STRUCTURED_HARD_TIMING_REQUIRED_RECOMPILE_SOURCES := $(RPB_ROOT)/src/workflow.cpp $(RPB_ROOT)/src/learning_curve_adapter.cpp $(RPB_ROOT)/src/early_mixer_adapter.cpp $(STRUCTURED_HARD_TIMING_ROOT)/structured_hard_timing_adapter.cpp $(STRUCTURED_HARD_TIMING_ROOT)/paired_timing_run.cpp $(STRUCTURED_HARD_TIMING_ROOT)/structured_hard_timing_main.cpp $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/structured_hard_timing.cpp $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/cross_feature_solvability.cpp

.PHONY: structured-hard-timing print-structured-hard-timing-sources print-structured-hard-timing-scopes print-structured-hard-timing-scope-inputs test-rpb-structured-hard-timing-adapter evaluate-structured-hard-timing
structured-hard-timing: $(STRUCTURED_HARD_TIMING_BIN)
print-structured-hard-timing-sources:
	@printf '%s\n' $(STRUCTURED_HARD_TIMING_INPUTS)
print-structured-hard-timing-scopes:
	@printf 'core_writer=%s\ncurve_training=%s\nearly_adapter=%s\nstructured_adapter=%s\n' '$(RPB_SOURCE_ID)' '$(EVALUATION_SOURCE_ID)' '$(EARLY_MIXER_RELIABILITY_SOURCE_ID)' '$(STRUCTURED_HARD_TIMING_SOURCE_ID)'
print-structured-hard-timing-scope-inputs:
	@printf 'core_writer|%s\n' $(RPB_PROVENANCE_INPUTS)
	@printf 'curve_training|%s\n' $(EVALUATION_PROVENANCE_INPUTS)
	@printf 'early_adapter|%s\n' $(EARLY_MIXER_RELIABILITY_INPUTS)
	@printf 'structured_adapter|%s\n' $(STRUCTURED_HARD_TIMING_INPUTS)

$(STRUCTURED_HARD_TIMING_OBJECT_DIR)/%.o: $(STRUCTURED_HARD_TIMING_ROOT)/%.cpp $(STRUCTURED_HARD_TIMING_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(STRUCTURED_HARD_TIMING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(STRUCTURED_HARD_TIMING_GENERATOR_OBJECT): $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/structured_hard_timing.cpp $(STRUCTURED_HARD_TIMING_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(STRUCTURED_HARD_TIMING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(STRUCTURED_HARD_TIMING_SOLVABILITY_OBJECT): $(STRUCTURED_HARD_TIMING_GENERATOR_ROOT)/cross_feature_solvability.cpp $(STRUCTURED_HARD_TIMING_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(STRUCTURED_HARD_TIMING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(STRUCTURED_HARD_TIMING_BIN): $(STRUCTURED_HARD_TIMING_OBJECT_DIR)/structured_hard_timing_main.o $(STRUCTURED_HARD_TIMING_RUN_OBJECT) $(STRUCTURED_HARD_TIMING_ADAPTER_OBJECT) $(STRUCTURED_HARD_TIMING_GENERATOR_OBJECT) $(STRUCTURED_HARD_TIMING_SOLVABILITY_OBJECT) $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(STRUCTURED_HARD_TIMING_OBJECT_DIR)/structured_hard_timing_adapter_test.o: $(STRUCTURED_HARD_TIMING_ROOT)/structured_hard_timing_adapter_test.cpp $(STRUCTURED_HARD_TIMING_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(STRUCTURED_HARD_TIMING_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(STRUCTURED_HARD_TIMING_TEST_BIN): $(STRUCTURED_HARD_TIMING_OBJECT_DIR)/structured_hard_timing_adapter_test.o $(STRUCTURED_HARD_TIMING_ADAPTER_OBJECT) $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-rpb-structured-hard-timing-adapter: $(STRUCTURED_HARD_TIMING_TEST_BIN)
	$(STRUCTURED_HARD_TIMING_TEST_BIN)

evaluate-structured-hard-timing: $(STRUCTURED_HARD_TIMING_BIN)
	@STRUCTURED_HARD_TIMING_BIN="$(abspath $(STRUCTURED_HARD_TIMING_BIN))" STRUCTURED_HARD_TIMING_SOURCE_INPUTS="$(STRUCTURED_HARD_TIMING_INPUTS)" bash $(STRUCTURED_HARD_TIMING_ROOT)/evaluate-structured-hard-timing.sh

-include $(wildcard $(STRUCTURED_HARD_TIMING_OBJECT_DIR)/*.d)
