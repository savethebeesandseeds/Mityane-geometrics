# Additive candidate-only build; historical Makefile remains unchanged.
include Makefile
.DEFAULT_GOAL := visible-difference
VISIBLE_DIFFERENCE_ROOT := code/evaluation/protocols/visible_difference_v1
VISIBLE_DIFFERENCE_OBJECT_DIR := $(OBJECT_DIR)/protocols/visible_difference_v1
VISIBLE_DIFFERENCE_BIN := $(BUILD_DIR)/embedding_visible_difference
VISIBLE_DIFFERENCE_ADAPTER_OBJECT := $(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_adapter.o
VISIBLE_DIFFERENCE_INPUTS := $(sort $(EARLY_MIXER_RELIABILITY_INPUTS) $(SDK_PROVENANCE_INPUTS) $(VISIBLE_DIFFERENCE_ROOT)/Makefile $(VISIBLE_DIFFERENCE_ROOT)/visible_difference_main.cpp $(VISIBLE_DIFFERENCE_ROOT)/prepare_saved_run.py $(VISIBLE_DIFFERENCE_ROOT)/check-visible-difference.sh $(VISIBLE_DIFFERENCE_ROOT)/evaluate-visible-difference.sh $(VISIBLE_DIFFERENCE_ROOT)/retained-roles.json $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/visible_difference.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/visible_difference_adapter.h $(RPB_ROOT)/src/visible_difference_adapter.cpp $(RPB_ROOT)/tests/visible_difference_model_test.cpp $(RPB_ROOT)/tests/visible_difference_adapter_test.cpp $(RPB_ROOT)/tests/pooled_context_old_model_reference.h code/evaluation/cards/visible_difference_v1.md)
VISIBLE_DIFFERENCE_SOURCE_ID := $(shell sha256sum $(VISIBLE_DIFFERENCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
VISIBLE_DIFFERENCE_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests -I$(EVALUATION_ROOT)/include -DVISIBLE_DIFFERENCE_SOURCE_ID=\"$(VISIBLE_DIFFERENCE_SOURCE_ID)\" -DVISIBLE_DIFFERENCE_ADAPTER_SOURCE_ID=\"$(VISIBLE_DIFFERENCE_SOURCE_ID)\"

.PHONY: visible-difference print-visible-difference-sources print-visible-difference-scope-inputs test-visible-difference-model test-visible-difference-adapter evaluate-visible-difference
visible-difference: $(VISIBLE_DIFFERENCE_BIN)
print-visible-difference-sources:
	@printf '%s\n' $(VISIBLE_DIFFERENCE_INPUTS)
print-visible-difference-scope-inputs:
	@printf 'core_writer|%s\n' $(RPB_PROVENANCE_INPUTS)
	@printf 'curve_training|%s\n' $(EVALUATION_PROVENANCE_INPUTS)
	@printf 'early_adapter|%s\n' $(EARLY_MIXER_RELIABILITY_INPUTS)
	@printf 'visible_adapter|%s\n' $(VISIBLE_DIFFERENCE_INPUTS)

$(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_main.o: $(VISIBLE_DIFFERENCE_ROOT)/visible_difference_main.cpp $(VISIBLE_DIFFERENCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(VISIBLE_DIFFERENCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(VISIBLE_DIFFERENCE_ADAPTER_OBJECT): $(RPB_ROOT)/src/visible_difference_adapter.cpp $(VISIBLE_DIFFERENCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(VISIBLE_DIFFERENCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(VISIBLE_DIFFERENCE_BIN): $(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_main.o $(VISIBLE_DIFFERENCE_ADAPTER_OBJECT) $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_model_test.o: $(RPB_ROOT)/tests/visible_difference_model_test.cpp $(VISIBLE_DIFFERENCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_model_test: $(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_model_test.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-visible-difference-model: $(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_model_test
	$(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_model_test
$(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_adapter_test.o: $(RPB_ROOT)/tests/visible_difference_adapter_test.cpp $(VISIBLE_DIFFERENCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(VISIBLE_DIFFERENCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_adapter_test: $(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_adapter_test.o $(VISIBLE_DIFFERENCE_ADAPTER_OBJECT) $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-visible-difference-adapter: $(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_adapter_test
	$(VISIBLE_DIFFERENCE_OBJECT_DIR)/visible_difference_adapter_test
evaluate-visible-difference: $(VISIBLE_DIFFERENCE_BIN)
	@VISIBLE_DIFFERENCE_BIN="$(abspath $(VISIBLE_DIFFERENCE_BIN))" bash $(VISIBLE_DIFFERENCE_ROOT)/evaluate-visible-difference.sh
-include $(wildcard $(VISIBLE_DIFFERENCE_OBJECT_DIR)/*.d)
