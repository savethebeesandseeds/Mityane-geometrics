ifeq ($(wildcard /.dockerenv),)
$(error Build and test inside the managed cuwacunu container using code/scripts/task.sh)
endif
CXX := g++
BUILD_DIR ?= /opt/cuwacunu/build/default
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
CPPFLAGS := -Icode/encoders/raw_patch_bottleneck_mae/include -Icode/data/include -Icode/config/include -Icode/encoders/rpb_spectral/include \
  -Icode/encoders/rpb_content/include \
  -Icode/encoders/rpb_v19/include -Icode/integration/include \
  -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include \
  -D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread

DATA_TEST := $(BUILD_DIR)/test_memory_mapped
ENCODER_TEST := $(BUILD_DIR)/test_rpb_v19_import
INTEGRATION_TEST := $(BUILD_DIR)/test_encoder_input
CONFIG_TEST := $(BUILD_DIR)/test_config
V20_TEST := $(BUILD_DIR)/test_rpb_v20
V20_CHECKPOINT_TEST := $(BUILD_DIR)/test_rpb_v20_checkpoint
V20_INTEGRATION_TEST := $(BUILD_DIR)/test_rpb_v20_input
CHECKPOINT_INTEGRITY_TEST := $(BUILD_DIR)/test_checkpoint_integrity
DATA_SAFETY_TEST := $(BUILD_DIR)/test_data_safety
V20_TESTS := $(CONFIG_TEST) $(V20_TEST) $(V20_CHECKPOINT_TEST) $(V20_INTEGRATION_TEST) $(CHECKPOINT_INTEGRITY_TEST)
TESTS := $(DATA_TEST) $(DATA_SAFETY_TEST) $(ENCODER_TEST) $(INTEGRATION_TEST) $(V20_TESTS)
OBJECTS := $(BUILD_DIR)/record_types.o $(BUILD_DIR)/test_memory_mapped.o \
  $(BUILD_DIR)/test_rpb_v19_import.o $(BUILD_DIR)/test_encoder_input.o \
  $(BUILD_DIR)/test_config.o $(BUILD_DIR)/test_rpb_v20.o \
  $(BUILD_DIR)/test_rpb_v20_checkpoint.o $(BUILD_DIR)/test_rpb_v20_input.o \
  $(BUILD_DIR)/test_checkpoint_integrity.o $(BUILD_DIR)/test_data_safety.o

.PHONY: all test test-data test-encoder test-integration test-reference test-v19 test-v20 build-v20 test-v20-integration test-config test-data-safety test-checkpoint-integrity test-stability test-setup
all: $(TESTS) build-primary
test: all
	bash code/scripts/test-setup.sh
	$(DATA_TEST)
	$(DATA_SAFETY_TEST)
	$(ENCODER_TEST)
	$(INTEGRATION_TEST)
	$(CONFIG_TEST)
	$(V20_TEST)
	$(V20_CHECKPOINT_TEST) config/encoder_v20.conf
	$(V20_INTEGRATION_TEST)
	$(CHECKPOINT_INTEGRITY_TEST)
	$(CONTENT_INPUT_TEST) config/encoder.conf
	$(PRIMARY_INPUT_TEST) config/encoder.conf
	bash code/scripts/test-reference.sh $(ENCODER_TEST)
test-data: $(DATA_TEST)
	$(DATA_TEST)
test-encoder: test-primary
test-v19: $(ENCODER_TEST)
	$(ENCODER_TEST)
test-integration: $(INTEGRATION_TEST)
	$(INTEGRATION_TEST)
test-reference: $(ENCODER_TEST)
	bash code/scripts/test-reference.sh $(ENCODER_TEST)
build-v20: $(V20_TESTS)
test-v20: build-v20
	$(CONFIG_TEST)
	$(V20_TEST)
	$(V20_CHECKPOINT_TEST) config/encoder_v20.conf
	$(V20_INTEGRATION_TEST)
	$(CHECKPOINT_INTEGRITY_TEST)
test-v20-integration: $(V20_INTEGRATION_TEST)
	$(V20_INTEGRATION_TEST)
test-config: $(CONFIG_TEST)
	$(CONFIG_TEST)
test-data-safety: $(DATA_SAFETY_TEST)
	$(DATA_SAFETY_TEST)
test-checkpoint-integrity: $(CHECKPOINT_INTEGRITY_TEST)
	$(CHECKPOINT_INTEGRITY_TEST)
test-stability: $(DATA_TEST) $(DATA_SAFETY_TEST) $(INTEGRATION_TEST) $(V20_TESTS) build-primary
	bash code/scripts/test-setup.sh
	$(DATA_TEST)
	$(DATA_SAFETY_TEST)
	$(INTEGRATION_TEST)
	$(CONFIG_TEST)
	$(V20_TEST)
	$(V20_CHECKPOINT_TEST) config/encoder_v20.conf
	$(V20_INTEGRATION_TEST)
	$(CHECKPOINT_INTEGRITY_TEST)
	$(CONTENT_INPUT_TEST) config/encoder.conf
	$(PRIMARY_INPUT_TEST) config/encoder.conf

test-setup:
	bash code/scripts/test-setup.sh

$(BUILD_DIR)/record_types.o: code/data/src/record_types.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_memory_mapped.o: code/data/tests/test_memory_mapped.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_rpb_v19_import.o: code/encoders/raw_patch_bottleneck_mae/versions/v19/tests/import_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_encoder_input.o: code/integration/tests/encoder_input_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_config.o: code/config/tests/document_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -Icode/config/include $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_rpb_v20.o: code/encoders/raw_patch_bottleneck_mae/versions/v20/tests/configurable_spectral_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_rpb_v20_checkpoint.o: code/encoders/raw_patch_bottleneck_mae/versions/v20/tests/configurable_checkpoint_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_rpb_v20_input.o: code/integration/tests/configurable_encoder_input_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_checkpoint_integrity.o: code/encoders/raw_patch_bottleneck_mae/versions/v20/tests/checkpoint_integrity_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_data_safety.o: code/data/tests/test_data_safety.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(DATA_TEST): $(BUILD_DIR)/test_memory_mapped.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(ENCODER_TEST): $(BUILD_DIR)/test_rpb_v19_import.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(INTEGRATION_TEST): $(BUILD_DIR)/test_encoder_input.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CONFIG_TEST): $(BUILD_DIR)/test_config.o
	$(CXX) $^ -o $@
$(V20_TEST): $(BUILD_DIR)/test_rpb_v20.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(V20_CHECKPOINT_TEST): $(BUILD_DIR)/test_rpb_v20_checkpoint.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(V20_INTEGRATION_TEST): $(BUILD_DIR)/test_rpb_v20_input.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CHECKPOINT_INTEGRITY_TEST): $(BUILD_DIR)/test_checkpoint_integrity.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(DATA_SAFETY_TEST): $(BUILD_DIR)/test_data_safety.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(OBJECTS:.o=.d)

# Primary content-only relation API; versioned v20 targets remain explicit.
include code/encoders/raw_patch_bottleneck_mae/versions/v22/content_ordered_relation.mk
include code/encoders/raw_patch_bottleneck_mae/shared/streaming_scaler.mk
include code/encoders/raw_patch_bottleneck_mae/versions/v22/training_state.mk
include code/evaluation/market_readouts.mk
include code/training/market_pilot.mk
include code/training/market_data_contract.mk
include code/training/market_study.mk
include code/encoders/raw_patch_bottleneck_mae/versions/v23/consecutive_change.mk
include code/training/market_changes.mk
include code/training/market_frozen_changes.mk
include code/training/market_separate_changes.mk

include code/training/market_separate_changes_native36.mk
include code/evaluation/market_synthetic_transfer_v1/transfer.mk
include code/evaluation/market_synthetic_transfer_v1/reader.mk
include code/evaluation/market_train_volatility_diagnosis_v1/diagnosis.mk
include code/encoders/raw_patch_bottleneck_mae/versions/v25/energy_change_test.mk
include code/training/market_change_energy.mk
include code/training/market_conditioned_changes.mk
include code/encoders/raw_patch_bottleneck_mae/versions/v26/conditioned_change_test.mk
include code/encoders/raw_patch_bottleneck_mae/versions/v27/parent_conditional_change_test.mk
include code/training/market_parent_conditional_changes.mk

include code/training/market_relative_changes.mk

include code/encoders/raw_patch_bottleneck_mae/versions/v28/relative_change_test.mk

include code/training/market_parent_modulated_decoder.mk

include code/encoders/raw_patch_bottleneck_mae/versions/v29/parent_modulated_decoder_test.mk

include code/encoders/raw_patch_bottleneck_mae/versions/v30/location_scale_decoder_test.mk
include code/training/market_location_scale_decoder.mk
include code/evaluation/market_location_scale_synthetic_transfer_v1/transfer.mk
include code/evaluation/market_location_scale_synthetic_transfer_v1/reader.mk
include code/encoders/causal_patch_bottleneck_mae/versions/v1/v1.mk
include code/evaluation/causal_patch_bottleneck_mae_v1/evaluation.mk
