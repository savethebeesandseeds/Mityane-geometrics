# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
ifeq ($(wildcard /.dockerenv),)
$(error Build and test inside the managed cuwacunu container using code/scripts/task.sh)
endif
CXX := g++
BUILD_DIR ?= /opt/cuwacunu/build/default
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
CPB_ROOT := code/encoders/causal_patch_bottleneck_mae/versions/v3
CPPFLAGS := -I$(CPB_ROOT)/include -Icode/data/include -Icode/config/include -Icode/integration/include \
  -Icode/artifacts/versions/v1/include \
  -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include \
  -D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread

DATA_TEST := $(BUILD_DIR)/test_memory_mapped
DATA_SAFETY_TEST := $(BUILD_DIR)/test_data_safety
CONFIG_TEST := $(BUILD_DIR)/test_config
PRIMARY_TEST := $(BUILD_DIR)/test_causal_encoder_input
ENCODE_HISTORY := $(BUILD_DIR)/encode_history
OBJECTS := $(BUILD_DIR)/record_types.o $(BUILD_DIR)/test_memory_mapped.o \
  $(BUILD_DIR)/test_data_safety.o $(BUILD_DIR)/test_config.o \
  $(BUILD_DIR)/causal_encoder_input_test.o $(BUILD_DIR)/encode_history.o

.PHONY: all test build-primary test-primary test-encoder test-integration \
  test-data test-data-safety test-config test-stability test-setup encode-history
all: $(DATA_TEST) $(DATA_SAFETY_TEST) $(CONFIG_TEST) build-primary
build-primary: $(PRIMARY_TEST) $(ENCODE_HISTORY)
test: all
	bash code/scripts/test-setup.sh
	$(DATA_TEST)
	$(DATA_SAFETY_TEST)
	$(CONFIG_TEST)
	$(PRIMARY_TEST) config/encoder.conf config/encoder_runtime.conf
test-primary: build-primary
	$(PRIMARY_TEST) config/encoder.conf config/encoder_runtime.conf
test-encoder: test-primary
test-integration: test-primary
test-data: $(DATA_TEST)
	$(DATA_TEST)
test-data-safety: $(DATA_SAFETY_TEST)
	$(DATA_SAFETY_TEST)
test-config: $(CONFIG_TEST)
	$(CONFIG_TEST)
test-stability: test
test-setup:
	bash code/scripts/test-setup.sh
# ARGS supplies an explicit cutoff and three semantic channel sources.
encode-history: $(ENCODE_HISTORY)
	$(ENCODE_HISTORY) $(ARGS)

$(BUILD_DIR)/record_types.o: code/data/src/record_types.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_memory_mapped.o: code/data/tests/test_memory_mapped.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_data_safety.o: code/data/tests/test_data_safety.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_config.o: code/config/tests/document_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -Icode/config/include $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/causal_encoder_input_test.o: code/integration/tests/causal_encoder_input_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/encode_history.o: code/integration/src/encode_history.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CONFIG_TEST): $(BUILD_DIR)/test_config.o
	$(CXX) $^ -o $@
$(DATA_TEST): $(BUILD_DIR)/test_memory_mapped.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(DATA_SAFETY_TEST): $(BUILD_DIR)/test_data_safety.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(PRIMARY_TEST): $(BUILD_DIR)/causal_encoder_input_test.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(ENCODE_HISTORY): $(BUILD_DIR)/encode_history.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(OBJECTS:.o=.d)

# Optional encoder-independent saved-feature readouts; no historical encoders.
include code/evaluation/market_readouts.mk
include code/latent_market/versions/v1/v1.mk

# Immutable artifact sidecars and checked integration; no numerical module changes.
include code/artifacts/versions/v1/v1.mk
include code/integration/snapshot.mk
include code/networks/direct_embedding_control/versions/v1/v1.mk
include code/latent_market/versions/v1/embedding_controls.mk
test: test-model-snapshots
test-primary: test-encoder-snapshot
test-latent-market-v1: test-artifacts-v1 test-predictive-snapshot
