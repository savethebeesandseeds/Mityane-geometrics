# Numeric-series dataloader. This tree is the whole build: headers, one
# translation unit, and its storage tests. It does not include an encoder.
HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/mityane-geometrics/dataloaders/build/numeric-series
CXX ?= g++
CPPFLAGS := -I$(HERE)/include \
	-isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include \
	-D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(LIBTORCH)/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread

DATA_TEST := $(BUILD_DIR)/test_memory_mapped
DATA_SAFETY_TEST := $(BUILD_DIR)/test_data_safety
SETTINGS_TEST := $(BUILD_DIR)/test_settings_and_raw_cache

.PHONY: test test-data test-data-safety test-settings
test: test-data test-data-safety test-settings
test-data: $(DATA_TEST)
	$(DATA_TEST)
test-data-safety: $(DATA_SAFETY_TEST)
	$(DATA_SAFETY_TEST)
test-settings: $(SETTINGS_TEST)
	$(SETTINGS_TEST)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/record_types.o: src/record_types.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_memory_mapped.o: tests/test_memory_mapped.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_data_safety.o: tests/test_data_safety.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_settings_and_raw_cache.o: tests/test_settings_and_raw_cache.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(DATA_TEST): $(BUILD_DIR)/test_memory_mapped.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(DATA_SAFETY_TEST): $(BUILD_DIR)/test_data_safety.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(SETTINGS_TEST): $(BUILD_DIR)/test_settings_and_raw_cache.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

-include $(BUILD_DIR)/*.d
