# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Invoke from an admitted source capsule root inside the existing cuwacunu container.
ifeq ($(wildcard /.dockerenv),)
$(error Build and run this research only inside the existing managed GPU container)
endif
CXX := g++
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/cuwacunu/build/causal-fresh-v8
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread
CPB_V8_RESEARCH_ROOT := code/research/causal_fresh_v8
CPB_V8_TRAIN := $(BUILD_DIR)/causal_fresh_v8_train
.DEFAULT_GOAL := all
.PHONY: all build-cpb-v8-train
all: build-cpb-v8-engineering build-cpb-v8-train build-cpb-v8-quality
build-cpb-v8-train: $(CPB_V8_TRAIN)
$(BUILD_DIR)/causal_fresh_v8_train.o: $(CPB_V8_RESEARCH_ROOT)/train_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V8_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V8_TRAIN): $(BUILD_DIR)/causal_fresh_v8_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_fresh_v8_train.d
CPB_V8_CPPFLAGS := -Icode/encoders/causal_patch_bottleneck_mae/versions/v8/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V8_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v8
CPB_V8_SEPARATION_TEST := $(BUILD_DIR)/test_cpb_v8_variance_gradient
CPB_V8_TEMPORAL_TEST := $(BUILD_DIR)/test_cpb_v8_temporal_pooling
.PHONY: build-cpb-v8-engineering
build-cpb-v8-engineering: $(CPB_V8_TEST) $(CPB_V8_SEPARATION_TEST) $(CPB_V8_TEMPORAL_TEST)
$(BUILD_DIR)/test_causal_patch_bottleneck_mae_v8.o: $(CPB_V8_RESEARCH_ROOT)/tests/v8_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V8_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_cpb_v8_variance_gradient.o: $(CPB_V8_RESEARCH_ROOT)/tests/v8_separation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V8_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V8_TEST): $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v8.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CPB_V8_SEPARATION_TEST): $(BUILD_DIR)/test_cpb_v8_variance_gradient.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test_cpb_v8_temporal_pooling.o: $(CPB_V8_RESEARCH_ROOT)/tests/v8_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V8_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V8_TEMPORAL_TEST): $(BUILD_DIR)/test_cpb_v8_temporal_pooling.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v8.d $(BUILD_DIR)/test_cpb_v8_variance_gradient.d $(BUILD_DIR)/test_cpb_v8_temporal_pooling.d
include $(CPB_V8_RESEARCH_ROOT)/evaluation/evaluation.mk
