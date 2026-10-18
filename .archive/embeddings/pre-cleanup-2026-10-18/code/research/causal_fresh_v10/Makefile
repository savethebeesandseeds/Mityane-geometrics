# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Invoke from an admitted source capsule root inside the existing cuwacunu container.
ifeq ($(wildcard /.dockerenv),)
$(error Build and run this research only inside the existing managed GPU container)
endif
CXX := g++
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/cuwacunu/build/causal-fresh-v10
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread
CPB_V10_RESEARCH_ROOT := code/research/causal_fresh_v10
CPB_V10_TRAIN := $(BUILD_DIR)/causal_fresh_v10_train
.DEFAULT_GOAL := all
.PHONY: all build-cpb-v10-train
all: build-cpb-v10-engineering build-cpb-v10-train build-cpb-v10-quality
build-cpb-v10-train: $(CPB_V10_TRAIN)
$(BUILD_DIR)/causal_fresh_v10_train.o: $(CPB_V10_RESEARCH_ROOT)/train_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V10_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V10_TRAIN): $(BUILD_DIR)/causal_fresh_v10_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_fresh_v10_train.d
CPB_V10_CPPFLAGS := -Icode/encoders/causal_patch_bottleneck_mae/versions/v10/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V10_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v10
CPB_V10_SEPARATION_TEST := $(BUILD_DIR)/test_cpb_v10_variance_gradient
CPB_V10_TEMPORAL_TEST := $(BUILD_DIR)/test_cpb_v10_temporal_pooling
.PHONY: build-cpb-v10-engineering
build-cpb-v10-engineering: $(CPB_V10_TEST) $(CPB_V10_SEPARATION_TEST) $(CPB_V10_TEMPORAL_TEST)
$(BUILD_DIR)/test_causal_patch_bottleneck_mae_v10.o: $(CPB_V10_RESEARCH_ROOT)/tests/v10_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V10_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_cpb_v10_variance_gradient.o: $(CPB_V10_RESEARCH_ROOT)/tests/v10_separation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V10_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V10_TEST): $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v10.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CPB_V10_SEPARATION_TEST): $(BUILD_DIR)/test_cpb_v10_variance_gradient.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test_cpb_v10_temporal_pooling.o: $(CPB_V10_RESEARCH_ROOT)/tests/v10_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V10_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V10_TEMPORAL_TEST): $(BUILD_DIR)/test_cpb_v10_temporal_pooling.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v10.d $(BUILD_DIR)/test_cpb_v10_variance_gradient.d $(BUILD_DIR)/test_cpb_v10_temporal_pooling.d
include $(CPB_V10_RESEARCH_ROOT)/evaluation/evaluation.mk
