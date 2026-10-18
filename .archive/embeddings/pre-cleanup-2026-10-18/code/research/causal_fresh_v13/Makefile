# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Invoke from an admitted source capsule root inside the existing cuwacunu container.
ifeq ($(wildcard /.dockerenv),)
$(error Build and run this research only inside the existing managed GPU container)
endif
CXX := g++
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/cuwacunu/build/causal-fresh-v13
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread
CPB_V13_RESEARCH_ROOT := code/research/causal_fresh_v13
CPB_V13_TRAIN := $(BUILD_DIR)/causal_fresh_v13_train
.DEFAULT_GOAL := all
.PHONY: all build-cpb-v13-train
all: build-cpb-v13-engineering build-cpb-v13-train build-cpb-v13-quality
build-cpb-v13-train: $(CPB_V13_TRAIN)
$(BUILD_DIR)/causal_fresh_v13_train.o: $(CPB_V13_RESEARCH_ROOT)/train_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V13_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V13_TRAIN): $(BUILD_DIR)/causal_fresh_v13_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_fresh_v13_train.d
CPB_V13_CPPFLAGS := -Icode/encoders/causal_patch_bottleneck_mae/versions/v13/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V13_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v13
CPB_V13_SEPARATION_TEST := $(BUILD_DIR)/test_cpb_v13_variance_gradient
CPB_V13_TEMPORAL_TEST := $(BUILD_DIR)/test_cpb_v13_temporal_pooling
.PHONY: build-cpb-v13-engineering
build-cpb-v13-engineering: $(CPB_V13_TEST) $(CPB_V13_SEPARATION_TEST) $(CPB_V13_TEMPORAL_TEST)
$(BUILD_DIR)/test_causal_patch_bottleneck_mae_v13.o: $(CPB_V13_RESEARCH_ROOT)/tests/v13_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V13_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_cpb_v13_variance_gradient.o: $(CPB_V13_RESEARCH_ROOT)/tests/v13_separation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V13_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V13_TEST): $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v13.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CPB_V13_SEPARATION_TEST): $(BUILD_DIR)/test_cpb_v13_variance_gradient.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test_cpb_v13_temporal_pooling.o: $(CPB_V13_RESEARCH_ROOT)/tests/v13_temporal_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V13_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V13_TEMPORAL_TEST): $(BUILD_DIR)/test_cpb_v13_temporal_pooling.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v13.d $(BUILD_DIR)/test_cpb_v13_variance_gradient.d $(BUILD_DIR)/test_cpb_v13_temporal_pooling.d
include $(CPB_V13_RESEARCH_ROOT)/evaluation/evaluation.mk
