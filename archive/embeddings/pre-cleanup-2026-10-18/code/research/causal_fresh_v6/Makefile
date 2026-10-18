# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Invoke from an admitted source capsule root inside the existing cuwacunu container.
ifeq ($(wildcard /.dockerenv),)
$(error Build and run this research only inside the existing managed GPU container)
endif
CXX := g++
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/cuwacunu/build/causal-fresh-v6
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread
CPB_V6_RESEARCH_ROOT := code/research/causal_fresh_v6
CPB_V6_TRAIN := $(BUILD_DIR)/causal_fresh_v6_train
.DEFAULT_GOAL := all
.PHONY: all build-cpb-v6-train
all: build-cpb-v6-engineering build-cpb-v6-train build-cpb-v6-quality
build-cpb-v6-train: $(CPB_V6_TRAIN)
$(BUILD_DIR)/causal_fresh_v6_train.o: $(CPB_V6_RESEARCH_ROOT)/train_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V6_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V6_TRAIN): $(BUILD_DIR)/causal_fresh_v6_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_fresh_v6_train.d
CPB_V6_CPPFLAGS := -Icode/encoders/causal_patch_bottleneck_mae/versions/v6/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V6_TEST := $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v6
CPB_V6_SEPARATION_TEST := $(BUILD_DIR)/test_cpb_v6_variance_gradient
.PHONY: build-cpb-v6-engineering
build-cpb-v6-engineering: $(CPB_V6_TEST) $(CPB_V6_SEPARATION_TEST)
$(BUILD_DIR)/test_causal_patch_bottleneck_mae_v6.o: $(CPB_V6_RESEARCH_ROOT)/tests/v6_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V6_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_cpb_v6_variance_gradient.o: $(CPB_V6_RESEARCH_ROOT)/tests/v6_separation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V6_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V6_TEST): $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v6.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CPB_V6_SEPARATION_TEST): $(BUILD_DIR)/test_cpb_v6_variance_gradient.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/test_causal_patch_bottleneck_mae_v6.d $(BUILD_DIR)/test_cpb_v6_variance_gradient.d
include $(CPB_V6_RESEARCH_ROOT)/evaluation/evaluation.mk
