# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Invoke from an admitted source capsule root inside the existing cuwacunu container.
ifeq ($(wildcard /.dockerenv),)
$(error Build and run this research only inside the existing managed GPU container)
endif
CXX := g++
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/cuwacunu/build/causal-fresh-v4
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread
CPB_V4_RESEARCH_ROOT := code/research/causal_fresh_v4
CPB_V4_TRAIN := $(BUILD_DIR)/causal_fresh_v4_train
.DEFAULT_GOAL := all
.PHONY: all build-cpb-v4-train
all: build-cpb-v4-engineering build-cpb-v4-train build-cpb-v4-quality
build-cpb-v4-train: $(CPB_V4_TRAIN)
$(BUILD_DIR)/causal_fresh_v4_train.o: $(CPB_V4_RESEARCH_ROOT)/train_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V4_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V4_TRAIN): $(BUILD_DIR)/causal_fresh_v4_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_fresh_v4_train.d
include code/encoders/causal_patch_bottleneck_mae/versions/v4/v4.mk
include $(CPB_V4_RESEARCH_ROOT)/evaluation/evaluation.mk
