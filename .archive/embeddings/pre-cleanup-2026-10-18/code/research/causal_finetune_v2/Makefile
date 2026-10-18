# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Invoke from an admitted source capsule root inside the existing cuwacunu container.
ifeq ($(wildcard /.dockerenv),)
$(error Build and run this research only inside the existing managed GPU container)
endif
CXX := g++
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/cuwacunu/build/causal-finetune-v2
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread
CPB_V2_RESEARCH_ROOT := code/research/causal_finetune_v2
CPB_V2_TRAIN := $(BUILD_DIR)/causal_finetune_v2_train
.DEFAULT_GOAL := all
.PHONY: all build-cpb-v2-train
all: build-cpb-v2-engineering build-cpb-v2-train build-cpb-v2-quality
build-cpb-v2-train: $(CPB_V2_TRAIN)
$(BUILD_DIR)/causal_finetune_v2_train.o: $(CPB_V2_RESEARCH_ROOT)/train_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V2_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V2_TRAIN): $(BUILD_DIR)/causal_finetune_v2_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/causal_finetune_v2_train.d
include code/encoders/causal_patch_bottleneck_mae/versions/v2/v2.mk
include $(CPB_V2_RESEARCH_ROOT)/evaluation/evaluation.mk
