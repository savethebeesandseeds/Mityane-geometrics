# Temporal Fusion Transformer. This tree is the whole build. The managed image
# has no CMake, so this file compiles the imported sources with g++. Products
# stay under /opt/mityane-geometrics. LibTorch stays at its inherited path.
HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/mityane-geometrics/embeddings/build/temporal-fusion-transformer
CXX ?= g++
CPPFLAGS := -I$(HERE)/include \
	-isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include \
	-D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS := -std=c++17 -O2 -g0 -Wall -Wextra -Wpedantic -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(LIBTORCH)/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread

LIB_SRCS := src/tft.cpp src/checkpoint.cpp src/evaluation.cpp src/calibration.cpp
LIB_OBJS := $(addprefix $(BUILD_DIR)/,$(notdir $(LIB_SRCS:.cpp=.o)))
CPU_TESTS := tft_test tft_numerics_test tft_reference_test tft_learning_test \
	tft_long_range_test tft_mask_test compatibility_test checkpoint_test \
	evaluation_test calibration_test
CPU_BINS := $(addprefix $(BUILD_DIR)/,$(CPU_TESTS))
DEMO := $(BUILD_DIR)/tft_demo

.PHONY: build test test-cuda

build: $(CPU_BINS) $(DEMO)

test: build
	@set -e; \
	for name in $(CPU_TESTS); do \
	  echo "RUN $$name"; \
	  $(BUILD_DIR)/$$name --device cpu; \
	done
	bash $(HERE)/tests/demo_workflow.sh $(DEMO) $(BUILD_DIR)/demo-workflow

test-cuda: build
	@set -e; \
	for name in $(CPU_TESTS); do \
	  echo "RUN $$name cuda"; \
	  $(BUILD_DIR)/$$name --device cuda; \
	done

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%: tests/%.cpp $(LIB_OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(DEMO): examples/train_synthetic.cpp $(LIB_OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

-include $(BUILD_DIR)/*.d
