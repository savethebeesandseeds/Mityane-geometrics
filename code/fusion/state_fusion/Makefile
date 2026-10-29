# State fusion. CMakeLists.txt is the build description. The managed image has
# no CMake, so this file compiles the package with g++ against the inherited
# LibTorch CUDA build. Products stay under /opt/mityane-geometrics.
HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
BUILD_DIR ?= /opt/mityane-geometrics/fusion/build/state-fusion
CXX ?= g++
CPPFLAGS := -I$(HERE)/include \
	-isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include \
	-D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS := -std=c++17 -O2 -g0 -Wall -Wextra -Wpedantic -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(LIBTORCH)/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread

LIB_SRCS := src/blocks.cpp src/embeddings.cpp src/model.cpp src/synthetic.cpp
LIB_OBJS := $(addprefix $(BUILD_DIR)/,$(notdir $(LIB_SRCS:.cpp=.o)))
INVARIANCE := $(BUILD_DIR)/state_fusion_invariance
FIDELITY := $(BUILD_DIR)/state_fusion_fidelity
TRAIN := $(BUILD_DIR)/state_fusion_train

.PHONY: build test train

build: $(INVARIANCE) $(FIDELITY) $(TRAIN)

test: build
	$(INVARIANCE)
	$(FIDELITY)
	$(TRAIN)

train: $(TRAIN)
	$(TRAIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(INVARIANCE): tests/test_invariance.cpp $(LIB_OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(FIDELITY): tests/test_fidelity.cpp $(LIB_OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(TRAIN): src/train.cpp $(LIB_OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB_OBJS) $(LDFLAGS) $(LDLIBS) -o $@

-include $(BUILD_DIR)/*.d
