# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Independently compiled immutable copies of the unchanged fixed head recipes.
CPB_V4_EVAL_ROOT := code/research/causal_fresh_v4/evaluation
CPB_V4_EVAL_CPPFLAGS := -I$(CPB_V4_EVAL_ROOT)/deps/include \
  -Icode/encoders/causal_patch_bottleneck_mae/versions/v4/include \
  -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include \
  -D_GLIBCXX_USE_CXX11_ABI=1
CPB_V4_EXTRACT := $(BUILD_DIR)/causal_fresh_v4_extract
CPB_V4_MARKET_READ := $(BUILD_DIR)/causal_fresh_v4_market_read
CPB_V4_SYNTHETIC_READ := $(BUILD_DIR)/causal_fresh_v4_synthetic_read
CPB_V4_HEAD_OBJECTS := $(addprefix $(BUILD_DIR)/cpb_v4_head_,data.o feature_harness.o fixed_feature_readouts.o)
CPB_V4_EVAL_OBJECTS := $(addprefix $(BUILD_DIR)/cpb_v4_,extract_main.o market_read_main.o synthetic_read_main.o fixed_regression.o) $(CPB_V4_HEAD_OBJECTS)
.PHONY: build-cpb-v4-quality
build-cpb-v4-quality: $(CPB_V4_EXTRACT) $(CPB_V4_MARKET_READ) $(CPB_V4_SYNTHETIC_READ)
$(BUILD_DIR)/cpb_v4_%_main.o: $(CPB_V4_EVAL_ROOT)/%_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V4_EVAL_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v4_fixed_regression.o: $(CPB_V4_EVAL_ROOT)/deps/src/fixed_regression.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V4_EVAL_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v4_head_%.o: $(CPB_V4_EVAL_ROOT)/deps/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V4_EVAL_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_V4_EXTRACT): $(BUILD_DIR)/cpb_v4_extract_main.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CPB_V4_MARKET_READ): $(BUILD_DIR)/cpb_v4_market_read_main.o $(BUILD_DIR)/cpb_v4_fixed_regression.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CPB_V4_SYNTHETIC_READ): $(BUILD_DIR)/cpb_v4_synthetic_read_main.o $(CPB_V4_HEAD_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(CPB_V4_EVAL_OBJECTS:.o=.d)
