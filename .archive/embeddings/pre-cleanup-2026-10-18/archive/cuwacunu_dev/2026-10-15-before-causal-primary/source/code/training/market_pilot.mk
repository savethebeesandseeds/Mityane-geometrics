MARKET_PILOT := $(BUILD_DIR)/market_history_pilot
CUDA_RUNTIME_INCLUDE ?= /usr/local/cuda-12.4/targets/x86_64-linux/include
.PHONY: build-market-pilot
build-market-pilot: $(MARKET_PILOT)
$(BUILD_DIR)/market_pilot_main.o: code/training/src/market_pilot_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_PILOT): $(BUILD_DIR)/market_pilot_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_pilot_main.d
