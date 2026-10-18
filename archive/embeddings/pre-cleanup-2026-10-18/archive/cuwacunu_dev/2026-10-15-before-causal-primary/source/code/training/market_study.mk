# Additive continuation/capacity runner. No default targets or frozen pilot edits.
MARKET_STUDY := $(BUILD_DIR)/market_study
CUDA_RUNTIME_INCLUDE ?= /usr/local/cuda-12.4/targets/x86_64-linux/include
.PHONY: build-market-study
build-market-study: $(MARKET_STUDY)
$(BUILD_DIR)/market_study_main.o: code/training/src/market_study_main.cpp code/training/include/cuwacunu/training/market_data.h
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_STUDY): $(BUILD_DIR)/market_study_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_study_main.d
