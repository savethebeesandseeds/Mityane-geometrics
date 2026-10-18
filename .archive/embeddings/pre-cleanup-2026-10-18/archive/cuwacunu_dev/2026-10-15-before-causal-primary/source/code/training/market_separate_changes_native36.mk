# Additive, separately declared RPB-v24.market-01.alt-01 native36 experiment.
MARKET_SEPARATE_NATIVE36 := $(BUILD_DIR)/market_separate_changes_native36
.PHONY: build-market-separate-changes-native36
build-market-separate-changes-native36: $(MARKET_SEPARATE_NATIVE36)
$(BUILD_DIR)/market_separate_changes_native36_main.o: code/training/src/market_separate_changes_native36_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(SEPARATE_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_SEPARATE_NATIVE36): $(BUILD_DIR)/market_separate_changes_native36_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_separate_changes_native36_main.d
include code/training/market_separate_native36_test.mk
