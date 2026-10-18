# Separately tagged RPB-v23 producer; preserves the completed study runner.
MARKET_CHANGES := $(BUILD_DIR)/market_changes
.PHONY: build-market-changes
build-market-changes: $(MARKET_CHANGES)
$(BUILD_DIR)/market_changes_main.o: code/training/src/market_changes_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CHANGE_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_CHANGES): $(BUILD_DIR)/market_changes_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_changes_main.d
