# Additive frozen-level recipe; measured v22/v23 runners stay unchanged.
FROZEN_CHANGE_TEST := $(BUILD_DIR)/test_frozen_changes
MARKET_FROZEN_CHANGES := $(BUILD_DIR)/market_frozen_changes
.PHONY: test-v23-frozen build-market-frozen-changes
test-v23-frozen: $(FROZEN_CHANGE_TEST)
	$(FROZEN_CHANGE_TEST)
build-market-frozen-changes: $(MARKET_FROZEN_CHANGES)
$(BUILD_DIR)/frozen_level_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v23/tests/frozen_level_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CHANGE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(FROZEN_CHANGE_TEST): $(BUILD_DIR)/frozen_level_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/market_frozen_changes_main.o: code/training/src/market_frozen_changes_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CHANGE_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_FROZEN_CHANGES): $(BUILD_DIR)/market_frozen_changes_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/frozen_level_test.d $(BUILD_DIR)/market_frozen_changes_main.d
