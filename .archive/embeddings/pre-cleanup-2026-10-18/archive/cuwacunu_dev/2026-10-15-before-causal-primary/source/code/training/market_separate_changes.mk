# Additive native-prefix-preserving RPB-v24 experiment.
SEPARATE_CPPFLAGS := $(CHANGE_CPPFLAGS) -Icode/encoders/rpb_separate/include
SEPARATE_TEST := $(BUILD_DIR)/test_separate_changes
MARKET_SEPARATE := $(BUILD_DIR)/market_separate_changes
.PHONY: test-v24 build-market-separate-changes
test-v24: $(SEPARATE_TEST)
	$(SEPARATE_TEST)
build-market-separate-changes: $(MARKET_SEPARATE)
$(BUILD_DIR)/separated_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v24/tests/separated_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(SEPARATE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(SEPARATE_TEST): $(BUILD_DIR)/separated_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/market_separate_changes_main.o: code/training/src/market_separate_changes_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(SEPARATE_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_SEPARATE): $(BUILD_DIR)/market_separate_changes_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/separated_change_test.d $(BUILD_DIR)/market_separate_changes_main.d
