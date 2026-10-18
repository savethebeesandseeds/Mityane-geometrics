EVALUATION_ROOT := code/evaluation
EVALUATION_CPPFLAGS := $(CPPFLAGS) -I$(EVALUATION_ROOT)/include
REGRESSION_OBJECT := $(BUILD_DIR)/fixed_regression.o
MARKET_READOUTS := $(BUILD_DIR)/market_fixed_regression
REGRESSION_TEST := $(BUILD_DIR)/test_fixed_regression

.PHONY: build-market-readouts test-market-readouts
build-market-readouts: $(MARKET_READOUTS)
test-market-readouts: $(REGRESSION_TEST)
	$(REGRESSION_TEST)
$(REGRESSION_OBJECT): $(EVALUATION_ROOT)/src/fixed_regression.cpp
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/market_readouts_main.o: $(EVALUATION_ROOT)/src/market_readouts_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/test_fixed_regression.o: $(EVALUATION_ROOT)/tests/fixed_regression_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MARKET_READOUTS): $(BUILD_DIR)/market_readouts_main.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(REGRESSION_TEST): $(BUILD_DIR)/test_fixed_regression.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/fixed_regression.d $(BUILD_DIR)/market_readouts_main.d $(BUILD_DIR)/test_fixed_regression.d
