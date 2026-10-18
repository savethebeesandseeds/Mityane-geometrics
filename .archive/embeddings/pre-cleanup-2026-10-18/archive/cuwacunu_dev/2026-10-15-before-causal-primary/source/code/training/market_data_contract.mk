# Focused all-past data-contract witness. No encoder/head construction or fitting.
MARKET_DATA_CONTRACT_ROOT := code/training
MARKET_DATA_CONTRACT_TEST := $(BUILD_DIR)/test_market_data_contract

.PHONY: build-market-data-contract test-market-data-contract
build-market-data-contract: $(MARKET_DATA_CONTRACT_TEST)
test-market-data-contract: build-market-data-contract
	$(MARKET_DATA_CONTRACT_TEST)

$(BUILD_DIR)/test_market_data_contract.o: $(MARKET_DATA_CONTRACT_ROOT)/tests/market_data_contract_test.cpp $(MARKET_DATA_CONTRACT_ROOT)/include/cuwacunu/training/market_data.h
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) -I$(MARKET_DATA_CONTRACT_ROOT)/include $(CXXFLAGS) -c $< -o $@
$(MARKET_DATA_CONTRACT_TEST): $(BUILD_DIR)/test_market_data_contract.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/test_market_data_contract.d
