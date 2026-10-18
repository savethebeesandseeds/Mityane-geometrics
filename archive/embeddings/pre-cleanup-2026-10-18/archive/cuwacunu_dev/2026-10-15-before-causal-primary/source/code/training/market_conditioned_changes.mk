# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separately declared RPB-v26.market-01 bank conditioning; new candidate objects only.
MARKET_CONDITIONED_CHANGES := $(BUILD_DIR)/market_conditioned_changes
MARKET_CONDITIONED_CPPFLAGS := $(SEPARATE_CPPFLAGS) -Icode/encoders/rpb_conditioned/include
.PHONY: build-market-conditioned-changes
build-market-conditioned-changes: $(MARKET_CONDITIONED_CHANGES)
$(BUILD_DIR)/market_conditioned_changes_main.o: code/training/src/market_conditioned_changes_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(MARKET_CONDITIONED_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_CONDITIONED_CHANGES): $(BUILD_DIR)/market_conditioned_changes_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_conditioned_changes_main.d


