# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separately declared RPB-v28.market-01 raw relative changes; new candidate objects only.
MARKET_RELATIVE_CHANGES := $(BUILD_DIR)/market_relative_changes
MARKET_RELATIVE_CPPFLAGS := $(SEPARATE_CPPFLAGS) -Icode/encoders/rpb_conditioned/include -Icode/encoders/rpb_relative/include
.PHONY: build-market-relative-changes
build-market-relative-changes: $(MARKET_RELATIVE_CHANGES)
$(BUILD_DIR)/market_relative_changes_main.o: code/training/src/market_relative_changes_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(MARKET_RELATIVE_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_RELATIVE_CHANGES): $(BUILD_DIR)/market_relative_changes_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_relative_changes_main.d


