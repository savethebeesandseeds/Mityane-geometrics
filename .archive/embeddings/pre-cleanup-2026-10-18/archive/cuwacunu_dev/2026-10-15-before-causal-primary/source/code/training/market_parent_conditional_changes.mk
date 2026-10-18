# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separately declared RPB-v27.market-01 parent-conditional branch; new candidate objects only.
MARKET_PARENT_CONDITIONAL_CHANGES := $(BUILD_DIR)/market_parent_conditional_changes
MARKET_PARENT_CONDITIONAL_CPPFLAGS := $(SEPARATE_CPPFLAGS) -Icode/encoders/rpb_conditioned/include -Icode/encoders/rpb_conditional/include
.PHONY: build-market-parent-conditional-changes
build-market-parent-conditional-changes: $(MARKET_PARENT_CONDITIONAL_CHANGES)
$(BUILD_DIR)/market_parent_conditional_changes_main.o: code/training/src/market_parent_conditional_changes_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(MARKET_PARENT_CONDITIONAL_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_PARENT_CONDITIONAL_CHANGES): $(BUILD_DIR)/market_parent_conditional_changes_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_parent_conditional_changes_main.d


