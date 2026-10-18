# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separately declared RPB-v25.market-01 objective; new candidate objects only.
MARKET_CHANGE_ENERGY := $(BUILD_DIR)/market_change_energy
MARKET_ENERGY_CPPFLAGS := $(SEPARATE_CPPFLAGS) -Icode/encoders/rpb_energy/include
.PHONY: build-market-change-energy
build-market-change-energy: $(MARKET_CHANGE_ENERGY)
$(BUILD_DIR)/market_change_energy_main.o: code/training/src/market_change_energy_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(MARKET_ENERGY_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_CHANGE_ENERGY): $(BUILD_DIR)/market_change_energy_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_change_energy_main.d
