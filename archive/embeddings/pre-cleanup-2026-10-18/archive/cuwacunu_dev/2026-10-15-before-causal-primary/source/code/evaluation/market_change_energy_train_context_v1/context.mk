# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Read saved TRAIN predictions only; no encoder or fitting object is linked.
MARKET_ENERGY_CONTEXT_ROOT := code/evaluation/market_change_energy_train_context_v1
MARKET_ENERGY_CONTEXT_OBJECT := $(BUILD_DIR)/market_change_energy_train_context_v1_main.o
MARKET_ENERGY_CONTEXT_BINARY := $(BUILD_DIR)/market_change_energy_train_context
.PHONY: build-market-change-energy-train-context
build-market-change-energy-train-context: $(MARKET_ENERGY_CONTEXT_BINARY)
$(MARKET_ENERGY_CONTEXT_OBJECT): $(MARKET_ENERGY_CONTEXT_ROOT)/main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MARKET_ENERGY_CONTEXT_BINARY): $(MARKET_ENERGY_CONTEXT_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(MARKET_ENERGY_CONTEXT_OBJECT:.o=.d)
