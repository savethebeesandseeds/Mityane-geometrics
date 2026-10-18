# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separately declared RPB-v29.market-01 parent-modulated decoder; new candidate objects only.
MARKET_PARENT_MODULATED_CHANGES := $(BUILD_DIR)/market_parent_modulated_decoder
MARKET_PARENT_MODULATED_CPPFLAGS := $(SEPARATE_CPPFLAGS) -Icode/encoders/rpb_conditioned/include -Icode/encoders/rpb_relative/include -Icode/encoders/rpb_modulated/include
.PHONY: build-market-parent-modulated-decoder
build-market-parent-modulated-decoder: $(MARKET_PARENT_MODULATED_CHANGES)
$(BUILD_DIR)/market_parent_modulated_decoder_main.o: code/training/src/market_parent_modulated_decoder_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(MARKET_PARENT_MODULATED_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_PARENT_MODULATED_CHANGES): $(BUILD_DIR)/market_parent_modulated_decoder_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_parent_modulated_decoder_main.d


