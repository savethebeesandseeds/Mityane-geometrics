# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separately declared RPB-v30.market-01 location-scale decoder; new candidate objects only.
MARKET_LOCATION_SCALE_CHANGES := $(BUILD_DIR)/market_location_scale_decoder
MARKET_LOCATION_SCALE_CPPFLAGS := $(SEPARATE_CPPFLAGS) -Icode/encoders/rpb_conditioned/include -Icode/encoders/rpb_relative/include -Icode/encoders/rpb_location_scale/include
.PHONY: build-market-location-scale-decoder
build-market-location-scale-decoder: $(MARKET_LOCATION_SCALE_CHANGES)
$(BUILD_DIR)/market_location_scale_decoder_main.o: code/training/src/market_location_scale_decoder_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(MARKET_LOCATION_SCALE_CPPFLAGS) -Icode/evaluation/include -Icode/training/include -isystem $(CUDA_RUNTIME_INCLUDE) $(CXXFLAGS) -c $< -o $@
$(MARKET_LOCATION_SCALE_CHANGES): $(BUILD_DIR)/market_location_scale_decoder_main.o $(BUILD_DIR)/record_types.o $(REGRESSION_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_location_scale_decoder_main.d


