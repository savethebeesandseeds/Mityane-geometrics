# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Frozen V30 full-API transfer; no training, decoder or readout dependencies.
LOCATION_SCALE_TRANSFER_EXTRACT := $(BUILD_DIR)/market_location_scale_synthetic_transfer_extract
LOCATION_SCALE_TRANSFER_CPPFLAGS := $(SEPARATE_CPPFLAGS) -Icode/encoders/rpb_conditioned/include -Icode/encoders/rpb_relative/include -Icode/encoders/rpb_location_scale/include
.PHONY: build-market-location-scale-synthetic-transfer-extract
build-market-location-scale-synthetic-transfer-extract: $(LOCATION_SCALE_TRANSFER_EXTRACT)
$(BUILD_DIR)/market_location_scale_synthetic_transfer_extract.o: code/evaluation/market_location_scale_synthetic_transfer_v1/extract_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LOCATION_SCALE_TRANSFER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(LOCATION_SCALE_TRANSFER_EXTRACT): $(BUILD_DIR)/market_location_scale_synthetic_transfer_extract.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_location_scale_synthetic_transfer_extract.d
