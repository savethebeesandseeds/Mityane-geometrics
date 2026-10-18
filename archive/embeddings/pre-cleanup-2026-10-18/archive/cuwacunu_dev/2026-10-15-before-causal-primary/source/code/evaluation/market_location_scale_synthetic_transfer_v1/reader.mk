# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# New metadata adapter; eight inherited fixed-head dependencies remain unchanged.
LOCATION_SCALE_TRANSFER_DEPS := code/evaluation/market_synthetic_transfer_v1/deps
LOCATION_SCALE_TRANSFER_READ_CPPFLAGS := $(CPPFLAGS) -I$(LOCATION_SCALE_TRANSFER_DEPS)/include
LOCATION_SCALE_TRANSFER_READ_OBJECTS := $(BUILD_DIR)/market_location_scale_synthetic_transfer_read_main.o \
  $(BUILD_DIR)/market_location_scale_synthetic_transfer_data.o \
  $(BUILD_DIR)/market_location_scale_synthetic_transfer_feature_harness.o \
  $(BUILD_DIR)/market_location_scale_synthetic_transfer_fixed_feature_readouts.o
LOCATION_SCALE_TRANSFER_READ := $(BUILD_DIR)/market_location_scale_synthetic_transfer_read
.PHONY: build-market-location-scale-synthetic-transfer-read
build-market-location-scale-synthetic-transfer-read: $(LOCATION_SCALE_TRANSFER_READ)
$(BUILD_DIR)/market_location_scale_synthetic_transfer_read_main.o: code/evaluation/market_location_scale_synthetic_transfer_v1/read_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LOCATION_SCALE_TRANSFER_READ_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/market_location_scale_synthetic_transfer_%.o: $(LOCATION_SCALE_TRANSFER_DEPS)/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LOCATION_SCALE_TRANSFER_READ_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(LOCATION_SCALE_TRANSFER_READ): $(LOCATION_SCALE_TRANSFER_READ_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(LOCATION_SCALE_TRANSFER_READ_OBJECTS:.o=.d)
