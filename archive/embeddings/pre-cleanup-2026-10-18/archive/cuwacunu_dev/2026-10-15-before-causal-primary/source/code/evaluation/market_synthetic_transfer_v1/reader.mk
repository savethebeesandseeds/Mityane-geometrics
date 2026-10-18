# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Saved-feature-only reader. Its inherited sources are byte-identical MIT copies.
TRANSFER_READOUT_ROOT := code/evaluation/market_synthetic_transfer_v1
TRANSFER_READOUT_CPPFLAGS := $(CPPFLAGS) -I$(TRANSFER_READOUT_ROOT)/deps/include
TRANSFER_READOUT_OBJECTS := $(BUILD_DIR)/market_synthetic_transfer_read_main.o \
  $(BUILD_DIR)/market_synthetic_transfer_data.o \
  $(BUILD_DIR)/market_synthetic_transfer_feature_harness.o \
  $(BUILD_DIR)/market_synthetic_transfer_fixed_feature_readouts.o
TRANSFER_READOUT := $(BUILD_DIR)/market_synthetic_transfer_read

.PHONY: build-market-synthetic-transfer-read
build-market-synthetic-transfer-read: $(TRANSFER_READOUT)

$(BUILD_DIR)/market_synthetic_transfer_read_main.o: $(TRANSFER_READOUT_ROOT)/read_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(TRANSFER_READOUT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/market_synthetic_transfer_%.o: $(TRANSFER_READOUT_ROOT)/deps/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(TRANSFER_READOUT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(TRANSFER_READOUT): $(TRANSFER_READOUT_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

-include $(TRANSFER_READOUT_OBJECTS:.o=.d)
