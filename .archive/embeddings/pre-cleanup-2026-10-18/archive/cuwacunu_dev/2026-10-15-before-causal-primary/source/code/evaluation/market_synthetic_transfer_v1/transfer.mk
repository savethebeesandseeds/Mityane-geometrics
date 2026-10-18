# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Frozen native36 market-checkpoint extraction; no training or decoder target.
MARKET_SYNTHETIC_TRANSFER_EXTRACT := $(BUILD_DIR)/market_synthetic_transfer_extract
.PHONY: build-market-synthetic-transfer-extract
build-market-synthetic-transfer-extract: $(MARKET_SYNTHETIC_TRANSFER_EXTRACT)
$(BUILD_DIR)/market_synthetic_transfer_extract.o: code/evaluation/market_synthetic_transfer_v1/extract_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(SEPARATE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MARKET_SYNTHETIC_TRANSFER_EXTRACT): $(BUILD_DIR)/market_synthetic_transfer_extract.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/market_synthetic_transfer_extract.d
