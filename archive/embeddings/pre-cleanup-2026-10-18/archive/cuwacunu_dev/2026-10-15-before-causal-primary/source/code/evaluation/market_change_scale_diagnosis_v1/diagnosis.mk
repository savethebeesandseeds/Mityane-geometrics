# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Archive arithmetic only: no encoder or fixed-head fitting object is linked.
MARKET_SCALE_DIAG_ROOT := code/evaluation/market_change_scale_diagnosis_v1
MARKET_SCALE_DIAG_OBJECT := $(BUILD_DIR)/market_change_scale_diagnosis_v1_main.o
MARKET_SCALE_DIAG_BINARY := $(BUILD_DIR)/market_change_scale_diagnosis
.PHONY: build-market-change-scale-diagnosis
build-market-change-scale-diagnosis: $(MARKET_SCALE_DIAG_BINARY)
$(MARKET_SCALE_DIAG_OBJECT): $(MARKET_SCALE_DIAG_ROOT)/main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MARKET_SCALE_DIAG_BINARY): $(MARKET_SCALE_DIAG_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(MARKET_SCALE_DIAG_OBJECT:.o=.d)
