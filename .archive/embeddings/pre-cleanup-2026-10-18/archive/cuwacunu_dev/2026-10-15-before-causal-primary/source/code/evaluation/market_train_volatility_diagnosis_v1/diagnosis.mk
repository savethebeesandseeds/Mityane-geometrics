# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Saved TRAIN arithmetic only. This target links no fitting or encoder object.
MARKET_TRAIN_DIAG_ROOT := code/evaluation/market_train_volatility_diagnosis_v1
MARKET_TRAIN_DIAG_OBJECT := $(BUILD_DIR)/market_train_volatility_diagnosis_v1_main.o
MARKET_TRAIN_DIAG_BINARY := $(BUILD_DIR)/market_train_volatility_diagnosis

.PHONY: build-market-train-volatility-diagnosis
build-market-train-volatility-diagnosis: $(MARKET_TRAIN_DIAG_BINARY)

$(MARKET_TRAIN_DIAG_OBJECT): $(MARKET_TRAIN_DIAG_ROOT)/main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(MARKET_TRAIN_DIAG_BINARY): $(MARKET_TRAIN_DIAG_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

-include $(MARKET_TRAIN_DIAG_OBJECT:.o=.d)
