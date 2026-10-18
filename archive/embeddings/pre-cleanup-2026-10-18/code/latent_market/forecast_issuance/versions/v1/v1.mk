# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
ifndef FORECAST_ISSUANCE_V1_INCLUDED
FORECAST_ISSUANCE_V1_INCLUDED := 1
ifeq ($(strip $(PREDICTIVE_V1_CPPFLAGS)),)
include code/networks/predictive_training/versions/v1/v1.mk
endif
ifeq ($(strip $(LD_EVAL_V1_OBJECT)),)
include code/evaluation/latent_dynamics/versions/v1/baselines.mk
endif
FORECAST_ISSUANCE_V1_ROOT := code/latent_market/forecast_issuance/versions/v1
FORECAST_ISSUANCE_V1_CPPFLAGS := $(CPPFLAGS) $(PREDICTIVE_V1_CPPFLAGS) \
  -Icode/artifacts/versions/v1/include -Icode/latent_market/versions/v1/include \
  -Icode/evaluation/latent_dynamics/versions/v1/include \
  -I$(FORECAST_ISSUANCE_V1_ROOT)/include
FORECAST_ISSUANCE_V1_TEST := $(BUILD_DIR)/forecast_issuance_v1_test
.PHONY: build-forecast-issuance-v1 test-forecast-issuance-v1
build-forecast-issuance-v1: $(FORECAST_ISSUANCE_V1_TEST)
test-forecast-issuance-v1: $(FORECAST_ISSUANCE_V1_TEST)
	$(FORECAST_ISSUANCE_V1_TEST)
$(BUILD_DIR)/forecast_issuance_v1_test.o: $(FORECAST_ISSUANCE_V1_ROOT)/tests/issuance_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(FORECAST_ISSUANCE_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(FORECAST_ISSUANCE_V1_TEST): $(BUILD_DIR)/forecast_issuance_v1_test.o $(LD_EVAL_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/forecast_issuance_v1_test.d
endif
