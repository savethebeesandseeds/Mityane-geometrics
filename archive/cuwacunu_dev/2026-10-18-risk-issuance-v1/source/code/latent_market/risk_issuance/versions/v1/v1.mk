# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
ifndef RISK_ISSUANCE_V1_INCLUDED
RISK_ISSUANCE_V1_INCLUDED := 1
ifeq ($(strip $(PREDICTIVE_V1_CPPFLAGS)),)
include code/networks/predictive_training/versions/v1/v1.mk
endif
ifeq ($(strip $(LD_RISK_V1_OBJECT)),)
include code/evaluation/latent_dynamics/versions/v1/baselines.mk
endif
RISK_ISSUANCE_V1_ROOT := code/latent_market/risk_issuance/versions/v1
RISK_ISSUANCE_V1_DC_SHA256 := $(shell python3 code/networks/direct_embedding_control/versions/v1/source_identity.py --root .)
RISK_ISSUANCE_V1_SOURCE_FILES := $(shell python3 $(RISK_ISSUANCE_V1_ROOT)/source_identity.py --root . --files)
RISK_ISSUANCE_V1_SOURCE_SHA256 := $(shell python3 $(RISK_ISSUANCE_V1_ROOT)/source_identity.py --root .)
RISK_ISSUANCE_V1_CPPFLAGS := $(CPPFLAGS) $(PREDICTIVE_V1_CPPFLAGS) \
  -Icode/latent_market/versions/v1/include -Icode/latent_market/forecast_issuance/versions/v1/include \
  -Icode/networks/direct_embedding_control/versions/v1/include -Icode/evaluation/latent_dynamics/versions/v1/include \
  -I$(RISK_ISSUANCE_V1_ROOT)/include \
  -DCUWACUNU_DIRECT_CONTROL_V1_SOURCE_SHA256=\"$(RISK_ISSUANCE_V1_DC_SHA256)\" \
  -DCUWACUNU_RISK_ISSUANCE_V1_SOURCE_SHA256=\"$(RISK_ISSUANCE_V1_SOURCE_SHA256)\"
RISK_ISSUANCE_V1_TEST := $(BUILD_DIR)/risk_issuance_v1_test
RISK_ISSUANCE_V1_PIPELINE := $(BUILD_DIR)/risk_issuance_v1_pipeline
.PHONY: test-risk-issuance-v1 test-risk-issuance-pipeline
test-risk-issuance-v1: $(RISK_ISSUANCE_V1_TEST)
	$(RISK_ISSUANCE_V1_TEST)
test-risk-issuance-pipeline: $(RISK_ISSUANCE_V1_PIPELINE)
	python3 $(RISK_ISSUANCE_V1_ROOT)/tests/pipeline_test.py $(RISK_ISSUANCE_V1_PIPELINE)
$(BUILD_DIR)/risk_issuance_v1_test.o: $(RISK_ISSUANCE_V1_ROOT)/tests/stage_test.cpp $(RISK_ISSUANCE_V1_SOURCE_FILES) $(RISK_ISSUANCE_V1_ROOT)/tests/fixture.h
	mkdir -p "$(@D)"
	$(CXX) $(RISK_ISSUANCE_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/risk_issuance_v1_pipeline.o: $(RISK_ISSUANCE_V1_ROOT)/src/pipeline_main.cpp $(RISK_ISSUANCE_V1_SOURCE_FILES) $(RISK_ISSUANCE_V1_ROOT)/tests/fixture.h
	mkdir -p "$(@D)"
	$(CXX) $(RISK_ISSUANCE_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(RISK_ISSUANCE_V1_TEST): $(BUILD_DIR)/risk_issuance_v1_test.o $(LD_EVAL_V1_OBJECT) $(LD_RISK_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(RISK_ISSUANCE_V1_PIPELINE): $(BUILD_DIR)/risk_issuance_v1_pipeline.o $(LD_EVAL_V1_OBJECT) $(LD_RISK_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/risk_issuance_v1_test.d $(BUILD_DIR)/risk_issuance_v1_pipeline.d
endif
