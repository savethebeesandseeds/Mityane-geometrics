# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
include code/networks/predictive_training/versions/v1/v1.mk
include code/evaluation/latent_dynamics/versions/v1/baselines.mk
LATENT_MARKET_V1_ROOT := code/latent_market/versions/v1
LATENT_MARKET_V1_CPPFLAGS := $(CPPFLAGS) $(PREDICTIVE_V1_CPPFLAGS) -I$(LATENT_MARKET_V1_ROOT)/include -I$(LD_EVAL_V1_ROOT)/include -Icode/encoders/causal_patch_bottleneck_mae/versions/v3/include -Icode/latent_market/forecast_issuance/versions/v1/include
LATENT_MARKET_V1_PREPARE := $(BUILD_DIR)/latent_market_v1_prepare
LATENT_MARKET_V1_STUDY := $(BUILD_DIR)/latent_market_v1_study
LATENT_MARKET_V1_UTILITY := $(BUILD_DIR)/latent_market_v1_utility
LATENT_MARKET_V1_TIMELINE_TEST := $(BUILD_DIR)/latent_market_v1_timeline_test
LATENT_MARKET_V1_FIXTURE := $(BUILD_DIR)/latent_market_v1_pipeline_fixture
LATENT_MARKET_V1_COORDINATE_FIXTURE := $(BUILD_DIR)/latent_market_v1_coordinate_fixture
.PHONY: build-latent-market-v1 test-latent-market-v1 test-latent-market-v1-timeline
build-latent-market-v1: $(LATENT_MARKET_V1_PREPARE) $(LATENT_MARKET_V1_STUDY) $(LATENT_MARKET_V1_UTILITY)
build-latent-market-v1-fixture: $(LATENT_MARKET_V1_FIXTURE) $(LATENT_MARKET_V1_COORDINATE_FIXTURE)
test-latent-market-v1: test-latent-market-v1-timeline test-latent-dynamics-v1-evaluation test-latent-risk-v1-readouts test-predictive-v1
test-latent-market-v1-timeline: $(LATENT_MARKET_V1_TIMELINE_TEST)
	$(LATENT_MARKET_V1_TIMELINE_TEST)
$(BUILD_DIR)/latent_market_v1_prepare.o: $(LATENT_MARKET_V1_ROOT)/src/prepare_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LATENT_MARKET_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/latent_market_v1_study.o: $(LATENT_MARKET_V1_ROOT)/src/study_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LATENT_MARKET_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/latent_market_v1_utility.o: $(LATENT_MARKET_V1_ROOT)/src/utility_main.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LATENT_MARKET_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/latent_market_v1_timeline_test.o: $(LATENT_MARKET_V1_ROOT)/tests/timeline_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(LATENT_MARKET_V1_ROOT)/include $(CXXFLAGS) -c $< -o $@
$(LATENT_MARKET_V1_PREPARE): $(BUILD_DIR)/latent_market_v1_prepare.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(LATENT_MARKET_V1_STUDY): $(BUILD_DIR)/latent_market_v1_study.o $(LD_EVAL_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(LATENT_MARKET_V1_UTILITY): $(BUILD_DIR)/latent_market_v1_utility.o $(LD_RISK_V1_OBJECT) $(LD_EVAL_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(LATENT_MARKET_V1_TIMELINE_TEST): $(BUILD_DIR)/latent_market_v1_timeline_test.o
	$(CXX) $^ -o $@
$(BUILD_DIR)/latent_market_v1_pipeline_fixture.o: $(LATENT_MARKET_V1_ROOT)/tests/pipeline_fixture.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(LATENT_MARKET_V1_ROOT)/include -Icode/artifacts/versions/v1/include $(CXXFLAGS) -c $< -o $@
$(LATENT_MARKET_V1_FIXTURE): $(BUILD_DIR)/latent_market_v1_pipeline_fixture.o
	$(CXX) $^ -o $@
$(BUILD_DIR)/latent_market_v1_coordinate_fixture.o: $(LATENT_MARKET_V1_ROOT)/tests/coordinate_fixture.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LATENT_MARKET_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(LATENT_MARKET_V1_COORDINATE_FIXTURE): $(BUILD_DIR)/latent_market_v1_coordinate_fixture.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/latent_market_v1_prepare.d $(BUILD_DIR)/latent_market_v1_study.d $(BUILD_DIR)/latent_market_v1_timeline_test.d
-include $(BUILD_DIR)/latent_market_v1_pipeline_fixture.d
-include $(BUILD_DIR)/latent_market_v1_coordinate_fixture.d
-include $(BUILD_DIR)/latent_market_v1_utility.d
