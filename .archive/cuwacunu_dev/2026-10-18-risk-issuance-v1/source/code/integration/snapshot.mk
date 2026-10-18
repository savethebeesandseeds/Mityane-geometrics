# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
SNAPSHOT_CPPFLAGS := $(CPPFLAGS) $(PREDICTIVE_V1_CPPFLAGS) -Icode/artifacts/versions/v1/include
include code/networks/predictive_training/versions/v1/artifact.mk
ADOPT_ENCODER_SNAPSHOT := $(BUILD_DIR)/adopt_encoder_snapshot
ENCODER_SNAPSHOT_TEST := $(BUILD_DIR)/test_encoder_snapshot
.PHONY: build-model-snapshots test-model-snapshots test-encoder-snapshot test-snapshot-pipeline
build-model-snapshots: $(ADOPT_ENCODER_SNAPSHOT) $(PREDICTIVE_SNAPSHOT_TEST) $(ENCODER_SNAPSHOT_TEST)
test-model-snapshots: test-artifacts-v1 test-predictive-snapshot test-encoder-snapshot
test-encoder-snapshot: $(ENCODER_SNAPSHOT_TEST)
	$(ENCODER_SNAPSHOT_TEST)
test-snapshot-pipeline: build-latent-market-v1 build-latent-market-v1-fixture
	python3 code/latent_market/versions/v1/tests/snapshot_pipeline_test.py $(BUILD_DIR)
$(BUILD_DIR)/adopt_encoder_snapshot.o: code/integration/src/adopt_encoder_snapshot.cpp
	mkdir -p "$(@D)"
	$(CXX) $(SNAPSHOT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(ADOPT_ENCODER_SNAPSHOT): $(BUILD_DIR)/adopt_encoder_snapshot.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/encoder_snapshot_test.o: code/integration/tests/encoder_snapshot_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(SNAPSHOT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(ENCODER_SNAPSHOT_TEST): $(BUILD_DIR)/encoder_snapshot_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/adopt_encoder_snapshot.d $(BUILD_DIR)/encoder_snapshot_test.d
