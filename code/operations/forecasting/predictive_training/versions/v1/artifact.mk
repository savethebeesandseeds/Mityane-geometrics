# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
ifndef PREDICTIVE_ARTIFACT_V1_INCLUDED
PREDICTIVE_ARTIFACT_V1_INCLUDED := 1
ifeq ($(strip $(PREDICTIVE_V1_CPPFLAGS)),)
include code/operations/forecasting/predictive_training/versions/v1/v1.mk
endif
PREDICTIVE_ARTIFACT_V1_CPPFLAGS := $(PREDICTIVE_V1_CPPFLAGS) -Icode/shared/artifacts/versions/v1/include
PREDICTIVE_SNAPSHOT_TEST := $(BUILD_DIR)/test_predictive_snapshot
.PHONY: build-predictive-snapshot test-predictive-snapshot
build-predictive-snapshot: $(PREDICTIVE_SNAPSHOT_TEST)
test-predictive-snapshot: build-predictive-snapshot
	$(PREDICTIVE_SNAPSHOT_TEST)
$(BUILD_DIR)/predictive_snapshot_test.o: code/operations/forecasting/predictive_training/versions/v1/tests/artifact_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(PREDICTIVE_ARTIFACT_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(PREDICTIVE_SNAPSHOT_TEST): $(BUILD_DIR)/predictive_snapshot_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/predictive_snapshot_test.d
endif
