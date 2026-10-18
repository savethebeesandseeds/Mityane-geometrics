# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
ARTIFACTS_V1_ROOT := code/artifacts/versions/v1
ARTIFACTS_V1_CPPFLAGS := -I$(ARTIFACTS_V1_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
ARTIFACTS_V1_TEST := $(BUILD_DIR)/test_artifacts_v1
.PHONY: build-artifacts-v1-engineering test-artifacts-v1
build-artifacts-v1-engineering: $(ARTIFACTS_V1_TEST)
test-artifacts-v1: build-artifacts-v1-engineering
	$(ARTIFACTS_V1_TEST)
$(BUILD_DIR)/artifacts_v1_test.o: $(ARTIFACTS_V1_ROOT)/tests/sidecar_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(ARTIFACTS_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(ARTIFACTS_V1_TEST): $(BUILD_DIR)/artifacts_v1_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/artifacts_v1_test.d
