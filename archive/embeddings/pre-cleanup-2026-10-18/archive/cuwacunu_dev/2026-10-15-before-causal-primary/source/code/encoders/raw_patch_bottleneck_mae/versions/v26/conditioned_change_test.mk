# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separate artificial CUDA admission; every historical target remains unchanged.
CONDITIONED_ROOT := code/encoders/rpb_conditioned
CONDITIONED_CPPFLAGS := $(SEPARATE_CPPFLAGS) -I$(CONDITIONED_ROOT)/include
CONDITIONED_TEST := $(BUILD_DIR)/test_conditioned_changes
.PHONY: build-v26-engineering test-v26-conditioned
build-v26-engineering: $(CONDITIONED_TEST)
test-v26-conditioned: build-v26-engineering
	$(CONDITIONED_TEST)
$(BUILD_DIR)/conditioned_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v26/tests/conditioned_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CONDITIONED_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CONDITIONED_TEST): $(BUILD_DIR)/conditioned_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/conditioned_change_test.d
