# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
RELATIVE_ROOT := code/encoders/rpb_relative
RELATIVE_CPPFLAGS := $(CONDITIONED_CPPFLAGS) -I$(RELATIVE_ROOT)/include
RELATIVE_TEST := $(BUILD_DIR)/test_relative_changes
.PHONY: build-v28-engineering test-v28-relative
build-v28-engineering: $(RELATIVE_TEST)
test-v28-relative: build-v28-engineering
	$(RELATIVE_TEST)
$(BUILD_DIR)/relative_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v28/tests/relative_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(RELATIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(RELATIVE_TEST): $(BUILD_DIR)/relative_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/relative_change_test.d
