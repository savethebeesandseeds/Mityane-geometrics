# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
CONDITIONAL_ROOT := code/encoders/rpb_conditional
CONDITIONAL_CPPFLAGS := $(CONDITIONED_CPPFLAGS) -I$(CONDITIONAL_ROOT)/include
CONDITIONAL_TEST := $(BUILD_DIR)/test_parent_conditional_changes
.PHONY: build-v27-engineering test-v27-parent-conditional
build-v27-engineering: $(CONDITIONAL_TEST)
test-v27-parent-conditional: build-v27-engineering
	$(CONDITIONAL_TEST)
$(BUILD_DIR)/parent_conditional_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v27/tests/parent_conditional_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CONDITIONAL_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CONDITIONAL_TEST): $(BUILD_DIR)/parent_conditional_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/parent_conditional_change_test.d
