# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Opt-in source-layout check; include after the integration Makefile.
LAYOUT_COMPATIBILITY_TEST := $(BUILD_DIR)/layout_compatibility
LAYOUT_COMPATIBILITY_CXXFLAGS := -std=c++20 -O0 -g0 -Wall -Wextra -MMD -MP
.PHONY: build-layout-compatibility
build-layout-compatibility: $(LAYOUT_COMPATIBILITY_TEST)
$(BUILD_DIR)/layout_compatibility.o: code/encoders/raw_patch_bottleneck_mae/tests/layout_compatibility.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) -Icode/encoders/rpb_changes/include -Icode/encoders/rpb_separate/include -Icode/encoders/rpb_energy/include -Icode/encoders/rpb_conditioned/include -Icode/encoders/rpb_conditional/include -Icode/encoders/rpb_relative/include -Icode/encoders/rpb_modulated/include -Icode/encoders/rpb_location_scale/include $(LAYOUT_COMPATIBILITY_CXXFLAGS) -c $< -o $@
$(LAYOUT_COMPATIBILITY_TEST): $(BUILD_DIR)/layout_compatibility.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/layout_compatibility.d
