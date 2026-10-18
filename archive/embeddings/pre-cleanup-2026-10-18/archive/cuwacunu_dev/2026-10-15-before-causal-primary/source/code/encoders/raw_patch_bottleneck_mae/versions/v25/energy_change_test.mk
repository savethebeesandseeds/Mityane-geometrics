# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Separate artificial CUDA admission; historical v24 targets remain unchanged.
ENERGY_ROOT := code/encoders/rpb_energy
ENERGY_CPPFLAGS := $(SEPARATE_CPPFLAGS) -I$(ENERGY_ROOT)/include
ENERGY_TEST := $(BUILD_DIR)/test_energy_changes
.PHONY: build-v25-engineering test-v25-energy
build-v25-engineering: $(ENERGY_TEST)
test-v25-energy: build-v25-engineering
	$(ENERGY_TEST)
$(BUILD_DIR)/energy_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v25/tests/energy_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(ENERGY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(ENERGY_TEST): $(BUILD_DIR)/energy_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/energy_change_test.d
