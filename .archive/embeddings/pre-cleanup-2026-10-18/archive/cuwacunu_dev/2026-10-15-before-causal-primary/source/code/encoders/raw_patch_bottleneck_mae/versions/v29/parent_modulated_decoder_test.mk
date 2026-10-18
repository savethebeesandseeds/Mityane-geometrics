# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
MODULATED_ROOT := code/encoders/rpb_modulated
MODULATED_CPPFLAGS := $(RELATIVE_CPPFLAGS) -I$(MODULATED_ROOT)/include
MODULATED_TEST := $(BUILD_DIR)/test_parent_modulated_decoder
.PHONY: build-v29-engineering test-v29-modulated
build-v29-engineering: $(MODULATED_TEST)
test-v29-modulated: build-v29-engineering
	$(MODULATED_TEST)
$(BUILD_DIR)/parent_modulated_decoder_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v29/tests/parent_modulated_decoder_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(MODULATED_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MODULATED_TEST): $(BUILD_DIR)/parent_modulated_decoder_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/parent_modulated_decoder_test.d
