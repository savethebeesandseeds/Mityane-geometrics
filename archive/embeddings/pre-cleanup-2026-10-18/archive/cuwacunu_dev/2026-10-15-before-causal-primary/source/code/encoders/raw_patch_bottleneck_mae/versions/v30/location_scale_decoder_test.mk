# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
LOCATION_SCALE_ROOT := code/encoders/rpb_location_scale
LOCATION_SCALE_CPPFLAGS := $(RELATIVE_CPPFLAGS) -I$(LOCATION_SCALE_ROOT)/include
LOCATION_SCALE_TEST := $(BUILD_DIR)/test_location_scale_decoder
.PHONY: build-v30-engineering test-v30-location-scale
build-v30-engineering: $(LOCATION_SCALE_TEST)
test-v30-location-scale: build-v30-engineering
	$(LOCATION_SCALE_TEST)
$(BUILD_DIR)/location_scale_decoder_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v30/tests/location_scale_decoder_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LOCATION_SCALE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(LOCATION_SCALE_TEST): $(BUILD_DIR)/location_scale_decoder_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/location_scale_decoder_test.d
