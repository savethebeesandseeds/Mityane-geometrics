# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Optional v30 artificial admission; portable context build paths only.
V30_TEST_DIR := $(RPB_TEST_DIR)/location_scale_decoder
V30_TEST := $(V30_TEST_DIR)/location_scale_decoder_test
.PHONY: rpb-v30 test-rpb-v30
rpb-v30: $(V30_TEST)
test-rpb-v30: rpb-v30
	$(V30_TEST)
$(V30_TEST_DIR)/location_scale_decoder_test.o: code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v30/tests/location_scale_decoder_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V30_TEST): $(V30_TEST_DIR)/location_scale_decoder_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V30_TEST_DIR)/location_scale_decoder_test.d
