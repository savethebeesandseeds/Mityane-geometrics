# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Optional v29 artificial admission; portable context build paths only.
V29_TEST_DIR := $(RPB_TEST_DIR)/parent_modulated_decoder
V29_TEST := $(V29_TEST_DIR)/parent_modulated_decoder_test
.PHONY: rpb-v29 test-rpb-v29
rpb-v29: $(V29_TEST)
test-rpb-v29: rpb-v29
	$(V29_TEST)
$(V29_TEST_DIR)/parent_modulated_decoder_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v29/tests/parent_modulated_decoder_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V29_TEST): $(V29_TEST_DIR)/parent_modulated_decoder_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V29_TEST_DIR)/parent_modulated_decoder_test.d
