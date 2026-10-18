# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Optional RPB-v25 engineering; historical targets/defaults remain unchanged.
V25_TEST_DIR := $(RPB_TEST_DIR)/energy_changes
V25_TEST := $(V25_TEST_DIR)/energy_change_test
.PHONY: rpb-v25 test-rpb-v25
rpb-v25: $(V25_TEST)
test-rpb-v25: rpb-v25
	$(V25_TEST)
$(V25_TEST_DIR)/energy_change_test.o: code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v25/tests/energy_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V25_TEST): $(V25_TEST_DIR)/energy_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V25_TEST_DIR)/energy_change_test.d
