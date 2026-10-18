# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Optional v28 artificial admission; portable context build paths only.
V28_TEST_DIR := $(RPB_TEST_DIR)/relative_changes
V28_TEST := $(V28_TEST_DIR)/relative_change_test
.PHONY: rpb-v28 test-rpb-v28
rpb-v28: $(V28_TEST)
test-rpb-v28: rpb-v28
	$(V28_TEST)
$(V28_TEST_DIR)/relative_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v28/tests/relative_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V28_TEST): $(V28_TEST_DIR)/relative_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V28_TEST_DIR)/relative_change_test.d
