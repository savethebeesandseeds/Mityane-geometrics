# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Portable optional v27 fixture build-path adapter; primary stays unchanged.
V27_TEST_DIR := $(RPB_TEST_DIR)/parent_conditional_changes
V27_TEST := $(V27_TEST_DIR)/parent_conditional_change_test
.PHONY: rpb-v27 test-rpb-v27
rpb-v27: $(V27_TEST)
test-rpb-v27: rpb-v27
	$(V27_TEST)
$(V27_TEST_DIR)/parent_conditional_change_test.o: code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v27/tests/parent_conditional_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V27_TEST): $(V27_TEST_DIR)/parent_conditional_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V27_TEST_DIR)/parent_conditional_change_test.d
