# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Optional v26 artificial admission; portable build-path adapter only.
V26_TEST_DIR := $(RPB_TEST_DIR)/conditioned_changes
V26_TEST := $(V26_TEST_DIR)/conditioned_change_test
.PHONY: rpb-v26 test-rpb-v26
rpb-v26: $(V26_TEST)
test-rpb-v26: rpb-v26
	$(V26_TEST)
$(V26_TEST_DIR)/conditioned_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v26/tests/conditioned_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V26_TEST): $(V26_TEST_DIR)/conditioned_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V26_TEST_DIR)/conditioned_change_test.d
