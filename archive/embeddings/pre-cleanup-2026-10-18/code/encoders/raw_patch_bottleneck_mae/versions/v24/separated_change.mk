# Separate RPB-v24 candidate; historical targets/defaults stay fixed.
V24_TEST_DIR := $(RPB_TEST_DIR)/separated_changes
V24_TEST := $(V24_TEST_DIR)/separated_change_test
.PHONY: rpb-v24 test-rpb-v24
rpb-v24: $(V24_TEST)
test-rpb-v24: rpb-v24
	$(V24_TEST)
$(V24_TEST_DIR)/separated_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v24/tests/separated_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V24_TEST): $(V24_TEST_DIR)/separated_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V24_TEST_DIR)/separated_change_test.d
