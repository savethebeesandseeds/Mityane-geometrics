# Optional RPB-v23 portable candidate; no market ingestion dependency.
V23_TEST_DIR := $(RPB_TEST_DIR)/consecutive_changes
V23_TEST := $(V23_TEST_DIR)/consecutive_change_test
.PHONY: rpb-v23 test-rpb-v23
rpb-v23: $(V23_TEST)
test-rpb-v23: rpb-v23
	$(V23_TEST)
$(V23_TEST_DIR)/consecutive_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v23/tests/consecutive_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V23_TEST): $(V23_TEST_DIR)/consecutive_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V23_TEST_DIR)/consecutive_change_test.d
