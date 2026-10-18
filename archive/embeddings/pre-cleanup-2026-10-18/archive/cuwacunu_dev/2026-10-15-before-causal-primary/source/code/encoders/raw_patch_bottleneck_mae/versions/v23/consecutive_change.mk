# Additive RPB-v23 candidate; primary and historical targets stay unchanged.
CHANGE_ROOT := code/encoders/rpb_changes
CHANGE_CPPFLAGS := $(CPPFLAGS) -I$(CHANGE_ROOT)/include
CHANGE_TEST := $(BUILD_DIR)/test_consecutive_changes
.PHONY: build-v23 test-v23
build-v23: $(CHANGE_TEST)
test-v23: build-v23
	$(CHANGE_TEST)
$(BUILD_DIR)/consecutive_change_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v23/tests/consecutive_change_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CHANGE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CHANGE_TEST): $(BUILD_DIR)/consecutive_change_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/consecutive_change_test.d
