# Independent frozen-level engineering fixture; existing v23 target is preserved.
V23_FROZEN_TEST := $(V23_TEST_DIR)/frozen_level_test
.PHONY: test-rpb-v23-frozen
test-rpb-v23-frozen: $(V23_FROZEN_TEST)
	$(V23_FROZEN_TEST)
$(V23_TEST_DIR)/frozen_level_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v23/tests/frozen_level_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V23_FROZEN_TEST): $(V23_TEST_DIR)/frozen_level_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V23_TEST_DIR)/frozen_level_test.d
