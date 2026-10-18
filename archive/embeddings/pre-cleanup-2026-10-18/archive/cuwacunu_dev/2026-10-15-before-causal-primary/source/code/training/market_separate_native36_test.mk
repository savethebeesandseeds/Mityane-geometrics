# Additive native36 engineering fixture; native40 target and artifacts stay fixed.
SEPARATE_NATIVE36_TEST := $(BUILD_DIR)/test_separate_changes_native36
.PHONY: test-v24-native36
test-v24-native36: $(SEPARATE_NATIVE36_TEST)
	$(SEPARATE_NATIVE36_TEST)
$(BUILD_DIR)/separated_change_native36_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v24/tests/separated_change_native36_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(SEPARATE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(SEPARATE_NATIVE36_TEST): $(BUILD_DIR)/separated_change_native36_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/separated_change_native36_test.d
