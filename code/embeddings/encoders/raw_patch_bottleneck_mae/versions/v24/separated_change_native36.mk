# Separate native36 engineering fixture; historical v24 targets stay fixed.
V24_NATIVE36_TEST_DIR := $(RPB_TEST_DIR)/separated_changes_native36
V24_NATIVE36_TEST := $(V24_NATIVE36_TEST_DIR)/separated_change_native36_test
.PHONY: rpb-v24-native36 test-rpb-v24-native36
rpb-v24-native36: $(V24_NATIVE36_TEST)
test-rpb-v24-native36: rpb-v24-native36
	$(V24_NATIVE36_TEST)
$(V24_NATIVE36_TEST_DIR)/separated_change_native36_test.o: code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v24/tests/separated_change_native36_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V24_NATIVE36_TEST): $(V24_NATIVE36_TEST_DIR)/separated_change_native36_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(V24_NATIVE36_TEST_DIR)/separated_change_native36_test.d
