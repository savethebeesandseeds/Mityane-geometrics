# Isolated candidate API and artificial CUDA engineering admission.
V21_TEST_DIR := $(RPB_TEST_DIR)/learned_ordered_relation
V21_CPPFLAGS := -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS)
V21_MODEL_TEST := $(V21_TEST_DIR)/learned_ordered_relation_test
V21_HEADERS := $(wildcard $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/*.h)

.PHONY: rpb-v21 test-rpb-v21
rpb-v21: $(V21_MODEL_TEST)
test-rpb-v21: rpb-v21
	$(V21_MODEL_TEST) code/encoders/raw_patch_bottleneck_mae/versions/v21/config/learned_ordered_relation.conf

$(V21_TEST_DIR)/learned_ordered_relation_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v21/tests/learned_ordered_relation_test.cpp $(V21_HEADERS)
	mkdir -p "$(@D)"
	$(CXX) $(V21_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V21_MODEL_TEST): $(V21_TEST_DIR)/learned_ordered_relation_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(V21_TEST_DIR)/*.d)
