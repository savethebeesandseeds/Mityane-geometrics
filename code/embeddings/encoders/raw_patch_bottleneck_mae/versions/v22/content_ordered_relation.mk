# Isolated candidate API and artificial CUDA engineering admission.
V22_TEST_DIR := $(RPB_TEST_DIR)/content_ordered_relation
V22_CPPFLAGS := -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS)
V22_MODEL_TEST := $(V22_TEST_DIR)/content_ordered_relation_test
V22_HEADERS := $(wildcard $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/*.h)

.PHONY: rpb-v22 test-rpb-v22
rpb-v22: $(V22_MODEL_TEST)
test-rpb-v22: rpb-v22
	$(V22_MODEL_TEST) code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/config/content_ordered_relation.conf

$(V22_TEST_DIR)/content_ordered_relation_test.o: code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/tests/content_ordered_relation_test.cpp $(V22_HEADERS)
	mkdir -p "$(@D)"
	$(CXX) $(V22_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V22_MODEL_TEST): $(V22_TEST_DIR)/content_ordered_relation_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(V22_TEST_DIR)/*.d)
