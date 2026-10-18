# Bounded artificial CUDA resume witness; independent of quality protocols.
TRAINING_STATE_TEST_DIR := $(RPB_TEST_DIR)/training_state
TRAINING_STATE_CPPFLAGS := -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS)
TRAINING_STATE_TEST := $(TRAINING_STATE_TEST_DIR)/training_state_test

.PHONY: build-training-state test-training-state
build-training-state: $(TRAINING_STATE_TEST)
test-training-state: build-training-state
	$(TRAINING_STATE_TEST)

$(TRAINING_STATE_TEST_DIR)/training_state_test.o: code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/tests/training_state_test.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/training_state.h
	mkdir -p "$(@D)"
	$(CXX) $(TRAINING_STATE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(TRAINING_STATE_TEST): $(TRAINING_STATE_TEST_DIR)/training_state_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(TRAINING_STATE_TEST_DIR)/*.d)
