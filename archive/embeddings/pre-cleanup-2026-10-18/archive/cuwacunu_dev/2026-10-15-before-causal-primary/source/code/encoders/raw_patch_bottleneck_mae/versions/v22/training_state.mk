# Bounded artificial CUDA resume witness; independent of market evaluation.
TRAINING_STATE_TEST := $(BUILD_DIR)/test_training_state

.PHONY: build-training-state test-training-state
build-training-state: $(TRAINING_STATE_TEST)
test-training-state: build-training-state
	$(TRAINING_STATE_TEST)

$(BUILD_DIR)/training_state_test.o: code/encoders/raw_patch_bottleneck_mae/versions/v22/tests/training_state_test.cpp code/encoders/rpb_content/include/embedding/encoders/raw_patch_bottleneck_mae/training_state.h
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(TRAINING_STATE_TEST): $(BUILD_DIR)/training_state_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/training_state_test.d
