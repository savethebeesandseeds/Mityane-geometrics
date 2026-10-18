# Bounded TRAIN preprocessing witness; no encoder/head/quality execution.
STREAMING_SCALER_ROOT := code/encoders/rpb_spectral
STREAMING_SCALER_TEST := $(BUILD_DIR)/test_streaming_scaler

.PHONY: build-streaming-scaler test-streaming-scaler
build-streaming-scaler: $(STREAMING_SCALER_TEST)
test-streaming-scaler: build-streaming-scaler
	$(STREAMING_SCALER_TEST)

$(BUILD_DIR)/test_streaming_scaler.o: code/encoders/raw_patch_bottleneck_mae/shared/tests/streaming_scaler_test.cpp $(STREAMING_SCALER_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(STREAMING_SCALER_TEST): $(BUILD_DIR)/test_streaming_scaler.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/test_streaming_scaler.d
