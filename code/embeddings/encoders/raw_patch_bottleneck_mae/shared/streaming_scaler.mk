# Bounded TRAIN preprocessing witness; no encoder/head/quality execution.
STREAMING_SCALER_TEST_DIR := $(RPB_TEST_DIR)/streaming_scaler
STREAMING_SCALER_CPPFLAGS := -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS)
STREAMING_SCALER_TEST := $(STREAMING_SCALER_TEST_DIR)/streaming_scaler_test

.PHONY: build-streaming-scaler test-streaming-scaler
build-streaming-scaler: $(STREAMING_SCALER_TEST)
test-streaming-scaler: build-streaming-scaler
	$(STREAMING_SCALER_TEST)

$(STREAMING_SCALER_TEST_DIR)/streaming_scaler_test.o: code/embeddings/encoders/raw_patch_bottleneck_mae/shared/tests/streaming_scaler_test.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h
	mkdir -p "$(@D)"
	$(CXX) $(STREAMING_SCALER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(STREAMING_SCALER_TEST): $(STREAMING_SCALER_TEST_DIR)/streaming_scaler_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(STREAMING_SCALER_TEST_DIR)/*.d)
