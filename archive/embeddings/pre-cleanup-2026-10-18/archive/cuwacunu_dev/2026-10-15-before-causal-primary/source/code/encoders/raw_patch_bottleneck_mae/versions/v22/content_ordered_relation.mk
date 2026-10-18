# Primary RPB-v22 API and loader witnesses, with explicit historical v20 tests.
CONTENT_ROOT := code/encoders/rpb_content
CONTENT_INPUT_TEST := $(BUILD_DIR)/test_rpb_v22_content_input
PRIMARY_INPUT_TEST := $(BUILD_DIR)/test_primary_encoder_input
CONTENT_CPPFLAGS := $(CPPFLAGS)
CONTENT_HEADERS := $(wildcard $(CONTENT_ROOT)/include/cuwacunu/encoder/*.h $(CONTENT_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/*.h)

.PHONY: build-v22 test-v22 build-primary test-primary
build-v22: $(CONTENT_INPUT_TEST)

test-v22: build-v22
	$(CONTENT_INPUT_TEST) config/encoder_v22.conf

build-primary: $(CONTENT_INPUT_TEST) $(PRIMARY_INPUT_TEST)

test-primary: build-primary
	$(CONTENT_INPUT_TEST) config/encoder.conf
	$(PRIMARY_INPUT_TEST) config/encoder.conf

$(BUILD_DIR)/test_rpb_v22_content_input.o: code/encoders/raw_patch_bottleneck_mae/versions/v22/tests/content_encoder_input_test.cpp $(CONTENT_HEADERS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTENT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CONTENT_INPUT_TEST): $(BUILD_DIR)/test_rpb_v22_content_input.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/test_primary_encoder_input.o: code/encoders/raw_patch_bottleneck_mae/versions/v22/tests/primary_encoder_input_test.cpp $(CONTENT_HEADERS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTENT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(PRIMARY_INPUT_TEST): $(BUILD_DIR)/test_primary_encoder_input.o $(BUILD_DIR)/record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/test_rpb_v22_content_input.d $(BUILD_DIR)/test_primary_encoder_input.d
