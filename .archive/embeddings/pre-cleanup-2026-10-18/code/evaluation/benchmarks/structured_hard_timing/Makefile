# Independent data-only fixtures; unchanged root encoder definitions.
include Makefile
STRUCTURED_TIMING_DATA_ROOT := code/evaluation/benchmarks/structured_hard_timing
STRUCTURED_TIMING_DATA_OBJECT_DIR := $(OBJECT_DIR)/benchmarks/structured_hard_timing
STRUCTURED_TIMING_DATA_TEST := $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/generator_test
STRUCTURED_TIMING_CROSS_TEST := $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/cross_feature_test
.PHONY: test-structured-hard-timing-data test-cross-feature-timing-data
$(STRUCTURED_TIMING_DATA_OBJECT_DIR)/%.o: $(STRUCTURED_TIMING_DATA_ROOT)/%.cpp $(STRUCTURED_TIMING_DATA_ROOT)/structured_hard_timing.h
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) -I$(STRUCTURED_TIMING_DATA_ROOT) $(CXXFLAGS) -c $< -o $@
$(STRUCTURED_TIMING_DATA_TEST): $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/generator_test.o $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/structured_hard_timing.o $(HARNESS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-structured-hard-timing-data: $(STRUCTURED_TIMING_DATA_TEST)
	$(STRUCTURED_TIMING_DATA_TEST)
$(STRUCTURED_TIMING_DATA_OBJECT_DIR)/cross_feature_test.o $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/cross_feature_solvability.o: $(STRUCTURED_TIMING_DATA_ROOT)/cross_feature_solvability.h
$(STRUCTURED_TIMING_CROSS_TEST): $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/cross_feature_test.o $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/cross_feature_solvability.o $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/structured_hard_timing.o $(HARNESS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-cross-feature-timing-data: $(STRUCTURED_TIMING_CROSS_TEST)
	$(STRUCTURED_TIMING_CROSS_TEST)
-include $(wildcard $(STRUCTURED_TIMING_DATA_OBJECT_DIR)/*.d)
