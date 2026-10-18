# Separate data-only fixture target; does not change any frozen encoder closure.
include Makefile

VARIABLE_DELAY_TIMING_ROOT := code/evaluation/benchmarks/variable_delay_timing
VARIABLE_DELAY_TIMING_OBJECT_DIR := $(OBJECT_DIR)/benchmarks/variable_delay_timing
VARIABLE_DELAY_TIMING_TEST := $(VARIABLE_DELAY_TIMING_OBJECT_DIR)/generator_test

.PHONY: test-variable-delay-timing
$(VARIABLE_DELAY_TIMING_OBJECT_DIR)/%.o: $(VARIABLE_DELAY_TIMING_ROOT)/%.cpp $(VARIABLE_DELAY_TIMING_ROOT)/variable_delay_timing.h
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) -I$(VARIABLE_DELAY_TIMING_ROOT) $(CXXFLAGS) -c $< -o $@

$(VARIABLE_DELAY_TIMING_TEST): $(VARIABLE_DELAY_TIMING_OBJECT_DIR)/generator_test.o $(VARIABLE_DELAY_TIMING_OBJECT_DIR)/variable_delay_timing.o $(HARNESS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-variable-delay-timing: $(VARIABLE_DELAY_TIMING_TEST)
	$(VARIABLE_DELAY_TIMING_TEST)

-include $(wildcard $(VARIABLE_DELAY_TIMING_OBJECT_DIR)/*.d)
