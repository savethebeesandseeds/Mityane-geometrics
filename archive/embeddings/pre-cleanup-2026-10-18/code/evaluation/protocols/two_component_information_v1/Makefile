# Data-only two-component engineering; no model or historical object rebuilds.
BUILD_DIR ?= /opt/cuwacunu_embedding/build/two-component-information-v1
include Makefile
.DEFAULT_GOAL := two-component-information
TC_ROOT := code/evaluation/benchmarks/two_component
TC_PROTOCOL := code/evaluation/protocols/two_component_information_v1
TC_OBJECT_DIR := $(OBJECT_DIR)/protocols/two_component_information_v1
TC_BIN := $(BUILD_DIR)/embedding_two_component_information
TC_GENERATOR_TEST := $(TC_OBJECT_DIR)/generator_test
TC_INFORMATION_TEST := $(TC_OBJECT_DIR)/information_test
TC_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
TC_SHARED_DIR := /opt/cuwacunu_embedding/build/rpb-architecture-screen/code/protocols/architecture_screen_v1/shared
TC_SHARED_OBJECTS := $(addprefix $(TC_SHARED_DIR)/,$(addsuffix .o,$(TC_SHARED_NAMES)))
TC_SHARED_SOURCES := $(addprefix code/shared/src/,$(addsuffix .cpp,$(TC_SHARED_NAMES)))
TC_INPUTS := $(sort Makefile $(SDK_PROVENANCE_INPUTS) code/evaluation/include/frozen_role_guard.h $(wildcard code/shared/include/embedding/shared/*.h) $(TC_SHARED_SOURCES) $(TC_ROOT)/two_component.h $(TC_ROOT)/two_component.cpp $(TC_ROOT)/generator_test.cpp $(TC_ROOT)/observed_information.h $(TC_ROOT)/observed_information.cpp $(TC_ROOT)/information_test.cpp $(TC_PROTOCOL)/Makefile $(TC_PROTOCOL)/information_main.cpp $(TC_PROTOCOL)/prepare.py $(TC_PROTOCOL)/check_saved.py code/evaluation/cards/two_component_information_v1.md code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/evaluation/protocols/visible_difference_v1/reader-v2/saved_cpu_math.py)
TC_SOURCE_ID := $(shell sha256sum $(TC_INPUTS) | sha256sum | cut -d ' ' -f1)
TC_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(TC_ROOT) -Icode/evaluation/include -DTWO_COMPONENT_SOURCE_ID=\"$(TC_SOURCE_ID)\"
.PHONY: two-component-information two-component-information-tests print-two-component-information-sources
two-component-information: $(TC_BIN)
two-component-information-tests: $(TC_GENERATOR_TEST) $(TC_INFORMATION_TEST)
print-two-component-information-sources:
	@printf '%s\n' $(TC_INPUTS)
$(TC_OBJECT_DIR)/two_component.o: $(TC_ROOT)/two_component.cpp $(TC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(TC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(TC_OBJECT_DIR)/observed_information.o: $(TC_ROOT)/observed_information.cpp $(TC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(TC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(TC_OBJECT_DIR)/generator_test.o: $(TC_ROOT)/generator_test.cpp $(TC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(TC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(TC_OBJECT_DIR)/information_test.o: $(TC_ROOT)/information_test.cpp $(TC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(TC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(TC_OBJECT_DIR)/information_main.o: $(TC_PROTOCOL)/information_main.cpp $(TC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(TC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(TC_SHARED_OBJECTS):
	@printf '%s\n' 'Required unchanged generic shared object is absent: $@' >&2; exit 1
$(TC_GENERATOR_TEST): $(TC_OBJECT_DIR)/generator_test.o $(TC_OBJECT_DIR)/two_component.o $(TC_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(TC_INFORMATION_TEST): $(TC_OBJECT_DIR)/information_test.o $(TC_OBJECT_DIR)/observed_information.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(TC_BIN): $(TC_OBJECT_DIR)/information_main.o $(TC_OBJECT_DIR)/two_component.o $(TC_OBJECT_DIR)/observed_information.o $(TC_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(TC_OBJECT_DIR)/*.d)
