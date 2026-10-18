# Separate configurable-v20 quality experiment; historical protocols stay frozen.
CC_ROOT := $(CODE_ROOT)/evaluation/protocols/configurable_capacity_v1
CC_DATA := $(CODE_ROOT)/evaluation/benchmarks/configurable_capacity
CC_DIR := $(OBJECT_DIR)/protocols/configurable_capacity_v1
CC_BIN := $(BUILD_DIR)/embedding_configurable_capacity
CC_WRITER := $(BUILD_DIR)/embedding_configurable_capacity_data
CC_READER := $(BUILD_DIR)/embedding_configurable_capacity_readout
CC_TEST := $(CC_DIR)/generator_test
CC_CARD := $(CODE_ROOT)/evaluation/cards/configurable_capacity_v1.md
CC_CONFIG := $(RPB_ROOT)/config/configurable_spectral.conf
CC_INPUTS := $(sort Makefile $(CC_CARD) $(CC_CONFIG) $(wildcard $(CC_ROOT)/*.cpp $(CC_ROOT)/*.h $(CC_ROOT)/*.py $(CC_ROOT)/*.sh $(CC_ROOT)/*.mk $(CC_ROOT)/README.md $(CC_DATA)/*.cpp $(CC_DATA)/*.h) $(RPB_CORE_HEADERS) $(RPB_ROOT)/configurable_spectral.mk $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp) $(CODE_ROOT)/shared/include/cuwacunu/config/document.h $(CODE_ROOT)/shared/include/piaabo/parse/simple_kv_block.h $(CODE_ROOT)/evaluation/include/frozen_role_guard.h $(CODE_ROOT)/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py $(CODE_ROOT)/scripts/task.sh $(SDK_PROVENANCE_INPUTS))
CC_SOURCE_ID := $(shell sha256sum $(CC_INPUTS) | sha256sum | cut -d ' ' -f1)
CC_CARD_SHA := $(shell sha256sum $(CC_CARD) | cut -d ' ' -f1)
CC_CPPFLAGS := $(COMMON_CPPFLAGS) -isystem /usr/local/cuda/include -I$(RPB_ROOT)/include -I$(CC_DATA) -I$(CODE_ROOT)/evaluation/include -DCONFIGURABLE_CAPACITY_SOURCE_ID=\"$(CC_SOURCE_ID)\" -DCONFIGURABLE_CAPACITY_CARD_SHA256=\"$(CC_CARD_SHA)\"
CC_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
CC_SHARED_OBJECTS := $(addprefix $(OBJECT_DIR)/shared/,$(addsuffix .o,$(CC_SHARED_NAMES)))
.PHONY: configurable-capacity configurable-capacity-data configurable-capacity-readout test-configurable-capacity-data test-configurable-capacity-runtime test-configurable-capacity-audit print-configurable-capacity-sources
configurable-capacity: $(CC_BIN) $(CC_WRITER) $(CC_READER) $(CC_TEST)
configurable-capacity-data: $(CC_WRITER)
configurable-capacity-readout: $(CC_READER)
print-configurable-capacity-sources:
	@printf '%s\n' $(CC_INPUTS)
$(CC_DIR)/%.o: $(CC_ROOT)/%.cpp $(CC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CC_DIR)/configurable_capacity.o: $(CC_DATA)/configurable_capacity.cpp $(CC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CC_DIR)/observed_information.o: $(CC_DATA)/observed_information.cpp $(CC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CC_DIR)/write_quality_data.o: $(CC_DATA)/main.cpp $(CC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CC_DIR)/generator_test.o: $(CC_DATA)/generator_test.cpp $(CC_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CC_BIN): $(CC_DIR)/producer_main.o $(CC_DIR)/configurable_capacity.o $(OBJECT_DIR)/shared/data.o $(OBJECT_DIR)/shared/feature_harness.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CC_WRITER): $(CC_DIR)/write_quality_data.o $(CC_DIR)/configurable_capacity.o $(CC_DIR)/observed_information.o $(OBJECT_DIR)/shared/data.o $(OBJECT_DIR)/shared/feature_harness.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CC_READER): $(CC_DIR)/reader_main.o $(CC_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CC_TEST): $(CC_DIR)/generator_test.o $(CC_DIR)/configurable_capacity.o $(CC_DIR)/observed_information.o $(OBJECT_DIR)/shared/data.o $(OBJECT_DIR)/shared/feature_harness.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-configurable-capacity-data: $(CC_TEST)
	$(CC_TEST)
test-configurable-capacity-runtime:
	python3 -B $(CC_ROOT)/capsule_runtime_test.py
	python3 -B -O $(CC_ROOT)/capsule_runtime_test.py
test-configurable-capacity-audit:
	python3 -B $(CC_ROOT)/evaluation_audit_test.py
	python3 -B -O $(CC_ROOT)/evaluation_audit_test.py
-include $(wildcard $(CC_DIR)/*.d)
