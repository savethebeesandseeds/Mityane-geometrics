# Additive content-only relation experiment. Measured protocols stay intact.
CCR_ROOT := $(CODE_ROOT)/evaluation/protocols/configurable_content_relation_v1
CCR_DATA := $(CODE_ROOT)/evaluation/benchmarks/configurable_capacity
CCR_DIR := $(OBJECT_DIR)/protocols/configurable_content_relation_v1
CCR_PRODUCER := $(BUILD_DIR)/embedding_configurable_content_relation
CCR_READER := $(BUILD_DIR)/embedding_configurable_content_relation_readout
CCR_CARD := $(CODE_ROOT)/evaluation/cards/configurable_content_relation_v1.md
CCR_CONFIG := code/encoders/raw_patch_bottleneck_mae/versions/v22/config/content_ordered_relation.conf
CCR_INPUTS := $(sort Makefile $(CCR_CARD) $(CCR_CONFIG) $(wildcard $(CCR_ROOT)/*.cpp $(CCR_ROOT)/*.h $(CCR_ROOT)/*.py $(CCR_ROOT)/*.sh $(CCR_ROOT)/*.mk $(CCR_ROOT)/README.md $(CCR_DATA)/*.cpp $(CCR_DATA)/*.h) $(RPB_CORE_HEADERS) code/encoders/raw_patch_bottleneck_mae/versions/v20/configurable_spectral.mk code/encoders/raw_patch_bottleneck_mae/versions/v22/content_ordered_relation.mk code/encoders/raw_patch_bottleneck_mae/versions/v22/tests/content_ordered_relation_test.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp) $(CODE_ROOT)/shared/include/cuwacunu/config/document.h $(CODE_ROOT)/shared/include/piaabo/parse/simple_kv_block.h $(CODE_ROOT)/evaluation/include/frozen_role_guard.h $(CODE_ROOT)/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py $(CODE_ROOT)/evaluation/protocols/configurable_capacity_v1/publish.py $(CODE_ROOT)/evaluation/protocols/configurable_capacity_v1/saved_audit.py $(CODE_ROOT)/evaluation/protocols/configurable_branch_v1/publish.py $(CODE_ROOT)/evaluation/protocols/configurable_learned_relation_v1/gate.py $(CODE_ROOT)/scripts/task.sh $(SDK_PROVENANCE_INPUTS))
CCR_SOURCE_ID := $(shell sha256sum $(CCR_INPUTS) | sha256sum | cut -d ' ' -f1)
CCR_CARD_SHA := $(shell sha256sum $(CCR_CARD) | cut -d ' ' -f1)
CCR_CPPFLAGS := $(COMMON_CPPFLAGS) -isystem /usr/local/cuda/include -I$(RPB_ROOT)/include -I$(CCR_DATA) -I$(CODE_ROOT)/evaluation/include -DCONFIGURABLE_CONTENT_RELATION_SOURCE_ID=\"$(CCR_SOURCE_ID)\" -DCONFIGURABLE_CONTENT_RELATION_CARD_SHA256=\"$(CCR_CARD_SHA)\"
CCR_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
CCR_SHARED_OBJECTS := $(addprefix $(OBJECT_DIR)/shared/,$(addsuffix .o,$(CCR_SHARED_NAMES)))
.PHONY: configurable-content-relation configurable-content-relation-readout print-configurable-content-relation-sources
configurable-content-relation: $(CCR_PRODUCER) $(CCR_READER)
configurable-content-relation-readout: $(CCR_READER)
print-configurable-content-relation-sources:
	@printf '%s\n' $(CCR_INPUTS)
$(CCR_DIR)/%.o: $(CCR_ROOT)/%.cpp $(CCR_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CCR_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CCR_DIR)/configurable_capacity.o: $(CCR_DATA)/configurable_capacity.cpp $(CCR_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CCR_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CCR_PRODUCER): $(CCR_DIR)/producer_main.o $(CCR_DIR)/configurable_capacity.o $(OBJECT_DIR)/shared/data.o $(OBJECT_DIR)/shared/feature_harness.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CCR_READER): $(CCR_DIR)/reader_main.o $(CCR_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(CCR_DIR)/*.d)
