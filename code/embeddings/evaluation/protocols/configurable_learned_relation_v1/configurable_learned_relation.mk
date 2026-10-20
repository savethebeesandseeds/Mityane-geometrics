# Additive learned-relation experiment. Existing measured protocols stay intact.
CLR_ROOT := code/embeddings/evaluation/protocols/configurable_learned_relation_v1
CLR_DATA := code/embeddings/evaluation/benchmarks/configurable_capacity
CLR_DIR := $(OBJECT_DIR)/protocols/configurable_learned_relation_v1
CLR_PRODUCER := $(BUILD_DIR)/embedding_configurable_learned_relation
CLR_READER := $(BUILD_DIR)/embedding_configurable_learned_relation_readout
CLR_CARD := code/embeddings/evaluation/cards/configurable_learned_relation_v1.md
CLR_CONFIG := code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v21/config/learned_ordered_relation.conf
CLR_INPUTS := $(sort Makefile $(CLR_CARD) $(CLR_CONFIG) $(wildcard $(CLR_ROOT)/*.cpp $(CLR_ROOT)/*.h $(CLR_ROOT)/*.py $(CLR_ROOT)/*.mjs $(CLR_ROOT)/*.sh $(CLR_ROOT)/*.mk $(CLR_ROOT)/README.md $(CLR_DATA)/*.cpp $(CLR_DATA)/*.h) $(RPB_CORE_HEADERS) code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v20/configurable_spectral.mk code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v21/learned_ordered_relation.mk code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v21/tests/learned_ordered_relation_test.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h $(OBSERVATION_PROVENANCE_INPUTS) $(wildcard code/shared/representation/include/embedding/shared/*.h code/shared/representation/src/*.cpp) code/shared/settings/include/cuwacunu/config/document.h code/shared/settings/include/piaabo/parse/simple_kv_block.h code/embeddings/evaluation/include/frozen_role_guard.h code/embeddings/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/embeddings/evaluation/protocols/configurable_capacity_v1/publish.py code/embeddings/evaluation/protocols/configurable_capacity_v1/saved_audit.py code/embeddings/evaluation/protocols/configurable_branch_v1/publish.py code/scripts/task.sh $(SDK_PROVENANCE_INPUTS) $(PROTOCOL_NODE_INPUTS) $(SCIENTIFIC_REFERENCE_INPUTS))
CLR_SOURCE_ID := $(shell sha256sum $(CLR_INPUTS) | sha256sum | cut -d ' ' -f1)
CLR_CARD_SHA := $(shell sha256sum $(CLR_CARD) | cut -d ' ' -f1)
CLR_CPPFLAGS := $(COMMON_CPPFLAGS) -isystem /usr/local/cuda/include -I$(RPB_ROOT)/include -I$(CLR_DATA) -Icode/embeddings/evaluation/include -DCONFIGURABLE_LEARNED_RELATION_SOURCE_ID=\"$(CLR_SOURCE_ID)\" -DCONFIGURABLE_LEARNED_RELATION_CARD_SHA256=\"$(CLR_CARD_SHA)\"
CLR_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
CLR_SHARED_OBJECTS := $(addprefix $(OBJECT_DIR)/shared/,$(addsuffix .o,$(CLR_SHARED_NAMES)))
.PHONY: configurable-learned-relation configurable-learned-relation-readout print-configurable-learned-relation-sources
configurable-learned-relation: $(CLR_PRODUCER) $(CLR_READER)
configurable-learned-relation-readout: $(CLR_READER)
print-configurable-learned-relation-sources:
	@printf '%s\n' $(CLR_INPUTS)
$(CLR_DIR)/%.o: $(CLR_ROOT)/%.cpp $(CLR_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CLR_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CLR_DIR)/configurable_capacity.o: $(CLR_DATA)/configurable_capacity.cpp $(CLR_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CLR_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CLR_PRODUCER): $(CLR_DIR)/producer_main.o $(CLR_DIR)/configurable_capacity.o $(OBJECT_DIR)/shared/data.o $(OBJECT_DIR)/shared/feature_harness.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(CLR_READER): $(CLR_DIR)/reader_main.o $(CLR_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(CLR_DIR)/*.d)
