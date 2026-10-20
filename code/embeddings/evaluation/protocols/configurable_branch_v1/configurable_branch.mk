# Saved-feature branch attribution; this binary never links an encoder.
CB_ROOT := code/embeddings/evaluation/protocols/configurable_branch_v1
CB_DIR := $(OBJECT_DIR)/protocols/configurable_branch_v1
CB_READER := $(BUILD_DIR)/embedding_configurable_branch_readout
CB_CARD := code/embeddings/evaluation/cards/configurable_branch_v1.md
CB_INPUTS := $(sort Makefile $(CB_CARD) $(wildcard $(CB_ROOT)/*.cpp $(CB_ROOT)/*.h $(CB_ROOT)/*.py $(CB_ROOT)/*.mjs $(CB_ROOT)/*.sh $(CB_ROOT)/*.mk $(CB_ROOT)/README.md) $(OBSERVATION_PROVENANCE_INPUTS) $(wildcard code/shared/representation/include/embedding/shared/*.h code/shared/representation/src/*.cpp) code/embeddings/evaluation/include/frozen_role_guard.h code/embeddings/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py code/scripts/task.sh $(SDK_PROVENANCE_INPUTS) $(PROTOCOL_NODE_INPUTS) $(SCIENTIFIC_REFERENCE_INPUTS))
CB_SOURCE_ID := $(shell sha256sum $(CB_INPUTS) | sha256sum | cut -d ' ' -f1)
CB_CARD_SHA := $(shell sha256sum $(CB_CARD) | cut -d ' ' -f1)
CB_CPPFLAGS := $(COMMON_CPPFLAGS) -Icode/embeddings/evaluation/include -DCONFIGURABLE_BRANCH_SOURCE_ID=\"$(CB_SOURCE_ID)\" -DCONFIGURABLE_BRANCH_CARD_SHA256=\"$(CB_CARD_SHA)\"
CB_SHARED_NAMES := data evaluation feature_harness feature_evaluation feature_stress archive_readout native_curve paired_pooling fixed_feature_readouts
CB_SHARED_OBJECTS := $(addprefix $(OBJECT_DIR)/shared/,$(addsuffix .o,$(CB_SHARED_NAMES)))
.PHONY: configurable-branch configurable-branch-readout print-configurable-branch-sources
configurable-branch configurable-branch-readout: $(CB_READER)
print-configurable-branch-sources:
	@printf '%s\n' $(CB_INPUTS)
$(CB_DIR)/reader_main.o: $(CB_ROOT)/reader_main.cpp $(CB_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CB_READER): $(CB_DIR)/reader_main.o $(CB_SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(CB_DIR)/*.d)
