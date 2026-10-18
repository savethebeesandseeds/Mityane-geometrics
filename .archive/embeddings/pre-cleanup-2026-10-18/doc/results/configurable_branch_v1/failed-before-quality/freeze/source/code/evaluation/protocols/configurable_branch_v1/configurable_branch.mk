# Saved-feature branch attribution; this binary never links an encoder.
CB_ROOT := $(CODE_ROOT)/evaluation/protocols/configurable_branch_v1
CB_DIR := $(OBJECT_DIR)/protocols/configurable_branch_v1
CB_READER := $(BUILD_DIR)/embedding_configurable_branch_readout
CB_CARD := $(CODE_ROOT)/evaluation/cards/configurable_branch_v1.md
CB_INPUTS := $(sort Makefile $(CB_CARD) $(wildcard $(CB_ROOT)/*.cpp $(CB_ROOT)/*.h $(CB_ROOT)/*.py $(CB_ROOT)/*.sh $(CB_ROOT)/*.mk $(CB_ROOT)/README.md) $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp) $(CODE_ROOT)/evaluation/include/frozen_role_guard.h $(CODE_ROOT)/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py $(CODE_ROOT)/scripts/task.sh $(SDK_PROVENANCE_INPUTS))
CB_SOURCE_ID := $(shell sha256sum $(CB_INPUTS) | sha256sum | cut -d ' ' -f1)
CB_CARD_SHA := $(shell sha256sum $(CB_CARD) | cut -d ' ' -f1)
CB_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(CODE_ROOT)/evaluation/include -DCONFIGURABLE_BRANCH_SOURCE_ID=\"$(CB_SOURCE_ID)\" -DCONFIGURABLE_BRANCH_CARD_SHA256=\"$(CB_CARD_SHA)\"
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
