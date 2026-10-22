# SPDX-License-Identifier: MIT
# Root-owned compiler/SDK settings; no component environment or installation.
MTF_FAMILY_SOURCE_ROOT ?= $(CURDIR)
MTF_FAMILY_ROOT := $(MTF_FAMILY_SOURCE_ROOT)/code/embeddings/encoders/mtf_jepa_mae_vicreg/family
MTF_CORE_ROOT := $(MTF_FAMILY_SOURCE_ROOT)/code/embeddings/encoders/mtf_jepa_mae_vicreg
MTF_SHARED_ROOT := $(MTF_FAMILY_SOURCE_ROOT)/code/shared/representation
MTF_DATA_ROOT := $(MTF_FAMILY_SOURCE_ROOT)/code/data/observations
MTF_ARTIFACT_ROOT := $(MTF_FAMILY_SOURCE_ROOT)/code/shared/artifacts/versions/v1
MTF_FAMILY_CPPFLAGS := -I$(MTF_FAMILY_ROOT)/include -I$(MTF_CORE_ROOT)/include \
  -I$(MTF_SHARED_ROOT)/include -I$(MTF_DATA_ROOT)/include -I$(MTF_ARTIFACT_ROOT)/include \
  -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
MTF_FAMILY_OBJECT_DIR := $(BUILD_DIR)/mtf-family
MTF_FAMILY_OBJECTS := $(addprefix $(MTF_FAMILY_OBJECT_DIR)/,family.o workflow.o evaluation.o shared_evaluation.o data.o)
MTF_FAMILY_TEST := $(BUILD_DIR)/test_mtf_family
MTF_FAMILY_TRAINER := $(BUILD_DIR)/mtf_family_train
MTF_FAMILY_OUTPUT ?= $(RUN_ROOT)/attempt-001
MTF_FAMILY_CLI_OUTPUT ?= $(RUN_ROOT)/cli-001
MTF_FAMILY_DELIVERY_OUTPUT ?= $(RUN_ROOT)/delivery-001

.PHONY: build-mtf-family build-mtf-family-trainer test-mtf-family test-mtf-family-cli test-mtf-family-exporter
.PHONY: test-mtf-family-delivery
build-mtf-family: $(MTF_FAMILY_TEST)
build-mtf-family-trainer: $(MTF_FAMILY_TRAINER)
test-mtf-family: build-mtf-family
	$(MTF_FAMILY_TEST) --config-directory "$(MTF_FAMILY_ROOT)/config" --output "$(MTF_FAMILY_OUTPUT)"
test-mtf-family-cli: build-mtf-family-trainer
	bash code/scripts/node.sh "$(MTF_FAMILY_ROOT)/tests/cli_example.mjs" --trainer "$(MTF_FAMILY_TRAINER)" --config "$(MTF_FAMILY_ROOT)/config/alternative.conf" --output "$(MTF_FAMILY_CLI_OUTPUT)"
test-mtf-family-exporter:
	bash code/scripts/node.sh --test "$(MTF_FAMILY_SOURCE_ROOT)/code/embeddings/scripts/tests/export-mtf-family.test.mjs"
test-mtf-family-delivery: build-mtf-family build-mtf-family-trainer
	bash code/scripts/node.sh "$(MTF_FAMILY_SOURCE_ROOT)/code/embeddings/scripts/verify-mtf-family-delivery.mjs" --test "$(MTF_FAMILY_TEST)" --trainer "$(MTF_FAMILY_TRAINER)" --source-root "$(MTF_FAMILY_SOURCE_ROOT)" --output "$(MTF_FAMILY_DELIVERY_OUTPUT)"

$(MTF_FAMILY_OBJECT_DIR)/family.o: $(MTF_FAMILY_ROOT)/src/family.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(MTF_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MTF_FAMILY_OBJECT_DIR)/workflow.o: $(MTF_CORE_ROOT)/src/workflow.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(MTF_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MTF_FAMILY_OBJECT_DIR)/evaluation.o: $(MTF_CORE_ROOT)/src/evaluation.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(MTF_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MTF_FAMILY_OBJECT_DIR)/shared_evaluation.o: $(MTF_SHARED_ROOT)/src/evaluation.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(MTF_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MTF_FAMILY_OBJECT_DIR)/data.o: $(MTF_DATA_ROOT)/src/data.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(MTF_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MTF_FAMILY_OBJECT_DIR)/family_test.o: $(MTF_FAMILY_ROOT)/tests/family_test.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(MTF_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MTF_FAMILY_OBJECT_DIR)/trainer.o: $(MTF_FAMILY_SOURCE_ROOT)/code/embeddings/scripts/mtf_family_train.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(MTF_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(MTF_FAMILY_TEST): $(MTF_FAMILY_OBJECT_DIR)/family_test.o $(MTF_FAMILY_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(MTF_FAMILY_TRAINER): $(MTF_FAMILY_OBJECT_DIR)/trainer.o $(MTF_FAMILY_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(MTF_FAMILY_OBJECT_DIR)/*.d)
