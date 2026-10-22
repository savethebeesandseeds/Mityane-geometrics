# SPDX-License-Identifier: MIT
# Target integration only; root build owns the compiler, SDK and environment.
RPB_FAMILY_SOURCE_ROOT ?= $(CURDIR)
RPB_FAMILY_ROOT := $(RPB_FAMILY_SOURCE_ROOT)/code/embeddings/encoders/raw_patch_bottleneck_mae/family
RPB_FAMILY_CORE := $(RPB_FAMILY_SOURCE_ROOT)/code/embeddings/encoders/raw_patch_bottleneck_mae
RPB_FAMILY_CPPFLAGS := -I$(RPB_FAMILY_ROOT)/include -I$(RPB_FAMILY_CORE)/include \
  -I$(RPB_FAMILY_SOURCE_ROOT)/code/shared/settings/include \
  -I$(RPB_FAMILY_SOURCE_ROOT)/code/data/observations/include \
  -I$(RPB_FAMILY_SOURCE_ROOT)/code/shared/artifacts/versions/v1/include \
  -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
RPB_FAMILY_OBJECT_DIR := $(BUILD_DIR)/rpb-family
RPB_FAMILY_TEST := $(BUILD_DIR)/test_rpb_family
RPB_FAMILY_TRAINER := $(BUILD_DIR)/rpb_family_train
RPB_FAMILY_FIXTURE := $(BUILD_DIR)/rpb_observation_windows_fixture
RPB_FAMILY_VARIANT := $(BUILD_DIR)/rpb_dataset_variant
RPB_FAMILY_OUTPUT ?= $(RUN_ROOT)/portability-001
RPB_FAMILY_EXAMPLE_OUTPUT ?= $(RUN_ROOT)/training-example-001

.PHONY: build-rpb-family build-rpb-family-trainer build-rpb-family-fixtures test-rpb-family test-rpb-family-training-example test-rpb-family-exporter
build-rpb-family: $(RPB_FAMILY_TEST)
build-rpb-family-trainer: $(RPB_FAMILY_TRAINER)
build-rpb-family-fixtures: $(RPB_FAMILY_FIXTURE) $(RPB_FAMILY_VARIANT)
test-rpb-family: build-rpb-family
	$(RPB_FAMILY_TEST) "$(RPB_FAMILY_ROOT)/config/single.conf" "$(RPB_FAMILY_ROOT)/config/multi.conf" "$(RPB_FAMILY_OUTPUT)"
test-rpb-family-training-example: build-rpb-family-trainer build-rpb-family-fixtures
	bash code/scripts/node.sh "$(RPB_FAMILY_ROOT)/tests/training_example.mjs" --trainer "$(RPB_FAMILY_TRAINER)" --fixture "$(RPB_FAMILY_FIXTURE)" --variant "$(RPB_FAMILY_VARIANT)" --config-directory "$(RPB_FAMILY_ROOT)/config" --output "$(RPB_FAMILY_EXAMPLE_OUTPUT)"
test-rpb-family-exporter:
	bash code/scripts/node.sh --test "$(RPB_FAMILY_SOURCE_ROOT)/code/embeddings/scripts/tests/export-rpb-family.test.mjs"

$(RPB_FAMILY_OBJECT_DIR)/portability_test.o: $(RPB_FAMILY_ROOT)/tests/portability_test.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(RPB_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(RPB_FAMILY_OBJECT_DIR)/trainer.o: $(RPB_FAMILY_SOURCE_ROOT)/code/embeddings/scripts/rpb_family_train.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(RPB_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(RPB_FAMILY_OBJECT_DIR)/observation_fixture.o: $(RPB_FAMILY_SOURCE_ROOT)/code/data/observations/tests/observation_windows_fixture.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(RPB_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(RPB_FAMILY_OBJECT_DIR)/dataset_variant.o: $(RPB_FAMILY_ROOT)/tests/dataset_variant.cpp
	@mkdir -p "$(@D)"
	$(CXX) $(RPB_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(RPB_FAMILY_TEST): $(RPB_FAMILY_OBJECT_DIR)/portability_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(RPB_FAMILY_TRAINER): $(RPB_FAMILY_OBJECT_DIR)/trainer.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(RPB_FAMILY_FIXTURE): $(RPB_FAMILY_OBJECT_DIR)/observation_fixture.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(RPB_FAMILY_VARIANT): $(RPB_FAMILY_OBJECT_DIR)/dataset_variant.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(RPB_FAMILY_OBJECT_DIR)/*.d)
