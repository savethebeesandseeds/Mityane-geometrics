# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Configurable fresh-family tooling has no research checkpoint dependency.
# The opt-in reference equivalence check requires an explicit artifact/hash.
CPB_FAMILY_ROOT := code/embeddings/encoders/causal_patch_bottleneck_mae/family
CPB_FAMILY_CORE := code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v16
CPB_FAMILY_CPPFLAGS := -I$(CPB_FAMILY_ROOT)/include -I$(CPB_FAMILY_CORE)/include -Icode/embeddings/encoders/include -Icode/data/observations/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPB_FAMILY_TEST := $(BUILD_DIR)/test_cpb_family
CPB_FAMILY_OUTPUT ?= $(RUN_ROOT)/attempt-001
.PHONY: build-cpb-family test-cpb-family
build-cpb-family: $(CPB_FAMILY_TEST)
test-cpb-family: build-cpb-family
	$(CPB_FAMILY_TEST) --config-directory $(CPB_FAMILY_ROOT)/config --output "$(CPB_FAMILY_OUTPUT)"
$(BUILD_DIR)/cpb_family_test.o: $(CPB_FAMILY_ROOT)/tests/family_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_FAMILY_TEST): $(BUILD_DIR)/cpb_family_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_family_test.d

CPB_FAMILY_EQUIVALENCE := $(BUILD_DIR)/test_cpb_family_reference_equivalence
CPB_FAMILY_TRAINER := $(BUILD_DIR)/cpb_family_train
OBSERVATION_WINDOW_FIXTURE := $(BUILD_DIR)/observation_windows_fixture
CPB_FAMILY_REFERENCE_CHECKPOINT ?=
CPB_FAMILY_REFERENCE_SHA256 ?=
CPB_FAMILY_EQUIVALENCE_OUTPUT ?= $(RUN_ROOT)/equivalence-001
CPB_FAMILY_TRAINING_EXAMPLE_OUTPUT ?= $(RUN_ROOT)/training-example-001
.PHONY: build-cpb-family-equivalence test-cpb-family-equivalence build-cpb-family-trainer build-observation-window-fixture test-cpb-family-training-example
build-cpb-family-equivalence: $(CPB_FAMILY_EQUIVALENCE)
test-cpb-family-equivalence: build-cpb-family-equivalence
	@test -n "$(CPB_FAMILY_REFERENCE_CHECKPOINT)" -a -n "$(CPB_FAMILY_REFERENCE_SHA256)"
	$(CPB_FAMILY_EQUIVALENCE) --checkpoint "$(CPB_FAMILY_REFERENCE_CHECKPOINT)" --expected-sha256 "$(CPB_FAMILY_REFERENCE_SHA256)" --output "$(CPB_FAMILY_EQUIVALENCE_OUTPUT)"
build-cpb-family-trainer: $(CPB_FAMILY_TRAINER)
build-observation-window-fixture: $(OBSERVATION_WINDOW_FIXTURE)
test-cpb-family-training-example: build-cpb-family-trainer build-observation-window-fixture
	bash code/scripts/node.sh $(CPB_FAMILY_ROOT)/tests/training_example.mjs --trainer "$(CPB_FAMILY_TRAINER)" --fixture "$(OBSERVATION_WINDOW_FIXTURE)" --settings "$(CPB_FAMILY_ROOT)/config/paired.conf" --output "$(CPB_FAMILY_TRAINING_EXAMPLE_OUTPUT)"
$(BUILD_DIR)/cpb_family_reference_equivalence.o: $(CPB_FAMILY_ROOT)/tests/reference_equivalence_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_FAMILY_EQUIVALENCE): $(BUILD_DIR)/cpb_family_reference_equivalence.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/cpb_family_train.o: code/embeddings/scripts/cpb_family_train.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(CPB_FAMILY_TRAINER): $(BUILD_DIR)/cpb_family_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/observation_windows_fixture.o: code/data/observations/tests/observation_windows_fixture.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_FAMILY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(OBSERVATION_WINDOW_FIXTURE): $(BUILD_DIR)/observation_windows_fixture.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_family_reference_equivalence.d $(BUILD_DIR)/cpb_family_train.d $(BUILD_DIR)/observation_windows_fixture.d

CPB_FAMILY_NUMERIC_SERIES := $(BUILD_DIR)/test_cpb_family_numeric_series
.PHONY: test-cpb-family-numeric-series
test-cpb-family-numeric-series: $(CPB_FAMILY_NUMERIC_SERIES)
	$(CPB_FAMILY_NUMERIC_SERIES)
$(BUILD_DIR)/cpb_family_numeric_series_history.o: $(CPB_FAMILY_ROOT)/tests/numeric_series_history_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_FAMILY_CPPFLAGS) -Icode/dataloaders/numeric_series/include $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/numeric_series_record_types.o: code/dataloaders/numeric_series/src/record_types.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPB_FAMILY_CPPFLAGS) -Icode/dataloaders/numeric_series/include $(CXXFLAGS) -c $< -o $@
$(CPB_FAMILY_NUMERIC_SERIES): $(BUILD_DIR)/cpb_family_numeric_series_history.o $(BUILD_DIR)/numeric_series_record_types.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_family_numeric_series_history.d $(BUILD_DIR)/numeric_series_record_types.d
