CXX := g++
CODE_ROOT := code
ENCODER_ROOT := $(CODE_ROOT)/encoders/mtf_jepa_mae_vicreg
BUILD_DIR ?= /opt/cuwacunu_embedding/build/baseline
OBJECT_DIR := $(BUILD_DIR)/code
LIBTORCH ?= /opt/cuwacunu_embedding/libtorch
SDK_PROVENANCE_INPUTS := setup.sh $(CODE_ROOT)/scripts/install-libtorch.py dependencies.lock
REFERENCE_DIR ?= $(CURDIR)/.build/reference
RUN_ROOT ?= $(CURDIR)/output/runs/baseline
BIN := $(BUILD_DIR)/embedding
COMMON_CPPFLAGS := -I$(CODE_ROOT)/shared/include -I$(CODE_ROOT)/shared/tests -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
CPPFLAGS := -I$(ENCODER_ROOT)/include -I$(ENCODER_ROOT)/tests $(COMMON_CPPFLAGS)
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,'$(abspath $(LIBTORCH))/lib' -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread

SHARED_OBJECTS := $(OBJECT_DIR)/shared/data.o $(OBJECT_DIR)/shared/evaluation.o
ENCODER_OBJECTS := $(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/workflow.o $(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/evaluation.o
ENCODER_TEST_DIR := $(OBJECT_DIR)/tests/encoders/mtf_jepa_mae_vicreg
SHARED_TEST_DIR := $(OBJECT_DIR)/tests/shared
ENCODER_TESTS := $(addprefix $(ENCODER_TEST_DIR)/,model_test workflow_test numerics_test masking_test evaluation_test)
SHARED_TESTS := $(addprefix $(SHARED_TEST_DIR)/,data_test objectives_test evaluation_test)

.PHONY: all test baseline smoke smoke-cuda evaluate periodic-check
all: $(BIN)

$(BIN): $(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/main.o $(SHARED_OBJECTS) $(ENCODER_OBJECTS)
	mkdir -p "$(@D)"
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

# Shared code compiles without encoder include directories.
$(OBJECT_DIR)/shared/%.o: $(CODE_ROOT)/shared/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/%.o: $(ENCODER_ROOT)/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(ENCODER_TEST_DIR)/%_test: $(ENCODER_ROOT)/tests/%_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(ENCODER_TEST_DIR)/%_test.o: $(ENCODER_ROOT)/tests/%_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(ENCODER_TEST_DIR)/workflow_test: $(ENCODER_TEST_DIR)/workflow_test.o $(SHARED_OBJECTS) $(ENCODER_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(ENCODER_TEST_DIR)/evaluation_test: $(ENCODER_TEST_DIR)/evaluation_test.o $(SHARED_OBJECTS) $(ENCODER_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(ENCODER_TEST_DIR)/baseline_test: $(ENCODER_ROOT)/tests/baseline_test.cpp $(REFERENCE_DIR)/reference_model.h
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) -I$(REFERENCE_DIR) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/objectives_test: $(CODE_ROOT)/shared/tests/objectives_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/evaluation_test.o: $(CODE_ROOT)/shared/tests/evaluation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/evaluation_test: $(SHARED_TEST_DIR)/evaluation_test.o $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/data_test.o: $(CODE_ROOT)/shared/tests/data_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/data_test: $(SHARED_TEST_DIR)/data_test.o $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test: $(ENCODER_TESTS) $(SHARED_TESTS)
	$(ENCODER_TEST_DIR)/model_test
	$(ENCODER_TEST_DIR)/workflow_test
	$(ENCODER_TEST_DIR)/numerics_test
	$(ENCODER_TEST_DIR)/masking_test
	$(ENCODER_TEST_DIR)/evaluation_test
	$(SHARED_TEST_DIR)/data_test
	$(SHARED_TEST_DIR)/objectives_test
	$(SHARED_TEST_DIR)/evaluation_test

baseline: $(ENCODER_TEST_DIR)/baseline_test
	$(ENCODER_TEST_DIR)/baseline_test

smoke: $(BIN)
	EMBEDDING_BIN="$(abspath $(BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(ENCODER_ROOT)/tests/cli_smoke.sh cpu

smoke-cuda: $(BIN)
	EMBEDDING_BIN="$(abspath $(BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(ENCODER_ROOT)/tests/cli_smoke.sh cuda

evaluate: $(BIN)
	EMBEDDING_BIN="$(abspath $(BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(ENCODER_ROOT)/scripts/evaluate.sh

periodic-check: $(BIN)
	EMBEDDING_BIN="$(abspath $(BIN))" bash $(ENCODER_ROOT)/tests/periodic_checkpoint.sh

# Build products use their own code/ namespace so retained dependencies from
# earlier layouts cannot refer to source paths that have moved.
-include $(wildcard $(OBJECT_DIR)/shared/*.d $(OBJECT_DIR)/encoders/mtf_jepa_mae_vicreg/*.d $(ENCODER_TEST_DIR)/*.d $(SHARED_TEST_DIR)/*.d)

# Independent model and evaluation boundaries. Baseline targets above retain
# their historical evaluator; ordinary RPB train/embed links neither evaluator.
RPB_ROOT := $(CODE_ROOT)/encoders/raw_patch_bottleneck_mae
RPB_OBJECT_DIR := $(OBJECT_DIR)/encoders/raw_patch_bottleneck_mae
RPB_TEST_DIR := $(OBJECT_DIR)/tests/encoders/raw_patch_bottleneck_mae
RPB_BIN ?= $(BUILD_DIR)/embedding_raw_patch_bottleneck_mae
EVALUATION_ROOT := $(CODE_ROOT)/evaluation
EVALUATION_OBJECT_DIR := $(OBJECT_DIR)/evaluation
EVALUATION_BIN ?= $(BUILD_DIR)/embedding_evaluate
HARNESS_BIN ?= $(BUILD_DIR)/feature_harness

RPB_CORE_HEADERS := $(filter-out $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/reconstruction_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/paired_pooling_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/optimization_diagnostic_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_replay_adapter.h $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_replication_adapter.h,$(wildcard $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/*.h))
RPB_PROVENANCE_INPUTS := $(sort $(RPB_CORE_HEADERS) $(RPB_ROOT)/src/workflow.cpp $(RPB_ROOT)/src/main.cpp $(wildcard $(RPB_ROOT)/config/*.conf) $(CODE_ROOT)/shared/include/embedding/shared/data.h $(CODE_ROOT)/shared/include/embedding/shared/types.h $(CODE_ROOT)/shared/include/embedding/shared/tensor_ops.h $(CODE_ROOT)/shared/src/data.cpp Makefile $(SDK_PROVENANCE_INPUTS))
RPB_SOURCE_ID := $(shell sha256sum $(RPB_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
SOURCE_GIT_HEAD := $(shell git -c safe.directory=$(CURDIR) rev-parse HEAD 2>/dev/null || printf unrecorded)
SOURCE_GIT_DIRTY := $(shell if source_status=$$(git -c safe.directory=$(CURDIR) status --porcelain 2>/dev/null); then if test -n "$$source_status"; then printf dirty; else printf clean; fi; else printf unrecorded; fi)
RPB_CPPFLAGS := -I$(RPB_ROOT)/include -I$(RPB_ROOT)/tests $(COMMON_CPPFLAGS) -DRPB_SOURCE_ID=\"$(RPB_SOURCE_ID)\" -DRPB_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DRPB_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"

MINIMUM_PROVENANCE_INPUTS := $(sort $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp $(ENCODER_ROOT)/include/embedding/encoders/mtf_jepa_mae_vicreg/*.h $(ENCODER_ROOT)/src/*.cpp $(ENCODER_ROOT)/config/*.conf $(EVALUATION_ROOT)/src/*.cpp $(EVALUATION_ROOT)/include/*.h) Makefile $(SDK_PROVENANCE_INPUTS))
EVALUATION_PROVENANCE_INPUTS := $(sort $(MINIMUM_PROVENANCE_INPUTS) $(RPB_PROVENANCE_INPUTS) $(RPB_ROOT)/src/evaluation_adapter.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h $(RPB_ROOT)/src/reconstruction_adapter.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/reconstruction_adapter.h $(RPB_ROOT)/src/learning_curve_adapter.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h $(wildcard $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h))
EVALUATION_PROVENANCE_INPUTS := $(sort $(EVALUATION_PROVENANCE_INPUTS) $(RPB_ROOT)/src/training_source_gain.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/training_source_gain.h)
MINIMUM_SOURCE_ID := $(shell sha256sum $(MINIMUM_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
EVALUATION_SOURCE_ID := $(shell sha256sum $(EVALUATION_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
EVALUATION_PROVENANCE_CPPFLAGS := -DEVALUATION_SOURCE_ID=\"$(EVALUATION_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
EVALUATION_CPPFLAGS := -I$(ENCODER_ROOT)/include $(COMMON_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS)
MINIMUM_CPPFLAGS := -I$(ENCODER_ROOT)/include $(COMMON_CPPFLAGS) -DEVALUATION_SOURCE_ID=\"$(MINIMUM_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"

HARNESS_OBJECT := $(OBJECT_DIR)/shared/feature_harness.o
FEATURE_EVALUATION_OBJECT := $(OBJECT_DIR)/shared/feature_evaluation.o
FEATURE_STRESS_OBJECT := $(OBJECT_DIR)/shared/feature_stress.o
EVALUATION_COMMON_OBJECTS := $(SHARED_OBJECTS) $(HARNESS_OBJECT) $(FEATURE_EVALUATION_OBJECT) $(FEATURE_STRESS_OBJECT)
RECONSTRUCTION_OBJECT := $(OBJECT_DIR)/shared/reconstruction_evaluation.o
RPB_WORKFLOW_OBJECT := $(RPB_OBJECT_DIR)/workflow.o
RPB_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/evaluation_adapter.o
RPB_RECONSTRUCTION_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/reconstruction_adapter.o
BASELINE_ADAPTER_OBJECT := $(EVALUATION_OBJECT_DIR)/baseline_adapter.o
MINIMUM_ADAPTER_OBJECT := $(EVALUATION_OBJECT_DIR)/minimum_baseline_adapter.o
RPB_CORE_TESTS := $(addprefix $(RPB_TEST_DIR)/,preprocessing_test numerics_test masking_test model_test)

.PHONY: rpb-mae evaluation feature-harness test-rpb-core test-rpb-mae test-rpb-reconstruction test-rpb-feature-adapter test-feature-harness test-feature-evaluation test-feature-stress test-reconstruction-evaluation smoke-rpb-mae smoke-rpb-mae-cuda periodic-rpb-mae evaluate-minimum evaluate-rpb-mae reconstruct-rpb-mae
rpb-mae: $(RPB_BIN)
evaluation: $(EVALUATION_BIN)
feature-harness: $(HARNESS_BIN)

$(RPB_OBJECT_DIR)/%.o: $(RPB_ROOT)/src/%.cpp
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_WORKFLOW_OBJECT) $(RPB_OBJECT_DIR)/main.o: $(RPB_PROVENANCE_INPUTS)

$(RPB_BIN): $(RPB_OBJECT_DIR)/main.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	mkdir -p "$(@D)"
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(EVALUATION_OBJECT_DIR)/main.o: $(EVALUATION_ROOT)/src/main.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -I$(EVALUATION_ROOT)/include -I$(RPB_ROOT)/include -DEMBEDDING_WITH_RPB=1 $(CXXFLAGS) -c $< -o $@

$(EVALUATION_OBJECT_DIR)/minimum_main.o: $(EVALUATION_ROOT)/src/main.cpp $(MINIMUM_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MINIMUM_CPPFLAGS) -I$(EVALUATION_ROOT)/include -DEMBEDDING_WITH_RPB=0 $(CXXFLAGS) -c $< -o $@

$(EVALUATION_OBJECT_DIR)/reconstruction_main.o: $(EVALUATION_ROOT)/src/reconstruction_main.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -I$(EVALUATION_ROOT)/include -I$(RPB_ROOT)/include -DEMBEDDING_WITH_RPB=1 $(CXXFLAGS) -c $< -o $@

$(EVALUATION_OBJECT_DIR)/minimum_reconstruction_main.o: $(EVALUATION_ROOT)/src/reconstruction_main.cpp $(MINIMUM_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MINIMUM_CPPFLAGS) -I$(EVALUATION_ROOT)/include -DEMBEDDING_WITH_RPB=0 $(CXXFLAGS) -c $< -o $@

$(BASELINE_ADAPTER_OBJECT): $(ENCODER_ROOT)/src/evaluation_adapter.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(MINIMUM_ADAPTER_OBJECT): $(ENCODER_ROOT)/src/evaluation_adapter.cpp $(MINIMUM_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MINIMUM_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_ADAPTER_OBJECT): $(RPB_ROOT)/src/evaluation_adapter.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_RECONSTRUCTION_ADAPTER_OBJECT): $(RPB_ROOT)/src/reconstruction_adapter.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EVALUATION_BIN): $(EVALUATION_OBJECT_DIR)/main.o $(EVALUATION_OBJECT_DIR)/reconstruction_main.o $(EVALUATION_COMMON_OBJECTS) $(RECONSTRUCTION_OBJECT) $(ENCODER_OBJECTS) $(BASELINE_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_RECONSTRUCTION_ADAPTER_OBJECT)
	mkdir -p "$(@D)"
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(HARNESS_BIN): $(EVALUATION_OBJECT_DIR)/minimum_main.o $(EVALUATION_OBJECT_DIR)/minimum_reconstruction_main.o $(EVALUATION_COMMON_OBJECTS) $(ENCODER_OBJECTS) $(MINIMUM_ADAPTER_OBJECT)
	mkdir -p "$(@D)"
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/%_test.o: $(RPB_ROOT)/tests/%_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/%_test: $(RPB_TEST_DIR)/%_test.o $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/workflow_test: $(RPB_TEST_DIR)/workflow_test.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/reconstruction_adapter_test: $(RPB_TEST_DIR)/reconstruction_adapter_test.o $(RPB_RECONSTRUCTION_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/evaluation_adapter_test: $(RPB_TEST_DIR)/evaluation_adapter_test.o $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/feature_harness_test.o: $(CODE_ROOT)/shared/tests/feature_harness_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/feature_harness_test: $(SHARED_TEST_DIR)/feature_harness_test.o $(SHARED_OBJECTS) $(HARNESS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/feature_evaluation_test.o: $(CODE_ROOT)/shared/tests/feature_evaluation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/feature_evaluation_test: $(SHARED_TEST_DIR)/feature_evaluation_test.o $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/feature_stress_test.o: $(CODE_ROOT)/shared/tests/feature_stress_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/feature_stress_test: $(SHARED_TEST_DIR)/feature_stress_test.o $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/reconstruction_evaluation_test.o: $(CODE_ROOT)/shared/tests/reconstruction_evaluation_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/reconstruction_evaluation_test: $(SHARED_TEST_DIR)/reconstruction_evaluation_test.o $(RECONSTRUCTION_OBJECT) $(SHARED_OBJECTS) $(HARNESS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-core: $(RPB_CORE_TESTS)
	$(RPB_TEST_DIR)/preprocessing_test
	$(RPB_TEST_DIR)/numerics_test
	$(RPB_TEST_DIR)/masking_test
	$(RPB_TEST_DIR)/model_test

test-rpb-mae: test-rpb-core $(RPB_TEST_DIR)/workflow_test
	$(RPB_TEST_DIR)/workflow_test

test-rpb-reconstruction: $(RPB_TEST_DIR)/reconstruction_adapter_test
	$(RPB_TEST_DIR)/reconstruction_adapter_test

test-rpb-feature-adapter: $(RPB_TEST_DIR)/evaluation_adapter_test
	$(RPB_TEST_DIR)/evaluation_adapter_test

test-feature-evaluation: $(SHARED_TEST_DIR)/feature_evaluation_test
	$(SHARED_TEST_DIR)/feature_evaluation_test

test-feature-stress: $(SHARED_TEST_DIR)/feature_stress_test
	$(SHARED_TEST_DIR)/feature_stress_test

test-feature-harness: $(SHARED_TEST_DIR)/feature_harness_test test-feature-evaluation test-feature-stress
	$(SHARED_TEST_DIR)/feature_harness_test

test-reconstruction-evaluation: $(SHARED_TEST_DIR)/reconstruction_evaluation_test
	$(SHARED_TEST_DIR)/reconstruction_evaluation_test

smoke-rpb-mae: $(RPB_BIN)
	EMBEDDING_BIN="$(abspath $(RPB_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(RPB_ROOT)/tests/cli_smoke.sh cpu

smoke-rpb-mae-cuda: $(RPB_BIN)
	EMBEDDING_BIN="$(abspath $(RPB_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(RPB_ROOT)/tests/cli_smoke.sh cuda

periodic-rpb-mae: $(RPB_BIN)
	EMBEDDING_BIN="$(abspath $(RPB_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(RPB_ROOT)/tests/periodic_checkpoint.sh

evaluate-minimum: $(HARNESS_BIN)
	EMBEDDING_BIN="$(abspath $(HARNESS_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-minimum.sh

evaluate-rpb-mae: $(EVALUATION_BIN)
	EVALUATION_BIN="$(abspath $(EVALUATION_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-rpb-mae.sh

reconstruct-rpb-mae: $(EVALUATION_BIN)
	EVALUATION_BIN="$(abspath $(EVALUATION_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/reconstruct-rpb-mae.sh

-include $(wildcard $(RPB_OBJECT_DIR)/*.d $(RPB_TEST_DIR)/*.d $(EVALUATION_OBJECT_DIR)/*.d)

# Learning-curve selection/scoring stays shared; only training and compact
# reconstruction live in the RPB adapter. Existing executables retain their CLI.
LEARNING_CURVE_BIN := $(BUILD_DIR)/embedding_learning_curve
LEARNING_CURVE_OBJECT := $(OBJECT_DIR)/shared/learning_curve.o
RPB_CURVE_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/learning_curve_adapter.o
.PHONY: learning-curve test-learning-curve test-rpb-learning-curve
learning-curve: $(LEARNING_CURVE_BIN)

$(EVALUATION_OBJECT_DIR)/learning_curve_main.o: $(EVALUATION_ROOT)/src/learning_curve_main.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -I$(RPB_ROOT)/include $(CXXFLAGS) -c $< -o $@

$(RPB_CURVE_ADAPTER_OBJECT): $(RPB_ROOT)/src/learning_curve_adapter.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(LEARNING_CURVE_BIN): $(EVALUATION_OBJECT_DIR)/learning_curve_main.o $(LEARNING_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/learning_curve_test.o: $(CODE_ROOT)/shared/tests/learning_curve_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/learning_curve_test: $(SHARED_TEST_DIR)/learning_curve_test.o $(LEARNING_CURVE_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/learning_curve_adapter_test: $(RPB_TEST_DIR)/learning_curve_adapter_test.o $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-learning-curve: $(SHARED_TEST_DIR)/learning_curve_test
	$(SHARED_TEST_DIR)/learning_curve_test

test-rpb-learning-curve: $(RPB_TEST_DIR)/learning_curve_adapter_test
	$(RPB_TEST_DIR)/learning_curve_adapter_test

# Frozen projection diagnostics link only shared math/archive code.
PROJECTION_DIAGNOSTIC_BIN := $(BUILD_DIR)/embedding_projection_diagnostic
PROJECTION_DIAGNOSTIC_OBJECT := $(OBJECT_DIR)/shared/projection_diagnostic.o
GLOBAL_BOTTLENECK_BIN := $(BUILD_DIR)/embedding_global_bottleneck
GLOBAL_BOTTLENECK_OBJECT := $(OBJECT_DIR)/shared/global_bottleneck_experiment.o
.PHONY: projection-diagnostic test-projection-diagnostic global-bottleneck test-global-bottleneck
projection-diagnostic: $(PROJECTION_DIAGNOSTIC_BIN)
global-bottleneck: $(GLOBAL_BOTTLENECK_BIN)

$(EVALUATION_OBJECT_DIR)/projection_diagnostic_main.o: $(EVALUATION_ROOT)/src/projection_diagnostic_main.cpp $(MINIMUM_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(EVALUATION_PROVENANCE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EVALUATION_OBJECT_DIR)/global_bottleneck_main.o: $(EVALUATION_ROOT)/src/global_bottleneck_main.cpp $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EVALUATION_CPPFLAGS) -I$(RPB_ROOT)/include $(CXXFLAGS) -c $< -o $@

$(PROJECTION_DIAGNOSTIC_BIN): $(EVALUATION_OBJECT_DIR)/projection_diagnostic_main.o $(PROJECTION_DIAGNOSTIC_OBJECT) $(OBJECT_DIR)/shared/feature_harness.o $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(GLOBAL_BOTTLENECK_BIN): $(EVALUATION_OBJECT_DIR)/global_bottleneck_main.o $(GLOBAL_BOTTLENECK_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/projection_diagnostic_test.o: $(CODE_ROOT)/shared/tests/projection_diagnostic_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/projection_diagnostic_test: $(SHARED_TEST_DIR)/projection_diagnostic_test.o $(PROJECTION_DIAGNOSTIC_OBJECT) $(OBJECT_DIR)/shared/feature_harness.o $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/global_bottleneck_experiment_test.o: $(CODE_ROOT)/shared/tests/global_bottleneck_experiment_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/global_bottleneck_experiment_test: $(SHARED_TEST_DIR)/global_bottleneck_experiment_test.o $(GLOBAL_BOTTLENECK_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-projection-diagnostic: $(SHARED_TEST_DIR)/projection_diagnostic_test
	$(SHARED_TEST_DIR)/projection_diagnostic_test

test-global-bottleneck: $(SHARED_TEST_DIR)/global_bottleneck_experiment_test
	$(SHARED_TEST_DIR)/global_bottleneck_experiment_test

.PHONY: test-rpb-legacy-checkpoint
$(RPB_TEST_DIR)/legacy_checkpoint_parity_test: $(RPB_TEST_DIR)/legacy_checkpoint_parity_test.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-legacy-checkpoint: $(RPB_TEST_DIR)/legacy_checkpoint_parity_test
	$(RPB_TEST_DIR)/legacy_checkpoint_parity_test

# Native archive controls use shared fitting only; no encoder is linked.
ARCHIVE_READOUT_BIN := $(BUILD_DIR)/embedding_archive_readout
ARCHIVE_READOUT_OBJECT := $(OBJECT_DIR)/shared/archive_readout.o
ARCHIVE_READOUT_PROVENANCE_INPUTS := $(sort $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp) $(EVALUATION_ROOT)/src/archive_readout_main.cpp $(EVALUATION_ROOT)/cards/archive_readout_v1.md Makefile dependencies.lock)
ARCHIVE_READOUT_SOURCE_ID := $(shell sha256sum $(ARCHIVE_READOUT_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
ARCHIVE_READOUT_CPPFLAGS := $(COMMON_CPPFLAGS) -DEVALUATION_SOURCE_ID=\"$(ARCHIVE_READOUT_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: archive-readout test-archive-readout evaluate-archive
archive-readout: $(ARCHIVE_READOUT_BIN)

$(EVALUATION_OBJECT_DIR)/archive_readout_main.o: $(EVALUATION_ROOT)/src/archive_readout_main.cpp $(ARCHIVE_READOUT_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(ARCHIVE_READOUT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(ARCHIVE_READOUT_BIN): $(EVALUATION_OBJECT_DIR)/archive_readout_main.o $(ARCHIVE_READOUT_OBJECT) $(HARNESS_OBJECT) $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/archive_readout_test.o: $(CODE_ROOT)/shared/tests/archive_readout_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/archive_readout_test: $(SHARED_TEST_DIR)/archive_readout_test.o $(ARCHIVE_READOUT_OBJECT) $(HARNESS_OBJECT) $(SHARED_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-archive-readout: $(SHARED_TEST_DIR)/archive_readout_test
	$(SHARED_TEST_DIR)/archive_readout_test

evaluate-archive: $(ARCHIVE_READOUT_BIN)
	ARCHIVE_READOUT_BIN="$(abspath $(ARCHIVE_READOUT_BIN))" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-archive.sh $(ARCHIVE_ARGS)

# Native-only curves select and score the served export. Historical curve
# executables/cards above retain their projection-based protocol semantics.
NATIVE_CURVE_BIN := $(BUILD_DIR)/embedding_native_curve
NATIVE_CURVE_OBJECT := $(OBJECT_DIR)/shared/native_curve.o
RPB_NATIVE_GATE_OBJECT := $(RPB_OBJECT_DIR)/native_curve_gate.o
NATIVE_CURVE_PROVENANCE_INPUTS := $(sort $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h $(CODE_ROOT)/shared/src/*.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/*.h) $(RPB_ROOT)/src/workflow.cpp $(RPB_ROOT)/src/evaluation_adapter.cpp $(RPB_ROOT)/src/learning_curve_adapter.cpp $(RPB_ROOT)/src/native_curve_gate.cpp $(RPB_ROOT)/config/learned_global.conf $(EVALUATION_ROOT)/src/native_curve_main.cpp $(EVALUATION_ROOT)/cards/native_curve_v1.md $(CODE_ROOT)/scripts/evaluate-native-curve.sh Makefile dependencies.lock)
NATIVE_CURVE_SOURCE_ID := $(shell sha256sum $(NATIVE_CURVE_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
NATIVE_CURVE_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(NATIVE_CURVE_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: native-curve test-native-curve test-rpb-native-gate evaluate-native-curve
native-curve: $(NATIVE_CURVE_BIN)

$(EVALUATION_OBJECT_DIR)/native_curve_main.o: $(EVALUATION_ROOT)/src/native_curve_main.cpp $(NATIVE_CURVE_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_CURVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_NATIVE_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(NATIVE_CURVE_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_CURVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(NATIVE_CURVE_BIN): $(EVALUATION_OBJECT_DIR)/native_curve_main.o $(NATIVE_CURVE_OBJECT) $(RPB_NATIVE_GATE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/native_curve_test.o: $(CODE_ROOT)/shared/tests/native_curve_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/native_curve_test: $(SHARED_TEST_DIR)/native_curve_test.o $(NATIVE_CURVE_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/native_curve_gate_test: $(RPB_TEST_DIR)/native_curve_gate_test.o $(RPB_NATIVE_GATE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-native-curve: $(SHARED_TEST_DIR)/native_curve_test
	$(SHARED_TEST_DIR)/native_curve_test

test-rpb-native-gate: $(RPB_TEST_DIR)/native_curve_gate_test
	$(RPB_TEST_DIR)/native_curve_gate_test

evaluate-native-curve: $(NATIVE_CURVE_BIN)
	NATIVE_CURVE_BIN="$(abspath $(NATIVE_CURVE_BIN))" NATIVE_CURVE_SOURCE_INPUTS="$(NATIVE_CURVE_PROVENANCE_INPUTS)" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-native-curve.sh

# One pooling candidate versus immutable retained native-only reference assets.
PAIRED_POOLING_BIN := $(BUILD_DIR)/embedding_paired_pooling
PAIRED_POOLING_OBJECT := $(OBJECT_DIR)/shared/paired_pooling.o
RPB_PAIRED_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/paired_pooling_adapter.o
RPB_PAIRED_GATE_OBJECT := $(RPB_OBJECT_DIR)/paired_pooling_gate.o
PAIRED_POOLING_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(RPB_ROOT)/config/learned_patch_global.conf $(EVALUATION_ROOT)/src/paired_pooling_main.cpp $(EVALUATION_ROOT)/cards/paired_pooling_v1.md $(CODE_ROOT)/scripts/evaluate-paired-pooling.sh)
PAIRED_POOLING_SOURCE_ID := $(shell sha256sum $(PAIRED_POOLING_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
PAIRED_POOLING_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(PAIRED_POOLING_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: paired-pooling test-paired-pooling evaluate-paired-pooling
paired-pooling: $(PAIRED_POOLING_BIN)

$(EVALUATION_OBJECT_DIR)/paired_pooling_main.o: $(EVALUATION_ROOT)/src/paired_pooling_main.cpp $(PAIRED_POOLING_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(PAIRED_POOLING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_PAIRED_ADAPTER_OBJECT): $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(PAIRED_POOLING_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(PAIRED_POOLING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_PAIRED_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(PAIRED_POOLING_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(PAIRED_POOLING_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(PAIRED_POOLING_BIN): $(EVALUATION_OBJECT_DIR)/paired_pooling_main.o $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_PAIRED_ADAPTER_OBJECT) $(RPB_PAIRED_GATE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/paired_pooling_test.o: $(CODE_ROOT)/shared/tests/paired_pooling_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/paired_pooling_test: $(SHARED_TEST_DIR)/paired_pooling_test.o $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-paired-pooling: $(SHARED_TEST_DIR)/paired_pooling_test
	$(SHARED_TEST_DIR)/paired_pooling_test

evaluate-paired-pooling: $(PAIRED_POOLING_BIN)
	PAIRED_POOLING_BIN="$(abspath $(PAIRED_POOLING_BIN))" PAIRED_POOLING_SOURCE_INPUTS="$(PAIRED_POOLING_PROVENANCE_INPUTS)" EMBEDDING_RUN_ROOT="$(RUN_ROOT)" bash $(CODE_ROOT)/scripts/evaluate-paired-pooling.sh

# Exact checkpoint continuation is an encoder adapter; all head fitting remains
# in the reusable archive reader, under a separate TRAIN/VALIDATION-only card.
OPTIMIZATION_DIAGNOSTIC_BIN := $(BUILD_DIR)/embedding_optimization_diagnostic
RPB_OPTIMIZATION_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/optimization_diagnostic_adapter.o
RPB_OPTIMIZATION_GATE_OBJECT := $(RPB_OBJECT_DIR)/optimization_diagnostic_gate.o
OPTIMIZATION_DIAGNOSTIC_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/optimization_diagnostic_adapter.cpp $(RPB_ROOT)/config/learned_patch_global.conf $(EVALUATION_ROOT)/src/optimization_diagnostic_main.cpp $(EVALUATION_ROOT)/cards/optimization_validation_v1.md $(CODE_ROOT)/scripts/evaluate-optimization-diagnostic.sh)
OPTIMIZATION_DIAGNOSTIC_SOURCE_ID := $(shell sha256sum $(OPTIMIZATION_DIAGNOSTIC_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
OPTIMIZATION_DIAGNOSTIC_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(OPTIMIZATION_DIAGNOSTIC_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: optimization-diagnostic test-rpb-optimization-diagnostic evaluate-optimization-diagnostic
optimization-diagnostic: $(OPTIMIZATION_DIAGNOSTIC_BIN)

$(EVALUATION_OBJECT_DIR)/optimization_diagnostic_main.o: $(EVALUATION_ROOT)/src/optimization_diagnostic_main.cpp $(OPTIMIZATION_DIAGNOSTIC_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(OPTIMIZATION_DIAGNOSTIC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_OPTIMIZATION_ADAPTER_OBJECT): $(RPB_ROOT)/src/optimization_diagnostic_adapter.cpp $(OPTIMIZATION_DIAGNOSTIC_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(OPTIMIZATION_DIAGNOSTIC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_OPTIMIZATION_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(OPTIMIZATION_DIAGNOSTIC_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(OPTIMIZATION_DIAGNOSTIC_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OPTIMIZATION_DIAGNOSTIC_BIN): $(EVALUATION_OBJECT_DIR)/optimization_diagnostic_main.o $(RPB_OPTIMIZATION_ADAPTER_OBJECT) $(RPB_OPTIMIZATION_GATE_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/optimization_diagnostic_adapter_test: $(RPB_TEST_DIR)/optimization_diagnostic_adapter_test.o $(RPB_OPTIMIZATION_ADAPTER_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-optimization-diagnostic: $(RPB_TEST_DIR)/optimization_diagnostic_adapter_test
	$(RPB_TEST_DIR)/optimization_diagnostic_adapter_test

evaluate-optimization-diagnostic: $(OPTIMIZATION_DIAGNOSTIC_BIN)
	OPTIMIZATION_DIAGNOSTIC_BIN="$(abspath $(OPTIMIZATION_DIAGNOSTIC_BIN))" OPTIMIZATION_DIAGNOSTIC_SOURCE_INPUTS="$(OPTIMIZATION_DIAGNOSTIC_PROVENANCE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-optimization-diagnostic.sh

# Same v4 inference tensors, one independently recorded TRAIN context policy.
CONTEXT_DELETION_BIN := $(BUILD_DIR)/embedding_context_deletion
RPB_CONTEXT_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/context_deletion_adapter.o
RPB_CONTEXT_GATE_OBJECT := $(RPB_OBJECT_DIR)/context_deletion_gate.o
CONTEXT_DELETION_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(RPB_ROOT)/src/context_deletion_adapter.cpp $(EVALUATION_ROOT)/src/context_deletion_main.cpp $(EVALUATION_ROOT)/cards/context_deletion_v1.md $(CODE_ROOT)/scripts/evaluate-context-deletion.sh $(CODE_ROOT)/scripts/check-context-deletion.sh)
CONTEXT_DELETION_SOURCE_ID := $(shell sha256sum $(CONTEXT_DELETION_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
CONTEXT_DELETION_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(CONTEXT_DELETION_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: context-deletion test-rpb-context-deletion test-rpb-training-policy evaluate-context-deletion print-context-deletion-sources
context-deletion: $(CONTEXT_DELETION_BIN)

print-context-deletion-sources:
	@printf '%s\n' $(CONTEXT_DELETION_PROVENANCE_INPUTS)

$(EVALUATION_OBJECT_DIR)/context_deletion_main.o: $(EVALUATION_ROOT)/src/context_deletion_main.cpp $(CONTEXT_DELETION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_DELETION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_ADAPTER_OBJECT): $(RPB_ROOT)/src/context_deletion_adapter.cpp $(CONTEXT_DELETION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_DELETION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(CONTEXT_DELETION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_DELETION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(CONTEXT_DELETION_BIN): $(EVALUATION_OBJECT_DIR)/context_deletion_main.o $(RPB_CONTEXT_ADAPTER_OBJECT) $(RPB_CONTEXT_GATE_OBJECT) $(RPB_PAIRED_ADAPTER_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/training_policy_test: $(RPB_TEST_DIR)/training_policy_test.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-training-policy: $(RPB_TEST_DIR)/training_policy_test
	$(RPB_TEST_DIR)/training_policy_test

$(RPB_TEST_DIR)/context_deletion_test: $(RPB_TEST_DIR)/context_deletion_test.o $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/context_deletion_adapter_test: $(RPB_TEST_DIR)/context_deletion_adapter_test.o $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-context-deletion: $(RPB_TEST_DIR)/context_deletion_test $(RPB_TEST_DIR)/context_deletion_adapter_test
	$(RPB_TEST_DIR)/context_deletion_test
	$(RPB_TEST_DIR)/context_deletion_adapter_test

evaluate-context-deletion: $(CONTEXT_DELETION_BIN)
	CONTEXT_DELETION_BIN="$(abspath $(CONTEXT_DELETION_BIN))" CONTEXT_DELETION_SOURCE_INPUTS="$(CONTEXT_DELETION_PROVENANCE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-context-deletion.sh

# Policy-preserving fresh replay is an encoder concern. Optional validation
# deletion views and their unchanged fitted heads belong to shared evaluation.
CONTEXT_OPTIMIZATION_BIN := $(BUILD_DIR)/embedding_context_optimization_validation
RPB_CONTEXT_REPLAY_OBJECT := $(RPB_OBJECT_DIR)/context_replay_adapter.o
RPB_CONTEXT_OPTIMIZATION_GATE_OBJECT := $(RPB_OBJECT_DIR)/context_optimization_gate.o
CONTEXT_OPTIMIZATION_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/context_replay_adapter.cpp $(EVALUATION_ROOT)/src/context_optimization_validation_main.cpp $(EVALUATION_ROOT)/cards/context_optimization_validation_v1.md $(CODE_ROOT)/scripts/evaluate-context-optimization-validation.sh $(CODE_ROOT)/scripts/check-context-optimization-validation.sh)
CONTEXT_OPTIMIZATION_SOURCE_ID := $(shell sha256sum $(CONTEXT_OPTIMIZATION_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
CONTEXT_OPTIMIZATION_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(CONTEXT_OPTIMIZATION_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: context-optimization-validation print-context-optimization-sources test-rpb-context-replay evaluate-context-optimization-validation
context-optimization-validation: $(CONTEXT_OPTIMIZATION_BIN)

print-context-optimization-sources:
	@printf '%s\n' $(CONTEXT_OPTIMIZATION_PROVENANCE_INPUTS)

$(EVALUATION_OBJECT_DIR)/context_optimization_validation_main.o: $(EVALUATION_ROOT)/src/context_optimization_validation_main.cpp $(CONTEXT_OPTIMIZATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_OPTIMIZATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_REPLAY_OBJECT): $(RPB_ROOT)/src/context_replay_adapter.cpp $(CONTEXT_OPTIMIZATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_OPTIMIZATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_OPTIMIZATION_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(CONTEXT_OPTIMIZATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_OPTIMIZATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(CONTEXT_OPTIMIZATION_BIN): $(EVALUATION_OBJECT_DIR)/context_optimization_validation_main.o $(RPB_CONTEXT_REPLAY_OBJECT) $(RPB_CONTEXT_OPTIMIZATION_GATE_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/context_replay_adapter_test: $(RPB_TEST_DIR)/context_replay_adapter_test.o $(RPB_CONTEXT_REPLAY_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-context-replay: $(RPB_TEST_DIR)/context_replay_adapter_test
	$(RPB_TEST_DIR)/context_replay_adapter_test

evaluate-context-optimization-validation: $(CONTEXT_OPTIMIZATION_BIN)
	CONTEXT_OPTIMIZATION_BIN="$(abspath $(CONTEXT_OPTIMIZATION_BIN))" CONTEXT_OPTIMIZATION_SOURCE_INPUTS="$(CONTEXT_OPTIMIZATION_PROVENANCE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-context-optimization-validation.sh

# Five fresh equal-budget context-policy pairs; no checkpoint selection.
CONTEXT_REPLICATION_BIN := $(BUILD_DIR)/embedding_context_replication
RPB_CONTEXT_REPLICATION_OBJECT := $(RPB_OBJECT_DIR)/context_replication_adapter.o
RPB_CONTEXT_REPLICATION_GATE_OBJECT := $(RPB_OBJECT_DIR)/context_replication_gate.o
RPB_CONTEXT_REPLICATION_RETAINED_OBJECT := $(RPB_OBJECT_DIR)/context_replication_retained.o
CONTEXT_REPLICATION_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/context_replication_adapter.cpp $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(EVALUATION_ROOT)/src/context_replication_main.cpp $(EVALUATION_ROOT)/cards/context_replication_v1.md $(CODE_ROOT)/scripts/evaluate-context-replication.sh $(CODE_ROOT)/scripts/check-context-replication.sh)
CONTEXT_REPLICATION_SOURCE_ID := $(shell sha256sum $(CONTEXT_REPLICATION_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
CONTEXT_REPLICATION_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(CONTEXT_REPLICATION_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: context-replication print-context-replication-sources test-rpb-context-replication evaluate-context-replication
context-replication: $(CONTEXT_REPLICATION_BIN)

print-context-replication-sources:
	@printf '%s\n' $(CONTEXT_REPLICATION_PROVENANCE_INPUTS)

$(EVALUATION_OBJECT_DIR)/context_replication_main.o: $(EVALUATION_ROOT)/src/context_replication_main.cpp $(CONTEXT_REPLICATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_REPLICATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_REPLICATION_OBJECT): $(RPB_ROOT)/src/context_replication_adapter.cpp $(CONTEXT_REPLICATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_REPLICATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_REPLICATION_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(CONTEXT_REPLICATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_REPLICATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_REPLICATION_RETAINED_OBJECT): $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(CONTEXT_REPLICATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_REPLICATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(CONTEXT_REPLICATION_BIN): $(EVALUATION_OBJECT_DIR)/context_replication_main.o $(RPB_CONTEXT_REPLICATION_OBJECT) $(RPB_CONTEXT_REPLICATION_GATE_OBJECT) $(RPB_CONTEXT_REPLICATION_RETAINED_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/context_replication_adapter_test: $(RPB_TEST_DIR)/context_replication_adapter_test.o $(RPB_CONTEXT_REPLICATION_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-context-replication: $(RPB_TEST_DIR)/context_replication_adapter_test
	$(RPB_TEST_DIR)/context_replication_adapter_test

evaluate-context-replication: $(CONTEXT_REPLICATION_BIN)
	CONTEXT_REPLICATION_BIN="$(abspath $(CONTEXT_REPLICATION_BIN))" CONTEXT_REPLICATION_SOURCE_INPUTS="$(CONTEXT_REPLICATION_PROVENANCE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-context-replication.sh

# One lighter training recipe on known development sources; no TEST API.
CONTEXT_LIGHTER_BIN := $(BUILD_DIR)/embedding_context_lighter_validation
RPB_CONTEXT_LIGHTER_PAIR_GATE_OBJECT := $(RPB_OBJECT_DIR)/context_lighter_pair_gate.o
RPB_CONTEXT_LIGHTER_RETAINED_OBJECT := $(RPB_OBJECT_DIR)/context_lighter_retained.o
RPB_CONTEXT_LIGHTER_SERVING_GATE_OBJECT := $(RPB_OBJECT_DIR)/context_lighter_serving_gate.o
CONTEXT_LIGHTER_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/context_replication_adapter.cpp $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(EVALUATION_ROOT)/src/context_lighter_validation_main.cpp $(EVALUATION_ROOT)/cards/context_lighter_validation_v1.md $(CODE_ROOT)/scripts/check-context-lighter-validation.sh $(CODE_ROOT)/scripts/evaluate-context-lighter-validation.sh)
CONTEXT_LIGHTER_SOURCE_ID := $(shell sha256sum $(CONTEXT_LIGHTER_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
CONTEXT_LIGHTER_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(CONTEXT_LIGHTER_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: context-lighter-validation print-context-lighter-sources test-rpb-context-lighter evaluate-context-lighter-validation
context-lighter-validation: $(CONTEXT_LIGHTER_BIN)

print-context-lighter-sources:
	@printf '%s\n' $(CONTEXT_LIGHTER_PROVENANCE_INPUTS)

$(EVALUATION_OBJECT_DIR)/context_lighter_validation_main.o: $(EVALUATION_ROOT)/src/context_lighter_validation_main.cpp $(CONTEXT_LIGHTER_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_LIGHTER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_LIGHTER_PAIR_GATE_OBJECT): $(RPB_ROOT)/src/context_replication_adapter.cpp $(CONTEXT_LIGHTER_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_LIGHTER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_LIGHTER_RETAINED_OBJECT): $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(CONTEXT_LIGHTER_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_LIGHTER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_LIGHTER_SERVING_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(CONTEXT_LIGHTER_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_LIGHTER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(CONTEXT_LIGHTER_BIN): $(EVALUATION_OBJECT_DIR)/context_lighter_validation_main.o $(RPB_CONTEXT_LIGHTER_PAIR_GATE_OBJECT) $(RPB_CONTEXT_LIGHTER_RETAINED_OBJECT) $(RPB_CONTEXT_LIGHTER_SERVING_GATE_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/context_lighter_test: $(RPB_TEST_DIR)/context_lighter_test.o $(RPB_CONTEXT_LIGHTER_PAIR_GATE_OBJECT) $(RPB_CONTEXT_REPLAY_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-context-lighter: $(RPB_TEST_DIR)/context_lighter_test
	$(RPB_TEST_DIR)/context_lighter_test

evaluate-context-lighter-validation: $(CONTEXT_LIGHTER_BIN)
	CONTEXT_LIGHTER_BIN="$(abspath $(CONTEXT_LIGHTER_BIN))" CONTEXT_LIGHTER_SOURCE_INPUTS="$(CONTEXT_LIGHTER_PROVENANCE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-context-lighter-validation.sh

# One alternating ordinary/deleted training recipe on known development sources.
CONTEXT_BALANCED_BIN := $(BUILD_DIR)/embedding_context_balanced_validation
RPB_CONTEXT_BALANCED_PAIR_GATE_OBJECT := $(RPB_OBJECT_DIR)/context_balanced_pair_gate.o
RPB_CONTEXT_BALANCED_RETAINED_OBJECT := $(RPB_OBJECT_DIR)/context_balanced_retained.o
RPB_CONTEXT_BALANCED_SERVING_GATE_OBJECT := $(RPB_OBJECT_DIR)/context_balanced_serving_gate.o
CONTEXT_BALANCED_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/context_replication_adapter.cpp $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(EVALUATION_ROOT)/src/context_balanced_validation_main.cpp $(EVALUATION_ROOT)/cards/context_balanced_validation_v1.md $(CODE_ROOT)/scripts/check-context-balanced-validation.sh $(CODE_ROOT)/scripts/evaluate-context-balanced-validation.sh)
CONTEXT_BALANCED_SOURCE_ID := $(shell sha256sum $(CONTEXT_BALANCED_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
CONTEXT_BALANCED_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(CONTEXT_BALANCED_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: context-balanced-validation print-context-balanced-sources test-rpb-context-balanced evaluate-context-balanced-validation
context-balanced-validation: $(CONTEXT_BALANCED_BIN)

print-context-balanced-sources:
	@printf '%s\n' $(CONTEXT_BALANCED_PROVENANCE_INPUTS)

$(EVALUATION_OBJECT_DIR)/context_balanced_validation_main.o: $(EVALUATION_ROOT)/src/context_balanced_validation_main.cpp $(CONTEXT_BALANCED_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_BALANCED_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_BALANCED_PAIR_GATE_OBJECT): $(RPB_ROOT)/src/context_replication_adapter.cpp $(CONTEXT_BALANCED_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_BALANCED_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_BALANCED_RETAINED_OBJECT): $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(CONTEXT_BALANCED_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_BALANCED_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_CONTEXT_BALANCED_SERVING_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(CONTEXT_BALANCED_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(CONTEXT_BALANCED_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(CONTEXT_BALANCED_BIN): $(EVALUATION_OBJECT_DIR)/context_balanced_validation_main.o $(RPB_CONTEXT_BALANCED_PAIR_GATE_OBJECT) $(RPB_CONTEXT_BALANCED_RETAINED_OBJECT) $(RPB_CONTEXT_BALANCED_SERVING_GATE_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/context_balanced_test: $(RPB_TEST_DIR)/context_balanced_test.o $(RPB_CONTEXT_BALANCED_PAIR_GATE_OBJECT) $(RPB_CONTEXT_BALANCED_RETAINED_OBJECT) $(RPB_CONTEXT_REPLAY_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-context-balanced: $(RPB_TEST_DIR)/context_balanced_test
	$(RPB_TEST_DIR)/context_balanced_test

evaluate-context-balanced-validation: $(CONTEXT_BALANCED_BIN)
	CONTEXT_BALANCED_BIN="$(abspath $(CONTEXT_BALANCED_BIN))" CONTEXT_BALANCED_SOURCE_INPUTS="$(CONTEXT_BALANCED_PROVENANCE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-context-balanced-validation.sh

# TRAIN-only frozen-weight diagnosis; no mixed-split evaluator or training driver.
TRAINING_OBJECTIVE_BIN := $(BUILD_DIR)/embedding_training_objective_diagnostic
RPB_TRAINING_OBJECTIVE_OBJECT := $(RPB_OBJECT_DIR)/training_objective_diagnostic.o
TRAINING_OBJECTIVE_PROVENANCE_INPUTS := $(sort $(RPB_CORE_HEADERS) $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h $(RPB_ROOT)/src/workflow.cpp $(RPB_ROOT)/src/training_objective_diagnostic.cpp $(RPB_ROOT)/tests/training_objective_diagnostic_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h $(wildcard $(CODE_ROOT)/shared/include/embedding/shared/*.h) $(CODE_ROOT)/shared/src/data.cpp $(EVALUATION_ROOT)/src/training_objective_diagnostic_main.cpp $(EVALUATION_ROOT)/cards/training_objective_diagnostic_v1.md $(CODE_ROOT)/scripts/prepare-training-objective-inputs.py $(CODE_ROOT)/scripts/check-training-objective-diagnostic.sh $(CODE_ROOT)/scripts/evaluate-training-objective-diagnostic.sh Makefile dependencies.lock)
TRAINING_OBJECTIVE_SOURCE_ID := $(shell sha256sum $(TRAINING_OBJECTIVE_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
TRAINING_OBJECTIVE_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(TRAINING_OBJECTIVE_SOURCE_ID)\"
.PHONY: training-objective-diagnostic print-training-objective-sources test-rpb-training-objective-diagnostic evaluate-training-objective-diagnostic
training-objective-diagnostic: $(TRAINING_OBJECTIVE_BIN)

print-training-objective-sources:
	@printf '%s\n' $(TRAINING_OBJECTIVE_PROVENANCE_INPUTS)

$(EVALUATION_OBJECT_DIR)/training_objective_diagnostic_main.o: $(EVALUATION_ROOT)/src/training_objective_diagnostic_main.cpp $(TRAINING_OBJECTIVE_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(TRAINING_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_TRAINING_OBJECTIVE_OBJECT): $(RPB_ROOT)/src/training_objective_diagnostic.cpp $(TRAINING_OBJECTIVE_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(TRAINING_OBJECTIVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(TRAINING_OBJECTIVE_BIN): $(EVALUATION_OBJECT_DIR)/training_objective_diagnostic_main.o $(RPB_TRAINING_OBJECTIVE_OBJECT) $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/training_objective_diagnostic_test.o: $(RPB_ROOT)/tests/training_objective_diagnostic_test.cpp $(TRAINING_OBJECTIVE_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(TRAINING_OBJECTIVE_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/training_objective_diagnostic_test: $(RPB_TEST_DIR)/training_objective_diagnostic_test.o $(RPB_TRAINING_OBJECTIVE_OBJECT) $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-training-objective-diagnostic: $(RPB_TEST_DIR)/training_objective_diagnostic_test
	$(RPB_TEST_DIR)/training_objective_diagnostic_test

evaluate-training-objective-diagnostic: $(TRAINING_OBJECTIVE_BIN)
	TRAINING_OBJECTIVE_BIN="$(abspath $(TRAINING_OBJECTIVE_BIN))" TRAINING_OBJECTIVE_SOURCE_INPUTS="$(TRAINING_OBJECTIVE_PROVENANCE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-training-objective-diagnostic.sh

# One native-space view-agreement objective; shared scoring stays unchanged.
NATIVE_VIEW_AGREEMENT_VALIDATION_BIN := $(BUILD_DIR)/embedding_native_view_agreement_validation
RPB_NATIVE_VIEW_AGREEMENT_OBJECT := $(RPB_OBJECT_DIR)/native_view_agreement.o
RPB_NATIVE_VIEW_AGREEMENT_PAIR_GATE_OBJECT := $(RPB_OBJECT_DIR)/native_view_agreement_pair_gate.o
RPB_NATIVE_VIEW_AGREEMENT_RETAINED_OBJECT := $(RPB_OBJECT_DIR)/native_view_agreement_retained.o
RPB_NATIVE_VIEW_AGREEMENT_SERVING_GATE_OBJECT := $(RPB_OBJECT_DIR)/native_view_agreement_serving_gate.o
NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/native_view_agreement.cpp $(RPB_ROOT)/src/context_replication_adapter.cpp $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(EVALUATION_ROOT)/src/native_view_agreement_validation_main.cpp $(EVALUATION_ROOT)/cards/native_view_agreement_validation_v1.md $(CODE_ROOT)/scripts/check-native-view-agreement-validation.sh $(CODE_ROOT)/scripts/evaluate-native-view-agreement-validation.sh)
NATIVE_VIEW_AGREEMENT_VALIDATION_SOURCE_ID := $(shell sha256sum $(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS) | sha256sum | cut -d ' ' -f 1)
NATIVE_VIEW_AGREEMENT_VALIDATION_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(NATIVE_VIEW_AGREEMENT_VALIDATION_SOURCE_ID)\" -DEVALUATION_GIT_HEAD=\"$(SOURCE_GIT_HEAD)\" -DEVALUATION_GIT_DIRTY=\"$(SOURCE_GIT_DIRTY)\"
.PHONY: native-view-agreement-validation print-native-view-agreement-validation-sources test-rpb-native-view-agreement evaluate-native-view-agreement-validation
native-view-agreement-validation: $(NATIVE_VIEW_AGREEMENT_VALIDATION_BIN)

print-native-view-agreement-validation-sources:
	@printf '%s\n' $(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS)

$(EVALUATION_OBJECT_DIR)/native_view_agreement_validation_main.o: $(EVALUATION_ROOT)/src/native_view_agreement_validation_main.cpp $(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_VIEW_AGREEMENT_VALIDATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_NATIVE_VIEW_AGREEMENT_OBJECT): $(RPB_ROOT)/src/native_view_agreement.cpp $(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_VIEW_AGREEMENT_VALIDATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_NATIVE_VIEW_AGREEMENT_PAIR_GATE_OBJECT): $(RPB_ROOT)/src/context_replication_adapter.cpp $(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_VIEW_AGREEMENT_VALIDATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_NATIVE_VIEW_AGREEMENT_RETAINED_OBJECT): $(RPB_ROOT)/src/paired_pooling_adapter.cpp $(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_VIEW_AGREEMENT_VALIDATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_NATIVE_VIEW_AGREEMENT_SERVING_GATE_OBJECT): $(RPB_ROOT)/src/native_curve_gate.cpp $(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_VIEW_AGREEMENT_VALIDATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(NATIVE_VIEW_AGREEMENT_VALIDATION_BIN): $(EVALUATION_OBJECT_DIR)/native_view_agreement_validation_main.o $(RPB_NATIVE_VIEW_AGREEMENT_OBJECT) $(RPB_NATIVE_VIEW_AGREEMENT_PAIR_GATE_OBJECT) $(RPB_NATIVE_VIEW_AGREEMENT_RETAINED_OBJECT) $(RPB_NATIVE_VIEW_AGREEMENT_SERVING_GATE_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/native_view_agreement_test.o: $(RPB_ROOT)/tests/native_view_agreement_test.cpp $(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(NATIVE_VIEW_AGREEMENT_VALIDATION_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/native_view_agreement_test: $(RPB_TEST_DIR)/native_view_agreement_test.o $(RPB_NATIVE_VIEW_AGREEMENT_OBJECT) $(RPB_NATIVE_VIEW_AGREEMENT_RETAINED_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-native-view-agreement: $(RPB_TEST_DIR)/native_view_agreement_test
	$(RPB_TEST_DIR)/native_view_agreement_test

evaluate-native-view-agreement-validation: $(NATIVE_VIEW_AGREEMENT_VALIDATION_BIN)
	NATIVE_VIEW_AGREEMENT_VALIDATION_BIN="$(abspath $(NATIVE_VIEW_AGREEMENT_VALIDATION_BIN))" NATIVE_VIEW_AGREEMENT_VALIDATION_SOURCE_INPUTS="$(NATIVE_VIEW_AGREEMENT_VALIDATION_PROVENANCE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-native-view-agreement-validation.sh

# Frozen v7 encoder, bounded decoder-only diagnosis; no readout fits or TEST.
V7_DECODER_CALIBRATION_BIN := $(BUILD_DIR)/embedding_v7_decoder_calibration
RPB_DECODER_CALIBRATION_OBJECT := $(RPB_OBJECT_DIR)/decoder_calibration.o
RPB_FROZEN_DECODER_CALIBRATION_OBJECT := $(RPB_OBJECT_DIR)/frozen_decoder_calibration.o
V7_DECODER_CALIBRATION_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h $(RPB_ROOT)/src/frozen_decoder_calibration.cpp $(RPB_ROOT)/src/decoder_calibration.cpp $(RPB_ROOT)/tests/decoder_calibration_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h $(CODE_ROOT)/shared/tests/shared_test_support.h $(EVALUATION_ROOT)/src/v7_decoder_calibration_main.cpp $(EVALUATION_ROOT)/include/frozen_role_guard.h $(EVALUATION_ROOT)/tests/frozen_role_guard_test.cpp $(EVALUATION_ROOT)/cards/v7_decoder_calibration_v1.md $(CODE_ROOT)/scripts/check-v7-decoder-calibration.sh $(CODE_ROOT)/scripts/evaluate-v7-decoder-calibration.sh $(CODE_ROOT)/scripts/prepare-v7-decoder-inputs.py)
V7_DECODER_CALIBRATION_SOURCE_ID := $(shell sha256sum $(V7_DECODER_CALIBRATION_INPUTS) | sha256sum | cut -d ' ' -f 1)
V7_DECODER_CALIBRATION_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(V7_DECODER_CALIBRATION_SOURCE_ID)\" -DDECODER_CALIBRATION_SOURCE_ID=\"$(V7_DECODER_CALIBRATION_SOURCE_ID)\"
.PHONY: v7-decoder-calibration print-v7-decoder-calibration-sources test-rpb-decoder-calibration test-frozen-role-guard evaluate-v7-decoder-calibration
v7-decoder-calibration: $(V7_DECODER_CALIBRATION_BIN)
print-v7-decoder-calibration-sources:
	@printf '%s\n' $(V7_DECODER_CALIBRATION_INPUTS)

$(EVALUATION_OBJECT_DIR)/v7_decoder_calibration_main.o: $(EVALUATION_ROOT)/src/v7_decoder_calibration_main.cpp $(V7_DECODER_CALIBRATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(V7_DECODER_CALIBRATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_DECODER_CALIBRATION_OBJECT): $(RPB_ROOT)/src/decoder_calibration.cpp $(V7_DECODER_CALIBRATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(V7_DECODER_CALIBRATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(V7_DECODER_CALIBRATION_BIN): $(EVALUATION_OBJECT_DIR)/v7_decoder_calibration_main.o $(RPB_DECODER_CALIBRATION_OBJECT) $(RPB_FROZEN_DECODER_CALIBRATION_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(RPB_WORKFLOW_OBJECT) $(EVALUATION_COMMON_OBJECTS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/decoder_calibration_test.o: $(RPB_ROOT)/tests/decoder_calibration_test.cpp $(V7_DECODER_CALIBRATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(V7_DECODER_CALIBRATION_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/decoder_calibration_test: $(RPB_TEST_DIR)/decoder_calibration_test.o $(RPB_DECODER_CALIBRATION_OBJECT) $(RPB_FROZEN_DECODER_CALIBRATION_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-decoder-calibration: $(RPB_TEST_DIR)/decoder_calibration_test
	$(RPB_TEST_DIR)/decoder_calibration_test

$(EVALUATION_OBJECT_DIR)/frozen_role_guard_test: $(EVALUATION_ROOT)/tests/frozen_role_guard_test.cpp $(EVALUATION_ROOT)/include/frozen_role_guard.h
	mkdir -p "$(@D)"
	$(CXX) -std=c++20 -O1 -Wall -Wextra -I$(EVALUATION_ROOT)/include $< -o $@

test-frozen-role-guard: $(EVALUATION_OBJECT_DIR)/frozen_role_guard_test
	$(EVALUATION_OBJECT_DIR)/frozen_role_guard_test

evaluate-v7-decoder-calibration: $(V7_DECODER_CALIBRATION_BIN)
	V7_DECODER_CALIBRATION_BIN="$(abspath $(V7_DECODER_CALIBRATION_BIN))" V7_DECODER_CALIBRATION_SOURCE_INPUTS="$(V7_DECODER_CALIBRATION_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-v7-decoder-calibration.sh

# Fresh v4/v7 encoders share the frozen-decoder implementation while retaining
# separate public protocol bindings, artifact identities and producer scopes.
FRESH_DECODER_REPLICATION_BIN := $(BUILD_DIR)/embedding_fresh_decoder_replication
FIXED_FEATURE_READOUTS_OBJECT := $(OBJECT_DIR)/shared/fixed_feature_readouts.o
FRESH_DECODER_REPLICATION_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h $(RPB_ROOT)/src/frozen_decoder_calibration.cpp $(RPB_ROOT)/src/decoder_calibration.cpp $(RPB_ROOT)/tests/frozen_decoder_calibration_test.cpp $(RPB_ROOT)/tests/decoder_calibration_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h $(CODE_ROOT)/shared/tests/shared_test_support.h $(CODE_ROOT)/shared/include/embedding/shared/fixed_feature_readouts.h $(CODE_ROOT)/shared/src/fixed_feature_readouts.cpp $(CODE_ROOT)/shared/tests/fixed_feature_readouts_test.cpp $(EVALUATION_ROOT)/src/fresh_decoder_replication_main.cpp $(EVALUATION_ROOT)/include/frozen_role_guard.h $(EVALUATION_ROOT)/tests/frozen_role_guard_test.cpp $(EVALUATION_ROOT)/cards/fresh_decoder_replication_v1.md $(CODE_ROOT)/scripts/check-fresh-decoder-replication.sh $(CODE_ROOT)/scripts/evaluate-fresh-decoder-replication.sh $(CODE_ROOT)/scripts/prepare-fresh-decoder-replication.py)
FRESH_DECODER_REPLICATION_SOURCE_ID := $(shell sha256sum $(FRESH_DECODER_REPLICATION_INPUTS) | sha256sum | cut -d ' ' -f 1)
FRESH_DECODER_REPLICATION_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(FRESH_DECODER_REPLICATION_SOURCE_ID)\" -DFROZEN_DECODER_CALIBRATION_SOURCE_ID=\"$(FRESH_DECODER_REPLICATION_SOURCE_ID)\"
.PHONY: fresh-decoder-replication print-fresh-decoder-replication-sources test-rpb-frozen-decoder-calibration test-fixed-feature-readouts evaluate-fresh-decoder-replication
fresh-decoder-replication: $(FRESH_DECODER_REPLICATION_BIN)
print-fresh-decoder-replication-sources:
	@printf '%s\n' $(FRESH_DECODER_REPLICATION_INPUTS)

$(EVALUATION_OBJECT_DIR)/fresh_decoder_replication_main.o: $(EVALUATION_ROOT)/src/fresh_decoder_replication_main.cpp $(FRESH_DECODER_REPLICATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FRESH_DECODER_REPLICATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_FROZEN_DECODER_CALIBRATION_OBJECT): $(RPB_ROOT)/src/frozen_decoder_calibration.cpp $(FRESH_DECODER_REPLICATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FRESH_DECODER_REPLICATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(FIXED_FEATURE_READOUTS_OBJECT): $(CODE_ROOT)/shared/src/fixed_feature_readouts.cpp $(CODE_ROOT)/shared/include/embedding/shared/fixed_feature_readouts.h $(CODE_ROOT)/shared/include/embedding/shared/feature_harness.h
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/fixed_feature_readouts_test.o: $(CODE_ROOT)/shared/tests/fixed_feature_readouts_test.cpp $(FRESH_DECODER_REPLICATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) -DEVALUATION_SOURCE_ID=\"$(FRESH_DECODER_REPLICATION_SOURCE_ID)\" $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/fixed_feature_readouts_test: $(SHARED_TEST_DIR)/fixed_feature_readouts_test.o $(FIXED_FEATURE_READOUTS_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-fixed-feature-readouts: $(SHARED_TEST_DIR)/fixed_feature_readouts_test
	$(SHARED_TEST_DIR)/fixed_feature_readouts_test

$(FRESH_DECODER_REPLICATION_BIN): $(EVALUATION_OBJECT_DIR)/fresh_decoder_replication_main.o $(RPB_FROZEN_DECODER_CALIBRATION_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/frozen_decoder_calibration_test.o: $(RPB_ROOT)/tests/frozen_decoder_calibration_test.cpp $(FRESH_DECODER_REPLICATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FRESH_DECODER_REPLICATION_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/frozen_decoder_calibration_test: $(RPB_TEST_DIR)/frozen_decoder_calibration_test.o $(RPB_FROZEN_DECODER_CALIBRATION_OBJECT) $(RPB_DECODER_CALIBRATION_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-frozen-decoder-calibration: $(RPB_TEST_DIR)/frozen_decoder_calibration_test
	$(RPB_TEST_DIR)/frozen_decoder_calibration_test

evaluate-fresh-decoder-replication: $(FRESH_DECODER_REPLICATION_BIN)
	FRESH_DECODER_REPLICATION_BIN="$(abspath $(FRESH_DECODER_REPLICATION_BIN))" FRESH_DECODER_REPLICATION_SOURCE_INPUTS="$(FRESH_DECODER_REPLICATION_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-fresh-decoder-replication.sh

# Generic saved-TRAIN arithmetic; no encoder or fitted-head implementation links.
SAVED_NATIVE_RELIABILITY_BIN := $(BUILD_DIR)/embedding_saved_feature_reliability
SAVED_FEATURE_RELIABILITY_OBJECT := $(OBJECT_DIR)/shared/saved_feature_reliability.o
SAVED_NATIVE_RELIABILITY_INPUTS := $(sort Makefile dependencies.lock $(CODE_ROOT)/shared/include/embedding/shared/types.h $(CODE_ROOT)/shared/include/embedding/shared/data.h $(CODE_ROOT)/shared/src/data.cpp $(CODE_ROOT)/shared/include/embedding/shared/saved_feature_reliability.h $(CODE_ROOT)/shared/src/saved_feature_reliability.cpp $(CODE_ROOT)/shared/tests/saved_feature_reliability_test.cpp $(CODE_ROOT)/shared/tests/shared_test_support.h $(EVALUATION_ROOT)/src/saved_feature_reliability_main.cpp $(EVALUATION_ROOT)/include/frozen_role_guard.h $(EVALUATION_ROOT)/tests/frozen_role_guard_test.cpp $(EVALUATION_ROOT)/cards/saved_native_reliability_v1.md $(RPB_ROOT)/FROZEN_NATIVE_RELIABILITY_PLAN.md $(CODE_ROOT)/scripts/check-saved-native-reliability.sh $(CODE_ROOT)/scripts/evaluate-saved-native-reliability.sh $(CODE_ROOT)/scripts/prepare-saved-native-reliability.py $(CODE_ROOT)/scripts/prepare-fresh-decoder-replication.py $(EVALUATION_ROOT)/src/fresh_decoder_replication_main.cpp $(CODE_ROOT)/shared/src/fixed_feature_readouts.cpp)
SAVED_NATIVE_RELIABILITY_SOURCE_ID := $(shell sha256sum $(SAVED_NATIVE_RELIABILITY_INPUTS) | sha256sum | cut -d ' ' -f 1)
SAVED_NATIVE_RELIABILITY_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(SAVED_NATIVE_RELIABILITY_SOURCE_ID)\"
.PHONY: saved-native-reliability print-saved-native-reliability-sources test-saved-feature-reliability evaluate-saved-native-reliability
saved-native-reliability: $(SAVED_NATIVE_RELIABILITY_BIN)
print-saved-native-reliability-sources:
	@printf '%s\n' $(SAVED_NATIVE_RELIABILITY_INPUTS)

$(EVALUATION_OBJECT_DIR)/saved_feature_reliability_main.o: $(EVALUATION_ROOT)/src/saved_feature_reliability_main.cpp $(SAVED_NATIVE_RELIABILITY_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(SAVED_NATIVE_RELIABILITY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SAVED_FEATURE_RELIABILITY_OBJECT): $(CODE_ROOT)/shared/src/saved_feature_reliability.cpp $(CODE_ROOT)/shared/include/embedding/shared/saved_feature_reliability.h $(CODE_ROOT)/shared/include/embedding/shared/data.h
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SAVED_NATIVE_RELIABILITY_BIN): $(EVALUATION_OBJECT_DIR)/saved_feature_reliability_main.o $(SAVED_FEATURE_RELIABILITY_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(SHARED_TEST_DIR)/saved_feature_reliability_test.o: $(CODE_ROOT)/shared/tests/saved_feature_reliability_test.cpp $(SAVED_NATIVE_RELIABILITY_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(SAVED_NATIVE_RELIABILITY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_TEST_DIR)/saved_feature_reliability_test: $(SHARED_TEST_DIR)/saved_feature_reliability_test.o $(SAVED_FEATURE_RELIABILITY_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-saved-feature-reliability: $(SHARED_TEST_DIR)/saved_feature_reliability_test
	$(SHARED_TEST_DIR)/saved_feature_reliability_test

evaluate-saved-native-reliability: $(SAVED_NATIVE_RELIABILITY_BIN)
	SAVED_NATIVE_RELIABILITY_BIN="$(abspath $(SAVED_NATIVE_RELIABILITY_BIN))" SAVED_NATIVE_RELIABILITY_SOURCE_INPUTS="$(SAVED_NATIVE_RELIABILITY_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-saved-native-reliability.sh

# Frozen encoder transfer: a separate backend object binds the new extractor
# identity without changing legacy binary/object scopes or quality protocols.
FROZEN_AMPLITUDE_TRANSFER_BIN := $(BUILD_DIR)/embedding_frozen_amplitude_transfer
FROZEN_NATIVE_FEATURE_OBJECT := $(RPB_OBJECT_DIR)/frozen_native_feature_adapter.o
FROZEN_AMPLITUDE_TRANSFER_INPUTS := $(sort $(NATIVE_CURVE_PROVENANCE_INPUTS) $(RPB_ROOT)/src/frozen_decoder_calibration.cpp $(RPB_ROOT)/src/decoder_calibration.cpp $(RPB_ROOT)/tests/frozen_native_feature_adapter_test.cpp $(RPB_ROOT)/tests/frozen_decoder_calibration_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h $(CODE_ROOT)/shared/tests/shared_test_support.h $(CODE_ROOT)/shared/tests/fixed_feature_readouts_test.cpp $(EVALUATION_ROOT)/src/frozen_amplitude_transfer_main.cpp $(EVALUATION_ROOT)/src/fresh_decoder_replication_main.cpp $(EVALUATION_ROOT)/include/frozen_role_guard.h $(EVALUATION_ROOT)/tests/frozen_role_guard_test.cpp $(EVALUATION_ROOT)/cards/frozen_amplitude_transfer_v1.md $(CODE_ROOT)/scripts/check-frozen-amplitude-transfer.sh $(CODE_ROOT)/scripts/evaluate-frozen-amplitude-transfer.sh $(CODE_ROOT)/scripts/prepare-frozen-amplitude-transfer.py $(CODE_ROOT)/scripts/prepare-fresh-decoder-replication.py)
FROZEN_AMPLITUDE_TRANSFER_SOURCE_ID := $(shell sha256sum $(FROZEN_AMPLITUDE_TRANSFER_INPUTS) | sha256sum | cut -d ' ' -f 1)
FROZEN_AMPLITUDE_TRANSFER_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(FROZEN_AMPLITUDE_TRANSFER_SOURCE_ID)\" -DFROZEN_NATIVE_FEATURE_SOURCE_ID=\"$(FROZEN_AMPLITUDE_TRANSFER_SOURCE_ID)\" -DFROZEN_DECODER_CALIBRATION_SOURCE_ID=\"$(FROZEN_AMPLITUDE_TRANSFER_SOURCE_ID)\"
.PHONY: frozen-amplitude-transfer print-frozen-amplitude-transfer-sources test-rpb-frozen-native-feature test-frozen-amplitude-legacy evaluate-frozen-amplitude-transfer
frozen-amplitude-transfer: $(FROZEN_AMPLITUDE_TRANSFER_BIN)
print-frozen-amplitude-transfer-sources:
	@printf '%s\n' $(FROZEN_AMPLITUDE_TRANSFER_INPUTS)

$(EVALUATION_OBJECT_DIR)/frozen_amplitude_transfer_main.o: $(EVALUATION_ROOT)/src/frozen_amplitude_transfer_main.cpp $(FROZEN_AMPLITUDE_TRANSFER_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FROZEN_AMPLITUDE_TRANSFER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(FROZEN_NATIVE_FEATURE_OBJECT): $(RPB_ROOT)/src/frozen_decoder_calibration.cpp $(FROZEN_AMPLITUDE_TRANSFER_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FROZEN_AMPLITUDE_TRANSFER_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(FROZEN_AMPLITUDE_TRANSFER_BIN): $(EVALUATION_OBJECT_DIR)/frozen_amplitude_transfer_main.o $(FROZEN_NATIVE_FEATURE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/frozen_native_feature_adapter_test.o: $(RPB_ROOT)/tests/frozen_native_feature_adapter_test.cpp $(FROZEN_AMPLITUDE_TRANSFER_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FROZEN_AMPLITUDE_TRANSFER_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/frozen_native_feature_adapter_test: $(RPB_TEST_DIR)/frozen_native_feature_adapter_test.o $(FROZEN_NATIVE_FEATURE_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-frozen-native-feature: $(RPB_TEST_DIR)/frozen_native_feature_adapter_test
	$(RPB_TEST_DIR)/frozen_native_feature_adapter_test

# Execute the existing legacy test source against the new shared backend.
$(RPB_TEST_DIR)/frozen_amplitude_legacy_test.o: $(RPB_ROOT)/tests/frozen_decoder_calibration_test.cpp $(FROZEN_AMPLITUDE_TRANSFER_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(FROZEN_AMPLITUDE_TRANSFER_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/frozen_amplitude_legacy_test: $(RPB_TEST_DIR)/frozen_amplitude_legacy_test.o $(FROZEN_NATIVE_FEATURE_OBJECT) $(RPB_DECODER_CALIBRATION_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-frozen-amplitude-legacy: $(RPB_TEST_DIR)/frozen_amplitude_legacy_test
	$(RPB_TEST_DIR)/frozen_amplitude_legacy_test

evaluate-frozen-amplitude-transfer: $(FROZEN_AMPLITUDE_TRANSFER_BIN)
	FROZEN_AMPLITUDE_TRANSFER_BIN="$(abspath $(FROZEN_AMPLITUDE_TRANSFER_BIN))" FROZEN_AMPLITUDE_TRANSFER_SOURCE_INPUTS="$(FROZEN_AMPLITUDE_TRANSFER_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-frozen-amplitude-transfer.sh

# Fresh matched early-mixer timing experiment. Historical targets and source
# scopes remain separate; snapshots never construct a CPU encoder provider.
EARLY_MIXER_RELIABILITY_BIN := $(BUILD_DIR)/embedding_early_mixer_reliability
EARLY_MIXER_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/early_mixer_adapter.o
EARLY_MIXER_LEGACY_FROZEN_OBJECT := $(RPB_OBJECT_DIR)/early_mixer_legacy_frozen.o
EARLY_MIXER_RELIABILITY_INPUTS := $(sort $(RPB_PROVENANCE_INPUTS) $(EVALUATION_PROVENANCE_INPUTS) $(NATIVE_CURVE_PROVENANCE_INPUTS) $(PAIRED_POOLING_PROVENANCE_INPUTS) $(ARCHIVE_READOUT_PROVENANCE_INPUTS) $(V7_DECODER_CALIBRATION_INPUTS) $(FRESH_DECODER_REPLICATION_INPUTS) $(wildcard $(RPB_ROOT)/src/*.cpp) $(RPB_ROOT)/tests/early_mixer_model_test.cpp $(RPB_ROOT)/tests/early_mixer_adapter_test.cpp $(RPB_ROOT)/tests/rpb_test_support.h $(CODE_ROOT)/shared/tests/shared_test_support.h $(CODE_ROOT)/shared/tests/fixed_feature_readouts_test.cpp $(EVALUATION_ROOT)/src/early_mixer_reliability_main.cpp $(EVALUATION_ROOT)/include/frozen_role_guard.h $(EVALUATION_ROOT)/tests/frozen_role_guard_test.cpp $(EVALUATION_ROOT)/cards/early_mixer_reliability_v1.md $(CODE_ROOT)/scripts/check-early-mixer-reliability.sh $(CODE_ROOT)/scripts/evaluate-early-mixer-reliability.sh $(CODE_ROOT)/scripts/prepare-early-mixer-reliability.py $(CODE_ROOT)/scripts/prepare-fresh-decoder-replication.py)
EARLY_MIXER_RELIABILITY_SOURCE_ID := $(shell sha256sum $(EARLY_MIXER_RELIABILITY_INPUTS) | sha256sum | cut -d ' ' -f 1)
EARLY_MIXER_RELIABILITY_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(EARLY_MIXER_RELIABILITY_SOURCE_ID)\" -DEARLY_MIXER_ADAPTER_SOURCE_ID=\"$(EARLY_MIXER_RELIABILITY_SOURCE_ID)\"
.PHONY: early-mixer-reliability print-early-mixer-reliability-sources test-rpb-early-mixer-model test-rpb-early-mixer-adapter evaluate-early-mixer-reliability
early-mixer-reliability: $(EARLY_MIXER_RELIABILITY_BIN)
print-early-mixer-reliability-sources:
	@printf '%s\n' $(EARLY_MIXER_RELIABILITY_INPUTS)

$(EVALUATION_OBJECT_DIR)/early_mixer_reliability_main.o: $(EVALUATION_ROOT)/src/early_mixer_reliability_main.cpp $(EARLY_MIXER_RELIABILITY_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_RELIABILITY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EARLY_MIXER_ADAPTER_OBJECT): $(RPB_ROOT)/src/early_mixer_adapter.cpp $(EARLY_MIXER_RELIABILITY_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_RELIABILITY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EARLY_MIXER_LEGACY_FROZEN_OBJECT): $(RPB_ROOT)/src/frozen_decoder_calibration.cpp $(EARLY_MIXER_RELIABILITY_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_RELIABILITY_CPPFLAGS) -DFROZEN_DECODER_CALIBRATION_SOURCE_ID=\"$(EARLY_MIXER_RELIABILITY_SOURCE_ID)\" -DFROZEN_NATIVE_FEATURE_SOURCE_ID=\"$(EARLY_MIXER_RELIABILITY_SOURCE_ID)\" $(CXXFLAGS) -c $< -o $@

$(EARLY_MIXER_RELIABILITY_BIN): $(EVALUATION_OBJECT_DIR)/early_mixer_reliability_main.o $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/early_mixer_model_test.o: $(RPB_ROOT)/tests/early_mixer_model_test.cpp $(EARLY_MIXER_RELIABILITY_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/early_mixer_model_test: $(RPB_TEST_DIR)/early_mixer_model_test.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-early-mixer-model: $(RPB_TEST_DIR)/early_mixer_model_test
	$(RPB_TEST_DIR)/early_mixer_model_test

$(RPB_TEST_DIR)/early_mixer_adapter_test.o: $(RPB_ROOT)/tests/early_mixer_adapter_test.cpp $(EARLY_MIXER_RELIABILITY_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_RELIABILITY_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/early_mixer_adapter_test: $(RPB_TEST_DIR)/early_mixer_adapter_test.o $(EARLY_MIXER_ADAPTER_OBJECT) $(EARLY_MIXER_LEGACY_FROZEN_OBJECT) $(RPB_DECODER_CALIBRATION_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-early-mixer-adapter: $(RPB_TEST_DIR)/early_mixer_adapter_test
	$(RPB_TEST_DIR)/early_mixer_adapter_test

evaluate-early-mixer-reliability: $(EARLY_MIXER_RELIABILITY_BIN)
	@EARLY_MIXER_RELIABILITY_BIN="$(abspath $(EARLY_MIXER_RELIABILITY_BIN))" EARLY_MIXER_RELIABILITY_SOURCE_INPUTS="$(EARLY_MIXER_RELIABILITY_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-early-mixer-reliability.sh

# Fresh continuous training curves keep their own source identity and adapter
# object. The closure includes every scope linked by the binary and admission.
EARLY_MIXER_LEARNING_CURVE_BIN := $(BUILD_DIR)/embedding_early_mixer_learning_curve
EARLY_MIXER_CURVE_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/early_mixer_curve_adapter.o
EARLY_MIXER_LEARNING_CURVE_INPUTS := $(sort $(EARLY_MIXER_RELIABILITY_INPUTS) $(RPB_ROOT)/tests/early_mixer_curve_adapter_test.cpp $(EVALUATION_ROOT)/src/early_mixer_learning_curve_main.cpp $(EVALUATION_ROOT)/cards/early_mixer_learning_curve_v1.md $(CODE_ROOT)/scripts/check-early-mixer-learning-curve.sh $(CODE_ROOT)/scripts/evaluate-early-mixer-learning-curve.sh $(CODE_ROOT)/scripts/prepare-early-mixer-learning-curve.py)
EARLY_MIXER_LEARNING_CURVE_SOURCE_ID := $(shell sha256sum $(EARLY_MIXER_LEARNING_CURVE_INPUTS) | sha256sum | cut -d ' ' -f 1)
EARLY_MIXER_LEARNING_CURVE_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(EARLY_MIXER_LEARNING_CURVE_SOURCE_ID)\" -DEARLY_MIXER_ADAPTER_SOURCE_ID=\"$(EARLY_MIXER_LEARNING_CURVE_SOURCE_ID)\"
.PHONY: early-mixer-learning-curve print-early-mixer-learning-curve-sources test-rpb-early-mixer-curve-adapter evaluate-early-mixer-learning-curve
early-mixer-learning-curve: $(EARLY_MIXER_LEARNING_CURVE_BIN)
print-early-mixer-learning-curve-sources:
	@printf '%s\n' $(EARLY_MIXER_LEARNING_CURVE_INPUTS)

$(EVALUATION_OBJECT_DIR)/early_mixer_learning_curve_main.o: $(EVALUATION_ROOT)/src/early_mixer_learning_curve_main.cpp $(EARLY_MIXER_LEARNING_CURVE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_LEARNING_CURVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EARLY_MIXER_CURVE_ADAPTER_OBJECT): $(RPB_ROOT)/src/early_mixer_adapter.cpp $(EARLY_MIXER_LEARNING_CURVE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_LEARNING_CURVE_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EARLY_MIXER_LEARNING_CURVE_BIN): $(EVALUATION_OBJECT_DIR)/early_mixer_learning_curve_main.o $(EARLY_MIXER_CURVE_ADAPTER_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/early_mixer_curve_adapter_test.o: $(RPB_ROOT)/tests/early_mixer_curve_adapter_test.cpp $(EARLY_MIXER_LEARNING_CURVE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_LEARNING_CURVE_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/early_mixer_curve_adapter_test: $(RPB_TEST_DIR)/early_mixer_curve_adapter_test.o $(EARLY_MIXER_CURVE_ADAPTER_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-early-mixer-curve-adapter: $(RPB_TEST_DIR)/early_mixer_curve_adapter_test
	$(RPB_TEST_DIR)/early_mixer_curve_adapter_test

evaluate-early-mixer-learning-curve: $(EARLY_MIXER_LEARNING_CURVE_BIN)
	@EARLY_MIXER_LEARNING_CURVE_BIN="$(abspath $(EARLY_MIXER_LEARNING_CURVE_BIN))" EARLY_MIXER_LEARNING_CURVE_SOURCE_INPUTS="$(EARLY_MIXER_LEARNING_CURVE_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-early-mixer-learning-curve.sh

# Default-disabled source-gain primitive is shared by the learner. Every legacy
# learner consumer links it explicitly; historical target recipes remain intact.
RPB_TRAINING_SOURCE_GAIN_OBJECT := $(RPB_OBJECT_DIR)/training_source_gain.o
$(RPB_TRAINING_SOURCE_GAIN_OBJECT): $(RPB_ROOT)/src/training_source_gain.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/training_source_gain.h $(EVALUATION_PROVENANCE_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include $(CXXFLAGS) -c $< -o $@

RPB_LEGACY_SOURCE_GAIN_CONSUMERS := $(LEARNING_CURVE_BIN) $(GLOBAL_BOTTLENECK_BIN) $(NATIVE_CURVE_BIN) $(PAIRED_POOLING_BIN) $(OPTIMIZATION_DIAGNOSTIC_BIN) $(CONTEXT_DELETION_BIN) $(CONTEXT_OPTIMIZATION_BIN) $(CONTEXT_REPLICATION_BIN) $(CONTEXT_LIGHTER_BIN) $(CONTEXT_BALANCED_BIN) $(NATIVE_VIEW_AGREEMENT_VALIDATION_BIN) $(FRESH_DECODER_REPLICATION_BIN) $(FROZEN_AMPLITUDE_TRANSFER_BIN) $(EARLY_MIXER_RELIABILITY_BIN) $(EARLY_MIXER_LEARNING_CURVE_BIN) $(RPB_TEST_DIR)/learning_curve_adapter_test $(RPB_TEST_DIR)/native_curve_gate_test $(RPB_TEST_DIR)/optimization_diagnostic_adapter_test $(RPB_TEST_DIR)/context_deletion_test $(RPB_TEST_DIR)/context_deletion_adapter_test $(RPB_TEST_DIR)/context_replay_adapter_test $(RPB_TEST_DIR)/context_replication_adapter_test $(RPB_TEST_DIR)/context_lighter_test $(RPB_TEST_DIR)/context_balanced_test $(RPB_TEST_DIR)/native_view_agreement_test $(RPB_TEST_DIR)/decoder_calibration_test $(RPB_TEST_DIR)/frozen_decoder_calibration_test $(RPB_TEST_DIR)/frozen_native_feature_adapter_test $(RPB_TEST_DIR)/frozen_amplitude_legacy_test $(RPB_TEST_DIR)/early_mixer_adapter_test $(RPB_TEST_DIR)/early_mixer_curve_adapter_test
$(RPB_LEGACY_SOURCE_GAIN_CONSUMERS): $(RPB_TRAINING_SOURCE_GAIN_OBJECT)

# A fresh fixed512 comparison of one original early-mixer TRAIN view and one
# matched-target source-gain view. Its enclosing capture unions all link scopes.
MATCHED_TARGET_GAIN_BIN := $(BUILD_DIR)/embedding_matched_target_gain
MATCHED_TARGET_GAIN_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/matched_target_gain_adapter.o
MATCHED_TARGET_GAIN_INPUTS := $(sort $(EARLY_MIXER_LEARNING_CURVE_INPUTS) $(SDK_PROVENANCE_INPUTS) $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/training_source_gain.h $(RPB_ROOT)/src/training_source_gain.cpp $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/matched_target_gain_adapter.h $(RPB_ROOT)/src/matched_target_gain_adapter.cpp $(RPB_ROOT)/tests/matched_target_gain_adapter_test.cpp $(EVALUATION_ROOT)/src/matched_target_gain_main.cpp $(EVALUATION_ROOT)/cards/matched_target_gain_v1.md $(CODE_ROOT)/scripts/prepare-matched-target-gain.py $(CODE_ROOT)/scripts/check-matched-target-gain.sh $(CODE_ROOT)/scripts/evaluate-matched-target-gain.sh)
MATCHED_TARGET_GAIN_SOURCE_ID := $(shell sha256sum $(MATCHED_TARGET_GAIN_INPUTS) | sha256sum | cut -d ' ' -f 1)
MATCHED_TARGET_GAIN_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(MATCHED_TARGET_GAIN_SOURCE_ID)\" -DMATCHED_TARGET_GAIN_ADAPTER_SOURCE_ID=\"$(MATCHED_TARGET_GAIN_SOURCE_ID)\"
.PHONY: matched-target-gain print-matched-target-gain-sources test-rpb-matched-target-gain-adapter evaluate-matched-target-gain
matched-target-gain: $(MATCHED_TARGET_GAIN_BIN)
print-matched-target-gain-sources:
	@printf '%s\n' $(MATCHED_TARGET_GAIN_INPUTS)

$(EVALUATION_OBJECT_DIR)/matched_target_gain_main.o: $(EVALUATION_ROOT)/src/matched_target_gain_main.cpp $(MATCHED_TARGET_GAIN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MATCHED_TARGET_GAIN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(MATCHED_TARGET_GAIN_ADAPTER_OBJECT): $(RPB_ROOT)/src/matched_target_gain_adapter.cpp $(MATCHED_TARGET_GAIN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MATCHED_TARGET_GAIN_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(MATCHED_TARGET_GAIN_BIN): $(EVALUATION_OBJECT_DIR)/matched_target_gain_main.o $(MATCHED_TARGET_GAIN_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/matched_target_gain_adapter_test.o: $(RPB_ROOT)/tests/matched_target_gain_adapter_test.cpp $(MATCHED_TARGET_GAIN_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(MATCHED_TARGET_GAIN_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/matched_target_gain_adapter_test: $(RPB_TEST_DIR)/matched_target_gain_adapter_test.o $(MATCHED_TARGET_GAIN_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-matched-target-gain-adapter: $(RPB_TEST_DIR)/matched_target_gain_adapter_test
	$(RPB_TEST_DIR)/matched_target_gain_adapter_test

evaluate-matched-target-gain: $(MATCHED_TARGET_GAIN_BIN)
	@MATCHED_TARGET_GAIN_BIN="$(abspath $(MATCHED_TARGET_GAIN_BIN))" MATCHED_TARGET_GAIN_SOURCE_INPUTS="$(MATCHED_TARGET_GAIN_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-matched-target-gain.sh

# Fixed pooled-context architecture comparison. Every linked legacy/admission
# scope is included, while the new adapter keeps its own enclosing source ID.
POOLED_CONTEXT_BIN := $(BUILD_DIR)/embedding_pooled_context
POOLED_CONTEXT_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/pooled_context_adapter.o
POOLED_CONTEXT_INPUTS := $(sort $(MATCHED_TARGET_GAIN_INPUTS) $(SDK_PROVENANCE_INPUTS) $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/pooled_context_adapter.h $(RPB_ROOT)/src/pooled_context_adapter.cpp $(RPB_ROOT)/tests/pooled_context_adapter_test.cpp $(RPB_ROOT)/tests/pooled_context_model_test.cpp $(RPB_ROOT)/tests/pooled_context_old_model_reference.h $(EVALUATION_ROOT)/src/pooled_context_main.cpp $(EVALUATION_ROOT)/cards/pooled_context_v1.md $(CODE_ROOT)/scripts/prepare-pooled-context.py $(CODE_ROOT)/scripts/check-pooled-context.sh $(CODE_ROOT)/scripts/evaluate-pooled-context.sh)
POOLED_CONTEXT_SOURCE_ID := $(shell sha256sum $(POOLED_CONTEXT_INPUTS) | sha256sum | cut -d ' ' -f 1)
POOLED_CONTEXT_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(POOLED_CONTEXT_SOURCE_ID)\" -DPOOLED_CONTEXT_ADAPTER_SOURCE_ID=\"$(POOLED_CONTEXT_SOURCE_ID)\"
.PHONY: pooled-context print-pooled-context-sources test-rpb-pooled-context-model test-rpb-pooled-context-adapter evaluate-pooled-context
pooled-context: $(POOLED_CONTEXT_BIN)
print-pooled-context-sources:
	@printf '%s\n' $(POOLED_CONTEXT_INPUTS)

$(EVALUATION_OBJECT_DIR)/pooled_context_main.o: $(EVALUATION_ROOT)/src/pooled_context_main.cpp $(POOLED_CONTEXT_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(POOLED_CONTEXT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(POOLED_CONTEXT_ADAPTER_OBJECT): $(RPB_ROOT)/src/pooled_context_adapter.cpp $(POOLED_CONTEXT_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(POOLED_CONTEXT_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(POOLED_CONTEXT_BIN): $(EVALUATION_OBJECT_DIR)/pooled_context_main.o $(POOLED_CONTEXT_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/pooled_context_model_test.o: $(RPB_ROOT)/tests/pooled_context_model_test.cpp $(POOLED_CONTEXT_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(RPB_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/pooled_context_model_test: $(RPB_TEST_DIR)/pooled_context_model_test.o $(RPB_WORKFLOW_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-pooled-context-model: $(RPB_TEST_DIR)/pooled_context_model_test
	$(RPB_TEST_DIR)/pooled_context_model_test

$(RPB_TEST_DIR)/pooled_context_adapter_test.o: $(RPB_ROOT)/tests/pooled_context_adapter_test.cpp $(POOLED_CONTEXT_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(POOLED_CONTEXT_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/pooled_context_adapter_test: $(RPB_TEST_DIR)/pooled_context_adapter_test.o $(POOLED_CONTEXT_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-pooled-context-adapter: $(RPB_TEST_DIR)/pooled_context_adapter_test
	$(RPB_TEST_DIR)/pooled_context_adapter_test

evaluate-pooled-context: $(POOLED_CONTEXT_BIN)
	@POOLED_CONTEXT_BIN="$(abspath $(POOLED_CONTEXT_BIN))" POOLED_CONTEXT_SOURCE_INPUTS="$(POOLED_CONTEXT_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-pooled-context.sh

# A fresh unchanged early/late confirmation delegates the existing reliability
# adapter under its truthful implementation namespace. The new wrapper/source
# identity binds the external cohort and fifth checkpoint companion separately.
EARLY_MIXER_CONFIRMATION_BIN := $(BUILD_DIR)/embedding_early_mixer_confirmation
EARLY_MIXER_CONFIRMATION_ADAPTER_OBJECT := $(RPB_OBJECT_DIR)/early_mixer_confirmation_adapter.o
EARLY_MIXER_CONFIRMATION_INPUTS := $(sort $(POOLED_CONTEXT_INPUTS) $(SDK_PROVENANCE_INPUTS) $(RPB_ROOT)/include/embedding/encoders/raw_patch_bottleneck_mae/early_mixer_confirmation_adapter.h $(RPB_ROOT)/src/early_mixer_confirmation_adapter.cpp $(RPB_ROOT)/tests/early_mixer_confirmation_adapter_test.cpp $(EVALUATION_ROOT)/src/early_mixer_confirmation_main.cpp $(EVALUATION_ROOT)/cards/early_mixer_confirmation_v1.md $(CODE_ROOT)/scripts/prepare-early-mixer-confirmation.py $(CODE_ROOT)/scripts/check-early-mixer-confirmation.sh $(CODE_ROOT)/scripts/evaluate-early-mixer-confirmation.sh)
EARLY_MIXER_CONFIRMATION_SOURCE_ID := $(shell sha256sum $(EARLY_MIXER_CONFIRMATION_INPUTS) | sha256sum | cut -d ' ' -f 1)
EARLY_MIXER_CONFIRMATION_CPPFLAGS := $(COMMON_CPPFLAGS) -I$(RPB_ROOT)/include -I$(EVALUATION_ROOT)/include -DEVALUATION_SOURCE_ID=\"$(EARLY_MIXER_CONFIRMATION_SOURCE_ID)\" -DEARLY_MIXER_CONFIRMATION_ADAPTER_SOURCE_ID=\"$(EARLY_MIXER_CONFIRMATION_SOURCE_ID)\"
.PHONY: early-mixer-confirmation print-early-mixer-confirmation-sources test-rpb-early-mixer-confirmation-adapter evaluate-early-mixer-confirmation
early-mixer-confirmation: $(EARLY_MIXER_CONFIRMATION_BIN)
print-early-mixer-confirmation-sources:
	@printf '%s\n' $(EARLY_MIXER_CONFIRMATION_INPUTS)

$(EVALUATION_OBJECT_DIR)/early_mixer_confirmation_main.o: $(EVALUATION_ROOT)/src/early_mixer_confirmation_main.cpp $(EARLY_MIXER_CONFIRMATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_CONFIRMATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EARLY_MIXER_CONFIRMATION_ADAPTER_OBJECT): $(RPB_ROOT)/src/early_mixer_confirmation_adapter.cpp $(EARLY_MIXER_CONFIRMATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_CONFIRMATION_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(EARLY_MIXER_CONFIRMATION_BIN): $(EVALUATION_OBJECT_DIR)/early_mixer_confirmation_main.o $(EARLY_MIXER_CONFIRMATION_ADAPTER_OBJECT) $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(PAIRED_POOLING_OBJECT) $(NATIVE_CURVE_OBJECT) $(ARCHIVE_READOUT_OBJECT) $(EVALUATION_COMMON_OBJECTS) $(FIXED_FEATURE_READOUTS_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(RPB_TEST_DIR)/early_mixer_confirmation_adapter_test.o: $(RPB_ROOT)/tests/early_mixer_confirmation_adapter_test.cpp $(EARLY_MIXER_CONFIRMATION_INPUTS)
	mkdir -p "$(@D)"
	$(CXX) $(EARLY_MIXER_CONFIRMATION_CPPFLAGS) -I$(RPB_ROOT)/tests $(CXXFLAGS) -c $< -o $@

$(RPB_TEST_DIR)/early_mixer_confirmation_adapter_test: $(RPB_TEST_DIR)/early_mixer_confirmation_adapter_test.o $(EARLY_MIXER_CONFIRMATION_ADAPTER_OBJECT) $(EARLY_MIXER_ADAPTER_OBJECT) $(RPB_TRAINING_SOURCE_GAIN_OBJECT) $(RPB_CURVE_ADAPTER_OBJECT) $(RPB_ADAPTER_OBJECT) $(RPB_WORKFLOW_OBJECT) $(HARNESS_OBJECT) $(OBJECT_DIR)/shared/data.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

test-rpb-early-mixer-confirmation-adapter: $(RPB_TEST_DIR)/early_mixer_confirmation_adapter_test
	$(RPB_TEST_DIR)/early_mixer_confirmation_adapter_test

evaluate-early-mixer-confirmation: $(EARLY_MIXER_CONFIRMATION_BIN)
	@EARLY_MIXER_CONFIRMATION_BIN="$(abspath $(EARLY_MIXER_CONFIRMATION_BIN))" EARLY_MIXER_CONFIRMATION_SOURCE_INPUTS="$(EARLY_MIXER_CONFIRMATION_INPUTS)" bash $(CODE_ROOT)/scripts/evaluate-early-mixer-confirmation.sh

# Portable configurable encoder; independent of market ingestion and evaluation.
include $(RPB_ROOT)/configurable_spectral.mk

# New dimension-aware synthetic capacity experiment; separate CUDA producer and CPU reader.
include $(EVALUATION_ROOT)/protocols/configurable_capacity_v1/configurable_capacity.mk

# Saved-coordinate usefulness diagnostic; no encoder construction or execution.
include $(EVALUATION_ROOT)/protocols/configurable_branch_v1/configurable_branch.mk
