# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
# Invoke task.sh SESSION TARGET in the managed container.
ifndef DIRECT_CONTROL_V1_INCLUDED
DIRECT_CONTROL_V1_INCLUDED := 1
ifndef LD_EVAL_V1_OBJECT
include code/operations/forecasting/evaluation/versions/v1/baselines.mk
endif
DIRECT_CONTROL_V1_ROOT := code/operations/forecasting/direct_embedding_control/versions/v1
DIRECT_CONTROL_V1_SOURCE_FILES := \
  $(DIRECT_CONTROL_V1_ROOT)/include/cuwacunu/networks/direct_embedding_control/v1/control.h \
  $(DIRECT_CONTROL_V1_ROOT)/include/cuwacunu/networks/direct_embedding_control/v1/artifact.h \
  code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/sidecar.h \
  code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/tensor_state.h \
  code/operations/forecasting/evaluation/versions/v1/include/cuwacunu/evaluation/latent_dynamics/v1/baselines.h \
  code/operations/forecasting/evaluation/versions/v1/src/baselines.cpp
DIRECT_CONTROL_V1_SOURCE_SHA256 := $(shell bash code/scripts/node.sh $(DIRECT_CONTROL_V1_ROOT)/source_identity.mjs --root .)
ifeq ($(DIRECT_CONTROL_V1_SOURCE_SHA256),)
$(error Cannot establish the compiled direct embedding control source identity)
endif
DIRECT_CONTROL_V1_CPPFLAGS := $(CPPFLAGS) -I$(DIRECT_CONTROL_V1_ROOT)/include \
  -Icode/operations/forecasting/evaluation/versions/v1/include -Icode/shared/artifacts/versions/v1/include \
  -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -DCUWACUNU_DIRECT_CONTROL_V1_SOURCE_SHA256=\"$(DIRECT_CONTROL_V1_SOURCE_SHA256)\"
DIRECT_CONTROL_V1_TEST := $(BUILD_DIR)/direct_embedding_control_v1_test
DIRECT_CONTROL_V1_ARTIFACT_TEST := $(BUILD_DIR)/direct_embedding_control_v1_artifact_test
.PHONY: build-direct-control-v1 test-direct-control-v1 test-direct-control-artifact
build-direct-control-v1: $(DIRECT_CONTROL_V1_TEST) $(DIRECT_CONTROL_V1_ARTIFACT_TEST)
test-direct-control-v1: $(DIRECT_CONTROL_V1_TEST)
	$(DIRECT_CONTROL_V1_TEST) "$(BUILD_DIR)/tmp/direct-control-$$(date +%s%N)"
test-direct-control-artifact: $(DIRECT_CONTROL_V1_ARTIFACT_TEST)
	$(DIRECT_CONTROL_V1_ARTIFACT_TEST)
$(BUILD_DIR)/direct_embedding_control_v1_test.o: $(DIRECT_CONTROL_V1_ROOT)/tests/control_test.cpp $(DIRECT_CONTROL_V1_SOURCE_FILES) $(DIRECT_CONTROL_V1_ROOT)/source_identity.mjs code/scripts/node.sh
	mkdir -p "$(@D)"
	$(CXX) $(DIRECT_CONTROL_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/direct_embedding_control_v1_artifact_test.o: $(DIRECT_CONTROL_V1_ROOT)/tests/artifact_test.cpp $(DIRECT_CONTROL_V1_SOURCE_FILES) $(DIRECT_CONTROL_V1_ROOT)/source_identity.mjs code/scripts/node.sh
	mkdir -p "$(@D)"
	$(CXX) $(DIRECT_CONTROL_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(DIRECT_CONTROL_V1_TEST): $(BUILD_DIR)/direct_embedding_control_v1_test.o $(LD_EVAL_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(DIRECT_CONTROL_V1_ARTIFACT_TEST): $(BUILD_DIR)/direct_embedding_control_v1_artifact_test.o $(LD_EVAL_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/direct_embedding_control_v1_test.d $(BUILD_DIR)/direct_embedding_control_v1_artifact_test.d
endif
