# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
# Include after the root Makefile, or invoke task.sh SESSION -f Makefile -f this.mk.
LD_EVAL_V1_ROOT := code/evaluation/latent_dynamics/versions/v1
LD_EVAL_V1_CPPFLAGS := -I$(LD_EVAL_V1_ROOT)/include -isystem $(LIBTORCH)/include \
  -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
LD_EVAL_V1_OBJECT := $(BUILD_DIR)/latent_dynamics_v1_baselines.o
LD_EVAL_V1_TEST_OBJECT := $(BUILD_DIR)/latent_dynamics_v1_baselines_test.o
LD_EVAL_V1_TEST := $(BUILD_DIR)/latent_dynamics_v1_baselines_test
.PHONY: build-latent-dynamics-v1-evaluation test-latent-dynamics-v1-evaluation
build-latent-dynamics-v1-evaluation: $(LD_EVAL_V1_TEST)
test-latent-dynamics-v1-evaluation: $(LD_EVAL_V1_TEST)
	$(LD_EVAL_V1_TEST) "$(BUILD_DIR)/tmp/baselines-$$(date +%s%N)"
$(LD_EVAL_V1_OBJECT): $(LD_EVAL_V1_ROOT)/src/baselines.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LD_EVAL_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(LD_EVAL_V1_TEST_OBJECT): $(LD_EVAL_V1_ROOT)/tests/baselines_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LD_EVAL_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(LD_EVAL_V1_TEST): $(LD_EVAL_V1_TEST_OBJECT) $(LD_EVAL_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(LD_EVAL_V1_OBJECT:.o=.d) $(LD_EVAL_V1_TEST_OBJECT:.o=.d)

LD_RISK_V1_OBJECT := $(BUILD_DIR)/latent_dynamics_v1_risk_readouts.o
LD_RISK_V1_TEST_OBJECT := $(BUILD_DIR)/latent_dynamics_v1_risk_readouts_test.o
LD_RISK_V1_TEST := $(BUILD_DIR)/latent_dynamics_v1_risk_readouts_test
.PHONY: build-latent-risk-v1-readouts test-latent-risk-v1-readouts
build-latent-risk-v1-readouts: $(LD_RISK_V1_TEST)
test-latent-risk-v1-readouts: $(LD_RISK_V1_TEST)
	$(LD_RISK_V1_TEST) "$(BUILD_DIR)/tmp/risk-readouts-$$(date +%s%N)"
$(LD_RISK_V1_OBJECT): $(LD_EVAL_V1_ROOT)/src/risk_readouts.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LD_EVAL_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(LD_RISK_V1_TEST_OBJECT): $(LD_EVAL_V1_ROOT)/tests/risk_readouts_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(LD_EVAL_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(LD_RISK_V1_TEST): $(LD_RISK_V1_TEST_OBJECT) $(LD_RISK_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(LD_RISK_V1_OBJECT:.o=.d) $(LD_RISK_V1_TEST_OBJECT:.o=.d)
