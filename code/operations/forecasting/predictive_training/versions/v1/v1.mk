# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
STATE_REPRESENTATION_V1_ROOT := code/embeddings/state_representation/versions/v1
LATENT_DYNAMICS_V1_ROOT := code/operations/forecasting/latent_dynamics/versions/v1
PREDICTIVE_TRAINING_V1_ROOT := code/operations/forecasting/predictive_training/versions/v1
PREDICTIVE_V1_CPPFLAGS := -I$(STATE_REPRESENTATION_V1_ROOT)/include -I$(LATENT_DYNAMICS_V1_ROOT)/include -I$(PREDICTIVE_TRAINING_V1_ROOT)/include -isystem $(LIBTORCH)/include -isystem $(LIBTORCH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
PREDICTIVE_V1_TEST := $(BUILD_DIR)/test_predictive_v1
.PHONY: build-predictive-v1-engineering test-predictive-v1
build-predictive-v1-engineering: $(PREDICTIVE_V1_TEST)
test-predictive-v1: build-predictive-v1-engineering
	$(PREDICTIVE_V1_TEST)
$(BUILD_DIR)/predictive_v1_test.o: $(PREDICTIVE_TRAINING_V1_ROOT)/tests/v1_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(PREDICTIVE_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(PREDICTIVE_V1_TEST): $(BUILD_DIR)/predictive_v1_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/predictive_v1_test.d
