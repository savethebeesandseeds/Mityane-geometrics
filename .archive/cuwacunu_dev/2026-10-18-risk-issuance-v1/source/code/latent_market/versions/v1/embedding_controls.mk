# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
EMBEDDING_CONTROL_STUDY := $(BUILD_DIR)/latent_market_v1_embedding_controls
.PHONY: build-embedding-control-study test-embedding-control-study
build-embedding-control-study: $(EMBEDDING_CONTROL_STUDY)
$(BUILD_DIR)/latent_market_v1_embedding_controls.o: code/latent_market/versions/v1/src/embedding_controls_main.cpp $(DIRECT_CONTROL_V1_SOURCE_FILES)
	mkdir -p "$(@D)"
	$(CXX) $(LATENT_MARKET_V1_CPPFLAGS) $(DIRECT_CONTROL_V1_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(EMBEDDING_CONTROL_STUDY): $(BUILD_DIR)/latent_market_v1_embedding_controls.o $(LD_EVAL_V1_OBJECT)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
test-embedding-control-study: build-embedding-control-study $(LATENT_MARKET_V1_FIXTURE)
	python3 -B code/latent_market/versions/v1/tests/embedding_controls_pipeline_test.py "$(BUILD_DIR)"
-include $(BUILD_DIR)/latent_market_v1_embedding_controls.d
