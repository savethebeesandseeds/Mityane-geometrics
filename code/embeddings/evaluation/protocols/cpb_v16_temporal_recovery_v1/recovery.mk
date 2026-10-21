# Copyright (c) 2026 Waajacu. See LICENSE.
# Extension of the authoritative root Makefile; no environment or lifecycle.
CPB_V16_RECOVERY_ROOT := code/embeddings/evaluation/protocols/cpb_v16_temporal_recovery_v1
.PHONY: build-cpb-v16-temporal-recovery test-cpb-v16-temporal-recovery
build-cpb-v16-temporal-recovery: $(BUILD_DIR)/cpb_v16_temporal_recovery_candidate_train
test-cpb-v16-temporal-recovery:
	bash code/scripts/node.sh --test $(CPB_V16_RECOVERY_ROOT)/reconcile.test.mjs
$(BUILD_DIR)/cpb_v16_temporal_recovery_candidate_train.o: $(CPB_V16_RECOVERY_ROOT)/recovery_train_main.cpp
	$(CPB_V16_TEMPORAL_GUARD)
	mkdir -p "$(@D)"
	$(CXX) $(CPB_V16_TEMPORAL_CPPFLAGS) -DCPB_OBJECTIVE_CANDIDATE=1 $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cpb_v16_temporal_recovery_candidate_train: $(BUILD_DIR)/cpb_v16_temporal_recovery_candidate_train.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(BUILD_DIR)/cpb_v16_temporal_recovery_candidate_train.d
