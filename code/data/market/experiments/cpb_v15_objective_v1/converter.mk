# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
CPB_V15_DATA_ROOT := code/data/market/experiments/cpb_v15_objective_v1
CPB_V15_DATA_BUILD := $(BUILD_DIR)/cpb-v15-data
.PHONY: build-cpb-v15-data test-cpb-v15-data
build-cpb-v15-data: $(CPB_V15_DATA_BUILD)/csv_to_cache88
$(CPB_V15_DATA_BUILD)/csv_to_cache88: $(CPB_V15_DATA_ROOT)/csv_to_cache88.cpp $(CPB_V15_DATA_ROOT)/cache88.h
	mkdir -p "$(@D)"
	$(CXX) -std=c++20 -O2 -g0 -Wall -Wextra -Werror -I$(CPB_V15_DATA_ROOT) $< -o $@
$(CPB_V15_DATA_BUILD)/converter_test: $(CPB_V15_DATA_ROOT)/converter_test.cpp $(CPB_V15_DATA_ROOT)/cache88.h
	mkdir -p "$(@D)"
	$(CXX) -std=c++20 -O2 -g0 -Wall -Wextra -Werror -I$(CPB_V15_DATA_ROOT) $< -o $@
test-cpb-v15-data: $(CPB_V15_DATA_BUILD)/converter_test
	$(CPB_V15_DATA_BUILD)/converter_test
	bash code/scripts/node.sh --test $(CPB_V15_DATA_ROOT)/prepare.test.mjs
