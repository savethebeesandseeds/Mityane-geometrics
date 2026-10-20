# The paired-response reference uses only the C++ standard library.
COUNTERFACTUALS_ROOT := code/operations/counterfactuals
COUNTERFACTUALS_BUILD_DIR := $(if $(strip $(BUILD_DIR)),$(BUILD_DIR),/opt/mityane-geometrics/embeddings/build/baseline)/counterfactuals
COUNTERFACTUALS_CPPFLAGS := -I$(COUNTERFACTUALS_ROOT)/include
COUNTERFACTUALS_CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -Wpedantic -MMD -MP
COUNTERFACTUALS_OBJECT := $(COUNTERFACTUALS_BUILD_DIR)/paired_response.o
COUNTERFACTUALS_TEST_OBJECT := $(COUNTERFACTUALS_BUILD_DIR)/paired_response_test.o
COUNTERFACTUALS_TEST := $(COUNTERFACTUALS_BUILD_DIR)/paired_response_test
COUNTERFACTUALS_HEADER := $(COUNTERFACTUALS_ROOT)/include/mityane/operations/counterfactuals/paired_response.h
COUNTERFACTUALS_EVALUATION_ROOT := $(COUNTERFACTUALS_ROOT)/evaluation/synthetic_response_v1
COUNTERFACTUALS_EVALUATION_CPPFLAGS := $(COUNTERFACTUALS_CPPFLAGS) -Icode/shared/artifacts/versions/v1/include
COUNTERFACTUALS_EVALUATION_OBJECT := $(COUNTERFACTUALS_BUILD_DIR)/synthetic_response.o
COUNTERFACTUALS_EVALUATION_BIN := $(COUNTERFACTUALS_BUILD_DIR)/synthetic_response
COUNTERFACTUALS_ARTIFACT_HEADER := code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/sidecar.h
COUNTERFACTUALS_TOOL_OBJECT := $(COUNTERFACTUALS_BUILD_DIR)/paired_response_tool.o
COUNTERFACTUALS_TOOL_BIN := $(COUNTERFACTUALS_BUILD_DIR)/paired_response_tool

.PHONY: test-counterfactuals counterfactuals-container-check build-counterfactuals-evaluation evaluate-counterfactuals build-counterfactuals-tool build-operation-tools test-operation-tools
build-operation-tools: build-counterfactuals-tool

build-counterfactuals-tool: $(COUNTERFACTUALS_TOOL_BIN) | counterfactuals-container-check

test-operation-tools: $(COUNTERFACTUALS_TOOL_BIN) | counterfactuals-container-check
	MITYANE_COUNTERFACTUALS_TOOL_BINARY="$(abspath $(COUNTERFACTUALS_TOOL_BIN))" bash code/scripts/node.sh --test $(COUNTERFACTUALS_ROOT)/tool/tool.test.mjs
test-counterfactuals: $(COUNTERFACTUALS_TEST)
	$(COUNTERFACTUALS_TEST)

build-counterfactuals-evaluation: $(COUNTERFACTUALS_EVALUATION_BIN) | counterfactuals-container-check

# Each evaluation requires a named, explicit new evidence destination. The
# operation-owned runner enforces study admission and exclusive publication.
evaluate-counterfactuals: $(COUNTERFACTUALS_EVALUATION_BIN) | counterfactuals-container-check
	@test -n "$(strip $(EVALUATION_ATTEMPT))" || { echo 'Set EVALUATION_ATTEMPT to an explicit absolute /mityane/.temp/operations/counterfactuals/ attempt path.' >&2; exit 1; }
	@case "$(EVALUATION_ATTEMPT)" in /mityane/.temp/operations/counterfactuals/*) ;; *) echo 'EVALUATION_ATTEMPT must be inside /mityane/.temp/operations/counterfactuals/.' >&2; exit 1 ;; esac
	bash code/scripts/node.sh $(COUNTERFACTUALS_EVALUATION_ROOT)/run.mjs --binary "$(abspath $(COUNTERFACTUALS_EVALUATION_BIN))" --attempt "$(EVALUATION_ATTEMPT)"

counterfactuals-container-check:
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }

$(COUNTERFACTUALS_OBJECT): $(COUNTERFACTUALS_ROOT)/src/paired_response.cpp $(COUNTERFACTUALS_HEADER) $(COUNTERFACTUALS_ROOT)/counterfactuals.mk | counterfactuals-container-check
	mkdir -p "$(@D)"
	$(CXX) $(COUNTERFACTUALS_CPPFLAGS) $(COUNTERFACTUALS_CXXFLAGS) -c $< -o $@

$(COUNTERFACTUALS_TEST_OBJECT): $(COUNTERFACTUALS_ROOT)/tests/paired_response_test.cpp $(COUNTERFACTUALS_HEADER) $(COUNTERFACTUALS_ROOT)/counterfactuals.mk | counterfactuals-container-check
	mkdir -p "$(@D)"
	$(CXX) $(COUNTERFACTUALS_CPPFLAGS) $(COUNTERFACTUALS_CXXFLAGS) -c $< -o $@

$(COUNTERFACTUALS_TEST): $(COUNTERFACTUALS_OBJECT) $(COUNTERFACTUALS_TEST_OBJECT)
	$(CXX) $^ -o $@

$(COUNTERFACTUALS_EVALUATION_OBJECT): $(COUNTERFACTUALS_EVALUATION_ROOT)/synthetic_response.cpp $(COUNTERFACTUALS_HEADER) $(COUNTERFACTUALS_ARTIFACT_HEADER) $(COUNTERFACTUALS_ROOT)/counterfactuals.mk | counterfactuals-container-check
	mkdir -p "$(@D)"
	$(CXX) $(COUNTERFACTUALS_EVALUATION_CPPFLAGS) $(COUNTERFACTUALS_CXXFLAGS) -c $< -o $@

$(COUNTERFACTUALS_EVALUATION_BIN): $(COUNTERFACTUALS_OBJECT) $(COUNTERFACTUALS_EVALUATION_OBJECT)
	$(CXX) $^ -o $@

$(COUNTERFACTUALS_TOOL_OBJECT): $(COUNTERFACTUALS_ROOT)/tool/native.cpp $(COUNTERFACTUALS_HEADER) $(COUNTERFACTUALS_ROOT)/counterfactuals.mk | counterfactuals-container-check
	mkdir -p "$(@D)"
	$(CXX) $(COUNTERFACTUALS_CPPFLAGS) $(COUNTERFACTUALS_CXXFLAGS) -c $< -o $@

$(COUNTERFACTUALS_TOOL_BIN): $(COUNTERFACTUALS_OBJECT) $(COUNTERFACTUALS_TOOL_OBJECT)
	$(CXX) $^ -o $@

-include $(COUNTERFACTUALS_OBJECT:.o=.d) $(COUNTERFACTUALS_TEST_OBJECT:.o=.d) $(COUNTERFACTUALS_EVALUATION_OBJECT:.o=.d) $(COUNTERFACTUALS_TOOL_OBJECT:.o=.d)
