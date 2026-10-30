# Saved-vector exact search uses only the C++ standard library.
RETRIEVAL_ROOT := code/operations/retrieval
RETRIEVAL_BUILD_DIR := $(if $(strip $(BUILD_DIR)),$(BUILD_DIR),/opt/mityane-geometrics/embeddings/build/baseline)/retrieval
RETRIEVAL_CPPFLAGS := -I$(RETRIEVAL_ROOT)/include
RETRIEVAL_CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -Wpedantic -MMD -MP
RETRIEVAL_OBJECT := $(RETRIEVAL_BUILD_DIR)/exact_search.o
RETRIEVAL_TEST_OBJECT := $(RETRIEVAL_BUILD_DIR)/exact_search_test.o
RETRIEVAL_TEST := $(RETRIEVAL_BUILD_DIR)/exact_search_test
RETRIEVAL_HEADER := $(RETRIEVAL_ROOT)/include/mityane/operations/retrieval/exact_search.h
RETRIEVAL_TOOL_OBJECT := $(RETRIEVAL_BUILD_DIR)/exact_search_tool.o
RETRIEVAL_TOOL_BIN := $(RETRIEVAL_BUILD_DIR)/exact_search_tool

.PHONY: test-retrieval build-retrieval-tool test-retrieval-tool retrieval-container-check

build-retrieval-tool: $(RETRIEVAL_TOOL_BIN)
test-retrieval-tool: $(RETRIEVAL_TOOL_BIN) | retrieval-container-check
	MITYANE_RETRIEVAL_TOOL_BINARY="$(abspath $(RETRIEVAL_TOOL_BIN))" bash code/scripts/node.sh --test $(RETRIEVAL_ROOT)/tool/tool.test.mjs

build-operation-tools: build-retrieval-tool
test-operation-tools: test-retrieval-tool

test-retrieval: $(RETRIEVAL_TEST)
	$(RETRIEVAL_TEST)

retrieval-container-check:
	@test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }

$(RETRIEVAL_OBJECT): $(RETRIEVAL_ROOT)/src/exact_search.cpp $(RETRIEVAL_HEADER) $(RETRIEVAL_ROOT)/retrieval.mk | retrieval-container-check
	mkdir -p "$(@D)"
	$(CXX) $(RETRIEVAL_CPPFLAGS) $(RETRIEVAL_CXXFLAGS) -c $< -o $@

$(RETRIEVAL_TEST_OBJECT): $(RETRIEVAL_ROOT)/tests/exact_search_test.cpp $(RETRIEVAL_HEADER) $(RETRIEVAL_ROOT)/retrieval.mk | retrieval-container-check
	mkdir -p "$(@D)"
	$(CXX) $(RETRIEVAL_CPPFLAGS) $(RETRIEVAL_CXXFLAGS) -c $< -o $@

$(RETRIEVAL_TEST): $(RETRIEVAL_OBJECT) $(RETRIEVAL_TEST_OBJECT)
	$(CXX) $^ -o $@

$(RETRIEVAL_TOOL_OBJECT): $(RETRIEVAL_ROOT)/tool/native.cpp $(RETRIEVAL_HEADER) $(RETRIEVAL_ROOT)/retrieval.mk | retrieval-container-check
	mkdir -p "$(@D)"
	$(CXX) $(RETRIEVAL_CPPFLAGS) $(RETRIEVAL_CXXFLAGS) -c $< -o $@

$(RETRIEVAL_TOOL_BIN): $(RETRIEVAL_OBJECT) $(RETRIEVAL_TOOL_OBJECT)
	$(CXX) $^ -o $@

-include $(RETRIEVAL_OBJECT).d $(RETRIEVAL_TEST_OBJECT).d $(RETRIEVAL_TOOL_OBJECT).d
