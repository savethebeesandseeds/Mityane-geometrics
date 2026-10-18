# Separate configurable RPB-v20 API and engineering checks. Historical CLI,
# default configuration and measured protocol targets remain unchanged.
V20_TEST_DIR := $(RPB_TEST_DIR)/configurable_spectral
V20_CPPFLAGS := -I$(RPB_ROOT)/include $(COMMON_CPPFLAGS)
V20_CONFIG_TEST := $(V20_TEST_DIR)/config_document_test
V20_MODEL_TEST := $(V20_TEST_DIR)/configurable_spectral_test
V20_CHECKPOINT_TEST := $(V20_TEST_DIR)/configurable_checkpoint_test
V20_INTEGRITY_TEST := $(V20_TEST_DIR)/checkpoint_integrity_test

.PHONY: rpb-v20 test-rpb-v20
rpb-v20: $(V20_CONFIG_TEST) $(V20_MODEL_TEST) $(V20_CHECKPOINT_TEST) $(V20_INTEGRITY_TEST)
test-rpb-v20: rpb-v20
	$(V20_CONFIG_TEST)
	$(V20_MODEL_TEST)
	$(V20_CHECKPOINT_TEST) $(RPB_ROOT)/config/configurable_spectral.conf
	$(V20_INTEGRITY_TEST)

$(V20_TEST_DIR)/config_document_test.o: $(CODE_ROOT)/shared/tests/config_document_test.cpp
	mkdir -p "$(@D)"
	$(CXX) -I$(CODE_ROOT)/shared/include $(CXXFLAGS) -c $< -o $@
$(V20_TEST_DIR)/configurable_spectral_test.o: $(RPB_ROOT)/tests/configurable_spectral_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(V20_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V20_TEST_DIR)/configurable_checkpoint_test.o: $(RPB_ROOT)/tests/configurable_checkpoint_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(V20_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V20_TEST_DIR)/checkpoint_integrity_test.o: $(RPB_ROOT)/tests/checkpoint_integrity_test.cpp
	mkdir -p "$(@D)"
	$(CXX) $(V20_CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(V20_CONFIG_TEST): $(V20_TEST_DIR)/config_document_test.o
	$(CXX) $^ -o $@
$(V20_MODEL_TEST): $(V20_TEST_DIR)/configurable_spectral_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(V20_CHECKPOINT_TEST): $(V20_TEST_DIR)/configurable_checkpoint_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(V20_INTEGRITY_TEST): $(V20_TEST_DIR)/checkpoint_integrity_test.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
-include $(wildcard $(V20_TEST_DIR)/*.d)
