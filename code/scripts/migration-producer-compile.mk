# Compile-only regression for producer translation units touched by the language
# migration. Included after the authoritative root rules: no copied flags, dummy
# provenance identities, linking, model execution or saved-artifact reads.
MIGRATION_PRODUCER_ROOT_NAMES := context_balanced_validation_main context_deletion_main \
  context_lighter_validation_main context_optimization_validation_main \
  early_mixer_confirmation_main early_mixer_learning_curve_main \
  frozen_amplitude_transfer_main matched_target_gain_main \
  native_view_agreement_validation_main optimization_diagnostic_main \
  paired_pooling_main pooled_context_main saved_feature_reliability_main \
  training_objective_diagnostic_main v7_decoder_calibration_main
MIGRATION_PRODUCER_ROOT_OBJECTS := $(addprefix $(EVALUATION_OBJECT_DIR)/,$(addsuffix .o,$(MIGRATION_PRODUCER_ROOT_NAMES)))
MIGRATION_PRODUCER_CONFIG_OBJECTS := $(foreach family,configurable_learned_relation_v1 configurable_content_relation_v1,$(addprefix $(OBJECT_DIR)/protocols/$(family)/,producer_main.o reader_main.o))
MIGRATION_PRODUCER_FAMILY_FILES := $(addprefix code/embeddings/evaluation/protocols/,$(addsuffix /Makefile,\
  visible_difference_v1 structured_hard_timing_v1 architecture_screen_v1 \
  fixed_prior_relation_screen_v1 partitioned_relation_screen_v1 unit_relation_screen_v1 \
  fixed_prior_fresh_confirmation_v1 two_component_information_v1 multiband_screen_v1 spectral_confirmation_v1))
MIGRATION_PRODUCER_FAMILY_OBJECTS := $(addprefix $(OBJECT_DIR)/protocols/,\
  visible_difference_v1/visible_difference_main.o structured_hard_timing_v1/paired_timing_run.o \
  architecture_screen_v1/architecture_screen_main.o \
  fixed_prior_relation_screen_v1/fixed_prior_relation_screen_main.o \
  partitioned_relation_screen_v1/partitioned_relation_screen_main.o \
  unit_relation_screen_v1/unit_relation_screen_main.o \
  fixed_prior_fresh_confirmation_v1/fresh_confirmation_main.o \
  fixed_prior_fresh_confirmation_v1/write_fresh_tempo3_data.o \
  two_component_information_v1/information_main.o multiband_screen_v1/producer_main.o \
  spectral_confirmation_v1/producer_main.o)
# A final recheck can recalculate every family macro after input-list changes.
# GNU make -W changes only its dependency assumptions, not source timestamps.
MIGRATION_PRODUCER_REBUILD_FAMILIES ?= 0
MIGRATION_PRODUCER_JOBS ?= 2
MIGRATION_PRODUCER_RECHECK_SOURCES ?=

.PHONY: test-migration-producer-compile
test-migration-producer-compile:
	@set -eu; \
	test -f /.dockerenv && test "$(CURDIR)" = /mityane || { echo 'Use container.ps1 in mityane-geometrics.' >&2; exit 1; }; \
	case '$(MIGRATION_PRODUCER_JOBS)' in 1|2|3|4|5|6) ;; *) echo 'Migration producer compiler jobs must be between 1 and 6.' >&2; exit 1;; esac; \
	build=$$(realpath -m -- '$(BUILD_DIR)'); \
	case "$$build" in /opt/mityane-geometrics/*) ;; *) echo 'Producer compile products must stay under /opt/mityane-geometrics.' >&2; exit 1;; esac; \
	task_tmp=$$(realpath -m -- "$$build/tmp"); \
	case "$$task_tmp" in "$$build"/*) ;; *) echo 'Producer compiler TMPDIR escapes its build directory.' >&2; exit 1;; esac; \
	mkdir -p -- "$$task_tmp"; test "$$(realpath -e -- "$$task_tmp")" = "$$task_tmp"; \
	export TMPDIR="$$task_tmp"; \
	$(MAKE) --no-print-directory -j$(MIGRATION_PRODUCER_JOBS) -f Makefile $(foreach file,$(MIGRATION_PRODUCER_FAMILY_FILES),-f $(file)) \
	  $(if $(filter 1,$(MIGRATION_PRODUCER_REBUILD_FAMILIES)),$(foreach file,$(MIGRATION_PRODUCER_FAMILY_FILES),-W $(file))) \
	  $(foreach source,$(MIGRATION_PRODUCER_RECHECK_SOURCES),-W $(source)) \
	  $(MIGRATION_PRODUCER_ROOT_OBJECTS) $(MIGRATION_PRODUCER_CONFIG_OBJECTS) $(MIGRATION_PRODUCER_FAMILY_OBJECTS); \
	printf 'Migration producer compile checks passed: 30 translation units; no binaries linked or run.\n'
