#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mityane::operations::counterfactuals::v1 {

struct ArtifactRef {
  std::string id;
  std::string sha256;
  bool operator==(const ArtifactRef&) const = default;
};

// Pins and evidence are caller-supplied. This library does not read artifacts,
// authenticate a provider, or certify the truth of upstream attestations.
struct Dependency {
  std::string role;
  ArtifactRef artifact;
  std::string fit_population;
  std::string selection_population;
  std::string pretraining_exposure; // known, unknown, or excluded_by_design
  std::optional<std::int64_t> available_at;
};

struct SpaceRef {
  ArtifactRef artifact;
  ArtifactRef provider;
  ArtifactRef preprocessing;
  ArtifactRef fitted_transform;
  std::string output_selector;
  std::vector<std::size_t> native_shape;
  std::string layout;
  std::string normalization;
  std::string geometry;
  std::string input_role;
  bool operator==(const SpaceRef&) const = default;
};

struct Axis {
  std::string id;
  std::string definition_revision;
  std::string units;
  bool operator==(const Axis&) const = default;
};

struct PipelineRef {
  ArtifactRef artifact;
  ArtifactRef response_model;
  std::string configuration_sha256;
  std::string stochastic_policy; // reference slice supports deterministic only
  bool provider_revision_known = false;
  std::vector<Dependency> dependencies;
};

enum class ClaimLevel { unspecified, controlled_model_response,
                        observational_contrast, identified_intervention_effect };
enum class TemporalMode { unspecified, non_temporal, retrospective, as_of };
enum class RepresentationKind { unspecified, observation_embedding, task_state,
                                predicted_state, transformed_candidate, generated_content };
enum class Origin { unspecified, observed, edited_input, simulated_input, latent_candidate };
enum class AttributeRole { unspecified, changed, held_fixed, coupled };
enum class CheckState { unspecified, passed, rejected };
enum class ResultState { success, abstained, failed, unresolved };
enum class ReasonCode {
  unsupported_claim, invalid_record, invalid_pairing, incompatible_role_or_space,
  stale_or_changed_pipeline, unverified_control, temporal_violation,
  upstream_failure, nonfinite_response_or_contrast, invalid_alternative,
  outside_declared_support, unsupported_interpretation, interrupted
};

struct AttributeRule {
  std::string name;
  std::string units;
  AttributeRole role = AttributeRole::unspecified;
  double minimum = 0;
  double maximum = 0;
};
struct ScalarValue {
  std::string name;
  std::string units;
  double value = 0;
};
struct RecordedCheck {
  CheckState state = CheckState::unspecified;
  ArtifactRef evidence;
  std::string detail;
};
struct TemporalSupport {
  std::optional<std::int64_t> support_start;
  std::optional<std::int64_t> support_end;
  std::optional<std::int64_t> available_at;
  std::string vintage;
};

struct Branch {
  ArtifactRef input;
  ArtifactRef source_observation;
  std::string parent_input_id;
  Origin origin = Origin::unspecified;
  ArtifactRef representation;
  RepresentationKind representation_kind = RepresentationKind::unspecified;
  std::string parent_representation_id;
  SpaceRef space;
  ArtifactRef pipeline;
  ArtifactRef response;
  std::string response_execution_id;
  std::int64_t output_position = 0;
  ArtifactRef mask;
  bool output_valid = false;
  bool upstream_succeeded = false;
  TemporalSupport timing;
  std::vector<ScalarValue> attributes;
  std::vector<Axis> axes;
  std::vector<double> response_values;
};

struct Pair {
  std::string id;
  std::string unit_id;
  std::string entity_id;
  std::string group_id;
  std::optional<Branch> baseline;
  std::optional<Branch> alternative;
  ArtifactRef constructor;
  std::string change_definition;
  std::string interpretation; // model_input_response or latent_coordinate_response
  ArtifactRef control_evidence;
  RecordedCheck alternative_validity;
  RecordedCheck support;
  RecordedCheck interpretability;
};

struct Request {
  std::string request_id;
  std::string operation_run_id;
  std::string contract_id;
  std::string contract_revision;
  std::string rer_id;
  std::string rer_revision;
  ClaimLevel claim_level = ClaimLevel::unspecified;
  std::string purpose;
  std::string population_id;
  std::string sampling_rule;
  std::string response_horizon;
  std::string permitted_interpretation;
  std::string permissions_ref;
  std::string resource_budget;
  std::string evidence_destination;
  std::string code_sha256;
  std::string configuration_sha256;
  TemporalMode temporal_mode = TemporalMode::unspecified;
  std::optional<std::int64_t> as_of;
  std::string non_temporal_justification;
  SpaceRef expected_space;
  PipelineRef expected_pipeline;
  std::vector<Axis> response_axes;
  std::int64_t selected_output_position = -1; // adapter-defined selector, explicitly bound
  std::vector<AttributeRule> attribute_rules;
  double control_tolerance = -1;
  std::vector<Pair> pairs; // complete, ordered admitted roster
};

struct Diagnostic {
  ReasonCode code;
  std::string record_id;
  std::string message;
};
struct Contrast {
  std::vector<double> baseline;
  std::vector<double> alternative;
  std::vector<double> delta;
};
struct PairResult {
  std::size_t pair_index = 0;
  std::string pair_id;
  ResultState state = ResultState::failed;
  std::vector<Diagnostic> diagnostics;
  std::optional<Contrast> contrast;
  std::optional<std::int64_t> available_at; // unknown if any dependency is unknown
};
struct Summary {
  std::size_t requested = 0;
  std::size_t succeeded = 0;
  std::size_t abstained = 0;
  std::size_t failed = 0;
  std::size_t unresolved = 0;
  std::optional<double> success_rate; // denominator is the full admitted roster
};
struct RunResult {
  // Retaining the exact typed input preserves source/parent/kind/axes, fitted
  // dependency, code/configuration and evidence identities even on refusal.
  Request input;
  bool admitted = false;
  std::vector<Diagnostic> admission_diagnostics;
  std::vector<PairResult> pairs;
  Summary summary;
};

// Validate and compare recorded deterministic responses. stop_after leaves all
// remaining admitted roster entries unresolved; it never truncates the roster.
RunResult compare(const Request& request,
                  std::optional<std::size_t> stop_after = std::nullopt);
std::string to_string(ResultState state);
std::string to_string(ReasonCode code);

} // namespace mityane::operations::counterfactuals::v1
