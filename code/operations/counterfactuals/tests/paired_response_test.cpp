#include "mityane/operations/counterfactuals/paired_response.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cf = mityane::operations::counterfactuals::v1;

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

cf::ArtifactRef artifact(const std::string& id) {
  // Deterministic synthetic identities, distinct for distinct fixture artifacts.
  std::uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char value : id) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }
  std::string hex;
  constexpr char digits[] = "0123456789abcdef";
  for (int shift = 60; shift >= 0; shift -= 4)
    hex.push_back(digits[(hash >> shift) & 15]);
  return {id, hex + hex + hex + hex};
}

cf::RecordedCheck passed(const std::string& id) {
  return {cf::CheckState::passed, artifact(id), "Recorded predeclared check passed"};
}

cf::SpaceRef space() {
  return {artifact("space-v1"), artifact("provider-v1"),
          artifact("preprocessing-v1"), artifact("transform-v1"),
          "embedding", {4}, "dense", "none", "euclidean", "response_input"};
}

cf::PipelineRef pipeline() {
  cf::PipelineRef result;
  result.artifact = artifact("pipeline-v1");
  result.response_model = artifact("response-model-v1");
  result.configuration_sha256 = std::string(64, 'b');
  result.stochastic_policy = "deterministic";
  result.provider_revision_known = true;
  for (const auto& role : {"provider", "preprocessing", "fitted_transform",
                           "response_model", "support_policy", "constructor"}) {
    const std::string name(role);
    auto ref = artifact(name + "-v1");
    if (name == "fitted_transform") ref = artifact("transform-v1");
    if (name == "response_model") ref = result.response_model;
    result.dependencies.push_back(
        {name, ref, "training-population-v1", "validation-population-v1", "known", 40});
  }
  return result;
}

cf::Pair pair(const std::string& suffix, const cf::SpaceRef& expected_space,
              const cf::PipelineRef& expected_pipeline,
              const std::vector<cf::Axis>& axes) {
  cf::Branch baseline;
  baseline.input = artifact("baseline-input-" + suffix);
  baseline.source_observation = artifact("source-observation-" + suffix);
  baseline.origin = cf::Origin::observed;
  baseline.representation = artifact("baseline-representation-" + suffix);
  baseline.representation_kind = cf::RepresentationKind::observation_embedding;
  baseline.space = expected_space;
  baseline.pipeline = expected_pipeline.artifact;
  baseline.response = artifact("baseline-response-" + suffix);
  baseline.response_execution_id = "baseline-execution-" + suffix;
  baseline.output_position = 0;
  baseline.mask = artifact("baseline-mask-" + suffix);
  baseline.output_valid = true;
  baseline.upstream_succeeded = true;
  baseline.timing = {10, 20, 30, "source-vintage-v1"};
  baseline.attributes = {{"setting", "unit", 1}, {"load", "unit", 2}};
  baseline.axes = axes;
  baseline.response_values = {2, -1};

  cf::Branch alternative = baseline;
  alternative.input = artifact("alternative-input-" + suffix);
  alternative.parent_input_id = baseline.input.id;
  alternative.origin = cf::Origin::edited_input;
  alternative.representation = artifact("alternative-representation-" + suffix);
  alternative.representation_kind = cf::RepresentationKind::transformed_candidate;
  alternative.parent_representation_id = baseline.representation.id;
  alternative.response = artifact("alternative-response-" + suffix);
  alternative.response_execution_id = "alternative-execution-" + suffix;
  alternative.mask = artifact("alternative-mask-" + suffix);
  alternative.timing.available_at = 35;
  alternative.attributes[0].value = 3;
  alternative.response_values = {5, -3};

  cf::Pair result;
  result.id = "pair-" + suffix;
  result.unit_id = "unit-" + suffix;
  result.entity_id = "entity-" + suffix;
  result.group_id = "group-" + suffix;
  result.baseline = std::move(baseline);
  result.alternative = std::move(alternative);
  result.constructor = artifact("constructor-v1");
  result.change_definition = "Change setting from 1 to 3; hold load fixed at 2";
  result.interpretation = "model_input_response";
  result.control_evidence = artifact("control-evidence-" + suffix);
  result.alternative_validity = passed("validity-evidence-" + suffix);
  result.support = passed("support-evidence-" + suffix);
  result.interpretability = passed("interpretability-evidence-" + suffix);
  return result;
}

cf::Request request() {
  cf::Request result;
  result.request_id = "request-v1";
  result.operation_run_id = "run-v1";
  result.contract_id = "OR06-PCR";
  result.contract_revision = "0.1.0";
  result.rer_id = "OR06-PCR-RER";
  result.rer_revision = "0.1.0";
  result.claim_level = cf::ClaimLevel::controlled_model_response;
  result.purpose = "Validate a fixed model response to a declared input edit";
  result.population_id = "assessment-population-v1";
  result.sampling_rule = "All predeclared eligible pairs in this synthetic fixture";
  result.response_horizon = "one-step";
  result.permitted_interpretation = "model_input_response";
  result.permissions_ref = "synthetic-fixture-permission-v1";
  result.resource_budget = "three synthetic pairs; no model execution";
  result.evidence_destination = "fixture-records-v1";
  result.code_sha256 = std::string(64, 'c');
  result.configuration_sha256 = std::string(64, 'd');
  result.temporal_mode = cf::TemporalMode::as_of;
  result.as_of = 100;
  result.expected_space = space();
  result.expected_pipeline = pipeline();
  result.response_axes = {{"temperature", "temperature-v1", "degC"},
                          {"energy", "energy-v1", "joule"}};
  result.selected_output_position = 0;
  result.attribute_rules = {{"setting", "unit", cf::AttributeRole::changed, 0, 10},
                            {"load", "unit", cf::AttributeRole::held_fixed, 0, 10}};
  result.control_tolerance = 0.01;
  result.pairs.push_back(pair("1", result.expected_space, result.expected_pipeline,
                              result.response_axes));
  return result;
}

bool has_reason(const std::vector<cf::Diagnostic>& diagnostics, cf::ReasonCode code) {
  for (const auto& diagnostic : diagnostics) {
    if (diagnostic.code == code) return true;
  }
  return false;
}

void check_summary(const cf::RunResult& result) {
  const auto& summary = result.summary;
  check(summary.succeeded + summary.abstained + summary.failed + summary.unresolved ==
            summary.requested,
        "summary must reconcile the full admitted roster");
  if (summary.requested != 0) {
    check(summary.success_rate.has_value(), "admitted roster needs a success rate");
    const double expected = static_cast<double>(summary.succeeded) /
                            static_cast<double>(summary.requested);
    check(*summary.success_rate == expected, "success rate must use full roster");
  }
}

cf::RunResult success(const cf::Request& input) {
  auto result = cf::compare(input);
  check(result.admitted, "valid request must be admitted");
  check(result.admission_diagnostics.empty(), "valid request has admission diagnostics");
  check(result.pairs.size() == input.pairs.size(), "all admitted pairs need envelopes");
  for (std::size_t i = 0; i < result.pairs.size(); ++i) {
    check(result.pairs[i].pair_index == i, "pair index must preserve roster order");
    check(result.pairs[i].pair_id == input.pairs[i].id, "pair ID must be retained");
    check(result.pairs[i].state == cf::ResultState::success, "pair must succeed");
    check(result.pairs[i].contrast.has_value(), "success must contain a contrast");
  }
  check(result.summary.succeeded == input.pairs.size(), "success count mismatch");
  check_summary(result);
  return result;
}

cf::RunResult refusal(const cf::Request& input, cf::ResultState state,
                     cf::ReasonCode reason) {
  auto result = cf::compare(input);
  check(result.admitted, "pair refusal must occur after admission");
  check(result.pairs.size() == input.pairs.size(), "refused pair must be retained");
  check(result.pairs.front().state == state, "wrong refusal state");
  check(!result.pairs.front().contrast, "refusal must not expose an ordinary contrast");
  check(has_reason(result.pairs.front().diagnostics, reason), "missing refusal reason");
  for (const auto& diagnostic : result.pairs.front().diagnostics) {
    check(!diagnostic.record_id.empty(), "diagnostic needs an affected record");
    check(!diagnostic.message.empty(), "diagnostic needs an explanation");
  }
  check_summary(result);
  return result;
}

cf::RunResult rejection(const cf::Request& input, cf::ReasonCode reason) {
  auto result = cf::compare(input);
  check(!result.admitted, "malformed request must be rejected before execution");
  check(result.pairs.empty(), "admission rejection must not invent executed pairs");
  check(has_reason(result.admission_diagnostics, reason), "missing admission reason");
  check(result.input.request_id == input.request_id, "rejected request must be retained");
  check(result.input.pairs.size() == input.pairs.size(), "rejected roster must be retained");
  return result;
}

template <typename Function>
void run_case(const std::string& name, Function&& function, int& passed_count,
              int& failed_count) {
  try {
    function();
    ++passed_count;
    std::cout << "PASS " << name << '\n';
  } catch (const std::exception& error) {
    ++failed_count;
    std::cerr << "FAIL " << name << ": " << error.what() << '\n';
  }
}

} // namespace

int main() {
  int passed_count = 0;
  int failed_count = 0;
  const auto test = [&](const std::string& name, auto&& function) {
    run_case(name, std::forward<decltype(function)>(function), passed_count, failed_count);
  };

  test("finite deltas preserve sign, axis order and response values", [] {
    const auto result = success(request());
    const auto& contrast = *result.pairs[0].contrast;
    check(contrast.baseline == std::vector<double>({2, -1}), "baseline values changed");
    check(contrast.alternative == std::vector<double>({5, -3}), "alternative values changed");
    check(contrast.delta == std::vector<double>({3, -2}), "delta must be alternative minus baseline");
    check(result.input.response_axes[1].units == "joule", "axis units must survive");
  });

  test("no-change control yields exact zero without a changed attribute", [] {
    auto input = request();
    input.attribute_rules[0].role = cf::AttributeRole::held_fixed;
    input.pairs[0].alternative->attributes = input.pairs[0].baseline->attributes;
    input.pairs[0].alternative->response_values = input.pairs[0].baseline->response_values;
    input.pairs[0].alternative->input.sha256 = input.pairs[0].baseline->input.sha256;
    input.pairs[0].change_definition = "Identity control: all recorded attributes held fixed";
    const auto result = success(input);
    check(result.pairs[0].contrast->delta == std::vector<double>({0, 0}), "identity delta must be zero");
  });

  test("source and parent lineage survives success and failure", [] {
    auto input = request();
    input.pairs[0].alternative->upstream_succeeded = false;
    const auto result = refusal(input, cf::ResultState::failed, cf::ReasonCode::upstream_failure);
    const auto& retained = *result.input.pairs[0].alternative;
    check(retained.input == input.pairs[0].alternative->input, "alternative input lost");
    check(retained.source_observation == input.pairs[0].baseline->source_observation, "source lost");
    check(retained.parent_input_id == input.pairs[0].baseline->input.id, "input parent lost");
    check(retained.parent_representation_id == input.pairs[0].baseline->representation.id,
          "representation parent lost");
    check(retained.representation_kind == cf::RepresentationKind::transformed_candidate,
          "derived kind lost");
    check(result.input.expected_pipeline.dependencies[0].fit_population == "training-population-v1",
          "fitting provenance lost");
    check(result.input.code_sha256 == input.code_sha256, "code identity lost");
  });

  test("availability is maximum of branches and fitted dependencies", [] {
    auto input = request();
    input.expected_pipeline.dependencies[4].available_at = 70;
    const auto result = success(input);
    check(result.pairs[0].available_at == 70, "availability must include support dependency");
  });

  test("retrospective unknown availability remains explicit", [] {
    auto input = request();
    input.temporal_mode = cf::TemporalMode::retrospective;
    input.as_of.reset();
    input.expected_pipeline.dependencies[0].available_at.reset();
    const auto result = success(input);
    check(!result.pairs[0].available_at, "unknown dependency must not acquire availability");
  });

  test("non-temporal request requires and retains its justification", [] {
    auto input = request();
    input.temporal_mode = cf::TemporalMode::non_temporal;
    input.as_of.reset();
    input.non_temporal_justification = "Synthetic independent scalar cases have no temporal claim";
    success(input);
    input.non_temporal_justification.clear();
    rejection(input, cf::ReasonCode::invalid_record);
  });

  test("unsupported causal and observational claims are rejected", [] {
    for (const auto claim : {cf::ClaimLevel::observational_contrast,
                             cf::ClaimLevel::identified_intervention_effect,
                             cf::ClaimLevel::unspecified}) {
      auto input = request();
      input.claim_level = claim;
      rejection(input, cf::ReasonCode::unsupported_claim);
    }
  });

  test("empty roster is not a zero-success completed run", [] {
    auto input = request();
    input.pairs.clear();
    rejection(input, cf::ReasonCode::invalid_pairing);
  });

  test("required request metadata and pinned revisions are enforced", [] {
    auto input = request();
    input.permissions_ref.clear();
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.contract_revision = "0.2.0";
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.rer_id = "another-rer";
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.request_id.clear();
    const auto result = cf::compare(input);
    check(!result.admitted && !result.admission_diagnostics.empty(), "missing request ID must reject admission");
    check(result.admission_diagnostics.front().record_id == "<missing-id>", "missing identity must remain explicit in diagnostics");
  });

  test("digests must contain exactly 64 lowercase hexadecimal characters", [] {
    for (const auto& digest : {std::string(63, 'a'), std::string(64, 'A'),
                              std::string(64, 'z')}) {
      auto input = request();
      input.code_sha256 = digest;
      rejection(input, cf::ReasonCode::invalid_record);
    }
  });

  test("duplicate pair keys are rejected before any pair executes", [] {
    auto input = request();
    input.pairs.push_back(input.pairs.front());
    rejection(input, cf::ReasonCode::invalid_pairing);
    input = request();
    input.pairs[0].id.clear();
    rejection(input, cf::ReasonCode::invalid_pairing);
  });

  test("duplicate baseline-alternative tuple cannot hide behind a new pair ID", [] {
    auto input = request();
    input.pairs.push_back(input.pairs[0]);
    input.pairs[1].id = "another-pair-id";
    rejection(input, cf::ReasonCode::invalid_pairing);
  });

  test("one baseline can support distinct alternatives with consistent immutable records", [] {
    auto input = request();
    auto second = pair("2", input.expected_space, input.expected_pipeline, input.response_axes);
    second.baseline = input.pairs[0].baseline;
    second.unit_id = input.pairs[0].unit_id;
    second.entity_id = input.pairs[0].entity_id;
    second.group_id = input.pairs[0].group_id;
    second.alternative->parent_input_id = second.baseline->input.id;
    second.alternative->parent_representation_id = second.baseline->representation.id;
    second.alternative->source_observation = second.baseline->source_observation;
    second.alternative->attributes[0].value = 4;
    second.change_definition = "Change setting 1 to 4, hold load fixed";
    input.pairs.push_back(second);
    success(input);
    input.pairs[1].baseline->response_values[0] = 7;
    rejection(input, cf::ReasonCode::invalid_record);
    input.pairs[1].baseline = input.pairs[0].baseline;
    input.pairs[1].baseline->attributes[0].value = 2;
    rejection(input, cf::ReasonCode::invalid_record);
  });

  test("conflicting digests for one immutable artifact identity reject admission", [] {
    auto input = request();
    input.pairs[0].alternative->space.preprocessing.sha256 = std::string(64, 'e');
    rejection(input, cf::ReasonCode::invalid_record);
  });

  test("missing branches fail their pair without losing the envelope", [] {
    auto input = request();
    input.pairs[0].alternative.reset();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_pairing);
  });

  test("alternative source and exact parents must match the baseline", [] {
    auto input = request();
    input.pairs[0].alternative->parent_input_id = "wrong-input";
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_pairing);
    input = request();
    input.pairs[0].alternative->parent_representation_id = "wrong-representation";
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_pairing);
    input = request();
    input.pairs[0].alternative->source_observation = artifact("wrong-source");
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_pairing);
  });

  test("changed pipeline artifact cannot produce a controlled contrast", [] {
    auto input = request();
    input.pairs[0].alternative->pipeline = artifact("pipeline-v2");
    refusal(input, cf::ResultState::failed, cf::ReasonCode::stale_or_changed_pipeline);
    input = request();
    input.pairs[0].constructor = artifact("constructor-v2");
    refusal(input, cf::ResultState::failed, cf::ReasonCode::stale_or_changed_pipeline);
  });

  test("identical input content cannot have inconsistent deterministic responses", [] {
    auto input = request();
    input.pairs[0].alternative->input.sha256 = input.pairs[0].baseline->input.sha256;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::unverified_control);
  });

  test("selected output positions must be held fixed", [] {
    auto input = request();
    input.pairs[0].alternative->output_position = 1;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::unverified_control);
    input = request();
    input.selected_output_position = -1;
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].baseline->output_position = 1;
    input.pairs[0].alternative->output_position = 1;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("equal dimensions do not excuse changed preprocessing or geometry", [] {
    auto input = request();
    input.pairs[0].alternative->space.preprocessing = artifact("preprocessing-v2");
    refusal(input, cf::ResultState::failed, cf::ReasonCode::incompatible_role_or_space);
    input = request();
    input.pairs[0].alternative->space.geometry = "cosine";
    refusal(input, cf::ResultState::failed, cf::ReasonCode::incompatible_role_or_space);
  });

  test("representation role and output kind cannot silently change", [] {
    auto input = request();
    input.pairs[0].alternative->space.input_role = "another-input-role";
    refusal(input, cf::ResultState::failed, cf::ReasonCode::incompatible_role_or_space);
    input = request();
    input.pairs[0].alternative->representation_kind = cf::RepresentationKind::observation_embedding;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].baseline->representation_kind = cf::RepresentationKind::generated_content;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("axis order, definition and units are part of compatibility", [] {
    auto input = request();
    std::swap(input.pairs[0].alternative->axes[0], input.pairs[0].alternative->axes[1]);
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].alternative->axes[0].units = "kelvin";
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].alternative->axes[0].definition_revision = "temperature-v2";
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("response length mismatch cannot be truncated", [] {
    auto input = request();
    input.pairs[0].alternative->response_values.pop_back();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("NaN and infinite recorded responses fail", [] {
    for (const auto value : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}) {
      auto input = request();
      input.pairs[0].alternative->response_values[0] = value;
      refusal(input, cf::ResultState::failed, cf::ReasonCode::nonfinite_response_or_contrast);
    }
  });

  test("finite responses whose subtraction overflows fail", [] {
    auto input = request();
    input.pairs[0].baseline->response_values[0] = -std::numeric_limits<double>::max();
    input.pairs[0].alternative->response_values[0] = std::numeric_limits<double>::max();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::nonfinite_response_or_contrast);
  });

  test("held-fixed differences honor a finite declared tolerance", [] {
    auto input = request();
    input.pairs[0].alternative->attributes[1].value = 2.005;
    success(input);
    input.pairs[0].alternative->attributes[1].value = 2.02;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::unverified_control);
    input = request();
    input.control_tolerance = 0;
    input.pairs[0].alternative->attributes[1].value = std::nextafter(2.0, 3.0);
    refusal(input, cf::ResultState::failed, cf::ReasonCode::unverified_control);
  });

  test("missing duplicate and wrong-unit controls are malformed", [] {
    auto input = request();
    input.pairs[0].alternative->attributes.pop_back();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].alternative->attributes.push_back(input.pairs[0].alternative->attributes[0]);
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].alternative->attributes[0].units = "another-unit";
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("nonfinite scalar controls do not become support abstentions", [] {
    auto input = request();
    input.pairs[0].alternative->attributes[0].value = std::numeric_limits<double>::quiet_NaN();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("declared coupled values are retained and compared within bounds", [] {
    auto input = request();
    input.attribute_rules[1].role = cf::AttributeRole::coupled;
    input.pairs[0].alternative->attributes[1].value = 4;
    success(input);
  });

  test("outside predeclared bounds abstains without a numeric contrast", [] {
    auto input = request();
    input.pairs[0].alternative->attributes[0].value = 11;
    refusal(input, cf::ResultState::abstained, cf::ReasonCode::outside_declared_support);
  });

  test("rejected validity support and interpretation checks each abstain", [] {
    auto input = request();
    input.pairs[0].alternative_validity.state = cf::CheckState::rejected;
    refusal(input, cf::ResultState::abstained, cf::ReasonCode::invalid_alternative);
    input = request();
    input.pairs[0].support.state = cf::CheckState::rejected;
    refusal(input, cf::ResultState::abstained, cf::ReasonCode::outside_declared_support);
    input = request();
    input.pairs[0].interpretability.state = cf::CheckState::rejected;
    refusal(input, cf::ResultState::abstained, cf::ReasonCode::unsupported_interpretation);
  });

  test("latent edits require an explicit coordinate-level interpretation", [] {
    auto input = request();
    input.pairs[0].alternative->origin = cf::Origin::latent_candidate;
    refusal(input, cf::ResultState::abstained, cf::ReasonCode::unsupported_interpretation);
    input.permitted_interpretation = "latent_coordinate_response";
    input.pairs[0].interpretation = "latent_coordinate_response";
    success(input);
  });

  test("missing check evidence is a failure rather than an abstention", [] {
    auto input = request();
    input.pairs[0].support.state = cf::CheckState::rejected;
    input.pairs[0].support.evidence.sha256.clear();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].alternative_validity.detail.clear();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("upstream failures take precedence over support abstention", [] {
    auto input = request();
    input.pairs[0].support.state = cf::CheckState::rejected;
    input.pairs[0].alternative->upstream_succeeded = false;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::upstream_failure);
    input = request();
    input.pairs[0].alternative->output_valid = false;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("as-of refuses unknown branch availability", [] {
    auto input = request();
    input.pairs[0].baseline->timing.available_at.reset();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::temporal_violation);
    input = request();
    input.pairs[0].baseline->timing.support_start.reset();
    input.pairs[0].baseline->timing.support_end.reset();
    refusal(input, cf::ResultState::failed, cf::ReasonCode::temporal_violation);
  });

  test("as-of refuses future support and future availability", [] {
    auto input = request();
    input.pairs[0].alternative->timing.support_end = 101;
    input.pairs[0].alternative->timing.available_at = 101;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::temporal_violation);
    input = request();
    input.pairs[0].alternative->timing.available_at = 101;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::temporal_violation);
  });

  test("reversed source support is malformed timing", [] {
    auto input = request();
    input.pairs[0].baseline->timing.support_start = 21;
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("as-of refuses unknown and future fitted dependency availability", [] {
    auto input = request();
    input.expected_pipeline.dependencies[2].available_at.reset();
    rejection(input, cf::ReasonCode::temporal_violation);
    input = request();
    input.expected_pipeline.dependencies[3].available_at = 101;
    rejection(input, cf::ReasonCode::temporal_violation);
    input = request();
    input.expected_pipeline.dependencies[0].pretraining_exposure = "unknown";
    rejection(input, cf::ReasonCode::temporal_violation);
    input = request();
    input.expected_pipeline.dependencies[0].fit_population = "unknown";
    rejection(input, cf::ReasonCode::temporal_violation);
  });

  test("dependency roster needs every role with fitted provenance", [] {
    auto input = request();
    input.expected_pipeline.dependencies.pop_back();
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.expected_pipeline.dependencies[0].fit_population.clear();
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.expected_pipeline.dependencies[0].selection_population.clear();
    rejection(input, cf::ReasonCode::invalid_record);
  });

  test("frozen dependencies must agree with the declared space and response model", [] {
    auto input = request();
    input.expected_pipeline.dependencies[1].artifact = artifact("unexpected-preprocessing");
    rejection(input, cf::ReasonCode::stale_or_changed_pipeline);
    input = request();
    input.expected_pipeline.dependencies[3].artifact = artifact("unexpected-model");
    rejection(input, cf::ReasonCode::stale_or_changed_pipeline);
  });

  test("unknown provider revision and stochastic policy are refused", [] {
    auto input = request();
    input.expected_pipeline.provider_revision_known = false;
    rejection(input, cf::ReasonCode::unverified_control);
    input = request();
    input.expected_pipeline.stochastic_policy = "unseeded";
    rejection(input, cf::ReasonCode::unverified_control);
  });

  test("finite nonnegative tolerances and finite ordered bounds are mandatory", [] {
    for (const auto value : {-1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
      auto input = request();
      input.control_tolerance = value;
      rejection(input, cf::ReasonCode::invalid_record);
    }
    auto input = request();
    input.attribute_rules[0].minimum = 11;
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.attribute_rules[0].maximum = std::numeric_limits<double>::infinity();
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.control_tolerance = 0;
    success(input);
  });

  test("invalid request enum values are refused explicitly", [] {
    auto input = request();
    input.claim_level = static_cast<cf::ClaimLevel>(999);
    rejection(input, cf::ReasonCode::unsupported_claim);
    input = request();
    input.temporal_mode = static_cast<cf::TemporalMode>(999);
    rejection(input, cf::ReasonCode::invalid_record);
    input = request();
    input.attribute_rules[0].role = static_cast<cf::AttributeRole>(999);
    rejection(input, cf::ReasonCode::invalid_record);
  });

  test("invalid branch and check enum values fail rather than fall through", [] {
    auto input = request();
    input.pairs[0].alternative->origin = static_cast<cf::Origin>(999);
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].baseline->representation_kind = static_cast<cf::RepresentationKind>(999);
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
    input = request();
    input.pairs[0].support.state = static_cast<cf::CheckState>(999);
    refusal(input, cf::ResultState::failed, cf::ReasonCode::invalid_record);
  });

  test("stop_after preserves unresolved roster entries and denominator", [] {
    auto input = request();
    input.pairs.push_back(pair("2", input.expected_space, input.expected_pipeline, input.response_axes));
    input.pairs.push_back(pair("3", input.expected_space, input.expected_pipeline, input.response_axes));
    const auto result = cf::compare(input, 1);
    check(result.admitted, "valid interrupted request must be admitted");
    check(result.pairs.size() == 3, "interruption must preserve complete roster");
    check(result.pairs[0].state == cf::ResultState::success, "first pair must complete");
    for (std::size_t i = 1; i < result.pairs.size(); ++i) {
      check(result.pairs[i].pair_index == i, "unresolved order changed");
      check(result.pairs[i].state == cf::ResultState::unresolved, "remaining pair must be unresolved");
      check(!result.pairs[i].contrast, "unresolved pair must not contain a contrast");
      check(has_reason(result.pairs[i].diagnostics, cf::ReasonCode::interrupted),
            "unresolved pair needs interruption evidence");
    }
    check(result.summary.requested == 3 && result.summary.succeeded == 1 &&
              result.summary.unresolved == 2, "interrupted summary mismatch");
    check_summary(result);
    const auto stopped = cf::compare(input, 0);
    check(stopped.summary.unresolved == 3 && stopped.summary.succeeded == 0,
          "stop before execution must preserve all unresolved pairs");
    check_summary(stopped);
  });

  test("mixed result states reconcile without silently excluding refused pairs", [] {
    auto input = request();
    input.pairs.push_back(pair("2", input.expected_space, input.expected_pipeline, input.response_axes));
    input.pairs.push_back(pair("3", input.expected_space, input.expected_pipeline, input.response_axes));
    input.pairs[1].support.state = cf::CheckState::rejected;
    input.pairs[2].alternative->upstream_succeeded = false;
    const auto result = cf::compare(input);
    check(result.admitted && result.pairs.size() == 3, "mixed run must retain all pairs");
    check(result.summary.succeeded == 1 && result.summary.abstained == 1 &&
              result.summary.failed == 1 && result.summary.unresolved == 0,
          "mixed summary counts mismatch");
    check(!result.pairs[1].contrast && !result.pairs[2].contrast, "refused contrast leaked");
    check_summary(result);
  });

  test("public state and reason names cover failure and interruption", [] {
    check(cf::to_string(cf::ResultState::success) == "success", "success name changed");
    check(cf::to_string(cf::ResultState::unresolved) == "unresolved", "unresolved name changed");
    check(!cf::to_string(cf::ReasonCode::temporal_violation).empty(), "temporal reason missing");
    check(!cf::to_string(cf::ReasonCode::interrupted).empty(), "interrupted reason missing");
  });

  std::cout << "Counterfactual paired-response cases: " << passed_count << " passed, "
            << failed_count << " failed\n";
  return failed_count == 0 ? 0 : 1;
}
