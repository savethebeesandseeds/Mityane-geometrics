#include "mityane/operations/counterfactuals/paired_response.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <utility>

namespace mityane::operations::counterfactuals::v1 {
static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits == 53 && sizeof(double) == 8,
              "The reference implementation requires IEEE 754 binary64 doubles.");
namespace {

bool text(const std::string& value) {
  return value.find_first_not_of(" \t\r\n") != std::string::npos;
}
bool digest(const std::string& value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
bool artifact(const ArtifactRef& value) { return text(value.id) && digest(value.sha256); }
void issue(std::vector<Diagnostic>& diagnostics, ReasonCode code,
           const std::string& id, std::string message) {
  diagnostics.push_back({code, text(id) ? id : "<missing-id>", std::move(message)});
}
bool space_valid(const SpaceRef& space) {
  if (!artifact(space.artifact) || !artifact(space.provider) ||
      !artifact(space.preprocessing) || !artifact(space.fitted_transform) ||
      !text(space.output_selector) || !text(space.layout) ||
      !text(space.normalization) || !text(space.geometry) ||
      !text(space.input_role) || space.native_shape.empty()) return false;
  std::size_t size = 1;
  for (const auto dim : space.native_shape) {
    if (dim == 0 || size > std::numeric_limits<std::size_t>::max() / dim) return false;
    size *= dim;
  }
  return true;
}
bool axes_valid(const std::vector<Axis>& axes) {
  if (axes.empty()) return false;
  std::set<std::string> names;
  for (const auto& axis : axes)
    if (!text(axis.id) || !text(axis.definition_revision) || !text(axis.units) ||
        !names.insert(axis.id).second) return false;
  return true;
}
bool kind_valid(RepresentationKind kind) {
  switch (kind) {
    case RepresentationKind::observation_embedding:
    case RepresentationKind::task_state:
    case RepresentationKind::predicted_state:
    case RepresentationKind::transformed_candidate:
    case RepresentationKind::generated_content: return true;
    default: return false;
  }
}
bool derived_origin(Origin origin) {
  return origin == Origin::edited_input || origin == Origin::simulated_input ||
         origin == Origin::latent_candidate;
}
const Dependency* dependency(const PipelineRef& pipeline, std::string_view role) {
  for (const auto& dep : pipeline.dependencies) if (dep.role == role) return &dep;
  return nullptr;
}
bool same_values(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i)
    if (std::bit_cast<std::uint64_t>(left[i]) != std::bit_cast<std::uint64_t>(right[i])) return false;
  return true;
}
auto scalar_manifest(const Branch& branch) {
  std::map<std::string, std::pair<std::string, std::uint64_t>> fields;
  for (const auto& field : branch.attributes)
    fields.emplace(field.name, std::pair{field.units, std::bit_cast<std::uint64_t>(field.value)});
  return fields;
}
bool same_timing(const TemporalSupport& a, const TemporalSupport& b) {
  return a.support_start == b.support_start && a.support_end == b.support_end &&
         a.available_at == b.available_at && a.vintage == b.vintage;
}

std::vector<Diagnostic> admission(const Request& request) {
  std::vector<Diagnostic> errors;
  const auto add = [&](ReasonCode code, std::string message) {
    issue(errors, code, request.request_id, std::move(message));
  };
  if (request.claim_level != ClaimLevel::controlled_model_response)
    add(ReasonCode::unsupported_claim, "Only controlled_model_response is supported.");
  if (request.contract_id != "OR06-PCR" || request.contract_revision != "0.1.0" ||
      request.rer_id != "OR06-PCR-RER" || request.rer_revision != "0.1.0")
    add(ReasonCode::invalid_record, "Unsupported contract or RER identity/revision.");
  if (!text(request.request_id) || !text(request.operation_run_id) ||
      !text(request.purpose) || !text(request.population_id) ||
      !text(request.sampling_rule) || !text(request.response_horizon) ||
      !text(request.permitted_interpretation) || !text(request.permissions_ref) ||
      !text(request.resource_budget) || !text(request.evidence_destination) ||
      !digest(request.code_sha256) || !digest(request.configuration_sha256))
    add(ReasonCode::invalid_record, "Request metadata or code/configuration pin is missing or malformed.");
  if (!std::isfinite(request.control_tolerance) || request.control_tolerance < 0)
    add(ReasonCode::invalid_record, "Control tolerance must be finite and nonnegative.");
  if (request.permitted_interpretation != "model_input_response" &&
      request.permitted_interpretation != "latent_coordinate_response")
    add(ReasonCode::invalid_record, "A supported model-input or latent-coordinate interpretation must be bound.");
  if (!space_valid(request.expected_space) || !axes_valid(request.response_axes))
    add(ReasonCode::invalid_record, "Expected space or numerical response axes are malformed.");
  if (request.selected_output_position < 0)
    add(ReasonCode::invalid_record, "A selected output position must be explicitly bound.");
  const auto& pipeline = request.expected_pipeline;
  if (!artifact(pipeline.artifact) || !artifact(pipeline.response_model) ||
      !digest(pipeline.configuration_sha256))
    add(ReasonCode::invalid_record, "Expected pipeline or fitted response identity is malformed.");
  if (pipeline.stochastic_policy != "deterministic" || !pipeline.provider_revision_known)
    add(ReasonCode::unverified_control, "Reference slice requires a known, fixed deterministic provider/pipeline.");

  const bool as_of = request.temporal_mode == TemporalMode::as_of;
  switch (request.temporal_mode) {
    case TemporalMode::as_of:
      if (!request.as_of) add(ReasonCode::temporal_violation, "As-of mode requires a cutoff.");
      break;
    case TemporalMode::non_temporal:
      if (!text(request.non_temporal_justification))
        add(ReasonCode::invalid_record, "Non-temporal mode requires a justification.");
      [[fallthrough]];
    case TemporalMode::retrospective:
      if (request.as_of) add(ReasonCode::temporal_violation, "A cutoff is only valid in as-of mode.");
      break;
    default: add(ReasonCode::invalid_record, "Temporal mode must be explicit and supported.");
  }
  std::set<std::string> dependency_roles;
  for (const auto& dep : pipeline.dependencies) {
    if (!text(dep.role) || !dependency_roles.insert(dep.role).second ||
        !artifact(dep.artifact) || !text(dep.fit_population) ||
        !text(dep.selection_population) ||
        (dep.pretraining_exposure != "known" && dep.pretraining_exposure != "unknown" &&
         dep.pretraining_exposure != "excluded_by_design"))
      add(ReasonCode::invalid_record, "Dependency role, identity or fit/exposure record is malformed.");
    if (as_of && (!dep.available_at || (request.as_of && *dep.available_at > *request.as_of) ||
                  dep.pretraining_exposure == "unknown" || dep.fit_population == "unknown" ||
                  dep.selection_population == "unknown"))
      issue(errors, ReasonCode::temporal_violation, dep.artifact.id,
            "Dependency availability or fit/selection exposure cannot support the cutoff.");
  }
  const std::pair<std::string_view, const ArtifactRef*> expected[] = {
    {"provider", &request.expected_space.provider},
    {"preprocessing", &request.expected_space.preprocessing},
    {"fitted_transform", &request.expected_space.fitted_transform},
    {"response_model", &pipeline.response_model}
  };
  for (const auto& [role, ref] : expected) {
    const auto* dep = dependency(pipeline, role);
    if (!dep || dep->artifact != *ref)
      add(ReasonCode::stale_or_changed_pipeline, "Missing or mismatched fitted dependency: " + std::string(role));
  }
  if (!dependency(pipeline, "support_policy") || !dependency(pipeline, "constructor"))
    add(ReasonCode::invalid_record, "Support/check policy and alternative constructor dependencies are required.");

  std::set<std::string> attributes;
  if (request.attribute_rules.empty()) add(ReasonCode::invalid_record, "Scalar attribute rules are required.");
  for (const auto& rule : request.attribute_rules) {
    if (!text(rule.name) || !text(rule.units) || !attributes.insert(rule.name).second ||
        !std::isfinite(rule.minimum) || !std::isfinite(rule.maximum) || rule.minimum > rule.maximum ||
        (rule.role != AttributeRole::changed && rule.role != AttributeRole::held_fixed &&
         rule.role != AttributeRole::coupled))
      add(ReasonCode::invalid_record, "Scalar attribute rule is malformed or duplicated.");
  }
  std::set<std::string> pairs;
  std::set<std::pair<std::string, std::string>> input_pairs;
  if (request.pairs.empty()) add(ReasonCode::invalid_pairing, "The complete pair roster must be nonempty.");
  for (const auto& pair : request.pairs) {
    if (!text(pair.id) || !pairs.insert(pair.id).second)
      issue(errors, ReasonCode::invalid_pairing, pair.id, "Pair IDs must be present and unique.");
    if (pair.baseline && pair.alternative &&
        !input_pairs.emplace(pair.baseline->input.id, pair.alternative->input.id).second)
      issue(errors, ReasonCode::invalid_pairing, pair.id, "The same baseline/alternative input pair is repeated.");
  }
  // Parent links use immutable IDs, so an ID cannot resolve to conflicting pins.
  std::map<std::string, std::string> pins;
  std::map<std::string, const Branch*> input_records, representation_records, response_records;
  const auto pin = [&](const ArtifactRef& ref) {
    if (!artifact(ref)) return; // malformed pair-local references fail per pair
    const auto [it, inserted] = pins.emplace(ref.id, ref.sha256);
    if (!inserted && it->second != ref.sha256)
      issue(errors, ReasonCode::invalid_record, ref.id, "One immutable artifact ID has conflicting SHA256 pins.");
  };
  const auto pin_space = [&](const SpaceRef& space) {
    pin(space.artifact); pin(space.provider); pin(space.preprocessing); pin(space.fitted_transform);
  };
  pin_space(request.expected_space);
  pin(pipeline.artifact); pin(pipeline.response_model);
  for (const auto& dep : pipeline.dependencies) pin(dep.artifact);
  for (const auto& pair : request.pairs) {
    pin(pair.constructor); pin(pair.control_evidence);
    for (const auto* check : {&pair.alternative_validity, &pair.support, &pair.interpretability}) pin(check->evidence);
    for (const auto* branch : {&pair.baseline, &pair.alternative}) if (*branch) {
      const auto& value = **branch;
      pin(value.input); pin(value.source_observation); pin(value.representation);
      pin_space(value.space); pin(value.pipeline); pin(value.response); pin(value.mask);
      if (artifact(value.input)) {
        const auto [it, inserted] = input_records.emplace(value.input.id, &value);
        if (!inserted) {
          const auto& old = *it->second;
          if (old.source_observation != value.source_observation || old.origin != value.origin ||
              old.parent_input_id != value.parent_input_id || !same_timing(old.timing, value.timing) ||
              old.attributes.size() != value.attributes.size() || scalar_manifest(old) != scalar_manifest(value))
            issue(errors, ReasonCode::invalid_record, value.input.id, "One input ID has contradictory source/control metadata.");
        }
      }
      if (artifact(value.representation)) {
        const auto [it, inserted] = representation_records.emplace(value.representation.id, &value);
        if (!inserted) {
          const auto& old = *it->second;
          if (old.input != value.input || old.representation_kind != value.representation_kind ||
              old.parent_representation_id != value.parent_representation_id || old.space != value.space ||
              old.mask != value.mask || old.output_valid != value.output_valid)
            issue(errors, ReasonCode::invalid_record, value.representation.id, "One representation ID has contradictory parent/kind/space metadata.");
        }
      }
      if (artifact(value.response)) {
        const auto [it, inserted] = response_records.emplace(value.response.id, &value);
        if (!inserted) {
          const auto& old = *it->second;
          if (old.input != value.input || old.representation != value.representation ||
              old.axes != value.axes || !same_values(old.response_values, value.response_values) ||
              old.pipeline != value.pipeline || old.response_execution_id != value.response_execution_id ||
              old.output_position != value.output_position || old.upstream_succeeded != value.upstream_succeeded)
            issue(errors, ReasonCode::invalid_record, value.response.id, "One response ID has contradictory numerical/execution metadata.");
        }
      }
    }
  }
  return errors;
}

std::optional<std::int64_t> availability(const Request& request, const Pair& pair) {
  if (!pair.baseline || !pair.alternative || !pair.baseline->timing.available_at ||
      !pair.alternative->timing.available_at) return std::nullopt;
  auto latest = std::max(*pair.baseline->timing.available_at, *pair.alternative->timing.available_at);
  for (const auto& dep : request.expected_pipeline.dependencies) {
    if (!dep.available_at) return std::nullopt;
    latest = std::max(latest, *dep.available_at);
  }
  return latest;
}

void validate_branch(const Request& request, const Branch& branch,
                     std::vector<Diagnostic>& errors) {
  const auto add = [&](ReasonCode code, std::string message) {
    issue(errors, code, branch.input.id, std::move(message));
  };
  if (!artifact(branch.input) || !artifact(branch.source_observation) ||
      !artifact(branch.representation) || !artifact(branch.response) ||
      !artifact(branch.mask) || !text(branch.response_execution_id) ||
      !kind_valid(branch.representation_kind) || branch.output_position != request.selected_output_position || !branch.output_valid)
    add(ReasonCode::invalid_record, "Branch identity, representation kind or selected mask/output position is invalid.");
  if (!branch.upstream_succeeded) add(ReasonCode::upstream_failure, "Upstream response execution did not succeed.");
  if (!space_valid(branch.space) || branch.space != request.expected_space)
    add(ReasonCode::incompatible_role_or_space, "Full role/space identity differs from the bound response input.");
  if (!artifact(branch.pipeline) || branch.pipeline != request.expected_pipeline.artifact)
    add(ReasonCode::stale_or_changed_pipeline, "Response pipeline differs from the bound fitted pipeline.");
  if (!axes_valid(branch.axes) || branch.axes != request.response_axes ||
      branch.response_values.size() != request.response_axes.size())
    add(ReasonCode::invalid_record, "Numerical response axes, definitions, units or value counts do not align.");
  for (double value : branch.response_values) if (!std::isfinite(value)) {
    add(ReasonCode::nonfinite_response_or_contrast, "Response contains a non-finite value.");
    break;
  }
  const auto& timing = branch.timing;
  if (!text(timing.vintage) || timing.support_start.has_value() != timing.support_end.has_value() ||
      (timing.support_start && *timing.support_start > *timing.support_end))
    add(ReasonCode::invalid_record, "Source support interval or vintage is malformed.");
  if (timing.support_end && timing.available_at && *timing.support_end > *timing.available_at)
    add(ReasonCode::temporal_violation, "Feature support extends beyond its declared information availability.");
  if (request.temporal_mode == TemporalMode::as_of &&
      (!timing.support_start || !timing.support_end || !timing.available_at ||
       (request.as_of && ((timing.available_at && *timing.available_at > *request.as_of) ||
                         (timing.support_end && *timing.support_end > *request.as_of)))))
    add(ReasonCode::temporal_violation, "Branch support/availability cannot satisfy the as-of cutoff.");
}

PairResult compare_pair(const Request& request, const Pair& pair, std::size_t index) {
  PairResult result;
  result.pair_index = index;
  result.pair_id = pair.id;
  result.available_at = availability(request, pair);
  auto& errors = result.diagnostics;
  const auto add = [&](ReasonCode code, std::string message) {
    issue(errors, code, pair.id, std::move(message));
  };
  if (!text(pair.unit_id) || !text(pair.entity_id) || !text(pair.group_id) ||
      !artifact(pair.constructor) || !artifact(pair.control_evidence) || !text(pair.change_definition))
    add(ReasonCode::invalid_record, "Pair population, constructor or control evidence is malformed.");
  const auto* constructor = dependency(request.expected_pipeline, "constructor");
  if (!constructor || pair.constructor != constructor->artifact)
    add(ReasonCode::stale_or_changed_pipeline, "Alternative constructor differs from its bound dependency.");
  const RecordedCheck* checks[] = {&pair.alternative_validity, &pair.support, &pair.interpretability};
  for (const auto* check : checks)
    if (!artifact(check->evidence) || !text(check->detail) ||
        (check->state != CheckState::passed && check->state != CheckState::rejected))
      add(ReasonCode::invalid_record, "Recorded validity/support/interpretation check is missing or malformed.");
  if (!pair.baseline || !pair.alternative) {
    add(ReasonCode::invalid_pairing, "Both baseline and alternative branches are required.");
    return result;
  }
  const auto& base = *pair.baseline;
  const auto& alt = *pair.alternative;
  validate_branch(request, base, errors);
  validate_branch(request, alt, errors);
  if (base.origin != Origin::observed || !derived_origin(alt.origin) ||
      base.representation_kind == RepresentationKind::transformed_candidate ||
      base.representation_kind == RepresentationKind::generated_content ||
      alt.representation_kind == RepresentationKind::observation_embedding)
    add(ReasonCode::invalid_record, "Baseline must retain observed origin and alternative must retain its derived origin/kind.");
  if (alt.parent_input_id != base.input.id || alt.parent_representation_id != base.representation.id ||
      alt.source_observation != base.source_observation || alt.input.id == base.input.id ||
      alt.representation.id == base.representation.id || alt.response.id == base.response.id)
    add(ReasonCode::invalid_pairing, "Alternative lineage, distinct record IDs or source-parent relationship is invalid.");
  if (base.output_position != alt.output_position)
    add(ReasonCode::unverified_control, "Selected response output positions differ across the pair.");

  const auto values = [&](const Branch& branch) {
    std::map<std::string, const ScalarValue*> map;
    for (const auto& value : branch.attributes)
      if (!text(value.name) || !text(value.units) || !std::isfinite(value.value) ||
          !map.emplace(value.name, &value).second)
        issue(errors, ReasonCode::invalid_record, branch.input.id, "Scalar manifest contains malformed, duplicate or non-finite values.");
    if (map.size() != request.attribute_rules.size())
      issue(errors, ReasonCode::invalid_record, branch.input.id, "Scalar manifest does not cover the complete attribute rule set.");
    return map;
  };
  const auto base_values = values(base);
  const auto alt_values = values(alt);
  bool outside = false;
  for (const auto& rule : request.attribute_rules) {
    const auto b = base_values.find(rule.name), a = alt_values.find(rule.name);
    if (b == base_values.end() || a == alt_values.end()) {
      add(ReasonCode::invalid_record, "Missing scalar manifest field: " + rule.name);
      continue;
    }
    if (b->second->units != rule.units || a->second->units != rule.units)
      add(ReasonCode::invalid_record, "Scalar manifest units differ from rule: " + rule.name);
    const double before = b->second->value, after = a->second->value;
    if (std::isfinite(before) && std::isfinite(after)) {
      if (rule.role == AttributeRole::held_fixed &&
          std::abs(static_cast<long double>(after) - static_cast<long double>(before)) > request.control_tolerance)
        add(ReasonCode::unverified_control, "A held-fixed scalar changed: " + rule.name);
      outside = outside || before < rule.minimum || before > rule.maximum ||
                           after < rule.minimum || after > rule.maximum;
    }
  }
  // Arithmetic errors are failures even when a validity check would abstain.
  std::vector<double> delta;
  if (base.response_values.size() == request.response_axes.size() &&
      alt.response_values.size() == request.response_axes.size()) {
    for (std::size_t k = 0; k < request.response_axes.size(); ++k) {
      const double value = alt.response_values[k] - base.response_values[k];
      if (!std::isfinite(value))
        add(ReasonCode::nonfinite_response_or_contrast, "Paired subtraction produced a non-finite contrast.");
      delta.push_back(value);
      if (base.input.sha256 == alt.input.sha256 && std::isfinite(value) &&
          std::abs(static_cast<long double>(value)) > request.control_tolerance)
        add(ReasonCode::unverified_control, "Identical input content has inconsistent deterministic responses.");
    }
  }
  if (!errors.empty()) return result;
  if (pair.alternative_validity.state == CheckState::rejected)
    add(ReasonCode::invalid_alternative, "Supplied alternative-validity check rejected the alternative.");
  if (outside || pair.support.state == CheckState::rejected)
    add(ReasonCode::outside_declared_support, "Pair is outside the declared scalar/support region.");
  if (pair.interpretability.state == CheckState::rejected ||
      pair.interpretation != request.permitted_interpretation ||
      (pair.interpretation != "model_input_response" && pair.interpretation != "latent_coordinate_response") ||
      (alt.origin == Origin::latent_candidate && pair.interpretation != "latent_coordinate_response"))
    add(ReasonCode::unsupported_interpretation, "The declared model/coordinate interpretation lacks support.");
  if (!errors.empty()) {
    result.state = ResultState::abstained;
    return result;
  }
  result.state = ResultState::success;
  result.contrast = Contrast{base.response_values, alt.response_values, std::move(delta)};
  return result;
}
} // namespace

RunResult compare(const Request& request, std::optional<std::size_t> stop_after) {
  RunResult result;
  result.input = request;
  result.summary.requested = request.pairs.size();
  result.admission_diagnostics = admission(request);
  if (!result.admission_diagnostics.empty()) return result;
  result.admitted = true;
  result.pairs.reserve(request.pairs.size());
  for (std::size_t i = 0; i < request.pairs.size(); ++i) {
    PairResult pair;
    if (stop_after && i >= *stop_after) {
      pair.pair_index = i;
      pair.pair_id = request.pairs[i].id;
      pair.state = ResultState::unresolved;
      pair.available_at = availability(request, request.pairs[i]);
      pair.diagnostics.push_back({ReasonCode::interrupted, pair.pair_id, "Not processed before the explicit stop boundary."});
    } else pair = compare_pair(request, request.pairs[i], i);
    switch (pair.state) {
      case ResultState::success: ++result.summary.succeeded; break;
      case ResultState::abstained: ++result.summary.abstained; break;
      case ResultState::failed: ++result.summary.failed; break;
      case ResultState::unresolved: ++result.summary.unresolved; break;
    }
    result.pairs.push_back(std::move(pair));
  }
  result.summary.success_rate = static_cast<double>(result.summary.succeeded) /
                                static_cast<double>(result.summary.requested);
  return result;
}

std::string to_string(ResultState state) {
  switch (state) {
    case ResultState::success: return "success";
    case ResultState::abstained: return "abstained";
    case ResultState::failed: return "failed";
    case ResultState::unresolved: return "unresolved";
  }
  return "invalid";
}
std::string to_string(ReasonCode code) {
  switch (code) {
    case ReasonCode::unsupported_claim: return "UNSUPPORTED_CLAIM";
    case ReasonCode::invalid_record: return "INVALID_RECORD";
    case ReasonCode::invalid_pairing: return "INVALID_PAIRING";
    case ReasonCode::incompatible_role_or_space: return "INCOMPATIBLE_ROLE_OR_SPACE";
    case ReasonCode::stale_or_changed_pipeline: return "STALE_OR_CHANGED_PIPELINE";
    case ReasonCode::unverified_control: return "UNVERIFIED_CONTROL";
    case ReasonCode::temporal_violation: return "TEMPORAL_VIOLATION";
    case ReasonCode::upstream_failure: return "UPSTREAM_FAILURE";
    case ReasonCode::nonfinite_response_or_contrast: return "NONFINITE_RESPONSE_OR_CONTRAST";
    case ReasonCode::invalid_alternative: return "INVALID_ALTERNATIVE";
    case ReasonCode::outside_declared_support: return "OUTSIDE_DECLARED_SUPPORT";
    case ReasonCode::unsupported_interpretation: return "UNSUPPORTED_INTERPRETATION";
    case ReasonCode::interrupted: return "INTERRUPTED";
  }
  return "INVALID";
}

} // namespace mityane::operations::counterfactuals::v1
