#include "mityane/operations/retrieval/geometry.h"

#include "numeric.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace mityane::operations::retrieval::v1 {
namespace {

using detail::binding_named;
using detail::finite_values;
using detail::prepare;
using detail::squared_norm;
using detail::text;

constexpr std::size_t k_max_constructor_points = 8;

ConstructorResult constructor_failure(Reason reason,
                                      ConstructorOutcome outcome = ConstructorOutcome::failed) {
  ConstructorResult result;
  result.outcome = outcome;
  result.reason = reason;
  return result;
}

SearchResult failure(Reason reason) {
  SearchResult result;
  result.outcome = Outcome::failed;
  result.reason = reason;
  return result;
}

Reason record_reason(const SpaceBinding& binding, const VectorRecord& record) {
  if (!binding_named(binding, true) || !text(record.source_id)) return Reason::invalid_request;
  if (record.values.size() != binding.dimensions) return Reason::dimension_mismatch;
  if (!finite_values(record.values)) return Reason::nonfinite_value;
  return Reason::none;
}

bool prepare_record(const SpaceBinding& binding, const VectorRecord& record,
                    std::vector<double>& prepared) {
  bool undefined = false;
  prepare(record.values, binding.metric, binding.normalization, prepared, undefined);
  return !undefined;
}

Reason corpus_reason(const Corpus& corpus) {
  if (!binding_named(corpus.binding, true)) return Reason::invalid_request;
  std::unordered_map<std::string, int> seen;
  for (const VectorRecord& record : corpus.records) {
    const Reason problem = record_reason(corpus.binding, record);
    if (problem != Reason::none) return problem;
    if (++seen[record.source_id] > 1) return Reason::duplicate_source_id;
  }
  return Reason::none;
}

bool radius_pair(const std::optional<double>& radius_min, const std::optional<double>& radius_max) {
  if (radius_min && (!std::isfinite(*radius_min) || *radius_min < 0)) return false;
  if (radius_max && (!std::isfinite(*radius_max) || *radius_max < 0)) return false;
  if (radius_min && radius_max && *radius_min > *radius_max) return false;
  return true;
}

bool inside_cone(const Cone& cone, const std::vector<double>& prepared) {
  const double sum = squared_norm(prepared);
  if (sum == 0) return false;
  if (cone.theta != std::numbers::pi_v<double>) {
    double dot = 0;
    for (std::size_t index = 0; index < prepared.size(); ++index) dot += prepared[index] * cone.axis[index];
    const double cosine = dot / std::sqrt(sum);
    if (cone.theta == 0) {
      if (cosine != 1) return false;
    } else if (cosine < std::cos(cone.theta)) {
      return false;
    }
  }
  const double radius = std::sqrt(sum);
  if (cone.radius_min && radius < *cone.radius_min) return false;
  if (cone.radius_max && radius > *cone.radius_max) return false;
  return true;
}

}  // namespace

ConstructorResult difference(const SpaceBinding& binding, const VectorRecord& origin,
                             const VectorRecord& target) {
  const Reason origin_problem = record_reason(binding, origin);
  if (origin_problem != Reason::none) return constructor_failure(origin_problem);
  const Reason target_problem = record_reason(binding, target);
  if (target_problem != Reason::none) return constructor_failure(target_problem);
  std::vector<double> prepared_origin;
  std::vector<double> prepared_target;
  if (!prepare_record(binding, origin, prepared_origin) ||
      !prepare_record(binding, target, prepared_target)) {
    return constructor_failure(Reason::undefined_geometry, ConstructorOutcome::abstained);
  }
  std::vector<double> values(binding.dimensions);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = prepared_target[index] - prepared_origin[index];
  }
  if (squared_norm(values) == 0) return constructor_failure(Reason::collinear_inputs);
  ConstructorResult result;
  result.outcome = ConstructorOutcome::constructed;
  result.reason = Reason::none;
  result.values = std::move(values);
  return result;
}

ConstructorResult shared_direction(const SpaceBinding& binding,
                                   const std::vector<VectorRecord>& points) {
  if (points.size() < 2 || points.size() > k_max_constructor_points) {
    return constructor_failure(Reason::invalid_request);
  }
  std::unordered_map<std::string, int> seen;
  std::vector<std::vector<double>> prepared;
  prepared.reserve(points.size());
  for (const VectorRecord& point : points) {
    const Reason problem = record_reason(binding, point);
    if (problem != Reason::none) return constructor_failure(problem);
    if (++seen[point.source_id] > 1) return constructor_failure(Reason::duplicate_source_id);
    std::vector<double> row;
    if (!prepare_record(binding, point, row)) {
      return constructor_failure(Reason::undefined_geometry, ConstructorOutcome::abstained);
    }
    prepared.push_back(std::move(row));
  }

  std::vector<std::vector<double>> basis;
  bool separated = false;
  for (std::size_t index = 1; index < prepared.size(); ++index) {
    std::vector<double> direction(binding.dimensions);
    for (std::size_t axis = 0; axis < binding.dimensions; ++axis) {
      direction[axis] = prepared[index][axis] - prepared[0][axis];
    }
    for (const std::vector<double>& kept : basis) {
      double projection = 0;
      for (std::size_t axis = 0; axis < binding.dimensions; ++axis) projection += direction[axis] * kept[axis];
      for (std::size_t axis = 0; axis < binding.dimensions; ++axis) direction[axis] -= projection * kept[axis];
    }
    const double sum = squared_norm(direction);
    if (sum == 0) continue;
    separated = true;
    const double norm = std::sqrt(sum);
    for (double& value : direction) value /= norm;
    basis.push_back(std::move(direction));
  }
  if (!separated) return constructor_failure(Reason::collinear_inputs);

  std::vector<double> residual = prepared[0];
  for (const std::vector<double>& kept : basis) {
    double projection = 0;
    for (std::size_t axis = 0; axis < binding.dimensions; ++axis) projection += residual[axis] * kept[axis];
    for (std::size_t axis = 0; axis < binding.dimensions; ++axis) residual[axis] -= projection * kept[axis];
  }
  if (squared_norm(residual) == 0) return constructor_failure(Reason::collinear_inputs);
  ConstructorResult result;
  result.outcome = ConstructorOutcome::constructed;
  result.reason = Reason::none;
  result.values = std::move(residual);
  return result;
}

ConeResult make_cone(const SpaceBinding& binding, const std::vector<double>& axis, double theta,
                     std::optional<double> radius_min, std::optional<double> radius_max) {
  if (!binding_named(binding, true) || !std::isfinite(theta) || theta < 0 ||
      theta > std::numbers::pi_v<double> || !radius_pair(radius_min, radius_max)) {
    return {ConstructorOutcome::failed, Reason::invalid_request, {}};
  }
  if (axis.size() != binding.dimensions) return {ConstructorOutcome::failed, Reason::dimension_mismatch, {}};
  if (!finite_values(axis)) return {ConstructorOutcome::failed, Reason::nonfinite_value, {}};
  std::vector<double> prepared;
  bool undefined = false;
  prepare(axis, binding.metric, binding.normalization, prepared, undefined);
  if (undefined || squared_norm(prepared) == 0) {
    return {ConstructorOutcome::abstained, Reason::undefined_geometry, {}};
  }
  const double norm = std::sqrt(squared_norm(prepared));
  for (double& value : prepared) value /= norm;
  Cone cone;
  cone.binding = binding;
  cone.axis = std::move(prepared);
  cone.theta = theta;
  cone.radius_min = radius_min;
  cone.radius_max = radius_max;
  return {ConstructorOutcome::constructed, Reason::none, std::move(cone)};
}

SearchResult search_cone(const Corpus& corpus, const Cone& cone, std::size_t k,
                         const ExclusionPolicy& exclusions) {
  if (k == 0) return failure(Reason::invalid_request);
  const Reason corpus_problem = corpus_reason(corpus);
  if (corpus_problem != Reason::none) return failure(corpus_problem);
  if (cone.binding != corpus.binding || cone.axis.size() != corpus.binding.dimensions ||
      !finite_values(cone.axis) || squared_norm(cone.axis) == 0 || !std::isfinite(cone.theta)) {
    return failure(Reason::incompatible_space);
  }

  Corpus eligible{corpus.binding, {}};
  SearchResult result;
  result.reason = Reason::none;
  for (const VectorRecord& record : corpus.records) {
    if (std::find(exclusions.source_ids.begin(), exclusions.source_ids.end(), record.source_id) !=
        exclusions.source_ids.end()) {
      result.skipped.push_back({record.source_id, SkipReason::excluded_source});
      continue;
    }
    std::vector<double> prepared;
    bool undefined = false;
    prepare(record.values, corpus.binding.metric, corpus.binding.normalization, prepared, undefined);
    if (undefined || squared_norm(prepared) == 0) {
      result.skipped.push_back({record.source_id, SkipReason::undefined_geometry});
      continue;
    }
    if (!inside_cone(cone, prepared)) {
      result.skipped.push_back({record.source_id, SkipReason::outside_cone});
      continue;
    }
    eligible.records.push_back(record);
  }

  Query probe;
  probe.binding = corpus.binding;
  probe.query_id = "cone-axis";
  probe.source_id.clear();
  probe.values = cone.axis;
  const SearchResult ranked = search(eligible, probe, k, {});
  if (ranked.outcome == Outcome::failed) return ranked;
  result.hits = ranked.hits;
  result.eligible_count = ranked.eligible_count;
  result.skipped.insert(result.skipped.end(), ranked.skipped.begin(), ranked.skipped.end());
  result.outcome = ranked.outcome;
  result.reason = ranked.reason;
  return result;
}

ConeReport cone_diagnostics(const Corpus& corpus, const Cone& cone, std::size_t k,
                            const std::vector<std::string>& labeled_source_ids) {
  ConeReport report;
  if (k == 0) {
    report.reason = Reason::invalid_request;
    return report;
  }
  const SearchResult found = search_cone(corpus, cone, corpus.records.empty() ? 1 : corpus.records.size(), {});
  if (found.outcome == Outcome::failed && found.reason != Reason::none) {
    report.reason = found.reason;
    return report;
  }
  std::unordered_set<std::string> inside;
  for (const Hit& hit : found.hits) inside.insert(hit.source_id);
  report.inside_count = inside.size();
  if (!labeled_source_ids.empty() && report.inside_count > 0) {
    const std::unordered_set<std::string> labels(labeled_source_ids.begin(), labeled_source_ids.end());
    std::size_t labeled = 0;
    for (const std::string& source_id : inside) {
      if (labels.contains(source_id)) ++labeled;
    }
    report.purity = static_cast<double>(labeled) / static_cast<double>(report.inside_count);
  }
  Corpus neighborhood_corpus{corpus.binding, {}};
  for (const VectorRecord& record : corpus.records) {
    if (inside.contains(record.source_id)) neighborhood_corpus.records.push_back(record);
  }
  report.neighborhood = neighborhood_diagnostics(neighborhood_corpus, k);
  if (report.neighborhood.outcome == Outcome::failed) {
    report.outcome = Outcome::failed;
    report.reason = report.neighborhood.reason;
    return report;
  }
  report.reason = Reason::none;
  if (report.inside_count == 0) report.outcome = Outcome::empty;
  else if (report.neighborhood.outcome == Outcome::abstained) report.outcome = Outcome::abstained;
  else report.outcome = Outcome::ranked;
  return report;
}

}  // namespace mityane::operations::retrieval::v1
