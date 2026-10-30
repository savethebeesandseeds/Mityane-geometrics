#include "mityane/operations/retrieval/exact_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace mityane::operations::retrieval::v1 {
namespace {

static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53 && sizeof(double) == 8,
              "The reference implementation requires IEEE 754 binary64 doubles.");

bool text(const std::string& value) {
  return value.find_first_not_of(" \t\r\n") != std::string::npos;
}

bool finite_values(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}

double squared_norm(const std::vector<double>& values) {
  double sum = 0;
  for (double value : values) sum += value * value;
  return sum;
}

SearchResult failure(Reason reason) {
  SearchResult result;
  result.outcome = Outcome::failed;
  result.reason = reason;
  return result;
}

void prepare(const std::vector<double>& values, Metric metric, Normalization normalization,
             std::vector<double>& prepared, bool& undefined) {
  prepared = values;
  const double sum = squared_norm(prepared);
  const bool needs_direction = normalization == Normalization::l2 || metric == Metric::cosine;
  if (needs_direction && sum == 0) {
    undefined = true;
    return;
  }
  if (normalization == Normalization::l2) {
    const double norm = std::sqrt(sum);
    for (double& value : prepared) value /= norm;
  }
  undefined = false;
}

struct Scored {
  std::string source_id;
  double distance = 0;
  double similarity = 0;
};

bool better(Metric metric, const Scored& left, const Scored& right) {
  if (metric == Metric::cosine) {
    if (left.similarity != right.similarity) return left.similarity > right.similarity;
  } else if (left.distance != right.distance) {
    return left.distance < right.distance;
  }
  return left.source_id < right.source_id;
}

void measure(Metric metric, const std::vector<double>& left, const std::vector<double>& right,
             double& distance, double& similarity) {
  double dot = 0;
  double left_squared = 0;
  double right_squared = 0;
  double squared = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double delta = left[index] - right[index];
    dot += left[index] * right[index];
    left_squared += left[index] * left[index];
    right_squared += right[index] * right[index];
    squared += delta * delta;
  }
  if (metric == Metric::cosine) {
    similarity = dot / std::sqrt(left_squared * right_squared);
    distance = 1 - similarity;
    return;
  }
  distance = std::sqrt(squared);
  similarity = 1 / (1 + distance);
}

bool binding_named(const SpaceBinding& binding, bool require_snapshot) {
  if (!text(binding.space_id) || binding.dimensions == 0) return false;
  if (require_snapshot && !text(binding.snapshot_id)) return false;
  return binding.metric == Metric::cosine || binding.metric == Metric::l2;
}

Reason corpus_reason(const Corpus& corpus) {
  if (!binding_named(corpus.binding, true)) return Reason::invalid_request;
  std::unordered_map<std::string, int> seen;
  for (const VectorRecord& record : corpus.records) {
    if (!text(record.source_id)) return Reason::invalid_request;
    if (record.values.size() != corpus.binding.dimensions) return Reason::dimension_mismatch;
    if (!finite_values(record.values)) return Reason::nonfinite_value;
    if (++seen[record.source_id] > 1) return Reason::duplicate_source_id;
  }
  return Reason::none;
}

}  // namespace

SearchResult search(const Corpus& corpus, const Query& query, std::size_t k,
                    const ExclusionPolicy& exclusions) {
  if (k == 0 || !text(query.query_id)) return failure(Reason::invalid_request);
  const Reason corpus_problem = corpus_reason(corpus);
  if (corpus_problem != Reason::none) return failure(corpus_problem);
  if (query.binding != corpus.binding) return failure(Reason::incompatible_space);
  if (query.values.size() != corpus.binding.dimensions) return failure(Reason::dimension_mismatch);
  if (!finite_values(query.values)) return failure(Reason::nonfinite_value);

  std::vector<double> query_values;
  bool query_undefined = false;
  prepare(query.values, corpus.binding.metric, corpus.binding.normalization, query_values,
          query_undefined);
  if (query_undefined) {
    SearchResult result;
    result.outcome = Outcome::abstained;
    result.reason = Reason::undefined_geometry;
    return result;
  }

  std::vector<Scored> eligible;
  SearchResult result;
  result.reason = Reason::none;
  for (const VectorRecord& record : corpus.records) {
    if (exclusions.exclude_self && text(query.source_id) && record.source_id == query.source_id) {
      result.skipped.push_back({record.source_id, SkipReason::self_match});
      continue;
    }
    if (std::find(exclusions.source_ids.begin(), exclusions.source_ids.end(), record.source_id) !=
        exclusions.source_ids.end()) {
      result.skipped.push_back({record.source_id, SkipReason::excluded_source});
      continue;
    }
    std::vector<double> candidate;
    bool undefined = false;
    prepare(record.values, corpus.binding.metric, corpus.binding.normalization, candidate,
            undefined);
    if (undefined) {
      result.skipped.push_back({record.source_id, SkipReason::undefined_geometry});
      continue;
    }
    Scored scored;
    scored.source_id = record.source_id;
    measure(corpus.binding.metric, query_values, candidate, scored.distance, scored.similarity);
    eligible.push_back(std::move(scored));
  }

  std::sort(eligible.begin(), eligible.end(), [&](const Scored& left, const Scored& right) {
    return better(corpus.binding.metric, left, right);
  });
  const std::size_t keep = std::min(k, eligible.size());
  result.eligible_count = eligible.size();
  result.hits.reserve(keep);
  for (std::size_t index = 0; index < keep; ++index) {
    Hit hit;
    hit.source_id = eligible[index].source_id;
    hit.rank = index + 1;
    hit.distance = eligible[index].distance;
    hit.similarity = eligible[index].similarity;
    result.hits.push_back(std::move(hit));
  }
  if (result.hits.empty()) {
    result.outcome = Outcome::empty;
    return result;
  }
  result.outcome = Outcome::ranked;
  return result;
}

PairwiseResult pairwise(const SpaceBinding& binding, const VectorRecord& left,
                        const VectorRecord& right) {
  PairwiseResult result;
  if (!binding_named(binding, false) || !text(left.source_id) || !text(right.source_id)) {
    result.reason = Reason::invalid_request;
    return result;
  }
  if (left.values.size() != binding.dimensions || right.values.size() != binding.dimensions) {
    result.reason = Reason::dimension_mismatch;
    return result;
  }
  if (!finite_values(left.values) || !finite_values(right.values)) {
    result.reason = Reason::nonfinite_value;
    return result;
  }
  std::vector<double> prepared_left;
  std::vector<double> prepared_right;
  bool left_undefined = false;
  bool right_undefined = false;
  prepare(left.values, binding.metric, binding.normalization, prepared_left, left_undefined);
  prepare(right.values, binding.metric, binding.normalization, prepared_right, right_undefined);
  if (left_undefined || right_undefined) {
    result.outcome = Outcome::abstained;
    result.reason = Reason::undefined_geometry;
    return result;
  }
  measure(binding.metric, prepared_left, prepared_right, result.distance, result.similarity);
  result.outcome = Outcome::ranked;
  result.reason = Reason::none;
  return result;
}

DiagnosticReport neighborhood_diagnostics(const Corpus& corpus, std::size_t k) {
  DiagnosticReport report;
  report.k = k;
  if (k == 0) {
    report.reason = Reason::invalid_request;
    return report;
  }
  const Reason corpus_problem = corpus_reason(corpus);
  if (corpus_problem != Reason::none) {
    report.reason = corpus_problem;
    return report;
  }

  ExclusionPolicy self_only;
  std::unordered_map<std::string, std::size_t> hubs;
  for (const VectorRecord& record : corpus.records) hubs.emplace(record.source_id, 0);

  bool any_ranked = false;
  bool any_abstained = false;
  report.rows.reserve(corpus.records.size());
  for (const VectorRecord& record : corpus.records) {
    Query query;
    query.binding = corpus.binding;
    query.query_id = record.source_id;
    query.source_id = record.source_id;
    query.values = record.values;
    const SearchResult found = search(corpus, query, k, self_only);
    NeighborDiagnostic row;
    row.source_id = record.source_id;
    row.query_outcome = found.outcome;
    row.query_reason = found.reason;
    row.neighbor_count = found.hits.size();
    if (found.outcome == Outcome::ranked) {
      any_ranked = true;
      double total = 0;
      for (const Hit& hit : found.hits) {
        total += hit.distance;
        hubs[hit.source_id] += 1;
      }
      row.mean_neighbor_distance = total / static_cast<double>(found.hits.size());
    } else if (found.outcome == Outcome::abstained) {
      any_abstained = true;
    } else if (found.outcome == Outcome::failed) {
      report.outcome = Outcome::failed;
      report.reason = found.reason;
      report.rows.clear();
      return report;
    }
    report.rows.push_back(std::move(row));
  }

  for (NeighborDiagnostic& row : report.rows) row.hub_count = hubs[row.source_id];
  std::sort(report.rows.begin(), report.rows.end(),
            [](const NeighborDiagnostic& left, const NeighborDiagnostic& right) {
              if (left.hub_count != right.hub_count) return left.hub_count > right.hub_count;
              return left.source_id < right.source_id;
            });
  report.reason = Reason::none;
  if (any_ranked) report.outcome = Outcome::ranked;
  else if (any_abstained) report.outcome = Outcome::abstained;
  else report.outcome = Outcome::empty;
  return report;
}

constexpr std::size_t k_max_constructor_points = 8;

ConstructorResult constructor_failure(Reason reason, ConstructorOutcome outcome = ConstructorOutcome::failed) {
  ConstructorResult result;
  result.outcome = outcome;
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
