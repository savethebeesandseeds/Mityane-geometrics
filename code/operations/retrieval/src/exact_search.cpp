#include "mityane/operations/retrieval/exact_search.h"

#include "numeric.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace mityane::operations::retrieval::v1 {
namespace {

using detail::binding_named;
using detail::finite_values;
using detail::prepare;
using detail::text;

static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53 && sizeof(double) == 8,
              "The reference implementation requires IEEE 754 binary64 doubles.");

SearchResult failure(Reason reason) {
  SearchResult result;
  result.outcome = Outcome::failed;
  result.reason = reason;
  return result;
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


}  // namespace mityane::operations::retrieval::v1
