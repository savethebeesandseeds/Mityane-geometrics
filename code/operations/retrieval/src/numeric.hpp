#pragma once

// Shared preparation for the ranker and the geometric constructors.
// This header is internal to the retrieval library.

#include "mityane/operations/retrieval/exact_search.h"

#include <cmath>
#include <string>
#include <vector>

namespace mityane::operations::retrieval::v1::detail {

inline bool text(const std::string& value) {
  return value.find_first_not_of(" \t\r\n") != std::string::npos;
}

inline bool finite_values(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}

inline double squared_norm(const std::vector<double>& values) {
  double sum = 0;
  for (double value : values) sum += value * value;
  return sum;
}

inline void prepare(const std::vector<double>& values, Metric metric, Normalization normalization,
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

inline bool binding_named(const SpaceBinding& binding, bool require_snapshot) {
  if (!text(binding.space_id) || binding.dimensions == 0) return false;
  if (require_snapshot && !text(binding.snapshot_id)) return false;
  return binding.metric == Metric::cosine || binding.metric == Metric::l2;
}

}  // namespace mityane::operations::retrieval::v1::detail
