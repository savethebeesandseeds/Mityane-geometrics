#include "mityane/operations/retrieval/combination.h"

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
using detail::squared_norm;
using detail::text;

constexpr double kSolverZero = 1e-10;

struct PreparedColumn {
  std::string source_id;
  std::vector<double> values;
  double norm = 0;
};

CombinationResult failed(Reason reason) {
  CombinationResult result;
  result.outcome = Outcome::failed;
  result.reason = reason;
  return result;
}

CombinationResult finished(Outcome outcome, Reason reason) {
  CombinationResult result;
  result.outcome = outcome;
  result.reason = reason;
  return result;
}

double dot(const std::vector<double>& left, const std::vector<double>& right) {
  double sum = 0;
  for (std::size_t index = 0; index < left.size(); ++index) sum += left[index] * right[index];
  return sum;
}

double norm_of(const std::vector<double>& values) { return std::sqrt(squared_norm(values)); }

bool parameter_ok(const CombinationParameters& parameters) {
  return std::isfinite(parameters.minimum_weight) && parameters.minimum_weight >= 0 &&
         std::isfinite(parameters.fit_floor) && parameters.fit_floor >= 0 && parameters.fit_floor <= 1 &&
         std::isfinite(parameters.directness_ceiling) && parameters.directness_ceiling >= 1 &&
         std::isfinite(parameters.early_stop_cosine) && parameters.early_stop_cosine >= 0 &&
         parameters.early_stop_cosine <= 1;
}

Reason corpus_problem(const Corpus& corpus) {
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

// Unconstrained least squares. A false return means the columns are dependent.
bool least_squares(const std::vector<const PreparedColumn*>& columns, const std::vector<double>& target,
                   std::vector<double>& weights) {
  const std::size_t count = columns.size();
  weights.assign(count, 0);
  if (count == 0) return true;
  std::vector<double> matrix(count * (count + 1), 0);
  double scale = 0;
  for (std::size_t row = 0; row < count; ++row) {
    for (std::size_t column = 0; column < count; ++column) {
      matrix[row * (count + 1) + column] = dot(columns[row]->values, columns[column]->values);
    }
    matrix[row * (count + 1) + count] = dot(columns[row]->values, target);
    scale = std::max(scale, std::abs(matrix[row * (count + 1) + row]));
  }
  const double pivot_floor = kSolverZero * std::max(1.0, scale);
  for (std::size_t column = 0; column < count; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < count; ++row) {
      if (std::abs(matrix[row * (count + 1) + column]) > std::abs(matrix[pivot * (count + 1) + column])) pivot = row;
    }
    if (std::abs(matrix[pivot * (count + 1) + column]) <= pivot_floor) return false;
    if (pivot != column) std::swap_ranges(matrix.begin() + static_cast<std::ptrdiff_t>(pivot * (count + 1)),
                                          matrix.begin() + static_cast<std::ptrdiff_t>((pivot + 1) * (count + 1)),
                                          matrix.begin() + static_cast<std::ptrdiff_t>(column * (count + 1)));
    const double diagonal = matrix[column * (count + 1) + column];
    for (std::size_t row = column + 1; row < count; ++row) {
      const double factor = matrix[row * (count + 1) + column] / diagonal;
      for (std::size_t entry = column; entry <= count; ++entry) {
        matrix[row * (count + 1) + entry] -= factor * matrix[column * (count + 1) + entry];
      }
    }
  }
  for (std::size_t row = count; row-- > 0;) {
    double value = matrix[row * (count + 1) + count];
    for (std::size_t column = row + 1; column < count; ++column) value -= matrix[row * (count + 1) + column] * weights[column];
    weights[row] = value / matrix[row * (count + 1) + row];
  }
  return true;
}

std::vector<double> residual_of(const std::vector<PreparedColumn>& columns, const std::vector<double>& weights,
                                const std::vector<double>& target) {
  std::vector<double> residual = target;
  for (std::size_t index = 0; index < columns.size(); ++index) {
    for (std::size_t axis = 0; axis < residual.size(); ++axis) {
      residual[axis] -= weights[index] * columns[index].values[axis];
    }
  }
  return residual;
}

bool represents(const std::vector<PreparedColumn>& columns, const std::vector<std::size_t>& subset,
                const std::vector<double>& projection, std::vector<double>& weights) {
  std::vector<const PreparedColumn*> chosen;
  chosen.reserve(subset.size());
  for (const std::size_t index : subset) chosen.push_back(&columns[index]);
  std::vector<double> solved;
  if (!least_squares(chosen, projection, solved)) return false;
  for (const double weight : solved) {
    if (!(weight > kSolverZero)) return false;
  }
  std::vector<double> gap = projection;
  for (std::size_t index = 0; index < subset.size(); ++index) {
    for (std::size_t axis = 0; axis < gap.size(); ++axis) gap[axis] -= solved[index] * columns[subset[index]].values[axis];
  }
  const double allowance = kSolverZero * std::max(1.0, norm_of(projection));
  if (norm_of(gap) > allowance) return false;
  weights.assign(columns.size(), 0);
  for (std::size_t index = 0; index < subset.size(); ++index) weights[subset[index]] = solved[index];
  return true;
}

// Lawson-Hanson gives one optimal projection. The weight vector for that
// projection can still be chosen in more than one way, so the face of columns
// with a zero gradient is searched from the smallest supports upward.
std::vector<double> non_negative_weights(const std::vector<PreparedColumn>& columns, const std::vector<double>& target) {
  const std::size_t count = columns.size();
  std::vector<double> weights(count, 0);
  if (count == 0) return weights;
  std::vector<bool> passive(count, false);
  std::vector<bool> blocked(count, false);
  const std::size_t limit = count * count + 4;
  for (std::size_t step = 0; step < limit; ++step) {
    const std::vector<double> residual = residual_of(columns, weights, target);
    std::size_t entering = count;
    double best = 0;
    for (std::size_t index = 0; index < count; ++index) {
      if (passive[index] || blocked[index]) continue;
      const double gradient = dot(columns[index].values, residual);
      if (!(gradient > kSolverZero)) continue;
      if (entering == count || gradient > best + kSolverZero ||
          (std::abs(gradient - best) <= kSolverZero && columns[index].source_id < columns[entering].source_id)) {
        best = gradient;
        entering = index;
      }
    }
    if (entering == count) break;
    passive[entering] = true;
    for (std::size_t inner = 0; inner < limit; ++inner) {
      std::vector<const PreparedColumn*> active;
      std::vector<std::size_t> active_index;
      for (std::size_t index = 0; index < count; ++index) {
        if (!passive[index]) continue;
        active.push_back(&columns[index]);
        active_index.push_back(index);
      }
      std::vector<double> solved;
      if (!least_squares(active, target, solved)) {
        passive[entering] = false;
        blocked[entering] = true;
        break;
      }
      bool positive = true;
      for (const double weight : solved) positive = positive && weight > kSolverZero;
      if (positive) {
        std::fill(weights.begin(), weights.end(), 0);
        for (std::size_t index = 0; index < active_index.size(); ++index) weights[active_index[index]] = solved[index];
        break;
      }
      double alpha = 1;
      for (std::size_t index = 0; index < active_index.size(); ++index) {
        if (solved[index] > kSolverZero) continue;
        const double current = weights[active_index[index]];
        const double candidate = current / (current - solved[index]);
        if (candidate < alpha) alpha = candidate;
      }
      for (std::size_t index = 0; index < active_index.size(); ++index) {
        const double mixed = weights[active_index[index]] + alpha * (solved[index] - weights[active_index[index]]);
        weights[active_index[index]] = mixed > kSolverZero ? mixed : 0;
        if (weights[active_index[index]] == 0) passive[active_index[index]] = false;
      }
    }
    if (!passive[entering]) blocked[entering] = true;
  }

  const std::vector<double> projection_residual = residual_of(columns, weights, target);
  std::vector<double> projection = target;
  for (std::size_t axis = 0; axis < projection.size(); ++axis) projection[axis] -= projection_residual[axis];
  std::vector<std::size_t> support;
  for (std::size_t index = 0; index < count; ++index) {
    if (weights[index] > kSolverZero) support.push_back(index);
  }
  if (support.empty()) return std::vector<double>(count, 0);

  std::vector<std::size_t> face = support;
  const double residual_norm = norm_of(projection_residual);
  const double projection_norm = norm_of(projection);
  for (std::size_t index = 0; index < count; ++index) {
    if (std::find(face.begin(), face.end(), index) != face.end()) continue;
    const bool on_face = residual_norm <= kSolverZero * std::max(1.0, norm_of(target))
                             ? dot(columns[index].values, projection) > kSolverZero * std::max(1.0, projection_norm)
                             : dot(columns[index].values, projection_residual) >= -kSolverZero * columns[index].norm * std::max(1.0, residual_norm);
    if (on_face) face.push_back(index);
  }
  std::sort(face.begin(), face.end(), [&columns](std::size_t left, std::size_t right) {
    return columns[left].source_id < columns[right].source_id;
  });

  const std::size_t widest = support.size();
  for (std::size_t width = 1; width <= widest && width <= face.size(); ++width) {
    std::vector<std::size_t> choice(width);
    for (std::size_t index = 0; index < width; ++index) choice[index] = index;
    while (true) {
      std::vector<std::size_t> subset;
      subset.reserve(width);
      for (const std::size_t offset : choice) subset.push_back(face[offset]);
      std::vector<double> sparse;
      if (represents(columns, subset, projection, sparse)) return sparse;
      std::size_t cursor = width;
      while (cursor > 0 && choice[cursor - 1] == face.size() - width + (cursor - 1)) --cursor;
      if (cursor == 0) break;
      ++choice[cursor - 1];
      for (std::size_t index = cursor; index < width; ++index) choice[index] = choice[index - 1] + 1;
    }
  }
  return weights;
}

void order_hits(CombinationResult& result) {
  std::sort(result.hits.begin(), result.hits.end(), [](const CombinationHit& left, const CombinationHit& right) {
    if (left.weight != right.weight) return left.weight > right.weight;
    return left.source_id < right.source_id;
  });
  for (std::size_t index = 0; index < result.hits.size(); ++index) result.hits[index].rank = index + 1;
}

CombinationResult from_weights(const std::vector<PreparedColumn>& columns, const std::vector<double>& weights,
                               const std::vector<double>& target, const CombinationParameters& parameters) {
  std::vector<double> reconstruction(target.size(), 0);
  double path = 0;
  bool positive = false;
  for (std::size_t index = 0; index < columns.size(); ++index) {
    if (!(weights[index] > kSolverZero)) continue;
    positive = true;
    path += weights[index] * columns[index].norm;
    for (std::size_t axis = 0; axis < reconstruction.size(); ++axis) {
      reconstruction[axis] += weights[index] * columns[index].values[axis];
    }
  }
  CombinationResult result = finished(Outcome::ranked, Reason::none);
  if (!positive) return finished(Outcome::empty, Reason::outside_cone);
  const double target_norm = norm_of(target);
  const double net = norm_of(reconstruction);
  result.reconstruction_cosine = net / target_norm;
  result.path_ratio = net > 0 ? path / net : std::numeric_limits<double>::infinity();
  if (result.reconstruction_cosine < parameters.fit_floor) {
    result.outcome = Outcome::empty;
    result.reason = Reason::fit;
    return result;
  }
  if (result.path_ratio > parameters.directness_ceiling) {
    result.outcome = Outcome::empty;
    result.reason = Reason::directness;
    return result;
  }
  for (std::size_t index = 0; index < columns.size(); ++index) {
    if (!(weights[index] > kSolverZero)) continue;
    result.hits.push_back({columns[index].source_id, 0, weights[index]});
  }
  order_hits(result);
  return result;
}

}  // namespace

CombinationResult combine(const Corpus& corpus, const std::vector<double>& target,
                          const CombinationParameters& parameters) {
  if (!parameter_ok(parameters)) return failed(Reason::invalid_request);
  const Reason problem = corpus_problem(corpus);
  if (problem != Reason::none) return failed(problem);
  if (target.size() != corpus.binding.dimensions) return failed(Reason::dimension_mismatch);
  if (!finite_values(target)) return failed(Reason::nonfinite_value);
  if (squared_norm(target) == 0) return finished(Outcome::abstained, Reason::undefined_geometry);

  bool target_undefined = false;
  std::vector<double> prepared_target;
  prepare(target, corpus.binding.metric, corpus.binding.normalization, prepared_target, target_undefined);
  if (target_undefined || squared_norm(prepared_target) == 0) return finished(Outcome::abstained, Reason::undefined_geometry);

  std::vector<PreparedColumn> columns;
  for (const VectorRecord& record : corpus.records) {
    bool undefined = false;
    PreparedColumn column;
    prepare(record.values, corpus.binding.metric, corpus.binding.normalization, column.values, undefined);
    if (undefined || squared_norm(column.values) == 0) continue;
    column.source_id = record.source_id;
    column.norm = norm_of(column.values);
    columns.push_back(std::move(column));
  }

  Query probe;
  probe.binding = corpus.binding;
  probe.query_id = "combination-target";
  probe.values = target;
  const SearchResult nearest = search(corpus, probe, 1, {});
  if (nearest.outcome == Outcome::failed) return failed(nearest.reason);
  if (nearest.outcome == Outcome::ranked && !nearest.hits.empty()) {
    const PreparedColumn* winner = nullptr;
    for (const PreparedColumn& column : columns) {
      if (column.source_id == nearest.hits[0].source_id) winner = &column;
    }
    if (winner != nullptr) {
      const double cosine = dot(winner->values, prepared_target) / (winner->norm * norm_of(prepared_target));
      const double weight = std::max(0.0, dot(winner->values, prepared_target)) / squared_norm(winner->values);
      if (cosine >= parameters.early_stop_cosine && weight > kSolverZero) {
        CombinationResult result = finished(Outcome::ranked, Reason::none);
        result.hits.push_back({winner->source_id, 1, weight});
        std::vector<double> reconstruction(prepared_target.size(), 0);
        for (std::size_t axis = 0; axis < reconstruction.size(); ++axis) reconstruction[axis] = weight * winner->values[axis];
        result.reconstruction_cosine = norm_of(reconstruction) / norm_of(prepared_target);
        result.path_ratio = 1;
        return result;
      }
    }
  }

  std::vector<double> weights = non_negative_weights(columns, prepared_target);
  for (;;) {
    std::vector<PreparedColumn> kept;
    for (std::size_t index = 0; index < columns.size(); ++index) {
      if (weights[index] < parameters.minimum_weight || !(weights[index] > kSolverZero)) continue;
      kept.push_back(columns[index]);
    }
    if (kept.size() == columns.size()) break;
    columns = std::move(kept);
    weights = non_negative_weights(columns, prepared_target);
  }
  return from_weights(columns, weights, prepared_target, parameters);
}

}  // namespace mityane::operations::retrieval::v1
