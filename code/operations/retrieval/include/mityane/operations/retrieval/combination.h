#pragma once

// Positive sparse combination, OR01-COMB. The registered tool does not call this.
// A target coordinate is approximated by non-negative weights on a frozen corpus.
// Encoding stays upstream. Weights are fit coefficients, not a relevance rule.

#include "mityane/operations/retrieval/exact_search.h"

#include <string>
#include <vector>

namespace mityane::operations::retrieval::v1 {

struct CombinationParameters {
  // Columns below this weight are dropped and the solve is refit. Zero keeps
  // the plain non-negative least-squares solution.
  double minimum_weight = 0;
  // Reconstruction cosine ||Xw|| / ||t|| must meet this floor.
  double fit_floor = 0;
  // Path length divided by ||Xw|| must not exceed this ceiling. At least 1.
  double directness_ceiling = 1;
  // Rank-1 exact-search cosine at or above this value returns that neighbor
  // before the solve. The gates below apply only to a solved combination.
  double early_stop_cosine = 1;
};

struct CombinationHit {
  std::string source_id;
  std::size_t rank = 0;
  double weight = 0;
};

struct CombinationResult {
  Outcome outcome = Outcome::failed;
  Reason reason = Reason::invalid_request;
  std::vector<CombinationHit> hits;
  double reconstruction_cosine = 0;
  double path_ratio = 0;
};

// Prepare the target and the corpus with the same rule as exact search.
// Among optimal basic solutions, keep the fewest positive weights, then the
// lexicographically smallest ascending source-ID list.
CombinationResult combine(const Corpus& corpus, const std::vector<double>& target,
                          const CombinationParameters& parameters);

}  // namespace mityane::operations::retrieval::v1
