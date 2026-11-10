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
  // reconstruct_k ignores this field.
  double early_stop_cosine = 1;
  // Skip the combinatorial face walk after Lawson-Hanson. combine() keeps the
  // walk unless this is set or OR01_COMB_SKIP_FACE is set. reconstruct_k does
  // not call that walk.
  bool skip_face = false;
  // Stop a peel chain when leftover / ||t|| is at most this. The depth is not a
  // requested count. Zero keeps peeling until the cap or no positive column.
  double leftover_floor = 0.2;
  // Safety cap on peel steps in one reconstruction. At least 1.
  std::size_t peel_cap = 16;
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

struct GeometricReconstruction {
  std::string start_source_id;
  double start_similarity = 0;
  double leftover_ratio = 0;
  std::size_t depth = 0;
  CombinationResult combination;
};

struct ReconstructionSet {
  Outcome outcome = Outcome::failed;
  Reason reason = Reason::invalid_request;
  std::vector<GeometricReconstruction> reconstructions;
  std::vector<Hit> nearest;
};

// K geometric retrievals of one point from one prepared pass. Retrieval i
// starts at the i-th nearest source and peels until leftover / ||t|| meets
// leftover_floor or peel_cap. Exact ranking of those anchors is in nearest.
// The registered tool does not call this.
ReconstructionSet reconstruct_k(const Corpus& corpus, const std::vector<double>& target,
                                std::size_t reconstruction_count,
                                const CombinationParameters& parameters);

}  // namespace mityane::operations::retrieval::v1
