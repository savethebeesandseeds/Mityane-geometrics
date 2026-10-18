// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_harness.h"
#include <string>
#include <vector>

namespace embedding::evaluation {

struct ProjectionArchiveInput {
  std::string id, architecture;
  uint64_t master_seed{0};
  int64_t checkpoint_steps{0};
  // Caller declares only frozen TRAINING/VALIDATION inputs; no testing API.
  // Feature rows must retain the source/label archive's original row order.
  std::string training_features, validation_features;
  std::string training_labels, validation_labels;
  std::string feature_values_key{"features"}, feature_valid_key{"valid"};
  std::string feature_provenance_key{"provenance"};
  std::string labels_key{"labels_scoring_only"}, source_ids_key{"source_ids_json"};
};

struct ProjectionRepetition {
  std::string id;
  uint64_t projection_seed{0}, probe_seed{0};
};

struct ProjectionDiagnosticRun {
  std::string output_directory;
  std::string source_fingerprint{"unrecorded"}, git_head{"unrecorded"}, git_dirty{"unrecorded"};
  std::vector<ProjectionArchiveInput> inputs;
  std::vector<int64_t> widths{12,24,36,48};
  std::vector<ProjectionRepetition> repetitions{
      {"rep-1",1701,2701},{"rep-2",1802,2802},{"rep-3",1903,2903}};
  int64_t threads{1}, bootstrap_replicates{1000};
};

// Seeded Gaussian reduced-QR map with orthonormal columns, float64 CPU.
// Local std::mt19937_64 does not consume/reseed Torch CPU/CUDA generators.
struct FrozenOrthonormalProjection {
  torch::Tensor components; // [native_dimensions, projected_dimensions].
  uint64_t seed{0};
  FrozenOrthonormalProjection(int64_t native_dimensions,int64_t projected_dimensions,uint64_t seed);
  FeatureSurface transform(const FeatureSurface &surface) const;
};

// New artifacts only; all inputs guarded byte-for-byte. Outer normalization
// and PCA fit valid TRAINING rows only. Random maps are data independent.
// PCA/random share actual named probe initialization per width/repetition.
// Every declared repetition is reported; no validation-selected best seed.
void run_projection_diagnostic(const ProjectionDiagnosticRun &run);

} // namespace embedding::evaluation
