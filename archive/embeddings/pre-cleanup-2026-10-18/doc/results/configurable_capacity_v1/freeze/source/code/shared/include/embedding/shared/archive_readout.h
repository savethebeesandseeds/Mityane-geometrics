// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/feature_harness.h"
#include <string>
#include <vector>

namespace embedding::evaluation {

struct ArchiveReadoutValidationView {
  std::string id, observations, features;
  std::string observations_sha256, features_sha256;
  std::string expected_feature_provenance, corruption_provenance;
};

struct ArchiveReadoutInput {
  std::string id, tag, task;
  uint64_t master_seed{0};
  int64_t checkpoint_steps{0};
  std::string producer_source_fingerprint, cohort_provenance;
  // Explicit TRAIN/VALIDATION roles only. No directory discovery or TEST API.
  std::string training_observations, validation_observations;
  std::string training_features, validation_features;
  std::string observations_key{"observed"}, observation_mask_key{"feature_mask"};
  std::string labels_key{"labels_scoring_only"}, source_ids_key{"source_ids_json"};
  std::string feature_values_key{"features"}, feature_valid_key{"valid"};
  std::string feature_provenance_key{"provenance"};
  // Audited adapters can bind row association to the exact producer archives.
  // Empty expected hashes permit generic dummy fixtures; all actual hashes are
  // always recorded and exact input-file bytes are checked before/after fitting.
  std::string training_observations_sha256, validation_observations_sha256;
  std::string training_features_sha256, validation_features_sha256;
  std::string expected_feature_provenance;
  // Optional declared VALIDATION views, never fitted. Rows/labels/source IDs
  // must exactly match ordinary VALIDATION; support can only be deleted,
  // retained values must be identical and hidden observation storage zero.
  // Every view uses this input's unchanged ordinary-TRAIN maps and readouts.
  std::vector<ArchiveReadoutValidationView> validation_views;
};

struct ArchiveReadoutRepetition {
  std::string id;
  uint64_t probe_seed{0};
};

struct ArchiveReadoutRun {
  std::string output_directory;
  std::string source_fingerprint{"unrecorded"};
  std::string git_head{"unrecorded"}, git_dirty{"unrecorded"};
  input_shape_t shape{3,32,3,torch::kFloat64,torch::kCPU};
  int64_t compact_width{32};
  int64_t threads{1}, bootstrap_replicates{1000};
  double ridge_penalty{1.0};
  int64_t tiny_hidden{16}, tiny_steps{100};
  double tiny_learning_rate{0.01};
  std::vector<ArchiveReadoutRepetition> repetitions{
      {"rep-1",2701},{"rep-2",2802},{"rep-3",2903}};
  std::vector<ArchiveReadoutInput> inputs;
};

// Frozen card precedes all deserialization, fits and scores. ObservationScaler
// fits observed TRAIN per-channel/feature statistics across rows/history in
// float64 (existing 1e-8 scale floor); missing standardized values are zero.
// Raw = flattened values followed by visibility flags. Standalone PCA fits raw
// TRAIN only. Native archived exports retain their exact dimensions, with no
// PCA after the encoder. Probe weights fit separately on TRAIN; equal-width
// methods share stream_seed(repetition.probe_seed,width). Every repetition and
// unsupported fit is retained. Raw/PCA support is any observed coordinate;
// native support may be a subset but never includes an all-missing row. Each
// method fits independently on its valid TRAIN rows. Validation effects use
// common-support paired source-group intervals. Only NEW outputs are allowed.
void run_archive_readout(const ArchiveReadoutRun &run);

} // namespace embedding::evaluation
