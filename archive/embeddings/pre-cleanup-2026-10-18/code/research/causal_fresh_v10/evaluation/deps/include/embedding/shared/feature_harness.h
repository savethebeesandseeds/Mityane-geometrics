// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/shared/data.h"
#include "embedding/shared/evaluation.h"
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::evaluation {

struct FeatureSurface {
  torch::Tensor values; // CPU [B,D]; invalid rows are excluded from every fit.
  torch::Tensor valid;  // CPU bool [B].
  std::string provenance;
};

void validate_features(const FeatureSurface &surface);
// Same population-std/effective-covariance-rank semantics as the legacy helper,
// using the smaller Gram matrix when feature width exceeds valid row count.
RepresentationDiagnostics feature_diagnostics(const FeatureSurface &surface);

struct FeatureNormalizer {
  torch::Tensor mean, scale;
  int64_t fitted_rows{0};
  explicit FeatureNormalizer(const FeatureSurface &training);
  FeatureSurface transform(const FeatureSurface &surface) const;
};

struct TrainPca {
  torch::Tensor mean, components, singular_values;
  int64_t fitted_rows{0}, numerical_rank{0};
  TrainPca(const FeatureSurface &training, int64_t dimensions);
  FeatureSurface transform(const FeatureSurface &surface) const;
};

struct RidgeProbe {
  FeatureNormalizer normalizer;
  torch::Tensor weights, intercept;
  RidgeProbe(const FeatureSurface &training, const torch::Tensor &labels,
             double penalty = 1.0);
  torch::Tensor predict(const FeatureSurface &surface) const;
};

// A fixed secondary interaction probe, never selected against test results.
struct TinyProbe {
  FeatureNormalizer normalizer;
  torch::Tensor w1, b1, w2, b2;
  TinyProbe(const FeatureSurface &training, const torch::Tensor &labels,
            uint64_t seed, int64_t steps = 100, int64_t hidden = 16,
            double learning_rate = 0.01);
  torch::Tensor predict(const FeatureSurface &surface) const;
};

struct Score {
  int64_t total{0}, valid{0}, correct{0};
  double accuracy{0}, coverage{0};
  bool supported{false};
};
Score score(const torch::Tensor &predictions, const torch::Tensor &labels,
            const torch::Tensor &valid);

struct GroupedInterval {
  double estimate{0}, lower{0}, upper{0};
  int64_t source_groups{0};
  bool supported{false};
};
// Percentile bootstrap over whole source groups; paired variants stay together.
// With comparison_predictions supplied, estimate a paired accuracy difference.
GroupedInterval grouped_accuracy_interval(const torch::Tensor &predictions,
    const torch::Tensor &labels, const torch::Tensor &valid,
    const std::vector<std::string> &source_ids, uint64_t seed,
    int64_t replicates = 1000,
    const torch::Tensor &comparison_predictions = {});

enum class Task { reversal, level, amplitude, lag_sign };
std::string task_name(Task task);
uint64_t stream_seed(uint64_t seed, uint64_t stream);

struct ControlledDataset {
  Batch clean, observed; // CPU float64; preparation must precede float32 casting.
  torch::Tensor labels;  // CPU int64 [B], both variants of every source present.
  std::vector<std::string> source_ids;
};
struct ControlledProtocol {
  Task task;
  uint64_t seed{0};
  input_shape_t shape;
  ControlledDataset training, validation, testing;
};

struct CoordinateDeletionView {
  Batch observations;
  torch::Tensor requested_erasure; // CPU bool BCHF, including already absent cells.
};

// Pure, role-neutral corruption: no labels, fitting, split discovery or Torch
// RNG. A source's paired rows share one C*H*F high53-bit mt19937_64 stream;
// rates are nested. Retained values are unchanged and hidden storage is zero.
// The caller freezes the seed/namespace and binds the observation role. The
// default namespace preserves the historical fixed-readout stress mask stream.
CoordinateDeletionView make_coordinate_deletion_view(const Batch &observations,
    const std::vector<std::string> &source_ids, Task task, uint64_t seed,
    double rate, const std::string &rng_namespace = "fixed-readout-stress-v1");

// Source groups are assigned before their paired observations/windows are drawn.
ControlledProtocol make_controlled_protocol(Task task, const input_shape_t &shape,
    int64_t training_pairs, int64_t validation_pairs, int64_t testing_pairs,
    uint64_t seed, double missing_rate = 0.1);
// Development generation never draws testing sources: testing tensors remain
// undefined and its source IDs empty. Validation is reserved for selection.
ControlledProtocol make_controlled_development_protocol(Task task, const input_shape_t &shape,
    int64_t training_pairs, int64_t validation_pairs,
    uint64_t seed, double missing_rate = 0.1);
// Generate only a final testing cohort, after selection has been frozen. The
// caller supplies the separately declared fresh test seed (for the learning
// curve: stream_seed(development_seed, 0x6375727665746573ULL)).
ControlledDataset make_controlled_test_dataset(Task task, const input_shape_t &shape,
    int64_t testing_pairs, uint64_t fresh_seed, double missing_rate = 0.1);
// The historical validator continues to require all three nonempty splits;
// it deliberately rejects a development-only partial protocol.
void validate_protocol(const ControlledProtocol &protocol);

struct OracleResult { torch::Tensor predictions, valid; };
// Legal observations only; no clean signal or supervised labels are available.
OracleResult raw_oracle(Task task, const Batch &observations);
OracleResult raw_oracle(Task task, const ControlledDataset &dataset);

// Per-channel/feature statistics for generic control preparation. This is not
// the legacy evaluator's [F] scaler and must never replace it for old weights.
struct ObservationScaler {
  torch::Tensor mean, scale, counts;
  explicit ObservationScaler(const Batch &training);
  Batch transform(const Batch &batch) const;
};

} // namespace embedding::evaluation
