// SPDX-License-Identifier: MIT
#pragma once

#include <torch/torch.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace embedding::evaluation::configurable_capacity {
inline constexpr const char *kProtocolId = "configurable-capacity-tempo5-v1";
inline constexpr const char *kDatasetId = "TEMPO-5";
inline constexpr int64_t kDesignedComplexity = 5, kComplexityMaximum = 5;
inline constexpr uint64_t kTaskStream = 0x74656d706f352d31ULL;
inline constexpr uint64_t kSplitStream = 0x6361702d73706c31ULL;
inline constexpr uint64_t kSignalStream = 0x6361702d73696731ULL;
inline constexpr uint64_t kDistractorStream = 0x6361702d64697331ULL;
inline constexpr uint64_t kAffineStream = 0x6361702d61666631ULL;
inline constexpr uint64_t kNoiseStream = 0x6361702d6e6f6931ULL;
inline constexpr uint64_t kMaskStream = 0x6361702d6d617331ULL;
inline constexpr uint64_t kGapStream = 0x6361702d67617031ULL;
inline constexpr uint64_t kOrderStream = 0x6361702d6f726431ULL;
inline constexpr uint64_t kDeletionStream = 0x6361702d64656c31ULL;

// This benchmark has three semantic channels and nine feature roles. History
// is configurable; its periods/delays always remain in original candle units.
struct Dimensions { int64_t channel_count = 3, history_length = 128, input_width = 9; };
struct Split {
  torch::Tensor observed;       // contiguous CPU float64 [N,3,H,9], hidden zero.
  torch::Tensor mask;           // contiguous CPU Bool, true=observed.
  torch::Tensor scoring_labels; // contiguous CPU Long [N], never model inputs.
  torch::Tensor source_ids;     // contiguous CPU Long [N], two rows per source.
};
struct Development { Split training, validation; };
void validate_dimensions(const Dimensions &dimensions);
void validate(const Split &source);
// Source assignment precedes paired variants. master<2^32, total sources<2^30;
// IDs=(master<<31)|(original_source_index<<1), namespaced as TEMPO-5 in JSON.
Development make_development(std::size_t training_pairs, std::size_t validation_pairs,
    uint64_t master, Dimensions dimensions = {});
Split delete_observations(const Split &source, double probability, uint64_t seed);
std::string source_ids_json(const Split &source);
// Immutable no-replace publication for this benchmark's observation/information
// evidence. An existing file, directory or dangling symlink is preserved.
void publish_archive(const std::string &path,torch::serialize::OutputArchive &archive);
void save_split_archive(const std::string &path, const Split &source, const std::string &role,
    uint64_t master, double additional_deletion = 0);
Split load_split_archive(const std::string &path);

inline constexpr const char *kInformationRule = "observed-tempo5-integer-two-band-fit-v1";
inline constexpr int kInformationMinimumFeatureTicks = 24;
inline constexpr int kInformationMinimumFeaturesPerChannel = 2;
inline constexpr int kInformationMinimumChannelUnionTicks = 40;
inline constexpr int kInformationMinimumChannelSpan = 48;
inline constexpr double kInformationScaledPivotFloor = 1e-8;
inline constexpr double kInformationEnergyFloor = 1e-12;
inline constexpr double kInformationMinimumComponentEnergyFraction = .02;
inline constexpr double kInformationMinimumPhaseCoherence = .95;
inline constexpr double kInformationMaximumNormalizedResidual = .0025;
inline constexpr double kInformationMinimumSlowPhaseMargin = .12;
inline constexpr double kInformationMaximumSlowPhaseMargin = .60;
struct InformationDecision {
  bool supported = false;
  int64_t predicted_label = 0, frequency_pairs_evaluated = 0;
  double slow_frequency = 0, fast_frequency = 0, normalized_residual = 0, margin = 0;
  // Rows c0f0..2,c1f0..2; columns constant,slow sine/cosine,fast sine/cosine.
  std::array<std::array<double,5>,6> coefficients{};
  std::array<bool,6> eligible{};
  std::string reason;
};
// No labels, source IDs, true frequencies, delays or hidden signals enter.
// Fit all 17x7 integer-period candidates with independent positive-affine
// nuisance coefficients per observed feature; retain unsupported populations.
std::vector<InformationDecision> infer_information(const torch::Tensor &observed,
    const torch::Tensor &mask);
namespace engineering {
// Engineering-only noise-free counterpart; never used for quality scores.
Development make_noiseless_development(std::size_t training_pairs,
    std::size_t validation_pairs, uint64_t master, Dimensions dimensions = {});
}
} // namespace embedding::evaluation::configurable_capacity
