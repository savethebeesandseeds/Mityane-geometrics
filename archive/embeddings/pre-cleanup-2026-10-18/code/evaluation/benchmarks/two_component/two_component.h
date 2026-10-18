// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <torch/torch.h>

namespace embedding::evaluation::two_component {

inline constexpr const char *kProtocolId = "two-component-v1";
inline constexpr int64_t kDesignedComplexity = 5;
inline constexpr int64_t kComplexityMaximum = 5;
inline constexpr uint64_t kSlowLagTaskStream = 0x7463342d6c616731ULL;
inline constexpr uint64_t kBalanceTaskStream = 0x616d70322d62616cULL;
inline constexpr uint64_t kSplitStream = 0x74632d73706c6974ULL;
inline constexpr uint64_t kSignalStream = 0x74632d7369676e6cULL;
inline constexpr uint64_t kDistractorStream = 0x74632d6469737472ULL;
inline constexpr uint64_t kAffineStream = 0x74632d616666696eULL;
inline constexpr uint64_t kNoiseStream = 0x74632d6e6f697365ULL;
inline constexpr uint64_t kMaskStream = 0x74632d6d61736b31ULL;
inline constexpr uint64_t kGapStream = 0x74632d6761707331ULL;
inline constexpr uint64_t kOrderStream = 0x74632d6f72646572ULL;
inline constexpr uint64_t kDeletionStream = 0x74632d64656c7631ULL;

enum class Task { SlowLagSign, ComponentBalance };
struct Split {
  torch::Tensor observed;       // contiguous CPU F64 [N,3,32,3], hidden exactly0.
  torch::Tensor mask;           // contiguous CPU Bool [N,3,32,3], true=observed.
  torch::Tensor scoring_labels; // contiguous CPU Long [N], never model inputs.
  torch::Tensor source_ids;     // contiguous CPU Long [N], paired identical IDs.
};
struct Development { Split training, validation; };

// master<2^32 and 0<train_pairs,0<validation_pairs, total_sources<2^30.
// ID=(master<<31)|(original_source_index<<1)|task_bit, with timing0/balance1.
// This is injective across the admitted seeds/tasks/sources. Source assignment
// happens before paired variants. No hidden waveform or latent parameter exits.
Development make_development(std::size_t train_pairs, std::size_t validation_pairs,
    std::uint64_t master, Task task);
// A pure source-keyed pair-shared view; no fitting, labels or Torch RNG draws.
// Preserve every row, label, source ID and retained observed value.
Split delete_observations(const Split &source, double probability, std::uint64_t seed);
std::string task_name(Task task);
std::string dataset_id(Task task);

namespace engineering {
// Same draws, phases, gain/offsets, source split, labels and masks as the quality
// factory, but noise multiplier0. Only legal observations are returned. This
// explicit fixture API is not the quality recipe; it exposes no hidden truth.
Development make_noiseless_development(std::size_t train_pairs, std::size_t validation_pairs,
    std::uint64_t master, Task task);
} // namespace engineering
} // namespace embedding::evaluation::two_component
