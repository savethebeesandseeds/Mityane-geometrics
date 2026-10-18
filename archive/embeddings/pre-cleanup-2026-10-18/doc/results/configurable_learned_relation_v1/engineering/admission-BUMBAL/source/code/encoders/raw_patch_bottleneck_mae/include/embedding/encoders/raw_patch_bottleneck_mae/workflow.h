// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/model.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/preprocessing.h"
#include <cstdint>
#include <string>

namespace embedding::encoders::raw_patch_bottleneck_mae {

inline constexpr const char *kEncoderId = "raw_patch_bottleneck_mae_v1";
inline constexpr const char *kOutputSemantics = "local_observed_global_equal_valid_v1";
inline constexpr const char *kEarlyMixerArchitectureId = "aligned-mixer-before-temporal-v1";
inline constexpr const char *kLateMixerArchitectureId = "aligned-mixer-after-temporal-v1";
// Prototype technical literals; no model-version tag or protocol is reserved.
inline constexpr const char *kPooledContextArchitectureId = "aligned-mixer-before-temporal-pooled-context-width-v1";
inline constexpr const char *kVisibleDifferenceArchitectureId = "aligned-mixer-before-temporal-visible-first-difference-v1";
inline constexpr const char *kVisibleDifferenceInputSemantics =
    "actual_visible_original_adjacent_backward_difference_and_pair_visibility_additive_projection_v1";

inline const char *visible_difference_input_semantics(const Config &config) {
  return config.temporal_difference_input == 1 ? kVisibleDifferenceInputSemantics :
      "raw_visible_values_and_visibility_v1";
}

inline const char *architecture_id(const Config &config) {
  if (config.temporal_difference_input == 1) return kVisibleDifferenceArchitectureId;
  if (config.global_pool_input_source == 1) return kPooledContextArchitectureId;
  return config.channel_mixer_placement == 1 ? kEarlyMixerArchitectureId : kLateMixerArchitectureId;
}

inline const char *global_pool_input_semantics(const Config &config) {
  return config.global_pool_input_source == 1 ?
      "semantic_ordered_temporally_pooled_W_before_diagnostic_D_projection_v1" :
      "semantic_ordered_projected_D_channel_summaries_v1";
}

inline const char *output_semantics(const Config &config) {
  if (config.temporal_difference_input == 1)
    return "local_observed_visible_first_difference_contextual_pretemporal_aligned_global_semantic_mlp_bottleneck_v1";
  if (config.global_pool_input_source == 1)
    return "local_diagnostic_observed_contextual_pretemporal_aligned_pooled_W_global_semantic_mlp_bottleneck_v1";
  if (config.channel_mixer_placement == 1)
    return "local_observed_contextual_pretemporal_aligned_global_semantic_mlp_bottleneck_v1";
  if (config.global_bottleneck_mode == 3)
    return config.channel_mixer_layers > 0 ?
        "local_diagnostic_observed_contextual_aligned_patch_state_global_semantic_mlp_bottleneck_v1" :
        "local_diagnostic_observed_patch_state_global_semantic_mlp_bottleneck_v1";
  if (config.global_bottleneck_mode == 1)
    return config.channel_mixer_layers > 0 ?
        "local_observed_contextual_aligned_global_mean_bottleneck_v1" :
        "local_observed_global_mean_bottleneck_v1";
  if (config.global_bottleneck_mode == 2)
    return config.channel_mixer_layers > 0 ?
        "local_observed_contextual_aligned_global_semantic_mlp_bottleneck_v1" :
        "local_observed_global_semantic_mlp_bottleneck_v1";
  return config.channel_mixer_layers > 0 ?
      "local_observed_contextual_aligned_global_equal_valid_v1" : kOutputSemantics;
}

inline const char *reconstruction_output_semantics(const Config &config) {
  if (config.temporal_difference_input == 1)
    return "exact_visible_first_difference_pretemporal_contextual_observed_global_semantic_mlp_export_v1";
  if (config.global_pool_input_source == 1)
    return "exact_pretemporal_contextual_observed_pooled_W_global_semantic_mlp_export_v1";
  if (config.channel_mixer_placement == 1)
    return "exact_pretemporal_contextual_observed_global_semantic_mlp_export_v1";
  if (config.global_bottleneck_mode == 3)
    return config.channel_mixer_layers > 0 ?
        "exact_contextual_observed_patch_state_global_semantic_mlp_export_v1" :
        "exact_observed_patch_state_global_semantic_mlp_export_v1";
  if (config.global_bottleneck_mode == 1)
    return config.channel_mixer_layers > 0 ?
        "exact_contextual_observed_global_mean_export_v1" :
        "exact_observed_global_mean_export_v1";
  if (config.global_bottleneck_mode == 2)
    return config.channel_mixer_layers > 0 ?
        "exact_contextual_observed_global_semantic_mlp_export_v1" :
        "exact_observed_global_semantic_mlp_export_v1";
  return config.channel_mixer_layers > 0 ?
      "exact_contextual_observed_aligned_export_v1" : "exact_local_observed_export_v1";
}

inline const char *global_readout_description(const Config &config, bool contextual) {
  if (config.temporal_difference_input == 1)
    return contextual ?
        "at least one channel has visible observations; raw and legal original-adjacent first-difference projected states mix before temporal blocks; contextual D summaries and semantic support bits through the learned global MLP" :
        "at least one channel has visible observations; raw and legal original-adjacent first-difference projected states through independent temporal blocks; local D summaries and semantic support bits through the learned global MLP";
  if (config.global_pool_input_source == 1)
    return contextual ?
        "at least one channel has visible observations; early contextual learned temporal W summaries and configured-semantic support bits through the global MLP before diagnostic D projection" :
        "at least one channel has visible observations; independent learned temporal W summaries and configured-semantic support bits through the global MLP before diagnostic D projection";
  if (config.channel_mixer_placement == 1 && contextual)
    return "at least one channel has visible observations; original-patch-aligned projected states mix across channels before temporal blocks; semantic-ordered contextual vectors and support bits through the learned global MLP";
  if (config.global_bottleneck_mode == 3)
    return contextual ?
        "at least one visible patch; semantic-channel-ordered original-patch-aligned contextual W states and slot-visible bits through a learned global MLP before diagnostic D compression" :
        "at least one visible patch; semantic-channel-ordered original-patch-aligned temporal W states and slot-visible bits through a learned global MLP before diagnostic D compression";
  if (config.global_bottleneck_mode == 2)
    return contextual ?
        "at least one channel has visible observations; semantic-ordered contextual vectors and observed-support bits through a learned global MLP" :
        "at least one channel has visible observations; semantic-ordered local vectors and observed-support bits through a learned global MLP";
  return contextual ?
      "at least one channel has visible observations; equal-valid-channel observed contextual mean" :
      "at least one channel has visible observations; equal-valid-channel local mean";
}

struct Settings {
  Config model{};
  int64_t steps{8}; // Additional completed optimizer updates per invocation.
  int64_t batch_size{4};
  int64_t seed{101};
  int64_t threads{1};
  int64_t log_every{1};
  int64_t checkpoint_every{0}; // Attempted batches; 0 means final save only.
  int64_t attempt_limit{1000}; // Additional attempted batches per invocation.
  double learning_rate{1e-3};
  double weight_decay{1e-4};
  double gradient_clip_norm{1.0};
};

// Versioned raw archives preserve float64 source precision before scaling.
// endpoints is required float64 [B]; channel_ids is int64 [C] or [B,C].
struct Dataset {
  Input input;
  std::string feature_units;
  std::string schema_id;
  std::string dataset_id;
};

struct Checkpoint {
  Settings settings;
  Model model{nullptr};
  FrozenScaler scaler;
  int64_t attempted_steps{0};
  int64_t completed_steps{0};
  std::string schema_id;
  std::string dataset_id;
  std::string scaler_fit_dataset_id;
  std::string source_fingerprint;
  std::string git_head;
  std::string git_dirty;
  // Empty means the ordinary training loop. Nonempty policies may share these
  // inference tensors, but require their matching training adapter to resume.
  std::string training_policy_id;
};

Settings default_settings();
void validate_settings(const Settings &settings);
Settings parse_settings(const std::string &text);
Settings read_settings(const std::string &path);
std::string settings_text(const Settings &settings);

Dataset synthetic_dataset(const Config &config, int64_t samples,
                          int64_t seed = 101);
// Validate already-cropped raw input and derive the exact archive identities.
// The returned input shares tensors with the caller; mutation requires a fresh
// description. Empty units explicitly selects unitless coordinates.
Dataset describe_dataset(const Input &input, const Config &config,
                         const std::string &feature_units = {});
std::string workflow_source_fingerprint();
void save_dataset(const std::string &path, const Dataset &dataset);
Dataset load_dataset(const std::string &path, const Config &config);
void save_scaler(const std::string &path, const FrozenScaler &scaler,
                 const Config &config, const std::string &schema_id,
                 const std::string &fit_dataset_id = "unrecorded");
FrozenScaler load_scaler(const std::string &path, const Config &config,
                         const std::string &schema_id);

void save_checkpoint(const std::string &path, const Checkpoint &checkpoint,
                     torch::optim::AdamW &optimizer);
Checkpoint load_checkpoint(const std::string &path,
                           const torch::Device &device = torch::kCPU);
void load_optimizer(const std::string &path, torch::optim::AdamW &optimizer,
                    const torch::Device &device = torch::kCPU);

// Throws on invalid arguments or an exhausted attempt budget. A final recovery
// checkpoint is saved before reporting exhaustion.
int run_cli(int argc, char **argv);

} // namespace embedding::encoders::raw_patch_bottleneck_mae
