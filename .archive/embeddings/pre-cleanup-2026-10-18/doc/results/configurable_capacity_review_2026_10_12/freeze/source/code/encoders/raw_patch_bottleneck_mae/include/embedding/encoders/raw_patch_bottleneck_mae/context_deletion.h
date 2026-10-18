// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/model.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {

// Explicitly identified training recipes on unchanged mode2/mixer1/native32
// inference. The old aggregate {true} retains the historical .30 recipe.
enum class ContextDeletionRecipe { coordinate30_v1, coordinate15_v1, balanced30_v1 };
struct ContextDeletionOptions {
  bool enabled{false};
  ContextDeletionRecipe recipe{ContextDeletionRecipe::coordinate30_v1};
};

namespace context_deletion {
inline constexpr double ratio = .30;
inline constexpr uint64_t stream = 0x6374782d64726f70ULL;
inline constexpr const char *policy_id = "rpb-training-context-deletion-v1";
inline constexpr const char *rng_policy =
    "splitmix64-counter-base-plus-semantic-canonical-BCHF-ordinal;top53-u01;independent-ctx-drop-stream-v1";
inline constexpr const char *repair_policy =
    "eligible-channels-only;restore-earliest-erased-original-patch-first-original-visible-coordinate-until-two-groups-v1";
inline constexpr const char *visibility_policy =
    "V=(O&~A)&~E;E-subset-original-visible;repair-only-E;original-O-A-Q-eligibility-Huber-unchanged-v1";
inline constexpr const char *balanced_schedule_policy = "absolute-attempt-even-ordinary-odd-coordinate30-v1";
inline constexpr const char *balanced_rate_scope = "active-odd-attempts-only;not-uniform-effective-rate";
inline constexpr const char *balanced_branch_count_policy =
    "absolute-attempt-prefix;no-skips;ordinary-plus-deletion-equals-attempted";
inline constexpr const char *balanced_skip_policy =
    "abort-before-update-on-ineligible-original-mask;no-replacement-or-schedule-shift";

struct RecipeDescriptor {
  double ratio;
  const char *ratio_text, *policy_id, *model_tag;
};

inline const RecipeDescriptor &descriptor(ContextDeletionRecipe recipe) {
  // The public constants above remain the immutable v6 .30 contract.
  static constexpr RecipeDescriptor coordinate30{ratio, "0.30", policy_id, "RPB-v6"};
  static constexpr RecipeDescriptor coordinate15{.15, "0.15", "rpb-training-context-deletion-015-v1", "RPB-v7"};
  static constexpr RecipeDescriptor balanced30{ratio, "0.30", "rpb-training-context-balanced-030-v1", "RPB-v8"};
  switch (recipe) {
    case ContextDeletionRecipe::coordinate30_v1: return coordinate30;
    case ContextDeletionRecipe::coordinate15_v1: return coordinate15;
    case ContextDeletionRecipe::balanced30_v1: return balanced30;
  }
  TORCH_CHECK(false, "[rpb context deletion] unknown bounded context recipe");
}

inline bool is_balanced(ContextDeletionRecipe recipe) {
  (void)descriptor(recipe);
  return recipe == ContextDeletionRecipe::balanced30_v1;
}

inline bool deletion_attempt(ContextDeletionRecipe recipe, int64_t attempt) {
  TORCH_CHECK(attempt >= 0, "[rpb context deletion] nonnegative absolute counters required");
  return !is_balanced(recipe) || attempt % 2 == 1;
}

inline void validate_options(const ContextDeletionOptions &options) {
  (void)descriptor(options.recipe);
  TORCH_CHECK(options.enabled || options.recipe == ContextDeletionRecipe::coordinate30_v1,
              "[rpb context deletion] disabled context must retain the default .30 recipe selector");
}

struct Plan {
  torch::Tensor visible, deleted, requested_deleted; // Bool BCHF on original mask device.
  int64_t requested_count{0}, actual_count{0}, restored_count{0};
};

// This function sees support/semantic identities only. All coordinates have a
// fixed counter ordinal, even when unobserved or a query, so support does not
// shift random draws. Physical channel storage order does not change E.
inline Plan make_plan(const MaskPlan &original, const torch::Tensor &ids,
                      const Config &config, int64_t seed, int64_t attempt,
                      ContextDeletionRecipe recipe) {
  const auto &selected = descriptor(recipe);
  TORCH_CHECK(seed >= 0 && attempt >= 0, "[rpb context deletion] nonnegative absolute counters required");
  const auto observed = original.visible.logical_or(original.target);
  const auto checked = mask_from_hidden(observed, original.hidden, config);
  TORCH_CHECK(torch::equal(checked.visible, original.visible) && torch::equal(checked.target, original.target) &&
                  torch::equal(checked.eligible_channels, original.eligible_channels),
              "[rpb context deletion] original legal masking plan required");
  const auto B = observed.size(0), C = config.channel_count;
  const auto H = config.history_length, F = config.input_width;
  const auto K = H / config.patch_length, PF = config.patch_length * F;
  (void)channel_indices(ids, config, B, torch::kCPU);
  if (!deletion_attempt(recipe, attempt)) {
    // Even attempts use the ordinary forward in the trainer, not this helper's
    // alternate forward. This support-only result consumes no deletion draws.
    const auto empty = torch::zeros_like(original.visible);
    return {original.visible.clone(), empty, empty.clone(), 0, 0, 0};
  }
  const auto rows = (ids.dim() == 1 ? ids.unsqueeze(0).expand({B, C}) : ids).to(torch::kCPU);
  const auto order = rows.argsort(int64_t{1}).contiguous();
  const auto canonical = order.accessor<int64_t, 2>();
  auto visible = original.visible.to(torch::kCPU).reshape({B, C, K, PF}).contiguous();
  auto requested = torch::zeros_like(visible);
  auto support = visible.accessor<bool, 4>();
  auto requests = requested.accessor<bool, 4>();
  auto eligible = original.eligible_channels.to(torch::kCPU).contiguous();
  auto original_eligible = eligible.accessor<bool, 2>();
  const auto base = training_detail::counter_seed(seed, attempt, stream);
  int64_t requested_count = 0, restored_count = 0;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t rank = 0; rank < C; ++rank) {
      const auto c = canonical[b][rank];
      for (int64_t k = 0; k < K; ++k) {
        for (int64_t j = 0; j < PF; ++j) {
          const auto ordinal = static_cast<uint64_t>((b * C + rank) * H * F + k * PF + j);
          const double uniform = static_cast<double>(training_detail::mixed(base + ordinal) >> 11) * 0x1.0p-53;
          requests[b][c][k][j] = support[b][c][k][j] && uniform < selected.ratio;
          requested_count += requests[b][c][k][j];
        }
      }
    }
  }
  auto deleted = requested.clone(); auto erasures = deleted.accessor<bool, 4>();
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t rank = 0; rank < C; ++rank) {
      const auto c = canonical[b][rank];
      if (!original_eligible[b][c]) continue;
      int64_t retained_groups = 0;
      for (int64_t k = 0; k < K; ++k) {
        bool retained = false;
        for (int64_t j = 0; j < PF; ++j) retained |= support[b][c][k][j] && !erasures[b][c][k][j];
        retained_groups += retained;
      }
      for (int64_t k = 0; k < K && retained_groups < 2; ++k) {
        bool retained = false; int64_t first = -1;
        for (int64_t j = 0; j < PF; ++j) {
          if (support[b][c][k][j] && first < 0) first = j;
          retained |= support[b][c][k][j] && !erasures[b][c][k][j];
        }
        if (!retained && first >= 0) {
          TORCH_CHECK(erasures[b][c][k][first], "[rpb context deletion] repair cannot invent support");
          erasures[b][c][k][first] = false;
          ++retained_groups; ++restored_count;
        }
      }
      TORCH_CHECK(retained_groups >= 2, "[rpb context deletion] original eligible support cannot be repaired");
    }
  }
  const auto device = original.visible.device();
  auto final_deleted = deleted.reshape_as(original.visible).to(device);
  return {original.visible.logical_and(final_deleted.logical_not()), final_deleted,
      requested.reshape_as(original.visible).to(device), requested_count,
      requested_count - restored_count, restored_count};
}

// Public legacy signature stays exactly bound to the original .30 recipe.
inline Plan make_plan(const MaskPlan &original, const torch::Tensor &ids,
                      const Config &config, int64_t seed, int64_t attempt) {
  return make_plan(original, ids, config, seed, attempt, ContextDeletionRecipe::coordinate30_v1);
}

// Targets never enter the encoder: only V with zeroed hidden storage is passed
// to encode. The existing decoder receives the exact served global32, while Q
// and hierarchical Huber denominators retain the original legal masking plan.
inline ForwardOutput training_forward(ModelImpl &model, const Input &input,
                                      const MaskPlan &original, const Plan &context) {
  const auto &config = model.config();
  TORCH_CHECK(config.global_bottleneck_mode == 2 && config.channel_mixer_layers == 1 && config.export_width == 32,
              "[rpb context deletion] fixed mode2/mixer1/native32 recipe required");
  validate_input(input, config, true);
  const auto checked = mask_from_hidden(input.observed.to(config.device), original.hidden, config);
  TORCH_CHECK(torch::equal(checked.visible, original.visible) && torch::equal(checked.target, original.target) &&
                  torch::equal(checked.eligible_channels, original.eligible_channels) &&
                  context.visible.defined() && context.visible.scalar_type() == torch::kBool &&
                  context.visible.sizes() == input.data.sizes() &&
                  !context.visible.logical_and(original.visible.logical_not()).any().item<bool>(),
              "[rpb context deletion] only original visible context may reach encoder");
  const auto groups = context.visible.reshape({input.data.size(0), config.channel_count,
      config.history_length / config.patch_length, -1}).any(-1).sum(-1);
  TORCH_CHECK(!original.eligible_channels.logical_and(groups.lt(2)).any().item<bool>(),
              "[rpb context deletion] original eligible channels require two retained visible groups");
  Input visible{torch::where(context.visible, input.data, torch::zeros_like(input.data)), context.visible,
      input.channel_ids, input.endpoints, input.sampling_interval};
  ForwardOutput out;
  out.encoding = model.encode(visible);
  out.reconstruction = model.decode(compact_reconstruction_export(out.encoding, config), input.channel_ids);
  const auto losses = hierarchical_huber(out.reconstruction, input.data.detach(), original.target,
      original.eligible_channels, config.huber_delta);
  out.loss = losses.loss; out.target_counts = losses.target_counts;
  out.eligible_channels = losses.eligible_channels; out.eligible_examples = losses.eligible_examples;
  out.eligible_channel_count = losses.eligible_channel_count;
  out.eligible_example_count = losses.eligible_example_count; out.target_cell_count = losses.target_cell_count;
  return out;
}
} // namespace context_deletion
} // namespace embedding::encoders::raw_patch_bottleneck_mae
