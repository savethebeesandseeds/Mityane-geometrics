// SPDX-License-Identifier: MIT
#pragma once

#include <limits>
#include <map>
#include <set>
#include <utility>
#include "embedding/encoders/mtf_jepa_mae_vicreg/types.h"

namespace embedding::encoders::mtf_jepa_mae_vicreg {

namespace detail {

inline void validate_mask_metadata(const mtf_token_batch_t &batch) {
  TORCH_CHECK(batch.token_mask.defined() && batch.token_mask.dim() == 2 &&
                  batch.token_mask.scalar_type() == torch::kBool,
              "[embedding] token_mask must be bool [B,N]");
  const auto N = batch.token_mask.size(1);
  for (const auto &field : {batch.metadata.start_index, batch.metadata.width,
                            batch.metadata.scale_id, batch.metadata.channel_id,
                            batch.metadata.domain_id}) {
    TORCH_CHECK(field.defined() && field.dim() == 1 && field.size(0) == N &&
                    field.scalar_type() == torch::kInt64,
                "[embedding] token metadata must be int64 [N]");
  }
  auto starts = batch.metadata.start_index.to(torch::kCPU).contiguous();
  auto widths = batch.metadata.width.to(torch::kCPU).contiguous();
  auto start = starts.accessor<int64_t, 1>();
  auto width = widths.accessor<int64_t, 1>();
  for (int64_t n = 0; n < N; ++n) {
    TORCH_CHECK(start[n] >= 0 && width[n] > 0 &&
                    start[n] <= std::numeric_limits<int64_t>::max() - width[n],
                "[embedding] invalid token temporal support");
  }
}

using support_interval_t = std::pair<int64_t, int64_t>;

inline std::vector<support_interval_t>
merge_support_intervals(std::vector<support_interval_t> intervals) {
  std::sort(intervals.begin(), intervals.end());
  std::vector<support_interval_t> merged;
  for (const auto &interval : intervals) {
    if (merged.empty() || merged.back().second < interval.first) {
      merged.push_back(interval);
    } else {
      merged.back().second = std::max(merged.back().second, interval.second);
    }
  }
  return merged;
}

} // namespace detail

// Audits conservative raw temporal support, using a union per sample/channel
// so repeated scales and time/frequency views do not double-count coverage.
// Token metadata cannot resolve missing individual features or timestamps;
// reported support therefore includes every position in each valid window.
inline jepa_mask_audit_t
audit_jepa_masks(const mtf_token_batch_t &batch,
                 const jepa_context_target_mask_t &masks) {
  detail::validate_mask_metadata(batch);
  const int64_t B = batch.token_mask.size(0);
  const int64_t N = batch.token_mask.size(1);
  for (const auto &mask : {masks.context_mask, masks.target_mask}) {
    TORCH_CHECK(mask.defined() && mask.sizes() == batch.token_mask.sizes() &&
                    mask.scalar_type() == torch::kBool,
                "[embedding] JEPA audit masks must be bool [B,N]");
  }
  auto valid = batch.token_mask.to(torch::kCPU).contiguous();
  auto context = masks.context_mask.to(torch::kCPU).contiguous();
  auto target = masks.target_mask.to(torch::kCPU).contiguous();
  TORCH_CHECK(!context.logical_and(target).any().item<bool>(),
              "[embedding] JEPA audit context/target masks are not disjoint");
  TORCH_CHECK(!context.logical_or(target).logical_and(valid.logical_not())
                   .any().item<bool>(),
              "[embedding] JEPA audit selects invalid tokens");
  auto channels = batch.metadata.channel_id.to(torch::kCPU).contiguous();
  auto starts = batch.metadata.start_index.to(torch::kCPU).contiguous();
  auto widths = batch.metadata.width.to(torch::kCPU).contiguous();
  auto valid_acc = valid.accessor<bool, 2>();
  auto context_acc = context.accessor<bool, 2>();
  auto target_acc = target.accessor<bool, 2>();
  auto channel_acc = channels.accessor<int64_t, 1>();
  auto start_acc = starts.accessor<int64_t, 1>();
  auto width_acc = widths.accessor<int64_t, 1>();
  jepa_mask_audit_t out{};
  out.sample_count = B;
  for (int64_t b = 0; b < B; ++b) {
    int64_t valid_count = 0;
    int64_t target_count = 0;
    std::map<int64_t, std::vector<detail::support_interval_t>> target_support;
    std::map<int64_t, std::vector<detail::support_interval_t>> context_support;
    for (int64_t n = 0; n < N; ++n) {
      valid_count += valid_acc[b][n] ? 1 : 0;
      const auto interval = std::make_pair(start_acc[n],
                                           start_acc[n] + width_acc[n]);
      if (context_acc[b][n]) {
        ++out.context_tokens;
        context_support[channel_acc[n]].push_back(interval);
      }
      if (target_acc[b][n]) {
        ++target_count;
        target_support[channel_acc[n]].push_back(interval);
      }
    }
    out.valid_tokens += valid_count;
    out.target_tokens += target_count;
    if (target_count == 0) {
      ++out.no_target_samples;
      if (valid_count > 0) ++out.valid_no_target_samples;
    }
    for (auto &[channel, intervals] : context_support)
      intervals = detail::merge_support_intervals(std::move(intervals));
    for (int64_t n = 0; n < N; ++n) {
      if (!target_acc[b][n]) continue;
      const auto found = context_support.find(channel_acc[n]);
      if (found == context_support.end()) continue;
      for (const auto &[start, end] : found->second) {
        if (std::max(start_acc[n], start) <
            std::min(start_acc[n] + width_acc[n], end)) {
          ++out.overlapping_target_tokens;
          break;
        }
      }
    }
    for (auto &[channel, intervals] : target_support) {
      const auto merged = detail::merge_support_intervals(std::move(intervals));
      const auto found = context_support.find(channel);
      for (const auto &[start, end] : merged) {
        out.target_support_points += end - start;
        if (found == context_support.end()) continue;
        for (const auto &[context_start, context_end] : found->second) {
          out.shared_support_points += std::max<int64_t>(
              0, std::min(end, context_end) - std::max(start, context_start));
        }
      }
    }
  }
  auto diagnostic_count = [&](const torch::Tensor &flag) {
    if (!flag.defined()) return int64_t{0};
    TORCH_CHECK(flag.dim() == 1 && flag.size(0) == B &&
                    flag.scalar_type() == torch::kBool,
                "[embedding] JEPA diagnostic flags must be bool [B]");
    return flag.to(torch::kCPU).sum().item<int64_t>();
  };
  out.support_relaxed_samples = diagnostic_count(masks.support_relaxed);
  out.target_reduced_samples = diagnostic_count(masks.targets_reduced);
  out.target_resampled_samples = diagnostic_count(masks.target_resampled);
  if (out.target_tokens > 0)
    out.target_overlap_fraction = static_cast<double>(out.overlapping_target_tokens) /
                                   static_cast<double>(out.target_tokens);
  if (out.target_support_points > 0)
    out.shared_support_ratio = static_cast<double>(out.shared_support_points) /
                               static_cast<double>(out.target_support_points);
  return out;
}

class JEPAContextTargetMasker {
public:
  explicit JEPAContextTargetMasker(mtf_jepa_mae_vicreg_config_t config)
      : config_(std::move(config)) {
    detail::validate_config(config_);
  }

  [[nodiscard]] jepa_context_target_mask_t
  create_masks(const mtf_token_batch_t &batch) const {
    if (config_.strict_jepa_support) return create_strict_masks(batch);
    return create_legacy_masks(batch);
  }

private:
  [[nodiscard]] jepa_context_target_mask_t
  create_strict_masks(const mtf_token_batch_t &batch) const {
    detail::validate_mask_metadata(batch);
    // Retain the legacy random target proposal, including channel/domain
    // coupling. Its context is discarded: strict context uses an unconditional
    // same-channel support ban across all scales and both descriptor domains.
    auto out = create_legacy_masks(batch);
    const int64_t B = batch.token_mask.size(0);
    const int64_t N = batch.token_mask.size(1);
    auto valid = batch.token_mask.to(torch::kCPU).contiguous();
    auto target = out.target_mask.to(torch::kCPU).contiguous();
    auto context = torch::zeros_like(target);
    auto reduced = out.targets_reduced.to(torch::kCPU).contiguous();
    auto resampled = torch::zeros(
        {B}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto channels = batch.metadata.channel_id.to(torch::kCPU).contiguous();
    auto starts = batch.metadata.start_index.to(torch::kCPU).contiguous();
    auto widths = batch.metadata.width.to(torch::kCPU).contiguous();
    auto valid_acc = valid.accessor<bool, 2>();
    auto target_acc = target.accessor<bool, 2>();
    auto context_acc = context.accessor<bool, 2>();
    auto reduced_acc = reduced.accessor<bool, 1>();
    auto resampled_acc = resampled.accessor<bool, 1>();
    auto channel_acc = channels.accessor<int64_t, 1>();
    auto start_acc = starts.accessor<int64_t, 1>();
    auto width_acc = widths.accessor<int64_t, 1>();
    for (int64_t b = 0; b < B; ++b) {
      std::vector<int64_t> valid_indices;
      std::set<int64_t> target_set;
      for (int64_t n = 0; n < N; ++n) {
        if (valid_acc[b][n]) valid_indices.push_back(n);
        if (target_acc[b][n]) target_set.insert(n);
      }
      const int64_t valid_count = static_cast<int64_t>(valid_indices.size());
      if (valid_count == 0) continue;
      const int64_t min_context = std::max<int64_t>(
          1, static_cast<int64_t>(std::ceil(config_.min_context_ratio * valid_count)));
      auto forbidden_context = [&](const std::set<int64_t> &targets) {
        std::set<int64_t> forbidden;
        for (const auto candidate : valid_indices) {
          if (targets.count(candidate) != 0) continue;
          for (const auto t : targets) {
            if (channel_acc[candidate] != channel_acc[t]) continue;
            const bool overlaps = std::max(start_acc[candidate], start_acc[t]) <
                std::min(start_acc[candidate] + width_acc[candidate],
                         start_acc[t] + width_acc[t]);
            if (config_.mask_same_channel_block || overlaps) {
              forbidden.insert(candidate);
              break;
            }
          }
        }
        return forbidden;
      };
      auto context_count = [&](const std::set<int64_t> &targets,
                                const std::set<int64_t> &forbidden) {
        return valid_count - static_cast<int64_t>(targets.size()) -
               static_cast<int64_t>(forbidden.size());
      };
      auto forbidden = forbidden_context(target_set);
      // Every support ban remains hard. Remove a target with the greatest
      // immediate context gain until enough context is available; if no
      // nonempty feasible subset remains, expose that fact via diagnostics.
      while (!target_set.empty() &&
             context_count(target_set, forbidden) < min_context) {
        reduced_acc[b] = true;
        int64_t best_target = *target_set.begin();
        int64_t best_context = -1;
        int64_t best_forbid_reduction = -1;
        for (const auto candidate : target_set) {
          auto smaller = target_set;
          smaller.erase(candidate);
          const auto smaller_forbidden = forbidden_context(smaller);
          const auto candidate_context = context_count(smaller, smaller_forbidden);
          const auto forbid_reduction = static_cast<int64_t>(forbidden.size()) -
              static_cast<int64_t>(smaller_forbidden.size());
          if (candidate_context > best_context ||
              (candidate_context == best_context &&
               forbid_reduction > best_forbid_reduction)) {
            best_target = candidate;
            best_context = candidate_context;
            best_forbid_reduction = forbid_reduction;
          }
        }
        target_set.erase(best_target);
        forbidden = forbidden_context(target_set);
      }
      if (target_set.empty() && valid_count > min_context) {
        // A sampled broad window can be infeasible even when a narrower
        // alternate target is possible. Try each valid singleton once, in a
        // random order, without ever relaxing separation or minimum context.
        resampled_acc[b] = true;
        auto order = torch::randperm(valid_count,
            torch::TensorOptions().dtype(torch::kInt64));
        auto order_acc = order.accessor<int64_t, 1>();
        for (int64_t i = 0; i < valid_count; ++i) {
          const int64_t candidate = valid_indices[
              static_cast<std::size_t>(order_acc[i])];
          std::set<int64_t> singleton{candidate};
          auto singleton_forbidden = forbidden_context(singleton);
          if (context_count(singleton, singleton_forbidden) >= min_context) {
            target_set = std::move(singleton);
            forbidden = std::move(singleton_forbidden);
            break;
          }
        }
      }
      for (const auto n : valid_indices) {
        target_acc[b][n] = target_set.count(n) != 0;
        context_acc[b][n] = target_set.count(n) == 0 && forbidden.count(n) == 0;
      }
    }
    out.target_mask = target.to(batch.token_mask.device());
    out.context_mask = context.to(batch.token_mask.device());
    out.targets_reduced = reduced.to(batch.token_mask.device());
    out.target_resampled = resampled.to(batch.token_mask.device());
    out.support_relaxed = torch::zeros_like(resampled).to(batch.token_mask.device());
    return out;
  }

  [[nodiscard]] jepa_context_target_mask_t
  create_legacy_masks(const mtf_token_batch_t &batch) const {
    TORCH_CHECK(batch.token_mask.defined() && batch.token_mask.dim() == 2,
                "[embedding] token_mask must be [B,N]");
    const int64_t B = batch.token_mask.size(0);
    const int64_t N = batch.token_mask.size(1);
    auto valid_cpu = batch.token_mask.to(torch::kCPU).contiguous();
    auto target_mask_cpu = torch::zeros(
        {B, N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto context_mask_cpu = torch::zeros_like(target_mask_cpu);
    auto support_relaxed_cpu = torch::zeros(
        {B}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto targets_reduced_cpu = torch::zeros_like(support_relaxed_cpu);
    auto support_relaxed_acc = support_relaxed_cpu.accessor<bool, 1>();
    auto targets_reduced_acc = targets_reduced_cpu.accessor<bool, 1>();
    auto domain_cpu = batch.metadata.domain_id.to(torch::kCPU).contiguous();
    auto channel_cpu = batch.metadata.channel_id.to(torch::kCPU).contiguous();
    auto scale_cpu = batch.metadata.scale_id.to(torch::kCPU).contiguous();
    auto start_cpu = batch.metadata.start_index.to(torch::kCPU).contiguous();
    auto width_cpu = batch.metadata.width.to(torch::kCPU).contiguous();
    auto valid_acc = valid_cpu.accessor<bool, 2>();
    auto target_acc = target_mask_cpu.accessor<bool, 2>();
    auto context_acc = context_mask_cpu.accessor<bool, 2>();
    auto domain_acc = domain_cpu.accessor<int64_t, 1>();
    auto channel_acc = channel_cpu.accessor<int64_t, 1>();
    auto scale_acc = scale_cpu.accessor<int64_t, 1>();
    auto start_acc = start_cpu.accessor<int64_t, 1>();
    auto width_acc = width_cpu.accessor<int64_t, 1>();


    for (int64_t b = 0; b < B; ++b) {
      std::vector<int64_t> valid_indices;
      std::vector<int64_t> time_indices;
      std::vector<int64_t> freq_indices;
      valid_indices.reserve(static_cast<std::size_t>(N));
      for (int64_t n = 0; n < N; ++n) {
        if (!valid_acc[b][n]) {
          continue;
        }
        valid_indices.push_back(n);
        if (domain_acc[n] == 0) {
          time_indices.push_back(n);
        } else {
          freq_indices.push_back(n);
        }
      }

      const int64_t valid_count = static_cast<int64_t>(valid_indices.size());
      if (valid_count == 0) {
        continue;
      }
      const int64_t min_context =
          std::max<int64_t>(1, static_cast<int64_t>(std::ceil(
                                   config_.min_context_ratio * valid_count)));
      const int64_t max_target =
          std::max<int64_t>(0, valid_count - min_context);
      std::set<int64_t> target_set;

      if (!time_indices.empty() && max_target > 0) {
        const int64_t desired_time = std::max<int64_t>(
            1, static_cast<int64_t>(std::llround(config_.mask_ratio_time *
                                                 time_indices.size())));
        const int64_t block_len = std::min<int64_t>(
            desired_time, static_cast<int64_t>(time_indices.size()));
        const int64_t max_start =
            static_cast<int64_t>(time_indices.size()) - block_len;
        int64_t block_start = 0;
        if (max_start > 0) {
          block_start =
              torch::randint(0, max_start + 1, {1},
                             torch::TensorOptions().dtype(torch::kInt64))
                  .item<int64_t>();
        }
        for (int64_t i = 0; i < block_len; ++i) {
          target_set.insert(
              time_indices[static_cast<std::size_t>(block_start + i)]);
        }
      }

      if (!freq_indices.empty() && max_target > 0 &&
          config_.mask_ratio_frequency > 0.0) {
        const int64_t desired_freq = static_cast<int64_t>(
            std::llround(config_.mask_ratio_frequency * freq_indices.size()));
        add_random_subset(freq_indices, desired_freq, target_set);
      }

      if (config_.mask_ratio_channel > 0.0 && max_target > 0) {
        for (int64_t c = 0; c < config_.channel_count; ++c) {
          const double draw = torch::rand({1}).item<double>();
          if (draw > config_.mask_ratio_channel) {
            continue;
          }
          for (const auto idx : valid_indices) {
            if (channel_acc[idx] == c) {
              target_set.insert(idx);
            }
          }
        }
      }

      if (target_set.empty() && max_target > 0 && valid_count > 1) {
        target_set.insert(valid_indices.back());
      }

      if (!target_set.empty() && max_target > 0) {
        std::vector<int64_t> seed_targets(target_set.begin(), target_set.end());
        for (const auto target_idx : seed_targets) {
          for (const auto candidate_idx : valid_indices) {
            if (candidate_idx == target_idx ||
                target_set.find(candidate_idx) != target_set.end()) {
              continue;
            }
            const bool same_window =
                channel_acc[candidate_idx] == channel_acc[target_idx] &&
                scale_acc[candidate_idx] == scale_acc[target_idx] &&
                start_acc[candidate_idx] == start_acc[target_idx] &&
                width_acc[candidate_idx] == width_acc[target_idx];
            const bool cross_domain =
                domain_acc[candidate_idx] != domain_acc[target_idx];
            const bool same_channel =
                channel_acc[candidate_idx] == channel_acc[target_idx];
            if (config_.mask_same_channel_block && same_channel &&
                static_cast<int64_t>(target_set.size()) < max_target) {
              target_set.insert(candidate_idx);
              continue;
            }
            if (config_.couple_time_frequency_masks &&
                config_.mask_same_window_across_domains && same_window &&
                cross_domain &&
                static_cast<int64_t>(target_set.size()) < max_target) {
              target_set.insert(candidate_idx);
            }
          }
        }
      }

      if (static_cast<int64_t>(target_set.size()) > max_target) {
        targets_reduced_acc[b] = true;
        std::vector<int64_t> capped(target_set.begin(), target_set.end());
        target_set.clear();
        add_random_subset(capped, max_target, target_set);
      }

      std::set<int64_t> hard_forbidden_context;
      std::set<int64_t> soft_forbidden_context;
      auto recompute_forbids = [&](const std::set<int64_t> &targets,
                                   std::set<int64_t> &hard,
                                   std::set<int64_t> &soft) {
        hard.clear();
        soft.clear();
        for (const auto target_idx : targets) {
          for (const auto candidate_idx : valid_indices) {
            if (candidate_idx == target_idx ||
                targets.find(candidate_idx) != targets.end()) {
              continue;
            }
            const bool same_window =
                channel_acc[candidate_idx] == channel_acc[target_idx] &&
                scale_acc[candidate_idx] == scale_acc[target_idx] &&
                start_acc[candidate_idx] == start_acc[target_idx] &&
                width_acc[candidate_idx] == width_acc[target_idx];
            const bool cross_domain =
                domain_acc[candidate_idx] != domain_acc[target_idx];
            const bool same_channel =
                channel_acc[candidate_idx] == channel_acc[target_idx];
            if ((config_.couple_time_frequency_masks &&
                 config_.mask_same_window_across_domains && same_window &&
                 cross_domain) ||
                (config_.mask_same_channel_block && same_channel)) {
              hard.insert(candidate_idx);
              soft.erase(candidate_idx);
              continue;
            }
            const double overlap = interval_overlap_ratio(
                start_acc[target_idx], width_acc[target_idx],
                start_acc[candidate_idx], width_acc[candidate_idx]);
            if (overlap > config_.max_context_target_time_overlap &&
                hard.find(candidate_idx) == hard.end()) {
              soft.insert(candidate_idx);
            }
          }
        }
      };

      auto context_count_with = [&](const std::set<int64_t> &targets,
                                    const std::set<int64_t> &hard,
                                    const std::set<int64_t> &soft) {
        int64_t count = 0;
        for (const auto idx : valid_indices) {
          if (targets.find(idx) == targets.end() &&
              hard.find(idx) == hard.end() && soft.find(idx) == soft.end()) {
            ++count;
          }
        }
        return count;
      };

      recompute_forbids(target_set, hard_forbidden_context,
                        soft_forbidden_context);

      std::set<int64_t> empty_forbidden_context;
      while (!target_set.empty() &&
             context_count_with(target_set, hard_forbidden_context,
                                empty_forbidden_context) < min_context) {
        targets_reduced_acc[b] = true;
        int64_t best_target = *target_set.begin();
        int64_t best_context_count = -1;
        int64_t best_hard_reduction = -1;
        for (const auto candidate_target : target_set) {
          auto reduced_targets = target_set;
          reduced_targets.erase(candidate_target);
          std::set<int64_t> reduced_hard;
          std::set<int64_t> reduced_soft;
          recompute_forbids(reduced_targets, reduced_hard, reduced_soft);
          const int64_t reduced_context = context_count_with(
              reduced_targets, reduced_hard, empty_forbidden_context);
          const int64_t hard_reduction =
              static_cast<int64_t>(hard_forbidden_context.size()) -
              static_cast<int64_t>(reduced_hard.size());
          if (reduced_context > best_context_count ||
              (reduced_context == best_context_count &&
               hard_reduction > best_hard_reduction)) {
            best_target = candidate_target;
            best_context_count = reduced_context;
            best_hard_reduction = hard_reduction;
          }
        }
        target_set.erase(best_target);
        recompute_forbids(target_set, hard_forbidden_context,
                          soft_forbidden_context);
      }

      auto context_count_after_forbid = [&]() {
        return context_count_with(target_set, hard_forbidden_context,
                                  soft_forbidden_context);
      };
      while (context_count_after_forbid() < min_context &&
             !soft_forbidden_context.empty()) {
        support_relaxed_acc[b] = true;
        auto it = soft_forbidden_context.end();
        --it;
        soft_forbidden_context.erase(it);
      }

      for (const auto idx : target_set) {
        target_acc[b][idx] = true;
      }
      for (const auto idx : valid_indices) {
        const bool is_target = target_set.find(idx) != target_set.end();
        const bool is_forbidden =
            hard_forbidden_context.find(idx) != hard_forbidden_context.end() ||
            soft_forbidden_context.find(idx) != soft_forbidden_context.end();
        context_acc[b][idx] = !is_target && !is_forbidden;
      }
    }

    jepa_context_target_mask_t out{};
    out.context_mask = context_mask_cpu.to(batch.token_mask.device());
    out.target_mask = target_mask_cpu.to(batch.token_mask.device());
    out.valid_mask = batch.token_mask;
    out.support_relaxed = support_relaxed_cpu.to(batch.token_mask.device());
    out.targets_reduced = targets_reduced_cpu.to(batch.token_mask.device());
    return out;
  }

  static void add_random_subset(const std::vector<int64_t> &indices,
                                int64_t desired,
                                std::set<int64_t> &target_set) {
    if (desired <= 0 || indices.empty()) {
      return;
    }
    const int64_t take =
        std::min<int64_t>(desired, static_cast<int64_t>(indices.size()));
    auto perm = torch::randperm(static_cast<int64_t>(indices.size()),
                                torch::TensorOptions().dtype(torch::kInt64));
    auto acc = perm.accessor<int64_t, 1>();
    for (int64_t i = 0; i < take; ++i) {
      target_set.insert(indices[static_cast<std::size_t>(acc[i])]);
    }
  }

  static double interval_overlap_ratio(int64_t start_a, int64_t width_a,
                                       int64_t start_b, int64_t width_b) {
    const int64_t end_a = start_a + width_a;
    const int64_t end_b = start_b + width_b;
    const int64_t overlap = std::max<int64_t>(
        0, std::min(end_a, end_b) - std::max(start_a, start_b));
    const int64_t denom = std::max<int64_t>(1, std::min(width_a, width_b));
    return static_cast<double>(overlap) / static_cast<double>(denom);
  }

  mtf_jepa_mae_vicreg_config_t config_{};
};

} // namespace embedding::encoders::mtf_jepa_mae_vicreg
