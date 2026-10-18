#include "embedding/encoders/mtf_jepa_mae_vicreg/masking.h"
#include "embedding/encoders/mtf_jepa_mae_vicreg/tokenization.h"
#include "test_support.h"

#include <cmath>
#include <iostream>

namespace mtf = embedding::encoders::mtf_jepa_mae_vicreg;

namespace {

using test::check;
using torch::indexing::Slice;

mtf::mtf_token_batch_t make_batch(
    const std::vector<int64_t> &starts, const std::vector<int64_t> &widths,
    const std::vector<int64_t> &channels, const std::vector<int64_t> &domains,
    int64_t samples = 1) {
  mtf::mtf_token_batch_t batch{};
  const auto N = static_cast<int64_t>(starts.size());
  const auto options = torch::TensorOptions().dtype(torch::kInt64);
  batch.token_mask = torch::ones({samples, N}, torch::kBool);
  batch.metadata.start_index = torch::tensor(starts, options);
  batch.metadata.width = torch::tensor(widths, options);
  batch.metadata.channel_id = torch::tensor(channels, options);
  batch.metadata.domain_id = torch::tensor(domains, options);
  batch.metadata.scale_id = torch::arange(N, options);
  return batch;
}

void check_minimum_context(const mtf::mtf_token_batch_t &batch,
                           const mtf::jepa_context_target_mask_t &masks,
                           const mtf::Config &config) {
  for (int64_t b = 0; b < batch.token_mask.size(0); ++b) {
    const auto valid = batch.token_mask[b].sum().item<int64_t>();
    const auto context = masks.context_mask[b].sum().item<int64_t>();
    const auto targets = masks.target_mask[b].sum().item<int64_t>();
    const auto minimum = valid == 0 ? int64_t{0} : std::max<int64_t>(
        1, static_cast<int64_t>(std::ceil(config.min_context_ratio * valid)));
    check(context >= minimum, "minimum context count was not retained");
    check(context + targets <= valid, "selected tokens exceed valid count");
  }
}

void check_finite_audit(const mtf::jepa_mask_audit_t &audit) {
  check(std::isfinite(audit.target_overlap_fraction) &&
            audit.target_overlap_fraction >= 0.0 &&
            audit.target_overlap_fraction <= 1.0,
        "target overlap fraction is invalid");
  check(std::isfinite(audit.shared_support_ratio) &&
            audit.shared_support_ratio >= 0.0 && audit.shared_support_ratio <= 1.0,
        "shared support coverage is invalid");
  check(audit.conservative_support, "audit must label conservative window support");
}

void test_union_support_audit() {
  // Duplicate domains and overlapping scales share one support union, while
  // another channel at the same timestamps contributes no same-channel leak.
  auto batch = make_batch({0, 0, 2, 3, 0}, {4, 4, 4, 3, 6},
                          {0, 0, 0, 0, 1}, {0, 1, 0, 1, 0}, 2);
  batch.token_mask[1].fill_(false);
  mtf::jepa_context_target_mask_t masks{};
  masks.context_mask = torch::zeros_like(batch.token_mask);
  masks.target_mask = torch::zeros_like(batch.token_mask);
  masks.target_mask.index_put_({0, Slice(0, 3)}, true);
  masks.context_mask.index_put_({0, Slice(3, 5)}, true);
  masks.support_relaxed = torch::tensor({true, false}, torch::kBool);
  masks.targets_reduced = torch::tensor({false, true}, torch::kBool);
  const auto audit = mtf::audit_jepa_masks(batch, masks);
  check(audit.sample_count == 2 && audit.valid_tokens == 5 &&
            audit.context_tokens == 2 && audit.target_tokens == 3,
        "audit token counts are wrong");
  check(audit.overlapping_target_tokens == 3 && audit.target_overlap_fraction == 1.0,
        "audit must count each overlapping target once");
  check(audit.target_support_points == 6 && audit.shared_support_points == 3 &&
            audit.shared_support_ratio == 0.5,
        "audit must use support unions across scales/domains");
  check(audit.no_target_samples == 1 && audit.valid_no_target_samples == 0 &&
            audit.support_relaxed_samples == 1 && audit.target_reduced_samples == 1,
        "audit diagnostic counts are wrong");
  check_finite_audit(audit);

  // Endpoint adjacency has no shared raw timestamp.
  batch = make_batch({0, 4}, {4, 4}, {0, 0}, {0, 1});
  masks.context_mask = torch::tensor({{false, true}}, torch::kBool);
  masks.target_mask = torch::tensor({{true, false}}, torch::kBool);
  masks.support_relaxed = torch::Tensor();
  masks.targets_reduced = torch::Tensor();
  const auto adjacent = mtf::audit_jepa_masks(batch, masks);
  check(adjacent.overlapping_target_tokens == 0 && adjacent.shared_support_points == 0,
        "adjacent support windows should not overlap");
  check_finite_audit(adjacent);
}

void test_no_target_and_legacy_relaxation() {
  auto config = test::small_config<mtf::Config>();
  config.channel_count = 1;
  config.mask_ratio_time = 0.25;
  config.mask_ratio_frequency = 0.0;
  config.min_context_ratio = 0.5;
  config.max_context_target_time_overlap = 0.0;
  config.couple_time_frequency_masks = false;
  config.mask_same_window_across_domains = false;
  // All tokens depend on the same raw support, so retaining even one context
  // token makes any strict target infeasible. Domain/scale IDs must not help.
  const auto batch = make_batch({0, 0, 0, 0}, {8, 8, 8, 8},
                                {0, 0, 0, 0}, {0, 1, 0, 1});
  for (int64_t seed = 0; seed < 8; ++seed) {
    config.strict_jepa_support = false;
    torch::manual_seed(seed);
    const auto legacy = mtf::JEPAContextTargetMasker(config).create_masks(batch);
    const auto legacy_audit = mtf::audit_jepa_masks(batch, legacy);
    check(legacy_audit.target_tokens > 0 && legacy_audit.support_relaxed_samples == 1 &&
              legacy_audit.target_overlap_fraction == 1.0 &&
              legacy_audit.shared_support_ratio == 1.0,
          "legacy soft exclusion fallback must remain visible and preserved");
    check_minimum_context(batch, legacy, config);
    check_finite_audit(legacy_audit);

    config.strict_jepa_support = true;
    torch::manual_seed(seed);
    const auto strict = mtf::JEPAContextTargetMasker(config).create_masks(batch);
    const auto strict_audit = mtf::audit_jepa_masks(batch, strict);
    check(strict_audit.target_tokens == 0 && strict_audit.context_tokens == 4 &&
              strict_audit.valid_no_target_samples == 1 &&
              strict_audit.target_reduced_samples == 1 &&
              strict_audit.target_resampled_samples == 1,
          "strict infeasibility must drop targets and retain all context");
    check(strict_audit.support_relaxed_samples == 0 &&
              strict_audit.overlapping_target_tokens == 0 &&
              strict_audit.shared_support_ratio == 0.0,
          "strict infeasibility must never relax support separation");
    check_minimum_context(batch, strict, config);
    check_finite_audit(strict_audit);
  }

  auto singleton = make_batch({0}, {1}, {0}, {0}, 2);
  singleton.token_mask[1].fill_(false);
  const auto masks = mtf::JEPAContextTargetMasker(config).create_masks(singleton);
  const auto audit = mtf::audit_jepa_masks(singleton, masks);
  check(audit.context_tokens == 1 && audit.target_tokens == 0 &&
            audit.no_target_samples == 2 && audit.valid_no_target_samples == 1,
        "singleton/empty sample diagnostics are wrong");
  check_minimum_context(singleton, masks, config);
  check_finite_audit(audit);
}

void test_cross_channel_context_and_channel_block() {
  auto config = test::small_config<mtf::Config>();
  config.strict_jepa_support = true;
  config.mask_ratio_time = 0.25;
  config.mask_ratio_frequency = 0.0;
  config.min_context_ratio = 0.25;
  config.couple_time_frequency_masks = false;
  config.mask_same_window_across_domains = false;
  const auto batch = make_batch({0, 0, 4, 4}, {4, 4, 4, 4},
                                {0, 1, 0, 1}, {0, 0, 0, 0});
  for (const bool block_channel : {false, true}) {
    config.mask_same_channel_block = block_channel;
    for (int64_t seed = 10; seed < 18; ++seed) {
      torch::manual_seed(seed);
      const auto masks = mtf::JEPAContextTargetMasker(config).create_masks(batch);
      const auto audit = mtf::audit_jepa_masks(batch, masks);
      check(audit.target_tokens > 0 && audit.context_tokens > 0 &&
                audit.overlapping_target_tokens == 0,
            "cross-channel context test needs separated nonempty masks");
      bool has_cross_channel_context = false;
      for (int64_t t = 0; t < 4; ++t) {
        if (!masks.target_mask[0][t].item<bool>()) continue;
        for (int64_t c = 0; c < 4; ++c) {
          if (!masks.context_mask[0][c].item<bool>()) continue;
          const bool same_channel = batch.metadata.channel_id[t].item<int64_t>() ==
                                     batch.metadata.channel_id[c].item<int64_t>();
          if (!same_channel) has_cross_channel_context = true;
          if (block_channel)
            check(!same_channel, "channel-block mode left same-channel context");
        }
      }
      check(has_cross_channel_context, "strict masking discarded correlated other channels");
      check_minimum_context(batch, masks, config);
      check_finite_audit(audit);
    }
  }
}

void test_alternative_target_when_proposal_is_infeasible() {
  auto config = test::small_config<mtf::Config>();
  config.channel_count = 1;
  config.strict_jepa_support = true;
  config.mask_ratio_time = 0.34;
  config.mask_ratio_frequency = 0.0;
  config.min_context_ratio = 0.25;
  config.couple_time_frequency_masks = false;
  // The only time token forces a full-history proposal. It cannot be a target
  // with same-channel context, but either frequency endpoint can use the other.
  const auto batch = make_batch({0, 0, 6}, {8, 2, 2},
                                {0, 0, 0}, {0, 1, 1});
  bool saw_resampling = false;
  for (int64_t seed = 30; seed < 46; ++seed) {
    torch::manual_seed(seed);
    const auto masks = mtf::JEPAContextTargetMasker(config).create_masks(batch);
    const auto audit = mtf::audit_jepa_masks(batch, masks);
    check(audit.target_tokens > 0 && audit.valid_no_target_samples == 0 &&
              audit.overlapping_target_tokens == 0 &&
              !masks.target_mask[0][0].item<bool>(),
          "strict mode must find a feasible narrow target after a broad proposal");
    saw_resampling = saw_resampling || audit.target_resampled_samples > 0;
    check_minimum_context(batch, masks, config);
    check_finite_audit(audit);
  }
  check(saw_resampling, "alternative-target test did not exercise resampling");
}

void test_multiscale_sparse_masks() {
  torch::NoGradGuard no_grad;
  const auto data = test::input();
  auto feature_mask = torch::ones_like(data, torch::kBool);
  feature_mask[1].fill_(false);
  feature_mask.index_put_({1, 0, 2, 0}, true);
  feature_mask.index_put_({1, 1, 12, 1}, true);
  feature_mask.index_put_({2, 1}, false);
  feature_mask.index_put_({2, 0, Slice(4, 16)}, false);
  feature_mask[3].fill_(false);
  for (const bool frequency : {false, true}) {
    auto config = test::small_config<mtf::Config>();
    config.use_frequency_tokens = frequency;
    auto tokenizer = mtf::TimeFrequencyViewBuilder(config);
    const auto batch = tokenizer->forward(data, feature_mask);
    for (const bool coupled : {false, true}) {
      for (const bool channel_block : {false, true}) {
        for (const double minimum : {0.0, 0.25, 0.75, 1.0}) {
          config.couple_time_frequency_masks = coupled;
          config.mask_same_channel_block = channel_block;
          config.min_context_ratio = minimum;
          config.mask_ratio_channel = channel_block ? 0.7 : 0.0;
          for (int64_t seed = 20; seed < 24; ++seed) {
            for (const bool strict : {false, true}) {
              config.strict_jepa_support = strict;
              torch::manual_seed(seed);
              const auto masks = mtf::JEPAContextTargetMasker(config).create_masks(batch);
              const auto audit = mtf::audit_jepa_masks(batch, masks);
              check(audit.valid_tokens == batch.token_mask.sum().item<int64_t>() &&
                        audit.no_target_samples >= 1,
                    "sparse/empty token counts are inconsistent");
              if (strict)
                check(audit.overlapping_target_tokens == 0 &&
                          audit.shared_support_points == 0 &&
                          audit.support_relaxed_samples == 0,
                      "strict multiscale/domain support leaked into context");
              check_minimum_context(batch, masks, config);
              check_finite_audit(audit);
            }
          }
        }
      }
    }
  }
}

} // namespace

int main() {
  try {
    torch::set_num_threads(1);
    test_union_support_audit();
    test_no_target_and_legacy_relaxation();
    test_cross_channel_context_and_channel_block();
    test_alternative_target_when_proposal_is_infeasible();
    test_multiscale_sparse_masks();
    std::cout << "PASS: JEPA support audits, strict masks, legacy fallback, sparse and empty inputs\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
