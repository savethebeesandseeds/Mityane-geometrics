#include "embedding/encoders/raw_patch_bottleneck_mae/masking.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/tokenization.h"
#include "rpb_test_support.h"
#include <iostream>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;
using torch::indexing::Slice;

void discrete_support_policy() {
  auto c = config(); auto raw = input(c); auto mask = rpb::make_training_mask(raw.observed, c, 17);
  check(mask.target.sum().item<int64_t>() == 96 && mask.visible.sum().item<int64_t>() == 288,
        "four patches must hide one and retain three");
  check(!mask.visible.logical_and(mask.target).any().item<bool>() && torch::equal(mask.visible.logical_or(mask.target), raw.observed), "visible-target support partition");
  check(torch::equal(mask.hidden, rpb::make_training_mask(raw.observed, c, 17).hidden), "mask seed reproducibility");
  auto missing = raw.observed.clone(); missing.index_put_({0, 0, Slice(0, 16)}, false);
  missing.index_put_({0, 1, Slice(0, 8), 1}, false);
  auto partial = rpb::make_training_mask(missing, c, 29);
  check(!partial.eligible_channels.index({0, 0}).item<bool>() && !partial.hidden.index({0, 0}).any().item<bool>(), "two observed patches must be training-ineligible");
  check(!partial.target.logical_and(missing.logical_not()).any().item<bool>(), "naturally absent cells became targets");
  auto illegal = torch::zeros_like(missing); illegal.index_put_({1, 0, 0, 0}, true);
  rejects([&] { rpb::mask_from_hidden(missing, illegal, c); }, "partial artificial patch mask");
  c.history_length = 16; auto two = torch::ones({1, 2, 16, 2}, torch::kBool);
  check(!rpb::make_training_mask(two, c, 17).eligible_channels.any().item<bool>(), "short history must skip rather than overlap");
}

void original_positions_and_absent_rows() {
  auto c = config(); auto raw = input(c); auto scaled = rpb::fit_scaler(raw, c).transform(raw, c);
  scaled.observed.index_put_({0, 0, Slice(8, 16)}, false);
  scaled.observed.index_put_({0, 1}, false);
  auto patches = rpb::tokenize_visible(scaled, c);
  check(patches.row_indices.numel() == 5, "absent channel row must be excluded");
  close(patches.positions[0], torch::tensor({0, 2, 3, -1}, torch::kInt64), "original packed positions", 0, 0);
  check(torch::equal(patches.valid[0], torch::tensor({true, true, true, false}, torch::kBool)), "visible token padding validity");
  check(!patches.visibility[0][3].any().item<bool>(), "padded visibility manufactured support");
}
}
int main() {
  try { torch::set_num_threads(1); discrete_support_policy(); original_positions_and_absent_rows(); std::cout << "RPB-MAE masking tests passed\n"; }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
