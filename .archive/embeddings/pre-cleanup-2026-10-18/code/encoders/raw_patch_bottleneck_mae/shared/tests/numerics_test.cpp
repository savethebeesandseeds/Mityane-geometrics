#include "embedding/encoders/raw_patch_bottleneck_mae/model.h"
#include "rpb_test_support.h"
#include <iostream>
#include <limits>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;

void hierarchical_loss() {
  auto predictions = torch::tensor({{{{1.0f, 1.0f}}, {{2.0f, 9.0f}}},
                                    {{{3.0f, 9.0f}}, {{9.0f, 9.0f}}}});
  auto target = torch::zeros_like(predictions);
  auto support = torch::tensor({{{{true, true}}, {{true, false}}},
                                {{{true, false}}, {{false, false}}}}, torch::kBool);
  auto eligible = support.any(-1).any(-1);
  auto result = rpb::hierarchical_huber(predictions, target, support, eligible, 1.0);
  // Channel means: .5,1.5; example means:1,2.5; equal-example mean:1.75.
  close(result.loss, torch::tensor(1.75f), "equal-channel/equal-example Huber", 0, 0);
  check(result.target_cell_count == 4 && result.eligible_channel_count == 3 && result.eligible_example_count == 2, "loss denominators");
  auto ignored = predictions.masked_fill(support.logical_not(), std::numeric_limits<float>::quiet_NaN());
  close(rpb::hierarchical_huber(ignored, target, support, eligible, 1).loss, result.loss, "absent predictions isolated", 0, 0);
  auto bad = predictions.clone(); bad.index_put_({0, 0, 0, 0}, std::numeric_limits<float>::infinity());
  rejects([&] { rpb::hierarchical_huber(bad, target, support, eligible, 1); }, "nonfinite fixed-Q prediction");
  auto none = torch::zeros_like(support); auto skipped = rpb::hierarchical_huber(predictions, target, none, none.any(-1).any(-1), 1);
  check(skipped.eligible_example_count == 0 && skipped.loss.item<float>() == 0 && !skipped.loss.requires_grad(), "empty batch loss skip contract");
}

void padding_and_public_validation() {
  auto c = config(); rpb::PreNormBlock block(c); block->eval();
  auto h = torch::randn({2, 4, 16}); auto valid = torch::tensor({{true, true, true, true}, {true, false, false, false}}, torch::kBool);
  auto original = block(h, valid);
  auto poison = h.masked_fill(valid.logical_not().unsqueeze(-1), std::numeric_limits<float>::quiet_NaN());
  close(block(poison, valid), original, "padding contents cannot affect attention", 0, 0);
  close(original[1].slice(0, 1), torch::zeros({3, 16}), "padded query states after residuals", 0, 0);
  auto all_invalid = torch::zeros_like(valid);
  rejects([&] { block(h, all_invalid); }, "all-invalid raw attention row");
  for (auto member : {&rpb::Config::dropout, &rpb::Config::layer_norm_epsilon, &rpb::Config::mask_ratio,
                      &rpb::Config::huber_delta, &rpb::Config::scale_floor, &rpb::Config::sampling_interval}) {
    auto bad = c; bad.*member = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { rpb::Model model(bad); }, "nonfinite public numerical config");
  }
  auto raw = input(c); raw.data = raw.data.to(torch::kFloat32);
  raw.data.index_put_({0, 0, 0, 0}, std::numeric_limits<float>::quiet_NaN());
  rpb::Model model(c);
  rejects([&] { model->encode(raw); }, "observed nonfinite model input");
}
}
int main() {
  try { torch::set_num_threads(1); torch::manual_seed(23); hierarchical_loss(); padding_and_public_validation(); std::cout << "RPB-MAE numerics tests passed\n"; }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
