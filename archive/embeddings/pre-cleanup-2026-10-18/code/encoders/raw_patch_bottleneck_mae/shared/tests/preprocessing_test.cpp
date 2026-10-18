#include "rpb_test_support.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/model.h"
#include <iostream>
#include <sstream>
#include <limits>

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
using namespace rpb_test;
using torch::indexing::Slice;

void precision_and_schema() {
  auto c = config(); auto raw = input(c);
  auto variation = torch::arange(c.history_length, torch::kFloat64).reshape({1, 1, c.history_length, 1}) * 0.01;
  raw.data = (variation + 1e12).expand({3, 2, c.history_length, 2}).clone().set_requires_grad(true);
  auto scaler = rpb::fit_scaler(raw, c); auto scaled = scaler.transform(raw, c);
  check(!scaler.mean.requires_grad() && !scaler.scale.requires_grad(), "fitted assets must be frozen");
  auto deviation = scaled.data.std(std::vector<int64_t>{0, 2}, false);
  check(deviation.gt(0.99).logical_and(deviation.lt(1.01)).all().item<bool>(), "source variation was lost before centering");
  check(raw.data.to(torch::kFloat32).max().item<float>() == raw.data.to(torch::kFloat32).min().item<float>(), "precision fixture must quantize in float32");
  auto order = torch::tensor({1, 0}, torch::kInt64);
  auto permuted = raw; permuted.data = raw.data.index_select(1, order); permuted.observed = raw.observed.index_select(1, order);
  permuted.channel_ids = raw.channel_ids.index_select(0, order);
  close(scaler.transform(permuted, c).data, scaled.data.index_select(1, order), "scaler follows semantic IDs", 0, 0);
  auto bad = raw; bad.channel_ids = torch::tensor({101, 101}, torch::kInt64);
  rejects([&] { scaler.transform(bad, c); }, "duplicate ID");
  bad.channel_ids = torch::tensor({101, 999}, torch::kInt64);
  rejects([&] { scaler.transform(bad, c); }, "unknown ID");

  std::stringstream stream; torch::serialize::OutputArchive archive; scaler.save(archive); archive.save_to(stream);
  torch::serialize::InputArchive loaded; loaded.load_from(stream); auto restored = rpb::FrozenScaler::load(loaded); restored.validate(c);
  check(restored.identity() == scaler.identity(), "scaler content identity roundtrip");
  check(torch::equal(restored.floor_applied, scaler.floor_applied), "scale floor coordinates roundtrip");
  close(restored.transform(raw, c).data, scaled.data, "scaler archive roundtrip", 0, 0);
  restored.mean = restored.mean.clone(); restored.mean.index_put_({0, 0}, restored.mean.index({0, 0}) + 1);
  check(restored.identity() != scaler.identity(), "scaler identity ignores changed statistics");
}

void missingness_and_floors() {
  auto c = config(); auto raw = input(c); raw.data.fill_(7);
  auto scaler = rpb::fit_scaler(raw, c);
  close(scaler.scale, torch::full_like(scaler.scale, c.scale_floor), "constant scale floor", 0, 0);
  check(scaler.floor_applied.all().item<bool>(), "constant coordinate floor policy must be recorded");
  close(scaler.transform(raw, c).data, torch::zeros_like(raw.data).to(torch::kFloat32), "constant centering", 0, 0);
  auto absent = raw; absent.observed = raw.observed.clone(); absent.observed.index_put_({0, 0, Slice(0, 8)}, false);
  absent.data = raw.data.masked_fill(absent.observed.logical_not(), std::numeric_limits<double>::quiet_NaN());
  finite(scaler.transform(absent, c).data, "masked NaN preprocessing");
  auto no_fit = raw; no_fit.observed = raw.observed.clone(); no_fit.observed.select(-1, 1).fill_(false);
  rejects([&] { rpb::fit_scaler(no_fit, c); }, "no-fit feature coordinate");
  auto poison = raw; poison.data = raw.data.clone(); poison.data.index_put_({0, 0, 0, 0}, std::numeric_limits<double>::infinity());
  rejects([&] { scaler.transform(poison, c); }, "observed infinity");
  auto overflow = raw; overflow.data = torch::full_like(raw.data, 1e100);
  rejects([&] { scaler.transform(overflow, c); }, "post-scaling float32 overflow");
}

void endpoint_preparation() {
  auto c = config(); auto raw = input(c); auto tail = torch::full({3, 2, 8, 2}, 12345.0, torch::kFloat64);
  auto full = torch::cat({raw.data, tail}, 2); auto support = torch::ones_like(full, torch::kBool);
  auto timestamps = torch::arange(40, torch::kFloat64);
  auto cropped = rpb::prepare_endpoint_history(full, support, raw.channel_ids, timestamps, raw.endpoints, c);
  close(cropped.data, raw.data, "endpoint support crop", 0, 0);
  auto scaler = rpb::fit_scaler(cropped, c); rpb::Model model(c); model->eval(); torch::NoGradGuard no_grad;
  auto before = model->encode(scaler.transform(cropped, c)).z_local;
  full.slice(2, 32).fill_(-98765);
  auto changed = rpb::prepare_endpoint_history(full, support, raw.channel_ids, timestamps, raw.endpoints, c);
  close(model->encode(scaler.transform(changed, c)).z_local, before, "preparation plus encoding future invariance", 0, 0);
  auto fractional = torch::full({3}, 31.5, torch::kFloat64);
  rejects([&] { rpb::prepare_endpoint_history(full, support, raw.channel_ids, timestamps, fractional, c); }, "unaligned fractional endpoint support metadata");
  auto early = torch::full({3}, 15.0, torch::kFloat64);
  rejects([&] { rpb::prepare_endpoint_history(full, support, raw.channel_ids, timestamps, early, c); }, "insufficient endpoint history");
  auto irregular = timestamps.clone(); irregular[5] += 0.1;
  rejects([&] { rpb::prepare_endpoint_history(full, support, raw.channel_ids, irregular, raw.endpoints, c); }, "irregular sampling");
}
}

int main() {
  try { torch::set_num_threads(1); torch::manual_seed(19); precision_and_schema(); missingness_and_floors(); endpoint_preparation(); std::cout << "RPB-MAE preprocessing tests passed\n"; }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
