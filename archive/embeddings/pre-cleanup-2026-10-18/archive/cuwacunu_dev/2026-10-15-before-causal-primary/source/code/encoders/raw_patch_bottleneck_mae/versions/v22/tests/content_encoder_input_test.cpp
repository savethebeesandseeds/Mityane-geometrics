// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Artificial primary-API witness, not market training or research evaluation.
#include "cuwacunu/encoder/primary_encoder.h"
#include "cuwacunu/encoder/configurable_artifact.h"
#include "cuwacunu/integration/configurable_encoder_input.h"
#include <torch/cuda.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <unistd.h>

namespace {
namespace enc = cuwacunu::encoder::primary;
namespace v20 = cuwacunu::encoder::configurable;
namespace integration = cuwacunu::integration;
namespace fs = std::filesystem;
void check(bool condition, const char *message) { TORCH_CHECK(condition, "[v22 primary API] ", message); }
template<class F> void rejects(F action, const char *message) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  check(rejected, message);
}
bool exact(const torch::Tensor &a, const torch::Tensor &b) {
  const auto x = a.detach().to(torch::kCPU).contiguous();
  const auto y = b.detach().to(torch::kCPU).contiguous();
  return x.scalar_type() == y.scalar_type() && x.sizes() == y.sizes() &&
      (!x.numel() || !std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.numel()*x.element_size()));
}
struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    const auto pattern = (fs::temp_directory_path()/"rpb-v22-input-XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end()); name.push_back('\0');
    const auto created = ::mkdtemp(name.data()); check(created != nullptr, "private checkpoint fixture");
    path = created;
  }
  ~TemporaryDirectory() { std::error_code error; fs::remove_all(path,error); }
};
std::vector<char> bytes(const fs::path &path) {
  std::ifstream f(path,std::ios::binary); check(f.good(), "read saved artificial checkpoint");
  return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
cuwacunu::data::sample_t sample(const enc::Config &c) {
  const auto o = torch::TensorOptions().dtype(torch::kFloat64);
  const auto t = torch::arange(c.history_length,o).view({1,1,c.history_length,1});
  const auto channel = torch::arange(c.channel_count,o).view({1,c.channel_count,1,1});
  const auto f = torch::arange(c.input_width,o).view({1,1,1,c.input_width});
  const auto b = torch::arange(2,o).view({2,1,1,1});
  cuwacunu::data::sample_t out;
  out.features = (t*.07+channel*.13+f*.17+b*.23).sin() +
      .4*(t*.31-channel*.17+f*.11).cos() + f*.03;
  out.mask = torch::ones({2,c.channel_count,c.history_length},torch::kBool);
  out.mask[0][1][7] = false;
  out.features[0][1][7][5] = std::numeric_limits<double>::quiet_NaN();
  out.features[1][2][11][8] = std::numeric_limits<double>::quiet_NaN();
  const auto keys = torch::arange(c.history_length,o)*c.sampling_interval+1700000000.;
  out.past_keys = keys.view({1,1,c.history_length}).expand({2,c.channel_count,c.history_length}).clone();
  out.future_features = torch::ones({2,c.channel_count,2,c.input_width},o);
  out.future_mask = torch::ones({2,c.channel_count,2},torch::kBool);
  out.future_keys = torch::zeros({2,c.channel_count,2},o);
  return out;
}
} // namespace

int main(int argc, char **argv) try {
  check(torch::cuda::is_available(), "actual CUDA required"); torch::set_num_threads(1);
  check(argc == 2, "explicit v22 settings path required");
  auto c = enc::read_settings(argv[1]);
  check(c.channel_count == 3 && c.history_length == 128 && c.input_width == 9 && c.export_width == 32,
        "v22 file defaults are C3/H128/F9/D32");
  (void)v20::read_settings("config/encoder_v20.conf");
  const auto primary = enc::read_settings("config/encoder.conf");
  const auto preserved = enc::read_settings("config/encoder_v22.conf");
  check(enc::implementation::configurable_checkpoint_detail::same_config(c,primary) &&
            enc::implementation::configurable_checkpoint_detail::same_config(c,preserved),
        "primary and preserved explicit v22 files select identical geometry/settings");
  rejects([&] { enc::read_settings("config/encoder_v20.conf"); }, "v22 reader refuses explicit v20 settings");
  rejects([&] { v20::read_settings(argv[1]); }, "v20 reader refuses v22 settings");
  c.history_length = 129; c.channel_ids = {10,20,30}; c.sampling_interval = 60.;
  c.encoder_width = 16; c.num_layers = 1; c.feedforward_width = 32; c.decoder_hidden_width = 16;
  enc::validate_config(c);
  auto batch = sample(c); const auto ids = torch::tensor(c.channel_ids,torch::kInt64);
  const auto input = integration::configurable_encoder_input(batch,ids,c);
  const auto observed = batch.mask.unsqueeze(-1).expand_as(batch.features).logical_and(torch::isfinite(batch.features));
  check(exact(input.observed,observed) && !input.observed[0][1][7].any().item<bool>() &&
            !input.observed[1][2][11][8].item<bool>() && input.observed[1][2][11][7].item<bool>(),
        "row gaps and nonfinite coordinates retain finite siblings across all nine features");
  for (int64_t f = 0; f < 9; ++f)
    check(exact(input.data.select(3,f),torch::where(observed.select(3,f),batch.features.select(3,f),
                                                 torch::zeros_like(batch.features.select(3,f)))),
          "every ordered kline feature reaches the shared adapter without reduction");
  auto changed_future = batch;
  changed_future.future_features = torch::full_like(batch.future_features,std::numeric_limits<double>::infinity());
  changed_future.future_mask = torch::zeros_like(batch.future_mask);
  changed_future.future_keys = torch::full_like(batch.future_keys,-999.);
  const auto same = integration::configurable_encoder_input(changed_future,ids,c);
  check(exact(input.data,same.data) && exact(input.observed,same.observed) && exact(input.endpoints,same.endpoints),
        "future values/masks/timestamps never enter v22 input");
  auto misaligned = batch; misaligned.past_keys = batch.past_keys.clone(); misaligned.past_keys.select(1,1).add_(1);
  rejects([&] { integration::configurable_encoder_input(misaligned,ids,c); }, "misaligned channel grids remain rejected");
  const auto scaler = enc::fit_train_scaler(input,c);
  const auto scaled = enc::transform(scaler,input,c);
  check(scaled.data.is_cuda() && scaled.observed.is_cuda() && scaled.data.size(2) == 129,
        "TRAIN scaling transfers logical nondivisible history to CUDA");
  torch::manual_seed(220012); enc::Encoder model(c);
  torch::Tensor before;
  {
    model->eval(); torch::NoGradGuard no_grad;
    before = enc::native_embedding(model->encode(scaled),c).clone();
    check(before.is_cuda() && before.sizes() == torch::IntArrayRef({2,32}) && torch::isfinite(before).all().item<bool>(),
          "primary facade serves finite configured CUDA native values");
    check(exact(before,model->base->encode(scaled).z_contextual_global), "zero-start native matches the preserved v20 child bytes");
  }
  const auto initial_relation = model->relation_projection->weight.detach().clone();
  torch::optim::AdamW optimizer(model->parameters(),torch::optim::AdamWOptions(.001).weight_decay(.0001));
  model->train(); optimizer.zero_grad();
  const auto masks = enc::make_training_mask(scaled.observed,c,220013);
  const auto output = model->forward(scaled,masks.hidden);
  check(output.loss.is_cuda() && output.reconstruction.is_cuda() &&
            output.reconstruction.sizes() == scaled.data.sizes() && output.eligible_example_count > 0 &&
            torch::isfinite(output.loss).item<bool>(), "masked waveform Huber and logical output execute on CUDA");
  output.loss.backward();
  const auto gradient = model->relation_projection->weight.grad();
  check(gradient.defined() && gradient.is_cuda() && torch::isfinite(gradient).all().item<bool>() &&
            gradient.abs().sum().item<double>() > 0, "new content weights receive actual finite nonzero CUDA gradients");
  for (const auto &p : model->named_parameters()) {
    check(p.value().is_cuda(), "all model parameters are CUDA");
    if (p.value().grad().defined()) check(p.value().grad().is_cuda() && torch::isfinite(p.value().grad()).all().item<bool>(),
                                       "all actual parameter gradients are finite CUDA");
  }
  torch::nn::utils::clip_grad_norm_(model->parameters(),1.); optimizer.step();
  check(!exact(initial_relation,model->relation_projection->weight), "one artificial optimizer update changes content weights");
  model->validate_state(); model->validate_fixed_prior(); model->eval(); torch::NoGradGuard no_grad;
  const auto native = enc::native_embedding(model->encode(scaled),c);
  const auto L = enc::shape_width(c), fixed = enc::spectral_width(c);
  check(exact(before.narrow(1,L,fixed),native.narrow(1,L,fixed)), "literal spectral tail survives the update exactly");
  auto poison = scaled; poison.data = torch::where(scaled.observed,scaled.data,
      torch::full_like(scaled.data,std::numeric_limits<float>::quiet_NaN()));
  check(exact(native,enc::native_embedding(model->encode(poison),c)), "masked storage poison cannot enter the active learned route");
  auto missing = poison; missing.observed = torch::zeros_like(scaled.observed);
  const auto empty = model->encode(missing);
  check(empty.z_contextual_global.eq(0).all().item<bool>() && !empty.sample_valid_mask.any().item<bool>(),
        "allmissing remains finite invalid zero after the update");
  TemporaryDirectory fixture; const auto file = fixture.path/"v22.pt";
  enc::save_checkpoint(file.string(),model,scaler,1); const auto saved = bytes(file);
  rejects([&] { enc::save_checkpoint(file.string(),model,scaler,2); }, "immutable typed save refuses replacement");
  check(bytes(file) == saved, "rejected replacement preserves checkpoint bytes");
  auto loaded = enc::load_checkpoint(file.string(),c.device,&c);
  check(loaded.training_updates == 1 && loaded.scaler.identity() == scaler.identity() &&
            exact(native,enc::native_embedding(loaded.model->encode(scaled),loaded.config)) &&
            exact(model->decode(native,ids),loaded.model->decode(native,ids)) &&
            exact(model->relation_projection->weight,loaded.model->relation_projection->weight),
        "v22 geometry/scaler/updated weights/native/reconstruction reload exactly");
  auto wrong = c; wrong.history_length = 130;
  rejects([&] { enc::load_checkpoint(file.string(),c.device,&wrong); }, "same padded but different logical history refuses load");
  rejects([&] { enc::load_checkpoint(file.string(),torch::kCPU); }, "CPU execution refuses load");
  rejects([&] { v20::load_checkpoint(file.string(),c.device,&c); }, "historical v20 typed loader refuses v22 artifact");
  const auto base_file = fixture.path/"v20.pt";
  v20::save_checkpoint(base_file.string(),model->base,scaler,1);
  rejects([&] { enc::load_checkpoint(base_file.string(),c.device,&c); }, "primary v22 typed loader refuses v20 artifact");
  torch::cuda::synchronize();
  std::cout << "PASS primary RPB-v22 API: nine-feature shared adapter, masks/future exclusion, H129 CUDA native, "
      "zero-start v20 bytes, one actual update, preserved fixed tail and immutable typed reload; no quality/head execution\n";
  return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
