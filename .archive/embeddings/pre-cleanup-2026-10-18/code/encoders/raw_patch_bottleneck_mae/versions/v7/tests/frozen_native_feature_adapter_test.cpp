// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_native_feature_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "rpb_test_support.h"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#ifndef FROZEN_NATIVE_FEATURE_SOURCE_ID
#ifdef EVALUATION_SOURCE_ID
#define FROZEN_NATIVE_FEATURE_SOURCE_ID EVALUATION_SOURCE_ID
#else
#define FROZEN_NATIVE_FEATURE_SOURCE_ID "unrecorded"
#endif
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
using namespace rpb_test;
using Policy = rpb::FrozenDecoderParentPolicy;

std::string bytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary); check(bool(in), "test-owned file readable");
  std::ostringstream out; out << in.rdbuf(); return out.str();
}
struct RuntimeWitness {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeWitness() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(
          at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  void unchanged(const std::string &label) const {
    check(threads == at::get_num_threads(), label + " thread count unchanged");
    for (size_t i = 0; i < states.size(); ++i)
      check(torch::equal(states[i], generators[i].get_state()), label + " ambient RNG unchanged");
  }
};
struct FixtureIsolation {
  RuntimeWitness before;
  ~FixtureIsolation() noexcept {
    try {
      for (size_t i = 0; i < before.states.size(); ++i)
        before.generators[i].set_state(before.states[i]);
      at::set_num_threads(before.threads);
    } catch (...) { std::terminate(); }
  }
};
template<class Function> auto fixture_call(Function function) {
  // Historical trainer/model setup owns its seeds. This isolation covers only
  // artificial fixture generation, not any NEW factory/extract/save callback.
  const FixtureIsolation isolate; return function();
}
torch::Tensor read(torch::serialize::InputArchive &in, const std::string &key) {
  torch::Tensor value; in.read(key, value, true); return value;
}
std::string text(torch::serialize::InputArchive &in, const std::string &key) {
  return embedding::archive::tensor_text(read(in, key));
}
void exact(const torch::Tensor &left, const torch::Tensor &right, const std::string &why) {
  const auto a = left.detach().to(torch::kCPU).contiguous(), b = right.detach().to(torch::kCPU).contiguous();
  check(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() &&
      (!a.numel() || std::memcmp(a.const_data_ptr(), b.const_data_ptr(), a.numel() * a.element_size()) == 0), why);
}
rpb::FrozenNativeFeatureOptions options(const std::string &path, Policy policy, int64_t updates) {
  torch::serialize::InputArchive audit; audit.load_from(path + ".audit.pt", torch::kCPU);
  return {path, policy, updates, text(audit, "core_writer_source_fingerprint"),
      text(audit, "training_producer_source_fingerprint")};
}
std::string clone_parent(const fs::path &directory, const std::string &source, const std::string &name) {
  const auto destination = (directory / name).string();
  for (const std::string suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt"})
    check(fs::copy_file(source + suffix, destination + suffix), "new malformed fixture copy");
  return destination;
}
void changed_audit(const std::string &path, const std::string &key, const torch::Tensor &value) {
  torch::serialize::InputArchive input; input.load_from(path + ".audit.pt", torch::kCPU);
  torch::serialize::OutputArchive out;
  for (const auto &name : input.keys()) out.write(name, name == key ? value : read(input, name), true);
  embedding::archive::save_archive(path + ".audit.pt", out);
}
void same_initialized_model(const rpb::Checkpoint &left, const rpb::Checkpoint &right) {
  const auto a = left.model->named_parameters(), b = right.model->named_parameters();
  check(a.size() == b.size(), "full initialized parameter name count");
  int64_t count = 0;
  for (const auto &p : a) { exact(p.value(), b[p.key()], "exact v4/v7 initialized parameter " + p.key()); count += p.value().numel(); }
  check(count == 225805, "complete initialized architecture");
  const auto c = left.model->named_buffers(), d = right.model->named_buffers();
  check(c.size() == d.size(), "initialized buffer count");
  for (const auto &p : c) exact(p.value(), d[p.key()], "initialized buffer " + p.key());
  check(left.scaler.identity() == right.scaler.identity(), "paired original scaler identity");
  for (const auto &[x, y] : std::vector<std::pair<torch::Tensor, torch::Tensor>>{
      {left.scaler.mean,right.scaler.mean}, {left.scaler.scale,right.scaler.scale},
      {left.scaler.count,right.scaler.count}, {left.scaler.channel_ids,right.scaler.channel_ids},
      {left.scaler.floor_applied,right.scaler.floor_applied}}) exact(x,y,"all initialized scaler tensors");
}
void cuda_contract() {
  auto settings = rpb::default_settings(); settings.model.device = torch::Device(torch::kCUDA, 0);
  settings.model.channel_mixer_layers = 1; settings.model.global_bottleneck_mode = 2;
  settings.model.export_width = 32; settings.model.dropout = 0;
  settings.steps = 4; settings.batch_size = 2; settings.log_every = 1; settings.attempt_limit = 16;
  auto raw = input(settings.model, 8); raw.observed[0][1][7][2] = false;
  raw.data.masked_fill_(raw.observed.logical_not(), 0);
  const embedding::Batch legal{raw.data.clone(),raw.observed.clone()};
  ev::ProviderFitInput fit{legal, {3,32,3,torch::kFloat64,torch::kCPU}, 9109,
      {"source-a","source-a","source-b","source-b","source-c","source-c","source-d","source-d"},
      {0,1,2}, "unitless,unitless,unitless", "fresh-decoder-replication-v1/lag_sign", 1.,31.};
  const auto directory = fs::path(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") /
      ("rpb-frozen-native-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(directory), "exclusive synthetic CUDA fixture directory");
  const RuntimeWitness overall;
  auto ordinary = fixture_call([&] { return rpb::make_learning_curve_trainer(settings)(fit); });
  auto lighter = fixture_call([&] { return rpb::make_learning_curve_trainer(
      settings, {true,rpb::ContextDeletionRecipe::coordinate15_v1})(fit); });
  const auto zero4 = (directory / "v4-zero.pt").string(), zero7 = (directory / "v7-zero.pt").string();
  ordinary.save_checkpoint(zero4); lighter.save_checkpoint(zero7);
  const auto initial4 = fixture_call([&] { return rpb::load_checkpoint(zero4,settings.model.device); });
  const auto initial7 = fixture_call([&] { return rpb::load_checkpoint(zero7,settings.model.device); });
  same_initialized_model(initial4,initial7);
  const RuntimeWitness callbacks;
  auto frozen_zero = rpb::make_frozen_native_feature_provider(options(zero4,Policy::ordinary_v4,0),fit);
  const auto zero_values = frozen_zero.extract(legal).at("curve_global");
  const auto old_zero = rpb::make_fresh_initial_snapshot(zero4,Policy::ordinary_v4,fit);
  exact(zero_values.values,old_zero.features.extract(legal).at("curve_global").values,"same CUDA point0 public-path parity");
  check(frozen_zero.surfaces.size() == 1 && frozen_zero.surfaces.count("curve_global"),"native-only provider has no decoder API");
  fixture_call([&] { ordinary.train_to(2); lighter.train_to(2); });
  const auto two4 = (directory / "v4-two.pt").string(), two7 = (directory / "v7-two.pt").string();
  ordinary.save_checkpoint(two4); lighter.save_checkpoint(two7);
  std::map<std::string,std::string> protected_bytes;
  for (const auto &path : {zero4,zero7,two4,two7}) for (const std::string suffix : {"", ".audit.pt", ".scaler.pt", ".training-raw.pt"})
    protected_bytes.emplace(path+suffix,bytes(path+suffix));
  for (const auto &[path,policy] : std::vector<std::pair<std::string,Policy>>{{two4,Policy::ordinary_v4},{two7,Policy::coordinate15_v7}}) {
    const auto declared = options(path,policy,2);
    auto provider = rpb::make_frozen_native_feature_provider(declared,fit);
    const auto baseline = provider.extract(legal).at("curve_global");
    check(baseline.values.device().is_cpu() && baseline.values.scalar_type() == torch::kFloat32 &&
        baseline.values.sizes() == torch::IntArrayRef({8,32}) && baseline.valid.all().item<bool>(),"actual CUDA native32 returned as finite CPU evidence");
    // Separate artificial old CUDA path. Its optimizer is test-owned; the NEW
    // provider itself is never a calibrator and constructs no optimizer.
    auto old = rpb::make_frozen_decoder_calibration({path,policy,2,1},fit);
    old.train_to(0); const auto old_asset = (directory/(policy == Policy::ordinary_v4 ? "old-v4.pt" : "old-v7.pt")).string();
    old.save(old_asset); const auto old_snapshot = old.snapshot(old_asset);
    exact(baseline.values,old_snapshot.features.extract(legal).at("curve_global").values,"same CUDA positive bare-parent/old snapshot parity");
    exact(baseline.valid,old_snapshot.features.extract(legal).at("curve_global").valid,"same CUDA original support parity");
    embedding::Batch transfer{legal.data.mul(2.),legal.feature_mask.clone()};
    const auto transferred = provider.extract(transfer).at("curve_global");
    check(torch::isfinite(transferred.values).all().item<bool>(),"new observations use the original frozen timing scaler");
    exact(baseline.values,provider.extract(legal).at("curve_global").values,"repeat stability after transfer extraction");
    auto absent = transfer; absent.data = torch::zeros_like(transfer.data); absent.feature_mask = torch::zeros_like(transfer.feature_mask);
    const auto missing = provider.extract(absent).at("curve_global");
    check(!missing.valid.any().item<bool>() && missing.values.eq(0).all().item<bool>(),"all-absent native support and zero storage");
    auto negative = declared; negative.expected_parent_updates = -1;
    rejects([&] { rpb::make_frozen_native_feature_provider(negative,fit); },"unspecified budget");
    negative = declared; negative.expected_parent_updates = 4;
    rejects([&] { rpb::make_frozen_native_feature_provider(negative,fit); },"wrong original completed budget");
    negative = declared; negative.parent_policy = static_cast<Policy>(-1);
    rejects([&] { rpb::make_frozen_native_feature_provider(negative,fit); },"unspecified original policy");
    negative = declared; negative.parent_policy = policy == Policy::ordinary_v4 ? Policy::coordinate15_v7 : Policy::ordinary_v4;
    rejects([&] { rpb::make_frozen_native_feature_provider(negative,fit); },"wrong original v4/v7 policy");
    negative = declared; negative.expected_parent_core_source_fingerprint = std::string(64,'0');
    rejects([&] { rpb::make_frozen_native_feature_provider(negative,fit); },"forged original core source");
    negative = declared; negative.expected_training_producer_source_fingerprint = std::string(64,'0');
    rejects([&] { rpb::make_frozen_native_feature_provider(negative,fit); },"forged original trainer source");
    auto wrong_fit = fit; wrong_fit.seed++;
    rejects([&] { rpb::make_frozen_native_feature_provider(declared,wrong_fit); },"new data master substituted as original master");
    wrong_fit = fit; wrong_fit.protocol_id = "frozen-amplitude-transfer-v1/amplitude";
    rejects([&] { rpb::make_frozen_native_feature_provider(declared,wrong_fit); },"new task namespace substituted as original fit");
    wrong_fit = fit; wrong_fit.training_observations = transfer;
    rejects([&] { rpb::make_frozen_native_feature_provider(declared,wrong_fit); },"amplitude observations substituted as original TRAIN");
    wrong_fit = fit; std::swap(wrong_fit.training_source_ids[0],wrong_fit.training_source_ids[2]);
    rejects([&] { rpb::make_frozen_native_feature_provider(declared,wrong_fit); },"original source order forgery");
    wrong_fit = fit; wrong_fit.channel_ids = {1,0,2};
    rejects([&] { rpb::make_frozen_native_feature_provider(declared,wrong_fit); },"original semantic order forgery");
    wrong_fit = fit; wrong_fit.shape.device = torch::kCUDA;
    rejects([&] { rpb::make_frozen_native_feature_provider(declared,wrong_fit); },"non-CPU original metadata");
    const auto asset_directory = directory/(policy == Policy::ordinary_v4 ? "audit-v4" : "audit-v7");
    check(fs::create_directory(asset_directory),"new encoder-only audit directory"); provider.save_assets(asset_directory.string());
    torch::serialize::InputArchive audit; audit.load_from((asset_directory/rpb::kFrozenNativeFeatureAuditFile).string(),torch::kCPU);
    check(text(audit,"artifact_kind") == rpb::kFrozenNativeFeatureArtifact && text(audit,"protocol_id") == rpb::kFrozenNativeFeatureProtocol && text(audit,"no_optimizer_created") == "true" &&
        text(audit,"encoder_updates") == "0" && text(audit,"decoder_updates") == "0" &&
        text(audit,"snapshot_loader_source_fingerprint") == FROZEN_NATIVE_FEATURE_SOURCE_ID &&
        text(audit,"parent_writer_source_fingerprint") == declared.expected_parent_core_source_fingerprint &&
        text(audit,"parent_training_producer_source_fingerprint") == declared.expected_training_producer_source_fingerprint,
        "encoder-only artifact and separate original/new producer scopes");
    check(!fs::exists(asset_directory/"decoder-calibration-snapshot-audit.pt"),"no decoder calibration audit masquerade");
    rejects([&] { provider.save_assets(asset_directory.string()); },"immutable audit output");
  }
  fixture_call([&] { ordinary.train_to(4); lighter.train_to(4); });
  const auto four4 = (directory/"v4-four.pt").string(); ordinary.save_checkpoint(four4);
  auto frozen_four = rpb::make_frozen_native_feature_provider(options(four4,Policy::ordinary_v4,4),fit);
  check(frozen_four.extract(legal).at("curve_global").valid.all().item<bool>(),"generic explicit nonnegative engineering budget4");
  exact(zero_values.values,frozen_zero.extract(legal).at("curve_global").values,"point0 immutable after separate live trainer advances");
  const auto bad_raw = clone_parent(directory,two4,"bad-raw.pt");
  auto changed_input = raw; changed_input.data = raw.data.clone(); changed_input.data[0][0][0][0] += .25;
  rpb::save_dataset(bad_raw+".training-raw.pt",rpb::describe_dataset(changed_input,settings.model,fit.feature_units));
  rejects([&] { rpb::make_frozen_native_feature_provider(options(bad_raw,Policy::ordinary_v4,2),fit); },"original TRAIN raw companion forgery");
  const auto bad_scaler = clone_parent(directory,two4,"bad-scaler.pt");
  auto loaded = fixture_call([&] { return rpb::load_checkpoint(two4,settings.model.device); });
  loaded.scaler.mean = loaded.scaler.mean.clone(); loaded.scaler.mean[0][0] += .125;
  rpb::save_scaler(bad_scaler+".scaler.pt",loaded.scaler,settings.model,loaded.schema_id,loaded.dataset_id);
  rejects([&] { rpb::make_frozen_native_feature_provider(options(bad_scaler,Policy::ordinary_v4,2),fit); },"original frozen scaler forgery");
  const auto bad_policy = clone_parent(directory,two7,"bad-policy.pt");
  changed_audit(bad_policy,"context_deletion_ratio_value",torch::tensor(.30,torch::kFloat64));
  rejects([&] { rpb::make_frozen_native_feature_provider(options(bad_policy,Policy::coordinate15_v7,2),fit); },"typed coordinate15 companion forgery");
  const auto bad_weight = clone_parent(directory,two4,"bad-weight.pt");
  auto nonfinite = fixture_call([&] { return rpb::load_checkpoint(two4,settings.model.device); });
  fixture_call([&] {
    torch::NoGradGuard no_grad; nonfinite.model->parameters().front().flatten()[0] = std::numeric_limits<float>::infinity();
    torch::optim::AdamW optimizer(nonfinite.model->parameters(),torch::optim::AdamWOptions(settings.learning_rate));
    rpb::save_checkpoint(bad_weight,nonfinite,optimizer);
  });
  rejects([&] { rpb::make_frozen_native_feature_provider(options(bad_weight,Policy::ordinary_v4,2),fit); },"nonfinite CUDA checkpoint parameters");
  for (const auto &[path,original] : protected_bytes) check(bytes(path) == original,"all original fixture parent bytes unchanged");
  callbacks.unchanged("new frozen native callback lifecycle"); overall.unchanged("bounded engineering runtime lifecycle");
  std::cout << "Generated artificial frozen-native fixtures at " << directory << '\n';
}
} // namespace

int main() {
  try {
    check(torch::cuda::is_available(),"CUDA required for frozen native feature admission; no CPU fallback");
    cuda_contract();
    std::cout << "Frozen native feature CUDA admission passed\n" << FROZEN_NATIVE_FEATURE_SOURCE_ID << '\n';
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
