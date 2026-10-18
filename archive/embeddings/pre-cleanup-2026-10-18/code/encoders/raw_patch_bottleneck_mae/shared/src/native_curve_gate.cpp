// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/native_curve_gate.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/feature_harness.h"
#include <ATen/Context.h>
#include <torch/cuda.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;
constexpr uint64_t preflight_seed = 424243;

void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error("[RPB learned-global CUDA gate] " + message);
}
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                         << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
std::string source_ids_json(const std::vector<std::string> &ids) {
  std::ostringstream out; out << '[';
  for (size_t i = 0; i < ids.size(); ++i) { if (i) out << ','; out << quote(ids[i]); }
  return out.str() + ']';
}
void save_text(const fs::path &path, const std::string &text) {
  require(!fs::exists(path) && !fs::is_symlink(fs::symlink_status(path)), "refusing an existing output");
  std::ofstream out(path, std::ios::binary);
  require(bool(out), "cannot create gate artifact");
  out << text << '\n'; out.close(); require(bool(out), "cannot save gate artifact");
}
void save_split(const fs::path &path, const ev::ControlledDataset &split) {
  torch::serialize::OutputArchive out;
  out.write("observed", split.observed.data, true);
  out.write("feature_mask", split.observed.feature_mask, true);
  out.write("labels_scoring_only", split.labels, true);
  out.write("source_ids_json", embedding::archive::text_tensor(source_ids_json(split.source_ids)), true);
  embedding::archive::save_archive(path.string(), out);
}
void save_features(const fs::path &path, const ev::FeatureSurface &surface) {
  torch::serialize::OutputArchive out;
  out.write("features", surface.values, true); out.write("valid", surface.valid, true);
  out.write("provenance", embedding::archive::text_tensor(surface.provenance), true);
  embedding::archive::save_archive(path.string(), out);
}
struct RuntimeIsolation {
  int old_threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try {
      for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]);
      at::set_num_threads(old_threads);
    } catch (...) { std::terminate(); }
  }
};
Input raw_input(const embedding::Batch &batch, const Config &config) {
  return {batch.data, batch.feature_mask, torch::tensor(resolved_channel_ids(config), torch::kInt64),
          torch::full({batch.data.size(0)}, (config.history_length - 1) * config.sampling_interval, torch::kFloat64),
          config.sampling_interval};
}
void freeze(Checkpoint &checkpoint) {
  checkpoint.model->eval();
  for (auto &parameter : checkpoint.model->parameters()) parameter.set_requires_grad(false);
}
void same_features(const ev::FeatureMap &before, const ev::FeatureMap &after, const std::string &label) {
  require(before.size() == after.size(), label + " changed feature count");
  for (const auto &[name, feature] : before) {
    require(after.count(name), label + " changed surface keys");
    const auto &other = after.at(name);
    require(torch::equal(feature.values, other.values) && torch::equal(feature.valid, other.valid)
            && feature.provenance == other.provenance, label + " changed values/support/provenance");
  }
}
void trace_prefix(const ev::CurveProgress &before, const ev::CurveProgress &after) {
  require(before.losses.size() <= after.losses.size(), "loss trace shrank");
  for (size_t i = 0; i < before.losses.size(); ++i) {
    const auto &a = before.losses[i], &b = after.losses[i];
    require(a.attempted == b.attempted && a.completed == b.completed && a.target_cells == b.target_cells
            && a.loss == b.loss && a.gradient_norm == b.gradient_norm, "loss trace lost its exact prefix");
  }
}
void optimizer_cuda(Checkpoint &checkpoint, const std::string &path, int64_t steps) {
  torch::optim::AdamW optimizer(checkpoint.model->parameters(),
      torch::optim::AdamWOptions(checkpoint.settings.learning_rate).weight_decay(checkpoint.settings.weight_decay));
  load_optimizer(path, optimizer, checkpoint.settings.model.device);
  require(!optimizer.state().empty(), "ordinary checkpoint lacks AdamW state");
  for (const auto &[key, state] : optimizer.state()) {
    (void)key;
    const auto *adam = dynamic_cast<const torch::optim::AdamWParamState *>(state.get());
    require(adam && adam->step() == steps && adam->exp_avg().is_cuda() && adam->exp_avg_sq().is_cuda()
            && torch::isfinite(adam->exp_avg()).all().item<bool>()
            && torch::isfinite(adam->exp_avg_sq()).all().item<bool>(), "AdamW step/moments/device differ from continuous updates");
  }
}
void cpu_export_parity(const ev::CurveSnapshot &snapshot, const std::string &path,
                       const embedding::Batch &batch) {
  const auto features = snapshot.features.extract(batch);
  require(features.size() == 2 && features.count("curve_global") && features.count("curve_channel_concatenation"),
          "snapshot did not preserve canonical surface aliases");
  const auto &description = snapshot.features.surfaces.at("curve_global");
  require(description.kind == ev::SurfaceKind::global && description.channel_order.empty(), "primary export is not typed global");
  auto checkpoint = load_checkpoint(path, torch::kCPU); freeze(checkpoint);
  torch::NoGradGuard no_grad;
  const auto encoding = checkpoint.model->encode(checkpoint.scaler.transform(raw_input(batch, checkpoint.settings.model), checkpoint.settings.model));
  const auto served = compact_reconstruction_export(encoding, checkpoint.settings.model);
  require(served.device().is_cpu() && served.sizes() == torch::IntArrayRef({batch.data.size(0), 32})
          && torch::equal(features.at("curve_global").values, served)
          && torch::equal(features.at("curve_global").valid, encoding.sample_valid_mask), "exact CPU checkpoint native32 export parity failed");
  require(torch::equal(features.at("curve_channel_concatenation").values, encoding.z_contextual.flatten(1))
          && torch::equal(features.at("curve_channel_concatenation").valid, encoding.channel_valid_mask.all(1)),
          "diagnostic concatenation differs from checkpoint observed channel vectors");
}
} // namespace

std::string run_native_curve_cuda_gate(const Settings &settings, const std::string &output_directory) {
  validate_settings(settings);
  require(settings.model.channel_mixer_placement == 0,
      "historical native curve gate rejects early channel mixer placement");
  require(settings.model.device.is_cuda() && torch::cuda::is_available(), "explicit CUDA required; no CPU fallback");
  require(settings.model.channel_mixer_layers == 1 &&
          (settings.model.global_bottleneck_mode == 2 || settings.model.global_bottleneck_mode == 3)
          && settings.model.export_width == 32, "learned-global gate requires mixer1/mode2-or3/native32");
  const bool direct_patch = settings.model.global_bottleneck_mode == 3;
  const std::string tag = direct_patch ? "RPB-v5" : "RPB-v4";
  const std::string gate_protocol = direct_patch ? "native-rpb-v5-cuda-gate-v1" : "native-rpb-v4-cuda-gate-v1";
  require(settings.model.channel_count >= 2 && settings.model.history_length >= 8
          && settings.model.history_length / settings.model.patch_length >= 3,
          "preflight needs lag-compatible geometry and three observed patch groups");
  require(settings.attempt_limit >= 4, "preflight requires at least four attempted batches");
  const std::string producer = EVALUATION_SOURCE_ID;
  require(producer.size() == 64 && producer.find_first_not_of("0123456789abcdef") == std::string::npos,
          "gate must be compiled with an explicit source fingerprint");
  const fs::path directory(output_directory);
  require(!output_directory.empty() && !fs::exists(directory) && !fs::is_symlink(fs::symlink_status(directory)),
          "gate requires a NEW output directory");
  auto gate_settings = settings;
  gate_settings.steps = 4; gate_settings.seed = static_cast<int64_t>(preflight_seed); gate_settings.checkpoint_every = 0;
  validate_settings(gate_settings);
  RuntimeIsolation isolation; at::set_num_threads(settings.threads);
  fs::create_directories(directory);
  const auto &config = gate_settings.model;
  const embedding::input_shape_t shape{config.channel_count, config.history_length, config.input_width, torch::kFloat64, torch::kCPU};
  std::string units;
  for (int64_t f = 0; f < config.input_width; ++f) { if (f) units += ','; units += "unitless"; }
  save_text(directory / "gate-card.json", "{\"version\":1,\"protocol\":" + quote(gate_protocol) + ",\"embedding_tag\":" + quote(tag) + ","
      "\"engineering_only\":true,\"preflight_seed\":424243,\"train_pairs\":4,\"validation_pairs\":2,\"test_generation\":false,"
      "\"training_label_access\":false,\"classifier_fitting\":false,\"main_experiment_training\":false,\"milestones\":[0,2,4],"
      "\"missing_rate\":0.1,\"settings\":" + quote(settings_text(gate_settings)) + ",\"requested_settings\":" + quote(settings_text(settings)) +
      ",\"gate_producer_source_fingerprint\":" + quote(producer) + '}');
  const auto protocol = ev::make_controlled_development_protocol(ev::Task::lag_sign, shape, 4, 2, preflight_seed, 0.1);
  require(!protocol.testing.observed.data.defined() && protocol.testing.source_ids.empty(), "preflight generated TEST data");
  save_split(directory / "controlled-training.pt", protocol.training);
  save_split(directory / "controlled-validation.pt", protocol.validation);
  ev::ProviderFitInput fit{protocol.training.observed, shape, preflight_seed, protocol.training.source_ids,
      resolved_channel_ids(config), units, gate_protocol, config.sampling_interval,
      (config.history_length - 1) * config.sampling_interval};
  auto trainer = make_learning_curve_trainer(gate_settings)(fit);
  const auto zero = trainer.train_to(0);
  require(zero.attempted == 0 && zero.completed == 0 && zero.sampled_rows == 0 && zero.training_seconds == 0
          && !zero.weights_changed && zero.parameter_count > 0 && zero.parameter_count == zero.cuda_parameter_count,
          "point zero is not exact untrained CUDA initialization");
  const auto zero_path = (directory / "point-0.pt").string(); trainer.save_checkpoint(zero_path);
  const auto snapshot_zero = trainer.snapshot(zero_path);
  const auto zero_features = snapshot_zero.features.extract(protocol.validation.observed);
  cpu_export_parity(snapshot_zero, zero_path, protocol.validation.observed);
  auto cpu_zero = load_checkpoint(zero_path, torch::kCPU); freeze(cpu_zero);
  auto cpu_config = config; cpu_config.device = torch::kCPU;
  torch::manual_seed(training_detail::mixed(preflight_seed ^ 0x7270622d696e6974ULL));
  Model initialized(cpu_config);
  for (size_t i = 0; i < initialized->parameters().size(); ++i)
    require(torch::equal(initialized->parameters()[i], cpu_zero.model->parameters()[i]), "saved point zero differs from hashed CPU initialization");
  const auto two = trainer.train_to(2);
  require(two.completed == 2 && two.attempted >= 2 && two.sampled_rows == two.attempted * gate_settings.batch_size
          && two.last_input_cuda && two.last_loss_cuda && two.finite_gradients && two.weights_changed,
          "first continuous CUDA segment failed");
  const auto two_path = (directory / "point-2.pt").string(); trainer.save_checkpoint(two_path);
  const auto snapshot_two = trainer.snapshot(two_path);
  const auto two_features = snapshot_two.features.extract(protocol.validation.observed);
  cpu_export_parity(snapshot_two, two_path, protocol.validation.observed);
  const int64_t B = protocol.validation.observed.data.size(0), C = config.channel_count;
  const auto patches = protocol.validation.observed.feature_mask.reshape({B, C, config.history_length / config.patch_length,
                                                                         config.patch_length * config.input_width}).any(-1);
  const auto query_eligible = patches.sum(-1).ge(3).logical_and(patches.select(2, 0));
  require(query_eligible.any().item<bool>(), "fixed preflight reconstruction query has no eligible channel");
  auto hidden = torch::zeros_like(protocol.validation.observed.feature_mask);
  hidden.narrow(2, 0, config.patch_length).copy_(query_eligible.unsqueeze(-1).unsqueeze(-1)
      .expand({B, C, config.patch_length, config.input_width}));
  const auto two_reconstruction = snapshot_two.reconstruct(protocol.validation.observed, hidden);
  const auto four = trainer.train_to(4);
  require(four.completed == 4 && four.attempted >= two.attempted && four.sampled_rows == four.attempted * gate_settings.batch_size
          && four.parameter_count == four.cuda_parameter_count && four.last_input_cuda && four.last_loss_cuda
          && four.finite_gradients && four.weights_changed && four.training_seconds >= two.training_seconds
          && four.preprocessing_id == zero.preprocessing_id && four.training_dataset_id == zero.training_dataset_id,
          "second continuous CUDA segment or frozen scaler identity failed");
  trace_prefix(two, four);
  const auto four_path = (directory / "point-4.pt").string(); trainer.save_checkpoint(four_path);
  const auto snapshot_four = trainer.snapshot(four_path);
  const auto four_features = snapshot_four.features.extract(protocol.validation.observed);
  cpu_export_parity(snapshot_four, four_path, protocol.validation.observed);
  auto cpu_four = load_checkpoint(four_path, torch::kCPU); freeze(cpu_four);
  std::map<std::string, bool> changed;
  const auto old_parameters = cpu_zero.model->named_parameters(), new_parameters = cpu_four.model->named_parameters();
  const std::vector<std::string> active_groups{direct_patch ? "global_patch_pool_first" : "global_pool_first",
      direct_patch ? "global_patch_pool_second" : "global_pool_second", "decoder_first", "decoder_second"};
  for (const std::string &prefix : active_groups) {
    bool any = false;
    for (const auto &parameter : new_parameters)
      if (parameter.key().rfind(prefix + '.', 0) == 0)
        any = any || !torch::equal(parameter.value(), old_parameters[parameter.key()]);
    require(any, "active learned-global/decoder weights did not change: " + prefix); changed.emplace(prefix, any);
  }
  same_features(zero_features, snapshot_zero.features.extract(protocol.validation.observed), "point-zero snapshot after training");
  same_features(two_features, snapshot_two.features.extract(protocol.validation.observed), "point-two snapshot after training");
  const auto earlier_reconstruction = snapshot_two.reconstruct(protocol.validation.observed, hidden);
  require(torch::equal(two_reconstruction.prediction, earlier_reconstruction.prediction)
          && torch::equal(two_reconstruction.target, earlier_reconstruction.target)
          && torch::equal(two_reconstruction.eligible, earlier_reconstruction.eligible), "earlier CUDA reconstruction snapshot changed");
  const auto restored_two = trainer.snapshot(two_path);
  same_features(two_features, restored_two.features.extract(protocol.validation.observed), "restored selected earlier checkpoint");
  const auto repeated = trainer.train_to(4);
  require(repeated.completed == four.completed && repeated.attempted == four.attempted
          && repeated.training_seconds == four.training_seconds, "same completed budget performed additional training");
  auto gpu_two = load_checkpoint(two_path, config.device); optimizer_cuda(gpu_two, two_path, 2);
  auto gpu_four = load_checkpoint(four_path, config.device); optimizer_cuda(gpu_four, four_path, 4); freeze(gpu_four);
  ev::CurveReconstruction reconstruction;
  torch::Tensor served, decoded;
  {
    torch::NoGradGuard no_grad;
    const auto normalized = gpu_four.scaler.transform(raw_input(protocol.validation.observed, config), gpu_four.settings.model);
    const auto expected = gpu_four.model->forward(normalized, hidden);
    served = compact_reconstruction_export(expected.encoding, gpu_four.settings.model);
    require(served.is_cuda() && served.sizes() == torch::IntArrayRef({B, 32}), "reconstruction export is not the sole CUDA native32 vector");
    decoded = gpu_four.model->decode(served, normalized.channel_ids);
    require(torch::equal(decoded, expected.reconstruction), "forward differs from decoding the exact served global vector");
    bool rejected = false;
    try { (void)gpu_four.model->decode(expected.encoding.z_contextual, normalized.channel_ids); }
    catch (const std::exception &) { rejected = true; }
    require(rejected, "global decoder accepted a per-channel bypass");
    reconstruction = snapshot_four.reconstruct(protocol.validation.observed, hidden);
    require(expected.reconstruction.is_cuda() && torch::equal(reconstruction.prediction, expected.reconstruction.to(torch::kCPU))
            && torch::equal(reconstruction.target, normalized.data.to(torch::kCPU))
            && torch::equal(reconstruction.eligible, expected.eligible_channels.to(torch::kCPU)), "exact frozen GPU checkpoint reconstruction parity failed");
    require(!torch::equal(decoded, gpu_four.model->decode(torch::zeros_like(served), normalized.channel_ids)), "decoder ignores its served global vector");
  }
  embedding::Batch changed_target{protocol.validation.observed.data.clone(), protocol.validation.observed.feature_mask.clone()};
  changed_target.data.masked_fill_(hidden.logical_and(changed_target.feature_mask), 4000);
  const auto concealed = snapshot_four.reconstruct(changed_target, hidden);
  require(torch::equal(concealed.prediction, reconstruction.prediction) && torch::equal(concealed.eligible, reconstruction.eligible)
          && !torch::equal(concealed.target, reconstruction.target), "hidden reconstruction target leaked into prediction");
  embedding::Batch absent{torch::full_like(protocol.validation.observed.data, std::numeric_limits<double>::quiet_NaN()),
                          torch::zeros_like(protocol.validation.observed.feature_mask)};
  for (const auto &[name, feature] : snapshot_four.features.extract(absent)) {
    (void)name; require(!feature.valid.any().item<bool>() && torch::equal(feature.values, torch::zeros_like(feature.values)), "all-absent export invented support");
  }
  require(!snapshot_four.reconstruct(absent, torch::zeros_like(hidden)).eligible.any().item<bool>(), "all-absent reconstruction invented eligibility");
  save_features(directory / "point-0-global-validation.pt", zero_features.at("curve_global"));
  save_features(directory / "point-2-global-validation.pt", two_features.at("curve_global"));
  save_features(directory / "point-4-global-validation.pt", four_features.at("curve_global"));
  torch::serialize::OutputArchive witness;
  witness.write("hidden", hidden, true); witness.write("prediction", reconstruction.prediction, true);
  witness.write("target", reconstruction.target, true); witness.write("eligible", reconstruction.eligible, true);
  witness.write("served_global", served.detach().to(torch::kCPU), true); witness.write("decoded_from_global", decoded.detach().to(torch::kCPU), true);
  embedding::archive::save_archive((directory / "reconstruction-witness.pt").string(), witness);
  std::ostringstream report; report << std::setprecision(17)
      << "{\"version\":1,\"protocol\":" << quote(gate_protocol) << ",\"status\":\"passed\",\"embedding_tag\":" << quote(tag) << ','
      << "\"engineering_only\":true,\"preflight_seed\":424243,\"train_rows\":8,\"validation_rows\":4,\"test_generation\":false,"
      << "\"classifier_fitting\":false,\"main_experiment_training\":false,\"milestones\":[0,2,4],\"completed\":4,\"attempted\":" << four.attempted
      << ",\"sampled_rows\":" << four.sampled_rows << ",\"parameters\":" << four.parameter_count << ",\"cuda_parameters\":" << four.cuda_parameter_count
      << ",\"input_cuda\":true,\"loss_cuda\":true,\"finite_gradients\":true,\"weights_changed\":true,\"global_bottleneck_mode\":" << config.global_bottleneck_mode << ",\"channel_mixer_layers\":1,"
      << "\"native_size\":32,\"point_zero_hashed_initialization_exact\":true,\"optimizer_moments_cuda\":true,\"optimizer_steps\":[2,4],"
      << "\"continuous_trace_prefix\":true,\"cpu_checkpoint_export_exact\":true,\"gpu_checkpoint_reconstruction_exact\":true,\"decoder_uses_exact_global\":true,"
      << "\"per_channel_decoder_bypass_rejected\":true,\"zero_global_changes_prediction\":true,\"hidden_target_isolation\":true,"
      << "\"earlier_snapshot_immutable\":true,\"restore_earlier_checkpoint_exact\":true,\"same_budget_no_update\":true,\"all_absent_no_inferred_support\":true,"
      << "\"parity_scope\":\"same-device frozen eval/no-grad models; no CPU-versus-CUDA bitwise claim\",\"parameter_groups_changed\":{";
  bool first = true;
  for (const auto &[name, value] : changed) { if (!first) report << ','; first = false; report << quote(name) << ':' << (value ? "true" : "false"); }
  report << "},\"training_seconds\":" << four.training_seconds << ",\"preprocessing_id\":" << quote(four.preprocessing_id)
      << ",\"training_dataset_id\":" << quote(four.training_dataset_id) << ",\"gate_producer_source_fingerprint\":" << quote(producer)
      << ",\"adapter_training_producer_source_fingerprint\":" << quote(trainer.audit_fields.at("training_producer_source_fingerprint"))
      << ",\"core_writer_source_fingerprint\":" << quote(workflow_source_fingerprint())
      << ",\"settings\":" << quote(settings_text(gate_settings)) << '}';
  save_text(directory / "gpu-check.json", report.str());
  return report.str();
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
