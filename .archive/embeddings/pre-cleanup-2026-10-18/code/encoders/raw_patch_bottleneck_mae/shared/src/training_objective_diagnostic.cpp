// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/training_objective_diagnostic.h"
#include "embedding/shared/data.h"
#include <ATen/Context.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/cuda.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace fs = std::filesystem;
namespace od = objective_diagnostic;
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error("[RPB training objective diagnostic] " + message);
}
torch::Tensor cpu(const torch::Tensor &value) { return value.detach().to(torch::kCPU).clone(); }
std::string quote(const std::string &value) {
  std::ostringstream out; out << '"';
  for (unsigned char ch : value) {
    if (ch == '"' || ch == '\\') out << '\\' << char(ch);
    else if (ch < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
    else out << char(ch);
  }
  out << '"'; return out.str();
}
std::string strings(const std::vector<std::string> &values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) { if (i) out += ','; out += quote(values[i]); }
  return out + ']';
}
std::vector<std::string> parse_strings(const std::string &value) {
  // Source IDs in this frozen protocol contain no escaped characters. Reject
  // rather than silently reinterpret an unsupported source-identity spelling.
  std::vector<std::string> out; size_t p = 0;
  const auto space = [&] { while (p < value.size() && std::isspace(static_cast<unsigned char>(value[p]))) ++p; };
  space(); require(p < value.size() && value[p++] == '[', "source IDs must be a JSON string array"); space();
  if (p < value.size() && value[p] == ']') { ++p; space(); require(p == value.size(), "trailing source ID JSON"); return out; }
  while (true) {
    space(); require(p < value.size() && value[p++] == '"', "invalid source ID JSON string");
    std::string id;
    while (p < value.size() && value[p] != '"') {
      require(value[p] != '\\' && static_cast<unsigned char>(value[p]) >= 32, "escaped/control source ID unsupported");
      id += value[p++];
    }
    require(!id.empty() && p < value.size() && value[p++] == '"', "empty/unterminated source ID"); out.push_back(id); space();
    require(p < value.size(), "unterminated source ID array");
    if (value[p] == ']') { ++p; break; }
    require(value[p++] == ',', "source ID array delimiter");
  }
  space(); require(p == value.size(), "trailing source ID JSON"); return out;
}
torch::Tensor tensor(torch::serialize::InputArchive &a, const std::string &key) {
  torch::Tensor out; a.read(key, out, true); return out;
}
std::string text(torch::serialize::InputArchive &a, const std::string &key) {
  return embedding::archive::tensor_text(tensor(a, key));
}
void write_text(torch::serialize::OutputArchive &a, const std::string &key, const std::string &value) {
  a.write(key, embedding::archive::text_tensor(value), true);
}
void write_file(const fs::path &path, const std::string &value) {
  require(!fs::exists(path), "refuse to overwrite diagnostic output");
  std::ofstream out(path, std::ios::binary); require(bool(out), "cannot create diagnostic JSON");
  out << value << '\n'; require(bool(out), "cannot write diagnostic JSON");
}
struct RuntimeIsolation {
  int threads{at::get_num_threads()};
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &generator : generators) states.push_back(generator.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try { for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]); at::set_num_threads(threads); }
    catch (...) { std::terminate(); }
  }
};
struct ModelIsolation {
  Model &model; bool training;
  std::vector<torch::Tensor> parameters, parameter_values, buffers, buffer_values;
  std::vector<bool> flags;
  explicit ModelIsolation(Model &m) : model(m), training(m->is_training()) {
    parameters = model->parameters(); buffers = model->buffers();
    for (const auto &p : parameters) { flags.push_back(p.requires_grad()); parameter_values.push_back(cpu(p)); }
    for (const auto &b : buffers) buffer_values.push_back(cpu(b));
  }
  void unchanged() const {
    for (size_t i = 0; i < parameters.size(); ++i) require(torch::equal(cpu(parameters[i]), parameter_values[i]), "diagnostic mutated model parameters");
    for (size_t i = 0; i < buffers.size(); ++i) require(torch::equal(cpu(buffers[i]), buffer_values[i]), "diagnostic mutated model buffers");
  }
  ~ModelIsolation() noexcept {
    try { for (size_t i = 0; i < parameters.size(); ++i) parameters[i].set_requires_grad(flags[i]); model->train(training); }
    catch (...) { std::terminate(); }
  }
};
void validate_head(const od::FrozenRidgeHead &head, int64_t width) {
  for (const auto &v : {head.outer_mean, head.outer_scale, head.ridge_mean, head.ridge_scale})
    require(v.defined() && v.scalar_type() == torch::kFloat64 && v.numel() == width && torch::isfinite(v).all().item<bool>(), "invalid saved Ridge normalizer");
  require(head.outer_scale.gt(0).all().item<bool>() && head.ridge_scale.gt(0).all().item<bool>() &&
      head.weights.scalar_type() == torch::kFloat64 && head.weights.sizes() == torch::IntArrayRef({width, 2}) &&
      head.intercept.scalar_type() == torch::kFloat64 && head.intercept.sizes() == torch::IntArrayRef({2}) &&
      torch::isfinite(head.weights).all().item<bool>() && torch::isfinite(head.intercept).all().item<bool>(), "invalid saved Ridge head");
}
uint64_t saved_head_seed(uint64_t seed, uint64_t width) {
  // Exact shared feature_harness::stream_seed law; no probe construction/link.
  uint64_t value=seed+0x9e3779b97f4a7c15ULL*width;
  value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
  value=(value^(value>>27))*0x94d049bb133111ebULL;
  return value^(value>>31);
}
torch::Tensor gradients(const torch::Tensor &loss, const std::vector<torch::Tensor> &parameters, bool retain) {
  const auto result = torch::autograd::grad({loss}, parameters, {}, retain, false, true);
  std::vector<torch::Tensor> pieces;
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto value = result[i].defined() ? result[i] : torch::zeros_like(parameters[i]);
    require(value.is_cuda() && torch::isfinite(value).all().item<bool>(), "non-CUDA/nonfinite encoder parameter gradient");
    pieces.push_back(value.detach().to(torch::kCPU, torch::kFloat64).reshape({-1}));
  }
  return torch::cat(pieces);
}
std::string group_name(const std::string &name) {
  if (name.rfind("block_", 0) == 0) return "temporal_blocks";
  if (name.rfind("channel_mixer_", 0) == 0) return "channel_mixer";
  if (name.rfind("global_pool_", 0) == 0) return "global_pool";
  if (name.rfind("pool_", 0) == 0 || name.rfind("export_projection", 0) == 0) return "channel_summary_pool";
  return "patch_temporal_input";
}
std::string comparison_json(const od::GradientComparison &value) {
  std::ostringstream out; out << std::setprecision(17) << "{\"supported\":" << (value.supported ? "true" : "false")
      << ",\"base_descent_margin_cosine\":";
  if (value.supported) out << value.base_descent_margin_cosine; else out << "null";
  out << ",\"combined_descent_margin_cosine\":";
  if (value.supported) out << value.combined_descent_margin_cosine; else out << "null";
  out << ",\"combined_descent_base_directional_derivative\":";
  if (value.supported) out << value.combined_descent_base_directional_derivative; else out << "null";
  out<<",\"base_norm\":"<<value.base_norm<<",\"auxiliary_norm\":"<<value.auxiliary_norm
      <<",\"margin_norm\":"<<value.margin_norm<<",\"combined_norm\":"<<value.combined_norm
      <<",\"base_gradient_margin_dot\":"<<value.base_margin_dot<<",\"combined_gradient_margin_dot\":"<<value.combined_margin_dot
      <<",\"combined_gradient_base_dot\":"<<value.combined_base_dot;
  return out.str() + '}';
}
}

namespace objective_diagnostic {
PairDifferenceLoss pair_difference_huber(const torch::Tensor &prediction, const torch::Tensor &target,
                                         const torch::Tensor &query, double delta) {
  require(prediction.dim() == 4 && target.sizes() == prediction.sizes() && query.sizes() == prediction.sizes() &&
      query.scalar_type() == torch::kBool && std::isfinite(delta) && delta > 0, "invalid pair-difference inputs");
  std::vector<torch::Tensor> losses, counts, valid;
  for (int64_t i = 0; i < prediction.size(1); ++i) for (int64_t j = i + 1; j < prediction.size(1); ++j) {
    const auto support = query.select(1, i).logical_and(query.select(1, j));
    const auto pi = torch::where(support, prediction.select(1, i), torch::zeros_like(prediction.select(1, i)));
    const auto pj = torch::where(support, prediction.select(1, j), torch::zeros_like(prediction.select(1, j)));
    const auto ti = torch::where(support, target.select(1, i).detach(), torch::zeros_like(target.select(1, i)));
    const auto tj = torch::where(support, target.select(1, j).detach(), torch::zeros_like(target.select(1, j)));
    require(torch::isfinite(pi).all().item<bool>() && torch::isfinite(pj).all().item<bool>() &&
        torch::isfinite(ti).all().item<bool>() && torch::isfinite(tj).all().item<bool>(), "nonfinite common-Q pair values");
    const auto error = ((pi - pj) - (ti - tj)) / std::sqrt(2.0);
    require(torch::isfinite(error).all().item<bool>(),"pair-difference arithmetic overflow on common Q");
    const auto absolute = error.abs();
    const auto cell = torch::where(absolute.le(delta), .5 * error.clamp(-delta,delta).square(), delta * (absolute - .5 * delta));
    const auto n = support.sum(std::vector<int64_t>{1, 2});
    losses.push_back(cell.sum(std::vector<int64_t>{1, 2}) / n.clamp_min(1).to(prediction.dtype()));
    counts.push_back(n); valid.push_back(n.gt(0));
  }
  require(!losses.empty(), "pair objective requires at least two channels");
  PairDifferenceLoss out;
  out.pair_losses = torch::stack(losses, 1); out.pair_target_counts = torch::stack(counts, 1);
  out.pair_valid = torch::stack(valid, 1); out.example_valid = out.pair_valid.any(1);
  out.supported_examples = out.example_valid.sum().item<int64_t>(); out.supported_pairs = out.pair_valid.sum().item<int64_t>();
  out.target_pair_cells = out.pair_target_counts.sum().item<int64_t>();
  const auto example = torch::where(out.pair_valid, out.pair_losses, torch::zeros_like(out.pair_losses)).sum(1) /
      out.pair_valid.sum(1).clamp_min(1).to(prediction.dtype());
  out.loss = out.supported_examples ? example.masked_select(out.example_valid).mean() : out.pair_losses.sum() * 0;
  require(torch::isfinite(out.loss).item<bool>(), "nonfinite pair objective"); return out;
}
torch::Tensor latent_from_zero_perturbation(const torch::Tensor &z, const torch::Tensor &u, const FrozenRidgeHead &head) {
  validate_head(head, z.size(1));
  require(z.dim() == 2 && z.scalar_type() == torch::kFloat32 && u.sizes() == z.sizes() &&
      u.scalar_type() == torch::kFloat64 && u.device() == z.device(), "invalid final-Ridge zero perturbation");
  return z.detach() + (u * (head.outer_scale * head.ridge_scale).to(u.device())).to(z.dtype());
}
torch::Tensor signed_ridge_margin(const torch::Tensor &z, const torch::Tensor &labels, const FrozenRidgeHead &head) {
  validate_head(head, z.size(1));
  require(labels.dim() == 1 && labels.size(0) == z.size(0) && labels.scalar_type() == torch::kInt64 &&
      labels.ge(0).logical_and(labels.le(1)).all().item<bool>(), "signed margin needs fixed binary labels");
  const auto device = z.device();
  const auto u = ((z.to(torch::kFloat64) - head.outer_mean.to(device)) / head.outer_scale.to(device) -
      head.ridge_mean.to(device)) / head.ridge_scale.to(device);
  const auto logits = u.matmul(head.weights.to(device)) + head.intercept.to(device);
  return (labels.to(device, torch::kFloat64) * 2 - 1) * (logits.select(1, 1) - logits.select(1, 0));
}
GradientComparison compare_gradients(const torch::Tensor &base, const torch::Tensor &auxiliary, const torch::Tensor &margin) {
  require(base.sizes() == auxiliary.sizes() && base.sizes() == margin.sizes() && base.numel() > 0, "gradient vector dimensions differ");
  const auto a = base.to(torch::kCPU, torch::kFloat64).flatten(), b = auxiliary.to(torch::kCPU, torch::kFloat64).flatten();
  const auto m = margin.to(torch::kCPU, torch::kFloat64).flatten(), combined = a + b;
  require(torch::isfinite(a).all().item<bool>() && torch::isfinite(b).all().item<bool>() && torch::isfinite(m).all().item<bool>(), "nonfinite comparison vectors");
  const auto an = a.norm().item<double>(), cn = combined.norm().item<double>(), mn = m.norm().item<double>();
  GradientComparison out;
  out.base_norm=an;out.auxiliary_norm=b.norm().item<double>();out.margin_norm=mn;out.combined_norm=cn;
  out.base_margin_dot=torch::dot(a,m).item<double>();out.combined_margin_dot=torch::dot(combined,m).item<double>();
  out.combined_base_dot=torch::dot(combined,a).item<double>();
  require(torch::isfinite(combined).all().item<bool>() && std::isfinite(an) && std::isfinite(cn) && std::isfinite(mn) &&
      std::isfinite(out.auxiliary_norm) && std::isfinite(out.base_margin_dot) && std::isfinite(out.combined_margin_dot) &&
      std::isfinite(out.combined_base_dot),"gradient norm/dot arithmetic overflow");
  if (an <= 1e-20 || cn <= 1e-20 || mn <= 1e-20) return out;
  out.supported = true; out.base_descent_margin_cosine = -torch::dot(a, m).item<double>() / (an * mn);
  out.combined_descent_margin_cosine = -torch::dot(combined, m).item<double>() / (cn * mn);
  out.combined_descent_base_directional_derivative = -torch::dot(combined, a).item<double>() / cn;
  return out;
}
BankEvidence inspect_bank(Model &model, const Input &normalized, const MaskPlan &original,
                          const torch::Tensor &visible, const torch::Tensor &labels, const FrozenRidgeHead &head) {
  RuntimeIsolation runtime; ModelIsolation state(model); const auto &config = model->config();
  require(torch::cuda::is_available() && config.device.is_cuda(), "actual CUDA model required");
  validate_input(normalized, config, true); validate_head(head, config.export_width);
  require(normalized.data.is_cuda() && visible.device() == normalized.data.device() && visible.scalar_type() == torch::kBool &&
      visible.sizes() == normalized.data.sizes() && !visible.logical_and(original.visible.logical_not()).any().item<bool>(), "only legal original-visible CUDA context may enter encoder");
  const auto checked = mask_from_hidden(normalized.observed, original.hidden, config);
  require(torch::equal(checked.visible,original.visible) && torch::equal(checked.target, original.target) && torch::equal(checked.eligible_channels, original.eligible_channels) &&
      original.eligible_channels.any().item<bool>(), "original Q/eligibility changed or entirely unsupported");
  const auto retained_groups=visible.reshape({normalized.data.size(0),config.channel_count,config.history_length/config.patch_length,-1}).any(-1).sum(-1);
  require(!original.eligible_channels.logical_and(retained_groups.lt(2)).any().item<bool>(),"original eligible channels require two retained visible patch groups");
  model->eval(); for (auto &p : state.parameters) p.set_requires_grad(false);
  Input input{torch::where(visible, normalized.data, torch::zeros_like(normalized.data)), visible,
      normalized.channel_ids, normalized.endpoints, normalized.sampling_interval};
  torch::Tensor frozen, prediction;
  {
    torch::NoGradGuard no_grad;
    const auto encoded = model->encode(input); frozen = compact_reconstruction_export(encoded, config).detach();
    prediction = model->decode(frozen, input.channel_ids).detach();
  }
  require(frozen.dim() == 2 && frozen.size(1) == 32 && frozen.is_cuda() && prediction.is_cuda(), "exact CUDA served global32 route required");
  BankEvidence out; out.frozen_z = cpu(frozen); out.prediction = cpu(prediction); out.target = cpu(normalized.data);
  out.query = cpu(original.target); out.visible = cpu(visible);
  auto perturbation = torch::zeros(frozen.sizes(), torch::TensorOptions().device(config.device).dtype(torch::kFloat64).requires_grad(true));
  const auto perturbed = latent_from_zero_perturbation(frozen, perturbation, head);
  require(torch::equal(perturbed.detach(), frozen), "zero perturbation changed served latent");
  const auto leaf_prediction = model->decode(perturbed, input.channel_ids);
  require(torch::equal(leaf_prediction.detach(), prediction), "zero perturbation changed frozen decoder prediction");
  const auto leaf_base = hierarchical_huber(leaf_prediction, normalized.data.detach(), original.target,
      original.eligible_channels, config.huber_delta);
  const auto leaf_aux = pair_difference_huber(leaf_prediction, normalized.data.detach(), original.target, config.huber_delta);
  const auto lb = torch::autograd::grad({leaf_base.loss}, {perturbation}, {}, true, false, false)[0];
  const auto la = torch::autograd::grad({leaf_aux.loss}, {perturbation}, {}, false, false, false)[0];
  require(lb.is_cuda() && la.is_cuda() && torch::isfinite(lb).all().item<bool>() && torch::isfinite(la).all().item<bool>(),"non-CUDA/nonfinite latent gradients");
  out.latent_base_gradient = cpu(lb); out.latent_auxiliary_gradient = cpu(la);
  out.latent_margin_gradient = cpu((labels.to(config.device, torch::kFloat64) * 2 - 1).unsqueeze(1) *
      (head.weights.select(1, 1) - head.weights.select(1, 0)).to(config.device).unsqueeze(0));
  std::map<std::string, torch::Tensor> sorted;
  for (const auto &p : model->named_parameters()) {
    require(p.value().is_cuda(), "non-CUDA model parameter"); out.cuda_parameter_count += p.value().numel();
    if (p.key().rfind("decoder_", 0) != 0) sorted.emplace(p.key(), p.value());
  }
  require(out.cuda_parameter_count == 225805 && !sorted.empty(), "fixed225805-parameter encoder required");
  std::vector<torch::Tensor> parameters; std::vector<int64_t> offsets{0};
  for (auto &[name, p] : sorted) {
    p.set_requires_grad(true); parameters.push_back(p); out.parameter_names.push_back(name); out.parameter_groups.push_back(group_name(name));
    offsets.push_back(offsets.back() + p.numel());
  }
  out.parameter_offsets = torch::tensor(offsets, torch::kInt64);
  const auto encoded = model->encode(input); const auto graph_z = compact_reconstruction_export(encoded, config);
  out.graph_z = cpu(graph_z); out.graph_export_max_error = (graph_z.detach() - frozen).abs().max().item<double>();
  require(out.graph_export_max_error <= 1e-6, "grad-enabled/frozen export branch difference exceeds1e-6");
  const auto graph_prediction = model->decode(graph_z, input.channel_ids);
  out.graph_prediction=cpu(graph_prediction);
  const auto base = hierarchical_huber(graph_prediction, normalized.data.detach(), original.target,
      original.eligible_channels, config.huber_delta);
  const auto auxiliary = pair_difference_huber(graph_prediction, normalized.data.detach(), original.target, config.huber_delta);
  const auto margins = signed_ridge_margin(graph_z, labels, head);
  const auto eligible_examples = original.eligible_channels.any(1);
  const auto margin = margins.masked_select(eligible_examples).mean();
  require(base.loss.is_cuda() && auxiliary.loss.is_cuda() && margin.is_cuda(), "all diagnostic gradients must originate on CUDA");
  out.encoder_base_gradient = gradients(base.loss, parameters, true);
  out.encoder_auxiliary_gradient = gradients(auxiliary.loss, parameters, true);
  out.encoder_margin_gradient = gradients(margin, parameters, false);
  out.base_loss = base.loss.item<double>(); out.auxiliary_loss = auxiliary.loss.item<double>();
  out.auxiliary = auxiliary; out.auxiliary.loss = cpu(auxiliary.loss); out.auxiliary.pair_losses = cpu(auxiliary.pair_losses);
  out.auxiliary.pair_target_counts = cpu(auxiliary.pair_target_counts); out.auxiliary.pair_valid = cpu(auxiliary.pair_valid);
  out.auxiliary.example_valid = cpu(auxiliary.example_valid);
  out.comparison = compare_gradients(out.encoder_base_gradient, out.encoder_auxiliary_gradient, out.encoder_margin_gradient);
  state.unchanged(); return out;
}
} // namespace objective_diagnostic

namespace {
struct Cohort {
  Checkpoint checkpoint;
  Dataset training;
  torch::Tensor labels;
  std::vector<std::string> sources;
  std::array<od::FrozenRidgeHead, 3> heads;
  std::array<std::map<std::string, torch::Tensor>, 3> fit_values;
  torch::Tensor archived_prediction, archived_target, archived_query, archived_visible, archived_eligible;
};
std::string source_manifest(const std::vector<std::string> &ids) {
  std::string out; for (const auto &id : ids) out += std::to_string(id.size()) + ':' + id; return out;
}
bool same_scaler(const FrozenScaler &a, const FrozenScaler &b) {
  return a.identity() == b.identity() && a.scale_floor == b.scale_floor && torch::equal(a.mean, b.mean) &&
      torch::equal(a.scale, b.scale) && torch::equal(a.count, b.count) && torch::equal(a.channel_ids, b.channel_ids) &&
      torch::equal(a.floor_applied, b.floor_applied);
}
ContextDeletionOptions policy(const TrainingObjectiveDiagnosticInstance &instance) {
  if (instance.model_tag == "RPB-v4") { require(instance.expected_policy_id.empty(), "v4 requires ordinary empty policy"); return {}; }
  const auto recipe = instance.model_tag == "RPB-v7" ? ContextDeletionRecipe::coordinate15_v1 : ContextDeletionRecipe::balanced30_v1;
  const auto &selected = context_deletion::descriptor(recipe);
  require(instance.model_tag == selected.model_tag && instance.expected_policy_id == selected.policy_id, "tag/policy binding differs");
  return {true, recipe};
}
void validate_policy(torch::serialize::InputArchive &audit, const TrainingObjectiveDiagnosticInstance &instance) {
  const auto options = policy(instance); torch::Tensor unused;
  if (!options.enabled) {
    require(!audit.try_read("training_policy_id", unused, true) && !audit.try_read("context_deletion_ratio", unused, true), "ordinary audit contains context policy"); return;
  }
  const auto &d = context_deletion::descriptor(options.recipe);
  require(text(audit, "model_tag") == d.model_tag && text(audit, "training_policy_id") == d.policy_id &&
      text(audit, "context_deletion_ratio") == d.ratio_text && tensor(audit, "context_deletion_ratio_value").item<double>() == d.ratio &&
      text(audit, "context_deletion_stream") == "0x6374782d64726f70" &&
      tensor(audit, "context_deletion_stream_value").item<int64_t>() == static_cast<int64_t>(context_deletion::stream) &&
      text(audit, "context_deletion_rng_policy") == context_deletion::rng_policy &&
      text(audit, "context_deletion_repair_policy") == context_deletion::repair_policy &&
      text(audit, "context_deletion_visibility_policy") == context_deletion::visibility_policy, "context companion constants differ");
  if (context_deletion::is_balanced(options.recipe))
    require(text(audit, "context_deletion_schedule_policy") == context_deletion::balanced_schedule_policy &&
        text(audit, "context_deletion_rate_scope") == context_deletion::balanced_rate_scope &&
        text(audit, "context_deletion_branch_count_policy") == context_deletion::balanced_branch_count_policy &&
        text(audit, "context_deletion_skip_policy") == context_deletion::balanced_skip_policy &&
        text(audit, "context_ordinary_attempts") == "256" && text(audit, "context_deletion_attempts") == "256" &&
        tensor(audit, "context_ordinary_attempts_value").item<int64_t>() == 256 && tensor(audit, "context_deletion_attempts_value").item<int64_t>() == 256,
        "balanced512 schedule/count companion differs");
}
Cohort admit(const TrainingObjectiveDiagnosticInstance &instance) {
  Cohort out; auto checkpoint_cpu = load_checkpoint(instance.checkpoint_path, torch::kCPU);
  const auto &c = checkpoint_cpu.settings.model;
  require(c.channel_mixer_placement == 0,
      "historical training objective diagnostic rejects early channel mixer placement");
  require(c.global_bottleneck_mode == 2 && c.channel_mixer_layers == 1 && c.channel_count == 3 && c.history_length == 32 &&
      c.input_width == 3 && c.patch_length == 8 && c.encoder_width == 64 && c.export_width == 32 && c.dropout == 0 &&
      c.huber_delta == 1 && c.num_layers==3 && c.num_heads==4 && c.feedforward_width==256 && c.decoder_hidden_width==128 &&
      checkpoint_cpu.settings.batch_size == 8 && checkpoint_cpu.settings.seed == static_cast<int64_t>(instance.master_seed) &&
      checkpoint_cpu.attempted_steps == 512 && checkpoint_cpu.completed_steps == 512 &&
      checkpoint_cpu.training_policy_id == instance.expected_policy_id && checkpoint_cpu.source_fingerprint == instance.core_writer_source_id,
      "checkpoint recipe/budget/policy/core writer differs from frozen role");
  out.training = load_dataset(instance.training_raw_path, c);
  require(out.training.dataset_id == checkpoint_cpu.dataset_id && out.training.schema_id == checkpoint_cpu.schema_id &&
      checkpoint_cpu.scaler_fit_dataset_id == out.training.dataset_id, "raw TRAIN/checkpoint/scaler association differs");
  const auto scaler = load_scaler(instance.scaler_path, c, out.training.schema_id);
  require(same_scaler(scaler, checkpoint_cpu.scaler), "saved raw scaler differs from checkpoint");
  torch::serialize::InputArchive controlled; controlled.load_from(instance.controlled_training_path, torch::kCPU);
  auto observed = tensor(controlled, "observed"), mask = tensor(controlled, "feature_mask");
  out.labels = tensor(controlled, "labels_scoring_only"); out.sources = parse_strings(text(controlled, "source_ids_json"));
  require(observed.scalar_type() == torch::kFloat64 && observed.sizes() == torch::IntArrayRef({256,3,32,3}) &&
      mask.scalar_type() == torch::kBool && mask.sizes() == observed.sizes() && torch::equal(observed, out.training.input.data) &&
      torch::equal(mask, out.training.input.observed) && out.labels.scalar_type() == torch::kInt64 && out.labels.sizes() == torch::IntArrayRef({256}) &&
      out.labels.ge(0).logical_and(out.labels.le(1)).all().item<bool>() && out.sources.size() == 256, "controlled legal TRAIN rows/support/labels differ");
  require(observed.masked_select(mask.logical_not()).eq(0).all().item<bool>(),"unobserved raw TRAIN storage must be zero");
  std::map<std::string, std::vector<int64_t>> groups;
  for (int64_t row = 0; row < 256; ++row) groups[out.sources[row]].push_back(row);
  require(groups.size() == 128, "TRAIN requires128 source pairs");
  for (const auto &[id, rows] : groups) require(rows.size() == 2 && out.labels[rows[0]].item<int64_t>() != out.labels[rows[1]].item<int64_t>() &&
      torch::equal(mask[rows[0]], mask[rows[1]]), "source pairs require opposite labels and identical natural support");
  torch::serialize::InputArchive audit; audit.load_from(instance.checkpoint_audit_path, torch::kCPU);
  auto recorded_settings=parse_settings(text(audit,"resolved_settings"));recorded_settings.model.device=torch::kCPU;
  require(settings_text(recorded_settings)==settings_text(checkpoint_cpu.settings) &&
      text(audit,"feature_units")==out.training.feature_units &&
      text(audit,"rng_policy")=="splitmix64-counter-rows-masks-torch-attempt-v1" &&
      text(audit,"sampling_policy")=="with_replacement_counter_rows;sampled_rows_includes_no_update_attempts" &&
      text(audit,"optimizer_policy")=="one_continuous_AdamW_state;absolute_completed_update_budgets",
      "recorded settings/units/counter policies differ");
  require(text(audit,"artifact_kind") == "rpb_learning_curve_training_audit_v1" &&
      text(audit,"protocol_id") == instance.training_namespace && instance.training_namespace == "native-development-v1/lag_sign" &&
      text(audit,"training_producer_source_fingerprint") == instance.training_producer_source_id &&
      text(audit,"core_writer_source_fingerprint") == instance.core_writer_source_id &&
      text(audit,"fit_source_manifest") == source_manifest(out.sources) &&
      text(audit,"training_dataset_id") == out.training.dataset_id && text(audit,"preprocessing_id") == scaler.identity() &&
      text(audit,"actual_training_seed") == std::to_string(instance.master_seed) &&
      text(audit,"initialization_seed") == std::to_string(training_detail::mixed(instance.master_seed ^ 0x7270622d696e6974ULL)) &&
      tensor(audit,"attempted_steps").item<int64_t>() == 512 && tensor(audit,"completed_steps").item<int64_t>() == 512 &&
      tensor(audit,"sampled_rows").item<int64_t>() == 4096 && torch::Device(text(audit,"training_device")).is_cuda(), "saved producer/source/counter association differs");
  const auto physical_ids=tensor(audit,"channel_order");
  const auto raw_ids=out.training.input.channel_ids.dim()==1?out.training.input.channel_ids:
      out.training.input.channel_ids.select(0,0);
  require(physical_ids.scalar_type()==torch::kInt64 && physical_ids.sizes()==torch::IntArrayRef({3}) && torch::equal(physical_ids,raw_ids) &&
      (out.training.input.channel_ids.dim()==1 || torch::equal(out.training.input.channel_ids,physical_ids.unsqueeze(0).expand({256,3}))) &&
      tensor(audit,"sampling_interval").item<double>()==out.training.input.sampling_interval &&
      out.training.input.endpoints.eq(tensor(audit,"endpoint").item<double>()).all().item<bool>(),"physical channel/endpoints/interval metadata differ");
  validate_policy(audit, instance);
  checkpoint_cpu.model->eval(); for (auto &p : checkpoint_cpu.model->parameters()) p.set_requires_grad(false);
  torch::Tensor full_z;
  {
    torch::NoGradGuard no_grad;
    const auto normalized = scaler.transform(out.training.input, c);
    const auto encoded = checkpoint_cpu.model->encode(normalized); full_z = cpu(compact_reconstruction_export(encoded,c));
    torch::serialize::InputArchive native; native.load_from(instance.native_training_path,torch::kCPU);
    require(torch::equal(tensor(native,"features"),full_z) && torch::equal(tensor(native,"valid"),cpu(encoded.sample_valid_mask)) &&
        torch::equal(tensor(native,"labels_scoring_only"),out.labels) && parse_strings(text(native,"source_ids_json")) == out.sources,
        "exact CPU checkpoint export/source/support parity failed");
  }
  for (size_t rep=0; rep<3; ++rep) {
    torch::serialize::InputArchive fit; fit.load_from(instance.ridge_fit_paths[rep],torch::kCPU);
    for (const std::string key : {"feature_mean","feature_scale","ridge_mean","ridge_scale","ridge_weights","ridge_intercept",
        "tiny_mean","tiny_scale","tiny_w1","tiny_b1","tiny_w2","tiny_b2"}) out.fit_values[rep].emplace(key,tensor(fit,key));
    for (const std::string key : {"pca_mean","pca_components","pca_singular_values"}) { torch::Tensor value; require(!fit.try_read(key,value,true),"native head must not contain PCA"); }
    require(tensor(fit,"fitted_rows").item<int64_t>() == 256 && text(fit,"actual_probe_seed_decimal") == std::to_string(saved_head_seed(2701 + rep*101,32)),"fixed head rows/width-derived seed differ");
    const auto &values=out.fit_values[rep]; out.heads[rep]={values.at("feature_mean"),values.at("feature_scale"),values.at("ridge_mean"),values.at("ridge_scale"),values.at("ridge_weights"),values.at("ridge_intercept")};
    validate_head(out.heads[rep],32);
    torch::serialize::InputArchive saved; saved.load_from(instance.saved_training_prediction_paths[rep],torch::kCPU);
    const auto outer=(full_z.to(torch::kFloat64)-values.at("feature_mean"))/values.at("feature_scale");
    const auto ridge=((outer-values.at("ridge_mean"))/values.at("ridge_scale")).matmul(values.at("ridge_weights"))+values.at("ridge_intercept");
    const auto hidden=torch::tanh(((outer-values.at("tiny_mean"))/values.at("tiny_scale")).matmul(values.at("tiny_w1"))+values.at("tiny_b1"));
    const auto tiny=hidden.matmul(values.at("tiny_w2"))+values.at("tiny_b2");
    require(torch::equal(tensor(saved,"probe_features"),outer) && torch::equal(tensor(saved,"ridge"),ridge.argmax(1)) &&
        torch::equal(tensor(saved,"tiny_secondary"),tiny.argmax(1)) && torch::equal(tensor(saved,"labels_scoring_only"),out.labels) &&
        tensor(saved,"valid").all().item<bool>() && parse_strings(text(saved,"source_ids_json")) == out.sources,"saved exact Ridge/tanh TRAIN equations/source parity failed");
    if (rep) for (const std::string key : {"feature_mean","feature_scale","ridge_mean","ridge_scale","ridge_weights","ridge_intercept"})
      require(torch::equal(values.at(key),out.fit_values[0].at(key)),"three deterministic Ridge fits must be exact duplicates");
  }
  torch::serialize::InputArchive reconstruction; reconstruction.load_from(instance.training_reconstruction_path,torch::kCPU);
  out.archived_prediction=tensor(reconstruction,"standardized_prediction"); out.archived_target=tensor(reconstruction,"standardized_target");
  out.archived_query=tensor(reconstruction,"target_mask"); out.archived_visible=tensor(reconstruction,"visible_mask");
  out.archived_eligible=tensor(reconstruction,"trial_channel_eligible");
  require(out.archived_prediction.sizes()==torch::IntArrayRef({4,256,3,32,3}) && out.archived_target.sizes()==out.archived_prediction.sizes() &&
      out.archived_query.sizes()==out.archived_prediction.sizes() && out.archived_visible.sizes()==out.archived_prediction.sizes() &&
      out.archived_eligible.sizes()==torch::IntArrayRef({4,256,3}) && parse_strings(text(reconstruction,"source_ids_json"))==out.sources,"saved TRAIN query witness dimensions/source differ");
  out.checkpoint=load_checkpoint(instance.checkpoint_path,torch::kCUDA);
  for (auto &p:out.checkpoint.model->parameters()) p.set_requires_grad(false); out.checkpoint.model->eval();
  return out;
}

std::string geometry_surface(torch::serialize::OutputArchive &archive,const std::string &prefix,
    const torch::Tensor &values,const torch::Tensor &valid,const torch::Tensor &labels,const std::vector<std::string> &sources,
    const od::FrozenRidgeHead *ridge=nullptr) {
  require(values.dim()==2 && valid.dim()==1 && valid.size(0)==values.size(0) && valid.any().item<bool>(),"geometry surface support invalid");
  const auto x=values.to(torch::kFloat64), selected=x.index({valid});
  const auto mean=selected.mean(0), scale=(selected-mean).square().mean(0).sqrt().clamp_min(1e-8);
  const auto u=(x-mean)/scale;
  std::map<std::string,std::vector<int64_t>> groups; for (int64_t i=0;i<x.size(0);++i) groups[sources[i]].push_back(i);
  std::vector<int64_t> zero,one; std::vector<std::string> retained;
  for (const auto &[source,rows]:groups) if (rows.size()==2 && valid[rows[0]].item<bool>() && valid[rows[1]].item<bool>()) {
    const auto a=labels[rows[0]].item<int64_t>()==0?rows[0]:rows[1]; const auto b=a==rows[0]?rows[1]:rows[0];
    zero.push_back(a);one.push_back(b);retained.push_back(source);
  }
  require(!zero.empty(),"zero supported paired geometry sources");
  const auto i0=torch::tensor(zero,torch::kInt64),i1=torch::tensor(one,torch::kInt64);
  const auto u0=u.index_select(0,i0),u1=u.index_select(0,i1), midpoint=(u0+u1)*.5,contrast=(u1-u0)*.5;
  const auto paired=torch::cat({u0,u1},0); const double total=(paired-paired.mean(0)).square().sum(1).mean().item<double>();
  const double mid=(midpoint-midpoint.mean(0)).square().sum(1).mean().item<double>();
  const double difference=contrast.square().sum(1).mean().item<double>(),coherent=contrast.mean(0).square().sum().item<double>();
  require(std::abs(total-mid-difference)<=1e-10*std::max(1.0,total),"paired variance identity failed");
  archive.write(prefix+"_values",values,true);archive.write(prefix+"_valid",valid,true);archive.write(prefix+"_mean",mean,true);archive.write(prefix+"_scale",scale,true);
  archive.write(prefix+"_standardized",u,true);archive.write(prefix+"_label0_rows",i0,true);archive.write(prefix+"_label1_rows",i1,true);
  archive.write(prefix+"_midpoint",midpoint,true);archive.write(prefix+"_half_difference",contrast,true);
  write_text(archive,prefix+"_paired_sources_json",strings(retained));
  double paired_margin=0;
  if(ridge){const auto margin=od::signed_ridge_margin(values,labels,*ridge);
    const auto along=(margin.index_select(0,i0)+margin.index_select(0,i1))*.5;
    archive.write(prefix+"_saved_ridge_signed_margin",margin,true);archive.write(prefix+"_saved_ridge_pair_contrast_margin",along,true);
    paired_margin=along.mean().item<double>();
  }
  std::ostringstream out;out<<std::setprecision(17)<<"{\"paired_sources\":"<<zero.size()<<",\"dimensions\":"<<values.size(1)
      <<",\"total_trace_variance\":"<<total<<",\"midpoint_trace_variance\":"<<mid<<",\"contrast_energy\":"<<difference
      <<",\"coherent_contrast_energy\":"<<coherent<<",\"incoherent_difference_variance\":"<<difference-coherent<<",\"coherence\":";
  if(difference>1e-20)out<<coherent/difference;else out<<"null";
  out<<",\"saved_ridge_pair_contrast_margin_mean\":";if(ridge)out<<paired_margin;else out<<"null";
  return out.str()+'}';
}

std::vector<int64_t> donor_rows(const std::vector<std::string> &sources,const torch::Tensor &labels,bool sibling) {
  std::map<std::string,std::array<int64_t,2>> groups;
  for(int64_t row=0;row<labels.size(0);++row) {
    if(!groups.count(sources[row]))groups[sources[row]]={-1,-1}; groups[sources[row]][labels[row].item<int64_t>()]=row;
  }
  std::vector<int64_t> result(labels.size(0),-1);
  if(sibling)for(const auto &[id,rows]:groups){require(rows[0]>=0&&rows[1]>=0,"sibling requires complete pair");result[rows[0]]=rows[1];result[rows[1]]=rows[0];}
  else {
    require(groups.size()>1,"cross-source derangement requires two source groups");
    std::vector<std::array<int64_t,2>> ordered;for(const auto &[id,rows]:groups)ordered.push_back(rows);
    for(size_t s=0;s<ordered.size();++s)for(size_t y=0;y<2;++y)result[ordered[s][y]]=ordered[(s+1)%ordered.size()][y];
  }
  return result;
}
torch::Tensor channel_losses(const torch::Tensor &prediction,const torch::Tensor &target,const torch::Tensor &query,double delta) {
  const auto safe_prediction=torch::where(query,prediction,torch::zeros_like(prediction));
  const auto safe_target=torch::where(query,target,torch::zeros_like(target));
  const auto error=safe_prediction-safe_target; const auto abs=error.abs();
  const auto loss=torch::where(abs.le(delta),.5*error.clamp(-delta,delta).square(),delta*(abs-.5*delta));const auto counts=query.sum(std::vector<int64_t>{2,3});
  return loss.sum(std::vector<int64_t>{2,3})/counts.clamp_min(1).to(loss.dtype());
}
std::string bank_output(const fs::path &directory,const std::string &name,Model &model,const Input &normalized,
    const MaskPlan &mask,const torch::Tensor &visible,const torch::Tensor &labels,const std::vector<std::string> &sources,
    const torch::Tensor &indices,const od::FrozenRidgeHead &head,const context_deletion::Plan *deletion,
    od::BankEvidence &evidence) {
  evidence=od::inspect_bank(model,normalized,mask,visible,labels,head);
  torch::serialize::OutputArchive archive;
  write_text(archive,"artifact_kind","rpb_training_objective_bank_v1");write_text(archive,"source_ids_json",strings(sources));
  write_text(archive,"parameter_names_json",strings(evidence.parameter_names));write_text(archive,"parameter_groups_json",strings(evidence.parameter_groups));
  for(const auto &[key,value]:std::map<std::string,torch::Tensor>{{"frozen_z",evidence.frozen_z},{"graph_z",evidence.graph_z},
      {"factual_prediction",evidence.prediction},{"graph_prediction",evidence.graph_prediction},{"standardized_target",evidence.target},{"query",evidence.query},{"visible",evidence.visible},
      {"original_hidden",cpu(mask.hidden)},{"original_visible",cpu(mask.visible)},{"original_eligible",cpu(mask.eligible_channels)},
      {"labels_scoring_only",cpu(labels)},{"training_row_indices",cpu(indices)}, {"parameter_offsets",evidence.parameter_offsets},
      {"latent_base_gradient",evidence.latent_base_gradient},{"latent_auxiliary_gradient",evidence.latent_auxiliary_gradient},
      {"latent_margin_gradient",evidence.latent_margin_gradient},{"encoder_base_gradient",evidence.encoder_base_gradient},
      {"encoder_auxiliary_gradient",evidence.encoder_auxiliary_gradient},{"encoder_margin_gradient",evidence.encoder_margin_gradient},
      {"auxiliary_pair_losses",evidence.auxiliary.pair_losses},{"auxiliary_pair_target_counts",evidence.auxiliary.pair_target_counts},
      {"auxiliary_pair_valid",evidence.auxiliary.pair_valid},{"auxiliary_example_valid",evidence.auxiliary.example_valid}})archive.write(key,value,true);
  archive.write("channel_ids",cpu(normalized.channel_ids),true);
  archive.write("outer_scale",head.outer_scale,true);archive.write("ridge_scale",head.ridge_scale,true);archive.write("ridge_weights",head.weights,true);
  archive.write("outer_mean",head.outer_mean,true);archive.write("ridge_mean",head.ridge_mean,true);archive.write("ridge_intercept",head.intercept,true);
  const auto empty=torch::zeros_like(evidence.visible);
  archive.write("context_requested_deleted",deletion?cpu(deletion->requested_deleted):empty,true);
  archive.write("context_actual_deleted",deletion?cpu(deletion->deleted):empty,true);
  // Source order is lexical and fixed. Sibling swaps intentionally flip the
  // source variant; the separate cyclic donor preserves class and source groups.
  std::map<std::string,torch::Tensor> intervention_z;
  if(name.rfind("query_",0)==0){
    const auto siblings=torch::tensor(donor_rows(sources,labels,true),torch::kInt64);
    const auto donors=torch::tensor(donor_rows(sources,labels,false),torch::kInt64);
    archive.write("sibling_donor_rows",siblings,true);archive.write("cross_source_donor_rows",donors,true);
    intervention_z={{"factual",evidence.frozen_z},
        {"mean",evidence.frozen_z.mean(0).unsqueeze(0).expand_as(evidence.frozen_z)},
        {"sibling_swap",evidence.frozen_z.index_select(0,siblings)}, {"cross_source_preserve_label",evidence.frozen_z.index_select(0,donors)}};
  }
  std::ostringstream interventions;interventions<<std::setprecision(17)<<'{';bool first=true;
  {
    torch::NoGradGuard no_grad;for(auto &p:model->parameters())p.set_requires_grad(false);model->eval();
    for(const auto &[id,z]:intervention_z){
      const auto prediction=cpu(model->decode(z.to(model->config().device),normalized.channel_ids));
      const auto loss=hierarchical_huber(prediction,evidence.target,evidence.query,cpu(mask.eligible_channels),model->config().huber_delta);
      const auto channel=channel_losses(prediction,evidence.target,evidence.query,model->config().huber_delta);
      archive.write("intervention_"+id+"_prediction",prediction,true);archive.write("intervention_"+id+"_channel_huber",channel,true);
      if(!first)interventions<<',';first=false;interventions<<quote(id)<<":{\"hierarchical_huber\":"<<loss.loss.item<double>()<<'}';
    }
  }
  interventions<<'}';
  const auto base_channel=channel_losses(evidence.prediction,evidence.target,evidence.query,model->config().huber_delta);
  const auto eligible=cpu(mask.eligible_channels),valid_examples=eligible.any(1);
  const auto contribution=torch::where(eligible,base_channel,torch::zeros_like(base_channel))/eligible.sum(1).clamp_min(1).to(base_channel.dtype()).unsqueeze(1)/valid_examples.sum().item<double>();
  archive.write("base_channel_huber",base_channel,true);archive.write("base_channel_contribution",contribution,true);
  auto b=evidence.latent_base_gradient.to(torch::kFloat64),a=evidence.latent_auxiliary_gradient.to(torch::kFloat64),m=evidence.latent_margin_gradient.to(torch::kFloat64);
  const auto bn=b.square().sum(1).sqrt(),an=(a+b).square().sum(1).sqrt(),mn=m.square().sum(1).sqrt();
  const auto latent_valid=bn.gt(1e-20).logical_and(an.gt(1e-20)).logical_and(mn.gt(1e-20));
  const auto latent_base_cos=-(b*m).sum(1)/(bn*mn).clamp_min(1e-40),latent_combined_cos=-((a+b)*m).sum(1)/(an*mn).clamp_min(1e-40);
  archive.write("latent_comparison_valid",latent_valid,true);archive.write("latent_base_descent_margin_cosine",latent_base_cos,true);archive.write("latent_combined_descent_margin_cosine",latent_combined_cos,true);
  archive.write("latent_base_norm",bn,true);archive.write("latent_combined_norm",an,true);archive.write("latent_margin_norm",mn,true);
  archive.write("latent_base_gradient_margin_dot",(b*m).sum(1),true);archive.write("latent_combined_gradient_margin_dot",((a+b)*m).sum(1),true);
  embedding::archive::save_archive((directory/(name+".pt")).string(),archive);
  std::map<std::string,std::vector<int64_t>> group_indices;
  const auto offsets=evidence.parameter_offsets.accessor<int64_t,1>();
  for(size_t i=0;i<evidence.parameter_names.size();++i)for(int64_t j=offsets[i];j<offsets[i+1];++j)group_indices[evidence.parameter_groups[i]].push_back(j);
  std::ostringstream groups;groups<<'{';first=true;
  for(const auto &[id,positions]:group_indices){const auto select=torch::tensor(positions,torch::kInt64);if(!first)groups<<',';first=false;
    groups<<quote(id)<<':'<<comparison_json(od::compare_gradients(evidence.encoder_base_gradient.index_select(0,select),evidence.encoder_auxiliary_gradient.index_select(0,select),evidence.encoder_margin_gradient.index_select(0,select)));}
  groups<<'}';
  std::ostringstream json;json<<std::setprecision(17)<<"{\"id\":"<<quote(name)<<",\"rows\":"<<labels.size(0)
      <<",\"target_cells\":"<<evidence.query.sum().item<int64_t>()<<",\"auxiliary_supported_rows\":"<<evidence.auxiliary.supported_examples
      <<",\"auxiliary_supported_pairs\":"<<evidence.auxiliary.supported_pairs<<",\"auxiliary_target_pair_cells\":"<<evidence.auxiliary.target_pair_cells
      <<",\"base_huber\":"<<evidence.base_loss<<",\"auxiliary_huber\":"<<evidence.auxiliary_loss
      <<",\"graph_export_max_error\":"<<evidence.graph_export_max_error<<",\"cuda_parameter_count\":"<<evidence.cuda_parameter_count
      <<",\"whole_encoder\":"<<comparison_json(evidence.comparison)<<",\"groups\":"<<groups.str()<<",\"decoder_interventions\":"<<interventions.str()
      <<",\"artifact\":"<<quote(name+".pt")<<'}';
  write_file(directory/(name+".json"),json.str());return json.str();
}
}

std::string run_training_objective_diagnostic(const TrainingObjectiveDiagnosticRun &run) {
  RuntimeIsolation runtime;
  at::set_num_threads(1);
  require(torch::cuda::is_available() && !run.source_fingerprint.empty() && !run.output_directory.empty() && run.instances.size()==15,
      "CUDA/frozen producer/fifteen explicit instances required");
  const std::set<uint64_t> masters{4404,5505,6606,7707,8808};std::set<std::pair<uint64_t,std::string>> seen;std::set<std::string> ids;
  for(const auto &instance:run.instances){
    require(masters.count(instance.master_seed)&&instance.budget==512&&seen.emplace(instance.master_seed,instance.model_tag).second&&
        !instance.input_id.empty()&&instance.input_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")==std::string::npos&&
        ids.insert(instance.input_id).second&&!instance.orchestration_source_id.empty()&&!instance.training_producer_source_id.empty()&&!instance.core_writer_source_id.empty(),"frozen instance/master/budget identity differs");
    (void)policy(instance);
    std::vector<std::string> paths{instance.controlled_training_path,instance.checkpoint_path,instance.checkpoint_audit_path,instance.scaler_path,
        instance.training_raw_path,instance.native_training_path,instance.training_reconstruction_path};
    paths.insert(paths.end(),instance.ridge_fit_paths.begin(),instance.ridge_fit_paths.end());paths.insert(paths.end(),instance.saved_training_prediction_paths.begin(),instance.saved_training_prediction_paths.end());
    for(const auto &path:paths)require(!path.empty()&&fs::is_regular_file(path),"explicit input role is not a regular existing file");
  }
  for(const auto master:masters)for(const std::string tag:{"RPB-v4","RPB-v7","RPB-v8"})require(seen.count({master,tag}),"missing frozen tag/master instance");
  const fs::path output=fs::absolute(run.output_directory);require(!fs::exists(output),"diagnostic output must be new");require(fs::create_directories(output),"exclusive new diagnostic directory creation failed");
  std::ostringstream reports;reports<<'[';bool first_instance=true;std::map<uint64_t,std::array<od::BankEvidence,4>> v7_actual;
  for(const auto &instance:run.instances){
    auto cohort=admit(instance);const auto &config=cohort.checkpoint.settings.model;const auto directory=output/instance.input_id;fs::create_directory(directory);
    auto &model=cohort.checkpoint.model;ModelIsolation model_state(model);
    const auto normalized=cohort.checkpoint.scaler.transform(cohort.training.input,config);
    torch::serialize::OutputArchive geometry;std::string contextual_json,global_json;
    {
      torch::NoGradGuard no_grad;const auto encoded=model->encode(normalized);
      const auto order=channel_indices(normalized.channel_ids,config,256,config.device).argsort(int64_t{1});
      const auto contextual=cpu(encoded.z_contextual.gather(1,order.unsqueeze(-1).expand({256,3,32}))).flatten(1);
      const auto channel_valid=cpu(encoded.channel_valid_mask.gather(1,order));
      contextual_json=geometry_surface(geometry,"contextual",contextual,channel_valid.all(1),cohort.labels,cohort.sources);
      global_json=geometry_surface(geometry,"global",cpu(compact_reconstruction_export(encoded,config)),cpu(encoded.sample_valid_mask),cohort.labels,cohort.sources,&cohort.heads[0]);
      geometry.write("labels_scoring_only",cohort.labels,true);write_text(geometry,"source_ids_json",strings(cohort.sources));
      geometry.write("channel_ids",torch::tensor(resolved_channel_ids(config),torch::kInt64),true);
    }
    embedding::archive::save_archive((directory/"geometry.pt").string(),geometry);
    write_file(directory/"geometry.json","{\"contextual\":"+contextual_json+",\"global\":"+global_json+'}');
    std::ostringstream banks;banks<<'[';bool first_bank=true;
    for(int64_t trial=0;trial<8;++trial){
      const bool actual=trial>=4;const auto attempt=512+trial-4;
      const auto indices=actual?training_detail::sampled_indices(256,8,cohort.checkpoint.settings.seed,attempt):torch::arange(256,torch::kInt64);
      const auto batch=actual?training_detail::selected(normalized,indices.to(config.device)):normalized;
      const auto labels=cohort.labels.index_select(0,indices);
      std::vector<std::string> sources;
      for(int64_t row=0;row<indices.numel();++row)sources.push_back(cohort.sources[indices[row].item<int64_t>()]);
      MaskPlan mask;
      if(actual){
        mask=make_training_mask(batch.observed,config,training_detail::counter_seed(cohort.checkpoint.settings.seed,attempt,0x6d61736bULL));
        torch::manual_seed(training_detail::counter_seed(cohort.checkpoint.settings.seed,attempt,0x746f726368ULL));
      }
      else{
        auto hidden=torch::zeros_like(batch.observed);hidden.narrow(2,trial*config.patch_length,config.patch_length).fill_(true);
        const auto visible=batch.observed.logical_and(hidden.logical_not());
        const auto eligible=visible.reshape({256,3,4,-1}).any(-1).sum(-1).ge(2).logical_and(batch.observed.logical_and(hidden).flatten(2).any(2));
        hidden=hidden.logical_and(eligible.unsqueeze(-1).unsqueeze(-1));mask=mask_from_hidden(batch.observed,hidden,config);
      }
      auto visible=mask.visible;context_deletion::Plan deletion;const auto options=policy(instance);
      const bool use_deletion=actual&&options.enabled&&context_deletion::deletion_attempt(options.recipe,attempt);
      if(use_deletion){deletion=context_deletion::make_plan(mask,batch.channel_ids,config,cohort.checkpoint.settings.seed,attempt,options.recipe);visible=deletion.visible;}
      if(!actual){
        torch::NoGradGuard no_grad;
        Input legal{torch::where(visible,batch.data,torch::zeros_like(batch.data)),visible,batch.channel_ids,batch.endpoints,batch.sampling_interval};
        const auto encoded=model->encode(legal);const auto prediction=cpu(model->decode(compact_reconstruction_export(encoded,config),batch.channel_ids)).to(torch::kFloat64);
        const auto target=cpu(batch.data).to(torch::kFloat64),query=cpu(mask.target);
        require(torch::equal(query,cohort.archived_query[trial])&&torch::equal(cpu(visible),cohort.archived_visible[trial])&&
            torch::equal(cpu(mask.eligible_channels),cohort.archived_eligible[trial])&&
            torch::equal(torch::where(query,prediction,torch::zeros_like(prediction)),cohort.archived_prediction[trial])&&
            torch::equal(torch::where(query,target,torch::zeros_like(target)),cohort.archived_target[trial]),"exact frozen-CUDA original TRAIN query parity failed");
      }
      od::BankEvidence evidence;const auto name=actual?"actual_attempt_"+std::to_string(attempt):"query_"+std::to_string(trial);
      if(!first_bank)banks<<',';first_bank=false;banks<<bank_output(directory,name,model,batch,mask,visible,labels,sources,indices,cohort.heads[0],use_deletion?&deletion:nullptr,evidence);
      if(instance.model_tag=="RPB-v7"&&actual)v7_actual[instance.master_seed][trial-4]=std::move(evidence);
    }
    banks<<']';model_state.unchanged();
    if(!first_instance)reports<<',';first_instance=false;
    reports<<"{\"input_id\":"<<quote(instance.input_id)<<",\"tag\":"<<quote(instance.model_tag)<<",\"master\":"<<instance.master_seed
        <<",\"training_policy_id\":"<<quote(instance.expected_policy_id)<<",\"training_producer_source_id\":"<<quote(instance.training_producer_source_id)
        <<",\"core_writer_source_id\":"<<quote(instance.core_writer_source_id)<<",\"orchestration_source_id\":"<<quote(instance.orchestration_source_id)
        <<",\"geometry\":{\"contextual\":"<<contextual_json<<",\"global\":"<<global_json<<"},\"banks\":"<<banks.str()<<'}';
  }
  reports<<']';bool trigger=true;int64_t conflicts=0;std::ostringstream gates;gates<<'[';bool first=true;
  for(const auto master:masters){const auto &banks=v7_actual.at(master);int64_t support=0;double base=0,combined=0,direction=0;bool supported=true;
    for(const auto &bank:banks){support+=bank.auxiliary.supported_examples;supported&=bank.comparison.supported;base+=bank.comparison.base_descent_margin_cosine/4;
      combined+=bank.comparison.combined_descent_margin_cosine/4;direction+=bank.comparison.combined_descent_base_directional_derivative/4;}
    const bool admitted=supported&&support>=16&&combined-base>1e-6&&direction<0;trigger&=admitted;if(supported&&base<0)++conflicts;
    if(!first)gates<<',';first=false;gates<<std::setprecision(17)<<"{\"master\":"<<master<<",\"supported_auxiliary_rows\":"<<support
        <<",\"total_actual_rows\":32,\"required_banks_supported\":"<<(supported?"true":"false")<<",\"base_descent_margin_cosine\":";
    if(supported)gates<<base;else gates<<"null";
    gates<<",\"combined_descent_margin_cosine\":";if(supported)gates<<combined;else gates<<"null";
    gates<<",\"combined_descent_base_directional_derivative\":";if(supported)gates<<direction;else gates<<"null";
    gates<<",\"admitted\":"<<(admitted?"true":"false")<<'}';
  }
  gates<<']';trigger&=conflicts>=3;
  std::ostringstream report;report<<"{\"protocol\":\"training-objective-diagnostic-v1\",\"status\":\"measured_zero_updates\",\"source_fingerprint\":"<<quote(run.source_fingerprint)
      <<",\"instances\":"<<reports.str()<<",\"hypothesis_gate\":{\"status\":"<<quote(trigger?"eligible_for_future_frozen_experiment":"null_stop_loss_hypothesis")
      <<",\"v7_actual_masks_only\":true,\"conflicting_masters\":"<<conflicts<<",\"masters\":"<<gates.str()
      <<"},\"limitations\":[\"TRAIN descriptive diagnosis only; no generalization measurement\",\"latent and encoder unit-descent cosines describe infinitesimal plain SGD, not saved AdamW updates\",\"geometry coherence is a coherent linear contrast, not proof of lost timing information\",\"common-patch queries have different pair support from actual per-channel training masks\",\"auxiliary is normalized reconstruction contrast reweighting, not new supervision\"],\"weight_updates\":0,\"head_refits\":0,\"test_reads\":0}";
  write_file(output/"report.json",report.str());return report.str();
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
