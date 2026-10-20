// SPDX-License-Identifier: MIT
// Separately tagged content-only relation training/export. Scoring is a separate
// CPU reader of these saved features; labels never enter this optimizer.
#include "configurable_capacity.h"
#include "frozen_role_guard.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_settings.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <regex>

#ifndef CONFIGURABLE_CONTENT_RELATION_SOURCE_ID
#define CONFIGURABLE_CONTENT_RELATION_SOURCE_ID "unrecorded"
#endif
#ifndef CONFIGURABLE_CONTENT_RELATION_CARD_SHA256
#define CONFIGURABLE_CONTENT_RELATION_CARD_SHA256 "unrecorded"
#endif

namespace {
namespace cap = embedding::evaluation::configurable_capacity;
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace io = embedding::evaluation::frozen_inputs;
namespace fs = std::filesystem;
using io::require;
using io::quote;
using Clock = std::chrono::steady_clock;
constexpr const char *protocol = "configurable-content-relation-v1";
constexpr const char *original_source = "8033ad0e20c8aa31cdda6befe340aed6001758233295fdaf78f14e3db2225fd1";
constexpr const char *original_inventory = "62f7e52a1d94815ec44ec90bae052c52ba2e89e9e809f851afdb11a15da0aa83";
constexpr const char *original_inputs = "6121773c0b97e5490f191a9f8ff013e90f27a9a557376ac8256bdbb2d6af6091";
constexpr int64_t batch_size = 8, quality_budget = 512;
constexpr uint64_t quality_masters[] = {941101, 941102, 941103};

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
torch::Tensor cpu(const torch::Tensor &value) {
  return value.detach().to(torch::kCPU).contiguous().clone();
}
bool exact(const torch::Tensor &a, const torch::Tensor &b) {
  const auto x = cpu(a), y = cpu(b);
  return x.scalar_type() == y.scalar_type() && x.sizes() == y.sizes() &&
      (!x.numel() || !std::memcmp(x.const_data_ptr(), y.const_data_ptr(), x.numel() * x.element_size()));
}
std::string tensor_digest(const torch::Tensor &value) {
  const auto t = cpu(value);
  return io::sha256(std::string(reinterpret_cast<const char *>(t.const_data_ptr()), t.numel() * t.element_size()));
}
std::string receipt_field(const std::string &json, const std::string &key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"\\\\]*)\"");
  std::sregex_iterator found(json.begin(), json.end(), pattern), end;
  require(found != end, "missing plain receipt field " + key); const auto value = (*found)[1].str();
  require(++found == end, "duplicate receipt field " + key); return value;
}
bool integer_receipt(const std::string &json, const std::string &key, int64_t value) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)(?=\\s*[,}])");
  std::sregex_iterator found(json.begin(), json.end(), pattern), end;
  if (found == end) return false;
  const auto actual = std::stoll((*found)[1].str());
  return ++found == end && actual == value;
}
double number_receipt(const std::string &json, const std::string &key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)(?=\\s*[,}])");
  std::sregex_iterator found(json.begin(), json.end(), pattern), end;
  require(found != end, "missing receipt number " + key); const auto value = std::stod((*found)[1].str());
  require(++found == end && std::isfinite(value), "unique finite receipt number " + key); return value;
}
void quality_admission(const fs::path &output, uint64_t master, const fs::path &settings, const fs::path &card,
    const fs::path &input_manifest, const std::string &source_sha, const std::string &card_sha) {
  require(output.is_absolute() && output.lexically_normal() == output && output.filename() == "master-" + std::to_string(master) &&
      output.parent_path().filename() == "candidates", "declared exclusive candidate output role");
  const auto capsule = output.parent_path().parent_path();
  require(capsule.parent_path() == fs::path("/mityane/.temp/embeddings/runs/rpb-v22-content-relation") &&
      capsule.filename().string().rfind("admission-", 0) == 0 &&
      fs::canonical(capsule) == capsule && !fs::is_symlink(capsule) &&
      fs::canonical(output.parent_path()) == output.parent_path(), "direct admitted capsule outside all historical evidence");
  require(settings == capsule / "freeze/config.conf" && card == capsule / "freeze/card.md" &&
      input_manifest == capsule / "freeze/inputs.sha256", "use exact frozen settings/card/input receipt roles");
  const auto frozen = io::bytes(capsule / "freeze/receipt.json"), admitted = io::bytes(capsule / "admission.json");
  require(receipt_field(frozen, "protocol") == protocol && receipt_field(frozen, "source_fingerprint") == source_sha &&
      receipt_field(frozen, "card_sha256") == card_sha && receipt_field(frozen, "config_sha256") == io::sha256(io::bytes(settings)) &&
      receipt_field(frozen, "input_manifest_sha256") == original_inputs &&
      io::sha256(io::bytes(capsule / "freeze/sources.sha256")) == source_sha, "complete frozen candidate provenance before construction");
  std::map<std::string, fs::path> live_roles, captured_roles; std::string role, previous;
  std::istringstream source_list(io::bytes(capsule / "freeze/source-list.txt"));
  while (std::getline(source_list, role)) {
    const fs::path relative(role);
    require(!role.empty() && !relative.is_absolute() && relative.lexically_normal() == relative &&
        role.find("..") == std::string::npos && (previous.empty() || previous < role) &&
        live_roles.emplace(role, fs::path("/mityane") / relative).second &&
        captured_roles.emplace(role, capsule / "freeze/source" / relative).second, "closed sorted source roles");
    previous = role;
  }
  require(!live_roles.empty() && integer_receipt(frozen, "source_files", int64_t(live_roles.size())), "complete frozen source population");
  io::Guard live_source(capsule / "freeze/sources.sha256", live_roles), captured_source(capsule / "freeze/sources.sha256", captured_roles);
  require(receipt_field(admitted, "protocol") == protocol && receipt_field(admitted, "status") == "admitted" &&
      receipt_field(admitted, "source_fingerprint") == source_sha && receipt_field(admitted, "card_sha256") == card_sha &&
      integer_receipt(admitted, "quality_encoder_runs", 0), "actual engineering admission before quality construction");
  const auto runner = io::bytes(capsule / "runner-engineering/complete.json"), reader = io::bytes(capsule / "readout-engineering/complete.json");
  require(receipt_field(runner, "protocol") == protocol && receipt_field(runner, "status") == "passed" &&
      integer_receipt(runner, "master", 941000) && integer_receipt(runner, "updates_each", 2) && integer_receipt(runner, "CPU_encoder_calls", 0) &&
      receipt_field(reader, "protocol") == std::string(protocol) + "-engineering" && receipt_field(reader, "status") == "passed" &&
      integer_receipt(reader, "fitted_pairs", 6) && integer_receipt(reader, "encoder_calls", 0), "two actual CUDA updates and saved-head admission receipts");
  const auto quality = io::bytes(capsule / "quality-admission.json");
  require(receipt_field(quality, "protocol") == protocol && receipt_field(quality, "status") == "admitted" &&
      receipt_field(quality, "source_fingerprint") == source_sha && receipt_field(quality, "card_sha256") == card_sha &&
      receipt_field(quality, "config_sha256") == io::sha256(io::bytes(settings)) &&
      receipt_field(quality, "input_manifest_sha256") == original_inputs &&
      receipt_field(quality, "runner_complete_sha256") == io::sha256(runner) &&
      receipt_field(quality, "runner_progress_sha256") == io::sha256(io::bytes(capsule / "runner-engineering/A/progress.json")) &&
      receipt_field(quality, "runner_checkpoint_sha256") == io::sha256(io::bytes(capsule / "runner-engineering/A/checkpoint.pt")) &&
      receipt_field(quality, "readout_complete_sha256") == io::sha256(reader) &&
      receipt_field(quality, "binaries_manifest_sha256") == io::sha256(io::bytes(capsule / "freeze/binaries.sha256")),
      "immutable admission binds actual engineering state and compiled binary receipt");
  const auto executable = fs::canonical("/proc/self/exe");
  const fs::path build(receipt_field(frozen, "build_directory"));
  std::map<std::string, fs::path> binary_roles;
  for (const std::string name : {"embedding_configurable_content_relation", "embedding_configurable_content_relation_readout",
      "code/tests/encoders/raw_patch_bottleneck_mae/content_ordered_relation/content_ordered_relation_test"})
    binary_roles.emplace((build / name).string(), build / name);
  io::Guard binaries(capsule / "freeze/binaries.sha256", binary_roles);
  require(executable == fs::path(receipt_field(frozen, "build_directory")) / "embedding_configurable_content_relation" &&
      receipt_field(quality, "producer_binary_sha256") == io::sha256(io::bytes(executable)), "actual running producer matches admitted binary bytes");
  const auto progress = io::bytes(capsule / "runner-engineering/A/progress.json");
  const auto losses = progress.find(",\"losses\":["); require(losses != std::string::npos, "actual artificial CUDA trace");
  const auto header = progress.substr(0, losses) + '}';
  require(receipt_field(header, "protocol") == protocol && receipt_field(header, "source_fingerprint") == source_sha &&
      receipt_field(header, "card_sha") == card_sha && receipt_field(header, "config_sha") == io::sha256(io::bytes(settings)) &&
      receipt_field(header, "training_device") == "cuda:0" && integer_receipt(header, "master", 941000) &&
      integer_receipt(header, "attempted", 2) && integer_receipt(header, "completed", 2) && integer_receipt(header, "skipped_attempts", 0) &&
      integer_receipt(header, "CPU_encoder_calls", 0) && number_receipt(header, "parameter_count") == number_receipt(header, "cuda_parameter_count") &&
      number_receipt(header, "relation_gradient_updates") > 0 && number_receipt(header, "relation_weight_max_absolute_change") > 0,
      "actual two-update CUDA progress has new projection gradients and weight movement");
  if (master == 941101) require(!fs::exists(capsule / "candidates/master-941102") &&
      !fs::exists(capsule / "candidates/master-941103") && !fs::exists(capsule / "screen") &&
      !fs::exists(capsule / "complete.json"), "first declared cohort precedes other quality outputs");
  if (master == 941102) {
    const auto first = capsule / "candidates/master-941101"; const auto complete = io::bytes(first / "complete.json");
    require(receipt_field(complete, "protocol") == protocol && receipt_field(complete, "status") == "complete" &&
        receipt_field(complete, "source_fingerprint") == source_sha && receipt_field(complete, "card_sha") == card_sha &&
        integer_receipt(complete, "master", 941101) && integer_receipt(complete, "updates", 512) &&
        receipt_field(complete, "report_sha256") == io::sha256(io::bytes(first / "report.json")), "complete first cohort before second allocation");
  }
}
void third_admission(const fs::path &output, const std::string &source, const std::string &card) {
  const auto capsule = output.parent_path().parent_path(), path = capsule / "third-admission.json";
  require(path.is_absolute() && fs::canonical(path) == path && !fs::is_symlink(path), "third requires direct screen admission receipt");
  const auto receipt = io::bytes(path);
  require(receipt_field(receipt, "protocol") == protocol && receipt_field(receipt, "status") == "admitted" &&
      receipt_field(receipt, "source_fingerprint") == source && receipt_field(receipt, "card_sha256") == card &&
      std::regex_search(receipt, std::regex("\"master\"\\s*:\\s*941103\\s*[,}]")), "third allocation/source/card binding");
  require(receipt_field(receipt, "screen_summary_sha256") == io::sha256(io::bytes(capsule / "screen/published/summary.json")) &&
      receipt_field(receipt, "screen_gate_sha256") == io::sha256(io::bytes(capsule / "screen/published/gate.json")) &&
      receipt_field(receipt, "consumed_manifest_sha256") == io::sha256(io::bytes(capsule / "screen/published/consumed-files.sha256")),
      "third admission binds the unchanged independently audited screen");
  const auto gate = io::bytes(capsule / "screen/published/gate.json");
  require(receipt_field(gate, "stage") == "two-master-screen" &&
      std::regex_search(gate, std::regex("\"passed\"\\s*:\\s*true")) &&
      !std::regex_search(gate, std::regex("\"passed\"\\s*:\\s*false")), "joint screen must pass before third model construction");
}
void text(torch::serialize::OutputArchive &a, const std::string &key, const std::string &value) {
  a.write(key, rpb::configurable_checkpoint_detail::text_tensor(value), true);
}
void integer(torch::serialize::OutputArchive &a, const std::string &key, int64_t value) {
  a.write(key, torch::tensor(value, torch::kInt64), true);
}
void save_archive(const fs::path &path, torch::serialize::OutputArchive &a) {
  rpb::configurable_checkpoint_detail::publish_checkpoint(a, path.string());
}
void new_leaf(const fs::path &path) {
  require(path.is_absolute() && path.lexically_normal() == path && !path.filename().empty(), "absolute normalized output leaf");
  require(fs::is_directory(path.parent_path()) && fs::canonical(path.parent_path()) == path.parent_path(), "direct existing output parent");
  for (auto p = path.parent_path(); p != p.root_path(); p = p.parent_path())
    require(!fs::is_symlink(p), "direct output ancestors");
  require(!fs::exists(path) && !fs::is_symlink(path) && fs::create_directory(path), "output leaf must be new");
}
struct RuntimeIsolation {
  int threads = at::get_num_threads();
  std::vector<at::Generator> generators;
  std::vector<torch::Tensor> states;
  RuntimeIsolation() {
    generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCPU)));
    for (size_t i = 0; i < at::getNumGPUs(); ++i)
      generators.push_back(at::globalContext().defaultGenerator(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(i))));
    for (const auto &g : generators) states.push_back(g.get_state().clone());
  }
  ~RuntimeIsolation() noexcept {
    try {
      for (size_t i = 0; i < states.size(); ++i) generators[i].set_state(states[i]);
      at::set_num_threads(threads);
    } catch (...) { std::terminate(); }
  }
};
struct MemoryPeak { int64_t allocated = 0, reserved = 0; };
MemoryPeak memory_peak() {
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto aggregate = static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
  return {stats.allocated_bytes[aggregate].peak, stats.reserved_bytes[aggregate].peak};
}

std::string tag(const std::string &profile) {
  require(profile == "A", "one closed learned relation profile");
  return "RPB-v22.alt-01";
}
rpb::Config config(const fs::path &path, const std::string &profile) {
  auto c = rpb::read_content_ordered_relation_settings(path);
  require(c.channel_count == 3 && c.history_length == 128 && c.input_width == 9 && c.patch_length == 8 &&
      c.encoder_width == 64 && c.export_width == 32 && c.num_layers == 3 && c.num_heads == 4 &&
      c.feedforward_width == 256 && c.decoder_hidden_width == 128 && c.mask_ratio == .25 &&
      c.huber_delta == 1. && c.scale_floor == 1e-6 && c.layer_norm_epsilon == 1e-5 &&
      c.sampling_interval == 1. && c.device == torch::Device(torch::kCUDA, 0) &&
      rpb::resolved_channel_ids(c) == std::vector<int64_t>({0, 1, 2}), "frozen base model settings");
  (void)tag(profile);
  rpb::validate_configurable_spectral_config(c);
  return c;
}
std::string config_json(const rpb::Config &c) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"model_tag\":\"RPB-v22\",\"architecture_id\":" << quote(rpb::kContentOrderedRelationArchitectureId)
      << ",\"native_layout\":" << quote(rpb::kContentOrderedRelationLayout)
      << ",\"relation_lags\":[1,2,4,8],\"relation_minimum_common\":2,\"relation_energy_floor\":1e-12,\"relation_projection_initialization\":\"zero\""
      << ",\"additive_support_coordinates\":false"
      << ",\"relation_bank_width\":" << rpb::content_ordered_relation_bank_width(c)
      << ",\"channels\":" << c.channel_count << ",\"history\":" << c.history_length
      << ",\"features\":" << c.input_width << ",\"patch_length\":" << c.patch_length
      << ",\"encoder_width\":" << c.encoder_width << ",\"embedding_size\":" << c.export_width
      << ",\"learned_coordinates\":" << rpb::configurable_spectral_shape_width(c)
      << ",\"fixed_coordinates\":" << rpb::configurable_spectral_width(c)
      << ",\"layers\":" << c.num_layers << ",\"heads\":" << c.num_heads
      << ",\"feedforward_width\":" << c.feedforward_width << ",\"decoder_hidden_width\":" << c.decoder_hidden_width
      << ",\"channel_mixer_layers\":1,\"channel_mixer_placement\":1,\"global_bottleneck_mode\":2,\"global_pool_input_source\":0,\"temporal_difference_input\":0"
      << ",\"mask_ratio\":" << c.mask_ratio << ",\"context_deletion_ratio\":0.15,\"huber_delta\":" << c.huber_delta
      << ",\"dropout\":0,\"layer_norm_epsilon\":" << c.layer_norm_epsilon << ",\"scale_floor\":" << c.scale_floor
      << ",\"sampling_interval\":" << c.sampling_interval << ",\"channel_ids\":[0,1,2],\"device\":\"cuda:0\",\"dtype\":\"float32\""
      << ",\"batch_size\":8,\"optimizer\":\"AdamW\",\"learning_rate\":0.001,\"weight_decay\":0.0001,\"gradient_clip_norm\":1}";
  return out.str();
}
rpb::Input input(const cap::Split &split, const rpb::Config &c) {
  return {split.observed, split.mask, torch::tensor(rpb::resolved_channel_ids(c), torch::kInt64),
      torch::full({split.observed.size(0)}, double(c.history_length - 1), torch::kFloat64), c.sampling_interval};
}
rpb::Input select(const rpb::Input &value, int64_t begin, int64_t count) {
  return rpb::training_detail::selected(value, torch::arange(begin, begin + count, torch::kInt64));
}
struct Cohort { cap::Split training, validation, deleted; };
void check_cohort(const Cohort &data, uint64_t master, int64_t train_rows, int64_t validation_rows) {
  const auto shape = [&](const cap::Split &s, int64_t rows) {
    cap::validate(s);
    require(s.observed.sizes() == torch::IntArrayRef({rows, 3, 128, 9}), "declared dataset geometry and rows");
    const auto ids = s.source_ids.accessor<int64_t, 1>();
    const auto labels = s.scoring_labels.accessor<int64_t, 1>();
    std::set<int64_t> sources;
    for (int64_t row = 0; row < rows; row += 2) {
      require(ids[row] >= 0 && uint64_t(ids[row]) >> 31 == master && ids[row] == ids[row + 1] &&
          sources.insert(ids[row]).second && labels[row] != labels[row + 1] &&
          torch::equal(s.mask[row], s.mask[row + 1]), "paired source identities, opposite labels and shared masks");
    }
    return sources;
  };
  const auto training = shape(data.training, train_rows), validation = shape(data.validation, validation_rows);
  shape(data.deleted, validation_rows);
  for (const auto id : validation) require(!training.count(id), "source-disjoint TRAIN/VALIDATION");
  require(torch::equal(data.validation.source_ids, data.deleted.source_ids) &&
      torch::equal(data.validation.scoring_labels, data.deleted.scoring_labels) &&
      !data.deleted.mask.logical_and(~data.validation.mask).any().item<bool>() &&
      torch::equal(data.deleted.observed.masked_select(data.deleted.mask), data.validation.observed.masked_select(data.deleted.mask)),
      "deleted validation preserves identities/labels and only original observations");
}

struct Binding {
  std::string profile, model_tag, source_fingerprint, card_sha, input_manifest_sha, config_sha;
  uint64_t master = 0;
  std::string json() const {
    return "\"protocol\":" + quote(protocol) + ",\"dataset_id\":\"TEMPO-5\",\"profile\":" + quote(profile) +
        ",\"model_tag\":" + quote(model_tag) + ",\"master\":" + std::to_string(master) +
        ",\"source_fingerprint\":" + quote(source_fingerprint) + ",\"card_sha\":" + quote(card_sha) +
        ",\"input_manifest_sha\":" + quote(input_manifest_sha) + ",\"config_sha\":" + quote(config_sha);
  }
  void write(torch::serialize::OutputArchive &a, int64_t point, const rpb::Config &c, const std::string &scaler_id) const {
    text(a, "protocol", protocol); text(a, "dataset_id", "TEMPO-5"); text(a, "profile", profile); text(a, "model_tag", model_tag);
    text(a, "source_fingerprint", source_fingerprint); text(a, "card_sha", card_sha); text(a, "input_manifest_sha", input_manifest_sha);
    text(a, "config_sha", config_sha); text(a, "scaler_identity", scaler_id); text(a, "resolved_settings_json", config_json(c));
    integer(a, "master", int64_t(master)); integer(a, "point", point); integer(a, "native_width", c.export_width);
  }
};
struct Trace {
  int64_t attempted = 0, completed = 0, target_cells = 0, requested = 0, actual = 0, restored = 0;
  uint64_t mask_seed = 0, torch_seed = 0;
  double loss = 0, gradient = 0, relation_gradient = 0;
  std::vector<int64_t> rows;
  std::string hidden_sha, visible_sha;
};
struct Trainer {
  rpb::Config c;
  Binding binding;
  rpb::Input raw;
  rpb::FrozenScaler scaler;
  rpb::ContentOrderedRelationModel model{nullptr};
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::vector<torch::Tensor> initial;
  std::vector<Trace> trace;
  int64_t parameter_count = 0, trainable_count = 0, frozen_count = 0;
  int64_t relation_gradient_updates = 0;
  double training_seconds = 0;
  MemoryPeak training_memory;

  Trainer(const rpb::Config &settings, const cap::Split &training, Binding identity)
      : c(settings), binding(std::move(identity)), raw(input(training, c)) {
    scaler = rpb::fit_configurable_spectral_scaler(raw, c); // TRAIN only; labels never passed.
    torch::manual_seed(rpb::training_detail::mixed(binding.master ^ 0x7270622d696e6974ULL));
    model = rpb::ContentOrderedRelationModel(c);
    validate_trainability();
    require(model->relation_projection->weight.eq(0).all().item<bool>(), "new projection must start exactly zero");
    for (const auto &p : model->parameters()) initial.push_back(cpu(p));
    optimizer = std::make_unique<torch::optim::AdamW>(model->parameters(), torch::optim::AdamWOptions(.001).weight_decay(.0001));
  }
  void validate_trainability() {
    parameter_count = trainable_count = frozen_count = 0;
    bool prior_found = false, relation_found = false;
    for (const auto &p : model->named_parameters()) {
      require(p.value().is_cuda() && p.value().scalar_type() == torch::kFloat32 &&
          torch::isfinite(p.value()).all().item<bool>(), "all parameters finite CUDA float32");
      const bool prior = p.key() == "base.grouped_odd_relation_projection.weight";
      if (p.key() == "relation_projection.weight") {
        relation_found = true;
        require(p.value().requires_grad() && p.value().dim() == 2 && p.value().size(0) == 20 && p.value().size(1) == 1944,
            "declared content-only relation CUDA projection20x1944");
      }
      require(p.value().requires_grad() != prior, "only literal spectral prior frozen");
      if (prior) {
        prior_found = true;
        frozen_count += p.value().numel();
        require(!p.value().grad().defined(), "frozen prior has no gradient");
      } else trainable_count += p.value().numel();
      parameter_count += p.value().numel();
    }
    require(prior_found && relation_found && frozen_count == 3 * 4 * 9 * 9 * 4 && trainable_count > 0, "expected all-feature fixed prior and learned relation projection");
    model->validate_fixed_prior();
  }
  bool changed() const {
    const auto parameters = model->parameters();
    for (size_t i = 0; i < parameters.size(); ++i) if (!exact(parameters[i], initial[i])) return true;
    return false;
  }
  double relation_weight_change() const {
    return cpu(model->relation_projection->weight).abs().max().item<double>();
  }
  void verify_frozen() {
    validate_trainability();
    const auto p = model->parameters();
    for (size_t i = 0; i < p.size(); ++i) if (!p[i].requires_grad()) require(exact(p[i], initial[i]), "frozen prior unchanged from initialization");
  }
  void train_to(int64_t budget) {
    require(budget >= int64_t(trace.size()) && budget <= quality_budget, "monotonic closed update budget");
    model->train();
    torch::cuda::synchronize(0);
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    const auto start = Clock::now();
    while (int64_t(trace.size()) < budget) {
      const auto attempt = int64_t(trace.size());
      const auto rows = rpb::training_detail::sampled_indices(raw.data.size(0), batch_size, int64_t(binding.master), attempt);
      const auto normalized = rpb::transform_configurable_spectral_input(scaler, rpb::training_detail::selected(raw, rows), c);
      require(normalized.data.is_cuda() && normalized.observed.is_cuda(), "CUDA training data and observed masks");
      Trace record; record.attempted = attempt + 1; record.completed = attempt + 1;
      record.rows.assign(rows.const_data_ptr<int64_t>(), rows.const_data_ptr<int64_t>() + rows.numel());
      record.mask_seed = rpb::training_detail::counter_seed(int64_t(binding.master), attempt, 0x6d61736bULL);
      record.torch_seed = rpb::training_detail::counter_seed(int64_t(binding.master), attempt, 0x746f726368ULL);
      const auto mask = rpb::configurable_spectral_make_training_mask(normalized.observed, c, record.mask_seed);
      require(mask.eligible_channels.any().item<bool>(), "no skipped original-mask attempts; abort if unsupported");
      // H128/P8 is divisible. The support-only shared deletion planner has no
      // D32 restriction; we intentionally do not use its legacy forward helper.
      const auto context = rpb::context_deletion::make_plan(mask, normalized.channel_ids, c,
          int64_t(binding.master), attempt, rpb::ContextDeletionRecipe::coordinate15_v1);
      record.requested = context.requested_count; record.actual = context.actual_count; record.restored = context.restored_count;
      record.hidden_sha = tensor_digest(mask.hidden); record.visible_sha = tensor_digest(context.visible);
      torch::manual_seed(record.torch_seed);
      const rpb::Input visible{torch::where(context.visible, normalized.data, torch::zeros_like(normalized.data)),
          context.visible, normalized.channel_ids, normalized.endpoints, normalized.sampling_interval};
      optimizer->zero_grad();
      const auto encoded = model->encode(visible);
      const auto native = rpb::compact_reconstruction_export(encoded, c);
      require(native.is_cuda() && native.sizes() == torch::IntArrayRef({batch_size, c.export_width}), "sole CUDA native configured bottleneck");
      const auto prediction = model->decode(native, normalized.channel_ids);
      const auto loss = rpb::hierarchical_huber(prediction, normalized.data.detach(), mask.target, mask.eligible_channels, c.huber_delta);
      require(prediction.is_cuda() && loss.loss.is_cuda() && loss.eligible_example_count > 0 &&
          torch::isfinite(loss.loss).item<bool>(), "finite eligible CUDA waveform Huber loss");
      loss.loss.backward();
      const auto relation_gradient = model->relation_projection->weight.grad();
      require(relation_gradient.defined() && relation_gradient.is_cuda() &&
          torch::isfinite(relation_gradient).all().item<bool>(), "actual finite CUDA gradient on new relation projection");
      record.relation_gradient = relation_gradient.norm().item<double>();
      require(std::isfinite(record.relation_gradient), "finite new relation gradient norm");
      if (record.relation_gradient > 0) ++relation_gradient_updates;
      for (const auto &p : model->parameters())
        require(!p.grad().defined() || (p.grad().is_cuda() && torch::isfinite(p.grad()).all().item<bool>()), "finite CUDA gradients");
      record.gradient = torch::nn::utils::clip_grad_norm_(model->parameters(), 1., 2., true);
      require(std::isfinite(record.gradient) && record.gradient > 0, "finite nonzero CUDA gradient norm");
      optimizer->step();
      record.target_cells = loss.target_cell_count; record.loss = loss.loss.item<double>();
      trace.push_back(std::move(record));
      if (trace.size() % 128 == 0) std::cout << binding.model_tag << " master=" << binding.master << " CUDA_updates=" << trace.size() << '/' << budget << '\n' << std::flush;
    }
    torch::cuda::synchronize(0);
    training_seconds += seconds(start); training_memory = memory_peak(); verify_frozen();
  }
  std::string progress() const {
    std::ostringstream out; out << std::setprecision(17) << '{' << binding.json() << ",\"resolved_settings\":" << config_json(c)
        << ",\"attempted\":" << trace.size() << ",\"completed\":" << trace.size() << ",\"skipped_attempts\":0,\"sampled_rows\":" << batch_size * trace.size()
        << ",\"parameter_count\":" << parameter_count << ",\"cuda_parameter_count\":" << parameter_count
        << ",\"trainable_parameter_count\":" << trainable_count << ",\"frozen_parameter_count\":" << frozen_count
        << ",\"frozen_parameter_name\":\"base.grouped_odd_relation_projection.weight\",\"training_device\":\"cuda:0\",\"CPU_encoder_calls\":0"
        << ",\"relation_gradient_updates\":" << relation_gradient_updates
        << ",\"relation_weight_max_absolute_change\":" << relation_weight_change()
        << ",\"training_seconds\":" << training_seconds << ",\"training_peak_allocated_bytes\":" << training_memory.allocated
        << ",\"training_peak_reserved_bytes\":" << training_memory.reserved << ",\"weights_changed\":" << (changed() ? "true" : "false")
        << ",\"scaler_identity\":" << quote(scaler.identity()) << ",\"initialization_seed\":" << rpb::training_detail::mixed(binding.master ^ 0x7270622d696e6974ULL)
        << ",\"context_recipe\":\"rpb-training-context-deletion-015-v1\",\"losses\":[";
    for (size_t i = 0; i < trace.size(); ++i) {
      const auto &t = trace[i]; if (i) out << ',';
      out << "{\"attempted\":" << t.attempted << ",\"completed\":" << t.completed << ",\"target_cells\":" << t.target_cells
          << ",\"total_loss\":" << t.loss << ",\"waveform_loss\":" << t.loss << ",\"dynamics_loss\":0,\"gradient_norm\":" << t.gradient << ",\"relation_gradient_norm\":" << t.relation_gradient
          << ",\"context_requested\":" << t.requested << ",\"context_actual\":" << t.actual << ",\"context_restored\":" << t.restored
          << ",\"mask_seed\":" << t.mask_seed << ",\"torch_seed\":" << t.torch_seed
          << ",\"hidden_sha256\":" << quote(t.hidden_sha) << ",\"visible_sha256\":" << quote(t.visible_sha) << ",\"sampled_indices\":[";
      for (size_t j = 0; j < t.rows.size(); ++j) { if (j) out << ','; out << t.rows[j]; }
      out << "]}";
    }
    return out.str() + "]}";
  }
  void save(const fs::path &path) {
    verify_frozen();
    rpb::save_content_ordered_relation_checkpoint(path.string(), model, scaler, int64_t(trace.size()));
  }
};

struct Features { torch::Tensor values, valid; };
void exact_reload_state(const rpb::ContentOrderedRelationModel &before, const rpb::ContentOrderedRelationModel &loaded) {
  const auto parameters = before->named_parameters(), restored = loaded->named_parameters();
  require(parameters.size() == restored.size(), "typed reload parameter inventory");
  for (const auto &p : parameters) require(restored.contains(p.key()) && restored[p.key()].is_cuda() &&
      exact(p.value(), restored[p.key()]), "typed CUDA reload exact parameter " + p.key());
  const auto buffers = before->named_buffers(), restored_buffers = loaded->named_buffers();
  require(buffers.size() == restored_buffers.size(), "typed reload buffer inventory");
  for (const auto &b : buffers) require(restored_buffers.contains(b.key()) &&
      exact(b.value(), restored_buffers[b.key()]), "typed reload exact buffer " + b.key());
  loaded->validate_state(); loaded->validate_fixed_prior();
}
Features extract(const cap::Split &split, rpb::LoadedContentOrderedRelationEncoder &loaded) {
  torch::NoGradGuard no_grad; loaded.model->eval();
  const auto raw = input(split, loaded.config);
  std::vector<torch::Tensor> values, validity;
  for (int64_t begin = 0; begin < raw.data.size(0); begin += batch_size) {
    const auto count = std::min(batch_size, raw.data.size(0) - begin);
    const auto normalized = rpb::transform_configurable_spectral_input(loaded.scaler, select(raw, begin, count), loaded.config);
    require(normalized.data.is_cuda(), "CUDA extraction input");
    const auto e = loaded.model->encode(normalized);
    const auto z = rpb::compact_reconstruction_export(e, loaded.config);
    require(z.is_cuda() && z.sizes() == torch::IntArrayRef({count, loaded.config.export_width}), "native CUDA export geometry");
    values.push_back(z.detach().to(torch::kCPU, torch::kFloat64).contiguous());
    validity.push_back(cpu(e.sample_valid_mask));
  }
  return {torch::cat(values), torch::cat(validity)};
}
struct FeatureViews { Features training, validation, deleted; };
struct OriginalControls {
  fs::path root;
  std::map<std::string, std::string> inventory, used;
  explicit OriginalControls(const fs::path &p) : root(p) {
    require(root.is_absolute() && fs::canonical(root) == root && !fs::is_symlink(root), "direct original capsule");
    require(io::sha256(io::bytes(root / "artifacts.sha256")) == original_inventory, "pinned original complete manifest");
    std::istringstream lines(io::bytes(root / "artifacts.sha256")); std::string line;
    while (std::getline(lines, line)) {
      require(line.size() > 66 && io::is_sha(line.substr(0, 64)) && line.substr(64, 2) == "  ", "closed original inventory line");
      const auto role = line.substr(66); const fs::path relative(role);
      require(!relative.is_absolute() && relative.lexically_normal() == relative && role.find("..") == std::string::npos &&
          inventory.emplace(role, line.substr(0, 64)).second, "safe unique original artifact role");
    }
    require(inventory.size() == 815, "all original artifact roles bound");
  }
  fs::path bind(const std::string &role) {
    require(inventory.count(role), "declared original artifact role"); const auto p = root / role;
    require(fs::is_regular_file(p) && !fs::is_symlink(p) && fs::canonical(p) == p &&
        io::sha256(io::bytes(p)) == inventory.at(role), "original control artifact exact bytes");
    used.emplace(role, inventory.at(role)); return p;
  }
  void verify() {
    require(io::sha256(io::bytes(root / "artifacts.sha256")) == original_inventory, "original manifest remains unchanged");
    for (const auto &[role, hash] : used) require(io::sha256(io::bytes(root / role)) == hash, "original control remained unchanged");
  }
  std::string json() const {
    std::ostringstream out; out << '['; bool first = true;
    for (const auto &[role, digest] : used) { if (!first) out << ','; first = false;
      out << "{\"path\":" << quote((root / role).string()) << ",\"sha256\":" << quote(digest) << '}'; }
    return out.str() + ']';
  }
};
FeatureViews saved_features(OriginalControls &original, uint64_t master, int64_t point, const Cohort &data,
    const std::string &scaler_identity) {
  const auto role = "profiles/A/master-" + std::to_string(master) + "/point-" + std::to_string(point) + "/features.pt";
  torch::serialize::InputArchive a; a.load_from(original.bind(role).string(), torch::kCPU);
  namespace d = rpb::configurable_checkpoint_detail;
  require(d::read_text(a, "protocol") == "configurable-capacity-v1" && d::read_text(a, "model_tag") == "RPB-v20.alt-01" &&
      d::read_text(a, "profile") == "A" && d::read_text(a, "dataset_id") == "TEMPO-5" &&
      d::read_text(a, "source_fingerprint") == original_source && d::read_text(a, "input_manifest_sha") == original_inputs &&
      d::read_text(a, "scaler_identity") == scaler_identity, "saved control source/geometry/TRAIN scaler identity");
  torch::Tensor scalar;
  a.read("master", scalar, true); require(scalar.item<int64_t>() == int64_t(master), "saved master");
  a.read("point", scalar, true); require(scalar.item<int64_t>() == point, "saved point");
  a.read("native_width", scalar, true); require(scalar.item<int64_t>() == 32, "saved native32 width");
  const auto view = [&](const std::string &name, const cap::Split &s) {
    Features f; a.read(name + "_values", f.values, true); a.read(name + "_valid", f.valid, true);
    require(f.values.device().is_cpu() && f.values.scalar_type() == torch::kFloat64 &&
        f.values.sizes() == torch::IntArrayRef({s.observed.size(0), 32}) && torch::isfinite(f.values).all().item<bool>() &&
        f.valid.scalar_type() == torch::kBool && f.valid.sizes() == torch::IntArrayRef({s.observed.size(0)}) &&
        exact(f.valid, s.mask.flatten(1).any(1)), "original row geometry/exact legal validity");
    return f;
  };
  return {view("training", data.training), view("validation", data.validation), view("deleted", data.deleted)};
}
void exact_initial_control(const FeatureViews &candidate, const FeatureViews &control) {
  for (const auto &pair : std::vector<std::pair<Features, Features>>{{candidate.training, control.training},
      {candidate.validation, control.validation}, {candidate.deleted, control.deleted}})
    require(exact(pair.first.values, pair.second.values) && exact(pair.first.valid, pair.second.valid),
        "actual CUDA candidate initialized native32 must exactly match saved v20 control");
}
void exact_fixed_control(const FeatureViews &candidate, const FeatureViews &control) {
  for (const auto &pair : std::vector<std::pair<Features, Features>>{{candidate.training, control.training},
      {candidate.validation, control.validation}, {candidate.deleted, control.deleted}})
    require(exact(pair.first.values.narrow(1, 20, 12), pair.second.values.narrow(1, 20, 12)) &&
        exact(pair.first.valid, pair.second.valid), "candidate fixed12 exactly matches original values and validity");
}
FeatureViews feature_views(const Cohort &data, rpb::LoadedContentOrderedRelationEncoder &loaded) {
  return {extract(data.training, loaded), extract(data.validation, loaded), extract(data.deleted, loaded)};
}
void write_features(const fs::path &path, const FeatureViews &features, const Binding &binding,
    int64_t point, const rpb::Config &c, const std::string &scaler_id, const Cohort &data) {
  torch::serialize::OutputArchive a; binding.write(a, point, c, scaler_id);
  text(a, "artifact_kind", "configurable_content_relation_native_feature_views_v1");
  for (const auto &entry : std::vector<std::pair<std::string, Features>>{{"training", features.training}, {"validation", features.validation}, {"deleted", features.deleted}}) {
    a.write(entry.first + "_values", entry.second.values, true);
    a.write(entry.first + "_valid", entry.second.valid, true);
    const auto &split = entry.first == "training" ? data.training : entry.first == "validation" ? data.validation : data.deleted;
    a.write(entry.first + "_source_ids", split.source_ids, true);
    text(a, entry.first + "_source_ids_json", cap::source_ids_json(split));
  }
  text(a, "provenance", std::string(protocol) + "/" + binding.model_tag + "/master-" + std::to_string(binding.master) +
      "/point-" + std::to_string(point) + "; exact native CUDA export; no post-encoder PCA");
  save_archive(path, a);
}
struct NativeMovement { int64_t changed_cells = 0; double max_absolute_change = 0; };
NativeMovement unchanged_fixed_features(const FeatureViews &initial, const FeatureViews &trained, const rpb::Config &c) {
  const auto start = rpb::configurable_spectral_shape_width(c), width = rpb::configurable_spectral_width(c);
  for (const auto &pair : std::vector<std::pair<Features, Features>>{{initial.training, trained.training}, {initial.validation, trained.validation}, {initial.deleted, trained.deleted}})
    require(exact(pair.first.valid, pair.second.valid) && exact(pair.first.values.narrow(1, start, width), pair.second.values.narrow(1, start, width)),
        "exact unchanged native fixed spectral coordinates at 0/512 for all legal views");
  const auto delta = trained.training.values.narrow(1, 0, start) - initial.training.values.narrow(1, 0, start);
  NativeMovement movement{delta.ne(0).sum().item<int64_t>(), delta.abs().max().item<double>()};
  require(movement.changed_cells > 0 && std::isfinite(movement.max_absolute_change) && movement.max_absolute_change > 0,
      "learned TRAIN native coordinates must move after CUDA training");
  return movement;
}

// Compact original query bank: every time/feature coordinate appears in one
// leave-one-patch-out context. trial_index_for_time explicitly identifies that
// context, avoiding K copies of zeros and never implying one shared embedding.
std::string write_queries(const fs::path &path, const cap::Split &split,
    rpb::LoadedContentOrderedRelationEncoder &loaded, const Binding &binding) {
  torch::NoGradGuard no_grad; loaded.model->eval();
  const auto &c = loaded.config;
  const auto B = split.observed.size(0), K = c.history_length / c.patch_length;
  auto prediction = torch::zeros_like(split.observed), target = torch::zeros_like(split.observed);
  auto query = torch::zeros_like(split.mask);
  auto requested = torch::zeros_like(split.mask);
  auto trial_eligible = torch::zeros({K, B, c.channel_count}, torch::kBool);
  const auto raw = input(split, c);
  int64_t calls = 0;
  for (int64_t begin = 0; begin < B; begin += batch_size) {
    const auto count = std::min(batch_size, B - begin);
    const auto normalized = rpb::transform_configurable_spectral_input(loaded.scaler, select(raw, begin, count), c);
    for (int64_t trial = 0; trial < K; ++trial) {
      auto hidden = torch::zeros_like(normalized.observed);
      hidden.narrow(2, trial * c.patch_length, c.patch_length).fill_(true);
      const auto requested_query = normalized.observed.logical_and(hidden);
      const auto visible = normalized.observed.logical_and(~hidden);
      const auto eligible = visible.reshape({count, c.channel_count, K, -1}).any(-1).sum(-1).ge(2)
          .logical_and(requested_query.flatten(2).any(2));
      hidden = hidden.logical_and(eligible.unsqueeze(-1).unsqueeze(-1));
      const auto mask = rpb::configurable_spectral_mask_from_hidden(normalized.observed, hidden, c);
      require(torch::equal(mask.eligible_channels, eligible), "fixed original-query eligibility");
      const rpb::Input context{torch::where(mask.visible, normalized.data, torch::zeros_like(normalized.data)),
          mask.visible, normalized.channel_ids, normalized.endpoints, normalized.sampling_interval};
      const auto e = loaded.model->encode(context);
      const auto z = rpb::compact_reconstruction_export(e, c);
      const auto p = loaded.model->decode(z, normalized.channel_ids); ++calls;
      require(p.is_cuda() && torch::isfinite(p).all().item<bool>(), "finite CUDA original query reconstruction");
      const auto target_mask = mask.target.logical_and(eligible.unsqueeze(-1).unsqueeze(-1));
      const auto start = trial * c.patch_length;
      prediction.narrow(0, begin, count).narrow(2, start, c.patch_length).copy_(cpu(torch::where(target_mask, p, torch::zeros_like(p))).narrow(2, start, c.patch_length));
      target.narrow(0, begin, count).narrow(2, start, c.patch_length).copy_(cpu(torch::where(target_mask, normalized.data, torch::zeros_like(normalized.data))).to(torch::kFloat64).narrow(2, start, c.patch_length));
      query.narrow(0, begin, count).narrow(2, start, c.patch_length).copy_(cpu(target_mask).narrow(2, start, c.patch_length));
      requested.narrow(0, begin, count).narrow(2, start, c.patch_length).copy_(cpu(requested_query).narrow(2, start, c.patch_length));
      trial_eligible[trial].narrow(0, begin, count).copy_(cpu(eligible));
    }
  }
  torch::cuda::synchronize(0);
  const auto error = prediction - target, absolute = error.abs();
  const auto huber = torch::where(absolute.le(c.huber_delta), .5 * error.clamp(-c.huber_delta, c.huber_delta).square(),
      c.huber_delta * (absolute - .5 * c.huber_delta));
  const std::vector<int64_t> axes{2, 3};
  const auto counts = query.sum(axes), channel_valid = counts.gt(0);
  const auto channel_mae = absolute.sum(axes) / counts.clamp_min(1).to(torch::kFloat64);
  const auto channel_huber = huber.sum(axes) / counts.clamp_min(1).to(torch::kFloat64);
  const auto example_valid = channel_valid.any(1);
  const auto channel_count = channel_valid.sum(1).clamp_min(1).to(torch::kFloat64);
  const auto example_mae = channel_mae.sum(1) / channel_count, example_huber = channel_huber.sum(1) / channel_count;
  const auto valid_examples = example_valid.sum().item<int64_t>(), valid_cells = counts.sum().item<int64_t>();
  require(valid_examples > 0 && torch::isfinite(example_mae).all().item<bool>(), "finite supported original-query summaries");
  torch::serialize::OutputArchive a; binding.write(a, loaded.training_updates, c, loaded.scaler.identity());
  text(a, "artifact_kind", "configurable_capacity_compact_original_query_bank_v1");
  text(a, "query_semantics", "K leave-one-complete-patch-out contexts; each coordinate once; context=time/P; original observations only; no extra context deletion");
  a.write("standardized_prediction", prediction, true); a.write("standardized_target", target, true);
  a.write("target_mask", query, true); a.write("requested_observed_target_mask", requested, true);
  a.write("original_observed_mask", split.mask, true); a.write("trial_channel_eligible", trial_eligible, true);
  a.write("trial_index_for_time", torch::arange(c.history_length, torch::kInt64).floor_divide(c.patch_length), true);
  a.write("channel_target_counts", counts, true); a.write("channel_valid", channel_valid, true);
  a.write("channel_standardized_mae", channel_mae, true); a.write("channel_standardized_huber", channel_huber, true);
  a.write("example_valid", example_valid, true); a.write("example_standardized_mae", example_mae, true);
  a.write("example_standardized_huber", example_huber, true); a.write("source_ids", split.source_ids, true);
  text(a, "source_ids_json", cap::source_ids_json(split)); integer(a, "patch_length", c.patch_length); integer(a, "query_cuda_forward_calls", calls);
  save_archive(path, a);
  std::ostringstream out; out << std::setprecision(17) << "{\"status\":\"measured\",\"units\":\"frozen TRAIN-scaler standardized\",\"total_examples\":" << B
      << ",\"valid_examples\":" << valid_examples << ",\"coverage\":" << double(valid_examples) / B
      << ",\"valid_target_cells\":" << valid_cells << ",\"requested_observed_target_cells\":" << requested.sum().item<int64_t>()
      << ",\"standardized_mae\":" << example_mae.masked_select(example_valid).mean().item<double>()
      << ",\"standardized_huber\":" << example_huber.masked_select(example_valid).mean().item<double>()
      << ",\"query_contexts_per_example\":" << K << ",\"CUDA_forward_calls\":" << calls << ",\"CPU_encoder_calls\":0,\"artifact\":" << quote(path.filename().string()) << '}';
  return out.str();
}

void save_training_state(const fs::path &path, Trainer &trainer) {
  torch::serialize::OutputArchive a, optimizer;
  trainer.binding.write(a, int64_t(trainer.trace.size()), trainer.c, trainer.scaler.identity());
  text(a, "artifact_kind", "configurable_content_relation_optimizer_witness_v1");
  text(a, "resume_contract", "state witness only; no resume CLI");
  trainer.optimizer->save(optimizer); a.write("optimizer", optimizer);
  a.write("rng_cpu", at::globalContext().defaultGenerator(at::Device(at::kCPU)).get_state().clone(), true);
  a.write("rng_cuda_0", at::globalContext().defaultGenerator(at::Device(at::kCUDA, 0)).get_state().clone(), true);
  save_archive(path, a);
}

void save_scaler_witness(const fs::path &path, Trainer &trainer, const cap::Split &training) {
  // CPU statistics only: audits can establish the TRAIN fitting population
  // without constructing or forwarding the CUDA model.
  torch::serialize::OutputArchive a;
  trainer.binding.write(a, int64_t(trainer.trace.size()), trainer.c, trainer.scaler.identity());
  text(a, "artifact_kind", "configurable_capacity_train_scaler_witness_v1");
  for (const auto &entry : std::vector<std::pair<std::string, torch::Tensor>>{
      {"mean", trainer.scaler.mean}, {"scale", trainer.scaler.scale}, {"count", trainer.scaler.count},
      {"channel_ids", trainer.scaler.channel_ids}, {"floor_applied", trainer.scaler.floor_applied}})
    a.write(entry.first, cpu(entry.second), true);
  a.write("scale_floor", torch::tensor(trainer.scaler.scale_floor, torch::kFloat64), true);
  a.write("training_source_ids", training.source_ids, true);
  text(a, "training_source_ids_json", cap::source_ids_json(training));
  save_archive(path, a);
}

void engineering(const fs::path &output, const fs::path &settings, bool &owned) {
  require(torch::cuda::is_available(), "CUDA admission mandatory"); new_leaf(output); owned = true;
  const auto development = cap::make_development(8, 4, 941000);
  Cohort fixture{development.training, development.validation,
      cap::delete_observations(development.validation, .30, rpb::training_detail::mixed(941000 ^ cap::kDeletionStream))};
  check_cohort(fixture, 941000, 16, 8);
  for (const std::string profile : {"A"}) {
    const auto dir = output / profile; require(fs::create_directory(dir), "new admission profile");
    const auto c = config(settings, profile);
    Binding binding{profile, tag(profile), CONFIGURABLE_CONTENT_RELATION_SOURCE_ID, CONFIGURABLE_CONTENT_RELATION_CARD_SHA256,
        "engineering-artificial", io::sha256(io::bytes(settings)), 941000};
    Trainer trainer(c, fixture.training, binding); trainer.train_to(2);
    require(trainer.changed() && trainer.relation_gradient_updates > 0 && trainer.relation_weight_change() > 0,
        "actual CUDA updates change new relation weights through finite gradients");
    trainer.save(dir / "checkpoint.pt");
    bool refused = false;
    try { trainer.save(dir / "checkpoint.pt"); } catch (const std::exception &) { refused = true; }
    require(refused, "typed checkpoint refuses replacing admission state");
    const auto normalized = rpb::transform_configurable_spectral_input(trainer.scaler, input(fixture.validation, c), c);
    torch::NoGradGuard no_grad; trainer.model->eval();
    const auto before = rpb::compact_reconstruction_export(trainer.model->encode(normalized), c);
    auto loaded = rpb::load_content_ordered_relation_checkpoint((dir / "checkpoint.pt").string(), c.device, &c);
    require(loaded.training_updates == 2 && loaded.scaler.identity() == trainer.scaler.identity(), "typed immutable checkpoint metadata");
    const auto after = extract(fixture.validation, loaded);
    require(exact(before.to(torch::kFloat64), after.values), "bit-exact CUDA checkpoint native reload");
    const auto features = feature_views(fixture, loaded); write_features(dir / "features.pt", features, binding, 2, c, trainer.scaler.identity(), fixture);
    const auto query = write_queries(dir / "reconstruction-validation.pt", fixture.validation, loaded, binding);
    io::write_new(dir / "progress.json", trainer.progress() + '\n');
    io::write_new(dir / "query.json", query + '\n');
    std::cout << binding.model_tag << " admission CUDA2 updates; parameters=" << trainer.parameter_count
        << " trainable=" << trainer.trainable_count << " frozen=" << trainer.frozen_count << " native=" << c.export_width << '\n' << std::flush;
  }
  io::write_new(output / "complete.json", "{\"status\":\"passed\",\"protocol\":" + quote(protocol) +
      ",\"master\":941000,\"profiles\":[\"A\"],\"updates_each\":2,\"CPU_encoder_calls\":0}\n");
}
} // namespace

int main(int argc, char **argv) {
  fs::path output; bool owned = false;
  try {
    if (argc == 2 && std::string(argv[1]) == "--source-id") { std::cout << CONFIGURABLE_CONTENT_RELATION_SOURCE_ID << '\n'; return 0; }
    if (argc == 2 && std::string(argv[1]) == "--card-sha256") { std::cout << CONFIGURABLE_CONTENT_RELATION_CARD_SHA256 << '\n'; return 0; }
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2)
      require(i + 1 < argc && std::string(argv[i]).rfind("--", 0) == 0 && args.emplace(argv[i], argv[i + 1]).second, "unique key/value CLI");
    RuntimeIsolation isolation; torch::set_num_threads(1);
    if (args.count("--engineering")) {
      require(args.size() == 2 && args.count("--config"), "engineering output and model config only");
      output = args.at("--engineering"); engineering(output, args.at("--config"), owned); return 0;
    }
    const std::set<std::string> required{"--data-root", "--inputs-sha256", "--original-capsule", "--output", "--profile", "--master", "--config", "--card", "--card-sha256", "--source-id"};
    require(args.size() == required.size(), "closed quality CLI");
    for (const auto &key : required) require(args.count(key), "required CLI: " + key);
    require(io::is_sha(CONFIGURABLE_CONTENT_RELATION_SOURCE_ID) && io::is_sha(CONFIGURABLE_CONTENT_RELATION_CARD_SHA256) &&
        args.at("--source-id") == CONFIGURABLE_CONTENT_RELATION_SOURCE_ID && args.at("--card-sha256") == CONFIGURABLE_CONTENT_RELATION_CARD_SHA256 &&
        io::sha256(io::bytes(args.at("--card"))) == args.at("--card-sha256"), "compiled source and exact prospective card binding");
    const auto master_text = args.at("--master");
    require(master_text == "941101" || master_text == "941102" || master_text == "941103", "one of three prospective quality masters");
    const auto master = std::stoull(master_text); const auto profile = args.at("--profile");
    const auto c = config(args.at("--config"), profile);
    const fs::path parent = args.at("--data-root");
    require(args.at("--original-capsule") == "/mityane/code/embeddings/output/runs/rpb-v20-capacity/admission-0gN4uD" &&
        parent == fs::path(args.at("--original-capsule")) / "quality-data", "exact original quality role root");
    require(parent.is_absolute() && fs::canonical(parent) == parent && !fs::is_symlink(parent), "direct data root");
    std::map<std::string, fs::path> roles;
    for (const auto m : quality_masters) for (const std::string role : {"training", "validation", "validation-deleted"}) {
      const auto name = "master-" + std::to_string(m) + "/" + role + ".pt"; roles.emplace(name, parent / name);
    }
    io::Guard inputs(args.at("--inputs-sha256"), roles);
    require(io::sha256(io::bytes(args.at("--inputs-sha256"))) == original_inputs,
        "reuse exact original nine quality role bytes without generation");
    std::map<uint64_t, Cohort> all;
    for (const auto m : quality_masters) {
      const auto prefix = "master-" + std::to_string(m) + "/";
      Cohort cohort{cap::load_split_archive(inputs.bind(prefix + "training.pt")), cap::load_split_archive(inputs.bind(prefix + "validation.pt")),
          cap::load_split_archive(inputs.bind(prefix + "validation-deleted.pt"))};
      check_cohort(cohort, m, 512, 256); all.emplace(m, std::move(cohort));
    }
    require(torch::cuda::is_available(), "quality CUDA required after complete data admission");
    output = args.at("--output");
    quality_admission(output, master, args.at("--config"), args.at("--card"), args.at("--inputs-sha256"),
        CONFIGURABLE_CONTENT_RELATION_SOURCE_ID, args.at("--card-sha256"));
    if (master == 941103) third_admission(output, CONFIGURABLE_CONTENT_RELATION_SOURCE_ID, args.at("--card-sha256"));
    new_leaf(output); owned = true;
    Binding binding{profile, tag(profile), CONFIGURABLE_CONTENT_RELATION_SOURCE_ID, args.at("--card-sha256"),
        io::sha256(io::bytes(args.at("--inputs-sha256"))), io::sha256(io::bytes(args.at("--config"))), master};
    const auto &data = all.at(master);
    auto start = Clock::now(); Trainer trainer(c, data.training, binding); double binding_seconds = seconds(start), feature_seconds = 0, query_seconds = 0;
    OriginalControls original(args.at("--original-capsule"));
    const auto control_initial = saved_features(original, master, 0, data, trainer.scaler.identity());
    const auto control_final = saved_features(original, master, 512, data, trainer.scaler.identity());
    exact_fixed_control(control_initial, control_final);
    FeatureViews initial_features;
    NativeMovement learned_movement;
    std::string training_query, validation_query, initial_training_query, initial_validation_query;
    io::write_new(output / "input-manifest.json", inputs.json() + '\n');
    for (const int64_t point : {int64_t{0}, quality_budget}) {
      if (point) {
        trainer.train_to(point);
        require(trainer.changed() && trainer.trace.size() == quality_budget && trainer.relation_gradient_updates > 0 &&
            trainer.relation_weight_change() > 0, "complete CUDA trajectory trains the new relation projection");
      }
      const auto directory = output / ("point-" + std::to_string(point)); require(fs::create_directory(directory), "new retained point");
      start = Clock::now(); trainer.save(directory / "checkpoint.pt"); save_training_state(directory / "optimizer-witness.pt", trainer);
      save_scaler_witness(directory / "scaler-witness.pt", trainer, data.training);
      auto loaded = rpb::load_content_ordered_relation_checkpoint((directory / "checkpoint.pt").string(), c.device, &c);
      exact_reload_state(trainer.model, loaded.model);
      require(loaded.training_updates == point && loaded.scaler.identity() == trainer.scaler.identity(), "served checkpoint metadata/scaler"); binding_seconds += seconds(start);
      start = Clock::now(); const auto features = feature_views(data, loaded);
      if (!point) exact_initial_control(features, control_initial);
      exact_fixed_control(features, control_initial); exact_fixed_control(features, control_final);
      write_features(directory / "features.pt", features, binding, point, c, loaded.scaler.identity(), data); torch::cuda::synchronize(0); feature_seconds += seconds(start);
      if (!point) initial_features = features;
      else learned_movement = unchanged_fixed_features(initial_features, features, c);
      start = Clock::now();
      training_query = write_queries(directory / "reconstruction-training.pt", data.training, loaded, binding);
      validation_query = write_queries(directory / "reconstruction-validation.pt", data.validation, loaded, binding);
      query_seconds += seconds(start);
      if (!point) { initial_training_query = training_query; initial_validation_query = validation_query; }
      io::write_new(directory / "query-report.json", "{" + binding.json() + ",\"point\":" + std::to_string(point) +
          ",\"training_reconstruction\":" + training_query + ",\"validation_reconstruction\":" + validation_query + "}\n");
      io::write_new(directory / "progress.json", trainer.progress() + '\n');
    }
    inputs.verify(); original.verify();
    require(io::sha256(io::bytes(args.at("--inputs-sha256"))) == binding.input_manifest_sha &&
        io::sha256(io::bytes(args.at("--config"))) == binding.config_sha && io::sha256(io::bytes(args.at("--card"))) == binding.card_sha,
        "metadata inputs unchanged");
    io::write_new(output / "progress.json", trainer.progress() + '\n');
    std::ostringstream report; report << std::setprecision(17) << '{' << binding.json() << ",\"progress\":" << trainer.progress()
        << ",\"training_reconstruction\":" << training_query << ",\"validation_reconstruction\":" << validation_query
        << ",\"initial_training_reconstruction\":" << initial_training_query << ",\"initial_validation_reconstruction\":" << initial_validation_query
        << ",\"learned_training_coordinate_movement\":{\"changed_cells\":" << learned_movement.changed_cells
        << ",\"max_absolute_change\":" << learned_movement.max_absolute_change << '}'
        << ",\"initial_native_matches_saved_v20\":true,\"initial_matching_native_cells\":32768,\"checkpoint_state_reload_exact\":true"
        << ",\"fixed_spectral_coordinates_exact_at_0_512\":true,\"fixed_coordinates_match_saved_v20\":true,\"original_control_files\":" << original.json()
        << ",\"costs\":{\"binding_checkpoint_snapshot_seconds\":" << binding_seconds
        << ",\"CUDA_feature_transfer_IO_seconds\":" << feature_seconds << ",\"CUDA_query_transfer_IO_seconds\":" << query_seconds
        << ",\"CPU_head_seconds\":0},\"counts\":{\"encoder_trajectories\":1,\"encoder_updates\":512,\"retained_points\":2,\"native_exports\":6,\"query_writer_calls\":4,\"query_cuda_forward_calls\":3072,\"generator_calls\":0,\"CPU_encoder_calls\":0,\"head_fits\":0},\"testing_accessed\":false,\"stress_accessed\":false,\"promotion\":false}";
    io::write_new(output / "report.json", report.str() + '\n');
    io::write_new(output / "complete.json", "{\"status\":\"complete\"," + binding.json() +
        ",\"updates\":512,\"report_sha256\":" + quote(io::sha256(report.str() + '\n')) + "}\n");
    std::cout << binding.model_tag << " master=" << master << " CUDA_updates=512 complete; native=" << c.export_width << " training_seconds=" << trainer.training_seconds << '\n';
    return 0;
  } catch (const std::exception &error) {
    if (owned && fs::is_directory(output) && !fs::exists(output / "failure.json"))
      try { io::write_new(output / "failure.json", "{\"status\":\"failed\",\"error\":" + quote(error.what()) + "}\n"); } catch (...) {}
    std::cerr << error.what() << '\n'; return 1;
  }
}
