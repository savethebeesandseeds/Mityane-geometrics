// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/context_replication_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/workflow.h"
#include <ATen/Context.h>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
constexpr const char *protocol = "native-development-v1/lag_sign";
constexpr const char *rng_policy = "splitmix64-counter-rows-masks-torch-attempt-v1";
constexpr const char *sampling_policy = "with_replacement_counter_rows;sampled_rows_includes_no_update_attempts";

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[RPB context replication audit] " + message);
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
    try {
      for (size_t i = 0; i < generators.size(); ++i) generators[i].set_state(states[i]);
      at::set_num_threads(threads);
    } catch (...) { std::terminate(); }
  }
};
std::string source_manifest(const std::vector<std::string> &ids) {
  std::string out;
  for (const auto &id : ids) out += std::to_string(id.size()) + ':' + id;
  return out;
}
std::string manifest_id(const std::string &manifest) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : manifest) { hash ^= byte; hash *= 1099511628211ULL; }
  std::ostringstream out; out << "rpb-fit-source-manifest-fnv1a-v1-" << std::hex << std::setw(16)
      << std::setfill('0') << hash; return out.str();
}
std::string text(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true); return embedding::archive::tensor_text(value);
}
int64_t integer(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true);
  require(value.scalar_type() == torch::kInt64 && value.numel() == 1, "invalid typed count: " + name);
  return value.item<int64_t>();
}
double floating(torch::serialize::InputArchive &archive, const std::string &name) {
  torch::Tensor value; archive.read(name, value, true);
  require(value.scalar_type() == torch::kFloat64 && value.numel() == 1 &&
      torch::isfinite(value).all().item<bool>(), "invalid typed scalar: " + name);
  return value.item<double>();
}
void validate_training(const Checkpoint &checkpoint, const ev::ProviderFitInput &fit) {
  const auto &config = checkpoint.settings.model;
  require(config.channel_mixer_placement == 0,
      "historical context replication admission rejects early channel mixer placement");
  require(fit.protocol_id == protocol && config.global_bottleneck_mode == 2 &&
      config.channel_mixer_layers == 1 && config.export_width == 32 &&
      fit.shape.dtype == torch::kFloat64 && fit.shape.device.is_cpu() &&
      fit.shape.channel_count == config.channel_count && fit.shape.history_length == config.history_length &&
      fit.shape.input_width == config.input_width && fit.training_observations.data.defined() &&
      fit.training_observations.data.dim() == 4 && fit.training_observations.data.device().is_cpu() &&
      fit.training_observations.data.scalar_type() == torch::kFloat64 && fit.training_observations.feature_mask.defined() &&
      fit.training_observations.feature_mask.device().is_cpu() &&
      fit.training_source_ids.size() == static_cast<size_t>(fit.training_observations.data.size(0)) &&
      !fit.feature_units.empty(), "fresh declared native-development TRAIN and unchanged mode2/mixer1/native32 required");
  for (const auto &id : fit.training_source_ids) require(!id.empty(), "empty fresh TRAIN source ID");
  Input input{fit.training_observations.data, fit.training_observations.feature_mask,
      torch::tensor(fit.channel_ids, torch::kInt64),
      torch::full({fit.training_observations.data.size(0)}, fit.endpoint, torch::kFloat64), fit.sampling_interval};
  validate_input(input, config);
  const auto described = describe_dataset(input, config, fit.feature_units);
  require(described.dataset_id == checkpoint.dataset_id && described.schema_id == checkpoint.schema_id &&
      checkpoint.scaler_fit_dataset_id == described.dataset_id &&
      checkpoint.settings.seed == static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL),
      "fresh checkpoint TRAIN/schema/scaler-fit/seed differs from its declared cohort");
}
bool same_scaler(const FrozenScaler &a, const FrozenScaler &b) {
  return a.identity() == b.identity() && a.scale_floor == b.scale_floor &&
      torch::equal(a.mean, b.mean) && torch::equal(a.scale, b.scale) && torch::equal(a.count, b.count) &&
      torch::equal(a.channel_ids, b.channel_ids) && torch::equal(a.floor_applied, b.floor_applied);
}
std::map<std::string, std::string> audit_point(const std::string &path, const Checkpoint &checkpoint,
                                            const ev::ProviderFitInput &fit, bool context,
                                            ContextDeletionRecipe expected_recipe) {
  torch::serialize::InputArchive archive; archive.load_from(path + ".audit.pt", torch::kCPU);
  require(text(archive, "artifact_kind") == "rpb_learning_curve_training_audit_v1", "unknown fresh producer audit kind");
  std::map<std::string, std::string> fields;
  for (const std::string name : {"protocol_id", "feature_units", "fit_source_manifest", "fit_source_manifest_id",
      "initialization_seed", "actual_training_seed", "rng_policy", "sampling_policy", "optimizer_policy"})
    fields.emplace(name, text(archive, name));
  auto saved_settings = parse_settings(text(archive, "resolved_settings")); saved_settings.model.device = torch::kCPU;
  const auto manifest = source_manifest(fit.training_source_ids);
  require(settings_text(saved_settings) == settings_text(checkpoint.settings) &&
      fields.at("protocol_id") == protocol && fields.at("feature_units") == fit.feature_units &&
      fields.at("fit_source_manifest") == manifest && fields.at("fit_source_manifest_id") == manifest_id(manifest) &&
      fields.at("initialization_seed") == std::to_string(training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL)) &&
      fields.at("actual_training_seed") == std::to_string(checkpoint.settings.seed) &&
      fields.at("rng_policy") == rng_policy && fields.at("sampling_policy") == sampling_policy &&
      fields.at("optimizer_policy") == "one_continuous_AdamW_state;absolute_completed_update_budgets" &&
      torch::Device(text(archive, "training_device")).is_cuda() &&
      text(archive, "training_producer_source_fingerprint").size() > 0 &&
      text(archive, "core_writer_source_fingerprint") == checkpoint.source_fingerprint &&
      text(archive, "training_dataset_id") == checkpoint.dataset_id &&
      text(archive, "scaler_fit_dataset_id") == checkpoint.scaler_fit_dataset_id &&
      text(archive, "preprocessing_id") == checkpoint.scaler.identity() &&
      text(archive, "training_observation_rows") == std::to_string(fit.training_source_ids.size()) &&
      text(archive, "training_source_groups") == std::to_string(std::set<std::string>(
          fit.training_source_ids.begin(), fit.training_source_ids.end()).size()),
      "fresh producer settings/source/initialization/counter/scaler association differs");
  torch::Tensor ids; archive.read("channel_order", ids, true);
  require(ids.scalar_type() == torch::kInt64 && ids.dim() == 1 &&
      torch::equal(ids, torch::tensor(fit.channel_ids, torch::kInt64)) &&
      floating(archive, "sampling_interval") == fit.sampling_interval && floating(archive, "endpoint") == fit.endpoint &&
      integer(archive, "attempted_steps") == checkpoint.attempted_steps &&
      integer(archive, "completed_steps") == checkpoint.completed_steps &&
      checkpoint.attempted_steps <= std::numeric_limits<int64_t>::max() / checkpoint.settings.batch_size &&
      integer(archive, "sampled_rows") == checkpoint.attempted_steps * checkpoint.settings.batch_size,
      "fresh physical channel/time metadata or typed attempted/completed/sample counters differ");
  if (checkpoint.completed_steps > 0) {
    torch::Tensor finite, changed; archive.read("finite_gradients", finite, true); archive.read("weights_changed", changed, true);
    require(finite.scalar_type() == torch::kBool && finite.numel() == 1 && finite.item<bool>() &&
        changed.scalar_type() == torch::kBool && changed.numel() == 1 && changed.item<bool>(),
        "fresh positive reference lacks finite-gradient/changed-weight evidence");
  }
  if (context) {
    const auto &selected = context_deletion::descriptor(expected_recipe);
    require(text(archive, "model_tag") == selected.model_tag && text(archive, "training_policy_id") == selected.policy_id &&
        text(archive, "context_deletion_ratio") == selected.ratio_text &&
        text(archive, "context_deletion_stream") == "0x6374782d64726f70" &&
        text(archive, "context_deletion_rng_policy") == context_deletion::rng_policy &&
        text(archive, "context_deletion_repair_policy") == context_deletion::repair_policy &&
        text(archive, "context_deletion_visibility_policy") == context_deletion::visibility_policy &&
        text(archive, "context_deletion_count_policy") ==
            "cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only" &&
        text(archive, "context_deletion_resume_policy") ==
            "fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API" &&
        floating(archive, "context_deletion_ratio_value") == selected.ratio &&
        integer(archive, "context_deletion_stream_value") == static_cast<int64_t>(context_deletion::stream),
        "fresh candidate companion differs from the fixed context policy");
    for (const std::string name : {"context_requested_deleted_coordinates", "context_actual_deleted_coordinates",
        "context_restored_coordinates"}) require(integer(archive, name) == 0, "candidate point zero has consumed context deletion");
    if (context_deletion::is_balanced(expected_recipe)) {
      require(text(archive, "context_deletion_schedule_policy") == context_deletion::balanced_schedule_policy &&
          text(archive, "context_deletion_rate_scope") == context_deletion::balanced_rate_scope &&
          text(archive, "context_deletion_branch_count_policy") == context_deletion::balanced_branch_count_policy &&
          text(archive, "context_deletion_skip_policy") == context_deletion::balanced_skip_policy &&
          text(archive, "context_ordinary_attempts") == "0" && text(archive, "context_deletion_attempts") == "0" &&
          integer(archive, "context_ordinary_attempts_value") == 0 &&
          integer(archive, "context_deletion_attempts_value") == 0,
          "fresh balanced candidate schedule/rate scope/branch-count companion differs");
    } else {
      for (const std::string name : {"context_deletion_schedule_policy", "context_deletion_rate_scope",
          "context_deletion_branch_count_policy", "context_deletion_skip_policy", "context_ordinary_attempts",
          "context_deletion_attempts", "context_ordinary_attempts_value", "context_deletion_attempts_value"}) {
        torch::Tensor value;
        require(!archive.try_read(name, value, true), "uniform context candidate contains a balanced schedule: " + name);
      }
    }
  } else {
    for (const std::string name : {"training_policy_id", "context_deletion_ratio", "context_deletion_stream",
        "context_requested_deleted_coordinates", "context_actual_deleted_coordinates", "context_restored_coordinates",
        "context_deletion_schedule_policy", "context_deletion_rate_scope", "context_deletion_branch_count_policy",
        "context_deletion_skip_policy", "context_ordinary_attempts", "context_deletion_attempts",
        "context_ordinary_attempts_value", "context_deletion_attempts_value"}) {
      torch::Tensor value;
      require(!archive.try_read(name, value, true), "ordinary fresh reference contains a training context policy: " + name);
    }
  }
  return fields;
}
} // namespace

std::map<std::string, std::string> audit_context_replication_initialization(const std::string &candidate_path,
    const ev::RetainedPoolingCohort &reference, const ev::ProviderFitInput &fit, int64_t expected_updates) {
  return audit_context_replication_initialization(candidate_path, reference, fit,
      ContextDeletionRecipe::coordinate30_v1, expected_updates);
}

std::map<std::string, std::string> audit_context_replication_initialization(const std::string &candidate_path,
    const ev::RetainedPoolingCohort &reference, const ev::ProviderFitInput &fit,
    ContextDeletionRecipe expected_recipe, int64_t expected_updates) {
  const RuntimeIsolation isolation;
  const auto &selected = context_deletion::descriptor(expected_recipe);
  require(expected_updates > 0 && reference.master_seed == fit.seed, "fresh reference master or declared positive budget differs");
  const auto candidate = load_checkpoint(candidate_path, torch::kCPU);
  const auto initial = load_checkpoint(reference.reference_initial_checkpoint, torch::kCPU);
  const auto positive = load_checkpoint(reference.reference_checkpoint, torch::kCPU);
  for (const auto *checkpoint : {&candidate, &initial, &positive}) validate_training(*checkpoint, fit);
  require(candidate.training_policy_id == selected.policy_id && initial.training_policy_id.empty() &&
      positive.training_policy_id.empty(), "fresh candidate/v4 checkpoint policy tags differ");
  require(candidate.attempted_steps == 0 && candidate.completed_steps == 0 &&
      initial.attempted_steps == 0 && initial.completed_steps == 0 &&
      positive.attempted_steps == expected_updates && positive.completed_steps == expected_updates,
      "fresh point-zero or equal-budget unskipped reference counters differ");
  require(settings_text(candidate.settings) == settings_text(initial.settings) &&
      settings_text(initial.settings) == settings_text(positive.settings), "fresh v4/v6 model or training settings differ");
  require(same_scaler(candidate.scaler, initial.scaler) && same_scaler(initial.scaler, positive.scaler),
      "fresh v4/v6 complete frozen TRAIN scalers differ");
  const auto a = initial.model->named_parameters(), b = candidate.model->named_parameters(), p = positive.model->named_parameters();
  require(a.size() == b.size() && a.size() == p.size(), "fresh architecture changed its parameter set");
  int64_t count = 0; bool positive_changed = false;
  for (const auto &parameter : a) {
    require(b.contains(parameter.key()) && p.contains(parameter.key()) &&
        parameter.value().scalar_type() == b[parameter.key()].scalar_type() &&
        parameter.value().scalar_type() == p[parameter.key()].scalar_type() &&
        parameter.value().sizes() == p[parameter.key()].sizes() && torch::equal(parameter.value(), b[parameter.key()]),
        "fresh initialized parameter differs: " + parameter.key());
    count += parameter.value().numel();
    positive_changed |= !torch::equal(parameter.value(), p[parameter.key()]);
  }
  require(count == 225805 && positive_changed, "fresh reference is not the complete trained 225805-parameter mode2 model");
  const auto x = initial.model->named_buffers(), y = candidate.model->named_buffers(), z = positive.model->named_buffers();
  require(x.size() == y.size() && x.size() == z.size(), "fresh architecture changed its buffer set");
  for (const auto &buffer : x)
    require(y.contains(buffer.key()) && z.contains(buffer.key()) &&
        buffer.value().scalar_type() == y[buffer.key()].scalar_type() &&
        buffer.value().scalar_type() == z[buffer.key()].scalar_type() &&
        torch::equal(buffer.value(), y[buffer.key()]) && torch::equal(buffer.value(), z[buffer.key()]),
        "fresh initialized or fixed reference buffer differs: " + buffer.key());
  const auto candidate_fields = audit_point(candidate_path, candidate, fit, true, expected_recipe);
  const auto initial_fields = audit_point(reference.reference_initial_checkpoint, initial, fit, false, expected_recipe);
  const auto positive_fields = audit_point(reference.reference_checkpoint, positive, fit, false, expected_recipe);
  require(candidate_fields == initial_fields && initial_fields == positive_fields,
      "fresh paired source order or original initialization/row/mask/Torch streams differ");
  std::map<std::string, std::string> result{{"common_parameters_exact", "true"}, {"all_parameters_exact_including_global_pool", "true"},
      {"common_parameter_count", std::to_string(count)}, {"common_tensors", std::to_string(a.size())},
      {"all_buffers_exact", "true"}, {"scaler_exact", "true"}, {"training_dataset_exact", "true"},
      {"counter_streams_exact", "true"}, {"training_policy_id", selected.policy_id},
      {"training_policy_companion_exact", "true"}, {"context_deletion_stream", std::to_string(context_deletion::stream)},
      {"training_dataset_id", candidate.dataset_id}, {"schema_id", candidate.schema_id},
      {"preprocessing_id", candidate.scaler.identity()}, {"initialization_seed", candidate_fields.at("initialization_seed")},
      {"training_seed", candidate_fields.at("actual_training_seed")}, {"training_protocol_id", protocol},
      {"reference_completed_updates", std::to_string(expected_updates)},
      {"reference_role", "fresh_equal_budget_training_reference"},
      {"input_byte_integrity_owner", "shared_engine_explicit_input_hash_guard"},
      {"scope", "fresh complete same-mode2 point0 weights/buffers/scaler/TRAIN/source order/original counter streams; context-only policy; fresh reference has no skips"}};
  if (context_deletion::is_balanced(expected_recipe)) {
    result.emplace("context_deletion_schedule_policy", context_deletion::balanced_schedule_policy);
    result.emplace("context_deletion_rate_scope", context_deletion::balanced_rate_scope);
    result.emplace("context_deletion_branch_count_policy", context_deletion::balanced_branch_count_policy);
    result.emplace("context_deletion_skip_policy", context_deletion::balanced_skip_policy);
    result.emplace("context_ordinary_attempts", "0");
    result.emplace("context_deletion_attempts", "0");
  }
  return result;
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
