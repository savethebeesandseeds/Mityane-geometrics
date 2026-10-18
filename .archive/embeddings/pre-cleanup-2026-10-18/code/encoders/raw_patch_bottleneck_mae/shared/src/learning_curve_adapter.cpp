// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/learning_curve_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/evaluation_adapter.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/masking.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_utils.h"
#include "embedding/shared/data.h"
#include <torch/cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef EVALUATION_SOURCE_ID
#define EVALUATION_SOURCE_ID "unrecorded"
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {
namespace {
namespace ev = embedding::evaluation;
namespace fs = std::filesystem;

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error("[rpb learning curve adapter] " + message);
}

Input protocol_input(const embedding::Batch &batch, const Config &config,
                     const ev::ProviderFitInput &fit) {
  Input out{batch.data, batch.feature_mask, torch::tensor(fit.channel_ids, torch::kInt64),
      torch::full({batch.data.size(0)}, fit.endpoint, torch::kFloat64), fit.sampling_interval};
  validate_input(out, config);
  return out;
}

std::string source_manifest(const std::vector<std::string> &ids) {
  std::string text;
  for (const auto &id : ids) text += std::to_string(id.size()) + ":" + id;
  return text;
}

std::string manifest_identity(const std::string &text) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ULL; }
  std::ostringstream out;
  out << "rpb-fit-source-manifest-fnv1a-v1-" << std::hex << std::setw(16)
      << std::setfill('0') << hash;
  return out.str();
}

std::string path_key(const std::string &path) {
  require(!path.empty(), "checkpoint path must be nonempty");
  return fs::weakly_canonical(fs::absolute(path)).string();
}

struct CurveState {
  Settings settings;
  ContextDeletionOptions context_options;
  ev::ProviderFitInput fit;
  Dataset training;
  FrozenScaler scaler;
  Model model{nullptr};
  std::unique_ptr<torch::optim::AdamW> optimizer;
  std::vector<torch::Tensor> initial_parameters;
  ev::CurveProgress progress;
  ev::CurveLossPoint last_loss;
  std::map<std::string, std::pair<int64_t, int64_t>> saved;
  std::array<int64_t, 3> context_counts{0, 0, 0}; // Requested, actual, restored.
  std::map<std::string, std::array<int64_t, 3>> saved_context_counts;
  std::array<int64_t, 2> branch_counts{0, 0}; // Balanced ordinary, deletion attempts.
  std::map<std::string, std::array<int64_t, 2>> saved_branch_counts;
  bool balanced_failed{false};
  bool continuation_witness{false};
  PooledContextInitializationOptions pooled_options;
  VisibleDifferenceOptions difference_options;
  std::map<std::string, torch::Tensor> difference_initial_parameters, difference_initial_buffers;
  std::map<std::string, std::string> difference_parent_content;
  torch::Tensor pooled_first_before_copy;
  std::map<std::string, torch::Tensor> pooled_initial_parameters, pooled_initial_buffers;
  std::map<std::string, std::string> pooled_reference_content;
  TrainingSourceGainOptions gain_options;
  training_source_gain::Manifest gain_manifest;
  std::vector<int64_t> gain_attempts;
  std::vector<torch::Tensor> gain_indices, gain_targets;
  std::map<std::string, std::string> audit;
  uint64_t initialization_seed{0};
};

ev::CurveProgress progress_report(const std::shared_ptr<CurveState> &state) {
  // Queried milestone endpoints are permanent entries, so subsequent reports
  // retain every previously delivered loss point as an exact positional prefix.
  if (state->progress.completed > 0 && (state->progress.losses.empty() ||
      state->progress.losses.back().completed != state->last_loss.completed))
    state->progress.losses.push_back(state->last_loss);
  auto out = state->progress;
  const auto parameters = state->model->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    require(torch::isfinite(parameters[i]).all().item<bool>(), "nonfinite CUDA model parameters");
    if (!torch::equal(parameters[i].detach(), state->initial_parameters[i]))
      out.weights_changed = true;
  }
  return out;
}

void write_text(torch::serialize::OutputArchive &archive, const std::string &key,
                const std::string &text) {
  archive.write(key, embedding::archive::text_tensor(text), true);
}

torch::Tensor cpu_state(const torch::Tensor &value) {
  return value.detach().to(torch::kCPU).contiguous().clone();
}

void write_named_state(torch::serialize::OutputArchive &archive, const std::string &key,
                       const std::map<std::string, torch::Tensor> &values) {
  torch::serialize::OutputArchive group;
  group.write("count", torch::tensor(static_cast<int64_t>(values.size())), true);
  int64_t index = 0;
  for (const auto &[name, value] : values) {
    torch::serialize::OutputArchive entry;
    write_text(entry, "parameter_name", name);
    entry.write("value", cpu_state(value), true);
    group.write("tensor_" + std::to_string(index++), entry);
  }
  archive.write(key, group);
}


std::string pooled_file_id(const std::string &path) {
  require(fs::is_regular_file(path) && !fs::is_symlink(fs::symlink_status(path)) &&
      fs::hard_link_count(path) == 1, "regular non-aliased point0 reference required");
  std::ifstream in(path, std::ios::binary); require(bool(in), "cannot read point0 reference");
  uint64_t hash = 14695981039346656037ULL; std::array<char,65536> bytes{};
  while (in) { in.read(bytes.data(), bytes.size());
    for (std::streamsize i=0;i<in.gcount();++i) { hash ^= static_cast<unsigned char>(bytes[i]); hash *= 1099511628211ULL; }
  }
  require(in.eof(), "point0 reference read failed");
  std::ostringstream out; out << "fnv1a64-runtime-content-v1-" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}
std::string pooled_text(torch::serialize::InputArchive &a,const std::string &key) {
  torch::Tensor v; a.read(key,v,true); return embedding::archive::tensor_text(v);
}
int64_t pooled_int(torch::serialize::InputArchive &a,const std::string &key) {
  torch::Tensor v; a.read(key,v,true); require(v.device().is_cpu() && v.scalar_type()==torch::kInt64 && v.dim()==0,
      "typed point0 scalar required: "+key); return v.item<int64_t>();
}
void prepare_pooled_initialization(const std::shared_ptr<CurveState> &state) {
  if (!state->pooled_options.enabled) return;
  const auto &config=state->settings.model;
  if (config.global_pool_input_source == 1) {
    state->pooled_first_before_copy=cpu_state(state->model->named_parameters()["global_pool_first.weight"]);
    const auto path=path_key(state->pooled_options.control_point0_checkpoint_path);
    for (const auto &suffix : {std::string(),std::string(".audit.pt"),std::string(".scaler.pt"),
        std::string(".training-raw.pt"),std::string(kLearningCurveContinuationSuffix)})
      state->pooled_reference_content.emplace(suffix,pooled_file_id(path+suffix));
    auto cp=load_checkpoint(path,config.device);
    auto expected=state->settings; expected.model.global_pool_input_source=0;
    require(cp.attempted_steps==0 && cp.completed_steps==0 && settings_text(cp.settings)==settings_text(expected) &&
        cp.settings.model.global_pool_input_source==0 && cp.training_policy_id=="rpb-training-context-deletion-015-v1" &&
        cp.source_fingerprint==workflow_source_fingerprint() && cp.dataset_id==state->training.dataset_id &&
        cp.schema_id==state->training.schema_id && cp.scaler_fit_dataset_id==cp.dataset_id &&
        cp.scaler.identity()==state->scaler.identity(), "compact reference point0 config/data/scaler/source differs");
    const auto raw=load_dataset(path+".training-raw.pt",cp.settings.model);
    const auto scaler=load_scaler(path+".scaler.pt",cp.settings.model,cp.schema_id);
    require(raw.dataset_id==cp.dataset_id && raw.schema_id==cp.schema_id && raw.feature_units==state->fit.feature_units &&
        torch::equal(raw.input.data,state->training.input.data) &&
        torch::equal(raw.input.observed,state->training.input.observed) &&
        torch::equal(raw.input.channel_ids,state->training.input.channel_ids) &&
        torch::equal(raw.input.endpoints,state->training.input.endpoints) &&
        scaler.identity()==state->scaler.identity(), "compact reference exact original TRAIN/scaler differs");
    torch::serialize::InputArchive audit; audit.load_from(path+".audit.pt",torch::kCPU);
    require(pooled_text(audit,"artifact_kind")=="rpb_learning_curve_training_audit_v1" &&
        pooled_text(audit,"protocol_id")==state->fit.protocol_id && pooled_text(audit,"model_tag")=="RPB-v10" &&
        pooled_text(audit,"architecture_id")==architecture_id(cp.settings.model) &&
        pooled_text(audit,"fit_source_manifest")==source_manifest(state->fit.training_source_ids) &&
        pooled_text(audit,"actual_training_seed")==std::to_string(state->settings.seed) &&
        pooled_text(audit,"initialization_seed")==std::to_string(state->initialization_seed) &&
        pooled_text(audit,"training_producer_source_fingerprint")==EVALUATION_SOURCE_ID &&
        pooled_text(audit,"core_writer_source_fingerprint")==workflow_source_fingerprint() &&
        pooled_int(audit,"attempted_steps")==0 && pooled_int(audit,"completed_steps")==0 && pooled_int(audit,"sampled_rows")==0 &&
        pooled_text(audit,"pooled_initialization_policy")=="compact-control-independent-initialization;no-copy",
        "compact reference point0 source/order/seed/counter/policy differs");
    auto resolved=parse_settings(pooled_text(audit,"resolved_settings")); resolved.model.device=config.device;
    require(settings_text(resolved)==settings_text(expected) &&
        pooled_text(audit,"training_policy_id")=="rpb-training-context-deletion-015-v1" &&
        pooled_text(audit,"context_deletion_ratio")=="0.15" &&
        pooled_text(audit,"context_deletion_rng_policy")==context_deletion::rng_policy &&
        pooled_text(audit,"context_deletion_repair_policy")==context_deletion::repair_policy &&
        pooled_text(audit,"context_deletion_visibility_policy")==context_deletion::visibility_policy &&
        pooled_int(audit,"global_pool_input_source_value")==0 && pooled_int(audit,"copied_parameter_values")==0 &&
        pooled_int(audit,"context_deletion_stream_value")==static_cast<int64_t>(context_deletion::stream) &&
        pooled_int(audit,"context_requested_deleted_coordinates")==0 && pooled_int(audit,"context_actual_deleted_coordinates")==0 &&
        pooled_int(audit,"context_restored_coordinates")==0,"compact point0 typed settings/context metadata differs");
    torch::Tensor seconds,changed,finite,ratio;
    audit.read("training_seconds",seconds,true); audit.read("weights_changed",changed,true); audit.read("finite_gradients",finite,true);
    audit.read("context_deletion_ratio_value",ratio,true);
    require(seconds.device().is_cpu() && seconds.scalar_type()==torch::kFloat64 && seconds.dim()==0 && seconds.item<double>()==0 &&
        ratio.device().is_cpu() && ratio.scalar_type()==torch::kFloat64 && ratio.dim()==0 && ratio.item<double>()==.15 &&
        changed.device().is_cpu() && changed.scalar_type()==torch::kBool && changed.dim()==0 && !changed.item<bool>() &&
        finite.device().is_cpu() && finite.scalar_type()==torch::kBool && finite.dim()==0 && !finite.item<bool>(),
        "reference point0 must have no update/time/gradient witnesses");
    torch::serialize::InputArchive continuation; continuation.load_from(path+kLearningCurveContinuationSuffix,torch::kCPU);
    torch::serialize::InputArchive opt; continuation.read("optimizer_state",opt);
    require(pooled_text(continuation,"artifact_kind")==kPooledContextContinuationArtifact &&
        pooled_text(continuation,"protocol_id")==state->fit.protocol_id && pooled_int(continuation,"attempted_steps")==0 &&
        pooled_int(continuation,"completed_steps")==0 && pooled_int(opt,"active_state_count")==0,
        "compact reference must have untouched point0 AdamW witness");
    torch::serialize::InputArchive scaler_state; continuation.read("scaler",scaler_state);
    require(FrozenScaler::load(scaler_state).identity()==state->scaler.identity() &&
        pooled_int(continuation,"global_pool_input_source_value")==0 && pooled_int(continuation,"sampled_rows")==0,
        "reference continuation scaler/source/sample count differs");
    torch::serialize::InputArchive named; continuation.read("model_parameters",named);
    const auto saved_parameters=cp.model->named_parameters();
    require(pooled_int(named,"count")==static_cast<int64_t>(saved_parameters.size()),"point0 named parameter count differs");
    for (int64_t i=0;i<pooled_int(named,"count");++i) {
      torch::serialize::InputArchive entry; named.read("tensor_"+std::to_string(i),entry);
      const auto name=pooled_text(entry,"parameter_name"); torch::Tensor value; entry.read("value",value,true);
      require(saved_parameters.contains(name) && value.device().is_cpu() && value.scalar_type()==saved_parameters[name].scalar_type() &&
          value.sizes()==saved_parameters[name].sizes() && torch::equal(value,cpu_state(saved_parameters[name])),
          "reference checkpoint/model continuation witness differs");
    }
    torch::serialize::InputArchive named_buffers; continuation.read("model_buffers",named_buffers);
    const auto saved_buffers=cp.model->named_buffers();
    require(pooled_int(named_buffers,"count")==static_cast<int64_t>(saved_buffers.size()),"point0 buffer count differs");
    for (int64_t i=0;i<pooled_int(named_buffers,"count");++i) {
      torch::serialize::InputArchive entry; named_buffers.read("tensor_"+std::to_string(i),entry);
      const auto name=pooled_text(entry,"parameter_name"); torch::Tensor value; entry.read("value",value,true);
      require(saved_buffers.contains(name) && value.device().is_cpu() && value.scalar_type()==saved_buffers[name].scalar_type() &&
          value.sizes()==saved_buffers[name].sizes() && torch::equal(value,cpu_state(saved_buffers[name])),
          "reference checkpoint/buffer continuation witness differs");
    }
    const auto reference=cp.model->named_parameters(); const auto candidate=state->model->named_parameters();
    require(reference.size()==candidate.size(), "paired parameter name count differs");
    int64_t copied=0,nonshared=0;
    torch::NoGradGuard guard;
    for (const auto &entry:candidate) {
      require(reference.contains(entry.key()), "missing reference parameter name");
      const auto &value=reference[entry.key()];
      require(value.is_cuda() && value.scalar_type()==entry.value().scalar_type() &&
          torch::isfinite(value).all().item<bool>(), "finite CUDA reference parameter required");
      if (entry.key()=="global_pool_first.weight") {
        require(value.sizes()==torch::IntArrayRef({64,99}) && entry.value().sizes()==torch::IntArrayRef({64,195}),
            "only first global weight may change shape"); ++nonshared; continue;
      }
      require(value.sizes()==entry.value().sizes(), "shared parameter shape differs");
      auto target=entry.value(); target.copy_(value); require(torch::equal(target,value), "shared parameter copy failed"); copied+=value.numel();
    }
    require(copied==219469 && nonshared==1,"exact219469 shared values and one wider first weight required");
    const auto reference_buffers=cp.model->named_buffers(); const auto candidate_buffers=state->model->named_buffers();
    require(reference_buffers.size()==candidate_buffers.size(),"paired buffer names differ");
    for (const auto &entry:candidate_buffers) {
      require(reference_buffers.contains(entry.key()) && reference_buffers[entry.key()].sizes()==entry.value().sizes() &&
          reference_buffers[entry.key()].scalar_type()==entry.value().scalar_type(),"paired buffer shape/dtype differs");
      auto target=entry.value(); target.copy_(reference_buffers[entry.key()]);
    }
    for (const auto &[suffix,id]:state->pooled_reference_content)
      require(pooled_file_id(path+suffix)==id,"point0 reference bytes changed during initialization");
  }
  for (const auto &entry:state->model->named_parameters()) state->pooled_initial_parameters.emplace(entry.key(),cpu_state(entry.value()));
  for (const auto &entry:state->model->named_buffers()) state->pooled_initial_buffers.emplace(entry.key(),cpu_state(entry.value()));
  if (config.global_pool_input_source==1)
    require(torch::equal(state->pooled_first_before_copy,state->pooled_initial_parameters.at("global_pool_first.weight")),
        "wider first global weight must retain its own initialization");
}
void verify_pooled_inactive(const std::shared_ptr<CurveState> &state) {
  if (!state->pooled_options.enabled || state->settings.model.global_pool_input_source==0) return;
  const auto p=state->model->named_parameters();
  for (const auto &name : {std::string("export_projection.weight"),std::string("export_projection.bias")}) {
    require(torch::equal(cpu_state(p[name]),state->pooled_initial_parameters.at(name)) &&
        state->optimizer->state().count(p[name].unsafeGetTensorImpl())==0,
        "diagnostic-only projection changed or acquired AdamW state");
  }
}

void prepare_visible_difference_initialization(const std::shared_ptr<CurveState> &state) {
  if (!state->difference_options.enabled) return;
  const auto &option=state->difference_options;
  const auto path=path_key(option.parent_point0_checkpoint_path);
  for (const auto &suffix : {std::string(),std::string(".audit.pt"),std::string(".scaler.pt"),std::string(".training-raw.pt")})
    require(fs::is_regular_file(path+suffix) && fs::hard_link_count(path+suffix)==1 &&
        fs::canonical(path+suffix)==fs::absolute(path+suffix).lexically_normal(),"complete direct original point0 roles required");
  for (const auto &suffix : {std::string(),std::string(".audit.pt"),std::string(".scaler.pt"),std::string(".training-raw.pt")}) {
    require(fs::canonical(path+suffix)==fs::absolute(path+suffix).lexically_normal(),"direct canonical original point0 role required");
    state->difference_parent_content.emplace(suffix,pooled_file_id(path+suffix));
  }
  auto expected=state->settings; expected.model.temporal_difference_input=0;
  torch::serialize::InputArchive audit; audit.load_from(path+".audit.pt",torch::kCPU);
  auto declared=parse_settings(pooled_text(audit,"resolved_settings")); declared.model.device=expected.model.device;
  require(settings_text(declared)==settings_text(expected) && pooled_text(audit,"protocol_id")==option.expected_parent_fit_protocol &&
      pooled_text(audit,"model_tag")=="RPB-v10" &&
      pooled_text(audit,"core_writer_source_fingerprint")==option.expected_parent_core_source_fingerprint &&
      pooled_text(audit,"training_producer_source_fingerprint")==option.expected_parent_training_producer_source_fingerprint &&
      pooled_text(audit,"fit_source_manifest")==source_manifest(state->fit.training_source_ids) &&
      pooled_text(audit,"preprocessing_id")==state->scaler.identity() && pooled_text(audit,"training_dataset_id")==state->training.dataset_id &&
      pooled_int(audit,"attempted_steps")==0 && pooled_int(audit,"completed_steps")==0 && pooled_int(audit,"sampled_rows")==0,
      "original point0 source/settings/TRAIN association differs before weights");
  const auto raw=load_dataset(path+".training-raw.pt",expected.model);
  const auto scaler=load_scaler(path+".scaler.pt",expected.model,state->training.schema_id);
  require(raw.dataset_id==state->training.dataset_id && raw.schema_id==state->training.schema_id && raw.feature_units==state->fit.feature_units &&
      torch::equal(raw.input.data,state->training.input.data) && torch::equal(raw.input.observed,state->training.input.observed) &&
      torch::equal(raw.input.channel_ids,state->training.input.channel_ids) && torch::equal(raw.input.endpoints,state->training.input.endpoints) &&
      scaler.identity()==state->scaler.identity(),"original point0 raw/scaler binding differs before weights");
  auto cp=load_checkpoint(path,state->settings.model.device);
  expected.model.device=cp.settings.model.device;
  require(settings_text(cp.settings)==settings_text(expected) && cp.attempted_steps==0 && cp.completed_steps==0 &&
      cp.training_policy_id=="rpb-training-context-deletion-015-v1" &&
      cp.source_fingerprint==option.expected_parent_core_source_fingerprint &&
      cp.dataset_id==state->training.dataset_id && cp.schema_id==state->training.schema_id &&
      cp.scaler_fit_dataset_id==cp.dataset_id && cp.scaler.identity()==state->scaler.identity(),
      "original early point0 settings/counters/data/scaler/writer differs");
  require(raw.dataset_id==cp.dataset_id && raw.schema_id==cp.schema_id && raw.feature_units==state->fit.feature_units &&
      torch::equal(raw.input.data,state->training.input.data) && torch::equal(raw.input.observed,state->training.input.observed) &&
      torch::equal(raw.input.channel_ids,state->training.input.channel_ids) && torch::equal(raw.input.endpoints,state->training.input.endpoints) &&
      scaler.identity()==state->scaler.identity(),"original point0 exact TRAIN/scaler differs");
  require(pooled_text(audit,"artifact_kind")=="rpb_learning_curve_training_audit_v1" &&
      pooled_text(audit,"protocol_id")==option.expected_parent_fit_protocol && pooled_text(audit,"model_tag")=="RPB-v10" &&
      pooled_text(audit,"architecture_id")==architecture_id(cp.settings.model) &&
      pooled_text(audit,"fit_source_manifest")==source_manifest(state->fit.training_source_ids) &&
      pooled_text(audit,"preprocessing_id")==state->scaler.identity() &&
      pooled_text(audit,"training_dataset_id")==state->training.dataset_id &&
      pooled_text(audit,"actual_training_seed")==std::to_string(state->settings.seed) &&
      pooled_text(audit,"initialization_seed")==std::to_string(state->initialization_seed) &&
      pooled_text(audit,"training_producer_source_fingerprint")==option.expected_parent_training_producer_source_fingerprint &&
      pooled_text(audit,"core_writer_source_fingerprint")==option.expected_parent_core_source_fingerprint &&
      pooled_text(audit,"training_policy_id")=="rpb-training-context-deletion-015-v1" &&
      pooled_text(audit,"context_deletion_ratio")=="0.15" &&
      pooled_text(audit,"context_deletion_rng_policy")==context_deletion::rng_policy &&
      pooled_text(audit,"context_deletion_repair_policy")==context_deletion::repair_policy &&
      pooled_text(audit,"context_deletion_visibility_policy")==context_deletion::visibility_policy &&
      pooled_int(audit,"attempted_steps")==0 && pooled_int(audit,"completed_steps")==0 && pooled_int(audit,"sampled_rows")==0 &&
      pooled_int(audit,"channel_mixer_placement_value")==1 &&
      pooled_int(audit,"context_deletion_stream_value")==static_cast<int64_t>(context_deletion::stream) &&
      pooled_int(audit,"context_requested_deleted_coordinates")==0 && pooled_int(audit,"context_actual_deleted_coordinates")==0 &&
      pooled_int(audit,"context_restored_coordinates")==0,"original point0 audit/source/order/policy differs");
  auto resolved=parse_settings(pooled_text(audit,"resolved_settings")); resolved.model.device=cp.settings.model.device;
  require(settings_text(resolved)==settings_text(cp.settings),"original point0 resolved audit settings differ");
  for (const auto &key : {"training_seconds","context_deletion_ratio_value"}) {
    torch::Tensor v; audit.read(key,v,true);
    require(v.device().is_cpu() && v.scalar_type()==torch::kFloat64 && v.dim()==0 &&
        v.item<double>()==(std::string(key)=="training_seconds" ? 0. : .15),"typed original point0 timing/ratio differs");
  }
  for (const auto &key : {"weights_changed","finite_gradients"}) {
    torch::Tensor v; audit.read(key,v,true);
    require(v.device().is_cpu() && v.scalar_type()==torch::kBool && v.dim()==0 && !v.item<bool>(),"point0 must be untouched");
  }
  const auto reference=cp.model->named_parameters(), candidate=state->model->named_parameters();
  require(candidate.size()==reference.size()+1 && candidate.contains("visible_difference_projection.weight"),"exact one added branch parameter required");
  for (size_t i=0;i<reference.size();++i)
    require(reference[i].key()==candidate[i].key(),"original common parameter registration prefix differs");
  require(candidate[reference.size()].key()=="visible_difference_projection.weight","zero difference child must be registered last");
  int64_t matched=0;
  for (const auto &entry:reference) {
    require(candidate.contains(entry.key()) && candidate[entry.key()].sizes()==entry.value().sizes() &&
        candidate[entry.key()].scalar_type()==entry.value().scalar_type() && entry.value().is_cuda() &&
        torch::isfinite(entry.value()).all().item<bool>(),"finite matching original common parameter required");
    require(torch::equal(candidate[entry.key()],entry.value()),"independent common initialization differs from original point0");
    matched+=entry.value().numel();
  }
  const auto branch=candidate["visible_difference_projection.weight"];
  require(matched==225805 && branch.sizes()==torch::IntArrayRef({64,48}) && branch.scalar_type()==torch::kFloat32 &&
      torch::equal(branch,torch::zeros_like(branch)),"225805 exact common values and zero3072 branch required");
  const auto rb=cp.model->named_buffers(), cb=state->model->named_buffers();
  require(rb.size()==cb.size(),"original point0 buffer count differs");
  for (size_t i=0;i<rb.size();++i) require(rb[i].key()==cb[i].key(),"original buffer registration order differs");
  for (const auto &entry:rb) {
    require(cb.contains(entry.key()) && cb[entry.key()].sizes()==entry.value().sizes() && cb[entry.key()].scalar_type()==entry.value().scalar_type(),
        "original buffer geometry differs");
    require(torch::equal(cb[entry.key()],entry.value()),"independent buffer initialization differs from original point0");
  }
  for (const auto &entry:candidate) state->difference_initial_parameters.emplace(entry.key(),cpu_state(entry.value()));
  for (const auto &entry:cb) state->difference_initial_buffers.emplace(entry.key(),cpu_state(entry.value()));
  for (const auto &[suffix,id]:state->difference_parent_content)
    require(pooled_file_id(path+suffix)==id,"original point0 bytes changed during initialization");
}

void save_continuation_state(const std::shared_ptr<CurveState> &state, const std::string &path) {
  require(state->progress.attempted == state->progress.completed &&
      state->progress.sampled_rows == state->progress.completed * state->settings.batch_size &&
      state->progress.losses.size() == static_cast<size_t>(state->progress.completed),
      "curve live-state witness requires the complete unskipped per-attempt trace");
  torch::serialize::OutputArchive archive;
  write_text(archive, "artifact_kind", training_source_gain::enabled(state->gain_options) ?
      kMatchedTargetGainContinuationArtifact : state->pooled_options.enabled ? kPooledContextContinuationArtifact :
      state->difference_options.enabled ? kVisibleDifferenceContinuationArtifact : kLearningCurveContinuationArtifact);
  for (const auto &[key, value] : state->audit) write_text(archive, key, value);
  write_text(archive, "fit_source_manifest", source_manifest(state->fit.training_source_ids));
  write_text(archive, "checkpoint_path", path_key(path));
  write_text(archive, "state_capture_policy",
      "live_named_CUDA_parameter_state_to_CPU;no_model_forward;no_optimizer_reload_or_step");
  archive.write("attempted_steps", torch::tensor(state->progress.attempted), true);
  archive.write("completed_steps", torch::tensor(state->progress.completed), true);
  archive.write("sampled_rows", torch::tensor(state->progress.sampled_rows), true);
  archive.write("channel_mixer_placement_value", torch::tensor(state->settings.model.channel_mixer_placement), true);
  archive.write("training_seconds", torch::tensor(state->progress.training_seconds, torch::kFloat64), true);
  archive.write("context_requested_deleted_coordinates", torch::tensor(state->context_counts[0]), true);
  archive.write("context_actual_deleted_coordinates", torch::tensor(state->context_counts[1]), true);
  archive.write("context_restored_coordinates", torch::tensor(state->context_counts[2]), true);
  std::vector<int64_t> counters;
  std::vector<double> losses;
  for (const auto &point : state->progress.losses) {
    counters.insert(counters.end(), {point.attempted, point.completed, point.target_cells});
    losses.insert(losses.end(), {point.loss, point.gradient_norm});
  }
  archive.write("loss_trace_counters", torch::tensor(counters, torch::kInt64).reshape({-1, 3}), true);
  archive.write("loss_trace_values", torch::tensor(losses, torch::kFloat64).reshape({-1, 2}), true);
  std::map<std::string, torch::Tensor> parameters, buffers;
  for (const auto &entry : state->model->named_parameters()) parameters.emplace(entry.key(), entry.value());
  for (const auto &entry : state->model->named_buffers()) buffers.emplace(entry.key(), entry.value());
  write_named_state(archive, "model_parameters", parameters);
  write_named_state(archive, "model_buffers", buffers);
  torch::serialize::OutputArchive scaler;
  state->scaler.save(scaler); archive.write("scaler", scaler);
  torch::serialize::OutputArchive optimizer;
  optimizer.write("parameter_count", torch::tensor(static_cast<int64_t>(parameters.size())), true);
  optimizer.write("active_state_count", torch::tensor(static_cast<int64_t>(state->optimizer->state().size())), true);
  int64_t index = 0, active = 0;
  for (const auto &[name, parameter] : parameters) {
    torch::serialize::OutputArchive entry;
    write_text(entry, "parameter_name", name);
    entry.write("parameter_shape", torch::tensor(parameter.sizes().vec(), torch::kInt64), true);
    const auto found = state->optimizer->state().find(parameter.unsafeGetTensorImpl());
    const bool has_state = found != state->optimizer->state().end();
    entry.write("has_state", torch::tensor(has_state, torch::kBool), true);
    if (has_state) {
      const auto *value = dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
      require(value && value->step() > 0 && value->step() <= state->progress.completed &&
          value->exp_avg().is_cuda() && value->exp_avg_sq().is_cuda() &&
          value->exp_avg().scalar_type() == parameter.scalar_type() &&
          value->exp_avg_sq().scalar_type() == parameter.scalar_type() &&
          value->exp_avg().sizes() == parameter.sizes() && value->exp_avg_sq().sizes() == parameter.sizes() &&
          torch::isfinite(value->exp_avg()).all().item<bool>() &&
          torch::isfinite(value->exp_avg_sq()).all().item<bool>(), "invalid live named CUDA AdamW state");
      entry.write("step", torch::tensor(value->step(), torch::kInt64), true);
      entry.write("exp_avg", cpu_state(value->exp_avg()), true);
      entry.write("exp_avg_sq", cpu_state(value->exp_avg_sq()), true);
      ++active;
    }
    optimizer.write("parameter_" + std::to_string(index++), entry);
  }
  require(active == static_cast<int64_t>(state->optimizer->state().size()),
      "AdamW state must associate with exactly one named model parameter");
  archive.write("optimizer_state", optimizer);
  if (state->difference_options.enabled) {
    archive.write("temporal_difference_input_value",torch::tensor(int64_t(1)),true);
    archive.write("common_parameter_values",torch::tensor(int64_t(225805)),true);
    archive.write("copied_parameter_values",torch::tensor(int64_t(0)),true);
    archive.write("difference_parameter_values",torch::tensor(int64_t(3072)),true);
    write_named_state(archive,"initial_model_parameters",state->difference_initial_parameters);
    write_named_state(archive,"initial_model_buffers",state->difference_initial_buffers);
  }
  if (state->pooled_options.enabled) {
    verify_pooled_inactive(state);
    archive.write("global_pool_input_source_value",torch::tensor(state->settings.model.global_pool_input_source),true);
    archive.write("shared_parameter_values",torch::tensor(int64_t(219469)),true);
    archive.write("copied_parameter_values",torch::tensor(int64_t(state->settings.model.global_pool_input_source ? 219469 : 0)),true);
    archive.write("inactive_projection_values",torch::tensor(int64_t(state->settings.model.global_pool_input_source ? 2080 : 0)),true);
    write_named_state(archive,"initial_model_parameters",state->pooled_initial_parameters);
    write_named_state(archive,"initial_model_buffers",state->pooled_initial_buffers);
    if (state->settings.model.global_pool_input_source==1)
      archive.write("candidate_first_before_copy",state->pooled_first_before_copy,true);
  }
  embedding::archive::save_archive(path + kLearningCurveContinuationSuffix, archive);
}

void save_gain_view(const std::shared_ptr<CurveState> &state, const std::string &path) {
  const auto completed = state->progress.completed;
  require(state->progress.attempted == completed && state->gain_attempts.size() == static_cast<size_t>(completed) &&
      state->gain_indices.size() == static_cast<size_t>(completed), "complete matched-view sample witness required");
  const auto &c = state->settings.model; const auto b = state->settings.batch_size;
  const bool candidate = training_source_gain::gained(state->gain_options);
  require(state->gain_targets.size() == (candidate ? static_cast<size_t>(completed) : 0), "candidate target witness length differs");
  torch::serialize::OutputArchive a;
  write_text(a,"artifact_kind",kMatchedTargetGainViewArtifact);
  training_source_gain::write_manifest(a,state->gain_manifest);
  for (const auto &[key,value]:state->audit)
    if (key != "source_gain_recipe" && key != "gain_manifest_id") write_text(a,key,value);
  write_text(a,"fit_source_manifest",source_manifest(state->fit.training_source_ids));
  write_text(a,"checkpoint_path",path_key(path));
  write_text(a,"target_witness_scope",candidate ? "all-completed-attempts-actual-normalized-CUDA-F32-target-copy" : "unit-control-no-target-copy;closed-zero-leading-dimension");
  a.write("attempted_steps",torch::tensor(state->progress.attempted),true);
  a.write("completed_steps",torch::tensor(completed),true);
  a.write("sampled_rows",torch::tensor(state->progress.sampled_rows),true);
  a.write("source_gain_stream_value",torch::tensor(static_cast<int64_t>(training_source_gain::stream)),true);
  a.write("source_gain_ln2_value",torch::tensor(training_source_gain::ln2,torch::kFloat64),true);
  a.write("attempt_indices",torch::tensor(state->gain_attempts,torch::kInt64),true);
  const auto indices = completed ? torch::stack(state->gain_indices) : torch::empty({0,b},torch::kInt64);
  a.write("sampled_row_indices",indices,true);
  a.write("sampled_gains",state->gain_manifest.row_gains.index_select(0,indices.reshape({-1})).reshape({completed,b}),true);
  a.write("actual_normalized_targets",candidate && completed ? torch::stack(state->gain_targets) :
      torch::empty({0,b,c.channel_count,c.history_length,c.input_width},torch::kFloat32),true);
  embedding::archive::save_archive(path+kMatchedTargetGainViewSuffix,a);
}

void save_point(const std::shared_ptr<CurveState> &state, const std::string &path) {
  require(!state->balanced_failed, "balanced policy previously aborted; saving is forbidden");
  const bool balanced = state->context_options.enabled && context_deletion::is_balanced(state->context_options.recipe);
  require(!balanced || (state->progress.attempted == state->progress.completed &&
      state->branch_counts[0] == state->progress.attempted / 2 + state->progress.attempted % 2 &&
      state->branch_counts[1] == state->progress.attempted / 2), "balanced absolute branch counters differ");
  const auto key = path_key(path);
  std::vector<std::string> outputs{path, path + ".training-raw.pt",
      path + ".scaler.pt", path + ".audit.pt"};
  if (state->continuation_witness) outputs.push_back(path + kLearningCurveContinuationSuffix);
  if (training_source_gain::enabled(state->gain_options)) outputs.push_back(path + kMatchedTargetGainViewSuffix);
  if (state->difference_options.enabled) outputs.push_back(path + kVisibleDifferenceBindingSuffix);
  for (const auto &output : outputs) {
    require(!fs::exists(output) && !fs::is_symlink(fs::symlink_status(output)),
        "refusing to replace an existing point artifact: " + output);
  }
  Checkpoint checkpoint;
  checkpoint.settings = state->settings;
  checkpoint.model = state->model;
  checkpoint.scaler = state->scaler;
  checkpoint.attempted_steps = state->progress.attempted;
  checkpoint.completed_steps = state->progress.completed;
  checkpoint.schema_id = state->training.schema_id;
  checkpoint.dataset_id = state->training.dataset_id;
  checkpoint.scaler_fit_dataset_id = state->training.dataset_id;
  checkpoint.source_fingerprint = workflow_source_fingerprint();
  if (state->context_options.enabled)
    checkpoint.training_policy_id = context_deletion::descriptor(state->context_options.recipe).policy_id;
  if (training_source_gain::gained(state->gain_options)) checkpoint.training_policy_id = kMatchedTargetGainPolicy;
  save_checkpoint(path, checkpoint, *state->optimizer);
  save_dataset(path + ".training-raw.pt", state->training);
  save_scaler(path + ".scaler.pt", state->scaler, state->settings.model,
              state->training.schema_id, state->training.dataset_id);
  torch::serialize::OutputArchive audit;
  write_text(audit, "artifact_kind", "rpb_learning_curve_training_audit_v1");
  for (const auto &[name, value] : state->audit) write_text(audit, name, value);
  write_text(audit, "fit_source_manifest", source_manifest(state->fit.training_source_ids));
  audit.write("attempted_steps", torch::tensor(state->progress.attempted), true);
  audit.write("completed_steps", torch::tensor(state->progress.completed), true);
  audit.write("sampled_rows", torch::tensor(state->progress.sampled_rows), true);
  audit.write("channel_order", torch::tensor(state->fit.channel_ids, torch::kInt64), true);
  audit.write("sampling_interval", torch::tensor(state->fit.sampling_interval, torch::kFloat64), true);
  audit.write("endpoint", torch::tensor(state->fit.endpoint, torch::kFloat64), true);
  audit.write("training_seconds", torch::tensor(state->progress.training_seconds, torch::kFloat64), true);
  write_text(audit, "model_weight_update_budget", std::to_string(state->progress.completed));
  const auto report = progress_report(state);
  audit.write("weights_changed", torch::tensor(report.weights_changed, torch::kBool), true);
  audit.write("finite_gradients", torch::tensor(report.finite_gradients, torch::kBool), true);
  if (state->pooled_options.enabled) {
    audit.write("global_pool_input_source_value",torch::tensor(state->settings.model.global_pool_input_source),true);
    audit.write("shared_parameter_values",torch::tensor(int64_t(219469)),true);
    audit.write("copied_parameter_values",torch::tensor(int64_t(state->settings.model.global_pool_input_source ? 219469 : 0)),true);
    audit.write("inactive_projection_values",torch::tensor(int64_t(state->settings.model.global_pool_input_source ? 2080 : 0)),true);
  }
  if (state->audit.count("architecture_id"))
    audit.write("channel_mixer_placement_value",
        torch::tensor(state->settings.model.channel_mixer_placement, torch::kInt64), true);
  if (state->difference_options.enabled)
    audit.write("temporal_difference_input_value",torch::tensor(int64_t(1)),true);
  if (state->context_options.enabled) {
    audit.write("context_deletion_ratio_value",
        torch::tensor(context_deletion::descriptor(state->context_options.recipe).ratio, torch::kFloat64), true);
    audit.write("context_deletion_stream_value", torch::tensor(static_cast<int64_t>(context_deletion::stream)), true);
    audit.write("context_requested_deleted_coordinates", torch::tensor(state->context_counts[0]), true);
    audit.write("context_actual_deleted_coordinates", torch::tensor(state->context_counts[1]), true);
    audit.write("context_restored_coordinates", torch::tensor(state->context_counts[2]), true);
    if (balanced) {
      write_text(audit, "context_ordinary_attempts", std::to_string(state->branch_counts[0]));
      write_text(audit, "context_deletion_attempts", std::to_string(state->branch_counts[1]));
      audit.write("context_ordinary_attempts_value", torch::tensor(state->branch_counts[0]), true);
      audit.write("context_deletion_attempts_value", torch::tensor(state->branch_counts[1]), true);
    }
  }
  embedding::archive::save_archive(path + ".audit.pt", audit);
  if (state->continuation_witness) save_continuation_state(state, path);
  if (training_source_gain::enabled(state->gain_options)) save_gain_view(state,path);
  if (state->difference_options.enabled) {
    torch::serialize::OutputArchive binding;
    write_text(binding,"artifact_kind",kVisibleDifferenceBindingArtifact);
    for (const auto &[name,value]:state->audit) write_text(binding,name,value);
    write_text(binding,"fit_source_manifest",source_manifest(state->fit.training_source_ids));
    write_text(binding,"checkpoint_path",key);
    binding.write("attempted_steps",torch::tensor(state->progress.attempted),true);
    binding.write("completed_steps",torch::tensor(state->progress.completed),true);
    binding.write("sampled_rows",torch::tensor(state->progress.sampled_rows),true);
    binding.write("temporal_difference_input_value",torch::tensor(int64_t(1)),true);
    for (const auto &suffix : {std::string(),std::string(".audit.pt"),std::string(".scaler.pt"),std::string(".training-raw.pt"),std::string(kLearningCurveContinuationSuffix)})
      write_text(binding,"checkpoint_content_id"+suffix,pooled_file_id(path+suffix));
    embedding::archive::save_archive(path+kVisibleDifferenceBindingSuffix,binding);
  }
  state->saved.emplace(key, std::make_pair(checkpoint.attempted_steps, checkpoint.completed_steps));
  if (state->context_options.enabled) state->saved_context_counts.emplace(key, state->context_counts);
  if (balanced) state->saved_branch_counts.emplace(key, state->branch_counts);
}

ev::CurveSnapshot snapshot_point(const std::shared_ptr<CurveState> &state,
                                const std::string &path) {
  require(state->settings.model.temporal_difference_input==0,"visible difference requires its new CUDA snapshot adapter");
  require(state->settings.model.global_pool_input_source == 0,"pooled source1 requires its new CUDA snapshot adapter");
  require(!state->continuation_witness,
      "curve snapshots require the protocol-bound CUDA adapter; historical CPU serving is forbidden");
  require(state->settings.model.channel_mixer_placement == 0,
      "early mixer snapshots require the protocol-bound CUDA adapter; historical CPU serving is forbidden");
  require(!state->balanced_failed, "balanced policy previously aborted; snapshots are forbidden");
  const auto saved = state->saved.find(path_key(path));
  require(saved != state->saved.end(), "snapshot requires a point saved by this trainer");
  // Separate model construction is essential: core keeps its construction device
  // in Config, and held-out callbacks must never mutate the live training model.
  auto checkpoint = std::make_shared<Checkpoint>(load_checkpoint(path, state->settings.model.device));
  require(checkpoint->dataset_id == state->training.dataset_id &&
      checkpoint->schema_id == state->training.schema_id &&
      checkpoint->scaler_fit_dataset_id == state->training.dataset_id &&
      checkpoint->scaler.identity() == state->scaler.identity() &&
      settings_text(checkpoint->settings) == settings_text(state->settings) &&
      checkpoint->attempted_steps == saved->second.first &&
      checkpoint->completed_steps == saved->second.second,
      "saved point settings, scaler, training identity or counters differ from the trainer");
  require(checkpoint->training_policy_id == (state->context_options.enabled ?
      context_deletion::descriptor(state->context_options.recipe).policy_id : ""),
      "saved training policy differs from the trainer");
  checkpoint->model->eval();
  for (auto &parameter : checkpoint->model->parameters()) parameter.set_requires_grad(false);
  EvaluationOptions options;
  options.checkpoint_path = path;
  options.surface_prefix = "curve_checkpoint";
  const auto frozen = make_evaluation_provider(options)(state->fit);
  const auto base = std::string("curve_checkpoint_trained") +
      (state->settings.model.channel_mixer_layers > 0 ? "_contextual" : "");
  ev::CurveSnapshot snapshot;
  snapshot.features.provenance = frozen.provenance +
      "; exact saved continuous CUDA training point; no matched random provider surface";
  if (state->context_options.enabled) {
    const auto &selected = context_deletion::descriptor(state->context_options.recipe);
    snapshot.features.provenance += "; model_tag=" + std::string(selected.model_tag) +
        "; training_policy=" + selected.policy_id + "; inference remains ordinary mode2/mixer1/native32";
  }
  snapshot.features.audit_fields = frozen.audit_fields;
  for (auto field = snapshot.features.audit_fields.begin(); field != snapshot.features.audit_fields.end();) {
    if (field->first.rfind("curve_checkpoint_untrained_", 0) == 0)
      field = snapshot.features.audit_fields.erase(field);
    else
      ++field;
  }
  for (const auto &[name, value] : state->audit) snapshot.features.audit_fields[name] = value;
  snapshot.features.audit_fields["checkpoint_path"] = path;
  snapshot.features.audit_fields["attempted_steps"] = std::to_string(saved->second.first);
  snapshot.features.audit_fields["completed_steps"] = std::to_string(saved->second.second);
  snapshot.features.audit_fields["snapshot_policy"] =
      "independent_CPU_feature_model_and_CUDA_reconstruction_model;frozen_train_scaler;no_refit";
  if (state->context_options.enabled) {
    const auto &counts = state->saved_context_counts.at(path_key(path));
    snapshot.features.audit_fields["context_requested_deleted_coordinates"] = std::to_string(counts[0]);
    snapshot.features.audit_fields["context_actual_deleted_coordinates"] = std::to_string(counts[1]);
    snapshot.features.audit_fields["context_restored_coordinates"] = std::to_string(counts[2]);
    if (context_deletion::is_balanced(state->context_options.recipe)) {
      const auto &branches = state->saved_branch_counts.at(path_key(path));
      snapshot.features.audit_fields["context_ordinary_attempts"] = std::to_string(branches[0]);
      snapshot.features.audit_fields["context_deletion_attempts"] = std::to_string(branches[1]);
    }
  }
  snapshot.features.surfaces.emplace("curve_global", frozen.surfaces.at(base + "_global"));
  snapshot.features.surfaces.emplace("curve_channel_concatenation",
      frozen.surfaces.at(base + "_channel_concatenation"));
  snapshot.features.extract = [frozen, base, enabled = state->context_options.enabled,
      recipe = state->context_options.recipe](const embedding::Batch &batch) {
    const auto all = frozen.extract(batch);
    ev::FeatureMap result;
    result.emplace("curve_global", all.at(base + "_global"));
    result.emplace("curve_channel_concatenation", all.at(base + "_channel_concatenation"));
    if (enabled) {
      const auto &selected = context_deletion::descriptor(recipe);
      for (auto &[name, surface] : result) {
        (void)name;
        surface.provenance += "; model_tag=" + std::string(selected.model_tag) + "; training_policy=" + selected.policy_id;
      }
    }
    return result;
  };
  // Ordinary checkpoint, raw/scaler companions and producer audit were saved
  // already. Do not re-export the frozen provider's unrelated random control.
  snapshot.features.save_assets = [audit_fields = snapshot.features.audit_fields](
      const std::string &directory) {
    const auto destination = (fs::path(directory) / "curve-snapshot-audit.pt").string();
    require(!fs::exists(destination), "refusing to replace snapshot provenance");
    torch::serialize::OutputArchive audit;
    write_text(audit, "artifact_kind", "rpb_learning_curve_snapshot_audit_v1");
    for (const auto &[name, value] : audit_fields) write_text(audit, name, value);
    embedding::archive::save_archive(destination, audit);
  };
  // Only metadata are captured; reconstruction cannot access training rows.
  auto metadata = state->fit;
  metadata.training_observations = {};
  metadata.training_source_ids.clear();
  snapshot.reconstruct = [checkpoint, metadata](const embedding::Batch &batch,
                                                const torch::Tensor &hidden) {
    torch::NoGradGuard no_grad;
    const auto raw = protocol_input(batch, checkpoint->settings.model, metadata);
    const auto normalized = checkpoint->scaler.transform(raw, checkpoint->settings.model);
    const auto output = checkpoint->model->forward(normalized, hidden);
    require(output.reconstruction.is_cuda() &&
        torch::isfinite(output.reconstruction).all().item<bool>(),
        "snapshot reconstruction must be finite CUDA model output");
    return ev::CurveReconstruction{output.reconstruction.detach().to(torch::kCPU),
        normalized.data.detach().to(torch::kCPU), output.eligible_channels.to(torch::kCPU)};
  };
  return snapshot;
}
} // namespace

ev::CurveTrainerFactory make_learning_curve_trainer(const Settings &settings) {
  return make_learning_curve_trainer(settings, ContextDeletionOptions{});
}

ev::CurveTrainerFactory make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options) {
  return make_learning_curve_trainer(settings, options, LearningCurveStateWitnessOptions{});
}

ev::CurveTrainerFactory make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options,
    LearningCurveStateWitnessOptions state_witness) {
  return make_learning_curve_trainer(settings,options,state_witness,TrainingSourceGainOptions{});
}

ev::CurveTrainerFactory make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options,
    LearningCurveStateWitnessOptions state_witness, TrainingSourceGainOptions gain) {
  return make_learning_curve_trainer(settings,options,state_witness,gain,PooledContextInitializationOptions{});
}

ev::CurveTrainerFactory make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options,
    LearningCurveStateWitnessOptions state_witness, TrainingSourceGainOptions gain, PooledContextInitializationOptions pooled) {
  return make_learning_curve_trainer(settings,options,state_witness,gain,pooled,VisibleDifferenceOptions{});
}

ev::CurveTrainerFactory make_learning_curve_trainer(const Settings &settings, ContextDeletionOptions options,
    LearningCurveStateWitnessOptions state_witness, TrainingSourceGainOptions gain, PooledContextInitializationOptions pooled,
    VisibleDifferenceOptions difference) {
  const auto valid_pin=[](const std::string &v) { return v.size()==64 && v.find_first_not_of("0123456789abcdef")==std::string::npos; };
  require(settings.model.temporal_difference_input==0 || difference.enabled,"historical learning factory rejects visible difference input1");
  require(difference.enabled || (difference.parent_point0_checkpoint_path.empty() && difference.expected_parent_core_source_fingerprint.empty() &&
      difference.expected_parent_training_producer_source_fingerprint.empty() && difference.expected_parent_fit_protocol.empty()),
      "disabled visible difference binding must be empty");
  require(!difference.enabled || (settings.model.temporal_difference_input==1 && settings.model.global_pool_input_source==0 &&
      settings.model.channel_mixer_placement==1 && state_witness.enabled && !pooled.enabled && !training_source_gain::enabled(gain) &&
      options.enabled && options.recipe==ContextDeletionRecipe::coordinate15_v1 && settings.log_every==1 &&
      !difference.parent_point0_checkpoint_path.empty() && valid_pin(difference.expected_parent_core_source_fingerprint) &&
      valid_pin(difference.expected_parent_training_producer_source_fingerprint) &&
      (difference.expected_parent_fit_protocol=="early-mixer-reliability-v1/lag_sign" ||
       difference.expected_parent_fit_protocol=="early-mixer-reliability-engineering-v1/lag_sign")),
      "closed visible-difference scope and explicit original source pins required");
  require(settings.model.global_pool_input_source == 0 || pooled.enabled,
      "historical learning factory rejects pooled source1");
  require(pooled.enabled || pooled.control_point0_checkpoint_path.empty(), "disabled pooled binding must be empty");
  require(!pooled.enabled || (state_witness.enabled && !training_source_gain::enabled(gain) && options.enabled &&
      options.recipe==ContextDeletionRecipe::coordinate15_v1 && settings.model.channel_mixer_placement==1 &&
      settings.model.global_bottleneck_mode==2 && settings.log_every==1 &&
      (settings.model.global_pool_input_source==1 ? !pooled.control_point0_checkpoint_path.empty() : pooled.control_point0_checkpoint_path.empty())),
      "closed pooled coordinate15 initialization binding required");
  training_source_gain::validate_options(gain);
  context_deletion::validate_options(options);
  validate_settings(settings);
  require(settings.model.device.is_cuda(), "learning-curve training requires an explicit CUDA device");
  require(torch::cuda::is_available(), "CUDA is unavailable; CPU fallback is not permitted");
  require(!options.enabled || (settings.model.global_bottleneck_mode == 2 &&
      settings.model.channel_mixer_layers == 1 && settings.model.export_width == 32),
      "context deletion requires the unchanged mode2/mixer1/native32 architecture");
  require(!training_source_gain::enabled(gain) || (state_witness.enabled && options.enabled &&
      options.recipe == ContextDeletionRecipe::coordinate15_v1 && settings.model.channel_mixer_placement == 1 &&
      settings.log_every == 1), "matched source gain requires early coordinate15 with complete live-state witness");
  return [settings, options, state_witness, gain, pooled, difference](const ev::ProviderFitInput &fit) {
    const bool difference_protocol=fit.protocol_id=="visible-difference-v1/lag_sign" ||
        fit.protocol_id=="visible-difference-engineering-v1/lag_sign";
    require(difference_protocol==difference.enabled,"visible-difference namespace and explicit option must agree");
    const bool pooled_protocol=fit.protocol_id=="pooled-context-v1/lag_sign" || fit.protocol_id=="pooled-context-engineering-v1/lag_sign";
    require(pooled_protocol==pooled.enabled,"pooled namespace and explicit initialization scope must agree");
    const bool gain_protocol = fit.protocol_id == kMatchedTargetGainFitProtocol || fit.protocol_id == kMatchedTargetGainFixtureFitProtocol;
    require(gain_protocol == training_source_gain::enabled(gain), "new matched-view namespace and explicit recipe must agree");
    const bool curve_protocol = fit.protocol_id == "early-mixer-learning-curve-v1/lag_sign" ||
        fit.protocol_id == "early-mixer-learning-curve-engineering-v1/lag_sign";
    const bool early_protocol = fit.protocol_id == "early-mixer-reliability-v1/lag_sign" ||
        fit.protocol_id == "early-mixer-reliability-engineering-v1/lag_sign" || curve_protocol || gain_protocol || pooled_protocol || difference_protocol;
    require(!state_witness.enabled || ((curve_protocol || gain_protocol || pooled_protocol || difference_protocol) && options.enabled &&
        options.recipe == ContextDeletionRecipe::coordinate15_v1 && settings.log_every == 1),
        "live continuation witness requires the separately bound coordinate15 curve and complete trace");
    require(settings.model.channel_mixer_placement == 0 ||
        (early_protocol && options.enabled && options.recipe == ContextDeletionRecipe::coordinate15_v1),
        "early placement requires the separately bound coordinate15 protocol");
    require(fit.shape.channel_count == settings.model.channel_count &&
        fit.shape.history_length == settings.model.history_length &&
        fit.shape.input_width == settings.model.input_width,
        "resolved configuration dimensions differ from the declared fit card");
    require(static_cast<int64_t>(fit.channel_ids.size()) == settings.model.channel_count &&
        std::set<int64_t>(fit.channel_ids.begin(), fit.channel_ids.end()).size() == fit.channel_ids.size(),
        "fit must declare unique semantic channel IDs");
    const auto ids = resolved_channel_ids(settings.model);
    require(std::set<int64_t>(ids.begin(), ids.end()) ==
        std::set<int64_t>(fit.channel_ids.begin(), fit.channel_ids.end()),
        "resolved configuration semantic IDs differ from the fit card");
    require(!fit.feature_units.empty() && std::isfinite(fit.endpoint) &&
        std::isfinite(fit.sampling_interval) && fit.sampling_interval > 0 &&
        std::abs(settings.model.sampling_interval - fit.sampling_interval) <=
            1e-12 * std::max(1.0, std::abs(settings.model.sampling_interval)),
        "fit units, endpoint or sampling interval are missing/inconsistent");
    require(fit.training_observations.data.defined() &&
        fit.training_observations.data.dim() == 4 &&
        fit.shape.device.is_cpu() &&
        fit.training_observations.data.device().is_cpu() &&
        fit.training_observations.data.scalar_type() == fit.shape.dtype &&
        fit.training_observations.feature_mask.defined() &&
        fit.training_observations.feature_mask.device().is_cpu() &&
        fit.training_source_ids.size() == static_cast<size_t>(fit.training_observations.data.size(0)),
        "fit requires CPU training observations matching declared precision and source IDs");
    for (const auto &id : fit.training_source_ids) require(!id.empty(), "empty permitted training source ID");

    auto state = std::make_shared<CurveState>();
    state->settings = settings;
    state->context_options = options;
    state->continuation_witness = state_witness.enabled;
    state->gain_options = gain;
    state->pooled_options = pooled;
    state->difference_options = difference;
    state->settings.seed = static_cast<int64_t>(fit.seed & 0x7fffffffffffffffULL);
    state->fit = fit;
    state->fit.training_observations.data = fit.training_observations.data.detach().clone();
    state->fit.training_observations.feature_mask = fit.training_observations.feature_mask.detach().clone();
    const auto raw = protocol_input(state->fit.training_observations, settings.model, state->fit);
    state->training = describe_dataset(raw, settings.model, fit.feature_units);
    state->scaler = fit_scaler(raw, settings.model);
    if (gain_protocol) state->gain_manifest = training_source_gain::make_manifest(fit.training_source_ids,state->settings.seed,gain);
    state->initialization_seed = training_detail::mixed(fit.seed ^ 0x7270622d696e6974ULL);
    torch::manual_seed(state->initialization_seed);
    // Construct with CUDA Config from the outset: Model::to alone cannot update
    // its private allocation/device contract. Module initialization remains CPU.
    state->model = Model(state->settings.model);
    state->model->train();
    prepare_pooled_initialization(state);
    prepare_visible_difference_initialization(state);
    state->optimizer = std::make_unique<torch::optim::AdamW>(state->model->parameters(),
        torch::optim::AdamWOptions(settings.learning_rate).weight_decay(settings.weight_decay));
    for (const auto &parameter : state->model->parameters()) {
      state->initial_parameters.push_back(parameter.detach().clone());
      state->progress.parameter_count += parameter.numel();
      if (parameter.is_cuda()) state->progress.cuda_parameter_count += parameter.numel();
    }
    require(state->progress.parameter_count > 0 &&
        state->progress.parameter_count == state->progress.cuda_parameter_count,
        "every model parameter must reside on CUDA");
    state->progress.training_device = settings.model.device.str();
    state->progress.preprocessing_id = state->scaler.identity();
    state->progress.training_dataset_id = state->training.dataset_id;
    state->audit = {{"provider_id", kEncoderId}, {"adapter_id", "rpb_continuous_cuda_learning_curve_v1"},
        {"protocol_id", fit.protocol_id}, {"fit_policy", "permitted_training_observations_only;scaler_fit_once"},
        {"resolved_settings", settings_text(state->settings)}, {"feature_units", fit.feature_units},
        {"initialization_seed", std::to_string(state->initialization_seed)},
        {"actual_training_seed", std::to_string(state->settings.seed)},
        {"rng_policy", "splitmix64-counter-rows-masks-torch-attempt-v1"},
        {"optimizer_policy", "one_continuous_AdamW_state;absolute_completed_update_budgets"},
        {"training_device", state->progress.training_device},
        {"parameter_count", std::to_string(state->progress.parameter_count)},
        {"cuda_parameter_count", std::to_string(state->progress.cuda_parameter_count)},
        {"output_semantics", output_semantics(settings.model)},
        {"reconstruction_export_semantics", reconstruction_output_semantics(settings.model)},
        {"training_dataset_id", state->training.dataset_id},
        {"scaler_fit_dataset_id", state->training.dataset_id}, {"preprocessing_id", state->scaler.identity()},
        {"fit_source_manifest_id", manifest_identity(source_manifest(fit.training_source_ids))},
        {"training_observation_rows", std::to_string(fit.training_source_ids.size())},
        {"training_source_groups", std::to_string(std::set<std::string>(
            fit.training_source_ids.begin(), fit.training_source_ids.end()).size())},
        {"core_writer_source_fingerprint", workflow_source_fingerprint()},
        {"training_producer_source_fingerprint", EVALUATION_SOURCE_ID},
        {"source_fingerprint_scope", "ordinary_checkpoint_field=core_writer;companion_audit_training_producer=evaluation_source"},
        {"trace_policy", "first_completed_update;every_log_every_updates;queried_milestone_endpoints_persist_as_cumulative_prefix"},
        {"sampling_policy", "with_replacement_counter_rows;sampled_rows_includes_no_update_attempts"}};
    if (options.enabled) {
      const auto &selected = context_deletion::descriptor(options.recipe);
      state->audit.emplace("model_tag", settings.model.channel_mixer_placement == 1 ? "RPB-v10" : selected.model_tag);
      state->audit.emplace("training_policy_id", selected.policy_id);
      state->audit.emplace("context_deletion_ratio", selected.ratio_text);
      state->audit.emplace("context_deletion_stream", "0x6374782d64726f70");
      state->audit.emplace("context_deletion_rng_policy", context_deletion::rng_policy);
      state->audit.emplace("context_deletion_repair_policy", context_deletion::repair_policy);
      state->audit.emplace("context_deletion_visibility_policy", context_deletion::visibility_policy);
      state->audit.emplace("context_deletion_count_policy", "cumulative-requested/actual/restored-coordinate-counts;eligible-forward-batches-only");
      state->audit.emplace("context_deletion_resume_policy", "fresh-continuous-only;ordinary-workflow-resume-rejected;no-augmented-resume-API");
      if (context_deletion::is_balanced(options.recipe)) {
        state->audit.emplace("context_deletion_schedule_policy", context_deletion::balanced_schedule_policy);
        state->audit.emplace("context_deletion_rate_scope", context_deletion::balanced_rate_scope);
        state->audit.emplace("context_deletion_branch_count_policy", context_deletion::balanced_branch_count_policy);
        state->audit.emplace("context_deletion_skip_policy", context_deletion::balanced_skip_policy);
      }
    }
    if (early_protocol) {
      state->audit.emplace("channel_mixer_placement", std::to_string(settings.model.channel_mixer_placement));
      state->audit.emplace("architecture_id", architecture_id(settings.model));
    }
    if (state_witness.enabled) {
      state->audit.emplace("continuation_state_artifact_kind", gain_protocol ? kMatchedTargetGainContinuationArtifact : pooled_protocol ? kPooledContextContinuationArtifact :
          difference_protocol ? kVisibleDifferenceContinuationArtifact : kLearningCurveContinuationArtifact);
      state->audit.emplace("continuation_state_suffix", kLearningCurveContinuationSuffix);
      state->audit.emplace("continuation_state_policy", "live_named_CPU_model_buffers_scaler_AdamW_and_complete_trace_v1");
      state->audit.emplace("curve_skip_policy", "abort_ineligible_attempt;no_skipped_update_prefix_permitted");
    }
    if (gain_protocol) {
      state->audit["model_tag"] = training_source_gain::gained(gain) ? "RPB-v11" : "RPB-v10";
      state->audit["training_policy_id"] = training_source_gain::gained(gain) ? kMatchedTargetGainPolicy : context_deletion::descriptor(options.recipe).policy_id;
      state->audit.emplace("context_component_policy_id",context_deletion::descriptor(options.recipe).policy_id);
      state->audit.emplace("source_gain_recipe",training_source_gain::recipe_name(gain.recipe));
      state->audit.emplace("source_gain_stream","0x6761696e2d763131");
      state->audit.emplace("source_gain_rng_policy",training_source_gain::rng_policy);
      state->audit.emplace("source_gain_policy",training_source_gain::gain_policy);
      state->audit.emplace("source_gain_target_policy",training_source_gain::target_policy);
      state->audit.emplace("source_gain_inference_policy",training_source_gain::inference_policy);
      state->audit.emplace("gain_manifest_id",state->gain_manifest.identity);
      state->audit.emplace("gain_view_artifact_kind",kMatchedTargetGainViewArtifact);
      state->audit.emplace("gain_view_suffix",kMatchedTargetGainViewSuffix);
      state->audit.emplace("source_gain_resume_policy","fresh-continuous-only;ordinary-and-historical-tagged-resume-rejected;no-gain-resume-API");
      state->audit.emplace("source_gain_evidence_cost_policy",training_source_gain::gained(gain) ?
          "update-loop-includes-per-attempt-CPU-normalized-target-copy;sample-index-capture;save-IO-separate" :
          "update-loop-includes-sample-index-capture;no-full-target-copy;save-IO-separate");
    }

    if (pooled_protocol) {
      const bool candidate=settings.model.global_pool_input_source==1;
      state->audit["model_tag"]=candidate ? "RPB-v12" : "RPB-v10";
      state->audit.emplace("global_pool_input_source",std::to_string(settings.model.global_pool_input_source));
      state->audit.emplace("global_pool_input_semantics",global_pool_input_semantics(settings.model));
      state->audit.emplace("pooled_initialization_policy",candidate ?
          "copy-all-shared-named-parameters-and-buffers-from-compact-point0-before-AdamW;except-global_pool_first.weight" :
          "compact-control-independent-initialization;no-copy");
      state->audit.emplace("pooled_shared_parameter_values","219469");
      state->audit.emplace("pooled_copied_parameter_values",candidate ? "219469" : "0");
      state->audit.emplace("pooled_inactive_projection_values",candidate ? "2080" : "0");
      state->audit.emplace("pooled_loss_reachable_parameter_values",candidate ? "229869" : "225805");
      state->audit.emplace("pooled_nonshared_parameter_name","global_pool_first.weight");
      state->audit.emplace("pooled_reference_point0_path",candidate ? path_key(pooled.control_point0_checkpoint_path) : "");
      for (const auto &[suffix,id]:state->pooled_reference_content) state->audit.emplace("pooled_reference_content_id"+suffix,id);
      std::ostringstream paired;
      paired << "{\"input_source\":" << settings.model.global_pool_input_source
          << ",\"copied_before_AdamW\":true,\"shared_parameter_values\":219469,\"copied_parameter_values\":"
          << (candidate ? 219469 : 0) << ",\"inactive_projection_values\":" << (candidate ? 2080 : 0)
          << ",\"nonshared_parameter_name\":\"global_pool_first.weight\",\"reference_point0_path\":"
          << std::quoted(candidate ? path_key(pooled.control_point0_checkpoint_path) : std::string()) << "}";
      state->audit.emplace("paired_initialization_json",paired.str());
    }

    if (difference_protocol) {
      state->audit["model_tag"]="RPB-v13";
      state->audit.emplace("temporal_difference_input","1");
      state->audit.emplace("temporal_difference_input_semantics",visible_difference_input_semantics(settings.model));
      state->audit.emplace("visible_difference_initialization_policy","independent-original-common-state-exact;last-zero-branch-no-RNG;assert-before-AdamW;no-copy");
      state->audit.emplace("visible_difference_common_parameter_values","225805");
      state->audit.emplace("visible_difference_added_parameter_values","3072");
      state->audit.emplace("visible_difference_copied_parameter_values","0");
      state->audit.emplace("visible_difference_parent_point0_path",path_key(difference.parent_point0_checkpoint_path));
      state->audit.emplace("visible_difference_parent_core_source_fingerprint",difference.expected_parent_core_source_fingerprint);
      state->audit.emplace("visible_difference_parent_training_producer_source_fingerprint",difference.expected_parent_training_producer_source_fingerprint);
      state->audit.emplace("visible_difference_parent_fit_protocol",difference.expected_parent_fit_protocol);
      state->audit.emplace("visible_difference_binding_artifact_kind",kVisibleDifferenceBindingArtifact);
      state->audit.emplace("visible_difference_binding_suffix",kVisibleDifferenceBindingSuffix);
      for (const auto &[suffix,id]:state->difference_parent_content) state->audit.emplace("visible_difference_parent_content_id"+suffix,id);
    }

    ev::CurveTrainer trainer;
    trainer.audit_fields = state->audit;
    trainer.train_to = [state](int64_t budget) {
      require(!state->balanced_failed, "balanced policy previously aborted; further training is forbidden");
      const bool balanced = state->context_options.enabled && context_deletion::is_balanced(state->context_options.recipe);
      require(budget >= state->progress.completed, "completed-update budgets must be monotonic");
      require(budget <= state->settings.steps, "requested completed updates exceed the frozen session budget");
      if (budget == state->progress.completed) return progress_report(state);
      state->model->train();
      torch::cuda::synchronize(state->settings.model.device.index());
      const auto began = std::chrono::steady_clock::now();
      const auto finish_timing = [&] {
        torch::cuda::synchronize(state->settings.model.device.index());
        state->progress.training_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - began).count();
      };
      try {
        while (state->progress.completed < budget &&
               state->progress.attempted < state->settings.attempt_limit) {
          const auto attempt = state->progress.attempted;
          require(state->progress.sampled_rows <= std::numeric_limits<int64_t>::max() -
              state->settings.batch_size, "sampled row counter overflow");
          const auto indices = training_detail::sampled_indices(state->training.input.data.size(0),
              state->settings.batch_size, state->settings.seed, attempt);
          const auto batch = training_source_gain::gained(state->gain_options) ?
              state->scaler.transform(training_source_gain::apply(training_detail::selected(state->training.input,indices),
                  indices,state->gain_manifest),state->settings.model) :
              state->scaler.transform(training_detail::selected(state->training.input, indices), state->settings.model);
          state->progress.last_input_cuda = batch.data.is_cuda();
          require(state->progress.last_input_cuda, "normalized training input is not on CUDA");
          const auto mask = make_training_mask(batch.observed, state->settings.model,
              training_detail::counter_seed(state->settings.seed, attempt, 0x6d61736bULL));
          torch::manual_seed(training_detail::counter_seed(state->settings.seed, attempt, 0x746f726368ULL));
          ++state->progress.attempted;
          state->progress.sampled_rows += state->settings.batch_size;
          if (!mask.eligible_channels.any().item<bool>()) {
            if (state->continuation_witness) {
              state->balanced_failed = true;
              require(false, "curve policy aborts an ineligible original masking attempt before update");
            }
            if (balanced) {
              state->balanced_failed = true;
              require(false, "balanced policy aborts an ineligible original masking attempt before update");
            }
            continue;
          }
          state->optimizer->zero_grad();
          ForwardOutput output;
          const bool deletion = state->context_options.enabled &&
              context_deletion::deletion_attempt(state->context_options.recipe, attempt);
          if (deletion) {
            const auto context = context_deletion::make_plan(mask, batch.channel_ids,
                state->settings.model, state->settings.seed, attempt, state->context_options.recipe);
            const std::array<int64_t, 3> counts{context.requested_count, context.actual_count, context.restored_count};
            for (size_t i = 0; i < counts.size(); ++i) {
              require(state->context_counts[i] <= std::numeric_limits<int64_t>::max() - counts[i],
                  "cumulative context-deletion counter overflow");
              state->context_counts[i] += counts[i];
            }
            output = context_deletion::training_forward(*state->model, batch, mask, context);
          } else {
            output = state->model->forward(batch, mask.hidden);
          }
          state->progress.last_loss_cuda = output.loss.is_cuda();
          require(state->progress.last_loss_cuda && output.eligible_example_count > 0 &&
              torch::isfinite(output.loss).all().item<bool>(), "invalid/non-CUDA training loss");
          output.loss.backward();
          for (const auto &parameter : state->model->parameters())
            require(!parameter.grad().defined() || parameter.grad().is_cuda(),
                "a model gradient is not on CUDA");
          const auto gradient_norm = torch::nn::utils::clip_grad_norm_(state->model->parameters(),
              state->settings.gradient_clip_norm > 0 ? state->settings.gradient_clip_norm :
                  std::numeric_limits<double>::infinity(), 2.0, true);
          state->progress.finite_gradients = std::isfinite(gradient_norm);
          require(state->progress.finite_gradients, "nonfinite training gradients");
          state->optimizer->step();
          ++state->progress.completed;
          if (training_source_gain::enabled(state->gain_options)) {
            state->gain_attempts.push_back(attempt); state->gain_indices.push_back(cpu_state(indices));
            if (training_source_gain::gained(state->gain_options)) state->gain_targets.push_back(cpu_state(batch.data));
          }
          if (balanced) ++state->branch_counts[deletion ? 1 : 0];
          state->last_loss = {state->progress.attempted, state->progress.completed,
              output.target_cell_count, output.loss.item<double>(), gradient_norm};
          if (state->progress.completed == 1 ||
              state->progress.completed % state->settings.log_every == 0)
            state->progress.losses.push_back(state->last_loss);
        }
      } catch (...) {
        if (balanced || training_source_gain::enabled(state->gain_options) || state->pooled_options.enabled || state->difference_options.enabled) state->balanced_failed = true;
        finish_timing();
        throw;
      }
      finish_timing();
      require(state->progress.completed == budget, "continuous training exhausted its cumulative attempt budget");
      return progress_report(state);
    };
    trainer.save_checkpoint = [state](const std::string &path) { save_point(state, path); };
    trainer.snapshot = [state](const std::string &path) { return snapshot_point(state, path); };
    return trainer;
  };
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae
