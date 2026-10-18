// Copyright (c) 2026 Waajacu. All rights reserved.
// See ../../../../../../../../../../../../LICENSE for the copyright and permission notice.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v4/consecutive_change_checkpoint.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <set>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v4::consecutive_change_training {

inline constexpr const char *kChangeTrainingStateKind =
    "rpb_consecutive_change_training_state_v1";
inline constexpr const char *kChangeTrainingCounterContract = "splitmix64-counter-v1";

// The caller binds these exact identities to immutable input/split/recipe
// manifests. This archive does not discover sources or choose a data split.
struct TrainingIdentity {
  std::string input_source_identity, split_identity, recipe_identity;
};

// Save only between completed attempts, with no gradient accumulation pending.
// A skipped ineligible batch consumes one attempt, but no optimizer update.
// The next row/mask/context streams use attempted_batches and these base seeds.
struct TrainingProgress {
  int64_t completed_updates{0}, attempted_batches{0};
  int64_t row_seed{0}, mask_seed{0}, context_seed{0};
};

inline int64_t training_counter_seed(int64_t base_seed, int64_t attempted_batches) {
  TORCH_CHECK(base_seed >= 0 && attempted_batches >= 0,
              "[rpb-v23 training] nonnegative counter stream required");
  uint64_t x = static_cast<uint64_t>(base_seed) +
      UINT64_C(0x9e3779b97f4a7c15) * (static_cast<uint64_t>(attempted_batches) + 1);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return static_cast<int64_t>((x ^ (x >> 31)) & UINT64_C(0x7fffffffffffffff));
}

inline torch::optim::AdamWOptions content_training_adamw_options() {
  return torch::optim::AdamWOptions(.001).weight_decay(.0001);
}

namespace training_state_detail {
namespace d = configurable_checkpoint_detail;

inline void validate_identity(const TrainingIdentity &i) {
  for (const auto *v : {&i.input_source_identity, &i.split_identity, &i.recipe_identity})
    TORCH_CHECK(!v->empty() && v->size() <= 4096 && v->find('\0') == std::string::npos,
                "[rpb-v23 training] nonempty bounded manifest identity required");
}
inline bool same_identity(const TrainingIdentity &a, const TrainingIdentity &b) {
  return a.input_source_identity == b.input_source_identity &&
      a.split_identity == b.split_identity && a.recipe_identity == b.recipe_identity;
}
inline void validate_progress(const TrainingProgress &p) {
  TORCH_CHECK(p.completed_updates >= 0 && p.attempted_batches >= p.completed_updates &&
                  p.attempted_batches < std::numeric_limits<int64_t>::max() &&
                  p.row_seed >= 0 && p.mask_seed >= 0 && p.context_seed >= 0,
              "[rpb-v23 training] invalid completed/attempted/seed counters");
}
inline std::string parameter_order(const ConsecutiveChangeModel &model) {
  std::string text;
  for (const auto &p : model->named_parameters()) text += p.key() + '\n';
  return text;
}
inline std::string optimizer_parameter_keys(const torch::optim::AdamW &optimizer) {
  std::string text;
  for (const auto &p : optimizer.param_groups().front().params())
    text += std::to_string(reinterpret_cast<uintptr_t>(p.unsafeGetTensorImpl())) + '\n';
  return text;
}
// LibTorch maps serialized pointer IDs by group position and silently discards
// orphan states. Bind that actual archived order and reject duplicate/orphan
// keys before invoking its loader, rather than trusting a field inventory.
inline void validate_optimizer_archive(torch::serialize::InputArchive &archive,
    const std::string &expected_keys, int64_t parameter_count) {
  torch::serialize::InputArchive groups, group, states;
  archive.read("param_groups", groups); archive.read("state", states);
  // InputArchive can reuse a defined tensor's TensorImpl when reading again;
  // distinct destinations prevent aliasing/mutating the archive's own fields.
  torch::Tensor group_count; groups.read("param_groups/size", group_count);
  TORCH_CHECK(group_count.scalar_type() == torch::kInt64 && group_count.dim() == 0 && group_count.item<int64_t>() == 1,
              "[rpb-v23 training] serialized single AdamW group required");
  groups.read("param_groups/0", group);
  torch::Tensor parameter_size; group.read("params/size", parameter_size);
  TORCH_CHECK(parameter_size.scalar_type() == torch::kInt64 && parameter_size.dim() == 0 &&
                  parameter_size.item<int64_t>() == parameter_count,
              "[rpb-v23 training] serialized parameter count mismatch");
  std::set<std::string> keys;
  std::string actual_keys;
  for (int64_t j = 0; j < parameter_count; ++j) {
    torch::IValue value; group.read("params/" + std::to_string(j), value);
    TORCH_CHECK(value.isString(), "[rpb-v23 training] serialized parameter key must be text");
    const std::string key = value.toStringRef();
    TORCH_CHECK(!key.empty() && std::all_of(key.begin(), key.end(), [](char c){return c >= '0' && c <= '9';}) &&
                    keys.insert(key).second, "[rpb-v23 training] malformed/duplicate optimizer parameter key");
    actual_keys += key + '\n';
  }
  TORCH_CHECK(actual_keys == expected_keys,
              "[rpb-v23 training] serialized optimizer parameter mapping mismatch");
  for (const auto &key : states.keys())
    TORCH_CHECK(keys.count(key), "[rpb-v23 training] orphan serialized optimizer state");
}

// One fixed recipe and one complete registered parameter group. Inactive
// diagnostic parameters may legitimately have no state; their -1 manifest
// entries must survive reload. Actual states must not reset or exceed updates.
inline std::vector<int64_t> validate_optimizer(const ConsecutiveChangeModel &model,
    const torch::optim::AdamW &optimizer, const TrainingProgress &progress) {
  validate_progress(progress);
  model->validate_state();
  const auto &c = model->config();
  const auto *defaults = dynamic_cast<const torch::optim::AdamWOptions *>(&optimizer.defaults());
  TORCH_CHECK(defaults && *defaults == content_training_adamw_options() &&
                  optimizer.param_groups().size() == 1,
              "[rpb-v23 training] unchanged single-group standard AdamW required");
  const auto &group = optimizer.param_groups().front();
  const auto *options = dynamic_cast<const torch::optim::AdamWOptions *>(&group.options());
  const auto parameters = model->parameters();
  TORCH_CHECK(options && *options == content_training_adamw_options() &&
                  group.params().size() == parameters.size(),
              "[rpb-v23 training] AdamW options/parameter count mismatch");
  const auto named = model->named_parameters();
  std::set<void *> known;
  std::vector<int64_t> steps;
  int64_t maximum_step = 0;
  for (size_t j = 0; j < parameters.size(); ++j) {
    const auto &p = parameters[j];
    auto *key = p.unsafeGetTensorImpl();
    const bool frozen = named[j].key() == "level.base.grouped_odd_relation_projection.weight";
    TORCH_CHECK(group.params()[j].unsafeGetTensorImpl() == key && known.insert(key).second &&
                    p.device() == c.device && p.scalar_type() == torch::kFloat32 &&
                    p.requires_grad() == !frozen,
                "[rpb-v23 training] parameter identity/order/trainability mismatch");
    const auto it = optimizer.state().find(key);
    if (it == optimizer.state().end()) { steps.push_back(-1); continue; }
    const auto *state = dynamic_cast<const torch::optim::AdamWParamState *>(it->second.get());
    TORCH_CHECK(!frozen && state && state->step() > 0 &&
                    state->step() <= progress.completed_updates && !state->max_exp_avg_sq().defined(),
                "[rpb-v23 training] invalid AdamW state/step/frozen parameter");
    for (const auto &moment : {state->exp_avg(), state->exp_avg_sq()})
      TORCH_CHECK(moment.defined() && moment.device() == p.device() &&
                      moment.scalar_type() == p.scalar_type() && moment.sizes() == p.sizes() &&
                      torch::isfinite(moment).all().item<bool>(),
                  "[rpb-v23 training] finite matching CUDA AdamW moments required");
    TORCH_CHECK(state->exp_avg_sq().ge(0).all().item<bool>(),
                "[rpb-v23 training] negative AdamW second moment");
    maximum_step = std::max(maximum_step, state->step());
    steps.push_back(state->step());
  }
  for (const auto &state : optimizer.state())
    TORCH_CHECK(known.count(state.first), "[rpb-v23 training] orphan optimizer state");
  TORCH_CHECK(maximum_step == progress.completed_updates,
              "[rpb-v23 training] completed updates disagree with optimizer step state");
  return steps;
}

// A nested, explicitly typed inference child. Existing inference writers and
// readers remain untouched; this training kind can never masquerade as one.
inline void write_encoder(torch::serialize::OutputArchive &a,
    const ConsecutiveChangeModel &model, const FrozenScaler &scaler, int64_t updates) {
  consecutive_change_checkpoint_detail::write(a,model,scaler,updates);
}

inline int64_t scalar_integer(torch::serialize::InputArchive &a, const char *key) {
  torch::Tensor t; a.read(key, t, true);
  TORCH_CHECK(t.scalar_type() == torch::kInt64 && t.dim() == 0,
              "[rpb-v23 training] malformed integer: ", key);
  return t.item<int64_t>();
}
inline LoadedConsecutiveChangeEncoder read_encoder(torch::serialize::InputArchive &a,
    const Config &expected, const std::string &expected_scaler_identity, int64_t updates) {
  auto out=consecutive_change_checkpoint_detail::read(a,expected.device,&expected,&expected_scaler_identity);
  TORCH_CHECK(out.training_updates==updates,"[rpb-v23 training] child update count mismatch");
  return out;
}
} // namespace training_state_detail

struct LoadedContentTrainingState {
  Config config;
  ConsecutiveChangeModel model{nullptr};
  FrozenScaler scaler;
  std::unique_ptr<torch::optim::AdamW> optimizer;
  TrainingIdentity identity;
  TrainingProgress progress;
};

inline void save_content_training_state(const std::string &path,
    const ConsecutiveChangeModel &model, const FrozenScaler &scaler,
    const torch::optim::AdamW &optimizer, const TrainingIdentity &identity,
    const TrainingProgress &progress) {
  TORCH_CHECK(model, "[rpb-v23 training] model required");
  namespace t = training_state_detail;
  namespace d = configurable_checkpoint_detail;
  t::validate_identity(identity);
  const auto steps = t::validate_optimizer(model, optimizer, progress);
  torch::serialize::OutputArchive a, encoder, optim;
  a.write("artifact_kind", d::text_tensor(kChangeTrainingStateKind), true);
  a.write("counter_contract", d::text_tensor(kChangeTrainingCounterContract), true);
  a.write("input_source_identity", d::text_tensor(identity.input_source_identity), true);
  a.write("split_identity", d::text_tensor(identity.split_identity), true);
  a.write("recipe_identity", d::text_tensor(identity.recipe_identity), true);
  a.write("progress_v1", torch::tensor(std::vector<int64_t>{progress.completed_updates,
      progress.attempted_batches, progress.row_seed, progress.mask_seed, progress.context_seed}, torch::kInt64), true);
  a.write("parameter_order", d::text_tensor(t::parameter_order(model)), true);
  a.write("optimizer_parameter_keys", d::text_tensor(t::optimizer_parameter_keys(optimizer)), true);
  a.write("optimizer_steps_v1", torch::tensor(steps, torch::kInt64), true);
  t::write_encoder(encoder, model, scaler, progress.completed_updates);
  optimizer.save(optim);
  a.write("encoder", encoder); a.write("optimizer", optim);
  d::publish_checkpoint(a, path);
}

inline LoadedContentTrainingState load_content_training_state(const std::string &path,
    const Config &expected, const TrainingIdentity &expected_identity,
    const std::string &expected_scaler_identity) {
  validate_consecutive_change_config(expected);
  namespace t = training_state_detail;
  namespace d = configurable_checkpoint_detail;
  t::validate_identity(expected_identity);
  torch::serialize::InputArchive a;
  a.load_from(path, expected.device);
  TORCH_CHECK(d::read_text(a, "artifact_kind") == kChangeTrainingStateKind &&
                  d::read_text(a, "counter_contract") == kChangeTrainingCounterContract,
              "[rpb-v23 training] explicit training kind/counter contract required");
  TrainingIdentity identity{d::read_text(a, "input_source_identity"), d::read_text(a, "split_identity"),
                            d::read_text(a, "recipe_identity")};
  t::validate_identity(identity);
  TORCH_CHECK(t::same_identity(identity, expected_identity),
              "[rpb-v23 training] input source/split/recipe identity mismatch");
  const auto p = d::read_vector(a, "progress_v1", torch::kInt64, 5);
  const auto x = p.accessor<int64_t, 1>();
  TrainingProgress progress{x[0], x[1], x[2], x[3], x[4]};
  t::validate_progress(progress);
  torch::serialize::InputArchive encoder, optim;
  a.read("encoder", encoder);
  auto child = t::read_encoder(encoder, expected, expected_scaler_identity, progress.completed_updates);
  TORCH_CHECK(d::read_text(a, "parameter_order") == t::parameter_order(child.model),
              "[rpb-v23 training] optimizer parameter-order contract mismatch");
  auto optimizer = std::make_unique<torch::optim::AdamW>(child.model->parameters(), content_training_adamw_options());
  a.read("optimizer", optim);
  t::validate_optimizer_archive(optim, d::read_text(a, "optimizer_parameter_keys"), child.model->parameters().size());
  optimizer->load(optim);
  const auto actual_steps = t::validate_optimizer(child.model, *optimizer, progress);
  const auto declared_steps = d::read_vector(a, "optimizer_steps_v1", torch::kInt64, actual_steps.size());
  TORCH_CHECK(torch::equal(declared_steps, torch::tensor(actual_steps, torch::kInt64)),
              "[rpb-v23 training] missing/reset/remapped optimizer step state");
  child.model->train(); optimizer->zero_grad();
  return {child.config, child.model, child.scaler, std::move(optimizer), identity, progress};
}

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v4
