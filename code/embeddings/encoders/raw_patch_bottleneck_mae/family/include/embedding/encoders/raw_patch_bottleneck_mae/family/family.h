// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Waajacu. See the repository LICENSE.
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_settings.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/content_ordered_relation_checkpoint.h"
#include "cuwacunu/artifacts/v1/sidecar.h"

namespace embedding::encoders::raw_patch_bottleneck_mae::family {

using Config = raw_patch_bottleneck_mae::Config;
using Input = raw_patch_bottleneck_mae::Input;
using FrozenScaler = raw_patch_bottleneck_mae::FrozenScaler;
using Model = raw_patch_bottleneck_mae::ContentOrderedRelationModel;
inline constexpr const char *kFamilyApi = "rpb-family-v1";

inline void validate_config(const Config &c) { validate_content_ordered_relation_config(c); }
inline Config read_settings(const std::filesystem::path &path) {
  return read_content_ordered_relation_settings(path);
}

struct Dimensions {
  int64_t channel_pairs, spectral_width, learned_width, relation_bank_width;
  int64_t padded_history, patch_count;
};
inline Dimensions dimensions(const Config &c) {
  family::validate_config(c);
  const auto padded = configurable_spectral_backbone_config(c).history_length;
  return {configurable_spectral_pair_count(c), configurable_spectral_width(c),
      configurable_spectral_shape_width(c), content_ordered_relation_bank_width(c),
      padded, padded / c.patch_length};
}
struct ParameterInfo {
  std::string name;
  std::vector<int64_t> shape;
  int64_t count;
  bool trainable;
};
struct ParameterInventory {
  int64_t total{0}, trainable{0}, frozen{0};
  std::vector<ParameterInfo> parameters;
};
inline ParameterInventory parameter_inventory(const Model &model) {
  TORCH_CHECK(model, "[rpb family] model required");
  model->validate_state();
  ParameterInventory out;
  for (const auto &p : model->named_parameters()) {
    const auto n = p.value().numel();
    TORCH_CHECK(out.total <= std::numeric_limits<int64_t>::max() - n,
                "[rpb family] parameter count overflow");
    out.parameters.push_back({p.key(), p.value().sizes().vec(), n, p.value().requires_grad()});
    out.total += n;
    (p.value().requires_grad() ? out.trainable : out.frozen) += n;
  }
  return out;
}

inline Model fresh(const Config &c, const FrozenScaler &scaler, int64_t initialization_seed) {
  family::validate_config(c);
  scaler.validate(configurable_spectral_input_config(c));
  TORCH_CHECK(initialization_seed >= 0, "[rpb family] nonnegative initialization seed required");
  torch::manual_seed(static_cast<uint64_t>(initialization_seed));
  return Model(c);
}

// Inference over an already admitted endpoint window. This is not prefix-causal.
// Raw input may be CPU float32/64; preprocessing precedes CUDA-float32 execution.
inline EncodeOutput encode(Model &model, const FrozenScaler &scaler, const Input &raw) {
  TORCH_CHECK(model, "[rpb family] model required");
  model->validate_state();
  torch::NoGradGuard guard;
  const bool previous_mode = model->is_training();
  struct RestoreMode { Model &model; bool mode; ~RestoreMode() { model->train(mode); } } restore{model,previous_mode};
  model->eval();
  return model->encode(transform_configurable_spectral_input(scaler, raw, model->config()));
}
inline void save_inference(const std::string &path, const Model &model,
                           const FrozenScaler &scaler, int64_t updates) {
  save_content_ordered_relation_checkpoint(path, model, scaler, updates);
}
inline LoadedContentOrderedRelationEncoder load_inference(const std::string &path,
    torch::Device device = torch::Device(torch::kCUDA,0), const Config *expected = nullptr) {
  return load_content_ordered_relation_checkpoint(path, device, expected);
}

namespace identity_detail {
using cuwacunu::artifacts::v1::Sha256;
using cuwacunu::artifacts::v1::hash_field;
inline void tensor(Sha256 &hash, const torch::Tensor &value) {
  if (!value.defined()) { hash_field(hash,"undefined"); return; }
  const auto cpu = value.detach().to(torch::kCPU).contiguous();
  hash_field(hash,"tensor-v1");
  hash_field(hash,std::to_string(static_cast<int64_t>(cpu.scalar_type())));
  hash_field(hash,std::to_string(cpu.dim()));
  for (const auto d : cpu.sizes()) hash_field(hash,std::to_string(d));
  hash_field(hash,std::string_view(static_cast<const char *>(cpu.const_data_ptr()),
                                  static_cast<size_t>(cpu.numel()*cpu.element_size())));
}
}
// Numerical SHA256 witnesses bind tensor dtype/shape/bytes, independent of
// archive serialization and pointer IDs. They do not authenticate application data.
inline std::string tensor_identity(const torch::Tensor &value) {
  identity_detail::Sha256 hash;
  identity_detail::hash_field(hash,"rpb-family-tensor-v1");
  identity_detail::tensor(hash,value);
  return hash.finish();
}
inline std::string config_identity(const Config &c) {
  family::validate_config(c);
  identity_detail::Sha256 hash;
  identity_detail::hash_field(hash,"rpb-family-config-v1");
  identity_detail::tensor(hash,torch::tensor(configurable_checkpoint_detail::integers(c),torch::kInt64));
  identity_detail::tensor(hash,torch::tensor(configurable_checkpoint_detail::numbers(c),torch::kFloat64));
  identity_detail::tensor(hash,torch::tensor(resolved_channel_ids(c),torch::kInt64));
  identity_detail::hash_field(hash,c.device.str());
  identity_detail::hash_field(hash,kContentOrderedRelationArchitectureId);
  identity_detail::hash_field(hash,kContentOrderedRelationLayout);
  return hash.finish();
}
inline std::string model_identity(const Model &model) {
  TORCH_CHECK(model, "[rpb family] model required");
  model->validate_state();
  identity_detail::Sha256 hash;
  identity_detail::hash_field(hash,"rpb-family-model-v1");
  identity_detail::hash_field(hash,config_identity(model->config()));
  for (const auto &p : model->named_parameters()) {
    identity_detail::hash_field(hash,"parameter");
    identity_detail::hash_field(hash,p.key());
    identity_detail::hash_field(hash,p.value().requires_grad()?"trainable":"frozen");
    identity_detail::tensor(hash,p.value());
  }
  for (const auto &p : model->named_buffers()) {
    identity_detail::hash_field(hash,"buffer");
    identity_detail::hash_field(hash,p.key());
    identity_detail::tensor(hash,p.value());
  }
  return hash.finish();
}
inline std::string gradient_identity(const Model &model) {
  TORCH_CHECK(model, "[rpb family] model required");
  identity_detail::Sha256 hash;
  identity_detail::hash_field(hash,"rpb-family-gradient-v1");
  for (const auto &p : model->named_parameters()) {
    identity_detail::hash_field(hash,p.key());
    identity_detail::tensor(hash,p.value().grad());
  }
  return hash.finish();
}
inline std::string optimizer_identity(const Model &model, const torch::optim::AdamW &optimizer) {
  TORCH_CHECK(model && optimizer.param_groups().size()==1,
              "[rpb family] model and single AdamW group required");
  const auto *options=dynamic_cast<const torch::optim::AdamWOptions *>(&optimizer.param_groups().front().options());
  TORCH_CHECK(options,"[rpb family] AdamW options required");
  identity_detail::Sha256 hash;
  identity_detail::hash_field(hash,"rpb-family-adamw-v1");
  const auto [beta1,beta2]=options->betas();
  identity_detail::tensor(hash,torch::tensor({options->lr(),options->weight_decay(),beta1,beta2,options->eps()},torch::kFloat64));
  identity_detail::hash_field(hash,options->amsgrad()?"amsgrad":"standard");
  const auto parameters=model->named_parameters();
  TORCH_CHECK(optimizer.param_groups().front().params().size()==parameters.size(),
              "[rpb family] optimizer parameter count mismatch");
  size_t active=0;
  for (size_t j=0;j<parameters.size();++j) {
    const auto &p=parameters[j];
    TORCH_CHECK(optimizer.param_groups().front().params()[j].unsafeGetTensorImpl()==p.value().unsafeGetTensorImpl(),
                "[rpb family] optimizer parameter order mismatch");
    identity_detail::hash_field(hash,p.key());
    const auto found=optimizer.state().find(p.value().unsafeGetTensorImpl());
    if(found==optimizer.state().end()) { identity_detail::hash_field(hash,"inactive"); continue; }
    ++active;
    const auto *state=dynamic_cast<const torch::optim::AdamWParamState *>(found->second.get());
    TORCH_CHECK(state,"[rpb family] AdamW state required");
    identity_detail::hash_field(hash,std::to_string(state->step()));
    identity_detail::tensor(hash,state->exp_avg());
    identity_detail::tensor(hash,state->exp_avg_sq());
    identity_detail::tensor(hash,state->max_exp_avg_sq());
  }
  TORCH_CHECK(active==optimizer.state().size(),"[rpb family] orphan optimizer state cannot have a numerical identity");
  return hash.finish();
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae::family
