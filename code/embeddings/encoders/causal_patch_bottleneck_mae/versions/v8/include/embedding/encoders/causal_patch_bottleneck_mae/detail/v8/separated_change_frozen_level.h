// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v8/separated_change_training_state.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v8/content_ordered_relation_checkpoint.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v8::separated_change_training {

// Copy values into the existing registered child: replacing its holder would
// detach the public child from the module used by parameters()/save().
inline void warm_start_level(const SeparatedChangeModel &model,
    const ContentOrderedRelationModel &parent) {
  TORCH_CHECK(model && parent, "[frozen level] models required");
  model->validate_state(); parent->validate_state();
  TORCH_CHECK(configurable_checkpoint_detail::same_config(model->config().level,parent->config()) &&
      model->change_decoder.eq(0).all().item<bool>(),
      "[frozen level] matching config and zero-start residual decoder required");
  torch::NoGradGuard guard;
  const auto copy=[](const auto &from,const auto &to) {
    TORCH_CHECK(from.size()==to.size(),"[frozen level] state count mismatch");
    for(const auto &item:from) {
      const auto *destination=to.find(item.key());
      TORCH_CHECK(destination && destination->sizes()==item.value().sizes() &&
          destination->device()==item.value().device() &&
          destination->scalar_type()==item.value().scalar_type(),
          "[frozen level] state contract mismatch: ",item.key());
      destination->copy_(item.value());
    }
  };
  copy(parent->named_parameters(),model->level->named_parameters());
  copy(parent->named_buffers(),model->level->named_buffers());
  model->validate_state();
}

// Train the change projection alone while preserving existing model/archive
// trainability contracts. Undefined gradients (not zero gradients) keep AdamW
// from allocating moments or applying weight decay to the protected child.
// Call discard_gradients after backward and before clipping/step; validate
// at checkpoints and segment boundaries. The decoder still propagates the
// reconstruction loss to the change projection during backward.
class FrozenLevelGuard {
public:
  explicit FrozenLevelGuard(const SeparatedChangeModel &model) {
    model->validate_state();
    for(const auto &p:model->level->named_parameters())
      parameters_.push_back({p.key(),p.value().unsafeGetTensorImpl(),
          p.value().detach().clone(),p.value().requires_grad()});
    for(const auto &b:model->level->named_buffers())
      buffers_.push_back({b.key(),b.value().unsafeGetTensorImpl(),
          b.value().detach().clone(),b.value().requires_grad()});
  }
  void discard_gradients(const SeparatedChangeModel &model,
      const torch::optim::AdamW &optimizer) const {
    const auto named=model->level->named_parameters();
    TORCH_CHECK(named.size()==parameters_.size(),"[frozen level] changed child");
    for(const auto &p:parameters_) {
      const auto *value=named.find(p.name);
      TORCH_CHECK(value && value->unsafeGetTensorImpl()==p.identity &&
          value->requires_grad()==p.trainable && !optimizer.state().count(p.identity),
          "[frozen level] changed registration/flags or existing optimizer state: ",p.name);
      value->mutable_grad()=torch::Tensor();
    }
  }
  void validate(const SeparatedChangeModel &model,
      const torch::optim::AdamW &optimizer) const {
    const auto verify=[](const auto &saved,const auto &actual) {
      TORCH_CHECK(saved.size()==actual.size(),"[frozen level] state count changed");
      for(const auto &p:saved) {
        const auto *value=actual.find(p.name);
        TORCH_CHECK(value && value->unsafeGetTensorImpl()==p.identity &&
            value->requires_grad()==p.trainable && torch::equal(*value,p.value),
            "[frozen level] protected state changed: ",p.name);
      }
    };
    verify(parameters_,model->level->named_parameters());
    verify(buffers_,model->level->named_buffers());
    for(const auto &p:parameters_) {
      const auto value=model->level->named_parameters()[p.name];
      TORCH_CHECK(!value.grad().defined() && !optimizer.state().count(p.identity),
          "[frozen level] gradient/optimizer state on protected parameter: ",p.name);
    }
  }
private:
  struct State { std::string name;void *identity;torch::Tensor value;bool trainable; };
  std::vector<State> parameters_,buffers_;
};

} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v8::separated_change_training
