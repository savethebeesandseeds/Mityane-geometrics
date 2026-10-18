// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_frozen_level.h"

namespace embedding::encoders::raw_patch_bottleneck_mae::parent_conditional_change_training {
using consecutive_change_training::TrainingIdentity;
using consecutive_change_training::TrainingProgress;
using consecutive_change_training::training_counter_seed;
using consecutive_change_training::content_training_adamw_options;
namespace common=consecutive_change_training::training_state_detail;
namespace d=configurable_checkpoint_detail;
inline constexpr const char *kTrainingStateKind="rpb_parent_conditional_change_training_state_v1";
inline void warm_start_from_separated(const ParentConditionalChangeModel &model,const SeparatedChangeModel &initial){
  TORCH_CHECK(model && initial,"[rpb-v27 warm start] models required");model->validate_state();initial->validate_state();
  TORCH_CHECK(d::same_config(model->config().level,initial->config().level) &&
      model->config().change_width==initial->config().change_width && initial->change_decoder.eq(0).all().item<bool>() &&
      model->encoder->change_decoder.eq(0).all().item<bool>(),
      "[rpb-v27 warm start] matching initial geometry and zero decoders required");
  torch::NoGradGuard guard;
  const auto copy=[](const auto &from,const auto &to){
    TORCH_CHECK(from.size()==to.size(),"[rpb-v27 warm start] state count mismatch");
    for(const auto &p:from){const auto *value=to.find(p.key());
      TORCH_CHECK(value && value->sizes()==p.value().sizes() && value->device()==p.value().device() &&
          value->scalar_type()==p.value().scalar_type(),"[rpb-v27 warm start] state mismatch: ",p.key());value->copy_(p.value());}
  };
  copy(initial->named_parameters(),model->encoder->named_parameters());copy(initial->named_buffers(),model->encoder->named_buffers());
  model->validate_state();
}
class FrozenLevelGuard {
public:
  explicit FrozenLevelGuard(const ParentConditionalChangeModel &model):guard_(model->encoder){model->validate_state();}
  void discard_gradients(const ParentConditionalChangeModel &model,const torch::optim::AdamW &optimizer)const{
    model->validate_state();guard_.discard_gradients(model->encoder,optimizer);
  }
  void validate(const ParentConditionalChangeModel &model,const torch::optim::AdamW &optimizer)const{
    model->validate_state();guard_.validate(model->encoder,optimizer);
  }
private:separated_change_training::FrozenLevelGuard guard_;
};
inline std::string parameter_order(const ParentConditionalChangeModel &model){
  std::string out;for(const auto &p:model->named_parameters())out+=p.key()+'\n';return out;
}
inline std::vector<int64_t> validate_optimizer(const ParentConditionalChangeModel &model,const torch::optim::AdamW &optimizer,
    const TrainingProgress &progress){
  common::validate_progress(progress);model->validate_state();
  const auto *defaults=dynamic_cast<const torch::optim::AdamWOptions *>(&optimizer.defaults());
  TORCH_CHECK(defaults && *defaults==content_training_adamw_options() && optimizer.param_groups().size()==1,
      "[rpb-v27 training] unchanged single-group AdamW required");
  const auto &group=optimizer.param_groups().front();const auto *options=dynamic_cast<const torch::optim::AdamWOptions *>(&group.options());
  const auto named=model->named_parameters();
  TORCH_CHECK(options && *options==content_training_adamw_options() && group.params().size()==named.size(),"[rpb-v27 training] group/options mismatch");
  std::set<void *> known;std::vector<int64_t> steps;int64_t maximum=0,active_count=0;
  for(size_t j=0;j<named.size();++j){
    const auto &p=named[j].value();auto *key=p.unsafeGetTensorImpl();
    const bool protected_level=named[j].key().rfind("encoder.level.",0)==0;
    TORCH_CHECK(group.params()[j].unsafeGetTensorImpl()==key && known.insert(key).second,"[rpb-v27 training] parameter mapping mismatch");
    TORCH_CHECK(protected_level || (p.requires_grad() && (named[j].key()=="encoder.change_projection" ||
        named[j].key()=="encoder.change_decoder")),"[rpb-v27 training] unexpected active parameter");
    if(!protected_level)++active_count;
    const auto it=optimizer.state().find(key);
    TORCH_CHECK(!protected_level || (it==optimizer.state().end() && !p.grad().defined()),"[rpb-v27 training] protected gradient/state forbidden");
    if(it==optimizer.state().end()){
      TORCH_CHECK(protected_level || progress.completed_updates==0,"[rpb-v27 training] missing active optimizer state");steps.push_back(-1);continue;}
    const auto *s=dynamic_cast<const torch::optim::AdamWParamState *>(it->second.get());
    TORCH_CHECK(!protected_level && s && s->step()==progress.completed_updates && s->step()>0 &&
        !s->max_exp_avg_sq().defined(),"[rpb-v27 training] active step/flags mismatch");
    for(const auto &m:{s->exp_avg(),s->exp_avg_sq()})TORCH_CHECK(m.defined() && m.device()==p.device() &&
        m.scalar_type()==p.scalar_type() && m.sizes()==p.sizes() && torch::isfinite(m).all().item<bool>(),"[rpb-v27 training] finite matching moments required");
    TORCH_CHECK(s->exp_avg_sq().ge(0).all().item<bool>(),"[rpb-v27 training] negative second moment");
    maximum=std::max(maximum,s->step());steps.push_back(s->step());
  }
  for(const auto &s:optimizer.state())TORCH_CHECK(known.count(s.first),"[rpb-v27 training] orphan optimizer state");
  TORCH_CHECK(active_count==2 && maximum==progress.completed_updates,"[rpb-v27 training] active-count/completed-step mismatch");return steps;
}
struct LoadedTrainingState {
  ParentConditionalChangeConfig config;ParentConditionalChangeModel model{nullptr};FrozenScaler scaler;
  std::unique_ptr<torch::optim::AdamW> optimizer;TrainingIdentity identity;TrainingProgress progress;
};
inline void save_training_state(const std::string &path,const ParentConditionalChangeModel &model,const FrozenScaler &scaler,
    const torch::optim::AdamW &optimizer,const TrainingIdentity &identity,const TrainingProgress &progress){
  common::validate_identity(identity);const auto steps=validate_optimizer(model,optimizer,progress);
  torch::serialize::OutputArchive a,e,o;
  a.write("artifact_kind",d::text_tensor(kTrainingStateKind),true);
  a.write("counter_contract",d::text_tensor(consecutive_change_training::kChangeTrainingCounterContract),true);
  a.write("input_source_identity",d::text_tensor(identity.input_source_identity),true);
  a.write("split_identity",d::text_tensor(identity.split_identity),true);a.write("recipe_identity",d::text_tensor(identity.recipe_identity),true);
  a.write("progress_v1",torch::tensor(std::vector<int64_t>{progress.completed_updates,progress.attempted_batches,
      progress.row_seed,progress.mask_seed,progress.context_seed},torch::kInt64),true);
  a.write("parameter_order",d::text_tensor(parameter_order(model)),true);
  a.write("optimizer_parameter_keys",d::text_tensor(common::optimizer_parameter_keys(optimizer)),true);
  a.write("optimizer_steps_v1",torch::tensor(steps,torch::kInt64),true);
  parent_conditional_change_checkpoint_detail::write(e,model,scaler,progress.completed_updates);optimizer.save(o);
  a.write("encoder",e);a.write("optimizer",o);d::publish_checkpoint(a,path);
}
inline LoadedTrainingState load_training_state(const std::string &path,const ParentConditionalChangeConfig &expected,
    const TrainingIdentity &identity,const std::string &scaler_identity,const std::string &normalizer_source_identity,const std::string &map_source_identity,
    const std::string *expected_normalizer_identity=nullptr,const std::string *expected_map_identity=nullptr){
  validate_parent_conditional_change_config(expected);common::validate_identity(identity);
  torch::serialize::InputArchive a;a.load_from(path,expected.level.device);
  TORCH_CHECK(d::read_text(a,"artifact_kind")==kTrainingStateKind && d::read_text(a,"counter_contract")==
      consecutive_change_training::kChangeTrainingCounterContract,"[rpb-v27 training] explicit kind/counter required");
  TrainingIdentity actual{d::read_text(a,"input_source_identity"),d::read_text(a,"split_identity"),d::read_text(a,"recipe_identity")};
  common::validate_identity(actual);TORCH_CHECK(common::same_identity(actual,identity),"[rpb-v27 training] manifest identity mismatch");
  const auto v=d::read_vector(a,"progress_v1",torch::kInt64,5);const auto x=v.accessor<int64_t,1>();
  TrainingProgress progress{x[0],x[1],x[2],x[3],x[4]};common::validate_progress(progress);
  torch::serialize::InputArchive e,o;a.read("encoder",e);
  auto child=parent_conditional_change_checkpoint_detail::read(e,expected.level.device,&expected,&scaler_identity,&normalizer_source_identity,&map_source_identity,expected_normalizer_identity,expected_map_identity);
  TORCH_CHECK(child.training_updates==progress.completed_updates && d::read_text(a,"parameter_order")==parameter_order(child.model),
      "[rpb-v27 training] child update/parameter order mismatch");
  auto optimizer=std::make_unique<torch::optim::AdamW>(child.model->parameters(),content_training_adamw_options());
  a.read("optimizer",o);common::validate_optimizer_archive(o,d::read_text(a,"optimizer_parameter_keys"),child.model->parameters().size());
  optimizer->load(o);const auto steps=validate_optimizer(child.model,*optimizer,progress);
  TORCH_CHECK(torch::equal(d::read_vector(a,"optimizer_steps_v1",torch::kInt64,steps.size()),torch::tensor(steps,torch::kInt64)),
      "[rpb-v27 training] missing/reset/remapped steps");child.model->train();optimizer->zero_grad();
  return{child.config,child.model,child.scaler,std::move(optimizer),actual,progress};
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae::parent_conditional_change_training
