// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_moments.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_checkpoint.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kEnergyChangeCheckpointKind="rpb_energy_change_inference_checkpoint_v1";
struct LoadedEnergyChangeEncoder {
  EnergyChangeConfig config;EnergyChangeModel model{nullptr};FrozenScaler scaler;
  MomentRmsScaler moment_scaler;int64_t training_updates{0};
};
namespace energy_change_checkpoint_detail {
namespace d=configurable_checkpoint_detail;
using separated_change_checkpoint_detail::integer;
inline std::vector<double> objective(){return{kEnergyChangeAuxiliaryWeight,kEnergyChangeHuberDelta,
    kEnergyChangeSignedWeight,kEnergyChangeEnergyWeight,kEnergyChangeMomentScaleFloor};}
inline void write_objective(torch::serialize::OutputArchive &a){
  a.write("moment_layout",d::text_tensor(kEnergyChangeMomentLayout),true);
  a.write("objective_constants_v1",torch::tensor(objective(),torch::kFloat64),true);
}
inline void read_objective(torch::serialize::InputArchive &a){
  TORCH_CHECK(d::read_text(a,"moment_layout")==kEnergyChangeMomentLayout &&
      torch::equal(d::read_vector(a,"objective_constants_v1",torch::kFloat64,5),torch::tensor(objective(),torch::kFloat64)),
      "[rpb-v25] objective/layout mismatch");
}
inline void write(torch::serialize::OutputArchive &a,const EnergyChangeModel &model,const FrozenScaler &scaler,
    const MomentRmsScaler &moments,int64_t updates){
  TORCH_CHECK(model && updates>=0,"[rpb-v25] model/nonnegative updates required");model->validate_state();
  const auto &s=model->config();const auto &c=s.level;
  scaler.validate(configurable_spectral_input_config(c));moments.validate(c);
  TORCH_CHECK(moments.frozen,"[rpb-v25] frozen TRAIN moment normalizer required");
  a.write("artifact_kind",d::text_tensor(kEnergyChangeCheckpointKind),true);
  a.write("model_tag",d::text_tensor(kEnergyChangeModelTag),true);
  a.write("architecture_id",d::text_tensor(kEnergyChangeArchitectureId),true);
  a.write("native_layout",d::text_tensor(kEnergyChangeLayout),true);
  a.write("dtype",d::text_tensor("float32"),true);a.write("training_device",d::text_tensor(c.device.str()),true);
  a.write("config_integer_v1",torch::tensor(d::integers(c),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(c),torch::kFloat64),true);
  a.write("channel_ids",torch::tensor(resolved_channel_ids(c),torch::kInt64),true);
  a.write("change_width",torch::tensor(s.change_width,torch::kInt64),true);
  a.write("native_width",torch::tensor(energy_change_native_width(s),torch::kInt64),true);
  a.write("change_moments",torch::tensor(kConsecutiveChangeMoments,torch::kInt64),true);
  a.write("change_minimum_count",torch::tensor(kConsecutiveChangeMinimumCount,torch::kInt64),true);
  a.write("change_rms_floor",torch::tensor(kConsecutiveChangeRmsFloor,torch::kFloat64),true);
  a.write("training_updates",torch::tensor(updates,torch::kInt64),true);
  a.write("scaler_identity",d::text_tensor(scaler.identity()),true);
  a.write("moment_source_identity",d::text_tensor(moments.source_identity),true);write_objective(a);
  torch::serialize::OutputArchive m,f,n;model->save(m);scaler.save(f);moments.save(n);
  a.write("model",m);a.write("scaler",f);a.write("moment_scaler",n);
}
inline LoadedEnergyChangeEncoder read(torch::serialize::InputArchive &a,torch::Device device,
    const EnergyChangeConfig *expected=nullptr,const std::string *expected_scaler=nullptr,
    const std::string *expected_moment_source=nullptr){
  TORCH_CHECK(device.is_cuda() && d::read_text(a,"artifact_kind")==kEnergyChangeCheckpointKind &&
      d::read_text(a,"model_tag")==kEnergyChangeModelTag && d::read_text(a,"architecture_id")==kEnergyChangeArchitectureId &&
      d::read_text(a,"native_layout")==kEnergyChangeLayout && d::read_text(a,"dtype")=="float32" &&
      torch::Device(d::read_text(a,"training_device")).is_cuda(),"[rpb-v25] typed architecture mismatch");
  EnergyChangeConfig c{d::read_config(a,device),integer(a,"change_width")};validate_energy_change_config(c);
  TORCH_CHECK(integer(a,"native_width")==energy_change_native_width(c),"[rpb-v25] native width mismatch");
  if(expected)TORCH_CHECK(d::same_config(c.level,expected->level) && c.change_width==expected->change_width,"[rpb-v25] config mismatch");
  torch::Tensor floor;a.read("change_rms_floor",floor,true);
  TORCH_CHECK(integer(a,"change_moments")==kConsecutiveChangeMoments && integer(a,"change_minimum_count")==kConsecutiveChangeMinimumCount &&
      floor.scalar_type()==torch::kFloat64 && floor.dim()==0 && floor.item<double>()==kConsecutiveChangeRmsFloor,"[rpb-v25] bank recipe mismatch");
  read_objective(a);
  const auto updates=integer(a,"training_updates");TORCH_CHECK(updates>=0,"[rpb-v25] negative updates");
  const auto id=d::read_text(a,"scaler_identity"),moment_source=d::read_text(a,"moment_source_identity");
  if(expected_scaler)TORCH_CHECK(id==*expected_scaler,"[rpb-v25] requested input TRAIN scaler mismatch");
  if(expected_moment_source)TORCH_CHECK(moment_source==*expected_moment_source,"[rpb-v25] requested moment TRAIN source mismatch");
  LoadedEnergyChangeEncoder out{c,EnergyChangeModel(c),{},{},updates};
  torch::serialize::InputArchive m,f,n;a.read("model",m);a.read("scaler",f);a.read("moment_scaler",n);
  out.model->load(m);out.scaler=FrozenScaler::load(f);out.moment_scaler=MomentRmsScaler::load(n,c.level,&moment_source);
  out.scaler.validate(configurable_spectral_input_config(c.level));
  TORCH_CHECK(out.scaler.identity()==id,"[rpb-v25] input scaler content mismatch");out.model->eval();return out;
}
} // namespace energy_change_checkpoint_detail
inline void save_energy_change_checkpoint(const std::string &path,const EnergyChangeModel &model,const FrozenScaler &scaler,
    const MomentRmsScaler &moments,int64_t updates){
  torch::serialize::OutputArchive a;energy_change_checkpoint_detail::write(a,model,scaler,moments,updates);
  configurable_checkpoint_detail::publish_checkpoint(a,path);
}
inline LoadedEnergyChangeEncoder load_energy_change_checkpoint(const std::string &path,
    torch::Device device=torch::Device(torch::kCUDA,0),const EnergyChangeConfig *expected=nullptr,
    const std::string *expected_scaler=nullptr,const std::string *expected_moment_source=nullptr){
  torch::serialize::InputArchive a;a.load_from(path,device);
  return energy_change_checkpoint_detail::read(a,device,expected,expected_scaler,expected_moment_source);
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
