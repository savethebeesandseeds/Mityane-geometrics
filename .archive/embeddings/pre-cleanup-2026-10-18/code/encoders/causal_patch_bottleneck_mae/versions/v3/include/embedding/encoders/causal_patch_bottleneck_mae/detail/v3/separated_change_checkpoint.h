// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/separated_change_model.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v3/configurable_spectral_checkpoint.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3 {
inline constexpr const char *kSeparatedChangeCheckpointKind="rpb_separated_change_inference_checkpoint_v1";
struct LoadedSeparatedChangeEncoder{SeparatedChangeConfig config;SeparatedChangeModel model{nullptr};FrozenScaler scaler;int64_t training_updates{0};};
namespace separated_change_checkpoint_detail {
namespace d=configurable_checkpoint_detail;
inline int64_t integer(torch::serialize::InputArchive &a,const char *key){
  torch::Tensor t;a.read(key,t,true);TORCH_CHECK(t.scalar_type()==torch::kInt64 && t.dim()==0,"[rpb-v24] scalar integer required: ",key);return t.item<int64_t>();
}
inline void write(torch::serialize::OutputArchive &a,const SeparatedChangeModel &model,const FrozenScaler &scaler,int64_t updates){
  TORCH_CHECK(model && updates>=0,"[rpb-v24] model/nonnegative updates required");model->validate_state();
  const auto &s=model->config();const auto &c=s.level;scaler.validate(configurable_spectral_input_config(c));
  a.write("artifact_kind",d::text_tensor(kSeparatedChangeCheckpointKind),true);
  a.write("model_tag",d::text_tensor(kSeparatedChangeModelTag),true);
  a.write("architecture_id",d::text_tensor(kSeparatedChangeArchitectureId),true);
  a.write("native_layout",d::text_tensor(kSeparatedChangeLayout),true);
  a.write("dtype",d::text_tensor("float32"),true);a.write("training_device",d::text_tensor(c.device.str()),true);
  a.write("config_integer_v1",torch::tensor(d::integers(c),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(c),torch::kFloat64),true);
  a.write("channel_ids",torch::tensor(resolved_channel_ids(c),torch::kInt64),true);
  a.write("change_width",torch::tensor(s.change_width,torch::kInt64),true);
  a.write("native_width",torch::tensor(separated_change_native_width(s),torch::kInt64),true);
  a.write("change_moments",torch::tensor(kConsecutiveChangeMoments,torch::kInt64),true);
  a.write("change_minimum_count",torch::tensor(kConsecutiveChangeMinimumCount,torch::kInt64),true);
  a.write("change_rms_floor",torch::tensor(kConsecutiveChangeRmsFloor,torch::kFloat64),true);
  a.write("training_updates",torch::tensor(updates,torch::kInt64),true);
  a.write("scaler_identity",d::text_tensor(scaler.identity()),true);
  torch::serialize::OutputArchive m,f;model->save(m);scaler.save(f);a.write("model",m);a.write("scaler",f);
}
inline LoadedSeparatedChangeEncoder read(torch::serialize::InputArchive &a,torch::Device device,
    const SeparatedChangeConfig *expected=nullptr,const std::string *expected_scaler=nullptr){
  TORCH_CHECK(device.is_cuda() && d::read_text(a,"artifact_kind")==kSeparatedChangeCheckpointKind &&
      d::read_text(a,"model_tag")==kSeparatedChangeModelTag && d::read_text(a,"architecture_id")==kSeparatedChangeArchitectureId &&
      d::read_text(a,"native_layout")==kSeparatedChangeLayout && d::read_text(a,"dtype")=="float32" &&
      torch::Device(d::read_text(a,"training_device")).is_cuda(),"[rpb-v24] typed architecture mismatch");
  SeparatedChangeConfig c{d::read_config(a,device),integer(a,"change_width")};validate_separated_change_config(c);
  TORCH_CHECK(integer(a,"native_width")==separated_change_native_width(c),"[rpb-v24] native width mismatch");
  if(expected)TORCH_CHECK(d::same_config(c.level,expected->level) && c.change_width==expected->change_width,"[rpb-v24] config mismatch");
  torch::Tensor floor;a.read("change_rms_floor",floor,true);
  TORCH_CHECK(integer(a,"change_moments")==kConsecutiveChangeMoments && integer(a,"change_minimum_count")==kConsecutiveChangeMinimumCount &&
      floor.scalar_type()==torch::kFloat64 && floor.dim()==0 && floor.item<double>()==kConsecutiveChangeRmsFloor,"[rpb-v24] bank recipe mismatch");
  const auto updates=integer(a,"training_updates");TORCH_CHECK(updates>=0,"[rpb-v24] negative updates");
  const auto id=d::read_text(a,"scaler_identity");if(expected_scaler)TORCH_CHECK(id==*expected_scaler,"[rpb-v24] requested TRAIN scaler mismatch");
  LoadedSeparatedChangeEncoder out{c,SeparatedChangeModel(c),{},updates};
  torch::serialize::InputArchive m,f;a.read("model",m);a.read("scaler",f);out.model->load(m);out.scaler=FrozenScaler::load(f);
  out.scaler.validate(configurable_spectral_input_config(c.level));TORCH_CHECK(out.scaler.identity()==id,"[rpb-v24] scaler content mismatch");
  out.model->eval();return out;
}
} // namespace separated_change_checkpoint_detail
inline void save_separated_change_checkpoint(const std::string &path,const SeparatedChangeModel &m,const FrozenScaler &s,int64_t updates){
  torch::serialize::OutputArchive a;separated_change_checkpoint_detail::write(a,m,s,updates);configurable_checkpoint_detail::publish_checkpoint(a,path);
}
inline LoadedSeparatedChangeEncoder load_separated_change_checkpoint(const std::string &path,torch::Device device=torch::Device(torch::kCUDA,0),
    const SeparatedChangeConfig *expected=nullptr){torch::serialize::InputArchive a;a.load_from(path,device);return separated_change_checkpoint_detail::read(a,device,expected);}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v3
