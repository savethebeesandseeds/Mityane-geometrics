// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once

#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v4/consecutive_change_model.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v4/configurable_spectral_checkpoint.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v4 {
inline constexpr const char *kConsecutiveChangeCheckpointKind="rpb_consecutive_change_inference_checkpoint_v1";
struct LoadedConsecutiveChangeEncoder {
  Config config;ConsecutiveChangeModel model{nullptr};FrozenScaler scaler;int64_t training_updates{0};
};
namespace consecutive_change_checkpoint_detail {
namespace d=configurable_checkpoint_detail;
inline void write(torch::serialize::OutputArchive &a,const ConsecutiveChangeModel &model,
    const FrozenScaler &scaler,int64_t updates) {
  TORCH_CHECK(model && updates>=0,"[rpb-v23 checkpoint] model/nonnegative updates required");
  const auto &c=model->config();validate_consecutive_change_config(c);model->validate_state();
  scaler.validate(configurable_spectral_input_config(c));
  a.write("artifact_kind",d::text_tensor(kConsecutiveChangeCheckpointKind),true);
  a.write("model_tag",d::text_tensor(kConsecutiveChangeModelTag),true);
  a.write("architecture_id",d::text_tensor(kConsecutiveChangeArchitectureId),true);
  a.write("native_layout",d::text_tensor(kConsecutiveChangeLayout),true);
  a.write("dtype",d::text_tensor("float32"),true);
  a.write("training_device",d::text_tensor(c.device.str()),true);
  a.write("config_integer_v1",torch::tensor(d::integers(c),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(c),torch::kFloat64),true);
  a.write("channel_ids",torch::tensor(resolved_channel_ids(c),torch::kInt64),true);
  a.write("change_moments",torch::tensor(kConsecutiveChangeMoments,torch::kInt64),true);
  a.write("change_minimum_count",torch::tensor(kConsecutiveChangeMinimumCount,torch::kInt64),true);
  a.write("change_rms_floor",torch::tensor(kConsecutiveChangeRmsFloor,torch::kFloat64),true);
  a.write("training_updates",torch::tensor(updates,torch::kInt64),true);
  a.write("scaler_identity",d::text_tensor(scaler.identity()),true);
  torch::serialize::OutputArchive m,s;model->save(m);scaler.save(s);a.write("model",m);a.write("scaler",s);
}
inline LoadedConsecutiveChangeEncoder read(torch::serialize::InputArchive &a,torch::Device device,
    const Config *expected=nullptr,const std::string *expected_scaler=nullptr) {
  TORCH_CHECK(device.is_cuda() && d::read_text(a,"artifact_kind")==kConsecutiveChangeCheckpointKind &&
      d::read_text(a,"model_tag")==kConsecutiveChangeModelTag &&
      d::read_text(a,"architecture_id")==kConsecutiveChangeArchitectureId &&
      d::read_text(a,"native_layout")==kConsecutiveChangeLayout &&
      d::read_text(a,"dtype")=="float32" &&
      torch::Device(d::read_text(a,"training_device")).is_cuda(),"[rpb-v23 checkpoint] typed architecture mismatch");
  const auto c=d::read_config(a,device);validate_consecutive_change_config(c);
  if(expected) TORCH_CHECK(d::same_config(c,*expected),"[rpb-v23 checkpoint] configuration mismatch");
  auto integer=[&](const char *key) {
    torch::Tensor t;a.read(key,t,true);TORCH_CHECK(t.scalar_type()==torch::kInt64 && t.dim()==0,
        "[rpb-v23 checkpoint] malformed integer: ",key);return t.item<int64_t>();
  };
  torch::Tensor floor;a.read("change_rms_floor",floor,true);
  TORCH_CHECK(integer("change_moments")==kConsecutiveChangeMoments &&
      integer("change_minimum_count")==kConsecutiveChangeMinimumCount &&
      floor.scalar_type()==torch::kFloat64 && floor.dim()==0 && floor.item<double>()==kConsecutiveChangeRmsFloor,
      "[rpb-v23 checkpoint] change recipe mismatch");
  const auto updates=integer("training_updates");TORCH_CHECK(updates>=0,"[rpb-v23 checkpoint] negative updates");
  const auto scaler_id=d::read_text(a,"scaler_identity");
  if(expected_scaler) TORCH_CHECK(scaler_id==*expected_scaler,"[rpb-v23 checkpoint] TRAIN scaler mismatch");
  LoadedConsecutiveChangeEncoder out{c,ConsecutiveChangeModel(c),{},updates};
  torch::serialize::InputArchive m,s;a.read("model",m);a.read("scaler",s);
  out.model->load(m);out.scaler=FrozenScaler::load(s);out.scaler.validate(configurable_spectral_input_config(c));
  TORCH_CHECK(out.scaler.identity()==scaler_id,"[rpb-v23 checkpoint] scaler content mismatch");
  out.model->eval();return out;
}
} // namespace consecutive_change_checkpoint_detail
inline void save_consecutive_change_checkpoint(const std::string &path,const ConsecutiveChangeModel &model,
    const FrozenScaler &scaler,int64_t updates) {
  torch::serialize::OutputArchive a;consecutive_change_checkpoint_detail::write(a,model,scaler,updates);
  configurable_checkpoint_detail::publish_checkpoint(a,path);
}
inline LoadedConsecutiveChangeEncoder load_consecutive_change_checkpoint(const std::string &path,
    torch::Device device=torch::Device(torch::kCUDA,0),const Config *expected=nullptr) {
  torch::serialize::InputArchive a;a.load_from(path,device);
  return consecutive_change_checkpoint_detail::read(a,device,expected);
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v4
