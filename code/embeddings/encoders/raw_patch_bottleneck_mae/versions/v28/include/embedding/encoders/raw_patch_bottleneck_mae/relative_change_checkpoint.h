// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/relative_change_model.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_checkpoint.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kRelativeChangeCheckpointKind="rpb_relative_change_inference_checkpoint_v1";
struct LoadedRelativeChangeEncoder {
  RelativeChangeConfig config;RelativeChangeModel model{nullptr};FrozenScaler scaler;int64_t training_updates{0};
};
namespace relative_change_checkpoint_detail {
namespace d=configurable_checkpoint_detail;using separated_change_checkpoint_detail::integer;
inline void write_recipe(torch::serialize::OutputArchive &a){
  a.write("bank_layout",d::text_tensor(kRelativeChangeBankLayout),true);
  a.write("normalization_recipe",d::text_tensor(kRelativeChangeNormalizationRecipe),true);
  a.write("objective",d::text_tensor("unchanged-hierarchical-waveform-huber-only-v1"),true);
  a.write("bank_family",d::text_tensor(kRelativeChangeBankFamily),true);
  a.write("raw_contract",d::text_tensor(kRelativeChangeRawContract),true);
  a.write("increment_recipe",d::text_tensor(kRelativeChangeIncrementRecipe),true);
  a.write("normalization_floor",torch::tensor(kRelativeChangeScaleFloor,torch::kFloat64),true);
  a.write("normalization_minimum_support",torch::tensor(kRelativeChangeSupportCount,torch::kInt64),true);
}
inline void read_recipe(torch::serialize::InputArchive &a){
  torch::Tensor floor;a.read("normalization_floor",floor,true);
  TORCH_CHECK(d::read_text(a,"bank_layout")==kRelativeChangeBankLayout &&
      d::read_text(a,"normalization_recipe")==kRelativeChangeNormalizationRecipe &&
      d::read_text(a,"objective")=="unchanged-hierarchical-waveform-huber-only-v1" &&
      d::read_text(a,"bank_family")==kRelativeChangeBankFamily && d::read_text(a,"raw_contract")==kRelativeChangeRawContract &&
      d::read_text(a,"increment_recipe")==kRelativeChangeIncrementRecipe &&
      integer(a,"normalization_minimum_support")==kRelativeChangeSupportCount &&
      floor.dim()==0 && floor.scalar_type()==torch::kFloat64 && floor.item<double>()==kRelativeChangeScaleFloor,
      "[rpb-v28] normalization/objective recipe mismatch");
}
inline void write(torch::serialize::OutputArchive &a,const RelativeChangeModel &model,const FrozenScaler &scaler,int64_t updates){
  TORCH_CHECK(model && updates>=0,"[rpb-v28] model/nonnegative updates required");model->validate_state();
  const auto &s=model->config();const auto &c=s.level;const auto &normalizer=model->normalizer();const auto scaler_id=scaler.identity();
  scaler.validate(configurable_spectral_input_config(c));normalizer.validate(c,nullptr,&scaler_id);
  a.write("artifact_kind",d::text_tensor(kRelativeChangeCheckpointKind),true);
  a.write("model_tag",d::text_tensor(kRelativeChangeModelTag),true);a.write("architecture_id",d::text_tensor(kRelativeChangeArchitectureId),true);
  a.write("native_layout",d::text_tensor(kRelativeChangeLayout),true);a.write("dtype",d::text_tensor("float32"),true);
  a.write("training_device",d::text_tensor(c.device.str()),true);a.write("config_integer_v1",torch::tensor(d::integers(c),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(c),torch::kFloat64),true);a.write("channel_ids",torch::tensor(resolved_channel_ids(c),torch::kInt64),true);
  a.write("change_width",torch::tensor(s.change_width,torch::kInt64),true);a.write("native_width",torch::tensor(relative_change_native_width(s),torch::kInt64),true);
  a.write("change_moments",torch::tensor(kConsecutiveChangeMoments,torch::kInt64),true);
  a.write("change_minimum_count",torch::tensor(kConsecutiveChangeMinimumCount,torch::kInt64),true);
  a.write("change_rms_floor",torch::tensor(kConsecutiveChangeRmsFloor,torch::kFloat64),true);
  a.write("training_updates",torch::tensor(updates,torch::kInt64),true);a.write("scaler_identity",d::text_tensor(scaler_id),true);
  a.write("normalizer_source_identity",d::text_tensor(normalizer.source_identity),true);
  a.write("normalizer_identity",d::text_tensor(normalizer.identity()),true);write_recipe(a);
  torch::serialize::OutputArchive m,f,n;model->save(m);scaler.save(f);normalizer.save(n);
  a.write("model",m);a.write("scaler",f);a.write("bank_standardizer",n);
}
inline LoadedRelativeChangeEncoder read(torch::serialize::InputArchive &a,torch::Device device,
    const RelativeChangeConfig *expected=nullptr,const std::string *expected_scaler=nullptr,
    const std::string *expected_normalizer_source=nullptr,const std::string *expected_normalizer_identity=nullptr){
  TORCH_CHECK(device.is_cuda() && d::read_text(a,"artifact_kind")==kRelativeChangeCheckpointKind &&
      d::read_text(a,"model_tag")==kRelativeChangeModelTag && d::read_text(a,"architecture_id")==kRelativeChangeArchitectureId &&
      d::read_text(a,"native_layout")==kRelativeChangeLayout && d::read_text(a,"dtype")=="float32" &&
      torch::Device(d::read_text(a,"training_device")).is_cuda(),"[rpb-v28] typed architecture mismatch");
  RelativeChangeConfig c{d::read_config(a,device),integer(a,"change_width")};validate_relative_change_config(c);
  TORCH_CHECK(integer(a,"native_width")==relative_change_native_width(c),"[rpb-v28] native width mismatch");
  if(expected)TORCH_CHECK(d::same_config(c.level,expected->level) && c.change_width==expected->change_width,"[rpb-v28] config mismatch");
  torch::Tensor floor;a.read("change_rms_floor",floor,true);
  TORCH_CHECK(integer(a,"change_moments")==kConsecutiveChangeMoments && integer(a,"change_minimum_count")==kConsecutiveChangeMinimumCount &&
      floor.dim()==0 && floor.scalar_type()==torch::kFloat64 && floor.item<double>()==kConsecutiveChangeRmsFloor,"[rpb-v28] bank recipe mismatch");
  read_recipe(a);const auto updates=integer(a,"training_updates");TORCH_CHECK(updates>=0,"[rpb-v28] negative updates");
  const auto scaler_id=d::read_text(a,"scaler_identity"),source=d::read_text(a,"normalizer_source_identity"),normalizer_id=d::read_text(a,"normalizer_identity");
  if(expected_scaler)TORCH_CHECK(scaler_id==*expected_scaler,"[rpb-v28] requested input scaler mismatch");
  if(expected_normalizer_source)TORCH_CHECK(source==*expected_normalizer_source,"[rpb-v28] requested normalizer source mismatch");
  if(expected_normalizer_identity)TORCH_CHECK(normalizer_id==*expected_normalizer_identity,"[rpb-v28] requested normalizer content mismatch");
  torch::serialize::InputArchive m,f,n;a.read("model",m);a.read("scaler",f);a.read("bank_standardizer",n);
  auto scaler=FrozenScaler::load(f);scaler.validate(configurable_spectral_input_config(c.level));
  TORCH_CHECK(scaler.identity()==scaler_id,"[rpb-v28] input scaler content mismatch");
  auto normalizer=RelativeBankStandardizer::load(n,c.level,&source,&scaler_id);
  TORCH_CHECK(normalizer.identity()==normalizer_id,"[rpb-v28] normalizer envelope content mismatch");
  auto model=RelativeChangeModel(c,normalizer);model->load(m);model->eval();return{c,model,scaler,updates};
}
} // namespace relative_change_checkpoint_detail
inline void save_relative_change_checkpoint(const std::string &path,const RelativeChangeModel &model,const FrozenScaler &scaler,int64_t updates){
  torch::serialize::OutputArchive a;relative_change_checkpoint_detail::write(a,model,scaler,updates);configurable_checkpoint_detail::publish_checkpoint(a,path);
}
inline LoadedRelativeChangeEncoder load_relative_change_checkpoint(const std::string &path,
    torch::Device device=torch::Device(torch::kCUDA,0),const RelativeChangeConfig *expected=nullptr,
    const std::string *expected_scaler=nullptr,const std::string *expected_normalizer_source=nullptr,const std::string *expected_normalizer_identity=nullptr){
  torch::serialize::InputArchive a;a.load_from(path,device);
  return relative_change_checkpoint_detail::read(a,device,expected,expected_scaler,expected_normalizer_source,expected_normalizer_identity);
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
