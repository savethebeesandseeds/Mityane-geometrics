// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_model.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_checkpoint.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kParentModulatedDecoderCheckpointKind="rpb_parent_modulated_decoder_inference_checkpoint_v1";
struct LoadedParentModulatedDecoderEncoder {
  ParentModulatedDecoderConfig config;ParentModulatedDecoderModel model{nullptr};FrozenScaler scaler;int64_t training_updates{0};
};
namespace parent_modulated_decoder_checkpoint_detail {
namespace d=configurable_checkpoint_detail;using separated_change_checkpoint_detail::integer;
inline void write_recipe(torch::serialize::OutputArchive &a){
  a.write("decoder_recipe",d::text_tensor(kParentModulatedDecoderRecipe),true);
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
  TORCH_CHECK(d::read_text(a,"decoder_recipe")==kParentModulatedDecoderRecipe &&
      d::read_text(a,"bank_layout")==kRelativeChangeBankLayout &&
      d::read_text(a,"normalization_recipe")==kRelativeChangeNormalizationRecipe &&
      d::read_text(a,"objective")=="unchanged-hierarchical-waveform-huber-only-v1" &&
      d::read_text(a,"bank_family")==kRelativeChangeBankFamily && d::read_text(a,"raw_contract")==kRelativeChangeRawContract &&
      d::read_text(a,"increment_recipe")==kRelativeChangeIncrementRecipe &&
      integer(a,"normalization_minimum_support")==kRelativeChangeSupportCount &&
      floor.dim()==0 && floor.scalar_type()==torch::kFloat64 && floor.item<double>()==kRelativeChangeScaleFloor,
      "[rpb-v29] normalization/objective recipe mismatch");
}
inline void write(torch::serialize::OutputArchive &a,const ParentModulatedDecoderModel &model,const FrozenScaler &scaler,int64_t updates){
  TORCH_CHECK(model && updates>=0,"[rpb-v29] model/nonnegative updates required");model->validate_state();
  TORCH_CHECK(updates!=0 || (model->encoder->change_decoder.eq(0).all().item<bool>() && model->waveform_gain.eq(0).all().item<bool>()),
      "[rpb-v29] initial archive requires zero decoder/gain");
  TORCH_CHECK(updates!=1 || model->waveform_gain.eq(0).all().item<bool>(),"[rpb-v29] first-update gain must remain zero");
  const auto &s=model->config();const auto &c=s.level;const auto &normalizer=model->normalizer();const auto scaler_id=scaler.identity();
  scaler.validate(configurable_spectral_input_config(c));normalizer.validate(c,nullptr,&scaler_id);
  a.write("artifact_kind",d::text_tensor(kParentModulatedDecoderCheckpointKind),true);
  a.write("model_tag",d::text_tensor(kParentModulatedDecoderModelTag),true);a.write("architecture_id",d::text_tensor(kParentModulatedDecoderArchitectureId),true);
  a.write("native_layout",d::text_tensor(kParentModulatedDecoderLayout),true);a.write("dtype",d::text_tensor("float32"),true);
  a.write("training_device",d::text_tensor(c.device.str()),true);a.write("config_integer_v1",torch::tensor(d::integers(c),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(c),torch::kFloat64),true);a.write("channel_ids",torch::tensor(resolved_channel_ids(c),torch::kInt64),true);
  a.write("change_width",torch::tensor(s.change_width,torch::kInt64),true);a.write("native_width",torch::tensor(parent_modulated_decoder_native_width(s),torch::kInt64),true);
  a.write("change_moments",torch::tensor(kConsecutiveChangeMoments,torch::kInt64),true);
  a.write("change_minimum_count",torch::tensor(kConsecutiveChangeMinimumCount,torch::kInt64),true);
  a.write("change_rms_floor",torch::tensor(kConsecutiveChangeRmsFloor,torch::kFloat64),true);
  a.write("training_updates",torch::tensor(updates,torch::kInt64),true);a.write("scaler_identity",d::text_tensor(scaler_id),true);
  a.write("normalizer_source_identity",d::text_tensor(normalizer.source_identity),true);
  a.write("normalizer_identity",d::text_tensor(normalizer.identity()),true);write_recipe(a);
  torch::serialize::OutputArchive m,f,n;model->save(m);scaler.save(f);normalizer.save(n);
  a.write("model",m);a.write("scaler",f);a.write("bank_standardizer",n);
}
inline LoadedParentModulatedDecoderEncoder read(torch::serialize::InputArchive &a,torch::Device device,
    const ParentModulatedDecoderConfig *expected=nullptr,const std::string *expected_scaler=nullptr,
    const std::string *expected_normalizer_source=nullptr,const std::string *expected_normalizer_identity=nullptr){
  TORCH_CHECK(device.is_cuda() && d::read_text(a,"artifact_kind")==kParentModulatedDecoderCheckpointKind &&
      d::read_text(a,"model_tag")==kParentModulatedDecoderModelTag && d::read_text(a,"architecture_id")==kParentModulatedDecoderArchitectureId &&
      d::read_text(a,"native_layout")==kParentModulatedDecoderLayout && d::read_text(a,"dtype")=="float32" &&
      torch::Device(d::read_text(a,"training_device")).is_cuda(),"[rpb-v29] typed architecture mismatch");
  ParentModulatedDecoderConfig c{d::read_config(a,device),integer(a,"change_width")};validate_parent_modulated_decoder_config(c);
  TORCH_CHECK(integer(a,"native_width")==parent_modulated_decoder_native_width(c),"[rpb-v29] native width mismatch");
  if(expected)TORCH_CHECK(d::same_config(c.level,expected->level) && c.change_width==expected->change_width,"[rpb-v29] config mismatch");
  torch::Tensor floor;a.read("change_rms_floor",floor,true);
  TORCH_CHECK(integer(a,"change_moments")==kConsecutiveChangeMoments && integer(a,"change_minimum_count")==kConsecutiveChangeMinimumCount &&
      floor.dim()==0 && floor.scalar_type()==torch::kFloat64 && floor.item<double>()==kConsecutiveChangeRmsFloor,"[rpb-v29] bank recipe mismatch");
  read_recipe(a);const auto updates=integer(a,"training_updates");TORCH_CHECK(updates>=0,"[rpb-v29] negative updates");
  const auto scaler_id=d::read_text(a,"scaler_identity"),source=d::read_text(a,"normalizer_source_identity"),normalizer_id=d::read_text(a,"normalizer_identity");
  if(expected_scaler)TORCH_CHECK(scaler_id==*expected_scaler,"[rpb-v29] requested input scaler mismatch");
  if(expected_normalizer_source)TORCH_CHECK(source==*expected_normalizer_source,"[rpb-v29] requested normalizer source mismatch");
  if(expected_normalizer_identity)TORCH_CHECK(normalizer_id==*expected_normalizer_identity,"[rpb-v29] requested normalizer content mismatch");
  torch::serialize::InputArchive m,f,n;a.read("model",m);a.read("scaler",f);a.read("bank_standardizer",n);
  auto scaler=FrozenScaler::load(f);scaler.validate(configurable_spectral_input_config(c.level));
  TORCH_CHECK(scaler.identity()==scaler_id,"[rpb-v29] input scaler content mismatch");
  auto normalizer=RelativeBankStandardizer::load(n,c.level,&source,&scaler_id);
  TORCH_CHECK(normalizer.identity()==normalizer_id,"[rpb-v29] normalizer envelope content mismatch");
  auto model=ParentModulatedDecoderModel(c,normalizer);model->load(m);
  TORCH_CHECK(updates!=0 || (model->encoder->change_decoder.eq(0).all().item<bool>() && model->waveform_gain.eq(0).all().item<bool>()),
      "[rpb-v29] initial archive decoder/gain mismatch");
  TORCH_CHECK(updates!=1 || model->waveform_gain.eq(0).all().item<bool>(),"[rpb-v29] first-update archived gain mismatch");
  model->eval();return{c,model,scaler,updates};
}
} // namespace parent_modulated_decoder_checkpoint_detail
inline void save_parent_modulated_decoder_checkpoint(const std::string &path,const ParentModulatedDecoderModel &model,const FrozenScaler &scaler,int64_t updates){
  torch::serialize::OutputArchive a;parent_modulated_decoder_checkpoint_detail::write(a,model,scaler,updates);configurable_checkpoint_detail::publish_checkpoint(a,path);
}
inline LoadedParentModulatedDecoderEncoder load_parent_modulated_decoder_checkpoint(const std::string &path,
    torch::Device device=torch::Device(torch::kCUDA,0),const ParentModulatedDecoderConfig *expected=nullptr,
    const std::string *expected_scaler=nullptr,const std::string *expected_normalizer_source=nullptr,const std::string *expected_normalizer_identity=nullptr){
  torch::serialize::InputArchive a;a.load_from(path,device);
  return parent_modulated_decoder_checkpoint_detail::read(a,device,expected,expected_scaler,expected_normalizer_source,expected_normalizer_identity);
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
