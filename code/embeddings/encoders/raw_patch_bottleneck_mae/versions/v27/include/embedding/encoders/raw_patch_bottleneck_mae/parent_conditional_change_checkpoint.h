// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_model.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_checkpoint.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kParentConditionalChangeCheckpointKind="rpb_parent_conditional_change_inference_checkpoint_v1";
struct LoadedParentConditionalChangeEncoder {
  ParentConditionalChangeConfig config;ParentConditionalChangeModel model{nullptr};FrozenScaler scaler;int64_t training_updates{0};
};
namespace parent_conditional_change_checkpoint_detail {
namespace d=configurable_checkpoint_detail;using separated_change_checkpoint_detail::integer;
inline void write_recipe(torch::serialize::OutputArchive &a){
  a.write("bank_layout",d::text_tensor(kParentConditionalChangeBankLayout),true);
  a.write("normalization_recipe",d::text_tensor(kParentConditionalChangeNormalizationRecipe),true);
  a.write("objective",d::text_tensor("unchanged-hierarchical-waveform-huber-only-v1"),true);
  a.write("conditional_map_recipe",d::text_tensor(kParentConditionalMapRecipe),true);
  a.write("normalization_floor",torch::tensor(kParentConditionalChangeScaleFloor,torch::kFloat64),true);
  a.write("normalization_minimum_support",torch::tensor(kParentConditionalChangeSupportCount,torch::kInt64),true);
}
inline void read_recipe(torch::serialize::InputArchive &a){
  torch::Tensor floor;a.read("normalization_floor",floor,true);
  TORCH_CHECK(d::read_text(a,"bank_layout")==kParentConditionalChangeBankLayout &&
      d::read_text(a,"normalization_recipe")==kParentConditionalChangeNormalizationRecipe &&
      d::read_text(a,"objective")=="unchanged-hierarchical-waveform-huber-only-v1" &&
      d::read_text(a,"conditional_map_recipe")==kParentConditionalMapRecipe &&
      integer(a,"normalization_minimum_support")==kParentConditionalChangeSupportCount &&
      floor.dim()==0 && floor.scalar_type()==torch::kFloat64 && floor.item<double>()==kParentConditionalChangeScaleFloor,
      "[rpb-v27] normalization/objective recipe mismatch");
}
inline void write(torch::serialize::OutputArchive &a,const ParentConditionalChangeModel &model,const FrozenScaler &scaler,int64_t updates){
  TORCH_CHECK(model && updates>=0,"[rpb-v27] model/nonnegative updates required");model->validate_state();
  const auto &s=model->config();const auto &c=s.level;const auto &normalizer=model->normalizer();const auto scaler_id=scaler.identity();
  scaler.validate(configurable_spectral_input_config(c));normalizer.validate(c,nullptr,&scaler_id);
  a.write("artifact_kind",d::text_tensor(kParentConditionalChangeCheckpointKind),true);
  a.write("model_tag",d::text_tensor(kParentConditionalChangeModelTag),true);a.write("architecture_id",d::text_tensor(kParentConditionalChangeArchitectureId),true);
  a.write("native_layout",d::text_tensor(kParentConditionalChangeLayout),true);a.write("dtype",d::text_tensor("float32"),true);
  a.write("training_device",d::text_tensor(c.device.str()),true);a.write("config_integer_v1",torch::tensor(d::integers(c),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(c),torch::kFloat64),true);a.write("channel_ids",torch::tensor(resolved_channel_ids(c),torch::kInt64),true);
  a.write("change_width",torch::tensor(s.change_width,torch::kInt64),true);a.write("native_width",torch::tensor(parent_conditional_change_native_width(s),torch::kInt64),true);
  a.write("change_moments",torch::tensor(kConsecutiveChangeMoments,torch::kInt64),true);
  a.write("change_minimum_count",torch::tensor(kConsecutiveChangeMinimumCount,torch::kInt64),true);
  a.write("change_rms_floor",torch::tensor(kConsecutiveChangeRmsFloor,torch::kFloat64),true);
  a.write("training_updates",torch::tensor(updates,torch::kInt64),true);a.write("scaler_identity",d::text_tensor(scaler_id),true);
  a.write("normalizer_source_identity",d::text_tensor(normalizer.source_identity),true);
  a.write("normalizer_identity",d::text_tensor(normalizer.identity()),true);write_recipe(a);
  a.write("conditional_map_source_identity",d::text_tensor(model->conditional_map().bindings.source_identity),true);
  a.write("conditional_map_identity",d::text_tensor(model->conditional_map().identity()),true);
  torch::serialize::OutputArchive m,f,n,k;model->save(m);scaler.save(f);normalizer.save(n);model->conditional_map().save(k);
  a.write("model",m);a.write("scaler",f);a.write("bank_standardizer",n);a.write("conditional_map",k);
}
inline LoadedParentConditionalChangeEncoder read(torch::serialize::InputArchive &a,torch::Device device,
    const ParentConditionalChangeConfig *expected=nullptr,const std::string *expected_scaler=nullptr,
    const std::string *expected_normalizer_source=nullptr,const std::string *expected_map_source=nullptr,
    const std::string *expected_normalizer_identity=nullptr,const std::string *expected_map_identity=nullptr){
  TORCH_CHECK(device.is_cuda() && d::read_text(a,"artifact_kind")==kParentConditionalChangeCheckpointKind &&
      d::read_text(a,"model_tag")==kParentConditionalChangeModelTag && d::read_text(a,"architecture_id")==kParentConditionalChangeArchitectureId &&
      d::read_text(a,"native_layout")==kParentConditionalChangeLayout && d::read_text(a,"dtype")=="float32" &&
      torch::Device(d::read_text(a,"training_device")).is_cuda(),"[rpb-v27] typed architecture mismatch");
  ParentConditionalChangeConfig c{d::read_config(a,device),integer(a,"change_width")};validate_parent_conditional_change_config(c);
  TORCH_CHECK(integer(a,"native_width")==parent_conditional_change_native_width(c),"[rpb-v27] native width mismatch");
  if(expected)TORCH_CHECK(d::same_config(c.level,expected->level) && c.change_width==expected->change_width,"[rpb-v27] config mismatch");
  torch::Tensor floor;a.read("change_rms_floor",floor,true);
  TORCH_CHECK(integer(a,"change_moments")==kConsecutiveChangeMoments && integer(a,"change_minimum_count")==kConsecutiveChangeMinimumCount &&
      floor.dim()==0 && floor.scalar_type()==torch::kFloat64 && floor.item<double>()==kConsecutiveChangeRmsFloor,"[rpb-v27] bank recipe mismatch");
  read_recipe(a);const auto updates=integer(a,"training_updates");TORCH_CHECK(updates>=0,"[rpb-v27] negative updates");
  const auto scaler_id=d::read_text(a,"scaler_identity"),source=d::read_text(a,"normalizer_source_identity"),normalizer_id=d::read_text(a,"normalizer_identity");
  if(expected_scaler)TORCH_CHECK(scaler_id==*expected_scaler,"[rpb-v27] requested input scaler mismatch");
  if(expected_normalizer_source)TORCH_CHECK(source==*expected_normalizer_source,"[rpb-v27] requested normalizer source mismatch");
  if(expected_normalizer_identity)TORCH_CHECK(normalizer_id==*expected_normalizer_identity,"[rpb-v27] requested normalizer content mismatch");
  const auto map_source=d::read_text(a,"conditional_map_source_identity"),map_id=d::read_text(a,"conditional_map_identity");
  if(expected_map_source)TORCH_CHECK(map_source==*expected_map_source,"[rpb-v27] requested map source mismatch");
  if(expected_map_identity)TORCH_CHECK(map_id==*expected_map_identity,"[rpb-v27] requested map content mismatch");
  torch::serialize::InputArchive m,f,n,k;a.read("model",m);a.read("scaler",f);a.read("bank_standardizer",n);a.read("conditional_map",k);
  auto scaler=FrozenScaler::load(f);scaler.validate(configurable_spectral_input_config(c.level));
  TORCH_CHECK(scaler.identity()==scaler_id,"[rpb-v27] input scaler content mismatch");
  auto normalizer=BankStandardizer::load(n,c.level,&source,&scaler_id);
  TORCH_CHECK(normalizer.identity()==normalizer_id,"[rpb-v27] normalizer envelope content mismatch");
  auto map=ParentConditionalMap::load(k,c.level);TORCH_CHECK(map.bindings.source_identity==map_source && map.bindings.input_scaler_identity==scaler_id &&
      map.bindings.normalizer_identity==normalizer_id && map.identity()==map_id,"[rpb-v27] map envelope binding mismatch");
  auto model=ParentConditionalChangeModel(c,normalizer,map);model->load(m);model->eval();return{c,model,scaler,updates};
}
} // namespace parent_conditional_change_checkpoint_detail
inline void save_parent_conditional_change_checkpoint(const std::string &path,const ParentConditionalChangeModel &model,const FrozenScaler &scaler,int64_t updates){
  torch::serialize::OutputArchive a;parent_conditional_change_checkpoint_detail::write(a,model,scaler,updates);configurable_checkpoint_detail::publish_checkpoint(a,path);
}
inline LoadedParentConditionalChangeEncoder load_parent_conditional_change_checkpoint(const std::string &path,
    torch::Device device=torch::Device(torch::kCUDA,0),const ParentConditionalChangeConfig *expected=nullptr,
    const std::string *expected_scaler=nullptr,const std::string *expected_normalizer_source=nullptr,const std::string *expected_map_source=nullptr,
    const std::string *expected_normalizer_identity=nullptr,const std::string *expected_map_identity=nullptr){
  torch::serialize::InputArchive a;a.load_from(path,device);
  return parent_conditional_change_checkpoint_detail::read(a,device,expected,expected_scaler,expected_normalizer_source,expected_map_source,expected_normalizer_identity,expected_map_identity);
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
