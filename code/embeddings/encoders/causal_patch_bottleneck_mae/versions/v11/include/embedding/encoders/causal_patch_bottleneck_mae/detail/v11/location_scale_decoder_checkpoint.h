// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v11/location_scale_decoder_model.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v11/separated_change_checkpoint.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v11 {
inline constexpr const char *kLocationScaleDecoderCheckpointKind="cpb_v11_location_scale_decoder_payload_v1";
struct LoadedLocationScaleDecoderEncoder {
  LocationScaleDecoderConfig config;LocationScaleDecoderModel model{nullptr};FrozenScaler scaler;int64_t training_updates{0};
};
namespace location_scale_decoder_checkpoint_detail {
namespace d=configurable_checkpoint_detail;using separated_change_checkpoint_detail::integer;
inline void write_temporal_pooling(torch::serialize::OutputArchive &a,const Config &logical){
  const auto partition=temporal_pooling_partition(configurable_spectral_backbone_config(logical));
  a.write("temporal_pooling_policy",d::text_tensor(kTemporalPoolingPolicy),true);
  a.write("temporal_pooling_partition_v1",torch::tensor(partition,torch::kInt64),true);
  a.write("temporal_pooling_mix_f64",torch::tensor(temporal_pooling_mix(),torch::kFloat64),true);
  a.write("empty_local_bin_policy",d::text_tensor(kEmptyLocalBinPolicy),true);
  a.write("temporal_moment_policy",d::text_tensor(kTemporalMomentPolicy),true);
  a.write("temporal_export_layout",d::text_tensor(kTemporalExportLayout),true);
  a.write("temporal_moment_partition_v1",torch::tensor(temporal_moment_partition(),torch::kInt64),true);
}
inline void read_temporal_pooling(torch::serialize::InputArchive &a,const Config &logical){
  const auto partition=temporal_pooling_partition(configurable_spectral_backbone_config(logical));
  TORCH_CHECK(d::read_text(a,"temporal_pooling_policy")==kTemporalPoolingPolicy &&
      torch::equal(d::read_vector(a,"temporal_pooling_partition_v1",torch::kInt64,5),torch::tensor(partition,torch::kInt64)) &&
      torch::equal(d::read_vector(a,"temporal_pooling_mix_f64",torch::kFloat64,2),torch::tensor(temporal_pooling_mix(),torch::kFloat64)) &&
      d::read_text(a,"empty_local_bin_policy")==kEmptyLocalBinPolicy &&
      d::read_text(a,"temporal_moment_policy")==kTemporalMomentPolicy &&
      d::read_text(a,"temporal_export_layout")==kTemporalExportLayout &&
      torch::equal(d::read_vector(a,"temporal_moment_partition_v1",torch::kInt64,2),torch::tensor(temporal_moment_partition(),torch::kInt64)),
      "[cpb-v11] exact policy/config/partition/mix/zero-empty/moment/export-layout contract required");
}
inline void write_objective(torch::serialize::OutputArchive &a){
  a.write("objective_policy",d::text_tensor(kLocationScaleDecoderObjectivePolicy),true);
  a.write("mean_huber_delta_f64",torch::tensor(kLocationScaleDecoderMeanHuberDelta,torch::kFloat64),true);
  a.write("loss_coefficients_f64",torch::tensor(std::vector<double>{kLocationScaleDecoderMeanWeight,kLocationScaleDecoderVarianceWeight},torch::kFloat64),true);
  a.write("variance_error_detached",torch::tensor(int64_t{1},torch::kInt64),true);
  a.write("direct_gradient_routes",d::text_tensor(kLocationScaleDecoderDirectGradientRoutes),true);
  a.write("reporting_nll_backpropagated",torch::tensor(int64_t{0},torch::kInt64),true);
}
inline void read_objective(torch::serialize::InputArchive &a){
  torch::Tensor delta;a.read("mean_huber_delta_f64",delta,true);
  TORCH_CHECK(d::read_text(a,"objective_policy")==kLocationScaleDecoderObjectivePolicy &&
      delta.dim()==0 && delta.scalar_type()==torch::kFloat64 && delta.item<double>()==kLocationScaleDecoderMeanHuberDelta &&
      torch::equal(d::read_vector(a,"loss_coefficients_f64",torch::kFloat64,2),
          torch::tensor(std::vector<double>{kLocationScaleDecoderMeanWeight,kLocationScaleDecoderVarianceWeight},torch::kFloat64)) &&
      integer(a,"variance_error_detached")==1 && integer(a,"reporting_nll_backpropagated")==0 &&
      d::read_text(a,"direct_gradient_routes")==kLocationScaleDecoderDirectGradientRoutes,
      "[cpb-v11] exact objective/delta/unit-weights/detached-error/direct-routing/report-only-NLL required");
}
inline void write_recipe(torch::serialize::OutputArchive &a){
  write_objective(a);
  a.write("decoder_recipe",d::text_tensor(kLocationScaleDecoderRecipe),true);
  a.write("gradient_policy",d::text_tensor(kLocationScaleDecoderGradientPolicy),true);
  a.write("link_constants_f32",torch::tensor(location_scale_decoder_constants(),torch::kFloat32),true);
  a.write("softplus_parameters_f64",torch::tensor(std::vector<double>{kLocationScaleDecoderSoftplusBeta,kLocationScaleDecoderSoftplusThreshold},torch::kFloat64),true);
  a.write("bank_layout",d::text_tensor(kRelativeChangeBankLayout),true);
  a.write("normalization_recipe",d::text_tensor(kRelativeChangeNormalizationRecipe),true);
  a.write("reporting_likelihood",d::text_tensor(kLocationScaleDecoderLikelihood),true);
  a.write("bank_family",d::text_tensor(kRelativeChangeBankFamily),true);
  a.write("raw_contract",d::text_tensor(kRelativeChangeRawContract),true);
  a.write("increment_recipe",d::text_tensor(kRelativeChangeIncrementRecipe),true);
  a.write("normalization_floor",torch::tensor(kRelativeChangeScaleFloor,torch::kFloat64),true);
  a.write("normalization_minimum_support",torch::tensor(kRelativeChangeSupportCount,torch::kInt64),true);
}
inline void read_recipe(torch::serialize::InputArchive &a){
  read_objective(a);
  torch::Tensor floor;a.read("normalization_floor",floor,true);
  TORCH_CHECK(d::read_text(a,"decoder_recipe")==kLocationScaleDecoderRecipe &&
      d::read_text(a,"gradient_policy")==kLocationScaleDecoderGradientPolicy &&
      torch::equal(d::read_vector(a,"link_constants_f32",torch::kFloat32,3),torch::tensor(location_scale_decoder_constants(),torch::kFloat32)) &&
      torch::equal(d::read_vector(a,"softplus_parameters_f64",torch::kFloat64,2),torch::tensor(std::vector<double>{kLocationScaleDecoderSoftplusBeta,kLocationScaleDecoderSoftplusThreshold},torch::kFloat64)) &&
      d::read_text(a,"bank_layout")==kRelativeChangeBankLayout &&
      d::read_text(a,"normalization_recipe")==kRelativeChangeNormalizationRecipe &&
      d::read_text(a,"reporting_likelihood")==kLocationScaleDecoderLikelihood &&
      d::read_text(a,"bank_family")==kRelativeChangeBankFamily && d::read_text(a,"raw_contract")==kRelativeChangeRawContract &&
      d::read_text(a,"increment_recipe")==kRelativeChangeIncrementRecipe &&
      integer(a,"normalization_minimum_support")==kRelativeChangeSupportCount &&
      floor.dim()==0 && floor.scalar_type()==torch::kFloat64 && floor.item<double>()==kRelativeChangeScaleFloor,
      "[rpb-v30] normalization/objective recipe mismatch");
}
inline void write(torch::serialize::OutputArchive &a,const LocationScaleDecoderModel &model,const FrozenScaler &scaler,int64_t updates){
  TORCH_CHECK(model && updates>=0,"[rpb-v30] model/nonnegative updates required");model->validate_state();
  const auto &s=model->config();const auto &c=s.level;const auto &normalizer=model->normalizer();const auto scaler_id=scaler.identity();
  scaler.validate(configurable_spectral_input_config(c));normalizer.validate(c,nullptr,&scaler_id);
  a.write("artifact_kind",d::text_tensor(kLocationScaleDecoderCheckpointKind),true);
  a.write("model_tag",d::text_tensor(kLocationScaleDecoderModelTag),true);a.write("architecture_id",d::text_tensor(kLocationScaleDecoderArchitectureId),true);
  a.write("native_layout",d::text_tensor(kLocationScaleDecoderLayout),true);a.write("dtype",d::text_tensor("float32"),true);
  a.write("training_device",d::text_tensor(c.device.str()),true);a.write("config_integer_v1",torch::tensor(d::integers(c),torch::kInt64),true);
  a.write("config_number_v1",torch::tensor(d::numbers(c),torch::kFloat64),true);a.write("channel_ids",torch::tensor(resolved_channel_ids(c),torch::kInt64),true);
  a.write("change_width",torch::tensor(s.change_width,torch::kInt64),true);a.write("mean_width",torch::tensor(s.change_width,torch::kInt64),true);
  a.write("uncertainty_width",torch::tensor(s.uncertainty_width,torch::kInt64),true);a.write("native_width",torch::tensor(location_scale_decoder_native_width(s),torch::kInt64),true);
  a.write("change_moments",torch::tensor(kConsecutiveChangeMoments,torch::kInt64),true);
  a.write("change_minimum_count",torch::tensor(kConsecutiveChangeMinimumCount,torch::kInt64),true);
  a.write("change_rms_floor",torch::tensor(kConsecutiveChangeRmsFloor,torch::kFloat64),true);
  a.write("training_updates",torch::tensor(updates,torch::kInt64),true);a.write("scaler_identity",d::text_tensor(scaler_id),true);
  a.write("normalizer_source_identity",d::text_tensor(normalizer.source_identity),true);
  a.write("normalizer_identity",d::text_tensor(normalizer.identity()),true);write_recipe(a);write_temporal_pooling(a,c);
  torch::serialize::OutputArchive m,f,n;model->save(m);scaler.save(f);normalizer.save(n);
  a.write("model",m);a.write("scaler",f);a.write("bank_standardizer",n);
}
inline LoadedLocationScaleDecoderEncoder read(torch::serialize::InputArchive &a,torch::Device device,
    const LocationScaleDecoderConfig *expected=nullptr,const std::string *expected_scaler=nullptr,
    const std::string *expected_normalizer_source=nullptr,const std::string *expected_normalizer_identity=nullptr){
  TORCH_CHECK(device.is_cuda() && d::read_text(a,"artifact_kind")==kLocationScaleDecoderCheckpointKind &&
      d::read_text(a,"model_tag")==kLocationScaleDecoderModelTag && d::read_text(a,"architecture_id")==kLocationScaleDecoderArchitectureId &&
      d::read_text(a,"native_layout")==kLocationScaleDecoderLayout && d::read_text(a,"dtype")=="float32" &&
      torch::Device(d::read_text(a,"training_device")).is_cuda(),"[rpb-v30] typed architecture mismatch");
  LocationScaleDecoderConfig c{d::read_config(a,device),integer(a,"change_width"),integer(a,"uncertainty_width")};validate_location_scale_decoder_config(c);
  TORCH_CHECK(integer(a,"mean_width")==c.change_width && integer(a,"native_width")==location_scale_decoder_native_width(c),"[cpb-v11] explicit mean/uncertainty/native width mismatch");
  if(expected)TORCH_CHECK(d::same_config(c.level,expected->level) && c.change_width==expected->change_width && c.uncertainty_width==expected->uncertainty_width,"[cpb-v11] config mismatch");
  torch::Tensor floor;a.read("change_rms_floor",floor,true);
  TORCH_CHECK(integer(a,"change_moments")==kConsecutiveChangeMoments && integer(a,"change_minimum_count")==kConsecutiveChangeMinimumCount &&
      floor.dim()==0 && floor.scalar_type()==torch::kFloat64 && floor.item<double>()==kConsecutiveChangeRmsFloor,"[rpb-v30] bank recipe mismatch");
  read_recipe(a);read_temporal_pooling(a,c.level);const auto updates=integer(a,"training_updates");TORCH_CHECK(updates>=0,"[rpb-v30] negative updates");
  const auto scaler_id=d::read_text(a,"scaler_identity"),source=d::read_text(a,"normalizer_source_identity"),normalizer_id=d::read_text(a,"normalizer_identity");
  if(expected_scaler)TORCH_CHECK(scaler_id==*expected_scaler,"[rpb-v30] requested input scaler mismatch");
  if(expected_normalizer_source)TORCH_CHECK(source==*expected_normalizer_source,"[rpb-v30] requested normalizer source mismatch");
  if(expected_normalizer_identity)TORCH_CHECK(normalizer_id==*expected_normalizer_identity,"[rpb-v30] requested normalizer content mismatch");
  torch::serialize::InputArchive m,f,n;a.read("model",m);a.read("scaler",f);a.read("bank_standardizer",n);
  auto scaler=FrozenScaler::load(f);scaler.validate(configurable_spectral_input_config(c.level));
  TORCH_CHECK(scaler.identity()==scaler_id,"[rpb-v30] input scaler content mismatch");
  auto normalizer=RelativeBankStandardizer::load(n,c.level,&source,&scaler_id);
  TORCH_CHECK(normalizer.identity()==normalizer_id,"[rpb-v30] normalizer envelope content mismatch");
  auto model=LocationScaleDecoderModel(c,normalizer);model->load(m);
  model->eval();return{c,model,scaler,updates};
}
} // namespace location_scale_decoder_checkpoint_detail
inline void save_location_scale_decoder_checkpoint(const std::string &path,const LocationScaleDecoderModel &model,const FrozenScaler &scaler,int64_t updates){
  torch::serialize::OutputArchive a;location_scale_decoder_checkpoint_detail::write(a,model,scaler,updates);configurable_checkpoint_detail::publish_checkpoint(a,path);
}
inline LoadedLocationScaleDecoderEncoder load_location_scale_decoder_checkpoint(const std::string &path,
    torch::Device device=torch::Device(torch::kCUDA,0),const LocationScaleDecoderConfig *expected=nullptr,
    const std::string *expected_scaler=nullptr,const std::string *expected_normalizer_source=nullptr,const std::string *expected_normalizer_identity=nullptr){
  torch::serialize::InputArchive a;a.load_from(path,device);
  return location_scale_decoder_checkpoint_detail::read(a,device,expected,expected_scaler,expected_normalizer_source,expected_normalizer_identity);
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v11
