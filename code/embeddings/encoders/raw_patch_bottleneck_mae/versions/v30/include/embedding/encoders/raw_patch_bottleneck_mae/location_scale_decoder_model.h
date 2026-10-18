// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/location_scale_decoder_config.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/relative_change_normalizer.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_model.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
struct LocationScaleDecoderTrace {
  torch::Tensor parent_mean,canonical_residual,mean,linear_score,score,u,u0,variance_CF,variance,eps32,c32,coefficient32;
};
struct LocationScaleNllOutput {
  torch::Tensor loss,cell_nll,target_counts,channel_loss,example_loss,eligible_channels,eligible_examples;
  int64_t eligible_channel_count{0},eligible_example_count{0},target_cell_count{0};
};
inline LocationScaleNllOutput location_scale_decoder_nll(const torch::Tensor &mean,const torch::Tensor &variance,
    const torch::Tensor &target,const torch::Tensor &support,const torch::Tensor &eligible){
  TORCH_CHECK(mean.defined() && mean.dim()==4 && mean.is_cuda() && mean.scalar_type()==torch::kFloat32 &&
      variance.sizes()==mean.sizes() && target.sizes()==mean.sizes() && support.sizes()==mean.sizes() &&
      eligible.sizes()==mean.sizes().slice(0,2) && variance.device()==mean.device() && target.device()==mean.device() &&
      support.device()==mean.device() && eligible.device()==mean.device() && variance.scalar_type()==torch::kFloat32 &&
      target.scalar_type()==torch::kFloat32 && support.scalar_type()==torch::kBool && eligible.scalar_type()==torch::kBool,
      "[rpb-v30] matching CUDA F32 distributions/targets and bool support/eligibility required");
  TORCH_CHECK(torch::isfinite(mean.masked_select(support)).all().item<bool>() &&
      torch::isfinite(target.masked_select(support)).all().item<bool>() &&
      torch::isfinite(variance.masked_select(support)).all().item<bool>() &&
      variance.masked_select(support).gt(0).all().item<bool>(),"[rpb-v30] finite supported mean/target and positive variance required");
  LocationScaleNllOutput out;out.target_counts=support.sum(std::vector<int64_t>{2,3});
  TORCH_CHECK(torch::equal(out.target_counts.gt(0),eligible),"[rpb-v30] eligibility must exactly match fixed target support");
  out.eligible_channels=eligible;out.eligible_examples=eligible.any(1);
  out.target_cell_count=out.target_counts.sum().item<int64_t>();
  out.eligible_channel_count=eligible.sum().item<int64_t>();out.eligible_example_count=out.eligible_examples.sum().item<int64_t>();
  // Sanitize before every arithmetic operation, then promote BEFORE subtraction/square/division/log.
  const auto safe_mean=torch::where(support,mean,torch::zeros_like(mean)).to(torch::kFloat64);
  const auto safe_target=torch::where(support,target.detach(),torch::zeros_like(target)).to(torch::kFloat64);
  const auto safe_variance=torch::where(support,variance,torch::ones_like(variance)).to(torch::kFloat64);
  const auto error=safe_mean-safe_target;
  out.cell_nll=.5*(error.square()/safe_variance+safe_variance.log());
  out.channel_loss=out.cell_nll.sum(std::vector<int64_t>{2,3})/out.target_counts.clamp_min(1).to(torch::kFloat64);
  out.example_loss=torch::where(eligible,out.channel_loss,torch::zeros_like(out.channel_loss)).sum(1)/
      eligible.sum(1).clamp_min(1).to(torch::kFloat64);
  out.loss=out.eligible_example_count?out.example_loss.masked_select(out.eligible_examples).mean():out.example_loss.sum();
  TORCH_CHECK(torch::isfinite(out.cell_nll).all().item<bool>() && torch::isfinite(out.loss).item<bool>(),"[rpb-v30] finite CUDA F64 Gaussian NLL required");
  return out;
}
struct LocationScaleForwardTrace {
  ForwardOutput output;LocationScaleDecoderTrace distribution;LocationScaleNllOutput nll;
  torch::Tensor targets,target_mask,visible;
};
struct LocationScaleDecoderModelImpl:torch::nn::Module {
  LocationScaleDecoderModelImpl(const LocationScaleDecoderConfig &c,const RelativeBankStandardizer &normalizer):
      config_(c),normalizer_(normalizer.clone()){
    validate_location_scale_decoder_config(c);normalizer_.validate(c.level);
    TORCH_CHECK(normalizer_.frozen,"[rpb-v30] frozen relative-bank calibration required");
    normalizer_identity_=normalizer_.identity();encoder=register_module("encoder",SeparatedChangeModel(c));
    const auto options=torch::TensorOptions().device(c.level.device).dtype(c.level.dtype);
    variance_projection=register_parameter("variance_projection",torch::zeros({c.level.channel_count*c.level.input_width,c.change_width},options));
    variance_bias=register_parameter("variance_bias",torch::zeros({c.level.channel_count*c.level.input_width},options));
    for(const auto &p:named_parameters())parameters_.push_back({p.key(),p.value().sizes().vec(),p.value().requires_grad()});
    validate_state();
  }
  const LocationScaleDecoderConfig &config()const{return config_;}
  const RelativeBankStandardizer &normalizer()const{return normalizer_;}
  EncodeOutput encode(const Input &raw,const FrozenScaler &scaler){
    const auto input=relative_change_parent_input(raw,scaler,config_.level);
    const auto values=normalizer_.transform(relative_change_targets(raw,config_.level));
    EncodeOutput out;{torch::NoGradGuard guard;out=encoder->level->encode(input);}
    const auto changes=torch::matmul(values,encoder->change_projection.transpose(0,1));
    out.z_contextual_global=torch::cat({out.z_contextual_global,
        torch::where(out.sample_valid_mask.unsqueeze(-1),changes,torch::zeros_like(changes))},1);
    TORCH_CHECK(torch::isfinite(out.z_contextual_global).all().item<bool>(),"[rpb-v30] finite native required");return out;
  }
  LocationScaleDecoderTrace decode_with_trace(const torch::Tensor &native,const torch::Tensor &ids){
    const auto &c=config_.level;
    TORCH_CHECK(native.defined() && native.dim()==2 && native.size(1)==location_scale_decoder_native_width(config_) &&
        native.device()==c.device && native.scalar_type()==c.dtype && torch::isfinite(native).all().item<bool>(),
        "[rpb-v30] configured CUDA F32 finite native required");
    const auto B=native.size(0),C=c.channel_count,H=c.history_length,F=c.input_width;
    const auto z=native.narrow(1,c.export_width,config_.change_width);
    LocationScaleDecoderTrace out;
    {torch::NoGradGuard guard;out.parent_mean=encoder->level->decode(native.narrow(1,0,c.export_width).detach(),ids);}
    out.canonical_residual=torch::matmul(z,encoder->change_decoder.transpose(0,1)).reshape({B,C,H,F});
    const auto semantic=channel_indices(ids,c,B,c.device);
    const auto gather=semantic.unsqueeze(-1).unsqueeze(-1).expand({B,C,H,F});
    out.mean=out.parent_mean+out.canonical_residual.gather(1,gather);
    out.linear_score=torch::matmul(z,variance_projection.transpose(0,1));out.score=out.linear_score+variance_bias;
    const auto constants=location_scale_decoder_constants();const auto options=native.options();
    out.eps32=torch::tensor(constants[0],options);out.c32=torch::tensor(constants[1],options);
    out.coefficient32=torch::ones({},options)-out.eps32;
    out.u=torch::softplus(out.score+out.c32,kLocationScaleDecoderSoftplusBeta,kLocationScaleDecoderSoftplusThreshold);
    out.u0=torch::softplus(out.c32,kLocationScaleDecoderSoftplusBeta,kLocationScaleDecoderSoftplusThreshold);
    const auto ratio=out.u/out.u0;
    out.variance_CF=(out.eps32+out.coefficient32*ratio).reshape({B,C,1,F});
    TORCH_CHECK(torch::isfinite(out.linear_score).all().item<bool>() && torch::isfinite(out.score).all().item<bool>() &&
        torch::isfinite(out.u).all().item<bool>() && torch::isfinite(out.u0).item<bool>() && out.u0.gt(0).item<bool>() &&
        torch::isfinite(ratio).all().item<bool>() && torch::isfinite(out.variance_CF).all().item<bool>() &&
        out.variance_CF.gt(0).all().item<bool>() && torch::isfinite(out.mean).all().item<bool>(),
        "[rpb-v30] finite positive link intermediates/distribution required; no rescue clamp");
    out.variance=out.variance_CF.expand({B,C,H,F}).gather(1,gather);return out;
  }
  torch::Tensor decode(const torch::Tensor &native,const torch::Tensor &ids){return decode_with_trace(native,ids).mean;}
  LocationScaleForwardTrace forward_with_trace(const Input &visible_raw,const torch::Tensor &scaled_target,
      const torch::Tensor &target_mask,const torch::Tensor &eligible_channels,const FrozenScaler &scaler){
    const auto &c=config_.level;const auto target=scaled_target.to(c.device).detach(),support=target_mask.to(c.device),eligible=eligible_channels.to(c.device);
    TORCH_CHECK(support.sizes()==visible_raw.observed.sizes() && support.scalar_type()==torch::kBool &&
        !(support&visible_raw.observed.to(c.device)).any().item<bool>(),"[rpb-v30] targets and final visible input must be disjoint");
    LocationScaleForwardTrace out;out.visible=visible_raw.observed.to(c.device);out.target_mask=support;
    out.output.encoding=encode(visible_raw,scaler);out.distribution=decode_with_trace(out.output.encoding.z_contextual_global,visible_raw.channel_ids);
    out.nll=location_scale_decoder_nll(out.distribution.mean,out.distribution.variance,target,support,eligible);
    out.targets=torch::where(support,target,torch::zeros_like(target));out.output.reconstruction=out.distribution.mean;
    out.output.loss=out.nll.loss;out.output.target_counts=out.nll.target_counts;out.output.eligible_channels=out.nll.eligible_channels;
    out.output.eligible_examples=out.nll.eligible_examples;out.output.eligible_channel_count=out.nll.eligible_channel_count;
    out.output.eligible_example_count=out.nll.eligible_example_count;out.output.target_cell_count=out.nll.target_cell_count;return out;
  }
  LocationScaleForwardTrace forward_with_trace(const Input &raw,const torch::Tensor &hidden,const FrozenScaler &scaler){
    const auto input=relative_change_parent_input(raw,scaler,config_.level);
    const auto masks=configurable_spectral_mask_from_hidden(raw.observed.to(config_.level.device),hidden.to(config_.level.device),config_.level);
    const Input visible{raw.data,masks.visible,raw.channel_ids,raw.endpoints,raw.sampling_interval};
    return forward_with_trace(visible,input.data,masks.target,masks.eligible_channels,scaler);
  }
  ForwardOutput forward(const Input &raw,const torch::Tensor &hidden,const FrozenScaler &scaler){return forward_with_trace(raw,hidden,scaler).output;}
  void load(torch::serialize::InputArchive &a)override{torch::nn::Module::load(a);validate_state();}
  void validate_fixed_prior()const{encoder->validate_fixed_prior();}
  void validate_state()const{
    normalizer_.validate(config_.level);TORCH_CHECK(normalizer_.frozen && normalizer_.identity()==normalizer_identity_,"[rpb-v30] frozen normalizer changed");
    TORCH_CHECK(encoder,"[rpb-v30] registered encoder required");encoder->validate_state();const auto named=named_parameters();
    TORCH_CHECK(named.size()==parameters_.size(),"[rpb-v30] parameter count mismatch");
    for(const auto &p:parameters_){const auto *v=named.find(p.name);
      TORCH_CHECK(v && v->sizes().vec()==p.shape && v->requires_grad()==p.trainable && v->device()==config_.level.device &&
          v->scalar_type()==config_.level.dtype && torch::isfinite(*v).all().item<bool>(),"[rpb-v30] parameter contract mismatch: ",p.name);}
    for(const auto &p:encoder->named_parameters()){const auto *v=named.find("encoder."+p.key());
      TORCH_CHECK(v && v->unsafeGetTensorImpl()==p.value().unsafeGetTensorImpl(),"[rpb-v30] registered child holder mismatch");}
    for(const auto &p:std::vector<std::pair<std::string,torch::Tensor>>{{"variance_projection",variance_projection},{"variance_bias",variance_bias},
        {"encoder.change_projection",encoder->change_projection},{"encoder.change_decoder",encoder->change_decoder}})
      TORCH_CHECK(named[p.first].unsafeGetTensorImpl()==p.second.unsafeGetTensorImpl(),"[rpb-v30] registered parameter holder mismatch: ",p.first);
    const auto buffers=named_buffers();TORCH_CHECK(buffers.size()==encoder->named_buffers().size(),"[rpb-v30] buffer count mismatch");
    for(const auto &b:encoder->named_buffers()){const auto *v=buffers.find("encoder."+b.key());
      TORCH_CHECK(v && v->unsafeGetTensorImpl()==b.value().unsafeGetTensorImpl(),"[rpb-v30] registered buffer holder mismatch");}
  }
  SeparatedChangeModel encoder{nullptr};torch::Tensor variance_projection,variance_bias;
private:
  struct Contract{std::string name;std::vector<int64_t> shape;bool trainable;};
  LocationScaleDecoderConfig config_;RelativeBankStandardizer normalizer_;std::string normalizer_identity_;std::vector<Contract> parameters_;
};
TORCH_MODULE(LocationScaleDecoderModel);
inline torch::Tensor location_scale_decoder_native(const EncodeOutput &out,const LocationScaleDecoderConfig &c){return separated_change_native(out,c);}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
