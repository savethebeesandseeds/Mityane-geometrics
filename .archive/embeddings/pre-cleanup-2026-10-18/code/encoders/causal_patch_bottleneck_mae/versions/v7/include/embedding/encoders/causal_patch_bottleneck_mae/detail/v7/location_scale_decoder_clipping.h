// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v7/location_scale_decoder_model.h"
#include <torch/nn/utils/clip_grad.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <vector>

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v7 {
// No module/parameter/buffer is registered here. Forward arithmetic and random
// constructor draws are unchanged. This is the complete optimization partition.
struct LocationScaleClipPartition {
  std::vector<torch::Tensor> mean,uncertainty;
  std::vector<std::string> mean_names,uncertainty_names,fixed_names;
  int64_t mean_values{0},uncertainty_values{0},fixed_values{0};
};
inline LocationScaleClipPartition location_scale_clip_partition(const LocationScaleDecoderModel &model){
  TORCH_CHECK(model,"[cpb-v7] owned model required for clipping partition");
  model->validate_state();model->validate_fixed_prior();LocationScaleClipPartition out;
  const std::set<std::string> uncertainty_names{"core.uncertainty_projection","core.variance_projection","core.variance_bias"};
  std::set<void *> seen;std::set<std::string> assigned_uncertainty;
  for(const auto &item:model->named_parameters()){
    const auto name="core."+item.key();const auto &p=item.value();
    TORCH_CHECK(seen.insert(p.unsafeGetTensorImpl()).second,"[cpb-v7] aliased clipping parameter");
    if(!p.requires_grad()){
      TORCH_CHECK(!uncertainty_names.count(name),"[cpb-v7] uncertainty parameter cannot be frozen");
      out.fixed_names.push_back(name);out.fixed_values+=p.numel();continue;
    }
    TORCH_CHECK(p.device()==model->config().level.device&&p.is_cuda()&&p.scalar_type()==torch::kFloat32,
        "[cpb-v7] learned clipping parameters must be configured CUDA F32");
    if(uncertainty_names.count(name)){
      out.uncertainty.push_back(p);out.uncertainty_names.push_back(name);out.uncertainty_values+=p.numel();assigned_uncertainty.insert(name);
    }else{out.mean.push_back(p);out.mean_names.push_back(name);out.mean_values+=p.numel();}
  }
  TORCH_CHECK(assigned_uncertainty==uncertainty_names&&out.mean.size()==73&&out.mean_values==293537&&
      out.uncertainty.size()==3&&out.uncertainty_values==1215&&out.fixed_names.size()==1&&out.fixed_values==3888&&seen.size()==77,
      "[cpb-v7] exact disjoint mean73/293537 uncertainty3/1215 fixed1/3888 partition required");
  return out;
}
inline std::string location_scale_clip_parameter_names(const LocationScaleClipPartition &p){
  std::string names;
  for(const auto &name:p.mean_names)names+="mean:"+name+'\n';
  for(const auto &name:p.uncertainty_names)names+="uncertainty:"+name+'\n';
  for(const auto &name:p.fixed_names)names+="fixed-excluded:"+name+'\n';
  return names;
}
struct LocationScaleClipResult {
  LocationScaleClipPartition partition;
  double mean_pre_clip_norm{0},uncertainty_pre_clip_norm{0},mean_clip_factor{1},uncertainty_clip_factor{1};
};
inline LocationScaleClipResult location_scale_clip_gradients(const LocationScaleDecoderModel &model){
  LocationScaleClipResult out;out.partition=location_scale_clip_partition(model);
  // Validate BOTH branches before mutating either. Undefined gradients are
  // allowed for currently unused parameters, as in the existing AdamW contract.
  for(const auto *group:{&out.partition.mean,&out.partition.uncertainty})for(const auto &p:*group)if(p.grad().defined()){
    const auto &g=p.grad();TORCH_CHECK(g.is_cuda()&&g.device()==p.device()&&g.scalar_type()==p.scalar_type()&&
        g.sizes()==p.sizes()&&!g.is_sparse()&&torch::isfinite(g).all().item<bool>(),"[cpb-v7] finite dense matching CUDA gradients required");
  }
  for(const auto &item:model->named_parameters())if(!item.value().requires_grad())
    TORCH_CHECK(!item.value().grad().defined(),"[cpb-v7] fixed prior excluded from clipping and gradients");
  out.mean_pre_clip_norm=torch::nn::utils::clip_grad_norm_(out.partition.mean,kLocationScaleDecoderMeanClipNorm,kLocationScaleDecoderClipNormType);
  out.uncertainty_pre_clip_norm=torch::nn::utils::clip_grad_norm_(out.partition.uncertainty,kLocationScaleDecoderUncertaintyClipNorm,kLocationScaleDecoderClipNormType);
  TORCH_CHECK(std::isfinite(out.mean_pre_clip_norm)&&std::isfinite(out.uncertainty_pre_clip_norm),"[cpb-v7] finite branch norms required");
  out.mean_clip_factor=std::min(1.,kLocationScaleDecoderMeanClipNorm/(out.mean_pre_clip_norm+1e-6));
  out.uncertainty_clip_factor=std::min(1.,kLocationScaleDecoderUncertaintyClipNorm/(out.uncertainty_pre_clip_norm+1e-6));
  // A single unchanged AdamW step follows both clips. The combined clipped L2
  // norm can approach sqrt(2); future residual/statistic/history coupling remains.
  return out;
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v7
