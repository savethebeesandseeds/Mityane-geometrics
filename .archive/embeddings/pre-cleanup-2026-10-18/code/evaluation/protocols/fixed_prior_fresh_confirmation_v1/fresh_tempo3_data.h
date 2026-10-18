// SPDX-License-Identifier: MIT
#pragma once
// This is evaluation data preparation only. It never constructs an encoder,
// scaler, optimizer or head, and delegates all sampling to the original recipe.
#include "structured_hard_timing.h"
#include "frozen_role_guard.h"
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace fixed_prior_fresh_confirmation {
namespace ev = embedding::evaluation;
using ev::frozen_inputs::require;
inline constexpr const char *kProtocol = "fixed-prior-fresh-confirmation-v1";
inline constexpr const char *kCardSha256 = "533afbcd14f03ad373f0dcfb928747d105b008e2e201a11f89f4a85f3d501c02";
inline constexpr std::array<uint64_t,5> kMasters{80787,81888,82989,84090,85191};
inline constexpr const char *kDeletionPurpose = "fixed-prior-fresh-confirmation-v1/lag_sign/validation-coordinate-dropout";
inline constexpr double kDeletionRate = .30;

struct FreshTempo3Data {
  uint64_t master;
  ev::ControlledDataset training, validation, validation_deleted;
  torch::Tensor requested_erasure;
};

// The quality runner's only data input is a Batch: labels and source identifiers
// are outside this type and remain scoring/fitting metadata.
inline const embedding::Batch &model_observations(const ev::ControlledDataset &data) {
  return data.observed;
}
inline bool exact_cpu(const torch::Tensor &a,const torch::Tensor &b) {
  if(!a.defined()||!b.defined()||!a.device().is_cpu()||!b.device().is_cpu()||
     !a.is_contiguous()||!b.is_contiguous()||a.scalar_type()!=b.scalar_type()||a.sizes()!=b.sizes())return false;
  return !a.numel()||std::memcmp(a.const_data_ptr(),b.const_data_ptr(),a.numel()*a.element_size())==0;
}
inline std::string source_prefix(uint64_t master) {
  return std::string(kProtocol)+"/lag_sign/"+ev::kStructuredHardTimingProtocol+"/seed-"+
      std::to_string(master)+"/lag_sign/source-";
}
inline void qualify(ev::ControlledDataset &data) {
  for(auto &id:data.source_ids)id=std::string(kProtocol)+"/lag_sign/"+id;
}
inline std::set<int64_t> validate_split(const ev::ControlledDataset &data,int64_t rows,uint64_t master) {
  require(rows==256||rows==128,"closed fresh split population");
  const auto &x=data.observed.data;const auto &mask=data.observed.feature_mask;
  const std::vector<int64_t> shape{rows,3,32,3};
  require(x.defined()&&x.device().is_cpu()&&x.scalar_type()==torch::kFloat64&&x.is_contiguous()&&
      x.sizes()==torch::IntArrayRef(shape),"fresh CPU F64 B3H32F3 values");
  require(mask.defined()&&mask.device().is_cpu()&&mask.scalar_type()==torch::kBool&&mask.is_contiguous()&&
      mask.sizes()==x.sizes(),"fresh CPU Bool observation mask");
  require(data.labels.defined()&&data.labels.device().is_cpu()&&data.labels.scalar_type()==torch::kInt64&&
      data.labels.is_contiguous()&&data.labels.sizes()==torch::IntArrayRef({rows}),"scoring-only CPU Long labels");
  require(torch::isfinite(x).all().item<bool>()&&x.masked_select(mask.logical_not()).eq(0).all().item<bool>(),
      "finite legal values and zero hidden storage");
  require(exact_cpu(data.clean.data,x)&&exact_cpu(data.clean.feature_mask,mask),"clean field contains only legal observations");
  require(data.source_ids.size()==size_t(rows),"whole ordered source association");
  const auto prefix=source_prefix(master);std::set<int64_t> sources;int64_t previous=-1;
  for(int64_t row=0;row<rows;row+=2) {
    const auto &id=data.source_ids[row];
    require(id.rfind(prefix,0)==0&&data.source_ids[row+1]==id,"qualified opposite-label source pair");
    const auto suffix=id.substr(prefix.size());
    require(!suffix.empty()&&suffix.find_first_not_of("0123456789")==std::string::npos&&suffix.size()<=3,"canonical original source integer");
    const auto source=std::stoll(suffix);
    require(source>=0&&source<192&&source>previous&&std::to_string(source)==suffix&&sources.insert(source).second,
        "ascending original source order, never renumbered after split");previous=source;
    const auto first=data.labels[row].item<int64_t>(),second=data.labels[row+1].item<int64_t>();
    require((first==0&&second==1)||(first==1&&second==0),"randomized opposite pair labels");
    require(exact_cpu(mask[row].contiguous(),mask[row+1].contiguous()),"shared paired natural/deleted masks");
    for(int64_t channel:{0,2})require(exact_cpu(x[row][channel].contiguous(),x[row+1][channel].contiguous()),
        "paired non-label channels share the original nuisance");
  }
  require(sources.size()==size_t(rows/2),"whole fresh source count");return sources;
}
inline void validate(const FreshTempo3Data &data) {
  const auto train=validate_split(data.training,256,data.master),val=validate_split(data.validation,128,data.master);
  (void)validate_split(data.validation_deleted,128,data.master);
  auto all=train;all.insert(val.begin(),val.end());
  require(train.size()==128&&val.size()==64&&all.size()==192,"original pair-disjoint192 universe");
  for(int64_t source=0;source<192;++source)require(all.count(source)==1,"complete original source universe");
  require(data.validation_deleted.source_ids==data.validation.source_ids&&exact_cpu(data.validation_deleted.labels,data.validation.labels),
      "deleted view preserves original source order and labels");
  const auto &e=data.requested_erasure;const auto &base=data.validation.observed;const auto &deleted=data.validation_deleted.observed;
  require(e.defined()&&e.device().is_cpu()&&e.scalar_type()==torch::kBool&&e.is_contiguous()&&e.sizes()==base.data.sizes(),
      "requested erasure CPU Bool B3H32F3");
  require(exact_cpu(deleted.feature_mask,base.feature_mask.logical_and(e.logical_not()).contiguous())&&
      exact_cpu(deleted.data,torch::where(deleted.feature_mask,base.data,torch::zeros_like(base.data)).contiguous()),
      "exact original deletion subset with unchanged retained values");
  for(int64_t row=0;row<128;row+=2)require(exact_cpu(e[row].contiguous(),e[row+1].contiguous()),"source-paired requested erasure");
}
inline FreshTempo3Data make_fresh_tempo3_data(uint64_t master) {
  require(master<=uint64_t(std::numeric_limits<int64_t>::max()),"signed archive seed range");
  auto original=ev::make_structured_hard_timing_development(128,64,master);
  require(original.seed==master,"authoritative generator seed association");
  qualify(original.training);qualify(original.validation);
  const auto view=ev::make_coordinate_deletion_view(original.validation.observed,original.validation.source_ids,
      ev::Task::lag_sign,ev::stream_seed(master,ev::kStructuredHardTimingDeletionStream),kDeletionRate,kDeletionPurpose);
  ev::ControlledDataset deleted{{view.observations.data.clone(),view.observations.feature_mask.clone()},
      view.observations,original.validation.labels.clone(),original.validation.source_ids};
  FreshTempo3Data out{master,std::move(original.training),std::move(original.validation),std::move(deleted),view.requested_erasure};
  validate(out);return out;
}
} // namespace fixed_prior_fresh_confirmation
