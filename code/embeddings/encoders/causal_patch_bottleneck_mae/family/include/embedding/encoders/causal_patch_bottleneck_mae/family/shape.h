// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v16/location_scale_decoder_config.h"
#include <initializer_list>

namespace embedding::encoders::causal_patch_bottleneck_mae::family {
namespace implementation=embedding::encoders::causal_patch_bottleneck_mae::detail::v16;
using Config=implementation::LocationScaleDecoderConfig;

// This family reuses the frozen V16 numerical implementation. Geometry and
// capacity belong to Config; none of these counts identifies a trained instance.
struct Dimensions {
  int64_t channels,history,features,patch_length,encoder_width,level_width,change_width;
  int64_t native_width,patch_count,padded_history,pair_count,spectral_width,shape_width;
  int64_t relation_bank_width,change_bank_width,decoder_cells;
};
struct ParameterCounts {int64_t learned_tensors,learned_values,fixed_tensors,fixed_values;};

namespace shape_detail {
inline int64_t add(std::initializer_list<int64_t> values){
  int64_t out=0;
  for(const auto value:values){
    TORCH_CHECK(value>=0 && out<=std::numeric_limits<int64_t>::max()-value,
        "[cpb-family] nonnegative dimension sum overflow");
    out+=value;
  }
  return out;
}
inline int64_t multiply(std::initializer_list<int64_t> values){
  int64_t out=1;
  for(const auto value:values){
    TORCH_CHECK(value>=0 && (!value || out<=std::numeric_limits<int64_t>::max()/value),
        "[cpb-family] nonnegative dimension product overflow");
    out*=value;
  }
  return out;
}
inline Dimensions dimensions(const Config &config){
  const auto &c=config.level;
  TORCH_CHECK(c.channel_count>0 && c.history_length>=17 && c.input_width>0 &&
      c.patch_length>0 && c.encoder_width>0 && c.export_width>0 && config.change_width>0 &&
      c.num_layers>0 && c.num_heads>0 && c.feedforward_width>0 && c.decoder_hidden_width>0,
      "[cpb-family] positive dimensions and H>=17 required");
  TORCH_CHECK(c.device.is_cuda() && c.device.index()>=0,
      "[cpb-family] an explicit CUDA device index is required");
  const auto C=c.channel_count,H=c.history_length,F=c.input_width,P=c.patch_length;
  // Match the original checked pair-count domain before its C*C validation.
  (void)multiply({C,C});
  const auto pairs=multiply({C,C-1})/2,spectral=multiply({4,pairs});
  TORCH_CHECK(c.export_width>spectral,"[cpb-family] level width must exceed four coordinates per channel pair");
  const auto K=add({H/P,H%P?1:0}),padded=multiply({K,P});
  (void)add({H,P-1}); // The unchanged core uses (H+P-1)/P.
  return{C,H,F,P,c.encoder_width,c.export_width,config.change_width,
      add({c.export_width,config.change_width}),K,padded,pairs,spectral,c.export_width-spectral,
      multiply({8,pairs,F,F}),multiply({10,C,F}),multiply({C,H,F})};
}
inline ParameterCounts counts(const Config &config,const Dimensions &d){
  const auto &c=config.level;
  const auto C=d.channels,F=d.features,P=d.patch_length,W=d.encoder_width,D=d.level_width;
  const auto U=d.change_width,K=d.patch_count,R=d.pair_count,Q=d.shape_width;
  const auto J=c.feedforward_width,M=c.decoder_hidden_width;
  const auto block=add({multiply({4,W,W}),multiply({2,W,J}),multiply({9,W}),J});
  const auto backbone=add({
      multiply({2,P,F,W}),W,multiply({K,W}),multiply({C,W}),
      multiply({add({c.num_layers,1}),block}),multiply({2,W}),W,1,K,
      multiply({D,W}),D,multiply({16,K}),multiply({16,C}),
      multiply({M,add({D,33})}),multiply({M,P,F}),multiply({P,F}),
      multiply({W,C,add({D,1})}),W,multiply({D,W}),D});
  const auto learned=add({backbone,multiply({Q,D}),multiply({8,Q,R,F,F}),
      multiply({10,U,C,F}),multiply({d.decoder_cells,U}),multiply({C,F,U}),multiply({C,F})});
  const auto fixed=multiply({16,R,F,F});
  // Also bound their aggregate and F32 storage size before allocating tensors.
  (void)multiply({4,add({learned,fixed})});
  return{add({39,multiply({12,c.num_layers})}),learned,1,fixed};
}
} // namespace shape_detail

inline ParameterCounts parameter_counts(const Config &config){
  const auto d=shape_detail::dimensions(config);
  const auto counts=shape_detail::counts(config,d);
  implementation::validate_location_scale_decoder_config(config);
  return counts;
}
inline Dimensions dimensions(const Config &config){
  const auto d=shape_detail::dimensions(config);
  (void)shape_detail::counts(config,d);
  implementation::validate_location_scale_decoder_config(config);
  return d;
}
inline void validate_config(const Config &config){(void)parameter_counts(config);}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::family
