// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/encoder/content_artifact.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/separated_change_settings.h"
namespace cuwacunu::encoder::separate {
namespace implementation=embedding::encoders::raw_patch_bottleneck_mae;
using Config=implementation::SeparatedChangeConfig;using Encoder=implementation::SeparatedChangeModel;
using Input=content::Input;using FrozenScaler=content::FrozenScaler;using EncodeOutput=content::EncodeOutput;
inline Config read_settings(const std::filesystem::path &p){return implementation::read_separated_change_settings(p);}
inline auto input_config(const Config &c){return content::input_config(c.level);}
inline auto transform(const FrozenScaler &s,const Input &i,const Config &c){return content::transform(s,i,c.level);}
inline auto native_embedding(const EncodeOutput &out,const Config &c){return implementation::separated_change_native(out,c);}
inline auto make_training_mask(const torch::Tensor &observed,const Config &c,int64_t seed){return content::make_training_mask(observed,c.level,seed);}
inline auto mask_from_hidden(const torch::Tensor &observed,const torch::Tensor &hidden,const Config &c){return content::mask_from_hidden(observed,hidden,c.level);}
inline void save_checkpoint(const std::string &p,const Encoder &m,const FrozenScaler &s,int64_t updates){implementation::save_separated_change_checkpoint(p,m,s,updates);}
} // namespace cuwacunu::encoder::separate
