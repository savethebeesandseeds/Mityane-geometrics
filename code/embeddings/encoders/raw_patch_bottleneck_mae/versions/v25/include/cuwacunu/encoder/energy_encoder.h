// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../LICENSE.
#pragma once
#include "cuwacunu/encoder/content_artifact.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_settings.h"

namespace cuwacunu::encoder::energy {
namespace implementation=embedding::encoders::raw_patch_bottleneck_mae;
using Config=implementation::EnergyChangeConfig;using Encoder=implementation::EnergyChangeModel;
using Input=content::Input;using FrozenScaler=content::FrozenScaler;using EncodeOutput=content::EncodeOutput;
using MomentRmsScaler=implementation::MomentRmsScaler;
inline Config read_settings(const std::filesystem::path &p){return implementation::read_energy_change_settings(p);}
inline auto input_config(const Config &c){return content::input_config(c.level);}
inline auto transform(const FrozenScaler &s,const Input &i,const Config &c){return content::transform(s,i,c.level);}
inline auto native_embedding(const EncodeOutput &out,const Config &c){return implementation::energy_change_native(out,c);}
inline auto make_training_mask(const torch::Tensor &observed,const Config &c,int64_t seed){return content::make_training_mask(observed,c.level,seed);}
inline auto mask_from_hidden(const torch::Tensor &observed,const torch::Tensor &hidden,const Config &c){return content::mask_from_hidden(observed,hidden,c.level);}
inline void save_checkpoint(const std::string &p,const Encoder &m,const FrozenScaler &s,const MomentRmsScaler &n,int64_t updates){implementation::save_energy_change_checkpoint(p,m,s,n,updates);}
} // namespace cuwacunu::encoder::energy
