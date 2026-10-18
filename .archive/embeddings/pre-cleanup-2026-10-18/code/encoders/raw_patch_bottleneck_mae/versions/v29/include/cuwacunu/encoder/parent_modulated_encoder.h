// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_checkpoint.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_modulated_decoder_settings.h"

namespace cuwacunu::encoder::parent_modulated {
namespace implementation=embedding::encoders::raw_patch_bottleneck_mae;
using Config=implementation::ParentModulatedDecoderConfig;using Encoder=implementation::ParentModulatedDecoderModel;
using Input=implementation::Input;using FrozenScaler=implementation::FrozenScaler;using EncodeOutput=implementation::EncodeOutput;
using BankStandardizer=implementation::RelativeBankStandardizer;
inline Config read_settings(const std::filesystem::path &p){return implementation::read_parent_modulated_decoder_settings(p);}
inline auto input_config(const Config &c){return implementation::configurable_spectral_input_config(c.level);}
inline auto transform(const FrozenScaler &s,const Input &i,const Config &c){return implementation::relative_change_parent_input(i,s,c.level);}
inline auto native_embedding(const EncodeOutput &out,const Config &c){return implementation::parent_modulated_decoder_native(out,c);}
inline auto make_training_mask(const torch::Tensor &observed,const Config &c,int64_t seed){return implementation::configurable_spectral_make_training_mask(observed,c.level,seed);}
inline auto mask_from_hidden(const torch::Tensor &observed,const torch::Tensor &hidden,const Config &c){return implementation::configurable_spectral_mask_from_hidden(observed,hidden,c.level);}
inline void save_checkpoint(const std::string &p,const Encoder &m,const FrozenScaler &s,int64_t updates){implementation::save_parent_modulated_decoder_checkpoint(p,m,s,updates);}
} // namespace cuwacunu::encoder::parent_modulated
