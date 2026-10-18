// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../LICENSE.
#pragma once
#include "cuwacunu/artifacts/v1/sidecar.h"
#include <bit>
#include <torch/torch.h>

namespace cuwacunu::artifacts::v1 {
using NamedTensor=std::pair<std::string,torch::Tensor>;
inline std::string tensor_state_sha256(std::vector<NamedTensor> tensors) {
  require(std::endian::native==std::endian::little,"tensor hashing requires little-endian host");
  std::sort(tensors.begin(),tensors.end(),[](const auto& a,const auto& b){return a.first<b.first;});
  Sha256 hash;hash_field(hash,"cuwacunu_component_tensor_state_v1");hash_field(hash,std::to_string(tensors.size()));
  std::string previous;
  for(const auto& [name,tensor]:tensors) {
    require(!name.empty()&&name!=previous,"empty/duplicate state tensor name " + name);
    require(tensor.defined()&&tensor.layout()==torch::kStrided&&!tensor.is_quantized()&&!tensor.is_complex(),"unsupported state tensor " + name);
    // Restrict the first encoding to explicitly supported stable scalar bytes.
    const auto dtype=tensor.scalar_type();
    require(dtype==torch::kBool||dtype==torch::kUInt8||dtype==torch::kInt8||dtype==torch::kInt16||dtype==torch::kInt32||dtype==torch::kInt64||dtype==torch::kFloat16||dtype==torch::kBFloat16||dtype==torch::kFloat32||dtype==torch::kFloat64,"unsupported state dtype " + name);
    const auto cpu=tensor.detach().to(torch::kCPU).contiguous();
    hash_field(hash,name);hash_field(hash,c10::toString(dtype));hash_field(hash,std::to_string(cpu.dim()));
    for(const auto size:cpu.sizes())hash_field(hash,std::to_string(size));
    hash_field(hash,std::string_view(cpu.nbytes()?static_cast<const char*>(cpu.const_data_ptr()):"",cpu.nbytes()));previous=name;
  }
  return hash.finish();
}
inline std::string module_state_sha256(const torch::nn::Module& module,std::vector<NamedTensor> extra_tensors={}) {
  for(const auto& item:module.named_parameters(true))extra_tensors.emplace_back("parameter."+item.key(),item.value());
  for(const auto& item:module.named_buffers(true))extra_tensors.emplace_back("buffer."+item.key(),item.value());
  return tensor_state_sha256(std::move(extra_tensors));
}
} // namespace cuwacunu::artifacts::v1
