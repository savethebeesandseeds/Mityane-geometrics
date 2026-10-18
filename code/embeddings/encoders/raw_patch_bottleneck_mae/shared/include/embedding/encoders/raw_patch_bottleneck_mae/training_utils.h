// SPDX-License-Identifier: MIT
#pragma once

#include "embedding/encoders/raw_patch_bottleneck_mae/types.h"
#include <cstdint>
#include <vector>

namespace embedding::encoders::raw_patch_bottleneck_mae::training_detail {

// Preserve the existing workflow's counter streams and row selection exactly.
inline uint64_t mixed(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline uint64_t counter_seed(int64_t seed, int64_t attempt, uint64_t stream) {
  return mixed(static_cast<uint64_t>(seed) ^ mixed(static_cast<uint64_t>(attempt)) ^ stream);
}

inline Input selected(const Input &input,const torch::Tensor &indices) {
  Input out=input;
  out.data=input.data.index_select(0,indices);
  out.observed=input.observed.index_select(0,indices);
  if (input.channel_ids.dim()==2) out.channel_ids=input.channel_ids.index_select(0,indices);
  out.endpoints=input.endpoints.index_select(0,indices);
  return out;
}

inline torch::Tensor sampled_indices(int64_t count,int64_t size,int64_t seed,int64_t attempt) {
  std::vector<int64_t> indices(static_cast<size_t>(size));
  const auto base=counter_seed(seed,attempt,0x726f7773ULL);
  for (int64_t i=0;i<size;++i) indices[static_cast<size_t>(i)]=
      static_cast<int64_t>(mixed(base+static_cast<uint64_t>(i))%static_cast<uint64_t>(count));
  return torch::tensor(indices,torch::kInt64);
}

} // namespace embedding::encoders::raw_patch_bottleneck_mae::training_detail
