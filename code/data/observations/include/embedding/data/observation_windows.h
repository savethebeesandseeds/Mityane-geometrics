// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE at repository root.
#pragma once
#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace embedding::data::observation_windows {
inline constexpr const char *kKind="observation_window_dataset_v1";
struct Dataset {
  torch::Tensor values,observed,channel_ids,timestamps,cutoffs,row_ids;
  double sampling_interval{1.};
  std::string source_identity,split_identity,role;
  int64_t rows()const{return values.size(0);}
};
inline void identity(const std::string &value){
  TORCH_CHECK(!value.empty()&&value.size()<=1024&&value.find('\0')==std::string::npos,
      "[observation windows] nonempty bounded source/split/role identity required");
}
inline torch::Tensor text_tensor(const std::string &value){
  return torch::tensor(std::vector<uint8_t>(value.begin(),value.end()),torch::kUInt8);
}
inline std::string read_text(torch::serialize::InputArchive &archive,const char *key){
  torch::Tensor value;archive.read(key,value,true);
  TORCH_CHECK(value.device().is_cpu()&&value.scalar_type()==torch::kUInt8&&value.dim()==1&&value.numel()<=1024,
      "[observation windows] malformed text: ",key);
  const auto bytes=value.contiguous();return std::string(reinterpret_cast<const char*>(bytes.const_data_ptr<uint8_t>()),bytes.numel());
}
inline void validate(const Dataset &data){
  identity(data.source_identity);identity(data.split_identity);identity(data.role);
  TORCH_CHECK(data.role=="TRAIN"||data.role=="VALIDATION"||data.role=="TEST",
      "[observation windows] declared split role must be TRAIN, VALIDATION or TEST");
  const auto &x=data.values;
  TORCH_CHECK(x.defined()&&x.device().is_cpu()&&x.scalar_type()==torch::kFloat64&&x.dim()==4&&
      x.size(0)>0&&x.size(1)>0&&x.size(2)>0&&x.size(3)>0,"[observation windows] CPU F64 B,C,N,F values required");
  const auto B=x.size(0),C=x.size(1),N=x.size(2);
  TORCH_CHECK(data.observed.defined()&&data.observed.device().is_cpu()&&data.observed.scalar_type()==torch::kBool&&
      data.observed.sizes()==x.sizes(),"[observation windows] matching CPU bool mask required");
  TORCH_CHECK(data.channel_ids.defined()&&data.channel_ids.device().is_cpu()&&data.channel_ids.scalar_type()==torch::kInt64&&
      ((data.channel_ids.dim()==1&&data.channel_ids.size(0)==C)||(data.channel_ids.dim()==2&&data.channel_ids.sizes()==torch::IntArrayRef({B,C}))),
      "[observation windows] CPU I64 semantic IDs C or B,C required");
  const auto ids=(data.channel_ids.dim()==1?data.channel_ids.unsqueeze(0):data.channel_ids).contiguous();
  for(int64_t b=0;b<ids.size(0);++b){const auto row=ids[b].contiguous();const auto *p=row.const_data_ptr<int64_t>();
    TORCH_CHECK(std::set<int64_t>(p,p+C).size()==size_t(C),"[observation windows] channel IDs must be unique within each row");}
  TORCH_CHECK(std::isfinite(data.sampling_interval)&&data.sampling_interval>0,"[observation windows] finite positive grid interval required");
  TORCH_CHECK(data.timestamps.defined()&&data.timestamps.device().is_cpu()&&data.timestamps.scalar_type()==torch::kFloat64&&
      ((data.timestamps.dim()==1&&data.timestamps.size(0)==N)||(data.timestamps.dim()==2&&data.timestamps.sizes()==torch::IntArrayRef({B,N})))&&
      torch::isfinite(data.timestamps).all().item<bool>(),"[observation windows] finite declared CPU F64 timestamp schema required");
  TORCH_CHECK(data.cutoffs.defined()&&data.cutoffs.device().is_cpu()&&data.cutoffs.scalar_type()==torch::kFloat64&&
      data.cutoffs.sizes()==torch::IntArrayRef({B})&&torch::isfinite(data.cutoffs).all().item<bool>(),"[observation windows] finite CPU F64 cutoffs B required");
  TORCH_CHECK(data.row_ids.defined()&&data.row_ids.device().is_cpu()&&data.row_ids.scalar_type()==torch::kInt64&&
      data.row_ids.sizes()==torch::IntArrayRef({B}),"[observation windows] explicit CPU I64 row IDs B required");
  const auto rows=data.row_ids.contiguous();const auto *row_pointer=rows.const_data_ptr<int64_t>();
  TORCH_CHECK(std::set<int64_t>(row_pointer,row_pointer+B).size()==size_t(B),"[observation windows] source row IDs must be unique");
  // Observation values and mask contents are intentionally not inspected here.
  // A consumer must crop at its cutoff before validating retained observations.
}
inline Dataset read(const std::filesystem::path &path){
  torch::serialize::InputArchive archive;archive.load_from(path.string(),torch::kCPU);
  TORCH_CHECK(read_text(archive,"artifact_kind")==kKind,"[observation windows] typed dataset required");
  Dataset data;data.source_identity=read_text(archive,"source_identity");data.split_identity=read_text(archive,"split_identity");data.role=read_text(archive,"role");
  archive.read("values",data.values,true);archive.read("observed",data.observed,true);archive.read("channel_ids",data.channel_ids,true);
  archive.read("timestamps",data.timestamps,true);archive.read("cutoffs",data.cutoffs,true);archive.read("row_ids",data.row_ids,true);
  torch::Tensor interval;archive.read("sampling_interval",interval,true);
  TORCH_CHECK(interval.device().is_cpu()&&interval.dim()==0&&interval.scalar_type()==torch::kFloat64,
      "[observation windows] F64 interval scalar required");data.sampling_interval=interval.item<double>();validate(data);return data;
}
inline void write(const std::filesystem::path &path,const Dataset &data){
  validate(data);TORCH_CHECK(!std::filesystem::exists(path),"[observation windows] output already exists");
  if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
  // Reserve a unique temporary directory; publish with a hard link so an
  // existing destination is never replaced, even if it appears concurrently.
  auto temporary=path;temporary+=".pending-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  TORCH_CHECK(std::filesystem::create_directory(temporary),"[observation windows] exclusive publication reservation required");
  const auto payload=temporary/"dataset.pt";
  torch::serialize::OutputArchive archive;
  archive.write("artifact_kind",text_tensor(kKind),true);archive.write("source_identity",text_tensor(data.source_identity),true);
  archive.write("split_identity",text_tensor(data.split_identity),true);archive.write("role",text_tensor(data.role),true);
  archive.write("values",data.values,true);archive.write("observed",data.observed,true);archive.write("channel_ids",data.channel_ids,true);
  archive.write("timestamps",data.timestamps,true);archive.write("cutoffs",data.cutoffs,true);archive.write("row_ids",data.row_ids,true);
  archive.write("sampling_interval",torch::tensor(data.sampling_interval,torch::kFloat64),true);archive.save_to(payload.string());
  std::filesystem::create_hard_link(payload,path);
  std::filesystem::remove(payload);std::filesystem::remove(temporary);
}
// This checked selector may contain repeated indices: dataset rows remain
// uniquely identified, while training can sample a batch with replacement.
inline Dataset subset(const Dataset &data,const std::vector<int64_t> &indices){
  TORCH_CHECK(!indices.empty(),"[observation windows] nonempty batch indices required");
  for(const auto index:indices)TORCH_CHECK(index>=0&&index<data.rows(),"[observation windows] batch index outside dataset");
  const auto index=torch::tensor(indices,torch::kInt64);Dataset batch;
  batch.values=data.values.index_select(0,index);batch.observed=data.observed.index_select(0,index);
  batch.channel_ids=data.channel_ids.dim()==1?data.channel_ids:data.channel_ids.index_select(0,index);
  batch.timestamps=data.timestamps.dim()==1?data.timestamps:data.timestamps.index_select(0,index);
  batch.cutoffs=data.cutoffs.index_select(0,index);batch.row_ids=data.row_ids.index_select(0,index);
  batch.sampling_interval=data.sampling_interval;batch.source_identity=data.source_identity;batch.split_identity=data.split_identity;batch.role=data.role;return batch;
}
} // namespace embedding::data::observation_windows
