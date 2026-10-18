// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "cuwacunu/training/market_data.h"
#include <bit>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cuwacunu::training::market_raw_f64 {
namespace md=market_data;
using Input=encoder::primary::Input;
using Config=encoder::primary::Config;
using Record=data::binary_record_type_t<data::kline_t>;
static_assert(std::is_same_v<Record,data::kline_cache_t> && sizeof(Record)==88);
static_assert(std::is_trivially_copyable_v<Record> && std::endian::native==std::endian::little);
inline constexpr const char *loader_recipe="pinned-raw-kline-cache88-direct-double-registry;strict-minute-past-role;legacy-f32-equality-v1";
struct RecordWitness { int64_t batch,channel,index,key;bool valid;std::string bytes; };
struct Channel {
  std::string path;int fd{-1};void *mapping{nullptr};size_t bytes{0},rows{0};int64_t first{0},last{0};
  explicit Channel(const std::string &file):path(file){
    fd=::open(path.c_str(),O_RDONLY);TORCH_CHECK(fd>=0,"[raw F64] open existing raw cache");
    struct stat status{};
    if(::fstat(fd,&status)!=0 || status.st_size<=0 || status.st_size%sizeof(Record)!=0){::close(fd);fd=-1;TORCH_CHECK(false,"[raw F64] nonempty complete typed cache required");}
    bytes=static_cast<size_t>(status.st_size);rows=bytes/sizeof(Record);
    mapping=::mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    if(mapping==MAP_FAILED){mapping=nullptr;::close(fd);fd=-1;TORCH_CHECK(false,"[raw F64] read-only mmap failed");}
    first=record(0).close_time;last=record(rows-1).close_time;
    for(size_t j=0;j<rows;++j)TORCH_CHECK(record(j).close_time==first+static_cast<int64_t>(j)*md::sampling_interval_ms,"[raw F64] original cache must retain exact minute lattice");
  }
  ~Channel(){if(mapping)::munmap(mapping,bytes);if(fd>=0)::close(fd);}
  Channel(const Channel&)=delete;Channel&operator=(const Channel&)=delete;
  Record record(size_t index)const{TORCH_CHECK(index<rows,"[raw F64] record bounds");Record out;std::memcpy(&out,static_cast<const char*>(mapping)+index*sizeof(Record),sizeof out);return out;}
  std::string record_hex(size_t index)const{
    TORCH_CHECK(index<rows,"[raw F64] record witness bounds");static constexpr char hex[]="0123456789abcdef";std::string out;out.reserve(sizeof(Record)*2);
    const auto *p=static_cast<const unsigned char*>(mapping)+index*sizeof(Record);
    for(size_t j=0;j<sizeof(Record);++j){out+=hex[p[j]>>4];out+=hex[p[j]&15];}return out;
  }
};
struct Dataset {
  md::Dataset legacy;std::vector<std::unique_ptr<Channel>> channels;
  int64_t load_calls{0},record_values_checked{0},legacy_values_checked{0},genuine_f64_differs_from_f32_count{0};
  Input last_legacy;std::vector<int64_t> witness_rows;std::vector<RecordWitness> first_records;
  void add_dataset(const std::string &csv,int64_t history,int64_t future,const std::string &normalization,bool rebuild){
    TORCH_CHECK(future==0 && normalization=="none" && !rebuild,"[raw F64] existing raw past-only cache, no conversion/rebuild");
    const auto cache=data::raw_binary_for_csv(csv);TORCH_CHECK(data::cache_file_is_strictly_newer(cache,csv),"[raw F64] existing fresh cache required");
    channels.push_back(std::make_unique<Channel>(cache));legacy.add_dataset(csv,history,0,"none",false);
  }
};
inline Input load_batch(Dataset &dataset,const std::vector<int64_t> &anchors,int64_t begin,int64_t end,const Config &c){
  encoder::primary::validate_config(c);
  TORCH_CHECK(!anchors.empty() && c.input_width==9 && dataset.channels.size()==static_cast<size_t>(c.channel_count) &&
      begin>=0 && end>=begin && c.sampling_interval==md::sampling_interval_ms,"[raw F64] role/registry/channel geometry required");
  const auto B=static_cast<int64_t>(anchors.size()),C=c.channel_count,H=c.history_length,F=c.input_width;
  auto values=torch::zeros({B,C,H,F},torch::kFloat64),observed=torch::zeros({B,C,H,F},torch::kBool);
  auto x=values.accessor<double,4>();auto mask=observed.accessor<bool,4>();const bool witness=dataset.load_calls==0;
  if(witness)dataset.witness_rows=anchors;
  for(int64_t b=0;b<B;++b)for(int64_t ch=0;ch<C;++ch){
    const auto &file=*dataset.channels[ch];const auto anchor=anchors[b];
    TORCH_CHECK(anchor>=begin && anchor<=end && anchor-begin>=(H-1)*md::sampling_interval_ms && anchor>=file.first && anchor<=file.last &&
        (anchor-file.first)%md::sampling_interval_ms==0,"[raw F64] exact role/past anchor required");
    const auto terminal=static_cast<size_t>((anchor-file.first)/md::sampling_interval_ms);
    TORCH_CHECK(terminal>=static_cast<size_t>(H-1),"[raw F64] sufficient original past records required");
    for(int64_t t=0;t<H;++t){
      const auto index=terminal-static_cast<size_t>(H-1-t);const auto record=file.record(index);
      TORCH_CHECK(record.close_time==anchor-(H-1-t)*md::sampling_interval_ms && record.close_time>=begin && record.close_time<=end,"[raw F64] actual record chronology/role");
      const auto features=record.tensor_features();TORCH_CHECK(features.size()==static_cast<size_t>(F),"[raw F64] complete original registry");
      for(int64_t f=0;f<F;++f){
        const auto value=features[f];const bool valid=record.is_valid() && std::isfinite(value);
        TORCH_CHECK(!valid || std::isfinite(static_cast<float>(value)),"[raw F64] observed original value must support finite historical F32 cast");
        x[b][ch][t][f]=valid?value:0.;mask[b][ch][t][f]=valid;
        TORCH_CHECK(!valid || std::bit_cast<uint64_t>(x[b][ch][t][f])==std::bit_cast<uint64_t>(value),"[raw F64] original record double preservation");
        ++dataset.record_values_checked;
        if(valid && value!=static_cast<double>(static_cast<float>(value)))++dataset.genuine_f64_differs_from_f32_count;
      }
      if(witness)dataset.first_records.push_back({b,ch,static_cast<int64_t>(index),record.close_time,record.is_valid(),file.record_hex(index)});
    }
  }
  Input out{values,observed,torch::tensor(encoder::primary::implementation::resolved_channel_ids(c),torch::kInt64),torch::tensor(anchors,torch::kFloat64),c.sampling_interval};
  dataset.last_legacy=md::load_batch(dataset.legacy,anchors,begin,end,c);
  TORCH_CHECK(torch::equal(out.data.to(torch::kFloat32),dataset.last_legacy.data) && torch::equal(out.observed,dataset.last_legacy.observed) &&
      torch::equal(out.channel_ids,dataset.last_legacy.channel_ids) && torch::equal(out.endpoints,dataset.last_legacy.endpoints) && out.sampling_interval==dataset.last_legacy.sampling_interval,
      "[raw F64] direct-record F64 cast/mask/metadata must match original loader exactly");
  ++dataset.load_calls;dataset.legacy_values_checked+=values.numel();return out;
}
} // namespace cuwacunu::training::market_raw_f64
