// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Owned read-only CACHE88 adapter. Layout/feature order copied from the MIT
// cuwacunu data registry; no application or historical model headers are used.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v2.h"
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <memory>

namespace causal_finetune_v2::market {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::v2;
namespace fs=std::filesystem;
inline constexpr int64_t interval=60000,train_begin=1704067259999LL,train_end=1705881599999LL,
    validation_begin=1705881659999LL,validation_end=1706486399999LL,unused_begin=1706486459999LL;
#pragma pack(push,1)
struct Record {
  int64_t open_time;double open_price,high_price,low_price,close_price,volume;
  int64_t close_time;double quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume;
};
#pragma pack(pop)
static_assert(sizeof(Record)==88 && offsetof(Record,close_time)==48 &&
    std::endian::native==std::endian::little && std::is_trivially_copyable_v<Record>);
inline void check(bool yes,const std::string &why){TORCH_CHECK(yes,"[CPB-v2 data] ",why);}
struct Channel {
  fs::path path;std::vector<char> bytes;int64_t first,last;
  explicit Channel(const fs::path &file):path(fs::canonical(file)){
    std::ifstream in(path,std::ios::binary|std::ios::ate);check(bool(in),"existing raw cache");
    const auto size=in.tellg();check(size>0 && static_cast<int64_t>(size)%88==0,"complete CACHE88");
    bytes.resize(static_cast<size_t>(size));in.seekg(0);in.read(bytes.data(),size);check(bool(in),"read raw cache");
    first=at(0).close_time;last=at(rows()-1).close_time;
    for(size_t i=0;i<rows();++i)check(at(i).close_time==first+static_cast<int64_t>(i)*interval,"actual minute lattice");
  }
  size_t rows()const{return bytes.size()/88;}
  Record at(size_t i)const{check(i<rows(),"record index");Record r;std::memcpy(&r,bytes.data()+88*i,88);return r;}
};
inline std::vector<int64_t> anchors(int64_t begin,int64_t end,int64_t history,int64_t n,bool nonoverlap){
  check(begin>=0 && end>=begin && (end-begin)%interval==0 && history>0 && n>=2,"selection geometry");
  const auto stride=nonoverlap?history*interval:interval,first=begin+(history-1)*interval;
  check(first<=end,"complete role history");const auto available=(end-first)/stride+1;
  check(n<=available,"enough role windows");std::vector<int64_t> out;
  for(int64_t i=0;i<n;++i)out.push_back(first+static_cast<int64_t>(static_cast<__int128>(i)*(available-1)/(n-1))*stride);
  return out;
}
inline std::vector<int64_t> slice(const std::vector<int64_t> &rows,size_t offset,size_t count){
  check(offset<=rows.size(),"slice offset");const auto take=std::min(count,rows.size()-offset);
  return{rows.begin()+offset,rows.begin()+offset+take};
}
struct Dataset {
  std::vector<Channel> channels;int64_t calls{0},record_cells{0},genuine_f64_cells{0};
  explicit Dataset(const std::vector<fs::path> &files){check(files.size()==3,"ordered BTC ETH SOL sources");for(const auto &p:files)channels.emplace_back(p);}
  cp::RawHistory load(const std::vector<int64_t> &rows,int64_t begin,int64_t end){
    check(!rows.empty(),"nonempty batch");const auto B=static_cast<int64_t>(rows.size());
    auto raw=torch::zeros({B,3,128,9},torch::kFloat64),mask=torch::zeros({B,3,128,9},torch::kBool),times=torch::zeros({B,128},torch::kFloat64);
    auto x=raw.accessor<double,4>();auto m=mask.accessor<bool,4>();auto t=times.accessor<double,2>();
    for(int64_t b=0;b<B;++b){const auto anchor=rows[b];check(anchor<=end && anchor-127*interval>=begin,"past window within declared role");
      for(int64_t h=0;h<128;++h){const auto key=anchor-(127-h)*interval;t[b][h]=static_cast<double>(key);
        for(int64_t c=0;c<3;++c){const auto &source=channels[c];check(key>=source.first && key<=source.last && (key-source.first)%interval==0,"exact cache key");
          const auto r=source.at(static_cast<size_t>((key-source.first)/interval));check(r.close_time==key,"actual record timestamp");
          const std::array<double,9> values{r.open_price,r.high_price,r.low_price,r.close_price,r.volume,r.quote_asset_volume,r.number_of_trades,r.taker_buy_base_volume,r.taker_buy_quote_volume};
          for(int64_t f=0;f<9;++f){const auto value=values[f];const bool valid=r.open_time!=INT64_MIN && std::isfinite(value);
            check(!valid || std::isfinite(static_cast<float>(value)),"supported historical F32 parent cast");x[b][c][h][f]=valid?value:0.;m[b][c][h][f]=valid;
            check(!valid || std::bit_cast<uint64_t>(x[b][c][h][f])==std::bit_cast<uint64_t>(value),"record F64 bits retained");
            ++record_cells;if(valid && value!=static_cast<double>(static_cast<float>(value)))++genuine_f64_cells;
          }
        }
      }
    }
    ++calls;return{raw,mask,torch::tensor({0,1,2},torch::kInt64),times,60000.};
  }
};
} // namespace causal_finetune_v2::market
