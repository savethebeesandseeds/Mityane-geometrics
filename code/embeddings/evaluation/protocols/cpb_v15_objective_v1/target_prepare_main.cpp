// Copyright (c) 2026 Waajacu. See LICENSE.
// The only label producer: copied frozen F32-loader -> F64 target arithmetic.
// Parity opens January only. Confirmation values require a prior passed parity receipt.
#include "include/market_data.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace {
namespace md=cpb_v15_objective::market;
namespace ev=cuwacunu::evaluation;
namespace fs=std::filesystem;
constexpr int64_t feb_begin=1706745659999LL,feb_end=1709251199999LL,feb_minutes=41760;
void require(bool value,const std::string &why){TORCH_CHECK(value,"[V15 target preparation] ",why);}
void write_new(const fs::path &file,const std::string &text){require(!fs::exists(file),"exclusive result leaf");std::ofstream out(file);require(bool(out),"create result");out<<text<<'\n';out.close();require(bool(out),"save result");}
ev::TargetRows targets(md::Dataset &data,const std::vector<int64_t> &rows,int64_t begin,int64_t end){
  std::vector<torch::Tensor> values,valid;
  for(size_t offset=0;offset<rows.size();offset+=8){
    const auto selected=md::slice(rows,offset,8);
    const auto raw=data.load(selected,begin,end);
    // Historical market_study_main.cpp used default F32 Dataset storage.
    // The frozen target function then promotes those rounded values to F64.
    const auto legacy=raw.data.to(torch::kFloat32).to(torch::kFloat64);
    const auto t=ev::past_window_targets(legacy,raw.observed);
    values.push_back(t.values);valid.push_back(t.valid);
  }
  const auto ids=torch::tensor(rows,torch::kInt64);
  return {{torch::cat(values),torch::cat(valid)},ids,ids.clone()};
}
void equal(const ev::TargetRows &actual,const ev::TargetRows &expected){
  require(torch::equal(actual.row_ids,expected.row_ids)&&torch::equal(actual.group_ids,expected.group_ids),"exact timestamp/source association");
  require(torch::equal(actual.targets.valid,expected.targets.valid),"exact target support");
  require(embedding::encoders::causal_patch_bottleneck_mae::CPB_OBJECTIVE_VERSION::boundary::same_bits(actual.targets.values,expected.targets.values),"bitwise-equal legacy F32 -> F64 labels; refuse February access on mismatch");
}
std::vector<int64_t> confirmation_rows(){
  std::vector<int64_t> rows;
  for(int64_t i=0;i<32;++i){
    const auto index=((2*i+1)*feb_minutes)/64;
    const auto bin_begin=(i*feb_minutes)/32,bin_end=((i+1)*feb_minutes)/32;
    require(index-255>=bin_begin&&index<bin_end,"full H256 support inside its declared calendar bin");
    const auto timestamp=feb_begin+index*60000;
    require(timestamp<=feb_end&&(rows.empty()||timestamp-rows.back()>=256*60000),"disjoint chronological H256 windows");
    rows.push_back(timestamp);
  }
  return rows;
}
int run(int argc,char **argv){
  require(fs::exists("/.dockerenv"),"managed Linux container");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard guard;
  require(argc>=2,"parity or confirmation mode required");const std::string mode=argv[1];
  std::map<std::string,fs::path> args;
  require((argc-2)%2==0,"paired arguments");
  for(int i=2;i<argc;i+=2)require(args.emplace(argv[i],argv[i+1]).second,"unique argument");
  for(const auto *name:{"--cache-btc","--cache-eth","--cache-sol","--output"})require(args.count(name),"required input/output path");
  std::vector<fs::path> caches;for(const auto *name:{"--cache-btc","--cache-eth","--cache-sol"})caches.push_back(args.at(name));
  const auto output=fs::absolute(args.at("--output"));require(!fs::exists(output)&&fs::is_directory(output.parent_path()),"new result in existing attempt");
  if(mode=="parity"){
    require(args.size()==5&&args.count("--shared"),"January parity has only explicit caches/shared/output");
    md::Dataset data(caches);
    const auto training=md::anchors(md::train_begin,md::train_end,256,112,true),validation=md::anchors(md::validation_begin,md::validation_end,256,32,true);
    const auto a=targets(data,training,md::train_begin,md::train_end),b=targets(data,validation,md::validation_begin,md::validation_end);
    equal(a,ev::load_targets((args.at("--shared")/"features/train-targets.pt").string()));
    equal(b,ev::load_targets((args.at("--shared")/"features/validation-targets.pt").string()));
    write_new(output,"{\"kind\":\"cpb_v15_target_precision_parity_numeric_v1\",\"status\":\"passed\",\"passed\":true,\"train_rows\":112,\"validation_rows\":32,\"target_precision\":\"legacy_f32_then_f64\",\"all_target_values_bitwise_equal\":true,\"all_masks_and_ids_exact\":true,\"February_values_opened\":false,\"encoder_calls\":0,\"head_fits\":0}");
  }else if(mode=="confirmation"){
    require(args.size()==6&&args.count("--target-parity")&&args.count("--target-parity-sha256"),"confirmation needs pinned already-passed January precision receipt");
    require(embedding::encoders::causal_patch_bottleneck_mae::CPB_OBJECTIVE_VERSION::boundary::sha256_file(args.at("--target-parity"))==args.at("--target-parity-sha256").string(),"exact source-reviewed parity receipt");
    // The Node preparation gate checks passed receipt schema/pins before this process starts.
    md::Dataset data(caches);for(const auto &channel:data.channels)require(channel.first==feb_begin&&channel.last==feb_end&&channel.rows()==feb_minutes,"exact frozen February2024 month");
    const auto rows=confirmation_rows();
    const auto result=targets(data,rows,feb_begin,feb_end);
    require(result.targets.valid.all().item<bool>(),"complete confirmation target support; no replacement selection");
    ev::save_targets(output.string(),result);
  }else require(false,"unknown target preparation mode");
  return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
