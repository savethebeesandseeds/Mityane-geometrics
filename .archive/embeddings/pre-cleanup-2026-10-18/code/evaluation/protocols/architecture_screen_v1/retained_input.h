// SPDX-License-Identifier: MIT
#pragma once
// Exact retained controlled-schema loader from the passed visible-difference
// producer. No generator or clean-data loader is involved in this screen.
#include "embedding/shared/paired_pooling.h"
#include "frozen_role_guard.h"
#include <regex>
#include <set>
#include <sys/stat.h>
namespace architecture_screen {
namespace fs=std::filesystem;
namespace ev=embedding::evaluation;
using namespace ev::frozen_inputs;
inline torch::Tensor read(torch::serialize::InputArchive &a,const std::string &key){torch::Tensor t;a.read(key,t,true);return t;}
inline void admit_files(const std::vector<fs::path>&paths){
  std::set<fs::path> names;std::set<std::pair<dev_t,ino_t>> nodes;
  for(const auto&p:paths){require(p.is_absolute()&&fs::is_regular_file(p)&&!fs::is_symlink(p)&&fs::canonical(p)==p,"canonical regular input");
    for(auto q=p;!q.empty()&&q!=q.root_path();q=q.parent_path())require(!fs::is_symlink(q),"input ancestor must be direct");
    struct stat s{};require(::stat(p.c_str(),&s)==0&&s.st_nlink==1&&names.insert(p).second&&nodes.emplace(s.st_dev,s.st_ino).second,"whole unique regular role matrix");}
}
inline ev::ControlledDataset load_controlled(const fs::path&path,int64_t rows,uint64_t master,bool deleted=false){
  require(rows==256||rows==128,"fixed retained controlled population");
  torch::serialize::InputArchive a;a.load_from(path.string(),torch::kCPU);
  const auto keys=a.keys();const std::set<std::string> actual(keys.begin(),keys.end());
  auto expected=std::set<std::string>{"observations","feature_mask","labels_scoring_only","source_ids_json"};
  if(deleted)expected.insert("requested_erasure");
  require(actual==expected,"exact retained controlled keys");
  ev::ControlledDataset out;out.observed={read(a,"observations"),read(a,"feature_mask")};out.labels=read(a,"labels_scoring_only");
  const std::vector<int64_t> shape{rows,3,32,3};
  require(out.observed.data.device().is_cpu()&&out.observed.data.scalar_type()==torch::kFloat64&&out.observed.data.sizes()==torch::IntArrayRef(shape)&&out.observed.data.is_contiguous(),"CPU F64 observations B3H32F3");
  require(out.observed.feature_mask.device().is_cpu()&&out.observed.feature_mask.scalar_type()==torch::kBool&&out.observed.feature_mask.sizes()==torch::IntArrayRef(shape)&&out.observed.feature_mask.is_contiguous(),"CPU Bool feature mask B3H32F3");
  require(out.labels.device().is_cpu()&&out.labels.scalar_type()==torch::kInt64&&out.labels.sizes()==torch::IntArrayRef({rows})&&out.labels.is_contiguous(),"CPU Long scoring labels B");
  require(torch::isfinite(out.observed.data).all().item<bool>()&&out.observed.data.masked_select(out.observed.feature_mask.logical_not()).eq(0).all().item<bool>(),"finite legal observations with hidden zeros");
  const auto ids=read(a,"source_ids_json");require(ids.device().is_cpu()&&ids.scalar_type()==torch::kUInt8&&ids.dim()==1&&ids.is_contiguous(),"CPU byte source JSON");
  const auto value=embedding::archive::tensor_text(ids);const std::regex quoted("\"([^\"\\\\]+)\"");
  for(auto it=std::sregex_iterator(value.begin(),value.end(),quoted);it!=std::sregex_iterator();++it)out.source_ids.push_back((*it)[1].str());
  require(out.source_ids.size()==size_t(rows)&&strings(out.source_ids)==value,"canonical ordered source JSON");
  const auto prefix="structured-hard-timing-comparison-v2/lag_sign/structured-hard-timing-v1/seed-"+std::to_string(master)+"/lag_sign/source-";
  int64_t previous=-1;
  for(int64_t i=0;i<rows;i+=2){require(out.source_ids[i].rfind(prefix,0)==0&&out.source_ids[i+1]==out.source_ids[i],"original paired source namespace");
    const auto suffix=out.source_ids[i].substr(prefix.size());require(!suffix.empty()&&suffix.find_first_not_of("0123456789")==std::string::npos,"canonical source integer");
    const auto source=std::stoll(suffix);require(source>=0&&source<192&&source>previous&&std::to_string(source)==suffix,"ascending original192 source universe");previous=source;
    const auto x=out.labels[i].item<int64_t>(),y=out.labels[i+1].item<int64_t>();require((x==0&&y==1)||(x==1&&y==0),"opposite randomized pair labels");
    require(torch::equal(out.observed.feature_mask[i],out.observed.feature_mask[i+1]),"paired observation masks");}
  if(deleted){const auto erasure=read(a,"requested_erasure");require(erasure.device().is_cpu()&&erasure.scalar_type()==torch::kBool&&erasure.sizes()==torch::IntArrayRef(shape)&&erasure.is_contiguous(),"CPU Bool requested erasure");
    require(!erasure.logical_and(out.observed.feature_mask).any().item<bool>(),"requested erasures remain hidden");}
  out.clean={out.observed.data.clone(),out.observed.feature_mask.clone()};return out;
}
inline void check_splits(const ev::ControlledDataset&t,const ev::ControlledDataset&v,const ev::ControlledDataset&d){
  const std::set<std::string> ti(t.source_ids.begin(),t.source_ids.end()),vi(v.source_ids.begin(),v.source_ids.end());auto all=ti;all.insert(vi.begin(),vi.end());
  require(ti.size()==128&&vi.size()==64&&all.size()==192&&d.source_ids==v.source_ids&&torch::equal(d.labels,v.labels),"disjoint original splits and unchanged deleted rows");
  require(!d.observed.feature_mask.logical_and(v.observed.feature_mask.logical_not()).any().item<bool>()&&
    torch::equal(d.observed.data.masked_select(d.observed.feature_mask),v.observed.data.masked_select(d.observed.feature_mask)),"deleted observation values/support are original subset");
}
inline embedding::Batch loader_engineering(const fs::path&root){
  using Fields=std::map<std::string,torch::Tensor>;
  auto fields=[](int64_t rows,bool deleted){Fields f;f["observations"]=torch::sin(torch::arange(rows*288,torch::kFloat64).reshape({rows,3,32,3})*.11);f["feature_mask"]=torch::ones({rows,3,32,3},torch::kBool);
    std::vector<int64_t> labels;std::vector<std::string> ids;for(int64_t i=0;i<rows;++i){labels.push_back((i/2)%2?1-i%2:i%2);const auto source=rows==128?3*(i/2)+2:i/2+(i/2)/2;
      ids.push_back("structured-hard-timing-comparison-v2/lag_sign/structured-hard-timing-v1/seed-75272/lag_sign/source-"+std::to_string(source));}
    f["labels_scoring_only"]=torch::tensor(labels,torch::kInt64);f["source_ids_json"]=embedding::archive::text_tensor(strings(ids));
    if(deleted){f["requested_erasure"]=torch::zeros({rows,3,32,3},torch::kBool);f["requested_erasure"].narrow(2,0,1).fill_(true);f["feature_mask"].narrow(2,0,1).fill_(false);f["observations"].narrow(2,0,1).zero_();}return f;};
  auto save=[&](const std::string&name,const Fields&f){const auto path=root/(name+".serialized-archive");require(!fs::exists(path),"new artificial loader archive");torch::serialize::OutputArchive a;for(const auto&[k,v]:f)a.write(k,v,true);a.save_to(path.string());return path;};
  std::map<std::pair<int64_t,bool>,ev::ControlledDataset> loaded;
  for(const auto rows:{256,128})for(const auto deleted:{false,true})loaded.emplace(std::make_pair(rows,deleted),load_controlled(save("valid-"+std::to_string(rows)+(deleted?"-deleted":""),fields(rows,deleted)),rows,75272,deleted));
  // The deleted synthetic observations must equal intact observations on their
  // retained support; both were built from the same deterministic values.
  check_splits(loaded.at({256,false}),loaded.at({128,false}),loaded.at({128,true}));
  int rejected=0;for(const auto defect:{0,1,2}){auto f=fields(128,false);if(defect==0)f["feature_mask"]=f.at("feature_mask").to(torch::kInt64);
    if(defect==1)f["observed"]=f.at("observations");
    if(defect==2)f["labels_scoring_only"].index_put_({1},0);
    const auto p=save("invalid-"+std::to_string(defect),f);bool failed=false;try{(void)load_controlled(p,128,75272);}catch(const std::exception&){failed=true;}require(failed,"serialized malformed loader fixture must reject");++rejected;}
  require(rejected==3,"all artificial loader negatives");const auto &t=loaded.at({256,false}).observed;return {t.data.narrow(0,0,16).clone(),t.feature_mask.narrow(0,0,16).clone()};
}
} // namespace architecture_screen
