// SPDX-License-Identifier: MIT
// CPU evaluation of saved features only. No encoder header or forward exists here.
#include "embedding/shared/fixed_feature_readouts.h"
#include "frozen_role_guard.h"
#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <regex>
#include <sys/stat.h>

#ifndef CONFIGURABLE_CAPACITY_SOURCE_ID
#define CONFIGURABLE_CAPACITY_SOURCE_ID "unrecorded"
#endif
#ifndef CONFIGURABLE_CAPACITY_CARD_SHA256
#define CONFIGURABLE_CAPACITY_CARD_SHA256 "unrecorded"
#endif
namespace {
namespace ev=embedding::evaluation;
namespace fi=ev::frozen_inputs;
namespace fs=std::filesystem;
using fi::require;using fi::quote;using fi::bytes;using fi::sha256;using fi::write_new;
constexpr const char* protocol="configurable-capacity-v1";
constexpr int64_t masters[]{941101,941102,941103};
constexpr int64_t repetitions[]{2701,2802,2903};
torch::Tensor read(torch::serialize::InputArchive&a,const std::string&key){torch::Tensor t;a.read(key,t,true);return t;}
std::string text(torch::serialize::InputArchive&a,const std::string&key){return embedding::archive::tensor_text(read(a,key));}
void put(torch::serialize::OutputArchive&a,const std::string&key,const std::string&value){a.write(key,embedding::archive::text_tensor(value),true);}
void direct(const fs::path&p,bool file=false){require(p.is_absolute()&&fs::canonical(p)==p&&!fs::is_symlink(p),"direct absolute input: "+p.string());
  if(file){struct stat s{};require(::stat(p.c_str(),&s)==0&&S_ISREG(s.st_mode)&&s.st_nlink==1,"single-link regular input");}}
void new_leaf(const fs::path&p){require(p.is_absolute()&&p.lexically_normal()==p&&!fs::exists(p)&&!fs::is_symlink(p),"new absolute output");direct(p.parent_path());require(fs::create_directory(p),"reserve output");}
void save(const fs::path&p,torch::serialize::OutputArchive&a){require(!fs::exists(p),"new archive required");embedding::archive::save_archive(p.string(),a);}
struct Split{embedding::Batch batch;torch::Tensor labels,source_numeric;std::vector<std::string>ids;};
std::vector<std::string> source_strings(const std::string&s){
  // Closed producer IDs have no escapes. Reject general JSON rather than guessing.
  require(s.size()>=2&&s.front()=='['&&s.back()==']',"source ID JSON array");std::vector<std::string>out;size_t i=1;
  while(i<s.size()-1){require(s[i]=='"',"source ID quote");const auto end=s.find('"',i+1);require(end!=std::string::npos,"source ID closing quote");
    const auto value=s.substr(i+1,end-i-1);require(!value.empty()&&value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/")==std::string::npos,"closed source ID string");out.push_back(value);i=end+1;
    if(i<s.size()-1){require(s[i++]==',',"source ID separator");require(i<s.size()-1,"no trailing source ID separator");}}
  return out;
}
Split load_split(const fs::path&p,int64_t rows){direct(p,true);torch::serialize::InputArchive a;a.load_from(p.string(),torch::kCPU);
  const auto keys=a.keys();require(std::set<std::string>(keys.begin(),keys.end())==std::set<std::string>{"observed","feature_mask","labels_scoring_only","source_ids","source_ids_json"},"exact controlled input keys");
  Split s{{read(a,"observed"),read(a,"feature_mask")},read(a,"labels_scoring_only"),read(a,"source_ids"),source_strings(text(a,"source_ids_json"))};
  const std::vector<int64_t>shape{rows,3,128,9};require(s.batch.data.device().is_cpu()&&s.batch.data.scalar_type()==torch::kFloat64&&s.batch.data.sizes()==torch::IntArrayRef(shape)&&s.batch.data.is_contiguous(),"CPU64 legal input geometry");
  require(s.batch.feature_mask.device().is_cpu()&&s.batch.feature_mask.scalar_type()==torch::kBool&&s.batch.feature_mask.sizes()==torch::IntArrayRef(shape),"Boolean legal feature mask");
  require(s.labels.scalar_type()==torch::kInt64&&s.labels.sizes()==torch::IntArrayRef({rows})&&s.source_numeric.scalar_type()==torch::kInt64&&s.source_numeric.sizes()==torch::IntArrayRef({rows})&&s.ids.size()==size_t(rows),"CPU label/source row association");
  require(torch::isfinite(s.batch.data).all().item<bool>()&&s.batch.data.masked_select(~s.batch.feature_mask).eq(0).all().item<bool>(),"finite input and zero hidden storage");
  std::set<std::string>groups;for(int64_t r=0;r<rows;r+=2){require(s.ids[r]==s.ids[r+1]&&groups.insert(s.ids[r]).second&&s.source_numeric[r].item<int64_t>()==s.source_numeric[r+1].item<int64_t>(),"complete unique paired sources");
    require(s.ids[r]=="TEMPO-5/"+std::to_string(s.source_numeric[r].item<int64_t>()),"numeric/text source identity");
    require(s.labels[r].item<int64_t>()>=0&&s.labels[r].item<int64_t>()<=1&&s.labels[r+1].item<int64_t>()==1-s.labels[r].item<int64_t>()&&torch::equal(s.batch.feature_mask[r],s.batch.feature_mask[r+1]),"opposite paired classes and label-independent masks");}
  return s;
}
struct Cohort{Split train,val,deleted;};
void check_splits(const Cohort&c){const std::set<std::string>t(c.train.ids.begin(),c.train.ids.end()),v(c.val.ids.begin(),c.val.ids.end());for(const auto&id:t)require(!v.count(id),"TRAIN/VAL source leakage");
  require(c.val.ids==c.deleted.ids&&torch::equal(c.val.labels,c.deleted.labels)&&torch::equal(c.val.source_numeric,c.deleted.source_numeric),"deleted role association");
  require(!c.deleted.batch.feature_mask.logical_and(~c.val.batch.feature_mask).any().item<bool>()&&torch::equal(c.deleted.batch.data,torch::where(c.deleted.batch.feature_mask,c.val.batch.data,torch::zeros_like(c.val.batch.data))),"deleted exact observation subset");}
ev::FeatureSurface raw(const ev::ObservationScaler&s,const embedding::Batch&b){auto z=s.transform(b);return {torch::cat({z.data.flatten(1),z.feature_mask.flatten(1).to(torch::kFloat64)},1),b.feature_mask.flatten(1).any(1),"TRAIN-only observation scaler; legal raw values plus feature mask"};}
ev::FixedFeatureReadoutRun heads(const fs::path&out,int64_t master,const Cohort&c){ev::FixedFeatureReadoutRun r;r.output_directory=out.string();r.master_seed=master;r.training_labels=c.train.labels;r.validation_labels=c.val.labels;r.training_source_ids=c.train.ids;r.validation_source_ids=c.val.ids;return r;}
std::string baseline(const fs::path&dir,int64_t master,const Cohort&c,const ev::FeatureSurface&t,const ev::FeatureSurface&v,const ev::FeatureSurface&d,const ev::FeatureNormalizer&outer,int64_t width){
  require(fs::create_directory(dir),"new baseline directory");auto h=heads(dir/"readouts",master,c);
  if(!width)h.methods.push_back({"raw",t,v,d,true,{}});
  else{
    const auto zero=[&](const ev::FeatureSurface&s){return ev::FeatureSurface{torch::zeros({s.values.size(0),width},torch::kFloat64),s.valid.clone(),"unsupported TRAIN-only raw PCA"};};
    try{ev::TrainPca pca(t,width);torch::serialize::OutputArchive a;a.write("feature_mean",outer.mean,true);a.write("feature_scale",outer.scale,true);a.write("fitted_rows",torch::tensor(outer.fitted_rows),true);
      a.write("pca_mean",pca.mean,true);a.write("pca_components",pca.components,true);a.write("pca_singular_values",pca.singular_values,true);a.write("pca_numerical_rank",torch::tensor(pca.numerical_rank),true);save(dir/"pca-preprocessing.pt",a);
      h.methods.push_back({"pca_only",pca.transform(t),pca.transform(v),pca.transform(d),true,{}});
    }catch(const std::exception&e){const std::string why=e.what();require(why.find("PCA dimensions exceed")!=std::string::npos,"unexpected PCA failure: "+why);require(!fs::exists(dir/"pca-preprocessing.pt"),"PCA failure after publication");h.methods.push_back({"pca_only",zero(t),zero(v),zero(d),true,why});}
  }
  return ev::run_fixed_feature_readouts(h);
}
ev::FixedFeatureMethod native(const fs::path&p,const std::string&profile,int64_t master,int64_t point,const Cohort&c,const std::string&input_sha,const std::string&source_sha,const std::string&card_sha){
  direct(p,true);torch::serialize::InputArchive a;a.load_from(p.string(),torch::kCPU);const int64_t width=profile=="C"?64:32;
  require(text(a,"protocol")==protocol&&text(a,"profile")==profile&&read(a,"master").item<int64_t>()==master&&read(a,"point").item<int64_t>()==point&&read(a,"native_width").item<int64_t>()==width,"native artifact geometry/cohort/point");
  const auto tag="RPB-v20.alt-0"+std::to_string(profile=="A"?1:profile=="B"?2:3);require(text(a,"model_tag")==tag,"native model tag");
  require(text(a,"input_manifest_sha")==input_sha&&text(a,"source_fingerprint")==source_sha&&text(a,"card_sha")==card_sha,"native immutable input/source/card binding");
  const auto surface=[&](const std::string&prefix,const Split&s){ev::FeatureSurface f{read(a,prefix+"_values"),read(a,prefix+"_valid"),std::string(protocol)+"/"+tag+"/master-"+std::to_string(master)+"/point-"+std::to_string(point)+";saved CUDA native export"};
    ev::validate_features(f);require(f.values.scalar_type()==torch::kFloat64&&f.values.sizes()==torch::IntArrayRef({s.labels.size(0),width})&&torch::isfinite(f.values).all().item<bool>(),"saved native CPU64 geometry/finiteness");
    require(torch::equal(f.valid,s.batch.feature_mask.flatten(1).any(1)),"native validity exact observed support");return f;};
  return {"native_"+profile+"_"+std::to_string(point),surface("training",c.train),surface("validation",c.val),surface("deleted",c.deleted),false,{}};
}
void close(const torch::Tensor&a,const torch::Tensor&b,const std::string&why){require(a.scalar_type()==b.scalar_type()&&a.sizes()==b.sizes(),why+" geometry");
  if(a.is_floating_point())require(torch::allclose(a,b,2e-9,2e-9),why+" arithmetic");else require(torch::equal(a,b),why+" exact classes/support");}
ev::FeatureSurface load_surface(const fs::path&p){torch::serialize::InputArchive a;a.load_from(p.string(),torch::kCPU);return {read(a,"features"),read(a,"valid"),text(a,"provenance")};}
void audit_baselines(const fs::path&dir,const Cohort&c){torch::serialize::InputArchive a;a.load_from((dir/"raw-preprocessing.pt").string(),torch::kCPU);const auto&b=c.train.batch;const auto count=b.feature_mask.sum(std::vector<int64_t>{0,2}).to(torch::kFloat64);const auto safe=torch::where(b.feature_mask,b.data,torch::zeros_like(b.data));const auto mean=safe.sum(std::vector<int64_t>{0,2})/count;
  const auto centered=torch::where(b.feature_mask,b.data-mean.unsqueeze(0).unsqueeze(2),torch::zeros_like(b.data));const auto scale=(centered.pow(2).sum(std::vector<int64_t>{0,2})/count).sqrt().clamp_min(1e-8);close(read(a,"counts"),count,"legal TRAIN raw scaler count");close(read(a,"mean"),mean,"legal TRAIN raw scaler mean");close(read(a,"scale"),scale,"legal TRAIN raw scaler scale");require(text(a,"source_ids_json")==fi::strings(c.train.ids),"raw scaler TRAIN source association");
  const auto unscaled=[&](const embedding::Batch&x){auto z=torch::where(x.feature_mask,(x.data-mean.unsqueeze(0).unsqueeze(2))/scale.unsqueeze(0).unsqueeze(2),torch::zeros_like(x.data));return torch::cat({z.flatten(1),x.feature_mask.flatten(1).to(torch::kFloat64)},1);};const auto rt=unscaled(c.train.batch);const auto om=rt.mean(0),os=(rt-om).pow(2).mean(0).sqrt().clamp_min(1e-8);close(read(a,"feature_mean"),om,"legal TRAIN raw outer mean");close(read(a,"feature_scale"),os,"legal TRAIN raw outer scale");
  for(const auto&entry:std::vector<std::pair<std::string,const Split*>>{{"training",&c.train},{"validation-intact",&c.val},{"validation-deleted",&c.deleted}}){const auto normalized=(unscaled(entry.second->batch)-om)/os;close(load_surface(dir/"baselines/raw/readouts/raw"/(entry.first+"-features.pt")).values,normalized,"legal raw preprocessing "+entry.first);
    for(const int64_t width:{32,64}){const auto root=dir/"baselines"/("pca"+std::to_string(width));if(!fs::exists(root/"pca-preprocessing.pt"))continue;torch::serialize::InputArchive p;p.load_from((root/"pca-preprocessing.pt").string(),torch::kCPU);close(read(p,"feature_mean"),om,"shared raw PCA outer mean");close(read(p,"feature_scale"),os,"shared raw PCA outer scale");const auto pc=read(p,"pca_components"),pm=read(p,"pca_mean");close(load_surface(root/"readouts/pca_only"/(entry.first+"-features.pt")).values,(normalized-pm).matmul(pc),"saved raw PCA projection "+entry.first);
      if(entry.first=="training"){const auto x=(rt-om)/os;close(pm,x.mean(0),"legal TRAIN raw PCA mean");close(pc.transpose(0,1).matmul(pc),torch::eye(width,pc.options()),"PCA orthonormal columns");const auto xc=x-pm,sv=read(p,"pca_singular_values").narrow(0,0,width);const auto lhs=xc.transpose(0,1).matmul(xc.matmul(pc)),rhs=pc*sv.pow(2).unsqueeze(0);require(torch::allclose(lhs,rhs,2e-8,2e-8),"TRAIN PCA covariance eigenvector witness; no SVD rerun");require(read(p,"fitted_rows").item<int64_t>()==512&&read(p,"pca_numerical_rank").item<int64_t>()>=width,"raw PCA fitting population/rank");}}
  }
}
std::string score_json(const ev::Score&s){std::ostringstream o;o<<std::setprecision(17)<<"{\"total\":"<<s.total<<",\"valid\":"<<s.valid<<",\"correct\":"<<s.correct<<",\"accuracy\":";if(s.supported)o<<s.accuracy;else o<<"null";o<<",\"coverage\":"<<s.coverage<<",\"full_population_correctness\":"<<double(s.correct)/s.total<<'}';return o.str();}
std::string interval_json(const ev::GroupedInterval&i){std::ostringstream o;o<<std::setprecision(17)<<"{\"source_groups\":"<<i.source_groups<<",\"estimate\":";if(i.source_groups)o<<i.estimate;else o<<"null";o<<",\"lower\":";if(i.supported)o<<i.lower;else o<<"null";o<<",\"upper\":";if(i.supported)o<<i.upper;else o<<"null";return o.str()+"}";}
uint64_t named(const std::string&s){uint64_t h=14695981039346656037ULL;for(unsigned char c:s){h^=c;h*=1099511628211ULL;}return h;}
int64_t audit_directory(const fs::path&root,fs::path output={}){direct(root);int64_t fits=0;std::ostringstream replay;replay<<std::setprecision(17)<<"{\"protocol\":\"saved-capacity-arithmetic-v1\",\"encoder_calls\":0,\"head_fits\":0,\"records\":[";bool first=true;
  for(const auto&entry:fs::recursive_directory_iterator(root)){if(entry.path().filename()!="fit.pt")continue;const auto dir=entry.path().parent_path();const auto method=dir.parent_path();
    torch::serialize::InputArchive f;f.load_from(entry.path().string(),torch::kCPU);const bool outer=read(f,"outer_normalizer_applied").item<bool>();const auto train=load_surface(method/"training-features.pt");
    const auto validrows=torch::nonzero(train.valid).flatten();const auto x=train.values.index_select(0,validrows);auto mean=x.mean(0),scale=(x-mean).pow(2).mean(0).sqrt().clamp_min(1e-8);
    if(outer){close(read(f,"feature_mean"),mean,"TRAIN outer mean");close(read(f,"feature_scale"),scale,"TRAIN outer scale");}
    const auto fitted=outer?(x-mean)/scale:x;mean=fitted.mean(0);scale=(fitted-mean).pow(2).mean(0).sqrt().clamp_min(1e-8);
    for(const std::string head:{"ridge","tiny"}){close(read(f,head+"_mean"),mean,"TRAIN probe mean");close(read(f,head+"_scale"),scale,"TRAIN probe scale");}
    require(read(f,"fitted_rows").item<int64_t>()==validrows.numel(),"fitted TRAIN row count");
    for(const std::string view:{"training","validation-intact","validation-deleted"}){const auto s=load_surface(method/(view+"-features.pt"));torch::serialize::InputArchive p;p.load_from((dir/(view+"-predictions.pt")).string(),torch::kCPU);
      auto input=torch::where(s.valid.unsqueeze(1),outer?(s.values-read(f,"feature_mean"))/read(f,"feature_scale"):s.values,torch::zeros_like(s.values));close(read(p,"probe_input_features"),input,"saved outer map");close(read(p,"valid"),s.valid,"saved prediction validity");
      const auto normalized=[&](const std::string&head){return torch::where(s.valid.unsqueeze(1),(input-read(f,head+"_mean"))/read(f,head+"_scale"),torch::zeros_like(input));};
      const auto ridge=normalized("ridge").matmul(read(f,"ridge_weights"))+read(f,"ridge_intercept");const auto hidden=normalized("tiny").matmul(read(f,"tiny_w1"))+read(f,"tiny_b1");const auto tiny=torch::tanh(hidden).matmul(read(f,"tiny_w2"))+read(f,"tiny_b2");
      close(read(p,"ridge_logits"),ridge,"saved ridge logits");close(read(p,"tiny_hidden_preactivation"),hidden,"saved neural hidden");close(read(p,"tiny_logits"),tiny,"saved neural logits");close(read(p,"ridge"),ridge.argmax(1),"saved ridge decision");close(read(p,"tiny_secondary"),tiny.argmax(1),"saved neural decision");
      torch::serialize::InputArchive original;original.load_from((method/(view+"-features.pt")).string(),torch::kCPU);close(read(p,"labels_scoring_only"),read(original,"labels_scoring_only"),"surface/prediction scoring label association");require(text(p,"source_ids_json")==text(original,"source_ids_json"),"surface/prediction source association");
      const auto truth=read(p,"labels_scoring_only");const auto ids=source_strings(text(p,"source_ids_json"));const auto name=method.filename().string(),rep=dir.filename().string();const auto master_dir=method.parent_path().filename()=="models"?method.parent_path().parent_path():method.parent_path().parent_path().parent_path().parent_path();
      const auto master_name=master_dir.filename().string();const int64_t master=master_name.rfind("master-",0)==0?std::stoll(master_name.substr(7)):941000;const auto seed=ev::stream_seed(master,named(name+"/"+rep+"/"+view));
      if(!first){replay<<',';}
      first=false;replay<<"{\"prediction\":"<<quote(fs::relative(dir/(view+"-predictions.pt"),root).string())<<",\"ridge\":"<<score_json(ev::score(ridge.argmax(1),truth,s.valid))<<",\"tiny_secondary\":"<<score_json(ev::score(tiny.argmax(1),truth,s.valid));
      if(view!="training"){
        replay<<",\"bootstrap_seed_decimal\":"<<quote(std::to_string(seed))<<",\"ridge_grouped_interval\":"<<interval_json(ev::grouped_accuracy_interval(ridge.argmax(1),truth,s.valid,ids,seed,1000))<<",\"tiny_grouped_interval\":"<<interval_json(ev::grouped_accuracy_interval(tiny.argmax(1),truth,s.valid,ids,seed,1000));
      }
      replay<<'}';
    }++fits;
  }require(fits>0,"no saved fitted heads to audit");replay<<"],\"fitted_pairs_replayed\":"<<fits<<"}";if(!output.empty())write_new(output,replay.str()+"\n");return fits;
}
void self_test(const fs::path&out){new_leaf(out);torch::manual_seed(941000);Cohort c;const auto split=[](int64_t groups,const std::string&prefix){Split s;std::vector<int64_t>label;for(int64_t i=0;i<groups;++i){s.ids.push_back(prefix+std::to_string(i));s.ids.push_back(prefix+std::to_string(i));label.push_back(0);label.push_back(1);}s.labels=torch::tensor(label,torch::kInt64);return s;};c.train=split(8,"train-");c.val=split(4,"val-");c.deleted=c.val;
  const auto f=[](const Split&s){auto x=torch::randn({s.labels.size(0),4},torch::kFloat64);x.select(1,0).add_(s.labels.to(torch::kFloat64)*2-1);return ev::FeatureSurface{x,torch::ones({s.labels.size(0)},torch::kBool),"artificial CPU readout engineering only"};};auto t=f(c.train),v=f(c.val),d=v;d.values=v.values.clone();d.values.select(1,1).zero_();auto r=heads(out/"readouts",941000,c);r.methods.push_back({"engineering",t,v,d,false,{}});ev::run_fixed_feature_readouts(r);require(audit_directory(out)==3,"three engineering fitted repetitions");std::cout<<"CPU saved-feature engineering passed; 3 fits replayed, no encoder calls\n";}
}
int main(int argc,char**argv){try{torch::set_num_threads(1);std::map<std::string,std::string>a;for(int i=1;i<argc;i+=2)require(i+1<argc&&a.emplace(argv[i],argv[i+1]).second,"unique CLI key/value");
  if(a.count("--self-test")){require(a.size()==1,"closed self-test CLI");self_test(a.at("--self-test"));return 0;}
  if(a.count("--audit-only")){require(a.size()==1,"closed audit CLI");std::cout<<"Saved CPU arithmetic audit passed: "<<audit_directory(a.at("--audit-only"))<<" fits, no fitting or encoder calls\n";return 0;}
  const std::set<std::string>required{"--data-root","--producer-root","--output","--inputs-sha256","--sources-sha256","--card-sha256"};require(a.size()==required.size(),"closed quality CLI");for(const auto&k:required)require(a.count(k),"missing CLI "+k);
  const fs::path data=a.at("--data-root"),producer=a.at("--producer-root"),output=a.at("--output");direct(data);direct(producer);const auto input_sha=sha256(bytes(a.at("--inputs-sha256"))),source_sha=sha256(bytes(a.at("--sources-sha256"))),card_sha=a.at("--card-sha256");
  require(fi::is_sha(card_sha)&&card_sha==CONFIGURABLE_CAPACITY_CARD_SHA256&&source_sha==CONFIGURABLE_CAPACITY_SOURCE_ID,"compiled prospective source/card SHA");std::map<std::string,fs::path>roles;for(const auto master:masters)for(const std::string view:{"training","validation","validation-deleted"}){const auto name="master-"+std::to_string(master)+"/"+view+".pt";roles[name]=data/name;}fi::Guard guard(a.at("--inputs-sha256"),roles);
  std::map<int64_t,Cohort>cohorts;std::map<fs::path,std::string>feature_hashes;for(const auto master:masters){const auto pre="master-"+std::to_string(master);Cohort c{load_split(data/pre/"training.pt",512),load_split(data/pre/"validation.pt",256),load_split(data/pre/"validation-deleted.pt",256)};check_splits(c);cohorts.emplace(master,std::move(c));
    for(const std::string p:{"A","B","C"})for(const int64_t point:{0,512}){const auto file=producer/"profiles"/p/pre/("point-"+std::to_string(point))/"features.pt";direct(file,true);feature_hashes[file]=sha256(bytes(file));}}
  new_leaf(output);int64_t pca_fits=0;std::ostringstream report;report<<std::setprecision(17)<<"{\"protocol\":"<<quote(protocol)<<",\"dataset\":\"TEMPO-5\",\"complexity\":5,\"complexity_max\":5,\"masters\":[941101,941102,941103],\"source_fingerprint\":"<<quote(source_sha)<<",\"card_sha\":"<<quote(card_sha)<<",\"input_manifest_sha\":"<<quote(input_sha)<<",\"encoder_calls\":0,\"native_PCA_fits\":0,\"baseline_cohorts\":3,\"raw_scaler_fits\":3,\"raw_outer_normalizer_fits\":3,\"raw_PCA_fit_attempts\":6,\"cohorts\":[";bool first=true;
  for(const auto master:masters){const auto&c=cohorts.at(master);const auto dir=output/("master-"+std::to_string(master));require(fs::create_directory(dir)&&fs::create_directory(dir/"baselines"),"reserve master/baselines");ev::ObservationScaler scaler(c.train.batch);auto rt=raw(scaler,c.train.batch),rv=raw(scaler,c.val.batch),rd=raw(scaler,c.deleted.batch);ev::FeatureNormalizer outer(rt);rt=outer.transform(rt);rv=outer.transform(rv);rd=outer.transform(rd);
    torch::serialize::OutputArchive scale;scale.write("mean",scaler.mean,true);scale.write("scale",scaler.scale,true);scale.write("counts",scaler.counts,true);scale.write("feature_mean",outer.mean,true);scale.write("feature_scale",outer.scale,true);scale.write("fitted_rows",torch::tensor(outer.fitted_rows),true);put(scale,"source_ids_json",fi::strings(c.train.ids));save(dir/"raw-preprocessing.pt",scale);
    const auto start=std::chrono::steady_clock::now();const auto raw_json=baseline(dir/"baselines/raw",master,c,rt,rv,rd,outer,0),p32=baseline(dir/"baselines/pca32",master,c,rt,rv,rd,outer,32),p64=baseline(dir/"baselines/pca64",master,c,rt,rv,rd,outer,64);
    pca_fits+=fs::exists(dir/"baselines/pca32/pca-preprocessing.pt")+fs::exists(dir/"baselines/pca64/pca-preprocessing.pt");auto h=heads(dir/"models",master,c);for(const std::string p:{"A","B","C"})for(const int64_t point:{0,512})h.methods.push_back(native(producer/"profiles"/p/("master-"+std::to_string(master))/("point-"+std::to_string(point))/"features.pt",p,master,point,c,input_sha,source_sha,card_sha));
    for(const std::string p:{"A","B","C"}){
      h.comparisons.push_back({"native_"+p+"_0","native_"+p+"_512"});
    }
    h.comparisons.push_back({"native_A_512","native_B_512"});h.comparisons.push_back({"native_A_512","native_C_512"});const auto model_json=ev::run_fixed_feature_readouts(h);audit_baselines(dir,c);
    if(!first){report<<',';}
    first=false;report<<"{\"master\":"<<master<<",\"CPU_head_and_PCA_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"raw\":"<<raw_json<<",\"pca32\":"<<p32<<",\"pca64\":"<<p64<<",\"models\":"<<model_json<<'}';std::cout<<"Fixed CPU readouts completed master "<<master<<std::endl;
  }
  report<<"],\"saved_feature_inputs\":[";first=true;
  for(const auto&[p,digest]:feature_hashes){
    require(sha256(bytes(p))==digest,"producer features changed during readout");
    if(!first){report<<',';}
    first=false;report<<"{\"path\":"<<quote(p.string())<<",\"sha256\":"<<quote(digest)<<'}';
  }
  report<<"],\"raw_PCA_fits\":"<<pca_fits<<"}";guard.verify();write_new(output/"report.json",report.str()+"\n");const auto fits=audit_directory(output,output/"arithmetic-audit.json");require(fits<=81,"at most 81 fixed paired TRAIN fits across 3 masters and9 methods");write_new(output/"complete.json","{\"protocol\":"+quote(protocol)+",\"status\":\"complete\",\"saved_arithmetic_fits_replayed\":"+std::to_string(fits)+",\"report_sha256\":"+quote(sha256(bytes(output/"report.json")))+"}\n");std::cout<<"Capacity saved-feature readouts and CPU arithmetic audit passed;"<<fits<<" fits;zero encoder calls\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
