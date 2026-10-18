// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Additive application runner. The completed pilot and portable models stay fixed.
#include "cuwacunu/config/document.h"
#include "cuwacunu/training/market_data.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_state.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;

namespace {
namespace fs=std::filesystem;
namespace enc=cuwacunu::encoder::primary;
namespace r=embedding::encoders::raw_patch_bottleneck_mae;
namespace ev=cuwacunu::evaluation;
namespace md=cuwacunu::training::market_data;
using Clock=std::chrono::steady_clock;
using Pin=std::pair<fs::path,std::string>;
void check(bool v,const std::string &why){TORCH_CHECK(v,"[market study] ",why);}
std::string sha256(const fs::path &path){
  check(fs::is_regular_file(path),"missing provenance file: "+path.string());
  int fds[2];check(::pipe(fds)==0,"checksum pipe");posix_spawn_file_actions_t actions;
  check(::posix_spawn_file_actions_init(&actions)==0,"checksum spawn actions");
  ::posix_spawn_file_actions_adddup2(&actions,fds[1],STDOUT_FILENO);
  ::posix_spawn_file_actions_addclose(&actions,fds[0]);::posix_spawn_file_actions_addclose(&actions,fds[1]);
  std::string name=path.string();char executable[]="sha256sum",separator[]="--";
  char *argv[]={executable,separator,name.data(),nullptr};pid_t pid;
  const int rc=::posix_spawnp(&pid,executable,&actions,nullptr,argv,environ);
  ::posix_spawn_file_actions_destroy(&actions);::close(fds[1]);
  if(rc){::close(fds[0]);check(false,"checksum spawn failed");}
  std::string output;char buffer[512];ssize_t n;
  while((n=::read(fds[0],buffer,sizeof buffer))>0)output.append(buffer,n);
  ::close(fds[0]);int status=0;
  check(::waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0&&output.size()>=64,"checksum failed");
  const auto hash=output.substr(0,64);
  check(hash.find_first_not_of("0123456789abcdef")==std::string::npos,"malformed SHA256");return hash;
}
std::string json(const std::string &s){
  std::ostringstream out;out<<'"';for(unsigned char c:s){
    if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;
  }out<<'"';return out.str();
}
std::string read_text(const fs::path &p){std::ifstream f(p,std::ios::binary);check(bool(f),"read "+p.string());return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
void write(const fs::path &p,const std::string &text){
  r::configurable_checkpoint_detail::StagedCheckpoint staged(p.string());
  std::ofstream f(staged.path(),std::ios::binary);check(bool(f),"open staged text");f<<text;f.close();check(bool(f),"write staged text");staged.publish();
}
void copy(const fs::path &from,const fs::path &to,const std::string &hash){
  check(sha256(from)==hash,"changed copied source: "+from.string());
  check(fs::copy_file(from,to),"exclusive copy destination");check(sha256(to)==hash,"copy hash mismatch");
}
std::vector<Pin> ledger(const fs::path &root,const fs::path &file,bool verify){
  std::istringstream input(read_text(file));std::string line;std::vector<Pin> result;std::set<std::string> names;
  while(std::getline(input,line)){
    check(line.size()>66&&line.substr(64,2)=="  "&&line.substr(0,64).find_first_not_of("0123456789abcdef")==std::string::npos,"malformed ledger");
    const fs::path relative=line.substr(66);
    check(!relative.is_absolute()&&relative.lexically_normal()==relative&&relative.string().find("..") == std::string::npos&&names.insert(relative.string()).second,"unsafe/duplicate ledger member");
    result.emplace_back(root/relative,line.substr(0,64));
    if(verify)check(sha256(result.back().first)==result.back().second,"ledger changed: "+relative.string());
  }check(!result.empty(),"empty ledger");return result;
}
void verify(const std::vector<Pin> &pins){for(const auto &[path,hash]:pins)check(sha256(path)==hash,"protected file changed: "+path.string());}
std::string closure(){
  std::vector<fs::path> files;
  for(const auto &root:{"code/data/include","code/data/src","code/config/include","code/integration/include",
      "code/encoders/rpb_v19/include","code/encoders/rpb_spectral/include","code/encoders/rpb_content/include","code/training/include"})
    for(const auto &entry:fs::recursive_directory_iterator(root))if(entry.is_regular_file())files.push_back(entry.path());
  for(const auto &p:{"code/training/src/market_study_main.cpp","code/training/market_study.mk",
      "code/evaluation/include/cuwacunu/evaluation/fixed_regression.h","code/evaluation/src/fixed_regression.cpp",
      "Makefile","code/scripts/task.sh"})files.emplace_back(p);
  std::sort(files.begin(),files.end());std::string text;
  for(const auto &p:files)text+=sha256(p)+"  "+p.generic_string()+"\n";return text;
}
std::string identity_json(const r::TrainingIdentity &i,const std::string &scaler){
  return "{\"input_source\":"+json(i.input_source_identity)+",\"split\":"+json(i.split_identity)+",\"recipe\":"+json(i.recipe_identity)+",\"scaler\":"+json(scaler)+"}\n";
}
struct Study{
  cuwacunu::config::Document document;enc::Config model;fs::path config,settings,card,manifest,output,shared,parent;
  std::string mode,profile,dataset,instance,config_sha,settings_sha,card_sha,manifest_sha,binary_sha,source_closure,source_identity;
  std::vector<fs::path> csv;std::vector<Pin> input_pins,parent_pins;
  int64_t tb,te,vb,ve,unused,train_n,probe_train_n,probe_val_n,batch,updates,max_attempts,init_seed,row_seed,mask_seed,context_seed,query_seed;
  std::vector<int64_t> checkpoints;
  Study(const fs::path &path,const std::string &requested_mode,const std::string &requested_profile):document(cuwacunu::config::Document::read(path)),config(fs::absolute(path)){
    document.reject_unknown_blocks({"STUDY","CONTINUATION"});const auto &b=document.block("STUDY");
    b.reject_unknown_keys({"MODE","PROFILE","DATASET_ID","INSTANCE_TAG","ENCODER_SETTINGS","BTC_CSV","ETH_CSV","SOL_CSV",
      "BTC_SHA256","ETH_SHA256","SOL_SHA256","INPUT_MANIFEST","CARD","OUTPUT_ROOT","SHARED_ROOT","TRAIN_BEGIN_CLOSE_MS","TRAIN_END_CLOSE_MS",
      "VALIDATION_BEGIN_CLOSE_MS","VALIDATION_END_CLOSE_MS","UNUSED_BEGIN_CLOSE_MS","TRAIN_WINDOWS","PROBE_TRAIN_WINDOWS","PROBE_VALIDATION_WINDOWS",
      "BATCH_SIZE","UPDATES","MAX_ATTEMPTS","CHECKPOINT_UPDATES","INITIALIZATION_SEED","ROW_SEED","MASK_SEED","CONTEXT_SEED","QUERY_SEED"});
    mode=b.string("MODE");profile=b.string("PROFILE");dataset=b.string("DATASET_ID");instance=b.string("INSTANCE_TAG");
    check(mode==requested_mode&&profile==requested_profile&&(mode=="continuation"||mode=="capacity"),"CLI must match fixed study mode/profile");
    check(profile=="base"||profile=="wide"||profile=="large"||profile=="long","unknown profile");
    settings=document.resolve_path(b.string("ENCODER_SETTINGS"));model=enc::read_settings(settings);
    card=document.resolve_path(b.string("CARD"));manifest=document.resolve_path(b.string("INPUT_MANIFEST"));output=document.resolve_path(b.string("OUTPUT_ROOT"));
    tb=b.integer("TRAIN_BEGIN_CLOSE_MS");te=b.integer("TRAIN_END_CLOSE_MS");vb=b.integer("VALIDATION_BEGIN_CLOSE_MS");ve=b.integer("VALIDATION_END_CLOSE_MS");unused=b.integer("UNUSED_BEGIN_CLOSE_MS");
    train_n=b.integer("TRAIN_WINDOWS");probe_train_n=b.integer("PROBE_TRAIN_WINDOWS");probe_val_n=b.integer("PROBE_VALIDATION_WINDOWS");
    batch=b.integer("BATCH_SIZE");updates=b.integer("UPDATES");max_attempts=b.integer("MAX_ATTEMPTS");checkpoints=b.integers("CHECKPOINT_UPDATES");
    init_seed=b.integer("INITIALIZATION_SEED");row_seed=b.integer("ROW_SEED");mask_seed=b.integer("MASK_SEED");context_seed=b.integer("CONTEXT_SEED");query_seed=b.integer("QUERY_SEED");
    check(tb==1704067259999&&te==1705881599999&&vb==1705881659999&&ve==1706486399999&&unused==1706486459999,"fixed January chronological roles");
    check(train_n==4096&&batch==8&&updates==2048&&max_attempts==4096&&checkpoints==std::vector<int64_t>({512,1024,2048})&&
      init_seed==942001&&row_seed==942002&&mask_seed==942003&&context_seed==942004&&query_seed==942005,"fixed budgets/seeds/checkpoints");
    const auto H=profile=="long"?256:128,W=profile=="wide"?128:64,D=profile=="large"?64:32;
    check(model.channel_count==3&&model.channel_ids==std::vector<int64_t>({0,1,2})&&model.input_width==9&&model.history_length==H&&model.encoder_width==W&&model.export_width==D&&
      model.patch_length==8&&model.num_layers==3&&model.num_heads==4&&model.feedforward_width==256&&model.decoder_hidden_width==128&&model.mask_ratio==.25&&model.huber_delta==1.&&
      model.scale_floor==1e-6&&model.layer_norm_epsilon==1e-5&&model.sampling_interval==60000&&model.device==torch::Device(torch::kCUDA,0),"fixed profile geometry/settings");
    if(mode=="continuation"){
      check(profile=="base"&&dataset=="MKT-HIST-1"&&instance=="RPB-v22.market-01.alt-01"&&probe_train_n==192&&probe_val_n==64&&!b.contains("SHARED_ROOT")&&document.blocks().size()==2,"continuation contract");
      const auto &p=document.block("CONTINUATION");p.reject_unknown_keys({"PARENT_RUN","PARENT_STATE_SHA256","PARENT_IDENTITIES_SHA256","PARENT_SELECTIONS_SHA256","PARENT_INITIALIZED_LEDGER_SHA256","PARENT_INFERENCE_SHA256"});
      parent=document.resolve_path(p.string("PARENT_RUN"));check(parent.filename()=="run-Ova1QQ","fixed parent run");
      for(const auto &[key,name]:std::vector<std::pair<std::string,std::string>>{{"PARENT_STATE_SHA256","checkpoints/training-512.pt"},{"PARENT_IDENTITIES_SHA256","identities.json"},{"PARENT_SELECTIONS_SHA256","selections.pt"},{"PARENT_INITIALIZED_LEDGER_SHA256","initialized-assets.sha256"},{"PARENT_INFERENCE_SHA256","checkpoints/inference-512.pt"}}){
        const auto hash=p.string(key);check(hash.size()==64&&sha256(parent/name)==hash,"parent pin mismatch: "+key);parent_pins.emplace_back(parent/name,hash);
      }
      check(parent_pins.front().second=="d68c0f83f4fc66acb57935f6f806ffa7c6fc80d3ef8e35ee88a7cf77f78f458e","exact original training state");
    }else{
      const std::string tag=profile=="base"?"RPB-v22.market-02":profile=="wide"?"RPB-v22.market-02.alt-01":profile=="large"?"RPB-v22.market-02.alt-02":"RPB-v22.market-02.alt-03";
      check(dataset=="MKT-HIST-2"&&instance==tag&&probe_train_n==112&&probe_val_n==32&&document.blocks().size()==1,"fresh capacity contract");shared=document.resolve_path(b.string("SHARED_ROOT"));
    }
    const std::array<std::string,3> raw_hashes{"67745d6acfcc42c3828802cea7bacdcd747ca033e3d3f3d6d49e0637cc6c6e58","0aad9d117ba86a58086b185064b41a92d7906c68fef9cebc7bc38cf009c2abaa","364fbf2c2e395b001b31d5c2daff6153f01812f679cb5002e52bd850c06e55a7"};
    size_t j=0;for(const std::string symbol:{"BTC","ETH","SOL"}){
      const auto file=document.resolve_path(b.string(symbol+"_CSV"));const auto hash=sha256(file);check(hash==b.string(symbol+"_SHA256"),"CSV changed: "+symbol);
      const fs::path cache=cuwacunu::data::raw_binary_for_csv(file.string());check(cuwacunu::data::cache_file_is_strictly_newer(cache.string(),file.string()),"existing cache must be fresh; rebuilds are forbidden");
      check(sha256(cache)==raw_hashes[j++],"fixed original mmap cache changed");csv.push_back(file);input_pins.emplace_back(file,hash);input_pins.emplace_back(cache,sha256(cache));
      source_identity+=symbol+":"+hash+";";
    }
    manifest_sha=sha256(manifest);check(manifest_sha=="828154b335670d4d97aeaac2f4a3839099abcdd31ef8bec1f031f54ff8ed3cd8","original input manifest required");
    source_identity+="manifest:"+manifest_sha;for(size_t j=0;j<csv.size();++j)source_identity+=";mmap:"+csv[j].filename().string()+":"+input_pins[j*2+1].second;
    config_sha=sha256(config);settings_sha=sha256(settings);card_sha=sha256(card);
    check(card_sha==(mode=="continuation"?"2dfbe0b43f3a6ab2542da3a482435edd45de9be7fe68796c66fec050fa896bd6":"dc64e6cdc81ff67cef0d8c127e910afb1f8e7b44f1212eb47a51f29b2c52e67b"),"frozen prospective card changed");
    binary_sha=sha256("/proc/"+std::to_string(::getpid())+"/exe");source_closure=closure();
  }
  enc::Config common_config()const{auto c=model;c.history_length=256;c.encoder_width=64;c.export_width=32;return c;}
  std::string roles()const{return std::to_string(tb)+":"+std::to_string(te)+":"+std::to_string(vb)+":"+std::to_string(ve)+":"+std::to_string(unused);}
  std::string common_text()const{return "MKT-HIST-2-shared-v1\nsource:"+source_identity+"\ncard:"+card_sha+"\nbinary:"+binary_sha+"\nroles:"+roles()+"\n4096:112:32;anchorH256;targetH128;B8;CFwindow-weighted-full-H256;P8-one-query-patch-per-channel;query942005\n"+source_closure;}
  void final_check()const{verify(input_pins);verify(parent_pins);check(sha256(config)==config_sha&&sha256(settings)==settings_sha&&sha256(card)==card_sha&&sha256(manifest)==manifest_sha&&closure()==source_closure&&sha256("/proc/"+std::to_string(::getpid())+"/exe")==binary_sha,"frozen execution bindings changed");}
};
void open_dataset(md::Dataset &dataset,const Study &p,const enc::Config &c){for(const auto &file:p.csv)dataset.add_dataset(file.string(),c.history_length,0,"none",false);verify(p.input_pins);}
enc::Input recent(const enc::Input &raw){return {raw.data.narrow(2,raw.data.size(2)-128,128),raw.observed.narrow(2,raw.observed.size(2)-128,128),raw.channel_ids,raw.endpoints,raw.sampling_interval};}
r::FrozenScaler fit_scaler(md::Dataset &dataset,const Study &p,const std::vector<int64_t> &anchors,const enc::Config &c){
  r::StreamingScaler stream(enc::input_config(c));for(size_t i=0;i<anchors.size();i+=p.batch)stream.add_train_batch(md::load_batch(dataset,md::slice(anchors,i,p.batch),p.tb,p.te,c));return stream.finish();
}
std::vector<int64_t> anchor_rows(torch::serialize::InputArchive &a,const char *key,const std::vector<int64_t> &expected){
  torch::Tensor t;a.read(key,t,true);check(t.scalar_type()==torch::kInt64&&t.dim()==1&&torch::equal(t,torch::tensor(expected,torch::kInt64)),"exact selected anchors: "+std::string(key));return expected;
}
void save_selections(const fs::path &path,const Study &p,const std::vector<int64_t> &train,const std::vector<int64_t> &pt,const std::vector<int64_t> &pv){
  torch::serialize::OutputArchive a;a.write("train_anchors",torch::tensor(train,torch::kInt64),true);a.write("probe_train_anchors",torch::tensor(pt,torch::kInt64),true);a.write("probe_validation_anchors",torch::tensor(pv,torch::kInt64),true);
  a.write("roles_and_strides",torch::tensor(std::vector<int64_t>{p.tb,p.te,p.vb,p.ve,p.unused,60000,256,128,256*60000},torch::kInt64),true);r::configurable_checkpoint_detail::publish_checkpoint(a,path.string());
}
void validate_selections(const fs::path &path,const Study &p,const std::vector<int64_t> &train,const std::vector<int64_t> &pt,const std::vector<int64_t> &pv,bool legacy){
  torch::serialize::InputArchive a;a.load_from(path.string(),torch::kCPU);anchor_rows(a,"train_anchors",train);anchor_rows(a,"probe_train_anchors",pt);anchor_rows(a,"probe_validation_anchors",pv);
  torch::Tensor t;a.read("roles_and_strides",t,true);const auto expected=legacy?std::vector<int64_t>{p.tb,p.te,p.vb,p.ve,p.unused,60000,128,128*60000}:std::vector<int64_t>{p.tb,p.te,p.vb,p.ve,p.unused,60000,256,128,256*60000};
  check(t.scalar_type()==torch::kInt64&&torch::equal(t,torch::tensor(expected,torch::kInt64)),"exact role/selection geometry");
}
const std::vector<std::string> shared_assets{"selections.pt","scaler.pt","query-plan-train.pt","query-plan-validation.pt","shared-common.txt","summary.json",
  "features/raw_recent128-train.pt","features/raw_recent128-validation.pt","features/raw_full256-train.pt","features/raw_full256-validation.pt","features/train-targets.pt","features/validation-targets.pt"};
void raw_panel(md::Dataset &dataset,const Study &p,const enc::Config &c,const r::FrozenScaler &scaler,const std::vector<int64_t> &rows,int64_t begin,int64_t end,const std::string &role){
  std::vector<torch::Tensor> raw128,raw256,flags128,flags256,targets,tvalid,hidden,qtarget,eligible;
  auto qc=c;qc.history_length=128;qc.mask_ratio=1./16.;
  for(size_t i=0;i<rows.size();i+=p.batch){
    const auto raw=md::load_batch(dataset,md::slice(rows,i,p.batch),begin,end,c);const auto last=recent(raw);
    const auto mu=scaler.mean.view({1,3,1,9}),sigma=scaler.scale.view({1,3,1,9});
    const auto scaled=torch::where(raw.observed,(raw.data.to(torch::kFloat64)-mu)/sigma,torch::zeros_like(raw.data));
    raw256.push_back(torch::cat({scaled.flatten(1),raw.observed.to(torch::kFloat64).flatten(1)},1));
    raw128.push_back(torch::cat({scaled.narrow(2,128,128).flatten(1),last.observed.to(torch::kFloat64).flatten(1)},1));
    flags128.push_back(last.observed.flatten(1).any(1));flags256.push_back(raw.observed.flatten(1).any(1));
    const auto labels=ev::past_window_targets(last.data,last.observed);targets.push_back(labels.values);tvalid.push_back(labels.valid);
    const auto plan=enc::make_training_mask(last.observed,qc,r::training_counter_seed(p.query_seed,static_cast<int64_t>(i/p.batch)));
    hidden.push_back(plan.hidden);qtarget.push_back(plan.target);eligible.push_back(plan.eligible_channels);
  }
  const auto ids=torch::tensor(rows,torch::kInt64);const auto directory=p.shared/"features";
  ev::save_features((directory/("raw_recent128-"+role+".pt")).string(),{torch::cat(raw128),torch::cat(flags128),ids});
  ev::save_features((directory/("raw_full256-"+role+".pt")).string(),{torch::cat(raw256),torch::cat(flags256),ids});
  ev::save_targets((directory/(role+"-targets.pt")).string(),{{torch::cat(targets),torch::cat(tvalid)},ids,ids.clone()});
  torch::serialize::OutputArchive a;a.write("row_ids",ids,true);a.write("hidden_recent128",torch::cat(hidden),true);a.write("target_recent128",torch::cat(qtarget),true);a.write("eligible_channels",torch::cat(eligible),true);
  r::configurable_checkpoint_detail::publish_checkpoint(a,(p.shared/("query-plan-"+role+".pt")).string());
}
r::FrozenScaler common_bank(const Study &p,const std::vector<int64_t> &train,const std::vector<int64_t> &pt,const std::vector<int64_t> &pv,std::vector<Pin> &pins){
  const auto c=p.common_config();
  if(p.profile=="base"){
    fs::create_directories(p.shared.parent_path());check(fs::create_directory(p.shared),"common bank must be a new exclusive leaf; partial banks are preserved");check(fs::create_directory(p.shared/"features"),"shared feature leaf");
    md::Dataset dataset;open_dataset(dataset,p,c);const auto scaler=fit_scaler(dataset,p,train,c);
    save_selections(p.shared/"selections.pt",p,train,pt,pv);torch::serialize::OutputArchive a,s;
    a.write("kind",r::configurable_checkpoint_detail::text_tensor("market_study_common_CF_scaler_v1"),true);a.write("identity",r::configurable_checkpoint_detail::text_tensor(scaler.identity()),true);scaler.save(s);a.write("scaler",s);r::configurable_checkpoint_detail::publish_checkpoint(a,(p.shared/"scaler.pt").string());
    write(p.shared/"shared-common.txt",p.common_text());raw_panel(dataset,p,c,scaler,pt,p.tb,p.te,"train");raw_panel(dataset,p,c,scaler,pv,p.vb,p.ve,"validation");
    write(p.shared/"summary.json","{\"kind\":\"market_capacity_common_bank_v1\",\"dataset\":\"MKT-HIST-2\",\"anchor_history\":256,\"target_history\":128,\"scaler_fit_windows\":4096,\"scaler_weighting\":\"observed cells across overlapping selected full H256 TRAIN windows; duplicate source ticks retain their declared window weights\",\"probe_train_rows\":112,\"probe_validation_rows\":32,\"raw_recent128_size\":6912,\"raw_full256_size\":13824,\"scaler_identity\":"+json(scaler.identity())+",\"owner_config_sha256\":"+json(p.config_sha)+",\"card_sha256\":"+json(p.card_sha)+",\"input_manifest_sha256\":"+json(p.manifest_sha)+",\"binary_sha256\":"+json(p.binary_sha)+",\"role_boundaries\":"+json(p.roles())+"}\n");
    std::string lines;for(const auto &name:shared_assets)lines+=sha256(p.shared/name)+"  "+name+"\n";write(p.shared/"shared-assets.sha256",lines);p.final_check();
  }
  check(read_text(p.shared/"shared-common.txt")==p.common_text(),"common bank source/card/bin/role/recipe mismatch");
  pins=ledger(p.shared,p.shared/"shared-assets.sha256",true);check(pins.size()==shared_assets.size(),"complete fixed shared bank");
  for(size_t j=0;j<shared_assets.size();++j)check(pins[j].first==p.shared/shared_assets[j],"shared bank exact member/order");
  pins.emplace_back(p.shared/"shared-assets.sha256",sha256(p.shared/"shared-assets.sha256"));validate_selections(p.shared/"selections.pt",p,train,pt,pv,false);
  torch::serialize::InputArchive a,s;a.load_from((p.shared/"scaler.pt").string(),torch::kCPU);check(r::configurable_checkpoint_detail::read_text(a,"kind")=="market_study_common_CF_scaler_v1","typed common scaler");a.read("scaler",s);
  auto scaler=r::FrozenScaler::load(s);scaler.validate(enc::input_config(c));check(scaler.identity()==r::configurable_checkpoint_detail::read_text(a,"identity"),"common scaler bytes");return scaler;
}
struct QueryPlan{torch::Tensor hidden,target,eligible;};
QueryPlan query_plan(const Study &p,const std::string &role,const std::vector<int64_t> &rows){
  torch::serialize::InputArchive a;a.load_from((p.shared/("query-plan-"+role+".pt")).string(),torch::kCPU);
  torch::Tensor ids;a.read("row_ids",ids,true);check(ids.scalar_type()==torch::kInt64&&torch::equal(ids,torch::tensor(rows,torch::kInt64)),"query plan row association");
  QueryPlan q;a.read("hidden_recent128",q.hidden,true);a.read("target_recent128",q.target,true);a.read("eligible_channels",q.eligible,true);
  for(const auto &t:{q.hidden,q.target})check(t.scalar_type()==torch::kBool&&t.sizes()==torch::IntArrayRef({static_cast<int64_t>(rows.size()),3,128,9}),"query recent128 geometry/type");
  check(q.eligible.scalar_type()==torch::kBool&&q.eligible.sizes()==torch::IntArrayRef({static_cast<int64_t>(rows.size()),3}),"query eligible shape");return q;
}
struct Capture{
  double mae{0},feature_seconds{0},query_seconds{0};int64_t valid{0},total{0},cells{0},encoder_calls{0};
  std::vector<double> row_mae;std::vector<int64_t> row_cells,row_channels;
};
std::string capture_json(const Capture &c){
  std::ostringstream s;s<<std::setprecision(17)<<"{\"mae_train_scaled\":"<<c.mae<<",\"eligible_rows\":"<<c.valid<<",\"total_rows\":"<<c.total<<",\"target_cells\":"<<c.cells
    <<",\"target_history\":128,\"feature_seconds_including_io\":"<<c.feature_seconds<<",\"query_seconds_including_io\":"<<c.query_seconds<<",\"CUDA_encoder_calls\":"<<c.encoder_calls<<",\"CPU_encoder_calls\":0,\"row_mae\":[";
  for(size_t i=0;i<c.row_mae.size();++i){if(i)s<<',';s<<c.row_mae[i];}s<<"],\"row_target_cells\":[";
  for(size_t i=0;i<c.row_cells.size();++i){if(i)s<<',';s<<c.row_cells[i];}s<<"],\"row_eligible_channels\":[";
  for(size_t i=0;i<c.row_channels.size();++i){if(i)s<<',';s<<c.row_channels[i];}s<<"]}";return s.str();
}
Capture capture(md::Dataset &dataset,const Study &p,const std::vector<int64_t> &rows,int64_t begin,int64_t end,
    enc::Encoder &model,const r::FrozenScaler &scaler,const fs::path &directory,const std::string &role,const std::string &method){
  torch::NoGradGuard guard;model->eval();std::vector<torch::Tensor> values,flags;Capture result;
  QueryPlan shared;if(p.mode=="capacity")shared=query_plan(p,role,rows);
  for(size_t offset=0;offset<rows.size();offset+=p.batch){
    torch::cuda::synchronize();auto started=Clock::now();
    const auto raw=md::load_batch(dataset,md::slice(rows,offset,p.batch),begin,end,p.model);const auto normalized=enc::transform(scaler,raw,p.model);
    check(normalized.data.is_cuda(),"actual CUDA feature input");const auto encoded=model->encode(normalized);const auto native=enc::native_embedding(encoded,p.model);
    check(native.is_cuda()&&native.sizes()==torch::IntArrayRef({raw.data.size(0),p.model.export_width}),"actual configured CUDA native export");
    values.push_back(native.to(torch::kCPU));flags.push_back(encoded.sample_valid_mask.to(torch::kCPU));++result.encoder_calls;
    torch::cuda::synchronize();result.feature_seconds+=std::chrono::duration<double>(Clock::now()-started).count();started=Clock::now();
    const auto B=raw.data.size(0);auto qc=p.model;qc.history_length=128;qc.mask_ratio=1./16.;
    const auto recent_observed=normalized.observed.narrow(2,p.model.history_length-128,128);
    const auto expected=enc::make_training_mask(recent_observed,qc,r::training_counter_seed(p.query_seed,static_cast<int64_t>(offset/p.batch)));
    torch::Tensor hidden_recent=expected.hidden,target_recent=expected.target,eligible=expected.eligible_channels;
    if(p.mode=="capacity"){
      hidden_recent=shared.hidden.narrow(0,offset,B).to(p.model.device);target_recent=shared.target.narrow(0,offset,B).to(p.model.device);eligible=shared.eligible.narrow(0,offset,B).to(p.model.device);
      check(torch::equal(hidden_recent,expected.hidden)&&torch::equal(target_recent,expected.target)&&torch::equal(eligible,expected.eligible_channels),"common query plan agrees with actual recent source support");
    }
    auto hidden=torch::zeros_like(normalized.observed);hidden.narrow(2,p.model.history_length-128,128).copy_(hidden_recent);
    const auto masks=enc::mask_from_hidden(normalized.observed,hidden,p.model);
    check(torch::equal(masks.target.narrow(2,p.model.history_length-128,128),target_recent)&&torch::equal(masks.eligible_channels,eligible)&&
      (p.model.history_length==128||!masks.target.narrow(2,0,p.model.history_length-128).any().item<bool>()),"only common recent128 target patch is hidden");
    const enc::Input visible{torch::where(masks.visible,normalized.data,torch::zeros_like(normalized.data)),masks.visible,normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    const auto prediction=model->decode(enc::native_embedding(model->encode(visible),p.model),normalized.channel_ids);++result.encoder_calls;
    check(prediction.is_cuda()&&torch::isfinite(prediction).all().item<bool>(),"finite CUDA query decoder");
    const auto counts=masks.target.sum(std::vector<int64_t>{2,3});
    const auto channel_mae=torch::where(masks.target,(prediction-normalized.data).abs(),torch::zeros_like(prediction)).sum(std::vector<int64_t>{2,3})/counts.clamp_min(1).to(torch::kFloat32);
    const auto channels=eligible.sum(1);const auto valid=channels.gt(0);
    const auto row=(channel_mae*eligible.to(torch::kFloat32)).sum(1)/channels.clamp_min(1).to(torch::kFloat32);
    const auto cpu_row=row.to(torch::kCPU),cpu_cells=counts.sum(1).to(torch::kCPU),cpu_channels=channels.to(torch::kCPU);
    for(int64_t j=0;j<B;++j){result.row_mae.push_back(cpu_row[j].item<double>());result.row_cells.push_back(cpu_cells[j].item<int64_t>());result.row_channels.push_back(cpu_channels[j].item<int64_t>());}
    result.mae+=row.masked_select(valid).sum().item<double>();result.valid+=valid.sum().item<int64_t>();result.total+=B;result.cells+=counts.sum().item<int64_t>();
    torch::cuda::synchronize();result.query_seconds+=std::chrono::duration<double>(Clock::now()-started).count();
  }
  const auto started=Clock::now();ev::save_features((directory/(method+"-"+role+".pt")).string(),{torch::cat(values),torch::cat(flags),torch::tensor(rows,torch::kInt64)});
  result.feature_seconds+=std::chrono::duration<double>(Clock::now()-started).count();check(result.valid>0,"query eligible population");result.mae/=result.valid;return result;
}
void copy_parent(const Study &p,const fs::path &run,std::vector<Pin> &pins){
  const std::vector<std::string> names{"features/initialized_v22-train.pt","features/initialized_v22-validation.pt","features/raw_mask-train.pt","features/raw_mask-validation.pt",
    "features/train-targets.pt","features/validation-targets.pt","checkpoints/inference-0.pt","initial-query.json","selections.pt"};
  const auto initialized=ledger(p.parent,p.parent/"initialized-assets.sha256",true);check(initialized.size()==names.size(),"parent initialized ledger exact size");
  for(size_t j=0;j<names.size();++j){check(initialized[j].first==p.parent/names[j],"parent initialized ledger exact order");copy(initialized[j].first,run/names[j],initialized[j].second);pins.push_back(initialized[j]);}
  copy(p.parent/"initialized-assets.sha256",run/"initialized-assets.sha256",sha256(p.parent/"initialized-assets.sha256"));
  copy(p.parent/"checkpoints/inference-512.pt",run/"checkpoints/inference-512.pt",p.parent_pins.back().second);
  copy(p.parent/"checkpoints/training-512.pt",run/"checkpoints/parent-training-512.pt",p.parent_pins.front().second);
  for(const std::string role:{"train","validation"}){
    const auto from=p.parent/"features"/("trained_v22-"+role+".pt");const auto hash=sha256(from);copy(from,run/"features"/("parent_512_v22-"+role+".pt"),hash);pins.emplace_back(from,hash);
  }
  for(const auto &name:{"summary.json","identities.json","source-closure.sha256"}){const auto file=p.parent/name;const auto hash=sha256(file);copy(file,run/(std::string("parent-")+name),hash);pins.emplace_back(file,hash);}
}
r::LoadedContentTrainingState restore_parent(const Study &p,const r::FrozenScaler &scaler,const std::vector<int64_t> &train,const std::vector<int64_t> &pt,const std::vector<int64_t> &pv){
  validate_selections(p.parent/"selections.pt",p,train,pt,pv,true);
  verify(ledger(fs::current_path(),p.parent/"source-closure.sha256",true));
  torch::serialize::InputArchive a,child;a.load_from((p.parent/"checkpoints/training-512.pt").string(),p.model.device);
  namespace d=r::configurable_checkpoint_detail;
  r::TrainingIdentity original{d::read_text(a,"input_source_identity"),d::read_text(a,"split_identity"),d::read_text(a,"recipe_identity")};a.read("encoder",child);
  const auto saved_scaler=d::read_text(child,"scaler_identity");
  check(identity_json(original,saved_scaler)==read_text(p.parent/"identities.json"),"exact hash-bound original identity JSON");
  check(original.input_source_identity==p.source_identity&&saved_scaler==scaler.identity(),"original source/cache/scaler identity remains exact");
  const auto expected_split="full-role-H128-step60000;uniform-integer-anchors-v1;nonoverlap-H-stride-probes-v1;"+p.roles()+";4096:192:64;selection-sha256:"+sha256(p.parent/"selections.pt");
  check(original.split_identity==expected_split,"original role/anchor identity");
  for(const auto &[key,file]:std::vector<std::pair<std::string,fs::path>>{{"pilot-config:",p.parent/"pilot.conf"},{"encoder-config:",p.parent/"encoder.conf"},{"card:",p.parent/"prospective-card.md"},
      {"runner:","code/training/src/market_pilot_main.cpp"},{"resume:","code/encoders/rpb_content/include/embedding/encoders/raw_patch_bottleneck_mae/training_state.h"},
      {"scaler:","code/encoders/rpb_spectral/include/embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h"},{"source-closure:",p.parent/"source-closure.sha256"},{"initialized-assets-sha256:",p.parent/"initialized-assets.sha256"}})
    check(original.recipe_identity.find(key+sha256(file))!=std::string::npos,"original recipe binding: "+key);
  auto restored=r::load_content_training_state((p.parent/"checkpoints/training-512.pt").string(),p.model,original,scaler.identity());
  check(restored.progress.completed_updates==512&&restored.progress.attempted_batches==512&&restored.progress.row_seed==p.row_seed&&restored.progress.mask_seed==p.mask_seed&&restored.progress.context_seed==p.context_seed,"exact original counters and stream bases");return restored;
}
fs::path new_run(const Study &p){
  fs::create_directories(p.output);auto pattern=(p.output/("run-"+p.profile+"-XXXXXX")).string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back('\0');
  check(::mkdtemp(name.data()),"exclusive results leaf");fs::path run=name.data();check(fs::create_directory(run/"features")&&fs::create_directory(run/"checkpoints"),"new output subdirectories");
  copy(p.config,run/"study.conf",p.config_sha);copy(p.settings,run/"encoder.conf",p.settings_sha);copy(p.card,run/"prospective-card.md",p.card_sha);copy(p.manifest,run/"input-manifest.json",p.manifest_sha);
  write(run/"source-closure.sha256",p.source_closure);std::string lines;for(const auto &[path,hash]:p.input_pins)lines+=hash+"  "+path.string()+"\n";write(run/"loaded-inputs.sha256",lines);
  std::cout<<"STARTED_RUN_DIRECTORY="<<run.string()<<std::endl;return run;
}
int execute(int argc,char **argv){
  check(argc==6&&std::string(argv[2])=="--mode"&&std::string(argv[4])=="--profile","usage: market_study CONFIG --mode continuation|capacity --profile base|wide|large|long");
  check(fs::exists("/.dockerenv")&&torch::cuda::is_available(),"managed Linux container and actual CUDA required");torch::set_num_threads(1);torch::set_num_interop_threads(1);
  Study p(argv[1],argv[3],argv[5]);const auto anchor_H=p.mode=="capacity"?256:128;
  const auto train=md::uniform_anchors(p.tb,p.te,anchor_H,p.train_n,false),pt=md::uniform_anchors(p.tb,p.te,anchor_H,p.probe_train_n,true),pv=md::uniform_anchors(p.vb,p.ve,anchor_H,p.probe_val_n,true);
  std::vector<Pin> reused_pins;md::Dataset dataset;open_dataset(dataset,p,p.model);
  r::FrozenScaler scaler=p.mode=="capacity"?common_bank(p,train,pt,pv,reused_pins):fit_scaler(dataset,p,train,p.model);
  auto run=new_run(p);enc::Encoder model{nullptr};std::unique_ptr<torch::optim::AdamW> optimizer;
  r::TrainingProgress progress{0,0,p.row_seed,p.mask_seed,p.context_seed};r::TrainingIdentity identity;
  const auto common_sha=p.mode=="capacity"?sha256(p.shared/"shared-assets.sha256"):sha256(p.parent/"selections.pt");
  const auto recipe="study-config:"+p.config_sha+";encoder-config:"+p.settings_sha+";card:"+p.card_sha+";executable:"+p.binary_sha+";source-closure:"+sha256(run/"source-closure.sha256")+
    ";row-mt19937_64-uniform-i64-v1;mask-splitmix64-attempt-v1;context-coordinate15-base-plus-attempt-v1;B8-adamw-.001-.0001-clip1;seeds942001:942002:942003:942004:942005;target-queryH128";
  if(p.mode=="continuation"){
    auto restored=restore_parent(p,scaler,train,pt,pv);copy_parent(p,run,reused_pins);
    model=restored.model;optimizer=std::move(restored.optimizer);scaler=restored.scaler;progress=restored.progress;identity=restored.identity;
    identity.recipe_identity+=";fork-parent-state:"+p.parent_pins.front().second+";"+recipe;
    write(run/"lineage.json","{\"parent_run\":"+json(p.parent.string())+",\"parent_state_sha256\":"+json(p.parent_pins.front().second)+",\"original_identity_verified_before_fork\":true,\"original_scaler_and_AdamW_preserved\":true,\"start_updates\":512,\"start_attempts\":512,\"reused_controls\":\"raw/init/512 saved banks and reports; no old head refits or encoder replay\"}\n");
    r::save_content_training_state((run/"checkpoints/training-512.pt").string(),model,scaler,*optimizer,identity,progress);
  }else{
    identity={p.source_identity,"shared-anchorH256-targetH128;"+p.roles()+";4096:112:32;shared-ledger:"+common_sha,recipe+";shared-ledger:"+common_sha};
    torch::manual_seed(p.init_seed);model=enc::Encoder(p.model);optimizer=std::make_unique<torch::optim::AdamW>(model->parameters(),r::content_training_adamw_options());
    enc::save_checkpoint((run/"checkpoints/inference-0.pt").string(),model,scaler,0);
    const auto initial_train=capture(dataset,p,pt,p.tb,p.te,model,scaler,run/"features","train","initialized_v22");
    const auto initial_val=capture(dataset,p,pv,p.vb,p.ve,model,scaler,run/"features","validation","initialized_v22");
    write(run/"initial-query.json","{\"train\":"+capture_json(initial_train)+",\"validation\":"+capture_json(initial_val)+"}\n");
    write(run/"shared-bank.txt",p.shared.string()+"\n"+common_sha+"\n");
  }
  write(run/"identities.json",identity_json(identity,scaler.identity()));
  const auto start_updates=progress.completed_updates,start_attempts=progress.attempted_batches;
  int64_t parameters=0,trainable=0;for(const auto &v:model->parameters()){parameters+=v.numel();if(v.requires_grad())trainable+=v.numel();}
  std::ofstream log(run/"training.csv");check(bool(log),"training log");log<<"attempt,update,loss,eligible_examples,target_cells,row_seed,mask_seed,anchors\n"<<std::setprecision(17);
  bool gpu_gradient=false,weight_changed=false;int64_t gradient_updates=0,total_target_cells=0,eligible_exposures=0;double first_loss=0,last_loss=0;
  model->train();torch::cuda::synchronize();const auto started=Clock::now();
  while(progress.completed_updates<p.updates&&progress.attempted_batches<p.max_attempts){
    const auto attempt=progress.attempted_batches,row_stream=r::training_counter_seed(p.row_seed,attempt),mask_stream=r::training_counter_seed(p.mask_seed,attempt);
    std::mt19937_64 rng(row_stream);std::uniform_int_distribution<int64_t> pick(0,train.size()-1);std::vector<int64_t> rows;for(int64_t j=0;j<p.batch;++j)rows.push_back(train[pick(rng)]);
    const auto normalized=enc::transform(scaler,md::load_batch(dataset,rows,p.tb,p.te,p.model),p.model);
    const auto masks=enc::make_training_mask(normalized.observed,p.model,mask_stream);
    if(!masks.eligible_channels.any().item<bool>()){++progress.attempted_batches;log<<attempt<<','<<progress.completed_updates<<",,0,0,"<<row_stream<<','<<mask_stream<<",\"";for(size_t j=0;j<rows.size();++j){if(j)log<<';';log<<rows[j];}log<<"\"\n";continue;}
    const auto context=r::context_deletion::make_plan(masks,normalized.channel_ids,p.model,p.context_seed,attempt,r::ContextDeletionRecipe::coordinate15_v1);
    const enc::Input visible{torch::where(context.visible,normalized.data,torch::zeros_like(normalized.data)),context.visible,normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    optimizer->zero_grad();const auto native=enc::native_embedding(model->encode(visible),p.model);const auto prediction=model->decode(native,normalized.channel_ids);
    const auto objective=r::hierarchical_huber(prediction,normalized.data.detach(),masks.target,masks.eligible_channels,p.model.huber_delta);
    check(normalized.data.is_cuda()&&native.is_cuda()&&prediction.is_cuda()&&objective.loss.is_cuda()&&torch::isfinite(objective.loss).item<bool>(),"finite actual CUDA model/loss");objective.loss.backward();
    for(const auto &v:model->parameters()){check(v.is_cuda(),"all parameters CUDA");if(v.grad().defined())check(v.grad().is_cuda()&&torch::isfinite(v.grad()).all().item<bool>(),"finite actual CUDA gradients");}
    const auto g=model->relation_projection->weight.grad();const bool nonzero=g.defined()&&g.is_cuda()&&g.abs().sum().item<double>()>0;gpu_gradient|=nonzero;if(nonzero)++gradient_updates;
    const auto before=model->relation_projection->weight.detach().clone();torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer->step();
    const bool moved=!torch::equal(before,model->relation_projection->weight.detach());weight_changed|=moved;
    ++progress.completed_updates;++progress.attempted_batches;last_loss=objective.loss.item<double>();
    if(progress.completed_updates==start_updates+1){check(nonzero&&moved,"first segment update must show nonzero CUDA relation gradient and changed relation weights before continuing");first_loss=last_loss;}
    total_target_cells+=objective.target_cell_count;eligible_exposures+=objective.eligible_example_count;
    log<<attempt<<','<<progress.completed_updates<<','<<last_loss<<','<<objective.eligible_example_count<<','<<objective.target_cell_count<<','<<row_stream<<','<<mask_stream<<",\"";
    for(size_t j=0;j<rows.size();++j){if(j)log<<';';log<<rows[j];}log<<"\"\n";
    if(progress.completed_updates%128==0)std::cout<<"update="<<progress.completed_updates<<" attempts="<<progress.attempted_batches<<" loss="<<last_loss<<std::endl;
    if(std::find(p.checkpoints.begin(),p.checkpoints.end(),progress.completed_updates)!=p.checkpoints.end()){
      log.flush();r::save_content_training_state((run/"checkpoints"/("training-"+std::to_string(progress.completed_updates)+".pt")).string(),model,scaler,*optimizer,identity,progress);
    }
  }
  torch::cuda::synchronize();const auto seconds=std::chrono::duration<double>(Clock::now()-started).count();log.close();check(bool(log),"persist training log");
  check(progress.completed_updates==2048&&gpu_gradient&&weight_changed,"complete declared budget with actual CUDA updates");model->validate_state();model->validate_fixed_prior();
  enc::save_checkpoint((run/"checkpoints/inference-2048.pt").string(),model,scaler,progress.completed_updates);
  const auto final_train=capture(dataset,p,pt,p.tb,p.te,model,scaler,run/"features","train","trained_v22"),final_val=capture(dataset,p,pv,p.vb,p.ve,model,scaler,run/"features","validation","trained_v22");
  verify(reused_pins);p.final_check();
  const auto memory=c10::cuda::CUDACachingAllocator::getDeviceStats(0);struct rusage usage{};::getrusage(RUSAGE_SELF,&usage);
  std::ostringstream summary;summary<<std::setprecision(17)<<"{\"kind\":\"market_training_study_v1\",\"dataset\":"<<json(p.dataset)<<",\"instance\":"<<json(p.instance)<<",\"mode\":"<<json(p.mode)<<",\"profile\":"<<json(p.profile)
    <<",\"H\":"<<p.model.history_length<<",\"W\":"<<p.model.encoder_width<<",\"D\":"<<p.model.export_width<<",\"parameters\":"<<parameters<<",\"trainable_parameters\":"<<trainable<<",\"frozen_parameters\":"<<parameters-trainable
    <<",\"completed_updates\":"<<progress.completed_updates<<",\"attempted_batches\":"<<progress.attempted_batches<<",\"segment_updates\":"<<progress.completed_updates-start_updates<<",\"segment_attempts\":"<<progress.attempted_batches-start_attempts
    <<",\"segment_window_presentations\":"<<(progress.attempted_batches-start_attempts)*8<<",\"segment_input_tick_presentations_per_channel\":"<<(progress.attempted_batches-start_attempts)*8*p.model.history_length
    <<",\"segment_eligible_window_exposures\":"<<eligible_exposures<<",\"segment_target_cells\":"<<total_target_cells<<",\"first_segment_loss\":"<<first_loss<<",\"last_segment_loss\":"<<last_loss<<",\"segment_training_seconds_including_checkpoints\":"<<seconds
    <<",\"segment_nonzero_relation_gradient_updates\":"<<gradient_updates<<",\"segment_cuda_gradient_verified\":"<<(gpu_gradient?"true":"false")<<",\"segment_cuda_parameter_update_verified\":"<<(weight_changed?"true":"false")<<",\"restored_adamw_state_verified\":"<<(p.mode=="continuation"?"true":"false")
    <<",\"target_history\":128,\"anchor_history\":"<<anchor_H<<",\"probe_train_rows\":"<<p.probe_train_n<<",\"probe_validation_rows\":"<<p.probe_val_n<<",\"encoder_device\":\"cuda:0\",\"CPU_encoder_calls\":0,\"host_peak_rss_kib\":"<<usage.ru_maxrss
    <<",\"cuda_peak_allocated_bytes\":"<<memory.allocated_bytes[0].peak<<",\"cuda_peak_reserved_bytes\":"<<memory.reserved_bytes[0].peak<<",\"trained_query\":{\"train\":"<<capture_json(final_train)<<",\"validation\":"<<capture_json(final_val)<<"},\"shared_bank\":"<<json(p.mode=="capacity"?p.shared.string():p.parent.string())
    <<",\"shared_or_parent_selection_sha256\":"<<json(common_sha)<<",\"scaler_identity\":"<<json(scaler.identity())<<",\"config_sha256\":"<<json(p.config_sha)<<",\"encoder_settings_sha256\":"<<json(p.settings_sha)<<",\"card_sha256\":"<<json(p.card_sha)<<",\"input_manifest_sha256\":"<<json(p.manifest_sha)<<",\"binary_sha256\":"<<json(p.binary_sha)<<",\"source_closure_sha256\":"<<json(sha256(run/"source-closure.sha256"))<<"}\n";
  write(run/"summary.json",summary.str());std::cout<<"RUN_DIRECTORY="<<run.string()<<std::endl;return 0;
}
} // namespace
int main(int argc,char **argv)try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
