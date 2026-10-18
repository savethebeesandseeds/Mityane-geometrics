// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Separately declared RPB-v25 change-energy objective; preserve all native36/native40 attempts.
#include "cuwacunu/config/document.h"
#include "cuwacunu/encoder/energy_encoder.h"
#include "cuwacunu/training/market_data.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/energy_change_training_state.h"
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
namespace enc=cuwacunu::encoder::content;
namespace r=embedding::encoders::raw_patch_bottleneck_mae;
namespace ev=cuwacunu::evaluation;
namespace tr=r::energy_change_training;
namespace md=cuwacunu::training::market_data;
using Clock=std::chrono::steady_clock;
using Pin=std::pair<fs::path,std::string>;
void check(bool v,const std::string &why){TORCH_CHECK(v,"[market change-energy study] ",why);}
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
      "code/encoders/rpb_v19/include","code/encoders/rpb_spectral/include","code/encoders/rpb_content/include","code/encoders/rpb_changes/include","code/encoders/rpb_separate/include","code/encoders/rpb_energy/include","code/training/include"})
    for(const auto &entry:fs::recursive_directory_iterator(root))if(entry.is_regular_file())files.push_back(entry.path());
  for(const auto &p:{"code/training/src/market_separate_changes_main.cpp","code/training/market_separate_changes.mk",
      "code/encoders/rpb_changes/consecutive_change.mk","code/encoders/rpb_separate/tests/separated_change_test.cpp",
      "code/scripts/run-market-separate-changes.sh",
      "code/evaluation/include/cuwacunu/evaluation/fixed_regression.h","code/evaluation/src/fixed_regression.cpp",
      "code/training/src/market_change_energy_main.cpp","code/training/market_change_energy.mk",
      "code/encoders/rpb_energy/tests/energy_change_test.cpp","code/encoders/rpb_energy/energy_change_test.mk",
      "code/scripts/run-market-change-energy.sh","config/market_studies/change_energy.conf",
      "config/market_studies/encoder_change_energy.conf","doc/MARKET_CHANGE_ENERGY_CARD.md",
      "Makefile","code/scripts/task.sh"})files.emplace_back(p);
  std::sort(files.begin(),files.end());std::string text;
  for(const auto &p:files)text+=sha256(p)+"  "+p.generic_string()+"\n";
  return text;
}
std::string identity_json(const tr::TrainingIdentity &i,const std::string &scaler){
  return "{\"input_source\":"+json(i.input_source_identity)+",\"split\":"+json(i.split_identity)+",\"recipe\":"+json(i.recipe_identity)+",\"scaler\":"+json(scaler)+"}\n";
}

struct Study {
  cuwacunu::config::Document document;enc::Config model;r::EnergyChangeConfig full;fs::path config,settings,card,manifest,output,shared;
  std::string mode="change_energy",profile="base",dataset,instance,config_sha,settings_sha,card_sha,manifest_sha,binary_sha,source_closure,source_identity;
  std::vector<fs::path> csv;std::vector<Pin> input_pins;
  int64_t tb,te,vb,ve,unused,train_n,probe_train_n,probe_val_n,batch,updates,max_attempts,init_seed,row_seed,mask_seed,context_seed,query_seed;
  std::vector<int64_t> checkpoints;
  explicit Study(const fs::path &path):document(cuwacunu::config::Document::read(path)),config(fs::absolute(path)) {
    document.reject_unknown_blocks({"STUDY"});const auto &b=document.block("STUDY");
    b.reject_unknown_keys({"MODE","PROFILE","DATASET_ID","INSTANCE_TAG","ENCODER_SETTINGS","BTC_CSV","ETH_CSV","SOL_CSV",
      "BTC_SHA256","ETH_SHA256","SOL_SHA256","INPUT_MANIFEST","CARD","OUTPUT_ROOT","SHARED_ROOT","TRAIN_BEGIN_CLOSE_MS","TRAIN_END_CLOSE_MS",
      "VALIDATION_BEGIN_CLOSE_MS","VALIDATION_END_CLOSE_MS","UNUSED_BEGIN_CLOSE_MS","TRAIN_WINDOWS","PROBE_TRAIN_WINDOWS","PROBE_VALIDATION_WINDOWS",
      "BATCH_SIZE","UPDATES","MAX_ATTEMPTS","CHECKPOINT_UPDATES","INITIALIZATION_SEED","ROW_SEED","MASK_SEED","CONTEXT_SEED","QUERY_SEED"});
    dataset=b.string("DATASET_ID");instance=b.string("INSTANCE_TAG");
    check(b.string("MODE")=="change_energy" && b.string("PROFILE")=="base" && dataset=="MKT-HIST-2" && instance=="RPB-v25.market-01","fixed candidate mode/profile/tag");
    settings=document.resolve_path(b.string("ENCODER_SETTINGS"));full=r::read_energy_change_settings(settings);model=full.level;
    card=document.resolve_path(b.string("CARD"));manifest=document.resolve_path(b.string("INPUT_MANIFEST"));output=document.resolve_path(b.string("OUTPUT_ROOT"));
    check(output.lexically_normal()==fs::absolute("output/runs/market-change-energy").lexically_normal(),"distinct change-energy output root");
    tb=b.integer("TRAIN_BEGIN_CLOSE_MS");te=b.integer("TRAIN_END_CLOSE_MS");vb=b.integer("VALIDATION_BEGIN_CLOSE_MS");ve=b.integer("VALIDATION_END_CLOSE_MS");unused=b.integer("UNUSED_BEGIN_CLOSE_MS");
    train_n=b.integer("TRAIN_WINDOWS");probe_train_n=b.integer("PROBE_TRAIN_WINDOWS");probe_val_n=b.integer("PROBE_VALIDATION_WINDOWS");
    batch=b.integer("BATCH_SIZE");updates=b.integer("UPDATES");max_attempts=b.integer("MAX_ATTEMPTS");checkpoints=b.integers("CHECKPOINT_UPDATES");
    init_seed=b.integer("INITIALIZATION_SEED");row_seed=b.integer("ROW_SEED");mask_seed=b.integer("MASK_SEED");context_seed=b.integer("CONTEXT_SEED");query_seed=b.integer("QUERY_SEED");
    check(tb==1704067259999&&te==1705881599999&&vb==1705881659999&&ve==1706486399999&&unused==1706486459999,"fixed January chronological roles");
    check(train_n==4096&&batch==8&&updates==2048&&max_attempts==4096&&checkpoints==std::vector<int64_t>({512,1024,2048})&&
      init_seed==942001&&row_seed==942002&&mask_seed==942003&&context_seed==942004&&query_seed==942005,"fixed budgets/seeds/checkpoints");
    check(model.channel_count==3&&model.channel_ids==std::vector<int64_t>({0,1,2})&&model.input_width==9&&model.history_length==128&&model.encoder_width==64&&model.export_width==32&&
      model.patch_length==8&&model.num_layers==3&&model.num_heads==4&&model.feedforward_width==256&&model.decoder_hidden_width==128&&model.mask_ratio==.25&&model.huber_delta==1.&&
      model.scale_floor==1e-6&&model.layer_norm_epsilon==1e-5&&model.sampling_interval==60000&&model.device==torch::Device(torch::kCUDA,0),"fixed candidate geometry");
    check(full.change_width==4 && r::energy_change_native_width(full)==36,"declared separate native36 geometry");check(probe_train_n==112 && probe_val_n==32,"shared probe counts");shared=document.resolve_path(b.string("SHARED_ROOT"));
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
    check(card_sha=="c67e5c37d56e5bc2801a9d32152589e3067ae11f8b18c8c804418b8cca807938","frozen RPB-v25.market-01 card required");
    binary_sha=sha256("/proc/"+std::to_string(::getpid())+"/exe");source_closure=closure();
  }
  void final_check()const { verify(input_pins);check(sha256(config)==config_sha&&sha256(settings)==settings_sha&&sha256(card)==card_sha&&sha256(manifest)==manifest_sha&&closure()==source_closure&&sha256("/proc/"+std::to_string(::getpid())+"/exe")==binary_sha,"frozen candidate bindings changed"); }
};
void open_dataset(md::Dataset &dataset,const Study &p,const enc::Config &c){for(const auto &file:p.csv)dataset.add_dataset(file.string(),c.history_length,0,"none",false);verify(p.input_pins);}
std::vector<int64_t> anchor_rows(torch::serialize::InputArchive &a,const char *key,const std::vector<int64_t> &expected){
  torch::Tensor t;a.read(key,t,true);check(t.scalar_type()==torch::kInt64&&t.dim()==1&&torch::equal(t,torch::tensor(expected,torch::kInt64)),"exact selected anchors: "+std::string(key));return expected;
}
void validate_selections(const fs::path &path,const Study &p,const std::vector<int64_t> &train,const std::vector<int64_t> &pt,const std::vector<int64_t> &pv,bool legacy){
  torch::serialize::InputArchive a;a.load_from(path.string(),torch::kCPU);anchor_rows(a,"train_anchors",train);anchor_rows(a,"probe_train_anchors",pt);anchor_rows(a,"probe_validation_anchors",pv);
  torch::Tensor t;a.read("roles_and_strides",t,true);const auto expected=legacy?std::vector<int64_t>{p.tb,p.te,p.vb,p.ve,p.unused,60000,128,128*60000}:std::vector<int64_t>{p.tb,p.te,p.vb,p.ve,p.unused,60000,256,128,256*60000};
  check(t.scalar_type()==torch::kInt64&&torch::equal(t,torch::tensor(expected,torch::kInt64)),"exact role/selection geometry");
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
    r::EnergyChangeModel &model,const r::FrozenScaler &scaler,const fs::path &directory,const std::string &role,const std::string &method){
  torch::NoGradGuard guard;model->eval();std::vector<torch::Tensor> values,flags;Capture result;
  const auto shared=query_plan(p,role,rows);
  for(size_t offset=0;offset<rows.size();offset+=p.batch){
    torch::cuda::synchronize();auto started=Clock::now();
    const auto raw=md::load_batch(dataset,md::slice(rows,offset,p.batch),begin,end,p.model);const auto normalized=enc::transform(scaler,raw,p.model);
    check(normalized.data.is_cuda(),"actual CUDA feature input");const auto encoded=model->encode(normalized);const auto native=r::energy_change_native(encoded,p.full);
    check(native.is_cuda()&&native.sizes()==torch::IntArrayRef({raw.data.size(0),r::energy_change_native_width(p.full)}),"actual configured CUDA native export");
    values.push_back(native.to(torch::kCPU));flags.push_back(encoded.sample_valid_mask.to(torch::kCPU));++result.encoder_calls;
    torch::cuda::synchronize();result.feature_seconds+=std::chrono::duration<double>(Clock::now()-started).count();started=Clock::now();
    const auto B=raw.data.size(0);auto qc=p.model;qc.history_length=128;qc.mask_ratio=1./16.;
    const auto recent_observed=normalized.observed.narrow(2,p.model.history_length-128,128);
    const auto expected=enc::make_training_mask(recent_observed,qc,tr::training_counter_seed(p.query_seed,static_cast<int64_t>(offset/p.batch)));
    torch::Tensor hidden_recent=expected.hidden,target_recent=expected.target,eligible=expected.eligible_channels;
    {
      hidden_recent=shared.hidden.narrow(0,offset,B).to(p.model.device);target_recent=shared.target.narrow(0,offset,B).to(p.model.device);eligible=shared.eligible.narrow(0,offset,B).to(p.model.device);
      check(torch::equal(hidden_recent,expected.hidden)&&torch::equal(target_recent,expected.target)&&torch::equal(eligible,expected.eligible_channels),"common query plan agrees with actual recent source support");
    }
    auto hidden=torch::zeros_like(normalized.observed);hidden.narrow(2,p.model.history_length-128,128).copy_(hidden_recent);
    const auto masks=enc::mask_from_hidden(normalized.observed,hidden,p.model);
    check(torch::equal(masks.target.narrow(2,p.model.history_length-128,128),target_recent)&&torch::equal(masks.eligible_channels,eligible)&&
      (p.model.history_length==128||!masks.target.narrow(2,0,p.model.history_length-128).any().item<bool>()),"only common recent128 target patch is hidden");
    const enc::Input visible{torch::where(masks.visible,normalized.data,torch::zeros_like(normalized.data)),masks.visible,normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    const auto prediction=model->decode(r::energy_change_native(model->encode(visible),p.full),normalized.channel_ids);++result.encoder_calls;
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

r::FrozenScaler load_shared(const Study &p,const std::vector<int64_t> &train,const std::vector<int64_t> &pt,const std::vector<int64_t> &pv,std::vector<Pin> &pins) {
  const auto file=p.shared/"shared-assets.sha256";
  check(sha256(file)=="dbc208fb7c858909ff6f04e70b939adaab1c3e8c27f8e72f261a620cf1b55e10","exact completed shared bank");
  pins=ledger(p.shared,file,true);check(pins.size()==12,"complete shared bank");pins.emplace_back(file,sha256(file));
  const fs::path previous="output/runs/market-capacity-study/run-base-gryiAV";
  for(const auto &[name,hash]:std::vector<std::pair<std::string,std::string>>{
      {"features/trained_v22-train.pt","c9f0ea724fd53f7189d03d7516e65b1a37c748d7247dc5f3ccea5920a776f323"},
      {"features/trained_v22-validation.pt","c5f9dfe1955a0cff68260024c14d222a146c5dc119a0e6eb1599cdc8eed2f8ae"},
      {"readouts/report.json","56791dc3e5eef9d87c5bd0d36b877bafdae39443d38162513bc7b096c8385b7b"},
      {"checkpoints/inference-2048.pt","e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c"}}) {
    check(sha256(previous/name)==hash,"historical control changed");pins.emplace_back(previous/name,hash);
  }
  const fs::path original="output/runs/market-separate-changes-native36/run-4SauBf";
  for(const auto &[name,hash]:std::vector<std::pair<std::string,std::string>>{
      {"checkpoints/inference-0.pt","2b614ca9866048a850cdeb73dc719c5816fd04bb39e1746b626d1abf1ef68108"},
      {"features/initialized_v24_native36-train.pt","bbe5cfa8f161f89eed503596e7e9ab29e49b6f80b85ee1e293b6dfe27f616ed7"},
      {"features/initialized_v24_native36-validation.pt","0ee14da3be12bab7aea52cfe3324bc2a13d27305b701ee19613c39a6b238ac5a"}}) {
    check(sha256(original/name)==hash,"matched native36 initial artifact changed");pins.emplace_back(original/name,hash);
  }
  validate_selections(p.shared/"selections.pt",p,train,pt,pv,false);
  torch::serialize::InputArchive a,s;a.load_from((p.shared/"scaler.pt").string(),torch::kCPU);
  check(r::configurable_checkpoint_detail::read_text(a,"kind")=="market_study_common_CF_scaler_v1","typed shared scaler");
  a.read("scaler",s);auto scaler=r::FrozenScaler::load(s);scaler.validate(enc::input_config(p.model));
  check(scaler.identity()=="rpb-scaler-v1-fnv1a64-a4b0f9ff085ab969"&&scaler.identity()==r::configurable_checkpoint_detail::read_text(a,"identity"),"exact saved TRAIN scaler");return scaler;
}

std::string vector_json(const torch::Tensor &value,bool integer=false){
  const auto cpu=value.detach().to(torch::kCPU).flatten();std::ostringstream out;out<<std::setprecision(17)<<'[';
  for(int64_t j=0;j<cpu.numel();++j){if(j)out<<',';if(integer)out<<cpu[j].item<int64_t>();else out<<cpu[j].item<double>();}
  out<<']';return out.str();
}
r::MomentRmsScaler fit_moment_scaler(md::Dataset &dataset,const Study &p,const std::vector<int64_t> &train,
    const r::FrozenScaler &scaler,const std::string &source_identity,double &seconds){
  torch::NoGradGuard guard;r::MomentRmsScaler result(p.model,source_identity);
  torch::cuda::synchronize();const auto started=Clock::now();
  for(size_t offset=0;offset<train.size();offset+=p.batch){
    const auto input=enc::transform(scaler,md::load_batch(dataset,md::slice(train,offset,p.batch),p.tb,p.te,p.model),p.model);
    // This pass computes only the existing generic moment bank and eligibility.
    // It never calls the encoder and never applies query/context deletion.
    const auto targets=r::energy_change_moment_targets(input,p.model);
    check(targets.values.is_cuda()&&targets.valid.is_cuda(),"CUDA full-natural TRAIN moment fit");
    result.add(targets);
  }
  result.freeze();result.validate(p.model,&source_identity);torch::cuda::synchronize();
  seconds=std::chrono::duration<double>(Clock::now()-started).count();return result;
}

int execute(int argc,char **argv) {
  check(argc==2,"usage: market_change_energy CONFIG");check(fs::exists("/.dockerenv")&&torch::cuda::is_available(),"managed container and CUDA required");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);Study p(argv[1]);
  const auto train=md::uniform_anchors(p.tb,p.te,256,p.train_n,false),pt=md::uniform_anchors(p.tb,p.te,256,p.probe_train_n,true),pv=md::uniform_anchors(p.vb,p.ve,256,p.probe_val_n,true);
  std::vector<Pin> shared_pins;const auto scaler=load_shared(p,train,pt,pv,shared_pins);md::Dataset dataset;open_dataset(dataset,p,p.model);
  fs::create_directories(p.output);auto pattern=(p.output/"run-XXXXXX").string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
  check(::mkdtemp(name.data()),"exclusive candidate leaf");const fs::path run=name.data();fs::create_directory(run/"features");fs::create_directory(run/"checkpoints");
  copy(p.config,run/"study.conf",p.config_sha);copy(p.settings,run/"encoder.conf",p.settings_sha);copy(p.card,run/"prospective-card.md",p.card_sha);copy(p.manifest,run/"input-manifest.json",p.manifest_sha);
  write(run/"source-closure.sha256",p.source_closure);std::string inputs;for(const auto &[path,hash]:p.input_pins)inputs+=hash+"  "+path.string()+"\n";write(run/"loaded-inputs.sha256",inputs);
  const auto shared_sha=sha256(p.shared/"shared-assets.sha256");
  const std::string moment_source="MKT-HIST-2;full-natural-TRAIN4096;no-query-no-context;shared-ledger:"+shared_sha+
      ";input:"+p.source_identity+";input-scaler:"+scaler.identity()+";existing-adjacent-global-quarter-bank-v1;validity-not-value;RMS-f64-floor1e-6-no-center";
  double moment_fit_seconds=0;const auto moment_scaler=fit_moment_scaler(dataset,p,train,scaler,moment_source,moment_fit_seconds);
  {
    torch::serialize::OutputArchive archive;moment_scaler.save(archive);
    r::configurable_checkpoint_detail::publish_checkpoint(archive,(run/"moment-rms-scaler.pt").string());
  }
  const auto moment_sha=sha256(run/"moment-rms-scaler.pt");
  write(run/"moment-rms-scaler.json","{\"kind\":\"market_change_energy_TRAIN_RMS_fit_v1\",\"TRAIN_anchors\":4096,\"validation_anchors\":0,\"encoder_calls\":0,\"source_identity\":"+json(moment_source)+
      ",\"counts\":"+vector_json(moment_scaler.counts,true)+",\"sumsq\":"+vector_json(moment_scaler.sumsq)+",\"scales\":"+vector_json(moment_scaler.scales)+
      ",\"support\":"+vector_json(moment_scaler.support,true)+",\"fit_seconds_including_io\":"+std::to_string(moment_fit_seconds)+",\"archive_sha256\":"+json(moment_sha)+"}\n");
  tr::TrainingIdentity identity{p.source_identity,"shared-anchorH256-targetH128;4096:112:32;shared-ledger:"+shared_sha,
      "RPB-v25.market-01;native36:protected32+separate4;matched-v24-init:2b614ca9866048a850cdeb73dc719c5816fd04bb39e1746b626d1abf1ef68108;zero-wave-and-moment-decoders;study:"+p.config_sha+";settings:"+p.settings_sha+";card:"+p.card_sha+";binary:"+p.binary_sha+";source:"+sha256(run/"source-closure.sha256")+";parent-v22:e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c;parent-updates2048;branch-updates2048;structural-NoGrad-level-encode-decode;undefined-level-gradients-before-clip-step-v1;row-mask-context-counter-v1;AdamW.001/.0001;B8;Huber1;mask25;context15;total=wave+0.1aux;aux=.5signed+.5energy;target=same-final-visible-bank-detached;validity-not-nonzero;moment-norm:"+moment_sha+";moment-source:"+moment_source+";no-label-loss"};
  torch::manual_seed(p.init_seed);r::EnergyChangeModel model(p.full);
  {
    const auto parent=r::load_separated_change_checkpoint(
        "output/runs/market-separate-changes-native36/run-4SauBf/checkpoints/inference-0.pt",p.model.device,&p.full);
    check(parent.training_updates==0 && parent.scaler.identity()==scaler.identity(),"exact matched native36 init updates/scaler");
    tr::warm_start_from_separated(model,parent.model);
  }
  torch::optim::AdamW optimizer(model->parameters(),tr::content_training_adamw_options());
  tr::FrozenLevelGuard frozen(model);frozen.validate(model,optimizer);
  tr::TrainingProgress progress{0,0,p.row_seed,p.mask_seed,p.context_seed};
  std::cout<<"STARTED_RUN_DIRECTORY="<<run.string()<<std::endl;
  write(run/"identities.json",identity_json(identity,scaler.identity()));r::save_energy_change_checkpoint((run/"checkpoints/inference-0.pt").string(),model,scaler,moment_scaler,0);
  const auto initial_train=capture(dataset,p,pt,p.tb,p.te,model,scaler,run/"features","train","initialized_v25"),initial_val=capture(dataset,p,pv,p.vb,p.ve,model,scaler,run/"features","validation","initialized_v25");
  write(run/"initial-query.json","{\"train\":"+capture_json(initial_train)+",\"validation\":"+capture_json(initial_val)+"}\n");
  for(const std::string role:{"train","validation"}) {
    const auto previous=fs::path("output/runs/market-capacity-study/run-base-gryiAV/features")/("trained_v22-"+role+".pt");
    const auto old=ev::load_features(previous.string()),now=ev::load_features((run/"features"/("initialized_v25-"+role+".pt")).string());
    check(torch::equal(old.values,now.values.narrow(1,0,p.model.export_width))&&torch::equal(old.valid,now.valid)&&torch::equal(old.row_ids,now.row_ids),"exact saved trained v22 prefix parity");
    const auto previous36=fs::path("output/runs/market-separate-changes-native36/run-4SauBf/features")/("initialized_v24_native36-"+role+".pt");
    const auto old36=ev::load_features(previous36.string());
    check(torch::equal(old36.values,now.values)&&torch::equal(old36.valid,now.valid)&&torch::equal(old36.row_ids,now.row_ids),"all36 match saved native36 initialization exactly");
  }
  write(run/"initialization-parity.json","{\"saved_v22_train_and_validation_prefix32_equal\":true,\"saved_v24_initial_train_and_validation_all36_equal\":true,\"old_encoder_replay_calls\":0,\"old_head_refits\":0}\n");
  const auto start_updates=progress.completed_updates;
  int64_t parameters=0,trainable=0;for(const auto &v:model->parameters()){parameters+=v.numel();if(v.requires_grad())trainable+=v.numel();}
  const auto active_parameters=model->encoder->change_projection.numel()+model->encoder->change_decoder.numel()+model->moment_decoder.numel();
  check(parameters==298505&&active_parameters==15984&&parameters-active_parameters==282521,"declared parameter geometry");
  std::ofstream log(run/"training.csv");check(bool(log),"training log");
  log<<"attempt,update,loss,wave_loss,aux_loss,signed_loss,energy_loss,eligible_examples,target_cells,aux_eligible_examples,signed_cells,energy_cells,projection_gradient_nonzero,wave_decoder_gradient_nonzero,moment_decoder_gradient_nonzero,projection_weight_changed,wave_decoder_weight_changed,moment_decoder_weight_changed,row_seed,mask_seed,anchors\n"<<std::setprecision(17);
  bool gpu_gradient=false,weight_changed=false;int64_t gradient_updates=0,decoder_gradient_updates=0,moment_gradient_updates=0,projection_weight_updates=0,decoder_weight_updates=0,moment_weight_updates=0,total_target_cells=0,eligible_exposures=0,aux_eligible_exposures=0,total_signed_cells=0,total_energy_cells=0;double first_loss=0,last_loss=0,last_wave_loss=0,last_aux_loss=0,last_signed_loss=0,last_energy_loss=0;
  model->train();torch::cuda::synchronize();const auto started=Clock::now();
  while(progress.completed_updates<p.updates&&progress.attempted_batches<p.max_attempts){
    const auto attempt=progress.attempted_batches,row_stream=tr::training_counter_seed(p.row_seed,attempt),mask_stream=tr::training_counter_seed(p.mask_seed,attempt);
    std::mt19937_64 rng(row_stream);std::uniform_int_distribution<int64_t> pick(0,train.size()-1);std::vector<int64_t> rows;for(int64_t j=0;j<p.batch;++j)rows.push_back(train[pick(rng)]);
    const auto normalized=enc::transform(scaler,md::load_batch(dataset,rows,p.tb,p.te,p.model),p.model);
    const auto masks=enc::make_training_mask(normalized.observed,p.model,mask_stream);
    if(!masks.eligible_channels.any().item<bool>()){++progress.attempted_batches;log<<attempt<<','<<progress.completed_updates<<",,,,,,0,0,0,0,0,0,0,0,0,0,0,"<<row_stream<<','<<mask_stream<<",\"";for(size_t j=0;j<rows.size();++j){if(j)log<<';';log<<rows[j];}log<<"\"\n";continue;}
    const auto context=r::context_deletion::make_plan(masks,normalized.channel_ids,p.model,p.context_seed,attempt,r::ContextDeletionRecipe::coordinate15_v1);
    const enc::Input visible{torch::where(context.visible,normalized.data,torch::zeros_like(normalized.data)),context.visible,normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    optimizer.zero_grad();const auto native=r::energy_change_native(model->encode(visible),p.full);const auto prediction=model->decode(native,normalized.channel_ids);
    const auto objective=r::hierarchical_huber(prediction,normalized.data.detach(),masks.target,masks.eligible_channels,p.model.huber_delta);
    const auto auxiliary=r::energy_change_moment_loss(model,visible,native,moment_scaler);
    const auto loss=objective.loss+.1*auxiliary.loss;
    check(normalized.data.is_cuda()&&native.is_cuda()&&prediction.is_cuda()&&objective.loss.is_cuda()&&auxiliary.loss.is_cuda()&&loss.is_cuda()&&torch::isfinite(loss).item<bool>(),"finite actual CUDA model/wave/moment/total loss");loss.backward();
    for(const auto &v:model->parameters()){check(v.is_cuda(),"all parameters CUDA");if(v.grad().defined())check(v.grad().is_cuda()&&torch::isfinite(v.grad()).all().item<bool>(),"finite actual CUDA gradients");}
    const auto g=model->encoder->change_projection.grad(),dg=model->encoder->change_decoder.grad(),mg=model->moment_decoder.grad();
    const bool decoder_nonzero=dg.defined()&&dg.is_cuda()&&dg.abs().sum().item<double>()>0;
    const bool moment_nonzero=mg.defined()&&mg.is_cuda()&&mg.abs().sum().item<double>()>0;
    const bool nonzero=g.defined()&&g.is_cuda()&&g.abs().sum().item<double>()>0;
    if(decoder_nonzero)++decoder_gradient_updates;
    if(moment_nonzero)++moment_gradient_updates;
    gpu_gradient|=nonzero;if(nonzero)++gradient_updates;
    frozen.discard_gradients(model,optimizer);
    const auto before=model->encoder->change_projection.detach().clone(),decoder_before=model->encoder->change_decoder.detach().clone(),moment_before=model->moment_decoder.detach().clone();
    torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer.step();
    const bool moved=!torch::equal(before,model->encoder->change_projection.detach()),decoder_moved=!torch::equal(decoder_before,model->encoder->change_decoder.detach()),moment_moved=!torch::equal(moment_before,model->moment_decoder.detach());
    if(moved)++projection_weight_updates;
    if(decoder_moved)++decoder_weight_updates;
    if(moment_moved)++moment_weight_updates;
    weight_changed|=moved&&nonzero;
    ++progress.completed_updates;++progress.attempted_batches;last_loss=loss.item<double>();last_wave_loss=objective.loss.item<double>();last_aux_loss=auxiliary.loss.item<double>();last_signed_loss=auxiliary.signed_loss.item<double>();last_energy_loss=auxiliary.energy_loss.item<double>();
    if(progress.completed_updates==start_updates+1){check(!nonzero&&decoder_nonzero&&decoder_moved&&moment_nonzero&&moment_moved,"first step must open both zero decoders with actual CUDA learning");first_loss=last_loss;}
    if(progress.completed_updates==start_updates+2) {
      check(nonzero&&moved&&decoder_nonzero&&decoder_moved&&moment_nonzero&&moment_moved,"second step must learn projection and both decoders on CUDA before continuing");
    }
    const auto signed_cells=auxiliary.signed_counts.sum().item<int64_t>(),energy_cells=auxiliary.energy_counts.sum().item<int64_t>();
    total_target_cells+=objective.target_cell_count;eligible_exposures+=objective.eligible_example_count;aux_eligible_exposures+=auxiliary.eligible_example_count;total_signed_cells+=signed_cells;total_energy_cells+=energy_cells;
    log<<attempt<<','<<progress.completed_updates<<','<<last_loss<<','<<last_wave_loss<<','<<last_aux_loss<<','<<last_signed_loss<<','<<last_energy_loss<<','<<objective.eligible_example_count<<','<<objective.target_cell_count<<','<<auxiliary.eligible_example_count<<','<<signed_cells<<','<<energy_cells<<','<<nonzero<<','<<decoder_nonzero<<','<<moment_nonzero<<','<<moved<<','<<decoder_moved<<','<<moment_moved<<','<<row_stream<<','<<mask_stream<<",\"";
    for(size_t j=0;j<rows.size();++j){if(j)log<<';';log<<rows[j];}log<<"\"\n";
    if(progress.completed_updates%128==0) {
      frozen.validate(model,optimizer);
      std::cout<<"update="<<progress.completed_updates<<" attempts="<<progress.attempted_batches<<" loss="<<last_loss<<std::endl;
    }
    if(std::find(p.checkpoints.begin(),p.checkpoints.end(),progress.completed_updates)!=p.checkpoints.end()){
      log.flush();tr::save_training_state((run/"checkpoints"/("training-"+std::to_string(progress.completed_updates)+".pt")).string(),model,scaler,moment_scaler,optimizer,identity,progress);
    }
  }
  torch::cuda::synchronize();const auto seconds=std::chrono::duration<double>(Clock::now()-started).count();log.close();check(bool(log),"persist training log");
  check(progress.completed_updates==2048&&gpu_gradient&&weight_changed&&moment_gradient_updates>0&&moment_weight_updates>0,"complete declared budget with actual CUDA updates on all active tensors");frozen.validate(model,optimizer);model->validate_state();model->validate_fixed_prior();moment_scaler.validate(p.model,&moment_source);
  r::save_energy_change_checkpoint((run/"checkpoints/inference-2048.pt").string(),model,scaler,moment_scaler,progress.completed_updates);
  const auto final_train=capture(dataset,p,pt,p.tb,p.te,model,scaler,run/"features","train","trained_v25"),final_val=capture(dataset,p,pv,p.vb,p.ve,model,scaler,run/"features","validation","trained_v25");
  for(const std::string role:{"train","validation"}) {
    const auto previous=fs::path("output/runs/market-capacity-study/run-base-gryiAV/features")/("trained_v22-"+role+".pt");
    const auto old=ev::load_features(previous.string()),now=ev::load_features((run/"features"/("trained_v25-"+role+".pt")).string());
    check(torch::equal(old.values,now.values.narrow(1,0,p.model.export_width)) && torch::equal(old.valid,now.valid) &&
        torch::equal(old.row_ids,now.row_ids),"final exact saved trained v22 prefix parity");
  }
  frozen.validate(model,optimizer);
  write(run/"final-prefix-parity.json","{\"saved_v22_train_and_validation_prefix32_equal\":true,\"old_encoder_replay_calls\":0,\"old_head_refits\":0}\n");
  verify(shared_pins);p.final_check();check(sha256(run/"moment-rms-scaler.pt")==moment_sha,"immutable TRAIN moment normalizer archive");
  const auto memory=c10::cuda::CUDACachingAllocator::getDeviceStats(0);struct rusage usage{};::getrusage(RUSAGE_SELF,&usage);
  std::ostringstream summary;summary<<std::setprecision(17)<<"{\"kind\":\"market_change_energy_v1\",\"dataset\":"<<json(p.dataset)<<",\"instance\":"<<json(p.instance)
    <<",\"H\":"<<p.model.history_length<<",\"W\":"<<p.model.encoder_width<<",\"D\":"<<r::energy_change_native_width(p.full)<<",\"level_width\":"<<p.model.export_width<<",\"change_width\":"<<p.full.change_width<<",\"parameters\":"<<parameters<<",\"requires_grad_flagged_parameters\":"<<trainable<<",\"optimizer_updated_parameters\":"<<active_parameters<<",\"protected_parameters\":"<<parameters-active_parameters
    <<",\"completed_updates\":"<<progress.completed_updates<<",\"attempted_batches\":"<<progress.attempted_batches<<",\"segment_updates\":"<<progress.completed_updates
    <<",\"window_presentations\":"<<progress.attempted_batches*8<<",\"target_cells\":"<<total_target_cells<<",\"wave_eligible_exposures\":"<<eligible_exposures<<",\"aux_eligible_exposures\":"<<aux_eligible_exposures<<",\"signed_target_cells\":"<<total_signed_cells<<",\"energy_target_cells\":"<<total_energy_cells<<",\"first_loss\":"<<first_loss<<",\"last_loss\":"<<last_loss<<",\"last_wave_loss\":"<<last_wave_loss<<",\"last_aux_loss\":"<<last_aux_loss<<",\"last_signed_loss\":"<<last_signed_loss<<",\"last_energy_loss\":"<<last_energy_loss
    <<",\"training_seconds_including_checkpoints\":"<<seconds<<",\"moment_normalizer_fit_seconds_including_io\":"<<moment_fit_seconds<<",\"moment_normalizer_TRAIN_anchors\":4096,\"moment_normalizer_validation_anchors\":0,\"moment_normalizer_encoder_calls\":0,\"moment_normalizer_sha256\":"<<json(moment_sha)<<",\"moment_source_identity\":"<<json(moment_source)
    <<",\"nonzero_decoder_gradient_updates\":"<<decoder_gradient_updates<<",\"nonzero_change_gradient_updates\":"<<gradient_updates<<",\"nonzero_moment_gradient_updates\":"<<moment_gradient_updates<<",\"projection_weight_updates\":"<<projection_weight_updates<<",\"decoder_weight_updates\":"<<decoder_weight_updates<<",\"moment_weight_updates\":"<<moment_weight_updates<<",\"cuda_learning_verified\":true,\"CPU_encoder_calls\":0,\"initial_prefix_matches_saved_trained_v22\":true,\"initial_all36_match_saved_v24_init\":true"
    <<",\"final_prefix_matches_saved_trained_v22\":true,\"protected_level_exact\":true,\"protected_optimizer_states\":0,\"parent_updates\":2048,\"parent_inference_sha256\":\"e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c\",\"probe_train_rows\":112,\"probe_validation_rows\":32,\"host_peak_rss_kib\":"<<usage.ru_maxrss<<",\"cuda_peak_allocated_bytes\":"<<memory.allocated_bytes[0].peak<<",\"cuda_peak_reserved_bytes\":"<<memory.reserved_bytes[0].peak
    <<",\"trained_query\":{\"train\":"<<capture_json(final_train)<<",\"validation\":"<<capture_json(final_val)<<"},\"scaler_identity\":"<<json(scaler.identity())
    <<",\"shared_ledger_sha256\":"<<json(shared_sha)<<",\"card_sha256\":"<<json(p.card_sha)<<",\"config_sha256\":"<<json(p.config_sha)<<",\"settings_sha256\":"<<json(p.settings_sha)
    <<",\"input_manifest_sha256\":"<<json(p.manifest_sha)<<",\"source_closure_sha256\":"<<json(sha256(run/"source-closure.sha256"))<<",\"binary_sha256\":"<<json(p.binary_sha)<<"}\n";
  write(run/"summary.json",summary.str());std::cout<<"RUN_DIRECTORY="<<run.string()<<std::endl;return 0;
}
} // namespace
int main(int argc,char **argv)try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
