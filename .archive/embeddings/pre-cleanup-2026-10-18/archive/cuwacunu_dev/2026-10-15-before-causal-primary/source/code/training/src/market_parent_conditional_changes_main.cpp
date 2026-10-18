// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Separately declared RPB-v27 parent-conditional changes; preserve every earlier attempt.
#include "cuwacunu/config/document.h"
#include "cuwacunu/encoder/content_artifact.h"
#include "cuwacunu/encoder/parent_conditional_encoder.h"
#include "cuwacunu/training/market_data.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_training_state.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <array>
#include <chrono>
#include <cmath>
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
namespace tr=r::parent_conditional_change_training;
namespace md=cuwacunu::training::market_data;
using Clock=std::chrono::steady_clock;
using Pin=std::pair<fs::path,std::string>;
void check(bool v,const std::string &why){TORCH_CHECK(v,"[market parent-conditional-change study] ",why);}
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
      "code/encoders/rpb_v19/include","code/encoders/rpb_spectral/include","code/encoders/rpb_content/include","code/encoders/rpb_changes/include","code/encoders/rpb_separate/include","code/encoders/rpb_conditioned/include","code/encoders/rpb_conditional/include","code/training/include"})
    for(const auto &entry:fs::recursive_directory_iterator(root))if(entry.is_regular_file())files.push_back(entry.path());
  for(const auto &p:{"code/training/src/market_separate_changes_main.cpp","code/training/market_separate_changes.mk",
      "code/encoders/rpb_changes/consecutive_change.mk","code/encoders/rpb_separate/tests/separated_change_test.cpp",
      "code/scripts/run-market-separate-changes.sh",
      "code/evaluation/include/cuwacunu/evaluation/fixed_regression.h","code/evaluation/src/fixed_regression.cpp",
      "code/training/src/market_parent_conditional_changes_main.cpp","code/training/market_parent_conditional_changes.mk",
      "code/encoders/rpb_conditional/tests/parent_conditional_change_test.cpp","code/encoders/rpb_conditional/parent_conditional_change_test.mk",
      "code/scripts/run-market-parent-conditional-changes.sh","config/market_studies/parent_conditional_changes.conf",
      "config/market_studies/encoder_parent_conditional_changes.conf","doc/MARKET_PARENT_CONDITIONAL_CHANGE_CARD.md",
      "Makefile","code/scripts/task.sh"})files.emplace_back(p);
  std::sort(files.begin(),files.end());std::string text;
  for(const auto &p:files)text+=sha256(p)+"  "+p.generic_string()+"\n";
  return text;
}
std::string identity_json(const tr::TrainingIdentity &i,const std::string &scaler){
  return "{\"input_source\":"+json(i.input_source_identity)+",\"split\":"+json(i.split_identity)+",\"recipe\":"+json(i.recipe_identity)+",\"scaler\":"+json(scaler)+"}\n";
}

struct Study {
  cuwacunu::config::Document document;enc::Config model;r::ParentConditionalChangeConfig full;fs::path config,settings,card,manifest,output,shared;
  std::string mode="parent_conditional_changes",profile="base",dataset,instance,config_sha,settings_sha,card_sha,manifest_sha,binary_sha,source_closure,source_identity;
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
    check(b.string("MODE")=="parent_conditional_changes" && b.string("PROFILE")=="base" && dataset=="MKT-HIST-2" && instance=="RPB-v27.market-01","fixed candidate mode/profile/tag");
    settings=document.resolve_path(b.string("ENCODER_SETTINGS"));full=r::read_parent_conditional_change_settings(settings);model=full.level;
    card=document.resolve_path(b.string("CARD"));manifest=document.resolve_path(b.string("INPUT_MANIFEST"));output=document.resolve_path(b.string("OUTPUT_ROOT"));
    check(output.lexically_normal()==fs::absolute("output/runs/market-parent-conditional-changes").lexically_normal(),"distinct conditioned-change output root");
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
    check(full.change_width==4 && r::parent_conditional_change_native_width(full)==36,"declared separate native36 geometry");check(probe_train_n==112 && probe_val_n==32,"shared probe counts");shared=document.resolve_path(b.string("SHARED_ROOT"));
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
    check(card_sha=="6e24e48bfdd05148f99446cf1421a713a7736d522f6c00159ba156e43b39c29b","frozen RPB-v27.market-01 card required");
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
std::string vector_json(const torch::Tensor &value,bool integer);
std::string matrix_json(const torch::Tensor &value);
std::string capture_json(const Capture &c){
  std::ostringstream s;s<<std::setprecision(17)<<"{\"mae_train_scaled\":"<<c.mae<<",\"eligible_rows\":"<<c.valid<<",\"total_rows\":"<<c.total<<",\"target_cells\":"<<c.cells
    <<",\"target_history\":128,\"feature_seconds_including_io\":"<<c.feature_seconds<<",\"query_seconds_including_io\":"<<c.query_seconds<<",\"CUDA_encoder_calls\":"<<c.encoder_calls<<",\"CPU_encoder_calls\":0,\"row_mae\":[";
  for(size_t i=0;i<c.row_mae.size();++i){if(i)s<<',';s<<c.row_mae[i];}s<<"],\"row_target_cells\":[";
  for(size_t i=0;i<c.row_cells.size();++i){if(i)s<<',';s<<c.row_cells[i];}s<<"],\"row_eligible_channels\":[";
  for(size_t i=0;i<c.row_channels.size();++i){if(i)s<<',';s<<c.row_channels[i];}s<<"]}";return s.str();
}
Capture capture(md::Dataset &dataset,const Study &p,const std::vector<int64_t> &rows,int64_t begin,int64_t end,
    r::ParentConditionalChangeModel &model,const r::FrozenScaler &scaler,const fs::path &directory,const std::string &role,const std::string &method){
  torch::NoGradGuard guard;model->eval();std::vector<torch::Tensor> values,flags,bank_values,bank_valid,conditioned_values,corrected_values;Capture result;
  const auto shared=query_plan(p,role,rows);
  for(size_t offset=0;offset<rows.size();offset+=p.batch){
    torch::cuda::synchronize();auto started=Clock::now();
    const auto raw=md::load_batch(dataset,md::slice(rows,offset,p.batch),begin,end,p.model);const auto normalized=enc::transform(scaler,raw,p.model);
    check(normalized.data.is_cuda(),"actual CUDA feature input");const auto encoded=model->encode(normalized);const auto native=r::parent_conditional_change_native(encoded,p.full);
    check(native.is_cuda()&&native.sizes()==torch::IntArrayRef({raw.data.size(0),r::parent_conditional_change_native_width(p.full)}),"actual configured CUDA native export");
    const auto targets=r::conditioned_change_targets(normalized,p.model);
    const auto conditioned=model->normalizer().transform(targets);
    const auto corrected=model->conditional_map().residual(native.narrow(1,0,p.model.export_width),encoded.sample_valid_mask,targets,model->normalizer());
    const auto projected=torch::matmul(corrected,model->encoder->change_projection.transpose(0,1));
    const auto expected_suffix=torch::where(encoded.sample_valid_mask.unsqueeze(1),projected,torch::zeros_like(projected));
    check(torch::equal(native.narrow(1,p.model.export_width,p.full.change_width),expected_suffix),"exact parent-conditional corrected bank/projection native suffix witness");
    bank_values.push_back(targets.values.to(torch::kCPU));bank_valid.push_back(targets.valid.to(torch::kCPU));conditioned_values.push_back(conditioned.to(torch::kCPU));corrected_values.push_back(corrected.to(torch::kCPU));
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
    const auto prediction=model->decode(r::parent_conditional_change_native(model->encode(visible),p.full),normalized.channel_ids);++result.encoder_calls;
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
  write(directory/(method+"-"+role+"-conditioning-witness.json"),"{\"kind\":\"market_parent_conditional_changes_projection_witness_v1\",\"method\":"+json(method)+",\"role\":"+json(role)+",\"rows\":"+std::to_string(rows.size())+",\"width\":270,\"native_width\":36,\"bank_dtype\":\"float32\",\"conditioned_dtype\":\"float32\",\"native_dtype\":\"float32\",\"normalizer_identity\":"+json(model->normalizer().identity())+",\"input_scaler_identity\":"+json(scaler.identity())+",\"row_ids\":"+vector_json(torch::tensor(rows,torch::kInt64),true)+",\"sample_valid\":"+vector_json(torch::cat(flags),true)+",\"values\":"+matrix_json(torch::cat(bank_values))+",\"valid\":"+matrix_json(torch::cat(bank_valid))+",\"conditioned\":"+matrix_json(torch::cat(conditioned_values))+",\"conditional_map_identity\":"+json(model->conditional_map().identity())+",\"parent\":"+matrix_json(torch::cat(values).narrow(1,0,p.model.export_width))+",\"corrected\":"+matrix_json(torch::cat(corrected_values))+",\"native\":"+matrix_json(torch::cat(values))+",\"projection\":"+matrix_json(model->encoder->change_projection)+",\"exact_projected_suffix\":true}\n");
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

std::string number(double value){std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();}
std::string vector_json(const torch::Tensor &value,bool integer=false){
  const auto cpu=value.detach().to(torch::kCPU).flatten();std::ostringstream out;out<<std::setprecision(17)<<'[';
  for(int64_t j=0;j<cpu.numel();++j){if(j)out<<',';if(integer)out<<cpu[j].item<int64_t>();else out<<cpu[j].item<double>();}out<<']';return out.str();
}
std::string matrix_json(const torch::Tensor &value){
  const auto cpu=value.detach().to(torch::kCPU,torch::kFloat64).contiguous();check(cpu.dim()==2,"matrix witness rank");
  const auto a=cpu.accessor<double,2>();std::ostringstream out;out<<std::setprecision(17)<<'[';
  for(int64_t i=0;i<cpu.size(0);++i){if(i)out<<',';out<<'[';for(int64_t j=0;j<cpu.size(1);++j){if(j)out<<',';check(std::isfinite(a[i][j]),"finite scalar witness");out<<a[i][j];}out<<']';}out<<']';return out.str();
}
std::string tensor3_json(const torch::Tensor &value){
  check(value.dim()==3,"rank-three sufficient statistic required");std::ostringstream out;out<<'[';
  for(int64_t j=0;j<value.size(0);++j){if(j)out<<',';out<<matrix_json(value[j]);}out<<']';return out.str();
}
struct ParentCalibration { torch::Tensor values,valid;double seconds{0};int64_t forwards{0}; };
ParentCalibration calibrate_parent(md::Dataset &dataset,const Study &p,const std::vector<int64_t> &train,
    const r::FrozenScaler &scaler,r::SeparatedChangeModel &initial,const fs::path &run){
  torch::NoGradGuard guard;initial->eval();ParentCalibration result;std::vector<torch::Tensor> values,valid;
  torch::cuda::synchronize();const auto started=Clock::now();
  for(size_t offset=0;offset<train.size();offset+=p.batch){
    const auto input=enc::transform(scaler,md::load_batch(dataset,md::slice(train,offset,p.batch),p.tb,p.te,p.model),p.model);
    check(input.data.is_cuda(),"actual CUDA parent calibration input");
    const auto encoded=initial->level->encode(input);++result.forwards;
    const auto native=enc::native_embedding(encoded,p.model);
    check(native.is_cuda() && native.scalar_type()==torch::kFloat32 && native.sizes()==torch::IntArrayRef({p.batch,32}),"actual frozen CUDA parent32 calibration");
    values.push_back(native.to(torch::kCPU));valid.push_back(encoded.sample_valid_mask.to(torch::kCPU));
  }
  check(result.forwards==512,"exactly512 new frozen parent calibration forwards");
  result.values=torch::cat(values);result.valid=torch::cat(valid);
  {torch::serialize::OutputArchive a;a.write("kind",r::configurable_checkpoint_detail::text_tensor("market_parent_conditional_parent_calibration_v1"),true);
    a.write("values",result.values,true);a.write("sample_valid",result.valid,true);a.write("row_ids",torch::tensor(train,torch::kInt64),true);
    r::configurable_checkpoint_detail::publish_checkpoint(a,(run/"calibration-parent.pt").string());}
  write(run/"calibration-parent-witness.json","{\"kind\":\"market_parent_conditional_parent_calibration_witness_v1\",\"rows\":4096,\"width\":32,\"batch_size\":8,\"CUDA_encoder_calls\":512,\"CPU_encoder_calls\":0,\"encoder_updates\":0,\"parent_identity\":\"e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c\",\"input_scaler_identity\":"+json(scaler.identity())+",\"row_ids\":"+vector_json(torch::tensor(train,torch::kInt64),true)+",\"sample_valid\":"+vector_json(result.valid,true)+",\"values\":"+matrix_json(result.values)+"}\n");
  torch::cuda::synchronize();result.seconds=std::chrono::duration<double>(Clock::now()-started).count();return result;
}
int execute(int argc,char **argv) {
  check(argc==2,"usage: market_parent_conditional_changes CONFIG");check(fs::exists("/.dockerenv")&&torch::cuda::is_available(),"managed container and CUDA required");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);Study p(argv[1]);
  const auto train=md::uniform_anchors(p.tb,p.te,256,p.train_n,false),pt=md::uniform_anchors(p.tb,p.te,256,p.probe_train_n,true),pv=md::uniform_anchors(p.vb,p.ve,256,p.probe_val_n,true);
  std::vector<Pin> shared_pins;const auto scaler=load_shared(p,train,pt,pv,shared_pins);md::Dataset dataset;open_dataset(dataset,p,p.model);
  fs::create_directories(p.output);auto pattern=(p.output/"run-XXXXXX").string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
  check(::mkdtemp(name.data()),"exclusive candidate leaf");const fs::path run=name.data();fs::create_directory(run/"features");fs::create_directory(run/"checkpoints");
  copy(p.config,run/"study.conf",p.config_sha);copy(p.settings,run/"encoder.conf",p.settings_sha);copy(p.card,run/"prospective-card.md",p.card_sha);copy(p.manifest,run/"input-manifest.json",p.manifest_sha);
  write(run/"source-closure.sha256",p.source_closure);std::string inputs;for(const auto &[path,hash]:p.input_pins)inputs+=hash+"  "+path.string()+"\n";write(run/"loaded-inputs.sha256",inputs);
  const auto shared_sha=sha256(p.shared/"shared-assets.sha256");
  const std::string bank_source="MKT-HIST-2;full-natural-TRAIN4096;recent128;ordered-B8;no-query-no-context;shared-ledger:"+shared_sha+
      ";input:"+p.source_identity+";input-scaler:"+scaler.identity()+";valid-existing-F32-adjacent-global-quarter-bank-v1;population-mean-std-f64-floor1e-6-countge2";
  const fs::path reused="output/runs/market-conditioned-changes/run-RjTTZZ";
  const std::string normalizer_sha="15e11373b14fe438fe920ccf13a72b86aa4719ea1c17dee8422249db1c185535";
  const std::string calibration_bank_sha="e375dbbfc29cbb6a0d70def2de42d83a68bc5d376a8cef6daf88edbfd7d73c08";
  check(sha256(reused/"bank-standardizer.pt")==normalizer_sha && sha256(reused/"calibration-bank.pt")==calibration_bank_sha,"exact reused v26 calibration archives");
  shared_pins.emplace_back(reused/"bank-standardizer.pt",normalizer_sha);shared_pins.emplace_back(reused/"calibration-bank.pt",calibration_bank_sha);
  torch::serialize::InputArchive normalizer_archive;normalizer_archive.load_from((reused/"bank-standardizer.pt").string(),torch::kCPU);
  const auto input_scaler_identity=scaler.identity();const auto normalizer=r::BankStandardizer::load(normalizer_archive,p.model,&bank_source,&input_scaler_identity);
  check(normalizer.identity()=="rpb-bank-standardizer-v1-fnv1a64-59b99cdeeeadc038","exact reused v26 normalizer content");
  copy(reused/"bank-standardizer.pt",run/"bank-standardizer.pt",normalizer_sha);
  const auto old_normalizer_receipt_sha=sha256(reused/"bank-standardizer.json");shared_pins.emplace_back(reused/"bank-standardizer.json",old_normalizer_receipt_sha);
  copy(reused/"bank-standardizer.json",run/"bank-standardizer.json",old_normalizer_receipt_sha);
  torch::serialize::InputArchive bank_archive;bank_archive.load_from((reused/"calibration-bank.pt").string(),torch::kCPU);
  check(r::configurable_checkpoint_detail::read_text(bank_archive,"kind")=="market_conditioned_changes_calibration_bank_v1","typed reused natural TRAIN bank");
  r::ConditionedChangeTargets bank;torch::Tensor bank_rows;bank_archive.read("values",bank.values,true);bank_archive.read("valid",bank.valid,true);bank_archive.read("row_ids",bank_rows,true);
  r::BankStandardizer::validate_targets(bank,270);check(bank.values.device().is_cpu() && bank.values.sizes()==torch::IntArrayRef({4096,270}) && bank_rows.scalar_type()==torch::kInt64 && torch::equal(bank_rows,torch::tensor(train,torch::kInt64)),"exact reused TRAIN bank row order/geometry");
  write(run/"calibration-reuse.json","{\"kind\":\"market_parent_conditional_calibration_reuse_v1\",\"new_bank_normalizer_fits\":0,\"new_bank_encoder_calls\":0,\"normalizer_source_archive\":"+json((reused/"bank-standardizer.pt").generic_string())+",\"normalizer_source_sha256\":"+json(normalizer_sha)+",\"bank_source_archive\":"+json((reused/"calibration-bank.pt").generic_string())+",\"bank_source_sha256\":"+json(calibration_bank_sha)+",\"normalizer_identity\":"+json(normalizer.identity())+"}\n");
  auto calibration_initial=r::load_separated_change_checkpoint("output/runs/market-separate-changes-native36/run-4SauBf/checkpoints/inference-0.pt",p.model.device,&p.full);
  check(calibration_initial.training_updates==0 && calibration_initial.scaler.identity()==scaler.identity(),"same saved v24 initial checkpoint for frozen parent calibration");
  const auto parent_calibration=calibrate_parent(dataset,p,train,scaler,calibration_initial.model,run);
  const std::string map_source="MKT-HIST-2;natural-TRAIN4096;recent128;ordered-B8;parent-calibration-once512CUDA;zero-updates;shared-ledger:"+shared_sha+
      ";bank-archive:"+calibration_bank_sha+";bank-normalizer-archive:"+normalizer_sha+";parent-init:2b614ca9866048a850cdeb73dc719c5816fd04bb39e1746b626d1abf1ef68108;parent-v22:e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c;parent-stats-F64-population-floor1e-6;map-F64-unit-slope-ridge-unpenalized-intercept-countge2";
  const r::ParentConditionalBindings bindings{map_source,scaler.identity(),normalizer.identity(),"e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c"};
  const r::ConditionedChangeTargets cuda_bank{bank.values.to(p.model.device),bank.valid.to(p.model.device)};
  const auto conditioned_calibration=normalizer.transform(cuda_bank).to(torch::kCPU);
  const auto map_started=Clock::now();
  const auto conditional_map=r::fit_parent_conditional_map(p.model,parent_calibration.values,parent_calibration.valid,cuda_bank,normalizer,bindings);
  conditional_map.validate(p.model,&bindings);
  {torch::serialize::OutputArchive a;conditional_map.save(a);r::configurable_checkpoint_detail::publish_checkpoint(a,(run/"conditional-map.pt").string());}
  const auto map_sha=sha256(run/"conditional-map.pt");
  const std::string map_binding_json="\"source_identity\":"+json(map_source)+",\"input_scaler_identity\":"+json(scaler.identity())+",\"normalizer_identity\":"+json(normalizer.identity())+",\"parent_identity\":"+json(bindings.parent_identity)+",\"recipe_identity\":"+json(conditional_map.recipe_identity)+",\"bank_layout\":"+json(conditional_map.bank_layout)+",\"input_geometry\":"+vector_json(torch::tensor(conditional_map.input_geometry,torch::kInt64),true)+",\"sampling_interval\":"+number(conditional_map.sampling_interval)+",\"floor\":"+number(conditional_map.floor)+",\"ridge\":"+number(conditional_map.ridge)+",\"minimum_support\":"+std::to_string(conditional_map.minimum_support)+",\"frozen\":true";
  write(run/"conditional-map.json","{\"kind\":\"market_parent_conditional_map_sufficient_statistics_v1\",\"identity\":"+json(conditional_map.identity())+","+map_binding_json+
      ",\"TRAIN_anchors\":4096,\"validation_anchors\":0,\"parent_width\":32,\"bank_width\":270,\"coefficient_count\":8910,\"penalty\":1,\"intercept_penalized\":false,\"minimum_count\":2,\"parent_scale_floor\":1e-6,\"gram_is_unpenalized\":true,\"residual_definition\":\"(gram+diag(ones32,0))@coefficients-rhs\",\"parent_counts\":"+vector_json(conditional_map.parent_counts,true)+",\"parent_sum\":"+vector_json(conditional_map.parent_sum)+",\"parent_sumsq\":"+vector_json(conditional_map.parent_sumsq)+",\"parent_mean\":"+vector_json(conditional_map.parent_mean)+",\"parent_scale\":"+vector_json(conditional_map.parent_scale)+",\"counts\":"+vector_json(conditional_map.counts,true)+",\"support\":"+vector_json(conditional_map.support,true)+",\"gram\":"+tensor3_json(conditional_map.gram)+",\"rhs\":"+matrix_json(conditional_map.rhs)+",\"target_sumsq\":"+vector_json(conditional_map.target_sumsq)+",\"coefficients\":"+matrix_json(conditional_map.coefficients)+",\"normal_equation_residual\":"+matrix_json(conditional_map.normal_equation_residual)+"}\n");
  write(run/"calibration-map-witness.json","{\"kind\":\"market_parent_conditional_calibration_map_witness_v1\",\"rows\":4096,\"parent_width\":32,\"bank_width\":270,\"batch_size\":8,\"conditional_map_identity\":"+json(conditional_map.identity())+","+map_binding_json+",\"bank_source_sha256\":"+json(calibration_bank_sha)+",\"row_ids\":"+vector_json(bank_rows,true)+",\"sample_valid\":"+vector_json(parent_calibration.valid,true)+",\"parent\":"+matrix_json(parent_calibration.values)+",\"values\":"+matrix_json(bank.values)+",\"valid\":"+matrix_json(bank.valid)+",\"conditioned\":"+matrix_json(conditioned_calibration)+"}\n");
  const double map_seconds=std::chrono::duration<double>(Clock::now()-map_started).count();
  write(run/"conditional-map-summary.json","{\"kind\":\"market_parent_conditional_map_summary_v1\",\"identity\":"+json(conditional_map.identity())+","+map_binding_json+",\"TRAIN_anchors\":4096,\"validation_anchors\":0,\"parent_width\":32,\"bank_width\":270,\"coefficient_count\":8910,\"penalty\":1,\"intercept_penalized\":false,\"minimum_count\":2,\"parent_scale_floor\":1e-6,\"supported_bank_coordinates\":"+std::to_string(conditional_map.support.sum().item<int64_t>())+",\"supported_parent_rows\":"+std::to_string(parent_calibration.valid.sum().item<int64_t>())+",\"archive_sha256\":"+json(map_sha)+",\"fit_seconds_including_io\":"+number(map_seconds)+"}\n");
  const double calibration_seconds=0;
  tr::TrainingIdentity identity{p.source_identity,"shared-anchorH256-targetH128;4096:112:32;shared-ledger:"+shared_sha,
      "RPB-v27.market-01;native36:protected32+separate4;matched-v24-init:2b614ca9866048a850cdeb73dc719c5816fd04bb39e1746b626d1abf1ef68108;zero-wave-residual-decoder;study:"+p.config_sha+";settings:"+p.settings_sha+";card:"+p.card_sha+";binary:"+p.binary_sha+";source:"+sha256(run/"source-closure.sha256")+
      ";parent-v22:e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c;parent-updates2048;branch-updates2048;structural-NoGrad-level-encode-decode;undefined-level-gradients-before-clip-step-v1;row-mask-context-counter-v1;AdamW.001/.0001;B8;Huber1;mask25;context15;wave-only-no-aux;uniform270-bank-conditioning;frozen-parent-conditional-map:"+map_sha+";map-source:"+map_source+";normalizer:"+normalizer_sha+";normalizer-source:"+bank_source+";no-label-loss"};
  torch::manual_seed(p.init_seed);r::ParentConditionalChangeModel model(p.full,normalizer,conditional_map);
  check(model->normalizer().identity()==normalizer.identity() && model->conditional_map().identity()==conditional_map.identity(),"exact frozen normalizer and conditional map copied into model");
  {
    const auto &initial=calibration_initial;
    check(initial.training_updates==0 && initial.scaler.identity()==scaler.identity(),"exact saved native36 initialization updates/scaler");
    tr::warm_start_from_separated(model,initial.model);
    const auto actual=model->encoder->named_parameters();
    for(const auto &v:initial.model->named_parameters())check(torch::equal(v.value(),actual[v.key()]),"exact copied initial encoder parameters");
    const auto actual_buffers=model->encoder->named_buffers();
    for(const auto &v:initial.model->named_buffers())check(torch::equal(v.value(),actual_buffers[v.key()]),"exact copied initial encoder buffers");
  }
  check(model->encoder->change_decoder.eq(0).all().item<bool>(),"literal-zero copied initial waveform residual decoder");
  write(run/"initial-parameter-parity.json","{\"saved_v24_initial_encoder_parameters_and_buffers_exact\":true,\"zero_wave_decoder\":true,\"full_initial36_equality_not_expected\":true}\n");
  torch::optim::AdamW optimizer(model->parameters(),tr::content_training_adamw_options());
  tr::FrozenLevelGuard frozen(model);frozen.validate(model,optimizer);
  tr::TrainingProgress progress{0,0,p.row_seed,p.mask_seed,p.context_seed};
  std::cout<<"STARTED_RUN_DIRECTORY="<<run.string()<<std::endl;
  write(run/"identities.json",identity_json(identity,scaler.identity()));r::save_parent_conditional_change_checkpoint((run/"checkpoints/inference-0.pt").string(),model,scaler,0);
  const auto initial_train=capture(dataset,p,pt,p.tb,p.te,model,scaler,run/"features","train","initialized_v27"),initial_val=capture(dataset,p,pv,p.vb,p.ve,model,scaler,run/"features","validation","initialized_v27");
  write(run/"initial-query.json","{\"train\":"+capture_json(initial_train)+",\"validation\":"+capture_json(initial_val)+"}\n");
  for(const std::string role:{"train","validation"}) {
    const auto previous=fs::path("output/runs/market-capacity-study/run-base-gryiAV/features")/("trained_v22-"+role+".pt");
    const auto old=ev::load_features(previous.string()),now=ev::load_features((run/"features"/("initialized_v27-"+role+".pt")).string());
    check(torch::equal(old.values,now.values.narrow(1,0,p.model.export_width))&&torch::equal(old.valid,now.valid)&&torch::equal(old.row_ids,now.row_ids),"exact saved trained v22 prefix parity");
  }
  write(run/"initialization-parity.json","{\"saved_v22_train_and_validation_prefix32_equal\":true,\"saved_v24_initial_encoder_parameters_and_buffers_exact\":true,\"full_initial36_equality_not_expected\":true,\"old_encoder_replay_calls\":0,\"old_head_refits\":0}\n");
  const auto start_updates=progress.completed_updates;
  int64_t parameters=0,trainable=0;for(const auto &v:model->parameters()){parameters+=v.numel();if(v.requires_grad())trainable+=v.numel();}
  check(parameters==297425 && model->encoder->change_projection.numel()+model->encoder->change_decoder.numel()==14904,"declared unchanged parameter counts");
  std::ofstream log(run/"training.csv");check(bool(log),"training log");log<<"attempt,update,loss,eligible_examples,target_cells,projection_gradient_nonzero,wave_decoder_gradient_nonzero,projection_weight_changed,wave_decoder_weight_changed,row_seed,mask_seed,anchors\n"<<std::setprecision(17);
  bool gpu_gradient=false,weight_changed=false;int64_t gradient_updates=0,decoder_gradient_updates=0,projection_weight_updates=0,decoder_weight_updates=0,total_target_cells=0,eligible_exposures=0;double first_loss=0,last_loss=0;
  model->train();torch::cuda::synchronize();const auto started=Clock::now();
  while(progress.completed_updates<p.updates&&progress.attempted_batches<p.max_attempts){
    const auto attempt=progress.attempted_batches,row_stream=tr::training_counter_seed(p.row_seed,attempt),mask_stream=tr::training_counter_seed(p.mask_seed,attempt);
    std::mt19937_64 rng(row_stream);std::uniform_int_distribution<int64_t> pick(0,train.size()-1);std::vector<int64_t> rows;for(int64_t j=0;j<p.batch;++j)rows.push_back(train[pick(rng)]);
    const auto normalized=enc::transform(scaler,md::load_batch(dataset,rows,p.tb,p.te,p.model),p.model);
    const auto masks=enc::make_training_mask(normalized.observed,p.model,mask_stream);
    if(!masks.eligible_channels.any().item<bool>()){++progress.attempted_batches;log<<attempt<<','<<progress.completed_updates<<",,0,0,0,0,0,0,"<<row_stream<<','<<mask_stream<<",\"";for(size_t j=0;j<rows.size();++j){if(j)log<<';';log<<rows[j];}log<<"\"\n";continue;}
    const auto context=r::context_deletion::make_plan(masks,normalized.channel_ids,p.model,p.context_seed,attempt,r::ContextDeletionRecipe::coordinate15_v1);
    const enc::Input visible{torch::where(context.visible,normalized.data,torch::zeros_like(normalized.data)),context.visible,normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    optimizer.zero_grad();const auto native=r::parent_conditional_change_native(model->encode(visible),p.full);const auto prediction=model->decode(native,normalized.channel_ids);
    const auto objective=r::hierarchical_huber(prediction,normalized.data.detach(),masks.target,masks.eligible_channels,p.model.huber_delta);
    check(normalized.data.is_cuda()&&native.is_cuda()&&prediction.is_cuda()&&objective.loss.is_cuda()&&torch::isfinite(objective.loss).item<bool>(),"finite actual CUDA model/loss");objective.loss.backward();
    for(const auto &v:model->parameters()){check(v.is_cuda(),"all parameters CUDA");if(v.grad().defined())check(v.grad().is_cuda()&&torch::isfinite(v.grad()).all().item<bool>(),"finite actual CUDA gradients");}
    const auto g=model->encoder->change_projection.grad();const auto dg=model->encoder->change_decoder.grad();const bool decoder_nonzero=dg.defined() && dg.is_cuda() && dg.abs().sum().item<double>()>0;if(decoder_nonzero)++decoder_gradient_updates;const bool nonzero=g.defined()&&g.is_cuda()&&g.abs().sum().item<double>()>0;gpu_gradient|=nonzero;if(nonzero)++gradient_updates;
    frozen.discard_gradients(model,optimizer);const auto before=model->encoder->change_projection.detach().clone(),decoder_before=model->encoder->change_decoder.detach().clone();torch::nn::utils::clip_grad_norm_(model->parameters(),1.);optimizer.step();
    const bool moved=!torch::equal(before,model->encoder->change_projection.detach()),decoder_moved=!torch::equal(decoder_before,model->encoder->change_decoder.detach());if(moved)++projection_weight_updates;if(decoder_moved)++decoder_weight_updates;weight_changed|=moved && nonzero;
    ++progress.completed_updates;++progress.attempted_batches;last_loss=objective.loss.item<double>();
    if(progress.completed_updates==start_updates+1){check(!nonzero && decoder_nonzero && decoder_moved,"first step must open zero residual decoder with actual CUDA learning");first_loss=last_loss;}
    if(progress.completed_updates==start_updates+2) {
      check(nonzero && moved && decoder_nonzero && decoder_moved,"second step must learn projection and decoder on CUDA before continuing");
    }
    total_target_cells+=objective.target_cell_count;eligible_exposures+=objective.eligible_example_count;
    log<<attempt<<','<<progress.completed_updates<<','<<last_loss<<','<<objective.eligible_example_count<<','<<objective.target_cell_count<<','<<nonzero<<','<<decoder_nonzero<<','<<moved<<','<<decoder_moved<<','<<row_stream<<','<<mask_stream<<",\"";
    for(size_t j=0;j<rows.size();++j){if(j)log<<';';log<<rows[j];}log<<"\"\n";
    if(progress.completed_updates%128==0) {
      frozen.validate(model,optimizer);
      std::cout<<"update="<<progress.completed_updates<<" attempts="<<progress.attempted_batches<<" loss="<<last_loss<<std::endl;
    }
    if(std::find(p.checkpoints.begin(),p.checkpoints.end(),progress.completed_updates)!=p.checkpoints.end()){
      log.flush();tr::save_training_state((run/"checkpoints"/("training-"+std::to_string(progress.completed_updates)+".pt")).string(),model,scaler,optimizer,identity,progress);
    }
  }
  torch::cuda::synchronize();const auto seconds=std::chrono::duration<double>(Clock::now()-started).count();log.close();check(bool(log),"persist training log");
  check(progress.completed_updates==2048&&gpu_gradient&&weight_changed,"complete declared budget with actual CUDA updates");frozen.validate(model,optimizer);model->validate_state();model->validate_fixed_prior();
  r::save_parent_conditional_change_checkpoint((run/"checkpoints/inference-2048.pt").string(),model,scaler,progress.completed_updates);
  const auto final_train=capture(dataset,p,pt,p.tb,p.te,model,scaler,run/"features","train","trained_v27"),final_val=capture(dataset,p,pv,p.vb,p.ve,model,scaler,run/"features","validation","trained_v27");
  for(const std::string role:{"train","validation"}) {
    const auto previous=fs::path("output/runs/market-capacity-study/run-base-gryiAV/features")/("trained_v22-"+role+".pt");
    const auto old=ev::load_features(previous.string()),now=ev::load_features((run/"features"/("trained_v27-"+role+".pt")).string());
    check(torch::equal(old.values,now.values.narrow(1,0,p.model.export_width)) && torch::equal(old.valid,now.valid) &&
        torch::equal(old.row_ids,now.row_ids),"final exact saved trained v22 prefix parity");
  }
  frozen.validate(model,optimizer);
  write(run/"final-prefix-parity.json","{\"saved_v22_train_and_validation_prefix32_equal\":true,\"old_encoder_replay_calls\":0,\"old_head_refits\":0}\n");
  verify(shared_pins);p.final_check();const auto scaler_identity=scaler.identity();model->normalizer().validate(p.model,&bank_source,&scaler_identity);check(model->normalizer().identity()==normalizer.identity(),"frozen calibration statistics unchanged in model");check(sha256(run/"bank-standardizer.pt")==normalizer_sha,"frozen normalizer archive unchanged");
  model->conditional_map().validate(p.model,&bindings);
  check(model->conditional_map().identity()==conditional_map.identity() && sha256(run/"conditional-map.pt")==map_sha,"frozen conditional map statistics and archive unchanged");
  int64_t map_frozen_storage_bytes=0;for(const auto &t:{conditional_map.parent_counts,conditional_map.parent_sum,conditional_map.parent_sumsq,conditional_map.parent_mean,conditional_map.parent_scale,conditional_map.counts,conditional_map.support,conditional_map.gram,conditional_map.rhs,conditional_map.target_sumsq,conditional_map.coefficients,conditional_map.normal_equation_residual})map_frozen_storage_bytes+=t.nbytes();
  const auto memory=c10::cuda::CUDACachingAllocator::getDeviceStats(0);struct rusage usage{};::getrusage(RUSAGE_SELF,&usage);
  std::ostringstream summary;summary<<std::setprecision(17)<<"{\"kind\":\"market_parent_conditional_changes_v1\",\"protocol\":\"market-parent-conditional-changes-v1\",\"bank_normalizer_identity\":"<<json(normalizer.identity())<<",\"dataset\":"<<json(p.dataset)<<",\"instance\":"<<json(p.instance)
    <<",\"H\":"<<p.model.history_length<<",\"W\":"<<p.model.encoder_width<<",\"D\":"<<r::parent_conditional_change_native_width(p.full)<<",\"level_width\":"<<p.model.export_width<<",\"change_width\":"<<p.full.change_width<<",\"parameters\":"<<parameters<<",\"requires_grad_flagged_parameters\":"<<trainable<<",\"optimizer_updated_parameters\":"<<(model->encoder->change_projection.numel()+model->encoder->change_decoder.numel())<<",\"protected_parameters\":"<<parameters-(model->encoder->change_projection.numel()+model->encoder->change_decoder.numel())
    <<",\"completed_updates\":"<<progress.completed_updates<<",\"attempted_batches\":"<<progress.attempted_batches<<",\"segment_updates\":"<<progress.completed_updates
    <<",\"window_presentations\":"<<progress.attempted_batches*8<<",\"target_cells\":"<<total_target_cells<<",\"first_loss\":"<<first_loss<<",\"last_loss\":"<<last_loss
    <<",\"conditional_map_identity\":"<<json(conditional_map.identity())<<",\"conditional_map_source_identity\":"<<json(map_source)<<",\"conditional_map_sha256\":"<<json(map_sha)<<",\"conditional_map_coefficient_count\":8910,\"conditional_map_coefficient_storage_bytes\":71280,\"conditional_map_frozen_tensor_storage_bytes\":"<<map_frozen_storage_bytes<<",\"conditional_map_fit_seconds_including_io\":"<<map_seconds
    <<",\"parent_calibration_TRAIN_anchors\":4096,\"parent_calibration_validation_anchors\":0,\"parent_calibration_CUDA_encoder_calls\":512,\"parent_calibration_CPU_encoder_calls\":0,\"parent_calibration_encoder_updates\":0,\"parent_calibration_seconds_including_io\":"<<parent_calibration.seconds<<",\"new_bank_normalizer_fits\":0,\"reused_calibration_bank_sha256\":"<<json(calibration_bank_sha)
    <<",\"bank_normalizer_fit_seconds_including_io\":"<<calibration_seconds<<",\"bank_normalizer_TRAIN_anchors\":4096,\"bank_normalizer_validation_anchors\":0,\"bank_normalizer_encoder_calls\":0,\"bank_normalizer_sha256\":"<<json(normalizer_sha)<<",\"bank_normalizer_source_identity\":"<<json(bank_source)<<",\"bank_normalizer_input_scaler_identity\":"<<json(scaler.identity())<<",\"initial_weights_match_saved_v24_init\":true,\"initial_all36_equality_to_v24_not_expected\":true,\"projection_weight_updates\":"<<projection_weight_updates<<",\"decoder_weight_updates\":"<<decoder_weight_updates<<",\"training_seconds_including_checkpoints\":"<<seconds<<",\"nonzero_decoder_gradient_updates\":"<<decoder_gradient_updates<<",\"nonzero_change_gradient_updates\":"<<gradient_updates<<",\"cuda_learning_verified\":true,\"CPU_encoder_calls\":0,\"initial_prefix_matches_saved_trained_v22\":true"
    <<",\"final_prefix_matches_saved_trained_v22\":true,\"protected_level_exact\":true,\"protected_optimizer_states\":0,\"parent_updates\":2048,\"parent_inference_sha256\":\"e58e26dfc178fc26bc2181f6d76b37565f5cb786c0f5b3dd3ad0c6bae564740c\",\"probe_train_rows\":112,\"probe_validation_rows\":32,\"host_peak_rss_kib\":"<<usage.ru_maxrss<<",\"cuda_peak_allocated_bytes\":"<<memory.allocated_bytes[0].peak<<",\"cuda_peak_reserved_bytes\":"<<memory.reserved_bytes[0].peak
    <<",\"trained_query\":{\"train\":"<<capture_json(final_train)<<",\"validation\":"<<capture_json(final_val)<<"},\"scaler_identity\":"<<json(scaler.identity())
    <<",\"shared_ledger_sha256\":"<<json(shared_sha)<<",\"card_sha256\":"<<json(p.card_sha)<<",\"config_sha256\":"<<json(p.config_sha)<<",\"settings_sha256\":"<<json(p.settings_sha)
    <<",\"input_manifest_sha256\":"<<json(p.manifest_sha)<<",\"source_closure_sha256\":"<<json(sha256(run/"source-closure.sha256"))<<",\"binary_sha256\":"<<json(p.binary_sha)<<"}\n";
  write(run/"summary.json",summary.str());std::cout<<"RUN_DIRECTORY="<<run.string()<<std::endl;return 0;
}
} // namespace
int main(int argc,char **argv)try{return execute(argc,argv);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}


