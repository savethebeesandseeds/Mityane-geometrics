// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
// Application runner: mmap, role boundaries and provenance live outside the model.
#include "cuwacunu/config/document.h"
#include "cuwacunu/data/memory_mapped_dataset.h"
#include "cuwacunu/encoder/primary_encoder.h"
#include "cuwacunu/integration/configurable_encoder_input.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/training_state.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/context_deletion.h"
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
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
namespace fs = std::filesystem;
namespace enc = cuwacunu::encoder::primary;
namespace r = embedding::encoders::raw_patch_bottleneck_mae;
namespace ev = cuwacunu::evaluation;
using Dataset = cuwacunu::data::MemoryMappedEdgeDataset<cuwacunu::data::kline_t>;
void check(bool value, const std::string &message) { TORCH_CHECK(value,"[market pilot] ",message); }

// No shell expansion; filenames remain one argv element.
std::string sha256(const fs::path &path) {
  check(fs::is_regular_file(path),"missing provenance file: "+path.string());
  int pipefd[2]; check(::pipe(pipefd)==0,"checksum pipe");
  posix_spawn_file_actions_t actions;
  check(::posix_spawn_file_actions_init(&actions)==0,"checksum spawn actions");
  ::posix_spawn_file_actions_adddup2(&actions,pipefd[1],STDOUT_FILENO);
  ::posix_spawn_file_actions_addclose(&actions,pipefd[0]);
  ::posix_spawn_file_actions_addclose(&actions,pipefd[1]);
  std::string name=path.string();
  char executable[]="sha256sum", separator[]="--";
  char *argv[]={executable,separator,name.data(),nullptr}; pid_t pid;
  const int rc=::posix_spawnp(&pid,executable,&actions,nullptr,argv,environ);
  ::posix_spawn_file_actions_destroy(&actions); ::close(pipefd[1]);
  if(rc) { ::close(pipefd[0]); check(false,"cannot run sha256sum"); }
  std::string output; char buffer[512]; ssize_t n;
  while((n=::read(pipefd[0],buffer,sizeof buffer))>0) output.append(buffer,n);
  ::close(pipefd[0]); int status=0;
  check(::waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0 && output.size()>=64,
        "checksum process failed");
  const auto hash=output.substr(0,64);
  check(hash.find_first_not_of("0123456789abcdef")==std::string::npos,"malformed SHA256"); return hash;
}
std::string json(const std::string &s) {
  std::ostringstream out; out << '"';
  for(unsigned char c:s) { if(c=='"'||c=='\\') out << '\\' << c;
    else if(c=='\n') out << "\\n"; else if(c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c; }
  out << '"'; return out.str();
}
void write(const fs::path &path,const std::string &text) {
  check(!fs::exists(path),"refuse to overwrite "+path.string());
  std::ofstream out(path); check(bool(out),"open output"); out << text; out.close(); check(bool(out),"write output");
}
std::string read_text(const fs::path &path) {
  std::ifstream file(path,std::ios::binary); check(bool(file),"read file "+path.string());
  return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
struct Pilot {
  cuwacunu::config::Document document;
  enc::Config model;
  fs::path settings,card,manifest,output;
  std::vector<fs::path> csv;
  std::string dataset,instance,card_hash,manifest_hash,config_hash,source_closure;
  int64_t tb,te,vb,ve,unused,train_n,probe_train_n,probe_validation_n,batch,updates,max_attempts,save_every;
  int64_t init_seed,row_seed,mask_seed,context_seed,query_seed;
  r::TrainingIdentity identity;
  explicit Pilot(const fs::path &path):document(cuwacunu::config::Document::read(path)) {
    document.reject_unknown_blocks({"PILOT"}); const auto &b=document.block("PILOT");
    b.reject_unknown_keys({"DATASET_ID","INSTANCE_TAG","ENCODER_SETTINGS","BTC_CSV","ETH_CSV","SOL_CSV",
      "BTC_SHA256","ETH_SHA256","SOL_SHA256","INPUT_MANIFEST","CARD","OUTPUT_ROOT","TRAIN_BEGIN_CLOSE_MS",
      "TRAIN_END_CLOSE_MS","VALIDATION_BEGIN_CLOSE_MS","VALIDATION_END_CLOSE_MS","UNUSED_BEGIN_CLOSE_MS",
      "TRAIN_WINDOWS","PROBE_TRAIN_WINDOWS","PROBE_VALIDATION_WINDOWS","BATCH_SIZE","UPDATES","MAX_ATTEMPTS",
      "CHECKPOINT_EVERY","INITIALIZATION_SEED","ROW_SEED","MASK_SEED","CONTEXT_SEED","QUERY_SEED"});
    dataset=b.string("DATASET_ID"); instance=b.string("INSTANCE_TAG");
    settings=document.resolve_path(b.string("ENCODER_SETTINGS")); model=enc::read_settings(settings);
    card=document.resolve_path(b.string("CARD")); manifest=document.resolve_path(b.string("INPUT_MANIFEST"));
    output=document.resolve_path(b.string("OUTPUT_ROOT"));
    tb=b.integer("TRAIN_BEGIN_CLOSE_MS"); te=b.integer("TRAIN_END_CLOSE_MS");
    vb=b.integer("VALIDATION_BEGIN_CLOSE_MS"); ve=b.integer("VALIDATION_END_CLOSE_MS"); unused=b.integer("UNUSED_BEGIN_CLOSE_MS");
    train_n=b.integer("TRAIN_WINDOWS"); probe_train_n=b.integer("PROBE_TRAIN_WINDOWS"); probe_validation_n=b.integer("PROBE_VALIDATION_WINDOWS");
    batch=b.integer("BATCH_SIZE"); updates=b.integer("UPDATES"); max_attempts=b.integer("MAX_ATTEMPTS"); save_every=b.integer("CHECKPOINT_EVERY");
    init_seed=b.integer("INITIALIZATION_SEED"); row_seed=b.integer("ROW_SEED"); mask_seed=b.integer("MASK_SEED");
    context_seed=b.integer("CONTEXT_SEED"); query_seed=b.integer("QUERY_SEED");
    check(dataset=="MKT-HIST-1" && instance=="RPB-v22.market-01","explicit pilot instance required");
    check(model.channel_count==3 && model.channel_ids==std::vector<int64_t>({0,1,2}) && model.input_width==9 &&
      model.history_length==128 && model.patch_length==8 && model.export_width==32 && model.encoder_width==64 &&
      model.num_layers==3 && model.num_heads==4 && model.feedforward_width==256 && model.decoder_hidden_width==128 &&
      model.mask_ratio==.25 && model.huber_delta==1. && model.scale_floor==1e-6 && model.layer_norm_epsilon==1e-5 &&
      model.sampling_interval==60000 && model.device==torch::Device(torch::kCUDA,0),"fixed prospective model recipe");
    check(tb>=0 && tb<te && te<vb && vb<ve && ve<unused && vb-te==60000 && unused-ve==60000 &&
      (te-tb)%60000==0 && (ve-vb)%60000==0,"ordered complete chronological roles");
    check(tb==1704067259999 && te==1705881599999 && vb==1705881659999 && ve==1706486399999 && unused==1706486459999,
      "fixed prospective January role boundaries");
    check(train_n==4096 && probe_train_n==192 && probe_validation_n==64 && batch==8 && updates==512 &&
      max_attempts==1024 && save_every==128 && init_seed==942001 && row_seed==942002 && mask_seed==942003 &&
      context_seed==942004 && query_seed==942005,"fixed prospective budget and streams");
    std::string source;
    for(const std::string symbol:{"BTC","ETH","SOL"}) {
      auto file=document.resolve_path(b.string(symbol+"_CSV")); auto hash=sha256(file);
      check(hash==b.string(symbol+"_SHA256"),"source changed: "+symbol); csv.push_back(file); source+=symbol+":"+hash+";";
    }
    card_hash=sha256(card); manifest_hash=sha256(manifest); config_hash=sha256(path);
    identity.input_source_identity=source+"manifest:"+manifest_hash;
    identity.split_identity="full-role-H128-step60000;uniform-integer-anchors-v1;nonoverlap-H-stride-probes-v1;"+
      std::to_string(tb)+":"+std::to_string(te)+":"+std::to_string(vb)+":"+std::to_string(ve)+":"+std::to_string(unused)+
      ";4096:192:64";
    identity.recipe_identity="pilot-config:"+config_hash+";encoder-config:"+sha256(settings)+";card:"+card_hash+
      ";runner:"+sha256("code/training/src/market_pilot_main.cpp")+
      ";resume:"+sha256("code/encoders/rpb_content/include/embedding/encoders/raw_patch_bottleneck_mae/training_state.h")+
      ";scaler:"+sha256("code/encoders/rpb_spectral/include/embedding/encoders/raw_patch_bottleneck_mae/streaming_scaler.h")+
      ";row-mt19937_64-uniform-i64-v1;mask-splitmix64-attempt-v1;context-coordinate15-base-plus-attempt-v1;adamw-.001-.0001-clip1";
    std::vector<fs::path> closure_files;
    for(const auto &root:{"code/data/include","code/data/src","code/config/include","code/integration/include",
        "code/encoders/rpb_v19/include","code/encoders/rpb_spectral/include","code/encoders/rpb_content/include"})
      for(const auto &entry:fs::recursive_directory_iterator(root)) if(entry.is_regular_file()) closure_files.push_back(entry.path());
    std::sort(closure_files.begin(),closure_files.end());
    for(const auto &file:closure_files) source_closure+=sha256(file)+"  "+file.generic_string()+"\n";
    identity.recipe_identity+=";executable:"+sha256("/proc/"+std::to_string(::getpid())+"/exe");
  }
};

// Floor-quantized uniform selection; no target-dependent choice.
std::vector<int64_t> anchors(int64_t begin,int64_t end,int64_t history,int64_t n,bool nonoverlap) {
  const int64_t stride=60000*(nonoverlap?history:1);
  const int64_t first=begin+(history-1)*60000, available=(end-first)/stride+1;
  check(first<=end && n>=2 && n<=available,"insufficient complete role windows");
  std::vector<int64_t> result; result.reserve(n);
  for(int64_t j=0;j<n;++j) result.push_back(first+(j*(available-1)/(n-1))*stride);
  for(size_t j=1;j<result.size();++j) check(result[j]-result[j-1]>=stride,"distinct/nonoverlapping anchors");
  return result;
}
enc::Input load_batch(Dataset &dataset,const std::vector<int64_t> &rows,int64_t begin,int64_t end,const enc::Config &c) {
  std::vector<cuwacunu::data::sample_t> samples; samples.reserve(rows.size());
  for(const auto anchor:rows) {
    check(anchor<=end && anchor-(c.history_length-1)*60000>=begin,"window crosses role boundary");
    samples.push_back(dataset.get_by_anchor_key(anchor));
  }
  const auto batch=cuwacunu::data::sample_t::collate_fn(samples);
  check(batch.past_keys.ge(begin).all().item<bool>() && batch.past_keys.le(end).all().item<bool>() &&
    torch::equal(batch.past_keys.select(1,0).select(1,c.history_length-1).to(torch::kInt64),torch::tensor(rows,torch::kInt64)),"mmap exact anchor association");
  return cuwacunu::integration::configurable_encoder_input(batch,torch::tensor(c.channel_ids,torch::kInt64),c);
}
std::vector<int64_t> slice(const std::vector<int64_t> &v,size_t begin,size_t count) {
  return {v.begin()+begin,v.begin()+std::min(v.size(),begin+count)};
}
struct Capture { double query_mae{0}; int64_t query_valid{0},query_total{0},query_cells{0}; };
Capture capture(Dataset &dataset,const Pilot &p,const std::vector<int64_t> &rows,int64_t begin,int64_t end,
    enc::Encoder &model,const enc::FrozenScaler &scaler,const fs::path &directory,const std::string &role,
    const std::string &method,bool save_raw_targets) {
  torch::NoGradGuard guard; model->eval();
  std::vector<torch::Tensor> features,valid,raw_features,target_values,target_valid;
  Capture receipt;
  for(size_t offset=0;offset<rows.size();offset+=p.batch) {
    auto raw=load_batch(dataset,slice(rows,offset,p.batch),begin,end,p.model);
    const auto normalized=enc::transform(scaler,raw,p.model); check(normalized.data.is_cuda(),"actual CUDA input");
    const auto encoded=model->encode(normalized); const auto native=enc::native_embedding(encoded,p.model);
    check(native.is_cuda() && native.sizes()==torch::IntArrayRef({raw.data.size(0),32}),"actual CUDA native32");
    features.push_back(native.to(torch::kCPU)); valid.push_back(encoded.sample_valid_mask.to(torch::kCPU));
    if(save_raw_targets) {
      const auto mu=scaler.mean.view({1,3,1,9}), sigma=scaler.scale.view({1,3,1,9});
      const auto scaled=torch::where(raw.observed,(raw.data.to(torch::kFloat64)-mu)/sigma,torch::zeros_like(raw.data));
      raw_features.push_back(torch::cat({scaled.flatten(1),raw.observed.to(torch::kFloat64).flatten(1)},1));
      const auto targets=ev::past_window_targets(raw.data,raw.observed);
      target_values.push_back(targets.values); target_valid.push_back(targets.valid);
    }
    auto query_config=p.model; query_config.mask_ratio=1./16.;
    const auto masks=enc::make_training_mask(normalized.observed,query_config,
      r::training_counter_seed(p.query_seed,static_cast<int64_t>(offset/p.batch)));
    const enc::Input visible{torch::where(masks.visible,normalized.data,torch::zeros_like(normalized.data)),masks.visible,
      normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    const auto prediction=model->decode(enc::native_embedding(model->encode(visible),p.model),normalized.channel_ids);
    check(prediction.is_cuda() && torch::isfinite(prediction).all().item<bool>(),"finite CUDA query decoder");
    const auto counts=masks.target.sum(std::vector<int64_t>{2,3});
    const auto channel_mae=torch::where(masks.target,(prediction-normalized.data).abs(),torch::zeros_like(prediction))
      .sum(std::vector<int64_t>{2,3})/counts.clamp_min(1).to(torch::kFloat32);
    const auto channels=masks.eligible_channels.sum(1); const auto eligible=channels.gt(0);
    const auto row_mae=(channel_mae*masks.eligible_channels.to(torch::kFloat32)).sum(1)/channels.clamp_min(1).to(torch::kFloat32);
    receipt.query_mae+=row_mae.masked_select(eligible).sum().item<double>();
    receipt.query_valid+=eligible.sum().item<int64_t>(); receipt.query_total+=raw.data.size(0);
    receipt.query_cells+=masks.target.sum().item<int64_t>();
  }
  const auto ids=torch::tensor(rows,torch::kInt64);
  ev::save_features((directory/(method+"-"+role+".pt")).string(),{torch::cat(features),torch::cat(valid),ids});
  if(save_raw_targets) {
    auto flags=torch::ones({static_cast<int64_t>(rows.size())},torch::kBool);
    ev::save_features((directory/("raw_mask-"+role+".pt")).string(),{torch::cat(raw_features),flags,ids});
    ev::save_targets((directory/(role+"-targets.pt")).string(),{{torch::cat(target_values),torch::cat(target_valid)},ids,ids.clone()});
  }
  check(receipt.query_valid>0,"query has eligible rows"); receipt.query_mae/=receipt.query_valid; return receipt;
}
std::string capture_json(const Capture &c) {
  std::ostringstream s; s << std::setprecision(17) << "{\"mae_train_scaled\":" << c.query_mae << ",\"eligible_rows\":" << c.query_valid
    << ",\"total_rows\":" << c.query_total << ",\"target_cells\":" << c.query_cells << "}"; return s.str();
}
const std::vector<std::string> initialized_assets{
  "features/initialized_v22-train.pt","features/initialized_v22-validation.pt",
  "features/raw_mask-train.pt","features/raw_mask-validation.pt",
  "features/train-targets.pt","features/validation-targets.pt",
  "checkpoints/inference-0.pt","initial-query.json","selections.pt"};
void save_selections(const fs::path &path,const Pilot &p,const std::vector<int64_t> &train,
    const std::vector<int64_t> &probe_train,const std::vector<int64_t> &probe_validation) {
  torch::serialize::OutputArchive a;
  a.write("train_anchors",torch::tensor(train,torch::kInt64),true);
  a.write("probe_train_anchors",torch::tensor(probe_train,torch::kInt64),true);
  a.write("probe_validation_anchors",torch::tensor(probe_validation,torch::kInt64),true);
  a.write("roles_and_strides",torch::tensor(std::vector<int64_t>{p.tb,p.te,p.vb,p.ve,p.unused,60000,128,128*60000},torch::kInt64),true);
  a.save_to(path.string());
}
void validate_selections(const fs::path &path,const Pilot &p,const std::vector<int64_t> &train,
    const std::vector<int64_t> &probe_train,const std::vector<int64_t> &probe_validation) {
  torch::serialize::InputArchive a; a.load_from(path.string(),torch::kCPU);
  const auto matches=[&](const std::string &key,const std::vector<int64_t> &expected) {
    torch::Tensor t; a.read(key,t,true); check(t.scalar_type()==torch::kInt64 && torch::equal(t,torch::tensor(expected,torch::kInt64)),"resume exact "+key);
  };
  matches("train_anchors",train); matches("probe_train_anchors",probe_train); matches("probe_validation_anchors",probe_validation);
  matches("roles_and_strides",{p.tb,p.te,p.vb,p.ve,p.unused,60000,128,128*60000});
}
void publish_initialized_ledger(const fs::path &run) {
  std::string text; for(const auto &name:initialized_assets) text+=sha256(run/name)+"  "+name+"\n";
  write(run/"initialized-assets.sha256",text);
}
void verify_and_copy_initialized(const fs::path &previous,const fs::path &run) {
  std::istringstream ledger(read_text(previous/"initialized-assets.sha256"));
  for(const auto &name:initialized_assets) {
    std::string line; check(bool(std::getline(ledger,line)),"complete initialized asset ledger");
    check(line.size()==66+name.size() && line.substr(64)=="  "+name && sha256(previous/name)==line.substr(0,64),"initialized asset checksum: "+name);
    fs::copy_file(previous/name,run/name); check(sha256(run/name)==line.substr(0,64),"copied asset checksum");
  }
  std::string extra; check(!std::getline(ledger,extra),"no unrecognized initialized assets");
  fs::copy_file(previous/"initialized-assets.sha256",run/"initialized-assets.sha256");
}

int execute(int argc,char **argv) {
  check(argc==2 || (argc==4 && std::string(argv[2])=="--resume"),"usage: market_history_pilot config/market_history_pilot.conf [--resume training-N.pt]");
  check(fs::exists("/.dockerenv") && torch::cuda::is_available(),"managed Linux container and actual CUDA required");
  torch::set_num_threads(1); torch::set_num_interop_threads(1);
  Pilot p(argv[1]);
  const auto train=anchors(p.tb,p.te,p.model.history_length,p.train_n,false);
  const auto probe_train=anchors(p.tb,p.te,p.model.history_length,p.probe_train_n,true);
  const auto probe_validation=anchors(p.vb,p.ve,p.model.history_length,p.probe_validation_n,true);
  Dataset dataset;
  for(const auto &path:p.csv) dataset.add_dataset(path.string(),p.model.history_length,0,"none",false);
  std::vector<std::pair<fs::path,std::string>> loaded_inputs;
  for(const auto &path:p.csv) {
    const fs::path binary=cuwacunu::data::raw_binary_for_csv(path.string());
    const auto hash=sha256(binary); p.identity.input_source_identity+=";mmap:"+path.filename().string()+":"+hash;
    loaded_inputs.emplace_back(path,sha256(path)); loaded_inputs.emplace_back(binary,hash);
  }
  r::StreamingScaler stream(enc::input_config(p.model));
  for(size_t offset=0;offset<train.size();offset+=p.batch)
    stream.add_train_batch(load_batch(dataset,slice(train,offset,p.batch),p.tb,p.te,p.model));
  auto scaler=stream.finish();
  fs::create_directories(p.output); auto pattern=(p.output/"run-XXXXXX").string();
  std::vector<char> name(pattern.begin(),pattern.end()); name.push_back('\0');
  check(::mkdtemp(name.data())!=nullptr,"exclusive output directory"); const fs::path run(name.data());
  std::cout << "STARTED_RUN_DIRECTORY=" << run.string() << std::endl;
  const auto checkpoints=run/"checkpoints",banks=run/"features"; fs::create_directory(checkpoints); fs::create_directory(banks);
  fs::copy_file(p.card,run/"prospective-card.md"); fs::copy_file(argv[1],run/"pilot.conf"); fs::copy_file(p.settings,run/"encoder.conf");
  fs::copy_file(p.manifest,run/"input-manifest.json");
  write(run/"source-closure.sha256",p.source_closure);
  std::string input_receipt;
  for(const auto &[path,hash]:loaded_inputs) input_receipt+=hash+"  "+path.string()+"\n";
  write(run/"loaded-inputs.sha256",input_receipt);
  p.identity.recipe_identity+=";source-closure:"+sha256(run/"source-closure.sha256");
  enc::Encoder model{nullptr}; std::unique_ptr<torch::optim::AdamW> optimizer;
  r::TrainingProgress progress{0,0,p.row_seed,p.mask_seed,p.context_seed};
  Capture initial_train,initial_validation;
  if(argc==4) {
    const auto previous=fs::absolute(argv[3]).parent_path().parent_path();
    verify_and_copy_initialized(previous,run);
    validate_selections(run/"selections.pt",p,train,probe_train,probe_validation);
    p.identity.split_identity+=";selection-sha256:"+sha256(run/"selections.pt");
    p.identity.recipe_identity+=";initialized-assets-sha256:"+sha256(run/"initialized-assets.sha256");
    auto restored=r::load_content_training_state(argv[3],p.model,p.identity,scaler.identity());
    check(restored.progress.row_seed==p.row_seed && restored.progress.mask_seed==p.mask_seed && restored.progress.context_seed==p.context_seed,
      "configured resume stream bases must match");
    check(restored.progress.completed_updates<=p.updates && restored.progress.attempted_batches<=p.max_attempts,"resume state within fixed budget");
    model=restored.model; scaler=restored.scaler; optimizer=std::move(restored.optimizer); progress=restored.progress;
    if(progress.completed_updates==p.updates) fs::copy_file(argv[3],checkpoints/"training-512.pt");
    write(run/"resume-source.txt",fs::absolute(argv[3]).string()+"\n"+sha256(argv[3])+"\n");
  } else {
    torch::manual_seed(p.init_seed); model=enc::Encoder(p.model);
    optimizer=std::make_unique<torch::optim::AdamW>(model->parameters(),r::content_training_adamw_options());
    enc::save_checkpoint((checkpoints/"inference-0.pt").string(),model,scaler,0);
    initial_train=capture(dataset,p,probe_train,p.tb,p.te,model,scaler,banks,"train","initialized_v22",true);
    initial_validation=capture(dataset,p,probe_validation,p.vb,p.ve,model,scaler,banks,"validation","initialized_v22",true);
    write(run/"initial-query.json","{\"train\":"+capture_json(initial_train)+",\"validation\":"+capture_json(initial_validation)+"}\n");
    save_selections(run/"selections.pt",p,train,probe_train,probe_validation);
    publish_initialized_ledger(run);
    p.identity.split_identity+=";selection-sha256:"+sha256(run/"selections.pt");
    p.identity.recipe_identity+=";initialized-assets-sha256:"+sha256(run/"initialized-assets.sha256");
    r::save_content_training_state((checkpoints/"training-0.pt").string(),model,scaler,*optimizer,p.identity,progress);
  }
  const auto identity_text="{\"input_source\":"+json(p.identity.input_source_identity)+",\"split\":"+json(p.identity.split_identity)+
    ",\"recipe\":"+json(p.identity.recipe_identity)+",\"scaler\":"+json(scaler.identity())+"}\n";
  if(argc==4) check(read_text(fs::absolute(argv[3]).parent_path().parent_path()/"identities.json")==identity_text,"resume source run identities");
  write(run/"identities.json",identity_text);
  const int64_t start_updates=progress.completed_updates, start_attempts=progress.attempted_batches;
  std::ofstream log(run/"training.csv"); log << "attempt,update,loss,eligible_examples,target_cells\n" << std::setprecision(17);
  model->train(); torch::cuda::synchronize();
  const auto started=std::chrono::steady_clock::now(); double first_loss=0,last_loss=0; bool gpu_gradient=false,weight_changed=false;
  while(progress.completed_updates<p.updates && progress.attempted_batches<p.max_attempts) {
    const auto attempt=progress.attempted_batches;
    std::mt19937_64 rng(r::training_counter_seed(p.row_seed,attempt)); std::uniform_int_distribution<int64_t> pick(0,train.size()-1);
    std::vector<int64_t> rows; for(int64_t j=0;j<p.batch;++j) rows.push_back(train[pick(rng)]);
    const auto normalized=enc::transform(scaler,load_batch(dataset,rows,p.tb,p.te,p.model),p.model);
    const auto masks=enc::make_training_mask(normalized.observed,p.model,r::training_counter_seed(p.mask_seed,attempt));
    if(!masks.eligible_channels.any().item<bool>()) { ++progress.attempted_batches; log << attempt << ',' << progress.completed_updates << ",,0,0\n"; continue; }
    const auto context=r::context_deletion::make_plan(masks,normalized.channel_ids,p.model,p.context_seed,attempt,r::ContextDeletionRecipe::coordinate15_v1);
    const enc::Input visible{torch::where(context.visible,normalized.data,torch::zeros_like(normalized.data)),context.visible,
      normalized.channel_ids,normalized.endpoints,normalized.sampling_interval};
    optimizer->zero_grad(); const auto native=enc::native_embedding(model->encode(visible),p.model);
    const auto prediction=model->decode(native,normalized.channel_ids);
    const auto objective=r::hierarchical_huber(prediction,normalized.data.detach(),masks.target,masks.eligible_channels,p.model.huber_delta);
    check(normalized.data.is_cuda() && native.is_cuda() && prediction.is_cuda() && objective.loss.is_cuda() &&
      torch::isfinite(objective.loss).item<bool>(),"finite actual CUDA model/loss");
    objective.loss.backward();
    for(const auto &parameter:model->parameters()) {
      check(parameter.is_cuda(),"all encoder parameters CUDA");
      if(parameter.grad().defined()) check(parameter.grad().is_cuda() && torch::isfinite(parameter.grad()).all().item<bool>(),"finite CUDA gradients");
    }
    const auto gradient=model->relation_projection->weight.grad();
    gpu_gradient|=gradient.defined() && gradient.is_cuda() && gradient.abs().sum().item<double>()>0;
    const auto before=model->relation_projection->weight.detach().clone();
    torch::nn::utils::clip_grad_norm_(model->parameters(),1.); optimizer->step();
    weight_changed|=!torch::equal(before,model->relation_projection->weight.detach());
    ++progress.completed_updates; ++progress.attempted_batches; last_loss=objective.loss.item<double>();
    if(progress.completed_updates==start_updates+1) first_loss=last_loss;
    log << attempt << ',' << progress.completed_updates << ',' << last_loss << ',' << objective.eligible_example_count << ',' << objective.target_cell_count << '\n';
    if(progress.completed_updates%64==0) std::cout << "update=" << progress.completed_updates << " attempts=" << progress.attempted_batches << " loss=" << last_loss << std::endl;
    if(progress.completed_updates%p.save_every==0) {
      log.flush(); r::save_content_training_state((checkpoints/("training-"+std::to_string(progress.completed_updates)+".pt")).string(),
        model,scaler,*optimizer,p.identity,progress);
    }
  }
  torch::cuda::synchronize(); const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  log.close(); check(bool(log),"persist attempt log");
  check(progress.completed_updates==p.updates && (progress.completed_updates==start_updates || (gpu_gradient && weight_changed)),
    "budget reached with actual nonzero CUDA projection updates or verified completed resume state");
  for(const auto &[path,hash]:loaded_inputs) check(sha256(path)==hash,"input changed during run: "+path.string());
  enc::save_checkpoint((checkpoints/"inference-512.pt").string(),model,scaler,progress.completed_updates);
  const auto final_train=capture(dataset,p,probe_train,p.tb,p.te,model,scaler,banks,"train","trained_v22",false);
  const auto final_validation=capture(dataset,p,probe_validation,p.vb,p.ve,model,scaler,banks,"validation","trained_v22",false);
  const auto memory=c10::cuda::CUDACachingAllocator::getDeviceStats(0); struct rusage usage{}; ::getrusage(RUSAGE_SELF,&usage);
  std::ostringstream summary; summary << std::setprecision(17) << "{\"dataset\":" << json(p.dataset) << ",\"instance\":" << json(p.instance)
    << ",\"completed_updates\":" << progress.completed_updates << ",\"attempted_batches\":" << progress.attempted_batches
    << ",\"segment_updates\":" << progress.completed_updates-start_updates << ",\"segment_attempts\":" << progress.attempted_batches-start_attempts
    << ",\"first_segment_loss\":" << (progress.completed_updates>start_updates?std::to_string(first_loss):"null")
    << ",\"last_segment_loss\":" << (progress.completed_updates>start_updates?std::to_string(last_loss):"null")
    << ",\"segment_training_seconds_including_checkpoints\":" << seconds
    << ",\"host_peak_rss_kib\":" << usage.ru_maxrss << ",\"cuda_peak_allocated_bytes\":" << memory.allocated_bytes[0].peak
    << ",\"cuda_peak_reserved_bytes\":" << memory.reserved_bytes[0].peak << ",\"scaler_state_coordinates\":27,\"max_input_batch\":[8,3,128,9],"
    << "\"encoder_device\":\"cuda:0\",\"segment_cuda_gradient_verified\":" << (gpu_gradient?"true":"false")
    << ",\"segment_cuda_parameter_update_verified\":" << (weight_changed?"true":"false")
    << ",\"restored_adamw_state_verified\":" << (argc==4?"true":"false") << ','
    << "\"trained_query\":{\"train\":" << capture_json(final_train) << ",\"validation\":" << capture_json(final_validation) << "},"
    << "\"card_sha256\":" << json(p.card_hash) << ",\"input_manifest_sha256\":" << json(p.manifest_hash) << "}\n";
  write(run/"summary.json",summary.str()); std::cout << "RUN_DIRECTORY=" << run.string() << std::endl; return 0;
}
} // namespace
int main(int argc,char **argv) try { return execute(argc,argv); }
catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
