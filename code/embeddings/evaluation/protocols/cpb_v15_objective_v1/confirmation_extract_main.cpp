// Copyright (c) 2026 Waajacu. See LICENSE.
// Open only after the Node runner verifies the sealed six-run development receipt.
#include "include/model_binding.h"
#include "include/market_data.h"
#include "cuwacunu/evaluation/fixed_regression.h"
#include <torch/cuda.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
namespace {
namespace cp=embedding::encoders::causal_patch_bottleneck_mae::CPB_OBJECTIVE_VERSION;
namespace md=cpb_v15_objective::market;
namespace ev=cuwacunu::evaluation;
namespace fs=std::filesystem;
void require(bool yes,const std::string &why){TORCH_CHECK(yes,"[V15 confirmation extraction] ",why);}
struct TensorState{std::string name;torch::Tensor value;bool requires_grad;torch::Device device;};
struct Guard {
  std::vector<TensorState> parameters,buffers;cp::FrozenScaler scaler;decltype(std::declval<cp::Encoder>()->normalizer().clone()) normalizer;
  explicit Guard(const cp::Encoder &m,const cp::FrozenScaler &s):scaler{s.mean.clone(),s.scale.clone(),s.count.clone(),s.channel_ids.clone(),s.scale_floor,s.floor_applied.clone()},normalizer(m->normalizer().clone()){
    for(const auto &p:m->named_parameters()){require(p.value().is_cuda()&&!p.value().grad().defined(),"actual CUDA parameters without gradients");parameters.push_back({p.key(),p.value().detach().cpu().clone(),p.value().requires_grad(),p.value().device()});}
    for(const auto &p:m->named_buffers())buffers.push_back({p.key(),p.value().detach().cpu().clone(),p.value().requires_grad(),p.value().device()});
    require(!parameters.empty(),"nonempty independent model");
  }
  void check(const cp::Encoder &m,const cp::FrozenScaler &s)const{
    require(!m->is_training(),"eval mode");
    const auto compare=[](const auto &now,const auto &before){require(now.size()==before.size(),"state count");for(const auto &p:before){const auto *v=now.find(p.name);require(v && v->sizes()==p.value.sizes() && v->scalar_type()==p.value.scalar_type() && v->device()==p.device && v->requires_grad()==p.requires_grad && !v->grad().defined() && torch::equal(v->detach().cpu(),p.value),"immutable model state: "+p.name);}};
    compare(m->named_parameters(),parameters);compare(m->named_buffers(),buffers);
    const auto &n=m->normalizer();
    for(const auto &[a,b]:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{n.counts,normalizer.counts},{n.sum,normalizer.sum},{n.sumsq,normalizer.sumsq},{n.mean,normalizer.mean},{n.scale,normalizer.scale},{n.support,normalizer.support}})
      require(a.device()==b.device()&&a.scalar_type()==b.scalar_type()&&torch::equal(a,b)&&!a.requires_grad()&&!a.grad().defined(),"immutable unregistered bank state");
    require(n.identity()==normalizer.identity() && n.source_identity==normalizer.source_identity && n.input_scaler_identity==normalizer.input_scaler_identity && n.input_geometry==normalizer.input_geometry && n.sampling_interval==normalizer.sampling_interval && n.floor==normalizer.floor && n.minimum_support==normalizer.minimum_support && n.frozen==normalizer.frozen && n.bank_family==normalizer.bank_family && n.raw_contract==normalizer.raw_contract && n.increment_recipe==normalizer.increment_recipe && n.layout_identity==normalizer.layout_identity && n.recipe_identity==normalizer.recipe_identity,"immutable normalizer metadata");
    require(s.identity()==scaler.identity()&&s.scale_floor==scaler.scale_floor&&torch::equal(s.mean,scaler.mean)&&torch::equal(s.scale,scaler.scale)&&torch::equal(s.count,scaler.count)&&torch::equal(s.channel_ids,scaler.channel_ids)&&torch::equal(s.floor_applied,scaler.floor_applied),"immutable checkpoint scaler");
  }
};

void save(torch::serialize::OutputArchive &a,const fs::path &file){require(!fs::exists(file),"exclusive archive");a.save_to(file.string());}
int run(int argc,char **argv){
  require(fs::exists("/.dockerenv")&&torch::cuda::is_available(),"managed actual CUDA container");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);torch::NoGradGuard no_grad;
  require(argc==23,"eleven explicit confirmation arguments required");
  std::map<std::string,fs::path> args;
  for(int i=1;i<argc;i+=2)require(args.emplace(argv[i],argv[i+1]).second,"unique flag");
  for(const auto *key:{"--settings","--checkpoint","--checkpoint-sha256","--targets","--query-geometry","--cache-btc","--cache-eth","--cache-sol","--seal","--seal-sha256","--output"})require(args.count(key),"required confirmation argument");
  require(cp::boundary::sha256_file(args.at("--seal"))==args.at("--seal-sha256").string(),"exact whole six-run seal");
  require(cp::boundary::sha256_file(args.at("--checkpoint"))==args.at("--checkpoint-sha256").string(),"sealed final checkpoint bytes");
  const auto config=cp::read_settings(args.at("--settings").string());
  auto loaded=cp::load_checkpoint(args.at("--checkpoint").string(),config.level.device,&config);
  require(loaded.source_updates==0&&loaded.causal_updates==2048,"only sealed fresh final2048 checkpoint");
  loaded.model->eval();loaded.model->clear_gradients();const Guard guard(loaded.model,loaded.scaler);
  const auto targets=ev::load_targets(args.at("--targets").string());
  require(targets.row_ids.numel()==32&&torch::equal(targets.row_ids,targets.group_ids),"exact32 timestamp-qualified confirmation groups");
  constexpr int64_t first=1706745659999LL,last=1709251199999LL,minutes=41760;
  std::vector<int64_t> all_rows;
  for(int64_t i=0;i<32;++i){const auto expected=first+((2*i+1)*minutes/64)*60000;require(targets.row_ids[i].item<int64_t>()==expected,"frozen calendar midpoint row; no selection from values");all_rows.push_back(expected);}
  torch::serialize::InputArchive geometry;geometry.load_from(args.at("--query-geometry").string(),torch::kCPU);
  torch::Tensor hidden;geometry.read("hidden_recent128",hidden,true);
  require(hidden.scalar_type()==torch::kBool&&hidden.sizes()==torch::IntArrayRef({32,3,128,9}),"only saved known-VALIDATION Boolean mask geometry");
  auto patches=hidden.reshape({32,3,16,72});
  require(torch::equal(patches,patches.select(-1,0).unsqueeze(-1).expand_as(patches))&&patches.select(-1,0).sum(-1).eq(1).all().item<bool>(),"one complete8x9 patch per row/channel; fail without reseeding");
  std::vector<fs::path> caches;for(const auto *key:{"--cache-btc","--cache-eth","--cache-sol"})caches.push_back(args.at(key));
  md::Dataset data(caches);for(const auto &channel:data.channels)require(channel.first==first&&channel.last==last&&channel.rows()==minutes,"exact admitted February2024 month");
  const auto output=fs::absolute(args.at("--output"));require(!fs::exists(output)&&fs::is_directory(output.parent_path())&&fs::create_directory(output),"exclusive confirmation leaf");
  std::vector<torch::Tensor> native,valid,mae,nll,count,means,variances,scaled_targets,supports;
  int64_t calls=0;
  for(size_t offset=0;offset<all_rows.size();offset+=8){
    const auto rows=md::slice(all_rows,offset,8);const auto raw=data.load(rows,first,last);
    const auto cutoffs=torch::tensor(rows,torch::kFloat64);
    const auto encoded=loaded.model->encode_at(raw,cutoffs,loaded.scaler);++calls;
    require(encoded.native.is_cuda()&&encoded.native.scalar_type()==torch::kFloat32&&encoded.native.sizes()==torch::IntArrayRef({8,36})&&encoded.valid.all().item<bool>(),"complete frozen CUDA native36");
    native.push_back(encoded.native.to(torch::kCPU,torch::kFloat64));valid.push_back(encoded.valid.cpu());
    const auto query=hidden.narrow(0,offset,8);
    const auto trace=loaded.model->forward_at(raw,cutoffs,query,loaded.scaler);++calls;
    require(trace.nll.eligible_example_count==8&&trace.nll.eligible_channels.all().item<bool>(),"full recomputed February query support");
    auto counts=trace.target_mask.sum(std::vector<int64_t>{2,3});
    const auto error=(trace.distribution.mean.to(torch::kFloat64)-trace.targets.to(torch::kFloat64)).abs();
    const auto per_channel=torch::where(trace.target_mask,error,torch::zeros_like(error)).sum(std::vector<int64_t>{2,3})/counts.clamp_min(1).to(torch::kFloat64);
    const auto per_example=per_channel.mean(1);
    require(torch::isfinite(per_example).all().item<bool>()&&torch::isfinite(trace.nll.example_loss).all().item<bool>(),"finite descriptive MAE/NLL");
    mae.push_back(per_example.cpu());nll.push_back(trace.nll.example_loss.cpu());count.push_back(counts.cpu());
    means.push_back(trace.distribution.mean.cpu());variances.push_back(trace.distribution.variance.cpu());
    scaled_targets.push_back(trace.targets.cpu());supports.push_back(trace.target_mask.cpu());guard.check(loaded.model,loaded.scaler);
  }
  torch::serialize::OutputArchive features;features.write("values",torch::cat(native),true);features.write("valid",torch::cat(valid),true);features.write("row_ids",targets.row_ids,true);save(features,output/"confirmation-features.pt");
  torch::serialize::OutputArchive witness;witness.write("row_ids",targets.row_ids,true);witness.write("mean",torch::cat(means),true);witness.write("variance",torch::cat(variances),true);witness.write("targets",torch::cat(scaled_targets),true);witness.write("target_mask",torch::cat(supports),true);witness.write("target_counts",torch::cat(count),true);witness.write("row_mae",torch::cat(mae),true);witness.write("row_nll",torch::cat(nll),true);save(witness,output/"confirmation-query-witness.pt");
  require(calls==8&&cp::boundary::sha256_file(args.at("--checkpoint"))==args.at("--checkpoint-sha256").string()&&cp::boundary::sha256_file(args.at("--seal"))==args.at("--seal-sha256").string(),"sealed inputs retained and exact confirmation call budget");
  std::ofstream report(output/"report.json");report<<std::setprecision(17)<<"{\"kind\":\"cpb_v15_confirmation_extraction_v1\",\"status\":\"passed\",\"rows\":32,\"native_width\":36,\"full_CUDA_calls\":8,\"encoder_updates\":0,\"normalizer_fits\":0,\"head_fits\":0,\"model_state_unchanged\":true,\"descriptive_only_no_gate_or_selection\":true,\"standardized_MAE\":"<<torch::cat(mae).mean().item<double>()<<",\"Gaussian_NLL_F64\":"<<torch::cat(nll).mean().item<double>()<<"}\n";report.close();require(bool(report),"persist report");
  return 0;
}
}
int main(int argc,char **argv){try{return run(argc,argv);}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}}
