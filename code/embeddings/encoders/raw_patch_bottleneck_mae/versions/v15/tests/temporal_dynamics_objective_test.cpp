// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/model.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/preprocessing.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/temporal_dynamics_objective.h"
#include "rpb_test_support.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>

#ifndef RPB_SOURCE_ID
#define RPB_SOURCE_ID "unrecorded"
#endif

namespace {
namespace rpb = embedding::encoders::raw_patch_bottleneck_mae;
namespace fs = std::filesystem;
using namespace rpb_test;

rpb::Config configuration() {
  rpb::Config c; c.device=torch::kCUDA; c.channel_ids={101,202,303};
  c.channel_mixer_layers=1;c.channel_mixer_placement=1;c.global_bottleneck_mode=2;
  c.mask_ratio=.15;return c;
}

void exact(const torch::Tensor &a,const torch::Tensor &b,const char *why) {
  check(a.scalar_type()==b.scalar_type() && a.sizes()==b.sizes() && torch::equal(a,b),why);
}
void scalar_near(double actual,double expected,double tolerance,const char *why) {
  check(std::isfinite(actual) && std::isfinite(expected) && std::abs(actual-expected)<=tolerance,why);
}

// Independent scalar reference: all four vectors are centred on the SAME
// eligible times; no target is inferred from a phase parameter or a label.
rpb::TemporalDynamicsTargets reference(const rpb::Input &in) {
  const auto x=in.data.detach().to(torch::kCPU,torch::kFloat64).contiguous();
  const auto mask=in.observed.to(torch::kCPU).contiguous();
  const auto B=x.size(0);auto values=torch::zeros({B,18},torch::kFloat64);
  auto counts=torch::zeros({B,18},torch::kInt64);auto v=values.accessor<double,2>();auto n=counts.accessor<int64_t,2>();
  auto a=x.accessor<double,4>();auto m=mask.accessor<bool,4>();
  for(int64_t b=0;b<B;++b) {
    int64_t d=0;
    for(const auto &p:std::array<std::array<int64_t,2>,3>{{{{0,1}},{{0,2}},{{1,2}}}})
      for(int64_t k:{1,2,4}) {
        double even=0,odd=0;int64_t pairs=0;
        for(int64_t f=0;f<3;++f)for(int64_t g=0;g<3;++g) {
          std::array<std::vector<double>,4> q;
          for(int64_t t=0;t<32-k;++t)if(m[b][p[0]][t][f]&&m[b][p[0]][t+k][f]&&m[b][p[1]][t][g]&&m[b][p[1]][t+k][g]) {
            q[0].push_back(a[b][p[0]][t][f]);q[1].push_back(a[b][p[0]][t+k][f]);
            q[2].push_back(a[b][p[1]][t][g]);q[3].push_back(a[b][p[1]][t+k][g]);
          }
          if(q[0].size()<4)continue;
          std::array<double,4> means{},variances{};
          for(size_t j=0;j<4;++j) {
            for(double value:q[j])means[j]+=value;
            means[j]/=q[j].size();
            for(double value:q[j])variances[j]+=(value-means[j])*(value-means[j]);
            variances[j]/=q[j].size();
          }
          if(std::any_of(variances.begin(),variances.end(),[](double z){return z<=1e-12;}))continue;
          double plus=0,minus=0;
          for(size_t i=0;i<q[0].size();++i) {
            plus+=(q[0][i]-means[0])*(q[3][i]-means[3]);
            minus+=(q[1][i]-means[1])*(q[2][i]-means[2]);
          }
          plus/=q[0].size()*std::sqrt(variances[0])*std::sqrt(variances[3]);
          minus/=q[0].size()*std::sqrt(variances[1])*std::sqrt(variances[2]);
          plus=std::clamp(plus,-1.,1.);minus=std::clamp(minus,-1.,1.);
          even+=.5*(plus+minus);odd+=.5*(plus-minus);++pairs;
        }
        v[b][d]=pairs?even/pairs:0;v[b][d+1]=pairs?odd/pairs:0;
        n[b][d]=n[b][d+1]=pairs;d+=2;
      }
  }
  return {values,counts.gt(0),counts};
}

void target_contract() {
  const auto c=configuration();auto original=input(c,4);
  original.data=original.data.to(torch::kCUDA);original.observed=original.observed.to(torch::kCUDA);
  original.data.set_requires_grad(true);
  // Exact four versus three common adjacent times, and a constant-energy row.
  original.observed[1].fill_(false);original.observed[1].narrow(1,0,5).fill_(true);
  original.observed[2].fill_(false);original.observed[2].narrow(1,0,4).fill_(true);
  original.data=original.data.detach();original.data[3].fill_(4.);original.data.set_requires_grad(true);
  const auto targets=rpb::temporal_dynamics_targets(original,c), expected=reference(original);
  check(!targets.values.requires_grad() && targets.values.is_cuda() && targets.values.scalar_type()==torch::kFloat64,
      "original legal CUDA targets detached in F64");
  check(torch::allclose(targets.values.to(torch::kCPU),expected.values,1e-10,1e-11),"independent scalar common-support correlation target law");
  exact(targets.valid.to(torch::kCPU),expected.valid,"independent common-support validity");
  exact(targets.supported_feature_pairs.to(torch::kCPU),expected.supported_feature_pairs,"independent supported feature-pair counts");
  for(int64_t pair=0;pair<3;++pair) {
    check(targets.valid[1][pair*6].item<bool>() && !targets.valid[1][pair*6+2].item<bool>(),"four supported times valid, three invalid");
  }
  check(!targets.valid[2].any().item<bool>() && !targets.valid[3].any().item<bool>(),"below-four and constant-energy rows invalid");
  check(targets.values[2].eq(0).all().item<bool>()&&targets.values[3].eq(0).all().item<bool>(),"invalid values exact zero");
  auto poisoned=original;poisoned.data=torch::where(original.observed,original.data,
      torch::full_like(original.data,std::numeric_limits<double>::quiet_NaN()));
  const auto hidden=rpb::temporal_dynamics_targets(poisoned,c);
  exact(targets.values,hidden.values,"hidden NaN isolation before centering");
  const auto permutation=torch::tensor({2,0,1},torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
  auto reordered=original;reordered.data=original.data.index_select(1,permutation);
  reordered.observed=original.observed.index_select(1,permutation);
  reordered.channel_ids=original.channel_ids.to(torch::kCUDA).index_select(0,permutation);
  const auto canonical=rpb::temporal_dynamics_targets(reordered,c);
  exact(targets.values,canonical.values,"physical channel permutation with semantic IDs");
  exact(targets.valid,canonical.valid,"canonical support under permutation");
  const auto gain=torch::tensor({.4,.9,2.,1.2,1.7,.8,2.3,.6,1.1},torch::kFloat64).reshape({1,3,1,3}).to(torch::kCUDA);
  const auto offset=torch::tensor({2.,-3.,1.,4.,2.,-.5,8.,-4.,3.},torch::kFloat64).reshape({1,3,1,3}).to(torch::kCUDA);
  auto affine=original;affine.data=original.data*gain+offset;
  const auto transformed=rpb::temporal_dynamics_targets(affine,c);
  check(torch::allclose(targets.values,transformed.values,1e-10,1e-11),"positive per-feature affine invariance on common support");
  exact(targets.valid,transformed.valid,"affine support unchanged above energy floor");
  auto absent=poisoned;absent.observed=torch::zeros_like(original.observed);
  const auto empty=rpb::temporal_dynamics_targets(absent,c);
  check(!empty.valid.any().item<bool>() && empty.values.eq(0).all().item<bool>() && empty.supported_feature_pairs.eq(0).all().item<bool>(),
      "all-absent default targets");
  auto bad=original;bad.observed=torch::zeros({4,3,31,3},original.observed.options());
  rejects([&]{rpb::temporal_dynamics_targets(bad,c);},"target mask geometry");
  bad=original;bad.data=bad.data.detach().clone();bad.data[0][0][0][0]=std::numeric_limits<double>::quiet_NaN();
  rejects([&]{rpb::temporal_dynamics_targets(bad,c);},"observed target NaN");
}

void map_and_loss_contract() {
  const auto c=configuration();auto x=input(c,5);x.data=x.data.to(torch::kCUDA);x.observed=x.observed.to(torch::kCUDA);
  const auto targets=rpb::temporal_dynamics_targets(x,c);
  const auto map=rpb::fit_temporal_dynamics_scaler(targets);
  const auto standardized=map.standardize(targets);
  check(map.mean.device().is_cpu()&&map.scale.device().is_cpu()&&map.count.device().is_cpu(),"frozen F64 TRAIN map on CPU");
  check(!standardized.values.requires_grad(),"standardized target remains detached");
  const auto values=targets.values.to(torch::kCPU),valid=targets.valid.to(torch::kCPU);
  for(int64_t d=0;d<18;++d) {
    const auto legal=values.select(1,d).masked_select(valid.select(1,d));
    check(map.count[d].item<int64_t>()==legal.numel(),"per-dimension original TRAIN fit count");
    if(legal.numel()) {
      scalar_near(map.mean[d].item<double>(),legal.mean().item<double>(),1e-12,"original TRAIN map mean");
      scalar_near(map.scale[d].item<double>(),std::max(legal.std(false).item<double>(),1e-6),1e-12,"original TRAIN population scale/floor");
    }
  }
  const auto dir=fs::temp_directory_path()/("rpb-temporal-dynamics-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(dir),"exclusive preserved synthetic objective archive directory");
  const auto path=dir/"dynamics-scaler.serialized-archive";
  torch::serialize::OutputArchive saved;map.save(saved);saved.save_to(path.string());
  torch::serialize::InputArchive loaded;loaded.load_from(path.string(),torch::kCPU);rpb::TemporalDynamicsScaler restored;restored.load(loaded);
  exact(map.mean,restored.mean,"map mean save/load");exact(map.scale,restored.scale,"map scale save/load");exact(map.count,restored.count,"map count save/load");
  auto invalid=map;invalid.scale=invalid.scale.clone();invalid.scale[0]=0;
  rejects([&]{invalid.standardize(targets);},"nonpositive frozen map scale");
  invalid=map;invalid.count=invalid.count.to(torch::kFloat64);
  rejects([&]{invalid.standardize(targets);},"wrong frozen map count dtype");
  auto none=targets;none.valid=torch::zeros_like(targets.valid);none.supported_feature_pairs=torch::zeros_like(targets.supported_feature_pairs);
  none.values=torch::zeros_like(targets.values);
  auto invalid_none=none;invalid_none.values=invalid_none.values.clone();invalid_none.values[0][0]=1;
  rejects([&]{rpb::fit_temporal_dynamics_scaler(invalid_none);},"invalid target dimensions must remain zero");
  const auto absent_map=rpb::fit_temporal_dynamics_scaler(none);
  check(absent_map.mean.eq(0).all().item<bool>()&&absent_map.scale.eq(1).all().item<bool>()&&absent_map.count.eq(0).all().item<bool>(),"no-support map defaults");
  rejects([&]{absent_map.standardize(targets);},"supported dimension absent from frozen TRAIN");
  // Exact example/support normalization, including unsupported NaN coordinates.
  auto prediction=torch::zeros({3,18},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto target=torch::zeros_like(prediction);auto support=torch::zeros({3,18},prediction.options().dtype(torch::kBool));
  support[0][0]=true;support[1][0]=true;support[1][1]=true;target[0][0]=2;target[1][0]=1;target[1][1]=3;
  prediction=torch::where(support,prediction,torch::full_like(prediction,std::numeric_limits<float>::quiet_NaN())).detach().set_requires_grad(true);
  target=torch::where(support,target,torch::full_like(target,std::numeric_limits<float>::quiet_NaN())).detach().set_requires_grad(true);
  const auto loss=rpb::temporal_dynamics_huber(prediction,target,support,1.0);
  scalar_near(loss.loss.item<double>(),1.5,1e-7,"Huber1 dimensions then examples equal normalization");
  check(loss.total_valid_dimensions==3&&loss.eligible_example_count==2,"objective support counts");loss.loss.backward();
  check(torch::isfinite(prediction.grad()).all().item<bool>()&&!target.grad().defined(),"legal prediction gradient only, detached target");
  check(prediction.grad().masked_select(support.logical_not()).eq(0).all().item<bool>(),"unsupported objective gradient zero");
  auto empty_prediction=prediction.detach().clone().set_requires_grad(true);
  const auto no_loss=rpb::temporal_dynamics_huber(empty_prediction,target,torch::zeros_like(support));no_loss.loss.backward();
  check(no_loss.loss.item<double>()==0&&empty_prediction.grad().eq(0).all().item<bool>(),"all-unsupported zero loss and gradient");
  rejects([&]{rpb::temporal_dynamics_huber(prediction,target,support,.5);},"fixed Huber delta");
  std::cout<<"Preserved temporal dynamics synthetic scaler archive: "<<path.string()<<'\n';
}

void cuda_graph_and_rng() {
  const auto c=configuration();torch::manual_seed(28171);rpb::Model original(c);
  const auto cpu_next=torch::rand({7}),cuda_next=torch::rand({7},torch::TensorOptions().device(torch::kCUDA));
  torch::manual_seed(28171);rpb::Model model(c);rpb::TemporalDynamicsDecoder head(c.device);
  exact(cpu_next,torch::rand({7}),"objective construction does not advance common CPU RNG");
  exact(cuda_next,torch::rand({7},torch::TensorOptions().device(torch::kCUDA)),"objective construction does not advance common CUDA RNG");
  const auto original_parameters=original->named_parameters(),parameters=model->named_parameters();int64_t backbone=0;
  check(original_parameters.size()==parameters.size(),"backbone registration count unchanged");
  for(size_t i=0;i<parameters.size();++i) {
    check(original_parameters[i].key()==parameters[i].key(),"backbone registration order unchanged");
    exact(original_parameters[i].value(),parameters[i].value(),"common original early initialization");backbone+=parameters[i].value().numel();
  }
  check(backbone==225805&&head->parameters().size()==1&&head->weight.sizes()==torch::IntArrayRef({18,32})&&
      head->weight.numel()==576&&head->weight.is_cuda()&&head->weight.ne(0).all().item<bool>(),"original225805 plus576 training-only bias-free values");
  torch::manual_seed(7);rpb::TemporalDynamicsDecoder second(c.device);exact(head->weight,second->weight,"head initialization deterministic independently of seed");
  auto raw=input(c,3);const auto frozen=rpb::fit_scaler(raw,c);auto x=frozen.transform(raw,c);
  const auto targets=rpb::temporal_dynamics_targets(x,c);
  const auto map=rpb::fit_temporal_dynamics_scaler(targets);
  const auto standardized=map.standardize(targets);const auto fixed_target=standardized.values.to(torch::kFloat32);
  auto all_parameters=model->parameters();const auto hp=head->parameters();all_parameters.insert(all_parameters.end(),hp.begin(),hp.end());
  torch::optim::AdamW optimizer(all_parameters,torch::optim::AdamWOptions(.001).weight_decay(.0001));
  const auto patch_weight=model->named_parameters()["patch_projection.weight"];
  const auto before=head->weight.detach().clone();const auto backbone_before=patch_weight.detach().clone();
  for(int64_t step=0;step<3;++step) {
    const auto mask=rpb::make_training_mask(x.observed,c,9611+step);optimizer.zero_grad();
    const auto forward=model->forward(x,mask.hidden);const auto native=rpb::compact_reconstruction_export(forward.encoding,c);
    check(native.is_cuda()&&native.sizes()==torch::IntArrayRef({3,32}),"objective input exact served native32 CUDA");
    const auto prediction=head->forward(native);const auto dynamics=rpb::temporal_dynamics_huber(prediction,fixed_target,standardized.valid);
    const auto total=forward.loss+dynamics.loss;check(torch::isfinite(total).item<bool>(),"fixed1:1 raw and dynamics loss finite");
    total.backward();check(head->weight.grad().defined()&&torch::isfinite(head->weight.grad()).all().item<bool>()&&
        head->weight.grad().abs().sum().item<double>()>0,"training-only576 decoder receives finite nonzero gradient");
    check(patch_weight.grad().defined()&&torch::isfinite(patch_weight.grad()).all().item<bool>()&&
        patch_weight.grad().abs().sum().item<double>()>0,"sole-native objective graph reaches CUDA backbone");
    torch::nn::utils::clip_grad_norm_(all_parameters,1.,2.,true);optimizer.step();
  }
  check(!torch::equal(before,head->weight)&&!torch::equal(backbone_before,patch_weight),"actual three CUDA updates change head and backbone");
  rejects([&]{head->forward(torch::zeros({3,33},head->weight.options()));},"objective rejects input beyond served native32");
  const auto dir=fs::temp_directory_path()/("rpb-temporal-dynamics-head-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(fs::create_directory(dir),"exclusive objective head evidence directory");
  const auto path=dir/"head.serialized-archive";torch::serialize::OutputArchive archive;head->save(archive);archive.save_to(path.string());
  rpb::TemporalDynamicsDecoder restored(c.device);torch::serialize::InputArchive input_archive;input_archive.load_from(path.string(),torch::kCUDA);restored->load(input_archive);
  exact(head->weight,restored->weight,"training-only head state save/load");
  std::cout<<"Preserved temporal dynamics synthetic head archive: "<<path.string()<<'\n';
}
} // namespace

int main() try {
  check(torch::cuda::is_available(),"temporal objective admission requires actual CUDA");torch::set_num_threads(1);
  target_contract();map_and_loss_contract();cuda_graph_and_rng();
  std::cout<<"Temporal dynamics objective CUDA admission passed\n"<<RPB_SOURCE_ID<<'\n';return 0;
} catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
