// SPDX-License-Identifier: MIT
#include "two_component.h"
#include "embedding/shared/feature_harness.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace embedding::evaluation::two_component {
namespace {
constexpr double pi = 3.14159265358979323846;
void require(bool ok,const char *why) {
  if(!ok)throw std::runtime_error(std::string("[two-component] ")+why);
}
uint64_t task_bit(Task task) {
  switch(task) {case Task::SlowLagSign:return 0;case Task::ComponentBalance:return 1;}
  throw std::runtime_error("[two-component] unknown task");
}
uint64_t task_stream(Task task) {return task_bit(task)==0?kSlowLagTaskStream:kBalanceTaskStream;}
double uniform53(std::mt19937_64 &rng) {
  return static_cast<double>(rng()>>11)*(1.0/9007199254740992.0);
}
uint64_t bounded_integer(std::mt19937_64 &rng,uint64_t bound) {
  const auto threshold=(uint64_t(0)-bound)%bound;
  uint64_t draw;do{draw=rng();}while(draw<threshold);return draw%bound;
}
double gaussian(std::mt19937_64 &rng) {
  const double u1=1.0-uniform53(rng),u2=uniform53(rng);
  return std::sqrt(-2.0*std::log(u1))*std::cos(2.0*pi*u2);
}
std::mt19937_64 source_rng(uint64_t namespaced,uint64_t role,uint64_t source) {
  return std::mt19937_64(stream_seed(stream_seed(namespaced,role),source+1));
}
std::array<double,2> amplitudes(double ratio) {
  const double denominator=std::sqrt(1.0+ratio*ratio);
  return {ratio/denominator,1.0/denominator};
}
Split assemble(const std::vector<torch::Tensor> &values,const std::vector<torch::Tensor> &masks,
    const std::vector<int64_t> &labels,const std::vector<int64_t> &ids) {
  const auto mask=torch::stack(masks);
  const auto observed=torch::where(mask,torch::stack(values),torch::zeros({int64_t(values.size()),3,32,3},torch::kFloat64));
  return {observed,mask,torch::tensor(labels,torch::kInt64),torch::tensor(ids,torch::kInt64)};
}

// Exact prospective draw law (all real uniforms use mt19937_64 high53):
// 1. Task namespace first. Split uses descending Fisher-Yates, i=total-1..1,
//    full64 bounded rejection threshold=(-bound)%bound. First TRAIN assignments.
// 2. For ascending original source indices, independent per-source role RNGs:
//    signal: Ps,Pf,phi_s,phi_f,Ls,Lf,r,slow-sign,fast-sign (nine draws).
//    distractor: Ps2,Pf2,phi_s2,phi_f2,r2 (five draws).
//    affine: channel0..2/feature0..2, gain then offset (18 draws).
//    gaps: channel0..2, start=floor(30*U), covering three original ticks.
//    noise: channel/time/feature lexical order, two Box-Muller draws per cell,
//      sqrt(-2log(1-U1))*cos(2pi*U2); the same additive noise for both variants.
//    masks: same lexical order, independent U>=.10, then shared channel gap.
//    order: one U, row0label=U>=.5 ?1:0; row1 opposite.
// 3. Label only selects a paired waveform variant; every nuisance is drawn
//    before variant construction. Task namespaces separate both data and split.
Development generate(std::size_t train_pairs,std::size_t validation_pairs,uint64_t master,Task task,double noise_sd) {
  const auto bit=task_bit(task);
  constexpr std::size_t source_limit=std::size_t(1)<<30;
  require(master<(uint64_t(1)<<32)&&train_pairs>0&&validation_pairs>0&&
      train_pairs<source_limit&&validation_pairs<source_limit&&
      train_pairs<source_limit-validation_pairs,"bounded injective seed/source domain required before draws/allocation");
  const auto total=train_pairs+validation_pairs;const auto namespaced=stream_seed(master,task_stream(task));
  std::vector<uint64_t> assignment(total);std::iota(assignment.begin(),assignment.end(),uint64_t(0));
  std::mt19937_64 split_rng(stream_seed(namespaced,kSplitStream));
  for(uint64_t i=total-1;i>0;--i)std::swap(assignment[i],assignment[bounded_integer(split_rng,i+1)]);
  std::vector<bool> train(total,false);for(std::size_t i=0;i<train_pairs;++i)train[assignment[i]]=true;
  std::vector<torch::Tensor> values[2],masks[2];std::vector<int64_t> labels[2],ids[2];
  for(uint64_t source=0;source<total;++source) {
    auto signal=source_rng(namespaced,kSignalStream,source),distractor=source_rng(namespaced,kDistractorStream,source);
    auto affine=source_rng(namespaced,kAffineStream,source),noise=source_rng(namespaced,kNoiseStream,source);
    auto natural=source_rng(namespaced,kMaskStream,source),gaps=source_rng(namespaced,kGapStream,source),order=source_rng(namespaced,kOrderStream,source);
    const double ps=12.0+8.0*uniform53(signal),pf=4.5+2.5*uniform53(signal);
    const double phi_s=2*pi*uniform53(signal),phi_f=2*pi*uniform53(signal);
    const double ls=.5+.5*uniform53(signal),lf=.3+.4*uniform53(signal);
    const double ratio=task==Task::SlowLagSign?.8+.45*uniform53(signal):1.8+.4*uniform53(signal);
    const double slow_sign=uniform53(signal)>=.5?1.0:-1.0,fast_sign=uniform53(signal)>=.5?1.0:-1.0;
    const double ps2=12.0+8.0*uniform53(distractor),pf2=4.5+2.5*uniform53(distractor);
    const double phi_s2=2*pi*uniform53(distractor),phi_f2=2*pi*uniform53(distractor);
    const auto a2=amplitudes(.8+.45*uniform53(distractor));
    std::array<std::array<double,3>,3> gain{},offset{};std::array<int64_t,3> gap{};
    for(int64_t c=0;c<3;++c)for(int64_t f=0;f<3;++f) {
      gain[c][f]=.5+uniform53(affine);offset[c][f]=-.75+1.5*uniform53(affine);
    }
    for(int64_t c=0;c<3;++c)gap[c]=static_cast<int64_t>(30*uniform53(gaps));
    const auto a_label0=amplitudes(task==Task::ComponentBalance?1.0/ratio:ratio),a_label1=amplitudes(ratio);
    auto a=torch::zeros({3,32,3},torch::kFloat64),b=torch::zeros_like(a),mask=torch::zeros_like(a,torch::kBool);
    auto av=a.accessor<double,3>(),bv=b.accessor<double,3>();auto observed=mask.accessor<bool,3>();
    for(int64_t c=0;c<3;++c)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f) {
      const double epsilon=noise_sd*gaussian(noise);
      const bool present=uniform53(natural)>=.10&&!(t>=gap[c]&&t<gap[c]+3);
      observed[c][t][f]=present;
      for(int label=0;label<2;++label) {
        double mixture;
        if(c==2)mixture=a2[0]*std::sin(2*pi*t/ps2+phi_s2)+a2[1]*std::sin(2*pi*t/pf2+phi_f2);
        else {
          const auto &amplitude=label?a_label1:a_label0;
          const double slow_shift=c==1?(task==Task::SlowLagSign?(label?ls:-ls):slow_sign*ls):0;
          const double fast_shift=c==1?fast_sign*lf:0;
          mixture=amplitude[0]*std::sin(2*pi*(t+slow_shift)/ps+phi_s)+
              amplitude[1]*std::sin(2*pi*(t+fast_shift)/pf+phi_f);
        }
        (label?bv:av)[c][t][f]=offset[c][f]+gain[c][f]*(1+.1*f)*mixture+epsilon;
      }
    }
    const int split=train[source]?0:1;const bool swap=uniform53(order)>=.5;
    const auto id=static_cast<int64_t>((master<<31)|(source<<1)|bit);
    for(int variant=0;variant<2;++variant) {
      const int label=swap?1-variant:variant;
      values[split].push_back(label?b:a);masks[split].push_back(mask.clone());labels[split].push_back(label);ids[split].push_back(id);
    }
  }
  return {assemble(values[0],masks[0],labels[0],ids[0]),assemble(values[1],masks[1],labels[1],ids[1])};
}

void validate(const Split &source) {
  const auto &x=source.observed,&mask=source.mask,&labels=source.scoring_labels,&ids=source.source_ids;
  require(x.defined()&&x.device().is_cpu()&&x.scalar_type()==torch::kFloat64&&x.is_contiguous()&&
      x.dim()==4&&x.size(0)>0&&x.size(0)%2==0&&x.size(1)==3&&x.size(2)==32&&x.size(3)==3,"CPU F64 paired N3H32F3 required");
  require(mask.defined()&&mask.device().is_cpu()&&mask.scalar_type()==torch::kBool&&mask.is_contiguous()&&mask.sizes()==x.sizes(),"CPU Bool mask required");
  require(labels.defined()&&ids.defined()&&labels.device().is_cpu()&&ids.device().is_cpu()&&
      labels.scalar_type()==torch::kInt64&&ids.scalar_type()==torch::kInt64&&labels.is_contiguous()&&ids.is_contiguous()&&
      labels.sizes()==torch::IntArrayRef({x.size(0)})&&ids.sizes()==labels.sizes(),"CPU Long scoring labels/source IDs required");
  require(torch::isfinite(x).all().item<bool>()&&x.masked_select(mask.logical_not()).eq(0).all().item<bool>(),"finite observed values with zero hidden storage required");
  std::map<int64_t,std::vector<int64_t>> rows;
  for(int64_t row=0;row<x.size(0);++row) {
    require(ids[row].item<int64_t>()>=0&&(labels[row].item<int64_t>()==0||labels[row].item<int64_t>()==1),"nonnegative IDs and binary scoring labels required");
    rows[ids[row].item<int64_t>()].push_back(row);
  }
  for(const auto &[id,pair]:rows) {
    (void)id;require(pair.size()==2&&labels[pair[0]].item<int64_t>()!=labels[pair[1]].item<int64_t>()&&
        torch::equal(mask[pair[0]],mask[pair[1]]),"whole opposite-label pair with shared mask required");
  }
}
} // namespace

std::string task_name(Task task) {
  switch(task){case Task::SlowLagSign:return "slow_lag_sign";case Task::ComponentBalance:return "component_balance";}
  throw std::runtime_error("[two-component] unknown task");
}
std::string dataset_id(Task task) {
  switch(task){case Task::SlowLagSign:return "TEMPO-4";case Task::ComponentBalance:return "AMP-2";}
  throw std::runtime_error("[two-component] unknown task");
}
Development make_development(std::size_t train_pairs,std::size_t validation_pairs,uint64_t master,Task task) {
  return generate(train_pairs,validation_pairs,master,task,.005);
}
Split delete_observations(const Split &source,double probability,uint64_t seed) {
  require(std::isfinite(probability)&&probability>=0&&probability<=1,"finite deletion probability in[0,1] required");validate(source);
  std::map<int64_t,torch::Tensor> erased;std::vector<torch::Tensor> rows;
  for(int64_t row=0;row<source.observed.size(0);++row) {
    const auto id=source.source_ids[row].item<int64_t>();
    if(!erased.count(id)) {
      std::mt19937_64 rng(stream_seed(stream_seed(seed,kDeletionStream),static_cast<uint64_t>(id)));
      auto e=torch::zeros({3,32,3},torch::kBool);auto m=e.accessor<bool,3>();
      for(int64_t c=0;c<3;++c)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f)m[c][t][f]=uniform53(rng)<probability;
      erased.emplace(id,e);
    }
    rows.push_back(erased.at(id));
  }
  const auto mask=source.mask.logical_and(torch::stack(rows).logical_not());
  return {torch::where(mask,source.observed,torch::zeros_like(source.observed)),mask,
      source.scoring_labels.clone(),source.source_ids.clone()};
}
namespace engineering {
Development make_noiseless_development(std::size_t train_pairs,std::size_t validation_pairs,uint64_t master,Task task) {
  return generate(train_pairs,validation_pairs,master,task,0.0);
}
} // namespace engineering
} // namespace embedding::evaluation::two_component
