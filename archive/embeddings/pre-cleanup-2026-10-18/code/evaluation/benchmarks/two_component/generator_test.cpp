// SPDX-License-Identifier: MIT
#include "two_component.h"
#include "embedding/shared/feature_harness.h"
#include <ATen/Context.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#ifndef TWO_COMPONENT_SOURCE_ID
#define TWO_COMPONENT_SOURCE_ID "unrecorded"
#endif
namespace tc=embedding::evaluation::two_component;
namespace ev=embedding::evaluation;
namespace {
int64_t checks=0,negatives=0;
void check(bool ok,const char *why) {++checks;if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F call) {
  bool rejected=false;try{call();}catch(const std::exception &){rejected=true;}check(rejected,"malformed generator/view accepted");++negatives;
}
bool exact(const torch::Tensor &a,const torch::Tensor &b) {
  return a.scalar_type()==b.scalar_type()&&a.sizes()==b.sizes()&&a.is_contiguous()&&b.is_contiguous()&&
      (!a.numel()||std::memcmp(a.const_data_ptr(),b.const_data_ptr(),a.numel()*a.element_size())==0);
}
void equal_split(const tc::Split &a,const tc::Split &b) {
  check(exact(a.observed,b.observed)&&exact(a.mask,b.mask)&&exact(a.scoring_labels,b.scoring_labels)&&exact(a.source_ids,b.source_ids),"deterministic full legal split");
}
std::set<int64_t> check_split(const tc::Split &s,int64_t pairs,uint64_t master,tc::Task task) {
  const std::vector<int64_t> shape{2*pairs,3,32,3};
  check(s.observed.device().is_cpu()&&s.observed.scalar_type()==torch::kFloat64&&s.observed.is_contiguous()&&s.observed.sizes()==torch::IntArrayRef(shape),"observed CPUF64 N3H32F3");
  check(s.mask.device().is_cpu()&&s.mask.scalar_type()==torch::kBool&&s.mask.is_contiguous()&&s.mask.sizes()==s.observed.sizes(),"legal Bool mask");
  check(s.scoring_labels.device().is_cpu()&&s.scoring_labels.scalar_type()==torch::kInt64&&s.source_ids.device().is_cpu()&&s.source_ids.scalar_type()==torch::kInt64&&
      s.scoring_labels.sizes()==torch::IntArrayRef({2*pairs})&&s.source_ids.sizes()==s.scoring_labels.sizes(),"labels/source Long N");
  check(torch::isfinite(s.observed).all().item<bool>()&&s.observed.masked_select(~s.mask).eq(0).all().item<bool>(),"finite legal values and hiddenzero");
  check(s.observed.const_data_ptr()!=s.scoring_labels.const_data_ptr()&&s.observed.const_data_ptr()!=s.source_ids.const_data_ptr()&&
      s.scoring_labels.const_data_ptr()!=s.source_ids.const_data_ptr()&&s.mask.const_data_ptr()!=s.scoring_labels.const_data_ptr(),"labels/IDs never alias observed values or each other");
  std::set<int64_t> ids;int64_t previous=-1;const int64_t bit=task==tc::Task::SlowLagSign?0:1;
  for(int64_t row=0;row<2*pairs;row+=2) {
    const auto id=s.source_ids[row].item<int64_t>();check(id>previous&&id==s.source_ids[row+1].item<int64_t>()&&(id&1)==bit&&uint64_t(id>>31)==master,"ascending injective seed/task/source IDs");previous=id;ids.insert(id);
    check(s.scoring_labels[row].item<int64_t>()+s.scoring_labels[row+1].item<int64_t>()==1,"opposite randomized labels in wholepair");
    check(torch::equal(s.mask[row],s.mask[row+1]),"nuisance mask not label-dependent");
    check(torch::equal(s.observed[row][2],s.observed[row+1][2]),"unrelated channel2 waveform/noise label-independent");
    if(task==tc::Task::SlowLagSign)check(torch::equal(s.observed[row][0],s.observed[row+1][0]),"timing channel0 shared waveform");
    check(!torch::equal(s.observed[row][1],s.observed[row+1][1]),"task variants contain a physical signal difference");
    for(int64_t channel=0;channel<3;++channel) {
      bool gap=false;for(int64_t start=0;start<30;++start)gap=gap||!s.mask[row][channel].narrow(0,start,3).any().item<bool>();
      check(gap,"three consecutive hidden original ticks perchannel acrossfeatures");
    }
  }
  check(ids.size()==size_t(pairs)&&s.scoring_labels.eq(0).sum().item<int64_t>()==pairs&&s.scoring_labels.eq(1).sum().item<int64_t>()==pairs,"balanced sourcepaired population");
  return ids;
}
std::set<int64_t> check_development(const tc::Development &d,int64_t train_pairs,int64_t val_pairs,uint64_t master,tc::Task task) {
  const auto train=check_split(d.training,train_pairs,master,task),val=check_split(d.validation,val_pairs,master,task);
  auto all=train;all.insert(val.begin(),val.end());check(all.size()==size_t(train_pairs+val_pairs),"TRAIN/VAL pair-disjoint beforevariants");
  std::set<uint64_t> original;for(const auto id:all)original.insert((uint64_t(id)&0x7fffffffULL)>>1);
  for(uint64_t source=0;source<uint64_t(train_pairs+val_pairs);++source)check(original.count(source)==1,"whole original source indices, no resplit/renumber");
  return all;
}

// Independent scalar reconstruction of the frozen engineering draw spec. The
// test knows the artificial seed; no latent parameters are exported or provided
// to an information diagnostic/model/head. This catches sign/ratio/feature-gain
// errors without testing a classifier or using hidden data in the quality API.
double uniform(std::mt19937_64 &rng){return double(rng()>>11)/9007199254740992.0;}
void waveform_reference(const tc::Split &s,uint64_t master,tc::Task task,bool with_noise=false) {
  constexpr double pi=3.14159265358979323846;const uint64_t task_stream=task==tc::Task::SlowLagSign?tc::kSlowLagTaskStream:tc::kBalanceTaskStream;
  const auto root=ev::stream_seed(master,task_stream);const auto x=s.observed.accessor<double,4>();const auto mask=s.mask.accessor<bool,4>();
  for(int64_t row=0;row<s.observed.size(0);row+=2) {
    const auto source=(uint64_t(s.source_ids[row].item<int64_t>())&0x7fffffffULL)>>1;
    auto rng=[&](uint64_t stream){return std::mt19937_64(ev::stream_seed(ev::stream_seed(root,stream),source+1));};
    auto signal=rng(tc::kSignalStream),distractor=rng(tc::kDistractorStream),affine=rng(tc::kAffineStream),order=rng(tc::kOrderStream);
    auto noise=rng(tc::kNoiseStream),natural=rng(tc::kMaskStream),gaps=rng(tc::kGapStream);
    const double ps=12+8*uniform(signal),pf=4.5+2.5*uniform(signal),phi_s=2*pi*uniform(signal),phi_f=2*pi*uniform(signal);
    const double ls=.5+.5*uniform(signal),lf=.3+.4*uniform(signal);
    const double r=task==tc::Task::SlowLagSign?.8+.45*uniform(signal):1.8+.4*uniform(signal);
    const double sign_s=uniform(signal)>=.5?1:-1,sign_f=uniform(signal)>=.5?1:-1;
    const double ps2=12+8*uniform(distractor),pf2=4.5+2.5*uniform(distractor),phi_s2=2*pi*uniform(distractor),phi_f2=2*pi*uniform(distractor);
    const double r2=.8+.45*uniform(distractor),norm2=std::sqrt(1+r2*r2);
    std::array<std::array<double,3>,3> gain{},offset{};
    for(int64_t c=0;c<3;++c)for(int64_t f=0;f<3;++f){gain[c][f]=.5+uniform(affine);offset[c][f]=-.75+1.5*uniform(affine);}
    std::array<int64_t,3> gap{};for(int64_t c=0;c<3;++c)gap[c]=int64_t(30*uniform(gaps));
    std::array<double,288> epsilon{};
    for(int64_t c=0;c<3;++c)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f) {
      const auto u1=1-uniform(noise),u2=uniform(noise);
      epsilon[c*96+t*3+f]=(with_noise?.005:0)*std::sqrt(-2*std::log(u1))*std::cos(2*pi*u2);
      const bool present=uniform(natural)>=.10&&!(t>=gap[c]&&t<gap[c]+3);
      check(mask[row][c][t][f]==present&&mask[row+1][c][t][f]==present,"independent natural mask/gap draws, shared pair");
    }
    const int64_t first_label=uniform(order)>=.5?1:0;
    check(first_label==s.scoring_labels[row].item<int64_t>()&&1-first_label==s.scoring_labels[row+1].item<int64_t>(),"independent pair-order stream excludes labelselection");
    for(int64_t variant=0;variant<2;++variant) {
      const auto label=s.scoring_labels[row+variant].item<int64_t>();const double ratio=task==tc::Task::ComponentBalance&&label==0?1/r:r;
      const double normalization=std::sqrt(1+ratio*ratio),as=ratio/normalization,af=1/normalization;
      check(std::abs(as*as+af*af-1)<1e-15,"unit squared component energy in both taskvariants");
      if(task==tc::Task::ComponentBalance)check((label==1&&as>af)||(label==0&&as<af),"balance target is relative componentstrength");
      for(int64_t c=0;c<3;++c)for(int64_t t=0;t<32;++t)for(int64_t f=0;f<3;++f)if(mask[row+variant][c][t][f]) {
        const double shift_s=c==1?(task==tc::Task::SlowLagSign?(label?ls:-ls):sign_s*ls):0,shift_f=c==1?sign_f*lf:0;
        const double mixture=c==2?(r2*std::sin(2*pi*t/ps2+phi_s2)+std::sin(2*pi*t/pf2+phi_f2))/norm2:
            as*std::sin(2*pi*(t+shift_s)/ps+phi_s)+af*std::sin(2*pi*(t+shift_f)/pf+phi_f);
        const double expected=offset[c][f]+gain[c][f]*(1+.1*f)*mixture+epsilon[c*96+t*3+f];
        check(std::abs(x[row+variant][c][t][f]-expected)<2e-14,"independent legal two-band/affine/sign/Box-Muller law");
      }
    }
  }
}
void noise_contract(const tc::Split &noisy,const tc::Split &noiseless) {
  check(exact(noisy.mask,noiseless.mask)&&exact(noisy.scoring_labels,noiseless.scoring_labels)&&exact(noisy.source_ids,noiseless.source_ids),"engineeringnoise switch preserves every otherdraw");
  const auto residual=noisy.observed-noiseless.observed;double sum=0,square=0;int64_t count=0;
  for(int64_t row=0;row<noisy.observed.size(0);row+=2) {
    const auto mask=noisy.mask[row];const auto first=residual[row].masked_select(mask),second=residual[row+1].masked_select(mask);
    check(torch::allclose(first,second,0,2e-15),"exact paired Gaussian draw up to independent additionroundoff");
    sum+=first.sum().item<double>();square+=first.square().sum().item<double>();count+=first.numel();
  }
  const auto mean=sum/count,sd=std::sqrt(square/count-mean*mean);
  check(count>1000&&std::abs(mean)<.001&&sd>.004&&sd<.006,"Box-Muller engineering noise mean/SD sane");
}
void deletion_contract(const tc::Split &source) {
  const auto zero=tc::delete_observations(source,0,991),full=tc::delete_observations(source,1,991);
  equal_split(source,zero);check(!full.mask.any().item<bool>()&&full.observed.eq(0).all().item<bool>(),"probability1 hides everycell with zero storage");
  const auto low=tc::delete_observations(source,.15,991),high=tc::delete_observations(source,.30,991),again=tc::delete_observations(source,.30,991);
  equal_split(high,again);check(!high.mask.logical_and(low.mask.logical_not()).any().item<bool>(),"nested same-source deletion rates");
  check(torch::equal(high.observed.masked_select(high.mask),source.observed.masked_select(high.mask))&&
      high.observed.masked_select(~high.mask).eq(0).all().item<bool>(),"exact retained values/hiddenzero");
  check(exact(high.scoring_labels,source.scoring_labels)&&exact(high.source_ids,source.source_ids),"no dropped/relabelled sources");
  check(high.observed.const_data_ptr()!=source.observed.const_data_ptr()&&high.scoring_labels.const_data_ptr()!=source.scoring_labels.const_data_ptr(),"pure view does not alias mutable scoring/values");
  for(int64_t row=0;row<source.observed.size(0);row+=2)check(torch::equal(high.mask[row],high.mask[row+1]),"deletionkeyedsource not label/row");
  const auto reverse=torch::arange(source.observed.size(0)-1,-1,-1,torch::kInt64);
  const tc::Split reordered{source.observed.index_select(0,reverse),source.mask.index_select(0,reverse),
      source.scoring_labels.index_select(0,reverse),source.source_ids.index_select(0,reverse)};
  const auto shuffled=tc::delete_observations(reordered,.30,991);
  check(torch::equal(shuffled.observed,high.observed.index_select(0,reverse))&&torch::equal(shuffled.mask,high.mask.index_select(0,reverse)),"deletion invariant to pair/variant row ordering");
  check(!torch::equal(high.mask,tc::delete_observations(source,.30,992).mask),"separate deletionseed changesview");
  auto bad=source;bad.mask=source.mask.to(torch::kInt64);rejects([&]{tc::delete_observations(bad,.3,991);});
  bad=source;bad.observed=source.observed.to(torch::kFloat32);rejects([&]{tc::delete_observations(bad,.3,991);});
  bad=source;bad.scoring_labels=source.scoring_labels.clone();bad.scoring_labels[1]=bad.scoring_labels[0];rejects([&]{tc::delete_observations(bad,.3,991);});
  bad=source;bad.source_ids=source.source_ids.clone();bad.source_ids[1]=bad.source_ids[2];rejects([&]{tc::delete_observations(bad,.3,991);});
  bad=source;bad.observed=source.observed.clone();bad.observed[0][0][0][0]=std::numeric_limits<double>::quiet_NaN();rejects([&]{tc::delete_observations(bad,.3,991);});
  rejects([&]{tc::delete_observations(source,-.1,991);});rejects([&]{tc::delete_observations(source,1.1,991);});
  rejects([&]{tc::delete_observations(source,std::numeric_limits<double>::quiet_NaN(),991);});
}
void fixtures() {
  auto &generator=at::globalContext().defaultGenerator(at::Device(at::kCPU));const auto before=generator.get_state().clone();
  std::set<int64_t> all_ids;
  for(const auto task:{tc::Task::SlowLagSign,tc::Task::ComponentBalance})for(const auto master:{910901ULL,910902ULL}) {
    const auto data=tc::make_development(12,7,master,task),repeat=tc::make_development(12,7,master,task);
    equal_split(data.training,repeat.training);equal_split(data.validation,repeat.validation);
    const auto ids=check_development(data,12,7,master,task);
    for(const auto id:ids)check(all_ids.insert(id).second,"IDs disjoint across task/engineering seeds");
    const auto noiseless=tc::engineering::make_noiseless_development(12,7,master,task);
    waveform_reference(noiseless.training,master,task);waveform_reference(noiseless.validation,master,task);
    waveform_reference(data.training,master,task,true);waveform_reference(data.validation,master,task,true);
    noise_contract(data.training,noiseless.training);noise_contract(data.validation,noiseless.validation);
    deletion_contract(data.validation);
  }
  check(all_ids.size()==76,"all four artificial task/seed universes retained");
  const auto maximal_id=(((uint64_t(1)<<32)-1)<<31)|(((uint64_t(1)<<30)-1)<<1)|uint64_t(1);
  check(maximal_id==uint64_t(std::numeric_limits<int64_t>::max()),"bounded packed IDs exactly fit signedLong without a new fixtureseed");
  rejects([]{tc::make_development(0,1,1,tc::Task::SlowLagSign);});rejects([]{tc::make_development(1,0,1,tc::Task::SlowLagSign);});
  rejects([]{tc::make_development(std::size_t(1)<<30,1,1,tc::Task::SlowLagSign);});
  rejects([]{tc::make_development((std::size_t(1)<<30)-1,1,1,tc::Task::SlowLagSign);});
  rejects([]{tc::make_development(1,1,uint64_t(1)<<32,tc::Task::SlowLagSign);});
  rejects([]{tc::make_development(1,1,1,static_cast<tc::Task>(99));});
  rejects([]{tc::task_name(static_cast<tc::Task>(99));});rejects([]{tc::dataset_id(static_cast<tc::Task>(99));});
  check(tc::task_name(tc::Task::SlowLagSign)=="slow_lag_sign"&&tc::dataset_id(tc::Task::SlowLagSign)=="TEMPO-4"&&
      tc::task_name(tc::Task::ComponentBalance)=="component_balance"&&tc::dataset_id(tc::Task::ComponentBalance)=="AMP-2","separate task/dataset names");
  check(exact(before,generator.get_state().contiguous()),"all generation/deletion/boundary fixtures consume zero Torch RNG");
  check(negatives==40,"all declared malformed generator/view cases");
}
} // namespace
int main() try {
  torch::set_num_threads(1);fixtures();
  std::cout<<"Two-component generator engineering passed: "<<checks<<" checks;"<<negatives<<" negatives;engineering seeds910901/910902;no models/heads/quality seeds\n"<<TWO_COMPONENT_SOURCE_ID<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
