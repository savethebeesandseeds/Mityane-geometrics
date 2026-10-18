// SPDX-License-Identifier: MIT
#include "configurable_capacity.h"
#include "embedding/shared/data.h"
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <unistd.h>

namespace cc=embedding::evaluation::configurable_capacity;
namespace fs=std::filesystem;
namespace {
void check(bool value,const char *why) { if(!value)throw std::runtime_error(why); }
template<class F>void rejects(F action,const char *why) {
  bool rejected=false;try{action();}catch(const std::exception &){rejected=true;}check(rejected,why);
}
void deterministic_contract() {
  torch::manual_seed(631);const auto expected=torch::rand({8},torch::kFloat64);
  torch::manual_seed(631);
  const auto data=cc::make_development(7,5,941000),again=cc::make_development(7,5,941000);
  check(torch::equal(torch::rand({8},torch::kFloat64),expected),"generator consumed Torch RNG");
  std::set<int64_t> all;
  for(const auto &[a,b]:{std::pair{&data.training,&again.training},std::pair{&data.validation,&again.validation}}) {
    cc::validate(*a);
    check(torch::equal(a->observed,b->observed)&&torch::equal(a->mask,b->mask)&&
        torch::equal(a->scoring_labels,b->scoring_labels)&&torch::equal(a->source_ids,b->source_ids),"same seed changed dataset");
    for(int64_t row=0;row<a->observed.size(0);row+=2) {
      const auto id=a->source_ids[row].item<int64_t>();
      check(all.insert(id).second&&a->source_ids[row+1].item<int64_t>()==id,"source overlap/fragmentation");
      check(a->scoring_labels[row].item<int64_t>()!=a->scoring_labels[row+1].item<int64_t>(),"pair labels not opposite");
      check(torch::equal(a->mask[row],a->mask[row+1]),"mask leaks label");
      check(torch::equal(a->observed[row][0],a->observed[row+1][0])&&
          torch::equal(a->observed[row][2],a->observed[row+1][2]),"unchanged channels leak label");
      using torch::indexing::Slice;
      check(torch::equal(a->observed[row].index({Slice(),Slice(),Slice(6,9)}),
          a->observed[row+1].index({Slice(),Slice(),Slice(6,9)})),"structured nuisances leak label");
      check(!torch::equal(a->observed[row][1],a->observed[row+1][1]),"timing variants identical");
      check(static_cast<uint64_t>(id)>>31==941000,"packed master identity changed");
    }
  }
  const auto other=cc::make_development(7,5,941000,{3,256,9});
  using torch::indexing::Slice;
  for(const auto &[a,b]:{std::pair{&data.training,&other.training},std::pair{&data.validation,&other.validation}}) {
    check(torch::equal(a->source_ids,b->source_ids)&&torch::equal(a->scoring_labels,b->scoring_labels),"history changed source/split/order");
    const auto long_prefix=b->observed.index({Slice(),Slice(),Slice(0,128),Slice()});
    const auto common=a->mask.logical_and(b->mask.index({Slice(),Slice(),Slice(0,128),Slice()}));
    check(torch::equal(a->observed.masked_select(common),long_prefix.masked_select(common)),"history stretched signals or changed noise prefix");
  }
  const auto zero=cc::delete_observations(data.validation,0,941000),one=cc::delete_observations(data.validation,1,941000);
  const auto low=cc::delete_observations(data.validation,.15,941000),high=cc::delete_observations(data.validation,.30,941000);
  const auto repeated=cc::delete_observations(data.validation,.30,941000);
  check(torch::equal(zero.observed,data.validation.observed)&&torch::equal(zero.mask,data.validation.mask),"zero deletion changed source");
  check(!one.mask.any().item<bool>()&&one.observed.eq(0).all().item<bool>(),"full deletion retained values");
  check(high.mask.logical_and(low.mask.logical_not()).any().item<bool>()==false,"deletion rates not nested");
  check(torch::equal(high.mask,repeated.mask)&&torch::equal(high.observed,repeated.observed),"deletion nondeterministic");
  check(torch::equal(high.scoring_labels,data.validation.scoring_labels)&&torch::equal(high.source_ids,data.validation.source_ids),"deletion changed population");
  check(torch::equal(high.observed.masked_select(high.mask),data.validation.observed.masked_select(high.mask)),"deletion changed legal values");
  cc::validate(high);
  const auto decisions=cc::infer_information(high.observed,high.mask);
  for(size_t i=0;i<decisions.size();++i)check(decisions[i].supported&&decisions[i].predicted_label==
      high.scoring_labels[i].item<int64_t>()&&decisions[i].frequency_pairs_evaluated==119,"noisy deleted timing witness failed");
}
void information_identity() {
  constexpr double tau=6.283185307179586476925286766559;
  for(const int period:{32,40,48})for(const int sign:{-1,1}) {
    auto x=torch::zeros({1,3,128,9},torch::kFloat64),mask=torch::ones_like(x,torch::kBool);
    auto a=x.accessor<double,4>();auto m=mask.accessor<bool,4>();
    for(int c=0;c<3;++c)for(int t=0;t<128;++t)for(int f=0;f<9;++f) {
      const double shift=c==1?sign*2.0:0;
      a[0][c][t][f]=.7+.2*f+(1+.1*f)*(std::sin(tau*(t+shift)/period+.83)+.25*std::sin(tau*(t+(c==1?.5:0))/11+1.17));
      if((t+3*c+f)%7==0||(t>=53&&t<58))m[0][c][t][f]=false;
    }
    const auto fit=cc::infer_information(x,mask)[0];
    check(fit.supported&&fit.predicted_label==(sign>0?1:0),"noiseless timing sign identity failed");
    check(std::abs(fit.margin-tau*sign*2.0/period)<1e-11&&std::abs(fit.slow_frequency-1.0/period)<1e-15&&
        std::abs(fit.fast_frequency-1.0/11)<1e-15&&fit.normalized_residual<1e-24,"noiseless phase/period/fit identity failed");
    auto poisoned=torch::where(mask,x,torch::full_like(x,std::numeric_limits<double>::quiet_NaN()));
    const auto hidden=cc::infer_information(poisoned,mask)[0];
    check(hidden.supported&&hidden.margin==fit.margin&&hidden.normalized_residual==fit.normalized_residual,
        "hidden values entered observed-only witness");
    // Independent selected-fit normal-equation stationarity in observation
    // units checks the exported coefficients, without rerunning grid search.
    for(int c=0;c<2;++c)for(int f=0;f<3;++f) {
      std::array<long double,5> gradient{};
      for(int t=0;t<128;++t)if(m[0][c][t][f]) {
        const std::array<double,5> basis{1,std::sin(tau*t/period),std::cos(tau*t/period),std::sin(tau*t/11),std::cos(tau*t/11)};
        double prediction=0;for(int col=0;col<5;++col)prediction+=basis[col]*fit.coefficients[c*3+f][col];
        for(int col=0;col<5;++col)gradient[col]+=basis[col]*(a[0][c][t][f]-prediction);
      }
      for(const auto value:gradient)check(std::abs(value)<1e-10,"selected fit not stationary");
    }
    const auto missing=cc::infer_information(x,torch::zeros_like(mask))[0];
    check(!missing.supported&&missing.frequency_pairs_evaluated==0,"all-missing row accepted");
    const auto constant=cc::infer_information(torch::ones_like(x),mask)[0];
    check(!constant.supported,"constant row accepted");
    auto bad=x.clone();bad[0][0][1][0]=std::numeric_limits<double>::infinity();
    rejects([&]{(void)cc::infer_information(bad,mask);},"legal nonfinite accepted");
  }
}
void archive_and_rejection_contract() {
  rejects([]{(void)cc::make_development(0,2,941000);},"empty TRAIN accepted");
  rejects([]{(void)cc::make_development(2,2,uint64_t(1)<<32);},"oversized master accepted");
  rejects([]{(void)cc::make_development(std::size_t(1)<<30,2,941000);},"oversized source count accepted");
  rejects([]{(void)cc::make_development(2,2,941000,{3,63,9});},"too short history accepted");
  rejects([]{(void)cc::make_development(2,2,941000,{3,128,8});},"undefined feature semantics accepted");
  const auto data=cc::make_development(2,2,941000);
  rejects([&]{(void)cc::delete_observations(data.validation,-.1,941000);},"negative deletion accepted");
  rejects([&]{(void)cc::delete_observations(data.validation,std::numeric_limits<double>::quiet_NaN(),941000);},"NaN deletion accepted");
  auto partial=data.validation;partial.scoring_labels=data.validation.scoring_labels.clone();partial.scoring_labels[0]=partial.scoring_labels[1];
  rejects([&]{cc::validate(partial);},"same-label pair accepted");
  auto odd=data.training;odd.source_ids=data.training.source_ids+1;
  rejects([&]{cc::validate(odd);},"nonzero source variant bit accepted");
  const auto parent=std::getenv("TMPDIR");fs::path base=parent?parent:"/tmp";
  std::string pattern=(base/"capacity-generator-test-XXXXXX").string();std::vector<char> writable(pattern.begin(),pattern.end());writable.push_back(0);
  const auto created=::mkdtemp(writable.data());check(created!=nullptr,"cannot create owned temporary fixture");fs::path directory=created;
  try {
    const auto path=directory/"observations.pt";
    cc::save_split_archive(path.string(),data.training,"TRAIN",941000);
    const auto loaded=cc::load_split_archive(path.string());
    check(torch::equal(loaded.observed,data.training.observed)&&torch::equal(loaded.mask,data.training.mask)&&
        torch::equal(loaded.scoring_labels,data.training.scoring_labels)&&torch::equal(loaded.source_ids,data.training.source_ids),"archive changed legal data or row metadata");
    rejects([&]{cc::save_split_archive(path.string(),data.training,"TRAIN",941000);},"existing archive overwritten");
    // Loading a known subset is insufficient: undeclared latent fields must
    // fail before this archive can reach a producer or analytic admission.
    const auto extra=directory/"extra-key.pt";
    torch::serialize::OutputArchive expanded;
    expanded.write("observed",data.training.observed,true);expanded.write("feature_mask",data.training.mask,true);
    expanded.write("labels_scoring_only",data.training.scoring_labels,true);expanded.write("source_ids",data.training.source_ids,true);
    expanded.write("source_ids_json",embedding::archive::text_tensor(cc::source_ids_json(data.training)),true);
    expanded.write("latent_periods",torch::ones({data.training.observed.size(0)},torch::kFloat64),true);
    expanded.save_to(extra.string());
    rejects([&]{(void)cc::load_split_archive(extra.string());},"undeclared latent archive field accepted");
    const auto link=directory/"existing-link.pt",target=directory/"missing-target";
    fs::create_symlink(target,link);
    rejects([&]{cc::save_split_archive(link.string(),data.training,"TRAIN",941000);},"dangling observation destination replaced");
    check(fs::is_symlink(link)&&fs::read_symlink(link)==target&&!fs::exists(target),"rejected symlink changed");
    const auto raced=directory/"concurrent.pt";
    std::atomic<int> published=0;std::barrier ready(3);
    const auto writer=[&] {
      ready.arrive_and_wait();
      try {cc::save_split_archive(raced.string(),data.training,"TRAIN",941000);++published;}
      catch(const std::exception &) {} // Exactly one competing publisher wins.
    };
    std::thread left(writer),right(writer);ready.arrive_and_wait();left.join();right.join();
    check(published==1,"competing publishers replaced an observation archive");
    const auto winner=cc::load_split_archive(raced.string());
    check(torch::equal(winner.observed,data.training.observed),"concurrent publication produced incomplete archive");
    const auto wrong_master=directory/"wrong-master.pt",odd_path=directory/"odd-source.pt";
    rejects([&]{cc::save_split_archive(wrong_master.string(),data.training,"TRAIN",941001);},"incorrect archive master accepted");
    rejects([&]{cc::save_split_archive(odd_path.string(),odd,"TRAIN",941000);},"odd packed source identity published");
    check(!fs::exists(wrong_master)&&!fs::exists(odd_path),"rejected metadata published an archive");
    fs::remove(raced);fs::remove(link);fs::remove(extra);fs::remove(path);fs::remove(directory);
  } catch(...) { throw; } // Keep failed evidence, including the owned directory.
}
}
int main() {
  try {
    torch::set_num_threads(1);torch::set_num_interop_threads(1);
    deterministic_contract();information_identity();archive_and_rejection_contract();
    std::cout<<"TEMPO-5 generator/observed-information engineering checks passed; no quality seed or encoder execution\n";
    return 0;
  } catch(const std::exception &error) {std::cerr<<"capacity generator test: "<<error.what()<<'\n';return 1;}
}
