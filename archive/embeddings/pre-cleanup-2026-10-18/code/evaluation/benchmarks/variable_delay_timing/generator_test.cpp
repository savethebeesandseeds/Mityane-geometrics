// SPDX-License-Identifier: MIT
#include "variable_delay_timing.h"
#include "shared_test_support.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>

namespace ev=embedding::evaluation;
namespace {
template<class F> void rejects(F action,const char *why) {
  bool rejected=false;
  try { action(); } catch(const std::exception &) { rejected=true; }
  test::check(rejected,why);
}
void source_and_rng_contract() {
  torch::manual_seed(739);
  const auto expected=torch::rand({12},torch::kFloat64);
  torch::manual_seed(739);
  const auto data=ev::make_variable_delay_timing_development(7,5,101);
  const auto repeated=ev::make_variable_delay_timing_development(7,5,101);
  test::close(torch::rand({12},torch::kFloat64),expected,"generator changed Torch RNG",0,0);
  std::set<std::string> all_sources;
  int split_index=0;
  for(const auto *split:{&data.training,&data.validation}) {
    const auto *again=split_index ? &repeated.validation : &repeated.training;
    const int64_t pairs=split_index ? 5 : 7;
    test::check(split->observed.data.sizes()==torch::IntArrayRef({2*pairs,3,32,3}),"closed geometry/count");
    test::check(split->observed.data.scalar_type()==torch::kFloat64 && split->observed.data.device().is_cpu(),"generation precision/device");
    test::close(split->observed.data,again->observed.data,"local seeded signal determinism",0,0);
    test::check(torch::equal(split->observed.feature_mask,again->observed.feature_mask) &&
        split->source_ids==again->source_ids && torch::equal(split->labels,again->labels),"mask/order/split determinism");
    test::check(split->labels.sum().item<int64_t>()==pairs,"balanced paired labels");
    test::close(split->clean.data,split->observed.data,"hidden clean signal escaped",0,0);
    test::check(torch::equal(split->clean.feature_mask,split->observed.feature_mask),"clean field reveals hidden support");
    test::check(split->observed.data.masked_select(split->observed.feature_mask.logical_not()).eq(0).all().item<bool>(),"hidden storage nonzero");
    std::map<std::string,std::vector<int64_t>> groups;
    for(int64_t row=0;row<2*pairs;++row) groups[split->source_ids[row]].push_back(row);
    test::check(groups.size()==size_t(pairs),"independent source count");
    for(const auto &[id,rows]:groups) {
      test::check(id.starts_with(std::string(ev::kVariableDelayTimingProtocol)+"/seed-101/lag_sign/source-") &&
          all_sources.insert(id).second,"source namespace or split overlap");
      test::check(rows.size()==2 && split->labels[rows[0]].item<int64_t>()!=split->labels[rows[1]].item<int64_t>(),"source lost paired variants");
      test::check(torch::equal(split->observed.feature_mask[rows[0]],split->observed.feature_mask[rows[1]]),"mask encodes sign");
      for(const int64_t channel:{0,2})
        test::close(split->observed.data[rows[0]][channel],split->observed.data[rows[1]][channel],"unrelated channel encodes sign",0,0);
    }
    ++split_index;
  }
  const auto other=ev::make_variable_delay_timing_development(7,5,102);
  test::check(!torch::equal(data.training.observed.data,other.training.observed.data),"seed does not change signals");
  rejects([]{(void)ev::make_variable_delay_timing_development(0,1,1);},"empty TRAIN accepted");
  rejects([]{(void)ev::make_variable_delay_timing_development(1,0,1);},"empty VALIDATION accepted");
  rejects([]{(void)ev::make_variable_delay_timing_development(std::numeric_limits<int64_t>::max(),1,1);},"count overflow accepted");
}
embedding::Batch noiseless(double period,double lag,double phase,int sign) {
  auto x=torch::zeros({1,3,32,3},torch::kFloat64);
  auto a=x.accessor<double,4>();
  constexpr double pi=3.14159265358979323846;
  for(int64_t c=0;c<3;++c)
    for(int64_t t=0;t<32;++t)
      for(int64_t f=0;f<3;++f)
        a[0][c][t][f]=(1+.1*f)*std::sin(2*pi*(t+(c==1 ? sign*lag : 0))/period+
            (c==2 ? phase+1.17 : phase));
  return {x,torch::ones_like(x,torch::kBool)};
}
void analytic_identity() {
  constexpr double pi=3.14159265358979323846;
  for(const double period:{12.,16.,20.})
    for(const double lag:{.5,1.,1.5})
      for(const double phase:{0.,1.31,5.98})
        for(const int sign:{-1,1}) {
          const auto input=noiseless(period,lag,phase,sign);
          const auto result=ev::variable_delay_timing_solvability(input);
          const double expected=sign*(1+1.1*1.1+1.2*1.2)/3*
              std::sin(2*pi/period)*std::sin(2*pi*lag/period);
          test::check(result.valid[0].item<bool>() && result.supported_time_positions[0].item<int64_t>()==31 &&
              result.predictions[0].item<int64_t>()==(sign==1),"clean physical sign/coverage");
          test::check(std::abs(result.margins[0].item<double>()-expected)<1e-12,"phase-independent analytic identity");
        }
}
void support_hidden_and_input_contract() {
  auto input=noiseless(14,.6,.81,1);
  input.data=input.data.repeat({4,1,1,1});
  input.feature_mask=torch::zeros_like(input.data,torch::kBool);
  input.feature_mask[0].slice(1,0,5).fill_(true); // four supported distinct times
  input.feature_mask[1].slice(1,0,4).fill_(true); // three: abstain
  input.feature_mask[2].fill_(true);
  input.data[2].zero_(); // supported geometry, zero signal: abstain
  for(int64_t t=0;t<32;t+=2) input.feature_mask[3].select(1,t).fill_(true); // no adjacent tuple
  const auto result=ev::variable_delay_timing_solvability(input);
  test::check(result.supported_time_positions[0].item<int64_t>()==4 && result.valid[0].item<bool>(),"four-time threshold");
  test::check(result.supported_time_positions[1].item<int64_t>()==3 && !result.valid[1].item<bool>(),"insufficient support did not abstain");
  test::check(result.supported_time_positions[2].item<int64_t>()==31 && !result.valid[2].item<bool>(),"zero margin did not abstain");
  test::check(result.supported_time_positions[3].item<int64_t>()==0 && !result.valid[3].item<bool>(),"nonadjacent cells became legal tuples");
  auto poisoned=embedding::Batch{input.data.clone(),input.feature_mask.clone()};
  poisoned.data.masked_fill_(poisoned.feature_mask.logical_not(),std::numeric_limits<double>::quiet_NaN());
  const auto hidden=ev::variable_delay_timing_solvability(poisoned);
  test::close(result.margins,hidden.margins,"hidden values affect analytic check",0,0);
  test::check(torch::equal(result.predictions,hidden.predictions) && torch::equal(result.valid,hidden.valid) &&
      torch::equal(result.supported_time_positions,hidden.supported_time_positions),"hidden values affect support/prediction");
  auto float_input=embedding::Batch{input.data.to(torch::kFloat32),input.feature_mask};
  const auto floats=ev::variable_delay_timing_solvability(float_input);
  test::check(torch::equal(floats.valid,result.valid) && torch::equal(floats.predictions,result.predictions),"float32 legal observations unsupported");
  auto observed_bad=embedding::Batch{input.data.clone(),input.feature_mask};
  observed_bad.data[0][0][0][0]=std::numeric_limits<double>::infinity();
  rejects([&]{(void)ev::variable_delay_timing_solvability(observed_bad);},"observed nonfinite accepted");
  auto wrong_mask=embedding::Batch{input.data,input.feature_mask.to(torch::kFloat32)};
  rejects([&]{(void)ev::variable_delay_timing_solvability(wrong_mask);},"nonboolean mask accepted");
  auto wrong_shape=embedding::Batch{input.data.slice(1,0,1),input.feature_mask.slice(1,0,1)};
  rejects([&]{(void)ev::variable_delay_timing_solvability(wrong_shape);},"single channel accepted");
}
} // namespace
int main() {
  try {
    at::set_num_threads(1);
    source_and_rng_contract(); analytic_identity(); support_hidden_and_input_contract();
    std::cout<<"Variable-delay timing data-only fixtures passed; no encoder/head fitting or quality evaluation\n";
    return 0;
  } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
