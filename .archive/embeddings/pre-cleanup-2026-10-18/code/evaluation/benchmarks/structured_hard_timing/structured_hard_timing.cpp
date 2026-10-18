// SPDX-License-Identifier: MIT
#include "structured_hard_timing.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
void require(bool ok, const char *why) {
  if (!ok) throw std::runtime_error(std::string("[structured hard timing] ") + why);
}
double uniform53(std::mt19937_64 &rng) {
  return static_cast<double>(rng() >> 11) / 9007199254740992.0;
}
ControlledDataset assemble(const std::vector<torch::Tensor> &values,
    const std::vector<torch::Tensor> &masks, const std::vector<int64_t> &labels,
    const std::vector<std::string> &sources) {
  const auto mask = torch::stack(masks);
  const auto x = torch::where(mask, torch::stack(values),
      torch::zeros({int64_t(values.size()),3,32,3}, torch::kFloat64));
  // The clean field is only a clone of the legal observations.
  return {{x.clone(),mask.clone()},{x,mask},torch::tensor(labels,torch::kInt64),sources};
}
} // namespace

StructuredHardTimingDevelopment make_structured_hard_timing_development(
    int64_t training_pairs, int64_t validation_pairs, uint64_t seed) {
  require(training_pairs>0 && validation_pairs>0 &&
      training_pairs<=std::numeric_limits<int64_t>::max()-validation_pairs,
      "positive nonoverflowing source counts required");
  const auto total=training_pairs+validation_pairs;
  require(total<=std::numeric_limits<int64_t>::max()/2,"paired row count overflow");
  std::vector<int64_t> assignments(static_cast<size_t>(total));
  std::iota(assignments.begin(),assignments.end(),0);
  std::mt19937_64 split_rng(stream_seed(seed,0x746d70332d73706cULL));
  std::shuffle(assignments.begin(),assignments.end(),split_rng);
  std::vector<bool> train(static_cast<size_t>(total),false);
  for(int64_t i=0;i<training_pairs;++i) train[assignments[i]]=true;
  std::vector<torch::Tensor> values[2],masks[2];
  std::vector<int64_t> labels[2]; std::vector<std::string> sources[2];
  constexpr double pi=3.14159265358979323846;
  for(int64_t source=0;source<total;++source) {
    std::mt19937_64 signal_rng(stream_seed(stream_seed(seed,0x746d70332d736967ULL),source+1));
    std::mt19937_64 mask_rng(stream_seed(stream_seed(seed,0x746d70332d6d6173ULL),source+1));
    std::mt19937_64 gap_rng(stream_seed(stream_seed(seed,0x746d70332d676170ULL),source+1));
    std::mt19937_64 order_rng(stream_seed(stream_seed(seed,0x746d70332d6f7264ULL),source+1));
    const double period=10+14*uniform53(signal_rng), delay=.25+.75*uniform53(signal_rng);
    const double phase=2*pi*uniform53(signal_rng);
    const double distractor_period=6+6*uniform53(signal_rng);
    const double distractor_phase=2*pi*uniform53(signal_rng);
    std::array<std::array<double,3>,3> gain{},offset{};
    std::array<int64_t,3> gap{};
    for(int64_t c=0;c<3;++c) {
      gap[c]=static_cast<int64_t>(30*uniform53(gap_rng)); // starts 0..29.
      for(int64_t f=0;f<3;++f) {
        gain[c][f]=(1+.1*f)*((c==2?1.0:.5)+uniform53(signal_rng));
        offset[c][f]=-.75+1.5*uniform53(signal_rng);
      }
    }
    std::normal_distribution<double> noise(0,.005);
    auto a=torch::zeros({3,32,3},torch::kFloat64), b=torch::zeros_like(a);
    auto m=torch::zeros_like(a,torch::kBool);
    auto av=a.accessor<double,3>(),bv=b.accessor<double,3>();
    auto mv=m.accessor<bool,3>();
    for(int64_t c=0;c<3;++c)
      for(int64_t t=0;t<32;++t)
        for(int64_t f=0;f<3;++f) {
          const double p=c==2?distractor_period:period;
          const double ph=c==2?distractor_phase:phase;
          const double shift=c==1?delay:0;
          const double nuisance=noise(signal_rng);
          av[c][t][f]=offset[c][f]+gain[c][f]*std::sin(2*pi*(t-shift)/p+ph)+nuisance;
          bv[c][t][f]=offset[c][f]+gain[c][f]*std::sin(2*pi*(t+shift)/p+ph)+nuisance;
          const bool observed=uniform53(mask_rng)>=.1;
          mv[c][t][f]=observed && !(t>=gap[c] && t<gap[c]+3);
        }
    const int split=train[source]?0:1;
    const bool swap=uniform53(order_rng)>=.5;
    const auto id=std::string(kStructuredHardTimingProtocol)+"/seed-"+
        std::to_string(seed)+"/lag_sign/source-"+std::to_string(source);
    for(int variant=0;variant<2;++variant) {
      const int label=swap?1-variant:variant;
      values[split].push_back(label?b:a); masks[split].push_back(m.clone());
      labels[split].push_back(label); sources[split].push_back(id);
    }
  }
  return {seed,assemble(values[0],masks[0],labels[0],sources[0]),
      assemble(values[1],masks[1],labels[1],sources[1])};
}

HardTimingSolvabilityResult structured_hard_timing_solvability(const Batch &observations) {
  const auto &input=observations.data,&mask=observations.feature_mask;
  require(input.defined() && input.device().is_cpu() && input.is_floating_point() &&
      input.dim()==4 && input.size(0)>0 && input.size(1)>=2 && input.size(2)>=6 && input.size(3)>0,
      "CPU floating observations [B,C>=2,H>=6,F>0] required");
  require(mask.defined() && mask.device().is_cpu() && mask.scalar_type()==torch::kBool &&
      mask.sizes()==input.sizes(),"exact CPU bool mask required");
  require(torch::isfinite(input.masked_select(mask)).all().item<bool>(),"observations must be finite");
  const auto x=input.to(torch::kFloat64).contiguous(),m=mask.contiguous();
  const auto xv=x.accessor<double,4>(); const auto mv=m.accessor<bool,4>();
  const auto B=x.size(0),H=x.size(2),F=x.size(3);
  HardTimingSolvabilityResult out{torch::zeros({B},torch::kInt64),torch::zeros({B},torch::kBool),
      torch::zeros({B},torch::kFloat64),torch::zeros({B},torch::kInt64)};
  auto predictions=out.predictions.accessor<int64_t,1>(); auto valid=out.valid.accessor<bool,1>();
  auto margins=out.margins.accessor<double,1>(); auto counts=out.supported_time_positions.accessor<int64_t,1>();
  for(int64_t row=0;row<B;++row) {
    double total=0;
    for(int64_t centre=1;centre<H-1;++centre) {
      double centre_sum=0; int spacings=0;
      for(int64_t k=1;k<=4;++k) {
        if(centre-k<0 || centre+k>=H) continue;
        double feature_sum=0; int64_t features=0;
        for(int64_t f=0;f<F;++f) {
          bool observed=true;
          for(int64_t c=0;c<2;++c)
            for(const auto t:{centre-k,centre,centre+k}) observed=observed && mv[row][c][t][f];
          if(!observed) continue;
          const double left0=xv[row][0][centre][f]-xv[row][0][centre-k][f];
          const double left1=xv[row][1][centre][f]-xv[row][1][centre-k][f];
          const double right0=xv[row][0][centre+k][f]-xv[row][0][centre][f];
          const double right1=xv[row][1][centre+k][f]-xv[row][1][centre][f];
          const double value=left1*right0-left0*right1;
          require(std::isfinite(value),"analytic tuple overflow"); feature_sum+=value; ++features;
        }
        if(features) {centre_sum+=feature_sum/features; ++spacings;}
      }
      if(spacings) {total+=centre_sum/spacings; ++counts[row];}
    }
    if(counts[row]) margins[row]=total/counts[row];
    require(std::isfinite(margins[row]),"analytic mean overflow");
    valid[row]=counts[row]>=4 && margins[row]!=0;
    if(valid[row]) predictions[row]=margins[row]>0;
  }
  return out;
}
} // namespace embedding::evaluation
