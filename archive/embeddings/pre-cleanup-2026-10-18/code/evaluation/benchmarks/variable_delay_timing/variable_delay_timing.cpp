// SPDX-License-Identifier: MIT
#include "variable_delay_timing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
void require(bool ok, const char *why) {
  if (!ok) throw std::runtime_error(std::string("[variable-delay timing] ") + why);
}
double uniform53(std::mt19937_64 &rng) {
  return static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
}
ControlledDataset assemble(const std::vector<torch::Tensor> &values,
    const std::vector<torch::Tensor> &masks, const std::vector<int64_t> &labels,
    const std::vector<std::string> &sources) {
  const auto mask = torch::stack(masks);
  const auto x = torch::where(mask, torch::stack(values),
                             torch::zeros({int64_t(values.size()),3,32,3}, torch::kFloat64));
  return {{x.clone(), mask.clone()}, {x, mask}, torch::tensor(labels, torch::kInt64), sources};
}
} // namespace

VariableDelayTimingDevelopment make_variable_delay_timing_development(
    int64_t training_pairs, int64_t validation_pairs, uint64_t seed) {
  require(training_pairs > 0 && validation_pairs > 0 &&
      training_pairs <= std::numeric_limits<int64_t>::max() - validation_pairs,
      "positive nonoverflowing TRAIN/VALIDATION source counts required");
  const auto total = training_pairs + validation_pairs;
  require(total <= std::numeric_limits<int64_t>::max()/2,
          "paired example count overflow");
  std::vector<int64_t> assignments(static_cast<size_t>(total));
  std::iota(assignments.begin(), assignments.end(), 0);
  std::mt19937_64 split_rng(stream_seed(seed, 0x7664742d73706c31ULL));
  std::shuffle(assignments.begin(), assignments.end(), split_rng);
  std::vector<bool> is_training(static_cast<size_t>(total), false);
  for (int64_t i=0; i<training_pairs; ++i) is_training[assignments[i]]=true;
  std::vector<torch::Tensor> values[2], masks[2];
  std::vector<int64_t> labels[2];
  std::vector<std::string> sources[2];
  constexpr double pi = 3.14159265358979323846;
  for (int64_t source=0; source<total; ++source) {
    std::mt19937_64 signal_rng(stream_seed(stream_seed(seed,0x7664742d73696731ULL),source+1));
    std::mt19937_64 mask_rng(stream_seed(stream_seed(seed,0x7664742d6d617331ULL),source+1));
    std::mt19937_64 order_rng(stream_seed(stream_seed(seed,0x7664742d6f726431ULL),source+1));
    const double period=12.0+8.0*uniform53(signal_rng);
    const double delay=.5+uniform53(signal_rng);
    const double phase=2*pi*uniform53(signal_rng);
    const double unrelated_phase=2*pi*uniform53(signal_rng);
    std::normal_distribution<double> noise(0,.005);
    auto a=torch::zeros({3,32,3},torch::kFloat64), b=torch::zeros_like(a);
    auto m=torch::zeros_like(a,torch::kBool);
    auto av=a.accessor<double,3>(), bv=b.accessor<double,3>();
    auto mv=m.accessor<bool,3>();
    for (int64_t c=0;c<3;++c)
      for (int64_t t=0;t<32;++t)
        for (int64_t f=0;f<3;++f) {
          const double coefficient=1.0+.1*f;
          const double p=c==2 ? unrelated_phase : phase;
          const double offset=c==1 ? delay : 0.0;
          const double nuisance=noise(signal_rng);
          av[c][t][f]=coefficient*std::sin(2*pi*(t-offset)/period+p)+nuisance;
          bv[c][t][f]=coefficient*std::sin(2*pi*(t+offset)/period+p)+nuisance;
          mv[c][t][f]=uniform53(mask_rng)>=.10;
        }
    const int split=is_training[source] ? 0 : 1;
    const bool swap=uniform53(order_rng)>=.5;
    const auto id=std::string(kVariableDelayTimingProtocol)+"/seed-"+
        std::to_string(seed)+"/lag_sign/source-"+std::to_string(source);
    for (int variant=0;variant<2;++variant) {
      const int label=swap ? 1-variant : variant;
      values[split].push_back(label ? b : a);
      masks[split].push_back(m.clone());
      labels[split].push_back(label);
      sources[split].push_back(id);
    }
  }
  return {seed,assemble(values[0],masks[0],labels[0],sources[0]),
               assemble(values[1],masks[1],labels[1],sources[1])};
}

TimingSolvabilityResult variable_delay_timing_solvability(const Batch &observations) {
  const auto &input=observations.data, &mask=observations.feature_mask;
  require(input.defined() && input.device().is_cpu() && input.is_floating_point() &&
      input.dim()==4 && input.size(0)>0 && input.size(1)>=2 &&
      input.size(2)>=5 && input.size(3)>0,
      "CPU floating observations [B,C>=2,H>=5,F>0] required");
  require(mask.defined() && mask.device().is_cpu() && mask.scalar_type()==torch::kBool &&
      mask.sizes()==input.sizes(), "exact CPU bool observation mask required");
  require(torch::isfinite(input.masked_select(mask)).all().item<bool>(),
          "observed values must be finite");
  const auto x=input.to(torch::kFloat64).contiguous(), m=mask.contiguous();
  const auto xv=x.accessor<double,4>();
  const auto mv=m.accessor<bool,4>();
  const auto B=x.size(0), H=x.size(2), F=x.size(3);
  TimingSolvabilityResult out{torch::zeros({B},torch::kInt64),
      torch::zeros({B},torch::kBool),torch::zeros({B},torch::kFloat64),
      torch::zeros({B},torch::kInt64)};
  auto predictions=out.predictions.accessor<int64_t,1>();
  auto valid=out.valid.accessor<bool,1>();
  auto margins=out.margins.accessor<double,1>();
  auto counts=out.supported_time_positions.accessor<int64_t,1>();
  for (int64_t row=0;row<B;++row) {
    double sum=0;
    for (int64_t t=0;t<H-1;++t) {
      double time_sum=0; int64_t features=0;
      for (int64_t f=0;f<F;++f) {
        if (!(mv[row][0][t][f] && mv[row][1][t][f] &&
              mv[row][0][t+1][f] && mv[row][1][t+1][f])) continue;
        const double value=xv[row][1][t][f]*xv[row][0][t+1][f]-
                           xv[row][0][t][f]*xv[row][1][t+1][f];
        require(std::isfinite(value),"analytic tuple overflow");
        time_sum+=value; ++features;
      }
      if (features) { sum+=time_sum/features; ++counts[row]; }
    }
    if (counts[row]) margins[row]=sum/counts[row];
    require(std::isfinite(margins[row]),"analytic margin overflow");
    valid[row]=counts[row]>=4 && margins[row]!=0;
    if (valid[row]) predictions[row]=margins[row]>0;
  }
  return out;
}
} // namespace embedding::evaluation
