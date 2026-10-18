// SPDX-License-Identifier: MIT
#include "cross_feature_solvability.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace embedding::evaluation {
namespace {
void require(bool ok, const char *why) {
  if (!ok) throw std::runtime_error(std::string("[cross-feature timing] ") + why);
}
} // namespace

HardTimingSolvabilityResult cross_feature_solvability(const Batch &observations) {
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
      double centre_sum=0; int64_t spacings=0;
      for(int64_t k=1;k<=4;++k) {
        if(centre-k<0 || centre+k>=H) continue;
        std::array<double,2> left{},right{};
        std::array<int64_t,2> features{};
        for(int64_t c=0;c<2;++c) {
          for(int64_t f=0;f<F;++f) {
            if(!mv[row][c][centre-k][f] || !mv[row][c][centre][f] || !mv[row][c][centre+k][f]) continue;
            const double l=xv[row][c][centre][f]-xv[row][c][centre-k][f];
            const double r=xv[row][c][centre+k][f]-xv[row][c][centre][f];
            require(std::isfinite(l) && std::isfinite(r),"analytic difference overflow");
            left[c]+=l; right[c]+=r; ++features[c];
          }
          if(features[c]) {left[c]/=features[c]; right[c]/=features[c];}
          require(std::isfinite(left[c]) && std::isfinite(right[c]),"analytic feature mean overflow");
        }
        if(!features[0] || !features[1]) continue;
        const double value=left[1]*right[0]-left[0]*right[1];
        require(std::isfinite(value),"analytic determinant overflow");
        centre_sum+=value; ++spacings;
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
