// SPDX-License-Identifier: MIT
#include "configurable_capacity.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace embedding::evaluation::configurable_capacity {
namespace {
constexpr int dimensions=5;
constexpr double tau=6.283185307179586476925286766559;
using Vector=std::array<double,dimensions>;
using Matrix=std::array<Vector,dimensions>;
struct Wave { std::vector<double> sine,cosine; };
struct Series {
  int channel=0,feature=0;
  std::vector<bool> legal;
  std::vector<double> centered;
  double energy=0,mean=0;
};
struct Fit { Vector coefficients{};double residual=0; };
struct Candidate {
  bool valid=false;double slow=0,fast=0,score=0;
  int64_t evaluations=0;std::vector<Fit> fits;
};
Wave wave(double frequency,int64_t history) {
  Wave out;out.sine.resize(history);out.cosine.resize(history);
  for(int64_t t=0;t<history;++t) {
    out.sine[t]=std::sin(tau*frequency*t);out.cosine[t]=std::cos(tau*frequency*t);
  }
  return out;
}
// The unchanged five-column normal-equation arithmetic follows the independent
// observed-two-band-affine-fit-v1 rule. Only the declared finite period family,
// dynamic history and timing-only task differ here; no source truth is used.
bool solve(const Matrix &gram,const Vector &rhs,Vector &coefficients) {
  Matrix lower{};Vector diagonal{},scaled_rhs{},temporary{},solution{};
  for(int i=0;i<dimensions;++i) {
    if(!(gram[i][i]>0)||!std::isfinite(gram[i][i]))return false;
    diagonal[i]=std::sqrt(gram[i][i]);scaled_rhs[i]=rhs[i]/diagonal[i];
    if(!std::isfinite(scaled_rhs[i]))return false;
  }
  for(int i=0;i<dimensions;++i)for(int j=0;j<=i;++j) {
    double value=gram[i][j]/diagonal[i]/diagonal[j];
    for(int k=0;k<j;++k)value-=lower[i][k]*lower[j][k];
    if(!std::isfinite(value))return false;
    if(i==j) {
      if(value<kInformationScaledPivotFloor)return false;
      lower[i][j]=std::sqrt(value);
    } else lower[i][j]=value/lower[j][j];
  }
  for(int i=0;i<dimensions;++i) {
    double value=scaled_rhs[i];for(int j=0;j<i;++j)value-=lower[i][j]*temporary[j];
    temporary[i]=value/lower[i][i];
  }
  for(int i=dimensions-1;i>=0;--i) {
    double value=temporary[i];for(int j=i+1;j<dimensions;++j)value-=lower[j][i]*solution[j];
    solution[i]=value/lower[i][i];coefficients[i]=solution[i]/diagonal[i];
    if(!std::isfinite(coefficients[i]))return false;
  }
  return true;
}
bool fit(const Series &series,const std::vector<Vector> &basis,Fit &out) {
  Matrix gram{};Vector rhs{};
  for(size_t t=0;t<basis.size();++t)if(series.legal[t]) {
    for(int i=0;i<dimensions;++i) {
      rhs[i]+=basis[t][i]*series.centered[t];
      for(int j=0;j<=i;++j)gram[i][j]+=basis[t][i]*basis[t][j];
    }
  }
  for(int i=0;i<dimensions;++i)for(int j=0;j<i;++j)gram[j][i]=gram[i][j];
  if(!solve(gram,rhs,out.coefficients))return false;
  long double squared_error=0;
  for(size_t t=0;t<basis.size();++t)if(series.legal[t]) {
    double predicted=0;for(int i=0;i<dimensions;++i)predicted+=basis[t][i]*out.coefficients[i];
    const long double error=series.centered[t]-predicted;squared_error+=error*error;
  }
  out.residual=static_cast<double>(squared_error/series.energy);
  const auto &c=out.coefficients;
  const double slow=c[1]*c[1]+c[2]*c[2],fast=c[3]*c[3]+c[4]*c[4],total=slow+fast;
  return std::isfinite(out.residual)&&std::isfinite(total)&&total>kInformationEnergyFloor&&
      slow/total>=kInformationMinimumComponentEnergyFraction&&fast/total>=kInformationMinimumComponentEnergyFraction;
}
bool phases(const std::vector<Series> &series,const std::vector<Fit> &fits,
    std::array<std::array<double,2>,2> &slow_direction) {
  for(int channel=0;channel<2;++channel)for(int component=0;component<2;++component) {
    const int first=1+2*component;double cosine=0,sine=0;int count=0;
    for(size_t i=0;i<series.size();++i)if(series[i].channel==channel) {
      const auto &c=fits[i].coefficients;const double amplitude=std::hypot(c[first],c[first+1]);
      if(!(amplitude>0)||!std::isfinite(amplitude))return false;
      cosine+=c[first]/amplitude;sine+=c[first+1]/amplitude;++count;
    }
    const double length=std::hypot(cosine,sine);
    if(count<kInformationMinimumFeaturesPerChannel||!std::isfinite(length)||
        length/count<kInformationMinimumPhaseCoherence)return false;
    if(component==0)slow_direction[channel]={cosine/length,sine/length};
  }
  return true;
}
Candidate evaluate(double slow,double fast,const Wave &slow_wave,const Wave &fast_wave,
    const std::vector<Series> &series) {
  Candidate out;out.slow=slow;out.fast=fast;
  std::vector<Vector> basis(slow_wave.sine.size());
  for(size_t t=0;t<basis.size();++t)basis[t]={1,slow_wave.sine[t],slow_wave.cosine[t],fast_wave.sine[t],fast_wave.cosine[t]};
  out.fits.resize(series.size());
  for(size_t i=0;i<series.size();++i) {
    if(!fit(series[i],basis,out.fits[i]))return out;
    out.score+=out.fits[i].residual;
  }
  std::array<std::array<double,2>,2> direction{};
  if(!phases(series,out.fits,direction))return out;
  out.score/=series.size();out.valid=std::isfinite(out.score);return out;
}
Candidate search(const std::vector<Series> &series,const std::vector<Wave> &slow_waves,const std::vector<Wave> &fast_waves) {
  Candidate best;int64_t evaluations=0;
  for(int ps=32;ps<=48;++ps)for(int pf=8;pf<=14;++pf) {
    const auto candidate=evaluate(1.0/ps,1.0/pf,slow_waves[ps-32],fast_waves[pf-8],series);++evaluations;
    // Ascending periods fix the tie order. Neither labels nor source identity
    // can affect search or tie-breaking.
    if(candidate.valid&&(!best.valid||candidate.score<best.score))best=candidate;
  }
  best.evaluations=evaluations;return best;
}
InformationDecision decide(const std::vector<Series> &series,const Candidate &best) {
  InformationDecision out;out.frequency_pairs_evaluated=best.evaluations;
  if(!best.valid){out.reason="no_stable_coherent_two_band_fit";return out;}
  out.slow_frequency=best.slow;out.fast_frequency=best.fast;out.normalized_residual=best.score;
  for(size_t i=0;i<series.size();++i) {
    const auto row=series[i].channel*3+series[i].feature;
    out.eligible[row]=true;out.coefficients[row]=best.fits[i].coefficients;
    out.coefficients[row][0]+=series[i].mean;
  }
  if(best.score>kInformationMaximumNormalizedResidual){out.reason="normalized_residual_above_limit";return out;}
  std::array<std::array<double,2>,2> direction{};
  if(!phases(series,best.fits,direction)){out.reason="incoherent_component_phases";return out;}
  const auto &a=direction[0],&b=direction[1];
  out.margin=std::atan2(a[0]*b[1]-a[1]*b[0],a[0]*b[0]+a[1]*b[1]);
  if(!std::isfinite(out.margin)||std::abs(out.margin)<kInformationMinimumSlowPhaseMargin||
      std::abs(out.margin)>kInformationMaximumSlowPhaseMargin) {
    out.reason="slow_phase_margin_outside_support";return out;
  }
  out.predicted_label=out.margin>0?1:0;out.supported=true;out.reason="supported";return out;
}
} // namespace

std::vector<InformationDecision> infer_information(const torch::Tensor &observed,const torch::Tensor &mask) {
  TORCH_CHECK(observed.defined()&&mask.defined()&&observed.device().is_cpu()&&mask.device().is_cpu()&&
      (observed.scalar_type()==torch::kFloat64||observed.scalar_type()==torch::kFloat32)&&
      mask.scalar_type()==torch::kBool&&observed.dim()==4&&observed.size(0)>0&&mask.sizes()==observed.sizes(),
      "[tempo5-information] CPU float64/32 observations and Bool mask required");
  validate_dimensions({observed.size(1),observed.size(2),observed.size(3)});
  const auto values=observed.detach().to(torch::kFloat64).contiguous(),legal=mask.contiguous();
  auto x=values.accessor<double,4>();auto m=legal.accessor<bool,4>();const int64_t history=values.size(2);
  for(int64_t b=0;b<values.size(0);++b)for(int c=0;c<3;++c)for(int64_t t=0;t<history;++t)for(int f=0;f<9;++f)
    TORCH_CHECK(!m[b][c][t][f]||std::isfinite(x[b][c][t][f]),"[tempo5-information] legal observations must be finite");
  std::vector<Wave> slow_waves,fast_waves;
  for(int ps=32;ps<=48;++ps)slow_waves.push_back(wave(1.0/ps,history));
  for(int pf=8;pf<=14;++pf)fast_waves.push_back(wave(1.0/pf,history));
  std::vector<InformationDecision> result;result.reserve(values.size(0));
  for(int64_t b=0;b<values.size(0);++b) {
    std::vector<Series> series;bool enough=true;
    for(int c=0;c<2;++c) {
      std::vector<bool> union_ticks(history,false);int features=0;
      for(int f=0;f<3;++f) {
        Series item;item.channel=c;item.feature=f;item.legal.resize(history);item.centered.resize(history);
        int64_t count=0;long double sum=0;
        for(int64_t t=0;t<history;++t) {
          item.legal[t]=m[b][c][t][f];if(item.legal[t]){++count;sum+=x[b][c][t][f];}
        }
        if(count<kInformationMinimumFeatureTicks)continue;
        const long double mean=sum/count;item.mean=static_cast<double>(mean);long double energy=0;
        for(int64_t t=0;t<history;++t)if(item.legal[t]) {
          item.centered[t]=static_cast<double>(static_cast<long double>(x[b][c][t][f])-mean);
          energy+=static_cast<long double>(item.centered[t])*item.centered[t];
        }
        item.energy=static_cast<double>(energy);
        if(!(item.energy>kInformationEnergyFloor)||!std::isfinite(item.energy))continue;
        for(int64_t t=0;t<history;++t)union_ticks[t]=union_ticks[t]||item.legal[t];
        ++features;series.push_back(std::move(item));
      }
      int64_t first=history,last=-1,ticks=0;
      for(int64_t t=0;t<history;++t)if(union_ticks[t]){first=std::min(first,t);last=t;++ticks;}
      enough=enough&&features>=kInformationMinimumFeaturesPerChannel&&ticks>=kInformationMinimumChannelUnionTicks&&
          last-first>=kInformationMinimumChannelSpan;
    }
    if(!enough) { InformationDecision out;out.reason="insufficient_observed_time_or_feature_support";result.push_back(out); }
    else result.push_back(decide(series,search(series,slow_waves,fast_waves)));
  }
  return result;
}
} // namespace embedding::evaluation::configurable_capacity
