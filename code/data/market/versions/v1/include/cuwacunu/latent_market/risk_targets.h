// Copyright (c) 2026 Waajacu. All rights reserved.
// See ../../../../../../../../LICENSE for the copyright and permission notice.
#pragma once

#include "cuwacunu/latent_market/timeline.h"

namespace cuwacunu::latent_market::v1 {

struct CloseObservation {
  std::int64_t endpoint_ms=0, available_ms=0;
  double close=0;
  bool observed=false;
  bool operator==(const CloseObservation&) const = default;
};

struct AssetCloseSeries {
  std::int64_t channel_id=0;
  std::vector<CloseObservation> rows;
  bool operator==(const AssetCloseSeries&) const = default;
};

enum RiskIssue : std::uint64_t {
  risk_no_issue=0, missing_origin_close=1ull<<0, missing_future_close=1ull<<1,
  masked_close=1ull<<2, nonpositive_close=1ull<<3, nonfinite_close=1ull<<4,
  risk_role_boundary=1ull<<5, origin_close_unavailable=1ull<<6,
  risk_decision_before_origin=1ull<<7, nonfinite_risk_value=1ull<<8
};

struct CloseSupport {
  std::int64_t endpoint_ms=0;
  std::optional<std::int64_t> available_ms;
  bool present=false, observed=false, positive_finite=false;
  bool operator==(const CloseSupport&) const = default;
};

struct RiskTargetRow {
  std::string pair_id;
  std::int64_t endpoint_ms=0, decision_ms=0;
  Role role=Role::unassigned;
  std::int64_t channel_id=0, horizon_ticks=0;
  bool eligible=false;
  std::uint64_t reasons=risk_no_issue;
  // Values are meaningful only when eligible. These are log-return units,
  // without sqrt(R), annualization, price scaling or a sample (R-1) divisor.
  double dispersion_1m=0, downside=0, mean_return=0;
  std::uint64_t eligible_returns=0, expected_returns=0;
  std::optional<std::int64_t> available_ms; // all required closes have matured
  std::vector<CloseSupport> support; // exact requested origin plus R future keys
  std::vector<std::uint8_t> return_valid; // R adjacent-pair validity indicators
  bool operator==(const RiskTargetRow&) const = default;
};

struct FutureRiskTargets {
  std::string timeline_id, source_hash;
  std::string formula_id="centered-population-adjacent-log-return-dispersion-and-origin-close-path-downside.v1";
  std::string dispersion_units="one-candle log-return standard deviation";
  std::string downside_units="cumulative log-return loss relative to origin";
  std::int64_t interval_ms=0;
  std::vector<std::int64_t> horizons_ticks;
  std::vector<RiskTargetRow> rows;
  bool operator==(const FutureRiskTargets&) const = default;
};

// The first protocol has complete-window eligibility. Partial-return targets
// require a new formula identity rather than silently replacing R by a count.
inline FutureRiskTargets build_future_risk_targets(const std::string& timeline_id,
    const std::vector<PairOrigin>& origins, const std::vector<AssetCloseSeries>& sources,
    std::int64_t interval_ms, const std::vector<std::int64_t>& horizons_ticks,
    const std::vector<RoleInterval>& roles, const std::string& source_hash) {
  using namespace temporal_detail;
  require(!timeline_id.empty() && !source_hash.empty() && interval_ms>0 &&
          !sources.empty() && !horizons_ticks.empty(),"risk identity, cadence, assets and horizons required");
  validate_roles(roles);
  std::set<std::int64_t> horizon_set,channel_ids;
  for (auto h:horizons_ticks) require(h>0 && h<std::numeric_limits<std::int64_t>::max() &&
    static_cast<std::uint64_t>(h)<std::numeric_limits<std::size_t>::max() &&
    horizon_set.insert(h).second,"positive unique representable risk horizons required");
  std::vector<std::map<std::int64_t,const CloseObservation*>> indexed;
  for (const auto& source:sources) {
    require(channel_ids.insert(source.channel_id).second,"duplicate risk semantic channel ID");
    std::map<std::int64_t,const CloseObservation*> index;
    for (std::size_t i=0;i<source.rows.size();++i) {
      const auto& row=source.rows[i];
      require(row.endpoint_ms>=0 && row.available_ms>=row.endpoint_ms &&
              (i==0 || source.rows[i-1].endpoint_ms<row.endpoint_ms),
              "close source keys must be ordered, unique and available after their endpoint");
      // An unobserved poison value is never inspected or used in arithmetic.
      index.emplace(row.endpoint_ms,&row);
    }
    indexed.push_back(std::move(index));
  }
  FutureRiskTargets out;
  out.timeline_id=timeline_id;out.source_hash=source_hash;
  out.interval_ms=interval_ms;out.horizons_ticks=horizons_ticks;
  std::set<std::string> ids;
  for (const auto& origin:origins) {
    require(!origin.pair_id.empty() && ids.insert(origin.pair_id).second &&
            valid_role(origin.role) && origin.role!=Role::unassigned &&
            origin.endpoint_ms>=0 && origin.decision_ms>=0,"invalid/duplicate risk origin");
    for (std::size_t asset=0;asset<sources.size();++asset) for (auto horizon:horizons_ticks) {
      RiskTargetRow row;
      row.pair_id=origin.pair_id;row.endpoint_ms=origin.endpoint_ms;row.decision_ms=origin.decision_ms;
      row.role=origin.role;row.channel_id=sources[asset].channel_id;row.horizon_ticks=horizon;
      row.expected_returns=static_cast<std::uint64_t>(horizon);
      const auto end_ms=shift_timestamp(origin.endpoint_ms,horizon,interval_ms);
      if (!contained_in_role(roles,origin.role,origin.endpoint_ms,end_ms)) row.reasons|=risk_role_boundary;
      if (origin.decision_ms<origin.endpoint_ms) row.reasons|=risk_decision_before_origin;
      std::vector<double> log_closes;
      bool all_present=true;
      std::int64_t matured=0;
      for (std::int64_t j=0;j<=horizon;++j) {
        CloseSupport support;
        support.endpoint_ms=shift_timestamp(origin.endpoint_ms,j,interval_ms);
        double log_close=0;
        const auto found=indexed[asset].find(support.endpoint_ms);
        if (found==indexed[asset].end()) {
          row.reasons|=j==0?missing_origin_close:missing_future_close;
          all_present=false;
        } else {
          const auto& close=*found->second;
          support.present=true;support.observed=close.observed;support.available_ms=close.available_ms;
          matured=std::max(matured,close.available_ms);
          if (j==0 && close.available_ms>origin.decision_ms) row.reasons|=origin_close_unavailable;
          if (!close.observed) row.reasons|=masked_close;
          else if (!std::isfinite(close.close)) row.reasons|=nonfinite_close;
          else if (close.close<=0) row.reasons|=nonpositive_close;
          else {support.positive_finite=true;log_close=std::log(close.close);}
        }
        row.support.push_back(support);log_closes.push_back(log_close);
      }
      row.available_ms=all_present?std::optional<std::int64_t>{matured}:std::nullopt;
      // Welford's centered population variance uses all R adjacent returns,
      // including the origin-to-first-future return. Invalid pairs contribute
      // neither a value nor a count and never bridge a missing candle.
      double mean=0,m2=0,downside=0;
      for (std::int64_t j=1;j<=horizon;++j) {
        const auto i=static_cast<std::size_t>(j);
        const bool valid=row.support[i-1].positive_finite && row.support[i].positive_finite;
        row.return_valid.push_back(valid?1:0);
        if (valid) {
          const double r=log_closes[i]-log_closes[i-1];
          ++row.eligible_returns;
          const double delta=r-mean;
          mean+=delta/static_cast<double>(row.eligible_returns);
          m2+=delta*(r-mean);
        }
        if (row.support[0].positive_finite && row.support[i].positive_finite)
          downside=std::max(downside,log_closes[0]-log_closes[i]);
      }
      row.eligible=row.reasons==risk_no_issue && row.eligible_returns==row.expected_returns;
      if (row.eligible) {
        row.mean_return=mean;
        row.dispersion_1m=std::sqrt(std::max(0.0,m2/static_cast<double>(horizon)));
        row.downside=downside;
        if (!std::isfinite(row.mean_return) || !std::isfinite(row.dispersion_1m) || !std::isfinite(row.downside)) {
          row.reasons|=nonfinite_risk_value;row.eligible=false;
          row.mean_return=0;row.dispersion_1m=0;row.downside=0;
        }
      }
      out.rows.push_back(std::move(row));
    }
  }
  return out;
}

inline bool risk_target_available_by(const RiskTargetRow& row, std::int64_t fit_cutoff_ms) {
  return row.eligible && row.available_ms && *row.available_ms<=fit_cutoff_ms;
}

} // namespace cuwacunu::latent_market::v1
