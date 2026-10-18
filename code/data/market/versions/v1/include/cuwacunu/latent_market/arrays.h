// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../LICENSE.
#pragma once
#include "cuwacunu/latent_market/timeline.h"
#include "cuwacunu/evaluation/latent_dynamics/v1/baselines.h"

namespace cuwacunu::latent_market::v1 {
namespace metrics=cuwacunu::evaluation::latent_dynamics::v1;
// One version-owned coordinate contract for fresh producers and consumers.
// Historical archives retain their original tokens; readers do not normalize them.
inline std::string target_state_coordinate_id(const std::string& timeline_id,
                                             const std::string& checkpoint_sha256) {
  return timeline_id+":Target State Representation Network:"+checkpoint_sha256;
}
inline metrics::Role metric_role(Role r) {
  if(r==Role::train)return metrics::Role::train;
  if(r==Role::validation)return metrics::Role::validation;
  if(r==Role::test)return metrics::Role::test;
  throw std::runtime_error("Unassigned timeline row cannot become an evaluation row");
}
inline metrics::ArrayBank array_bank(const EmbeddingTimeline& t) {
  validate_timeline(t);metrics::ArrayBank b;b.coordinate_id=t.identity.artifact_id+":native-e";
  b.interval_ms=t.interval_ms;b.window_ticks=t.window_ticks;
  b.values=torch::zeros({static_cast<int64_t>(t.rows.size()),static_cast<int64_t>(t.embedding_width)},torch::kFloat64);
  b.valid=torch::zeros({static_cast<int64_t>(t.rows.size())},torch::kBool);
  for(size_t n=0;n<t.rows.size();++n) {
    const auto& r=t.rows[n];const auto s=summarize_support(t,r,SupportPolicy{});
    b.values[static_cast<int64_t>(n)].copy_(torch::tensor(r.values,torch::kFloat64));b.valid[static_cast<int64_t>(n)]=r.sample_valid;
    metrics::ArrayRow a;a.row_id=n+1;a.endpoint_ms=r.endpoint_ms;a.available_ms=r.available_ms;
    a.support_begin_ms=r.support.timestamps_ms.front();a.support_end_ms=r.endpoint_ms;a.role=metric_role(r.role);
    a.complete_support=s.complete;a.stale=s.endpoint_staleness_ticks>0;
    a.observed_fraction=s.expected_values?double(s.observed_values)/s.expected_values:0;b.rows.push_back(a);
  }
  return b;
}
inline std::vector<metrics::Association> associations(const EmbeddingTimeline& t,const DynamicsPairs& pairs) {
  validate_pairs(t,pairs);std::vector<metrics::Association> result;
  for(size_t n=0;n<pairs.rows.size();++n) {
    const auto& p=pairs.rows[n];
    for(const auto& f:p.futures) {
      metrics::Association a;a.pair_id=result.size()+1;a.origin_row_id=p.history.back().row_index?*p.history.back().row_index+1:-1;
      a.target_row_id=f.target.row_index?*f.target.row_index+1:-1;a.horizon_ticks=f.horizon_ticks;
      a.decision_ms=p.decision_ms;a.input_eligible=p.input_reasons==no_issue;
      a.target_eligible=f.target.reasons==no_issue && !(f.reasons&raw_envelope_role_boundary);
      a.ledger_input_reasons=p.input_reasons;a.ledger_target_reasons=f.target.reasons|(f.reasons&raw_envelope_role_boundary);
      a.actual_overlap_fraction=f.overlap.mask_aware_shared_fraction.value_or(f.overlap.geometric_fraction);
      a.dependence_group=p.dependence_group;
      for(const auto& h:p.history)a.history_row_ids.push_back(h.row_index?*h.row_index+1:-1);
      result.push_back(std::move(a));
    }
  }
  return result;
}
inline torch::Tensor training_window_indices(const EmbeddingTimeline& t,const std::vector<RoleInterval>& roles,int64_t fit_cutoff) {
  std::vector<int64_t> index;
  for(size_t n=0;n<t.rows.size();++n) {
    const auto& r=t.rows[n];
    if(r.role==Role::train && r.sample_valid && r.available_ms<=fit_cutoff &&
       contained_in_role(roles,Role::train,r.support.timestamps_ms.front(),r.endpoint_ms) && summarize_support(t,r,SupportPolicy{}).eligible)
      index.push_back(static_cast<int64_t>(n));
  }
  TORCH_CHECK(index.size()>=2,"At least two complete unique TRAIN windows must mature before fit cutoff");return torch::tensor(index,torch::kInt64);
}
} // namespace cuwacunu::latent_market::v1
