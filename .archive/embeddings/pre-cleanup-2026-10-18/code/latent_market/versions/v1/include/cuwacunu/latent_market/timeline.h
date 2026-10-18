// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuwacunu::latent_market::v1 {

enum class Role : std::uint8_t { unassigned, train, validation, test };

inline const char* role_name(Role role) {
  switch (role) {
    case Role::unassigned: return "UNASSIGNED";
    case Role::train: return "TRAIN";
    case Role::validation: return "VALIDATION";
    case Role::test: return "TEST";
  }
  throw std::invalid_argument("invalid chronological role");
}

namespace temporal_detail {
inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("[latent-market v1] " + message);
}
inline std::size_t product(std::size_t a, std::size_t b) {
  require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
          "size product overflow");
  return a * b;
}
inline std::int64_t nonnegative_product(std::int64_t a, std::int64_t b) {
  require(a >= 0 && b >= 0 && (b == 0 || a <= std::numeric_limits<std::int64_t>::max()/b),
          "timestamp/geometry product overflow");
  return a*b;
}
inline bool valid_role(Role role) {
  return static_cast<std::uint8_t>(role) <= static_cast<std::uint8_t>(Role::test);
}
} // namespace temporal_detail

// Exact signed tick arithmetic. In particular, a missing timestamp is never
// replaced by a neighboring row and a timestamp overflow is never wrapped.
inline std::int64_t shift_timestamp(std::int64_t endpoint_ms,
                                    std::int64_t ticks,
                                    std::int64_t interval_ms) {
  using namespace temporal_detail;
  require(endpoint_ms >= 0 && interval_ms > 0, "nonnegative timestamp and positive interval required");
  if (ticks >= 0) {
    const auto delta = nonnegative_product(ticks, interval_ms);
    require(endpoint_ms <= std::numeric_limits<std::int64_t>::max()-delta,
            "future timestamp overflow");
    return endpoint_ms+delta;
  }
  require(ticks != std::numeric_limits<std::int64_t>::min(), "past tick overflow");
  const auto delta = nonnegative_product(-ticks, interval_ms);
  require(endpoint_ms >= delta, "past timestamp before Unix epoch");
  return endpoint_ms-delta;
}

struct RoleInterval {
  Role role = Role::unassigned;
  std::int64_t begin_ms = 0, end_ms = 0; // inclusive exchange-close keys
  bool operator==(const RoleInterval&) const = default;
};

inline void validate_roles(const std::vector<RoleInterval>& roles) {
  using namespace temporal_detail;
  require(!roles.empty(), "role intervals are required");
  std::set<Role> seen;
  auto sorted = roles;
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.begin_ms < b.begin_ms;
  });
  for (std::size_t i=0; i<sorted.size(); ++i) {
    const auto& r=sorted[i];
    require(valid_role(r.role) && r.role != Role::unassigned && r.begin_ms >= 0 &&
            r.end_ms >= r.begin_ms, "invalid role interval");
    require(seen.insert(r.role).second, "one contiguous interval per role required");
    require(i == 0 || sorted[i-1].end_ms < r.begin_ms, "roles overlap");
  }
}

inline bool contained_in_role(const std::vector<RoleInterval>& roles, Role role,
                              std::int64_t begin_ms, std::int64_t end_ms) {
  return std::any_of(roles.begin(), roles.end(), [&](const auto& r) {
    return role == r.role && begin_ms >= r.begin_ms && end_ms <= r.end_ms;
  });
}

struct TimelineIdentity {
  std::string artifact_id, dataset_id;
  std::string encoder_family, encoder_version, checkpoint_hash, scaler_hash;
  std::string encoder_config_hash, source_hash, cache_hash, export_layout;
  std::string availability_policy, source_closure_hash, executable_hash;
  std::int64_t encoder_fit_cutoff_ms = 0;
  std::vector<std::int64_t> channel_ids;
  std::vector<std::string> feature_names;
  bool operator==(const TimelineIdentity&) const = default;
};

inline void validate_identity(const TimelineIdentity& id) {
  using namespace temporal_detail;
  const std::string* required[] = {&id.artifact_id,&id.dataset_id,&id.encoder_family,
    &id.encoder_version,&id.checkpoint_hash,&id.scaler_hash,&id.encoder_config_hash,
    &id.source_hash,&id.cache_hash,&id.export_layout,&id.availability_policy,
    &id.source_closure_hash,&id.executable_hash};
  for (const auto* field: required) require(!field->empty(), "all timeline identity bindings are required");
  require(id.encoder_fit_cutoff_ms >= 0, "negative encoder fit cutoff");
  require(!id.channel_ids.empty() && !id.feature_names.empty(), "semantic channels/features required");
  require(std::set<std::int64_t>(id.channel_ids.begin(),id.channel_ids.end()).size()==id.channel_ids.size(),
          "duplicate semantic channel ID");
  require(std::set<std::string>(id.feature_names.begin(),id.feature_names.end()).size()==id.feature_names.size(),
          "duplicate feature name");
  for (const auto& name:id.feature_names) require(!name.empty(), "empty feature name");
}

struct WindowSupport {
  // One actual candle timestamp per temporal position. Non-regular timestamps
  // can be archived, but the first protocol refuses them for associations.
  std::vector<std::int64_t> timestamps_ms;
  std::size_t channels = 0, features = 0;
  std::vector<std::uint8_t> observed; // [H,C,F], time-major, exactly 0 or 1
  bool operator==(const WindowSupport&) const = default;
};

struct EmbeddingRow {
  std::string window_id;
  std::int64_t endpoint_ms = 0, available_ms = 0;
  Role role = Role::unassigned;
  bool sample_valid = false; // Encoder sample validity, independent of eligibility
  WindowSupport support;
  std::vector<double> values; // complete native CPU export; no appended metadata
  bool operator==(const EmbeddingRow&) const = default;
};

struct EmbeddingTimeline {
  TimelineIdentity identity;
  std::int64_t interval_ms = 0, window_ticks = 0;
  std::size_t embedding_width = 0;
  std::vector<EmbeddingRow> rows; // strictly increasing exact endpoints
  bool operator==(const EmbeddingTimeline&) const = default;
};

inline std::string expected_window_id(const TimelineIdentity& id, std::int64_t endpoint_ms) {
  return id.artifact_id + ":window:" + std::to_string(endpoint_ms);
}

inline void validate_timeline(const EmbeddingTimeline& timeline) {
  using namespace temporal_detail;
  validate_identity(timeline.identity);
  require(timeline.interval_ms > 0 && timeline.window_ticks > 0 && timeline.embedding_width > 0,
          "positive timeline geometry required");
  nonnegative_product(timeline.window_ticks-1,timeline.interval_ms);
  std::set<std::string> ids;
  for (std::size_t i=0; i<timeline.rows.size(); ++i) {
    const auto& row=timeline.rows[i];
    require(!row.window_id.empty() && ids.insert(row.window_id).second, "duplicate/empty window ID");
    require(row.endpoint_ms >= 0 && row.available_ms >= row.endpoint_ms && valid_role(row.role),
            "invalid endpoint, availability or role");
    require(i == 0 || timeline.rows[i-1].endpoint_ms < row.endpoint_ms,
            "timeline endpoints must be unique and strictly ordered");
    require(row.values.size()==timeline.embedding_width, "embedding width mismatch");
    // Invalid rows remain present, but native stored arrays never carry poison.
    for (auto value:row.values) require(std::isfinite(value), "nonfinite native embedding value");
    const auto& s=row.support;
    require(s.channels==timeline.identity.channel_ids.size() &&
            s.features==timeline.identity.feature_names.size() &&
            s.timestamps_ms.size()==static_cast<std::size_t>(timeline.window_ticks),
            "support geometry/semantic order mismatch");
    require(s.observed.size()==product(product(s.timestamps_ms.size(),s.channels),s.features),
            "support mask shape mismatch");
    require(s.timestamps_ms.back()==row.endpoint_ms, "actual final support key differs from endpoint");
    for (std::size_t j=0;j<s.timestamps_ms.size();++j) {
      require(s.timestamps_ms[j]>=0 && (j==0 || s.timestamps_ms[j-1]<s.timestamps_ms[j]),
              "actual support keys must be increasing nonnegative timestamps");
    }
    for (auto mask:s.observed) require(mask<=1, "observed mask must be binary");
    require(!row.sample_valid || std::any_of(s.observed.begin(),s.observed.end(),[](auto m){return m!=0;}),
            "valid encoder sample has no observed support");
  }
}

struct SupportPolicy {
  bool require_complete_window = true;
  double minimum_observed_fraction = 1.0;
  double minimum_channel_fraction = 1.0;
  double minimum_feature_fraction = 1.0;
  std::size_t maximum_missing_run_ticks = std::numeric_limits<std::size_t>::max();
  std::size_t maximum_endpoint_staleness_ticks = 0;
  bool operator==(const SupportPolicy&) const = default;
};

struct SupportSummary {
  std::uint64_t observed_values = 0, expected_values = 0;
  std::vector<std::uint64_t> channel_observed, feature_observed;
  std::size_t maximum_missing_run_ticks = 0, endpoint_staleness_ticks = 0;
  bool regular_timestamps = false, complete = false, eligible = false;
  bool operator==(const SupportSummary&) const = default;
};

inline void validate_support_policy(const SupportPolicy& policy) {
  for (auto fraction: {policy.minimum_observed_fraction,policy.minimum_channel_fraction,
                       policy.minimum_feature_fraction})
    temporal_detail::require(std::isfinite(fraction) && fraction>=0 && fraction<=1,
                             "support fractions must lie in [0,1]");
}

inline SupportSummary summarize_support(const EmbeddingTimeline& t, const EmbeddingRow& row,
                                         const SupportPolicy& policy) {
  validate_support_policy(policy);
  const auto& s=row.support;
  SupportSummary out;
  out.expected_values=s.observed.size();
  out.channel_observed.assign(s.channels,0);
  out.feature_observed.assign(s.features,0);
  out.regular_timestamps=true;
  for (std::size_t j=0;j<s.timestamps_ms.size();++j) {
    const auto ticks=static_cast<std::int64_t>(s.timestamps_ms.size()-1-j);
    // A support beginning before epoch is invalid for this geometry, not a
    // wrapped timestamp or an exception that discards the association.
    const auto lookback=temporal_detail::nonnegative_product(ticks,t.interval_ms);
    if (row.endpoint_ms<lookback || s.timestamps_ms[j]!=row.endpoint_ms-lookback)
      out.regular_timestamps=false;
  }
  for (std::size_t c=0;c<s.channels;++c) for (std::size_t f=0;f<s.features;++f) {
    std::size_t missing_run=0;
    for (std::size_t j=0;j<s.timestamps_ms.size();++j) {
      const auto mask=s.observed[(j*s.channels+c)*s.features+f];
      if (mask) {
        ++out.observed_values; ++out.channel_observed[c]; ++out.feature_observed[f];
        missing_run=0;
      } else {
        ++missing_run;
        out.maximum_missing_run_ticks=std::max(out.maximum_missing_run_ticks,missing_run);
      }
    }
    out.endpoint_staleness_ticks=std::max(out.endpoint_staleness_ticks,missing_run);
  }
  out.complete=out.observed_values==out.expected_values;
  out.eligible=out.regular_timestamps && (!policy.require_complete_window || out.complete) &&
    static_cast<double>(out.observed_values)>=policy.minimum_observed_fraction*out.expected_values &&
    out.maximum_missing_run_ticks<=policy.maximum_missing_run_ticks &&
    out.endpoint_staleness_ticks<=policy.maximum_endpoint_staleness_ticks;
  for (auto n:out.channel_observed)
    out.eligible=out.eligible && static_cast<double>(n)>=policy.minimum_channel_fraction*
      static_cast<double>(s.timestamps_ms.size()*s.features);
  for (auto n:out.feature_observed)
    out.eligible=out.eligible && static_cast<double>(n)>=policy.minimum_feature_fraction*
      static_cast<double>(s.timestamps_ms.size()*s.channels);
  return out;
}

enum PairIssue : std::uint64_t {
  no_issue=0, missing_input=1ull<<0, missing_target=1ull<<1,
  invalid_input=1ull<<2, invalid_target=1ull<<3,
  input_support=1ull<<4, target_support=1ull<<5,
  input_timestamp_gap=1ull<<6, target_timestamp_gap=1ull<<7,
  input_role_boundary=1ull<<8, target_role_boundary=1ull<<9,
  raw_envelope_role_boundary=1ull<<10, input_unavailable=1ull<<11,
  decision_before_origin=1ull<<12
};

inline std::vector<std::string> pair_issue_names(std::uint64_t issues) {
  const std::pair<PairIssue,const char*> names[]={{missing_input,"missing_input"},
    {missing_target,"missing_target"},{invalid_input,"invalid_input"},{invalid_target,"invalid_target"},
    {input_support,"input_support"},{target_support,"target_support"},
    {input_timestamp_gap,"input_timestamp_gap"},{target_timestamp_gap,"target_timestamp_gap"},
    {input_role_boundary,"input_role_boundary"},{target_role_boundary,"target_role_boundary"},
    {raw_envelope_role_boundary,"raw_envelope_role_boundary"},{input_unavailable,"input_unavailable"},
    {decision_before_origin,"decision_before_origin"}};
  std::vector<std::string> result;
  for (const auto& [bit,name]:names) if (issues & bit) result.emplace_back(name);
  return result;
}

struct PairSpec {
  std::int64_t history_steps = 1, spacing_ticks = 1;
  std::vector<std::int64_t> horizons_ticks, future_label_ticks;
  SupportPolicy input_support, target_support;
  bool operator==(const PairSpec&) const = default;
};

inline void validate_pair_spec(const PairSpec& spec) {
  using namespace temporal_detail;
  require(spec.history_steps>0 && spec.spacing_ticks>0 && !spec.horizons_ticks.empty(),
          "positive history, spacing and nonempty horizon panel required");
  nonnegative_product(spec.history_steps-1,spec.spacing_ticks);
  for (const auto* panel:{&spec.horizons_ticks,&spec.future_label_ticks}) {
    std::set<std::int64_t> seen;
    for (auto h:*panel) require(h>0 && seen.insert(h).second, "positive unique horizons required");
  }
  validate_support_policy(spec.input_support); validate_support_policy(spec.target_support);
}

struct PairOrigin {
  std::string pair_id;
  std::int64_t endpoint_ms = 0, decision_ms = 0;
  Role role = Role::unassigned;
  std::string dependence_group;
  bool operator==(const PairOrigin&) const = default;
};

struct WindowAssociation {
  std::int64_t endpoint_ms = 0;
  std::string window_id;
  std::optional<std::size_t> row_index;
  std::uint64_t reasons = no_issue;
  SupportSummary support;
  bool operator==(const WindowAssociation&) const = default;
};

struct OverlapSummary {
  double geometric_fraction = 0;
  std::uint64_t shared_observed_values = 0, origin_observed_values = 0, target_observed_values = 0;
  std::optional<double> mask_aware_shared_fraction; // denominator: observed target values
  bool operator==(const OverlapSummary&) const = default;
};

struct HorizonAssociation {
  std::int64_t horizon_ticks = 0;
  WindowAssociation target;
  std::uint64_t reasons = no_issue; // union of input, target and full-envelope issues
  bool eligible = false;
  OverlapSummary overlap;
  bool operator==(const HorizonAssociation&) const = default;
};

struct DynamicsPair {
  std::string pair_id;
  std::int64_t endpoint_ms = 0, decision_ms = 0;
  Role role = Role::unassigned;
  std::string dependence_group;
  std::vector<WindowAssociation> history; // oldest first, ending at origin
  std::uint64_t input_reasons = no_issue;
  std::int64_t raw_begin_ms = 0, raw_end_ms = 0; // full conservative envelope, all labels included
  std::vector<HorizonAssociation> futures;
  bool operator==(const DynamicsPair&) const = default;
};

struct DynamicsPairs {
  std::string timeline_id;
  PairSpec spec;
  std::vector<RoleInterval> roles;
  std::vector<DynamicsPair> rows; // every requested origin, including invalid ones
  bool operator==(const DynamicsPairs&) const = default;
};

inline std::vector<std::int64_t> required_window_endpoints(const std::vector<PairOrigin>& origins,
                                                          const PairSpec& spec,
                                                          std::int64_t interval_ms) {
  validate_pair_spec(spec);
  std::set<std::int64_t> times;
  for (const auto& origin:origins) {
    for (std::int64_t k=spec.history_steps; k>0; --k)
      times.insert(shift_timestamp(origin.endpoint_ms,-temporal_detail::nonnegative_product(k-1,spec.spacing_ticks),interval_ms));
    for (auto h:spec.horizons_ticks) times.insert(shift_timestamp(origin.endpoint_ms,h,interval_ms));
  }
  return {times.begin(),times.end()};
}

inline OverlapSummary support_overlap(const EmbeddingTimeline& t, const EmbeddingRow& origin,
                                      const EmbeddingRow& target, std::int64_t horizon_ticks) {
  OverlapSummary out;
  out.geometric_fraction=static_cast<double>(std::max(t.window_ticks-horizon_ticks,std::int64_t{0}))/t.window_ticks;
  const auto& a=origin.support; const auto& b=target.support;
  const auto width=temporal_detail::product(a.channels,a.features);
  for (auto m:a.observed) out.origin_observed_values+=m;
  for (auto m:b.observed) out.target_observed_values+=m;
  std::size_t i=0,j=0;
  while (i<a.timestamps_ms.size() && j<b.timestamps_ms.size()) {
    if (a.timestamps_ms[i]<b.timestamps_ms[j]) {++i;continue;}
    if (a.timestamps_ms[i]>b.timestamps_ms[j]) {++j;continue;}
    for (std::size_t cell=0;cell<width;++cell)
      out.shared_observed_values+=a.observed[i*width+cell] && b.observed[j*width+cell];
    ++i;++j;
  }
  if (out.target_observed_values)
    out.mask_aware_shared_fraction=static_cast<double>(out.shared_observed_values)/out.target_observed_values;
  return out;
}

inline DynamicsPairs build_pairs(const EmbeddingTimeline& t, const std::vector<PairOrigin>& origins,
                                 const PairSpec& spec, const std::vector<RoleInterval>& roles) {
  using namespace temporal_detail;
  validate_timeline(t); validate_pair_spec(spec); validate_roles(roles);
  std::map<std::int64_t,std::size_t> index;
  for (std::size_t i=0;i<t.rows.size();++i) index.emplace(t.rows[i].endpoint_ms,i);
  std::set<std::string> pair_ids;
  DynamicsPairs out{t.identity.artifact_id,spec,roles,{}};
  auto associate=[&](std::int64_t time,const PairOrigin& origin,bool input) {
    WindowAssociation a;
    a.endpoint_ms=time; a.window_id=expected_window_id(t.identity,time);
    const auto found=index.find(time);
    if (found==index.end()) {a.reasons=input?missing_input:missing_target;return a;}
    a.row_index=found->second;
    const auto& row=t.rows[*a.row_index]; a.window_id=row.window_id;
    a.support=summarize_support(t,row,input?spec.input_support:spec.target_support);
    if (!row.sample_valid) a.reasons|=input?invalid_input:invalid_target;
    if (!a.support.eligible) a.reasons|=input?input_support:target_support;
    if (!a.support.regular_timestamps) a.reasons|=input?input_timestamp_gap:target_timestamp_gap;
    const auto raw_begin=row.support.timestamps_ms.front();
    if (row.role!=origin.role || !contained_in_role(roles,origin.role,raw_begin,time))
      a.reasons|=input?input_role_boundary:target_role_boundary;
    if (input && row.available_ms>origin.decision_ms) a.reasons|=input_unavailable;
    return a;
  };
  auto q=*std::max_element(spec.horizons_ticks.begin(),spec.horizons_ticks.end());
  for (auto r:spec.future_label_ticks) q=std::max(q,r);
  const auto history_span=nonnegative_product(spec.history_steps-1,spec.spacing_ticks);
  require(t.window_ticks-1<=std::numeric_limits<std::int64_t>::max()-history_span,
          "full support tick envelope overflow");
  for (const auto& origin:origins) {
    require(!origin.pair_id.empty() && pair_ids.insert(origin.pair_id).second,
            "duplicate/empty pair ID");
    require(valid_role(origin.role) && origin.role!=Role::unassigned && origin.endpoint_ms>=0 &&
            origin.decision_ms>=0 && !origin.dependence_group.empty(),
            "origin role, timestamp and declared dependence group required");
    DynamicsPair pair;
    pair.pair_id=origin.pair_id;pair.endpoint_ms=origin.endpoint_ms;
    pair.decision_ms=origin.decision_ms;pair.role=origin.role;pair.dependence_group=origin.dependence_group;
    pair.raw_begin_ms=shift_timestamp(origin.endpoint_ms,-(t.window_ticks-1+history_span),t.interval_ms);
    pair.raw_end_ms=shift_timestamp(origin.endpoint_ms,q,t.interval_ms);
    if (origin.decision_ms<origin.endpoint_ms) pair.input_reasons|=decision_before_origin;
    for (std::int64_t k=spec.history_steps;k>0;--k) {
      const auto time=shift_timestamp(origin.endpoint_ms,-nonnegative_product(k-1,spec.spacing_ticks),t.interval_ms);
      pair.history.push_back(associate(time,origin,true));
      pair.input_reasons|=pair.history.back().reasons;
    }
    const bool envelope_ok=contained_in_role(roles,origin.role,pair.raw_begin_ms,pair.raw_end_ms);
    for (auto h:spec.horizons_ticks) {
      HorizonAssociation future;
      future.horizon_ticks=h;
      future.target=associate(shift_timestamp(origin.endpoint_ms,h,t.interval_ms),origin,false);
      future.reasons=pair.input_reasons|future.target.reasons;
      if (!envelope_ok) future.reasons|=raw_envelope_role_boundary;
      future.eligible=future.reasons==no_issue;
      future.overlap.geometric_fraction=static_cast<double>(std::max(t.window_ticks-h,std::int64_t{0}))/t.window_ticks;
      if (pair.history.back().row_index && future.target.row_index)
        future.overlap=support_overlap(t,t.rows[*pair.history.back().row_index],t.rows[*future.target.row_index],h);
      pair.futures.push_back(std::move(future));
    }
    out.rows.push_back(std::move(pair));
  }
  return out;
}

// Independent reconstruction catches tampered IDs, eligibility, availability,
// support counts or coordinate associations when opening an external ledger.
inline void validate_pairs(const EmbeddingTimeline& timeline, const DynamicsPairs& pairs) {
  temporal_detail::require(pairs.timeline_id==timeline.identity.artifact_id,"pair/timeline identity mismatch");
  std::vector<PairOrigin> origins;
  for (const auto& p:pairs.rows)
    origins.push_back({p.pair_id,p.endpoint_ms,p.decision_ms,p.role,p.dependence_group});
  temporal_detail::require(build_pairs(timeline,origins,pairs.spec,pairs.roles)==pairs,
                           "pair ledger differs from exact timestamp/support reconstruction");
}

// This serving view intentionally has no target IDs, target validity, future
// masks, realized errors, future availability or full-support endpoint.
struct OriginInputs {
  std::string pair_id, timeline_id;
  std::int64_t endpoint_ms=0, decision_ms=0;
  std::vector<std::string> window_ids;
  std::vector<std::int64_t> endpoints_ms, available_ms;
  std::vector<double> flattened_values; // [K,D], oldest first
  bool operator==(const OriginInputs&) const = default;
};

inline OriginInputs origin_inputs(const EmbeddingTimeline& timeline, const DynamicsPair& pair) {
  temporal_detail::require(pair.input_reasons==no_issue && !pair.history.empty(),
                           "ineligible/empty origin cannot enter a model");
  OriginInputs out;
  out.pair_id=pair.pair_id;out.timeline_id=timeline.identity.artifact_id;
  out.endpoint_ms=pair.endpoint_ms;out.decision_ms=pair.decision_ms;
  for (const auto& a:pair.history) {
    temporal_detail::require(a.row_index && *a.row_index<timeline.rows.size() && a.reasons==no_issue &&
      a.support.eligible,"missing/ineligible input association");
    const auto& row=timeline.rows[*a.row_index];
    temporal_detail::require(row.window_id==a.window_id && row.endpoint_ms==a.endpoint_ms &&
      row.available_ms<=pair.decision_ms && row.endpoint_ms<=pair.endpoint_ms && row.sample_valid,
      "input identity/availability mismatch");
    out.window_ids.push_back(row.window_id);out.endpoints_ms.push_back(row.endpoint_ms);
    out.available_ms.push_back(row.available_ms);
    out.flattened_values.insert(out.flattened_values.end(),row.values.begin(),row.values.end());
  }
  return out;
}

inline bool targets_available_by(const EmbeddingTimeline& timeline, const DynamicsPair& pair,
                                 std::int64_t fit_cutoff_ms) {
  for (const auto& future:pair.futures) {
    if (!future.eligible || !future.target.row_index || *future.target.row_index>=timeline.rows.size()) return false;
    const auto& row=timeline.rows[*future.target.row_index];
    if (row.window_id!=future.target.window_id || row.endpoint_ms!=future.target.endpoint_ms ||
        row.available_ms>fit_cutoff_ms) return false;
  }
  return !pair.futures.empty();
}

} // namespace cuwacunu::latent_market::v1
