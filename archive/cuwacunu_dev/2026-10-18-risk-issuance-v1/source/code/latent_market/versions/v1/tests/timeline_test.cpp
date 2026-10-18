// Copyright (c) 2026 Waajacu. All rights reserved.
// See LICENSE for the copyright and permission notice.
#include "cuwacunu/latent_market/ledger_io.h"
#include <iostream>
#include <sstream>

namespace lm=cuwacunu::latent_market::v1;
namespace {
std::size_t checks=0;
void check(bool condition,const std::string& message) {
  ++checks;if (!condition) throw std::runtime_error(message);
}
template<class Fn> void refuses(Fn fn,const std::string& message) {
  bool refused=false;try {fn();} catch (const std::exception&) {refused=true;}
  check(refused,message);
}
bool near(double a,double b,double tolerance=1e-12) {return std::abs(a-b)<=tolerance;}

lm::EmbeddingTimeline fixture() {
  lm::EmbeddingTimeline t;
  auto& id=t.identity;
  id.artifact_id="synthetic-timeline-v1";id.dataset_id="synthetic-temporal-admission-v1";
  id.encoder_family="synthetic-fixture";id.encoder_version="v1";
  id.checkpoint_hash="fixture-checkpoint";id.scaler_hash="fixture-scaler";
  id.encoder_config_hash="fixture-config";id.source_hash="fixture-source";
  id.cache_hash="fixture-cache";id.export_layout="complete-native-2";
  id.availability_policy="SIMULATED endpoint+5ms, not measured";
  id.source_closure_hash="fixture-source-closure";id.executable_hash="fixture-executable";
  id.channel_ids={11,22};id.feature_names={"close","base_volume"};id.encoder_fit_cutoff_ms=0;
  t.interval_ms=60000;t.window_ticks=4;t.embedding_width=2;
  for (std::int64_t n=0;n<=24;++n) {
    lm::EmbeddingRow row;
    row.endpoint_ms=600000+n*t.interval_ms;row.available_ms=row.endpoint_ms+5;
    row.window_id=lm::expected_window_id(id,row.endpoint_ms);row.role=lm::Role::train;row.sample_valid=true;
    row.values={static_cast<double>(n),static_cast<double>(n*n)};
    row.support.channels=2;row.support.features=2;
    for (std::int64_t k=3;k>=0;--k) row.support.timestamps_ms.push_back(row.endpoint_ms-k*t.interval_ms);
    row.support.observed.assign(16,1);t.rows.push_back(std::move(row));
  }
  lm::validate_timeline(t);return t;
}
std::vector<lm::RoleInterval> roles() {
  return {{lm::Role::train,0,2400000},{lm::Role::validation,2460000,3600000}};
}
lm::PairSpec spec() {
  lm::PairSpec s;s.history_steps=4;s.spacing_ticks=2;s.horizons_ticks={1,4};s.future_label_ticks={4};return s;
}
std::vector<lm::PairOrigin> origins() {
  return {{"origin-10",1200000,1200005,lm::Role::train,"synthetic-block-1"}};
}

void association_checks() {
  auto t=fixture();const auto o=origins();auto s=spec();
  const auto required=lm::required_window_endpoints(o,s,t.interval_ms);
  check(required==std::vector<std::int64_t>({840000,960000,1080000,1200000,1260000,1440000}),
        "history must use exact spacing and futures must have their own endpoints");
  const auto p=lm::build_pairs(t,o,s,roles());const auto& row=p.rows[0];
  check(row.history.size()==4 && row.futures.size()==2,"ledger geometry");
  check(row.history[0].row_index==4 && row.history[1].row_index==6 && row.history[3].row_index==10,
        "associations resolve exact endpoint rows");
  check(row.raw_begin_ms==660000 && row.raw_end_ms==1440000,"full raw footprint includes earliest input window");
  check(row.futures[0].eligible && row.futures[1].eligible,"complete valid fixture admitted");
  check(near(row.futures[0].overlap.geometric_fraction,0.75) &&
        row.futures[0].overlap.shared_observed_values==12 &&
        near(*row.futures[0].overlap.mask_aware_shared_fraction,0.75),"short horizon overlap counts");
  check(row.futures[1].overlap.shared_observed_values==0 &&
        row.futures[1].overlap.geometric_fraction==0,"h>=H has no literal overlap");
  check(!lm::targets_available_by(t,row,1440004) && lm::targets_available_by(t,row,1440005),
        "fit cutoff waits for future target availability");
  lm::validate_pairs(t,p);
  auto altered=p;altered.rows[0].history[0].window_id="forged";
  refuses([&]{lm::validate_pairs(t,altered);},"tampered association identity refused");
  altered=p;altered.rows[0].futures[0].eligible=false;
  refuses([&]{lm::validate_pairs(t,altered);},"tampered eligibility refused");

  const auto before=lm::origin_inputs(t,row);
  t.rows[11].values={1e30,-1e30};t.rows[14].support.observed.assign(16,0);
  t.rows[14].sample_valid=false;t.rows[14].available_ms+=1000000;
  const auto changed=lm::build_pairs(t,o,s,roles());
  check(lm::origin_inputs(t,changed.rows[0])==before,"future values/masks/availability cannot alter serving inputs");
  check(changed.rows[0].futures[0].eligible && !changed.rows[0].futures[1].eligible,
        "future perturbation changes target-side eligibility only");

  t=fixture();t.rows.erase(t.rows.begin()+6);
  const auto missing=lm::build_pairs(t,o,s,roles());
  check(missing.rows.size()==1 && !missing.rows[0].history[1].row_index &&
        missing.rows[0].history[1].endpoint_ms==960000 &&
        (missing.rows[0].futures[0].reasons&lm::missing_input),"missing input remains explicit, no row shifting");
  refuses([&]{lm::origin_inputs(t,missing.rows[0]);},"missing input cannot enter serving model");
  t=fixture();t.rows.erase(t.rows.begin()+14);
  const auto no_target=lm::build_pairs(t,o,s,roles());
  check(no_target.rows[0].futures[0].eligible && !no_target.rows[0].futures[1].target.row_index &&
        (no_target.rows[0].futures[1].reasons&lm::missing_target),"missing target preserves other horizons");
  check(lm::origin_inputs(t,no_target.rows[0])==before,"missing future does not invalidate legal origin inputs");

  t=fixture();t.rows[10].available_ms+=1;
  check(lm::build_pairs(t,o,s,roles()).rows[0].input_reasons&lm::input_unavailable,"input availability is enforced");
  t=fixture();t.rows[10].support.observed[0]=0;
  auto partial=lm::build_pairs(t,o,s,roles());
  check(partial.rows[0].input_reasons&lm::input_support,"default complete-window policy refuses partial support");
  s.input_support.require_complete_window=false;s.input_support.minimum_observed_fraction=0.9;
  s.input_support.minimum_channel_fraction=0.8;s.input_support.minimum_feature_fraction=0.8;
  partial=lm::build_pairs(t,o,s,roles());
  check(partial.rows[0].futures[0].eligible && partial.rows[0].history.back().support.observed_values==15,
        "declared partial-support view has counts and explicit thresholds");
  s.input_support.maximum_missing_run_ticks=0;
  check(lm::build_pairs(t,o,s,roles()).rows[0].input_reasons&lm::input_support,"missing-run threshold enforced");
  s=spec();t=fixture();t.rows[10].support.timestamps_ms[1]-=1;
  check(lm::build_pairs(t,o,s,roles()).rows[0].input_reasons&lm::input_timestamp_gap,"irregular actual timestamps refused");

  t=fixture();auto r=roles();r[0].begin_ms=720000;
  const auto purged=lm::build_pairs(t,o,s,r);
  check((purged.rows[0].futures[0].reasons&lm::raw_envelope_role_boundary) &&
        (purged.rows[0].input_reasons&lm::input_role_boundary),"purge uses full support, not origin row role");
  r=roles();r[0].end_ms=1500000;s.future_label_ticks={6};
  check(lm::build_pairs(t,o,s,r).rows[0].futures[0].reasons&lm::raw_envelope_role_boundary,
        "future risk support participates in full envelope even for shorter latent horizons");
  refuses([&]{lm::validate_roles({{lm::Role::train,0,100},{lm::Role::validation,100,200}});},"cross-role overlap refused");
  refuses([&]{lm::shift_timestamp(std::numeric_limits<std::int64_t>::max(),1,60000);},"future overflow refused");
  refuses([&]{lm::shift_timestamp(0,-1,60000);},"past underflow refused");
  refuses([&]{lm::shift_timestamp(1,std::numeric_limits<std::int64_t>::min(),60000);},"signed negation overflow refused");
  auto duplicates=o;duplicates.push_back(o[0]);
  refuses([&]{lm::build_pairs(t,duplicates,s,roles());},"duplicate pair IDs refused");
  t.rows.push_back(t.rows.back());
  refuses([&]{lm::validate_timeline(t);},"duplicate timestamp/window ID refused");
}

void risk_checks() {
  const std::vector<lm::PairOrigin> o={{"risk-origin",600000,600005,lm::Role::train,"risk-block"}};
  // Returns +a,-a,+a,-a have mean 0, dispersion a, downside 0.
  // All-negative constant returns have dispersion 0 and downside 4*a.
  constexpr double a=0.02;
  std::vector<lm::AssetCloseSeries> sources={{11,{}},{22,{}}};
  for (std::int64_t j=0;j<=4;++j) {
    const auto time=600000+j*60000;
    sources[0].rows.push_back({time,time+5,100*std::exp(j%2?a:0),true});
    sources[1].rows.push_back({time,time+5,100*std::exp(-a*j),true});
  }
  const auto targets=lm::build_future_risk_targets("fixture",o,sources,60000,{4},roles(),"fixture-source");
  lm::validate_risk_targets(targets);
  check(targets.rows.size()==2 && targets.rows[0].eligible && targets.rows[1].eligible,"per-asset risk rows");
  check(near(targets.rows[0].dispersion_1m,a) && near(targets.rows[0].mean_return,0) &&
        near(targets.rows[0].downside,0),"centered population dispersion uses all first-through-R returns");
  check(near(targets.rows[1].dispersion_1m,0) && near(targets.rows[1].mean_return,-a) &&
        near(targets.rows[1].downside,4*a),"dispersion differs from uncentered RV and downside uses origin");
  check(!lm::risk_target_available_by(targets.rows[0],840004) &&
        lm::risk_target_available_by(targets.rows[0],840005),"risk fit cutoff waits for every future close");
  auto absent=sources;absent[0].rows.erase(absent[0].rows.begin()+2);
  const auto gap=lm::build_future_risk_targets("fixture",o,absent,60000,{4},roles(),"fixture-source");
  check(!gap.rows[0].eligible && gap.rows[0].eligible_returns==2 && !gap.rows[0].available_ms &&
        gap.rows[0].return_valid==std::vector<std::uint8_t>({1,0,0,1}),"missing close cannot bridge two one-minute returns");
  check(gap.rows[1].eligible,"one asset's gap does not silently remove another asset's row");
  auto poison=sources;poison[0].rows[1].observed=false;
  poison[0].rows[1].close=std::numeric_limits<double>::quiet_NaN();
  const auto masked=lm::build_future_risk_targets("fixture",o,poison,60000,{4},roles(),"fixture-source");
  check(!masked.rows[0].eligible && (masked.rows[0].reasons&lm::masked_close) &&
        !(masked.rows[0].reasons&lm::nonfinite_close) && masked.rows[0].dispersion_1m==0,
        "masked poison close is not inspected or turned into an observed zero");
  poison=sources;poison[0].rows[0].close=0;
  check(lm::build_future_risk_targets("fixture",o,poison,60000,{4},roles(),"fixture-source").rows[0].reasons&
        lm::nonpositive_close,"log returns require positive observed origin close");
  poison=sources;poison[0].rows[0].available_ms+=1;
  check(lm::build_future_risk_targets("fixture",o,poison,60000,{4},roles(),"fixture-source").rows[0].reasons&
        lm::origin_close_unavailable,"risk origin cannot use a close unavailable at decision");
  auto limited=roles();limited[0].end_ms=839999;
  check(lm::build_future_risk_targets("fixture",o,sources,60000,{4},limited,"fixture-source").rows[0].reasons&
        lm::risk_role_boundary,"future risk support must lie inside origin role");
  std::stringstream binary(std::ios::in|std::ios::out|std::ios::binary);
  lm::write_risk_targets(binary,targets);
  check(lm::read_risk_targets(binary)==targets,"risk serialization exact roundtrip");
  std::stringstream invalid_binary(std::ios::in|std::ios::out|std::ios::binary);
  lm::write_risk_targets(invalid_binary,masked);
  check(lm::read_risk_targets(invalid_binary)==masked,"masked invalid risk rows survive serialization");
}

void serialization_checks() {
  auto t=fixture();t.rows[14].sample_valid=false;t.rows[14].support.observed.assign(16,0);
  t.rows[14].values.assign(2,0); // these stored zeros remain invalid
  const auto p=lm::build_pairs(t,origins(),spec(),roles());
  std::stringstream timeline(std::ios::in|std::ios::out|std::ios::binary);
  lm::write_timeline(timeline,t);const auto bytes=timeline.str();
  check(lm::read_timeline(timeline)==t,"timeline exact identity/value/mask roundtrip");
  std::stringstream pairs(std::ios::in|std::ios::out|std::ios::binary);lm::write_pairs(pairs,p);
  check(lm::read_pairs(pairs,t)==p,"pair ledger invalid associations survive exact reconstruction roundtrip");
  std::stringstream repeat(std::ios::in|std::ios::out|std::ios::binary);lm::write_timeline(repeat,t);
  check(repeat.str()==bytes,"serialization deterministic");
  std::stringstream truncated(bytes.substr(0,bytes.size()-1),std::ios::in|std::ios::binary);
  refuses([&]{lm::read_timeline(truncated);},"truncated artifact refused");
  std::stringstream trailing(bytes+"x",std::ios::in|std::ios::binary);
  refuses([&]{lm::read_timeline(trailing);},"trailing bytes refused");
  std::stringstream wrong_kind(bytes,std::ios::in|std::ios::binary);
  refuses([&]{lm::read_pairs(wrong_kind);},"wrong typed artifact refused");
  std::stringstream bounded(bytes,std::ios::in|std::ios::binary);lm::LedgerReadLimits limits;
  limits.maximum_allocated_bytes=1;
  refuses([&]{lm::read_timeline(bounded,limits);},"allocation budget enforced before vector allocation");

#ifdef __linux__
  const auto dir=std::filesystem::temp_directory_path()/
    ("cuwacunu-timeline-test-"+std::to_string(::getpid())+"-"+
     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  check(std::filesystem::create_directory(dir),"exclusive test artifact leaf");
  try {
    lm::publish_timeline(dir/"timeline.bin",t);lm::publish_pairs(dir/"pairs.bin",p);
    check(lm::load_timeline(dir/"timeline.bin")==t,"immutable file publication roundtrip");
    check(lm::load_pairs(dir/"pairs.bin",t)==p,"immutable pair file publication roundtrip");
    auto changed=t;changed.identity.artifact_id="different";
    refuses([&]{lm::publish_timeline(dir/"timeline.bin",changed);},"second publication cannot replace artifact");
    check(lm::load_timeline(dir/"timeline.bin")==t,"failed overwrite preserves original bytes");
    std::filesystem::remove(dir/"timeline.bin");std::filesystem::remove(dir/"pairs.bin");
    check(std::filesystem::remove(dir),"publication leaves no staging files");
  } catch (...) {
    std::filesystem::remove(dir/"timeline.bin");std::filesystem::remove(dir/"pairs.bin");
    std::filesystem::remove(dir);throw;
  }
#endif
}
} // namespace

int main() {
  try {
    association_checks();risk_checks();serialization_checks();
    std::cout<<"latent-market v1 temporal admission: "<<checks<<" checks passed\n";return 0;
  } catch (const std::exception& e) {
    std::cerr<<"latent-market v1 temporal admission failed after "<<checks<<" checks: "<<e.what()<<'\n';return 1;
  }
}
