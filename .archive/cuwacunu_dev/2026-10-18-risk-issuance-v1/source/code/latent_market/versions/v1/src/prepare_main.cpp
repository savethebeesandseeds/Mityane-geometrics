// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Application-owned capture. The Encoder is called once for each unique window.
#include "cuwacunu/latent_market/plan.h"
#include "cuwacunu/latent_market/ledger_io.h"
#include "cuwacunu/latent_market/risk_targets.h"
#include "cuwacunu/data/memory_mapped_dataset.h"
#include "cuwacunu/data/kline_feature_registry.h"
#include "cuwacunu/encoder/encoder.h"
#include "cuwacunu/artifacts/selected_encoder.h"
#include <torch/cuda.h>
#include <iostream>
#include <memory>

namespace lm=cuwacunu::latent_market::v1;
namespace primary=cuwacunu::encoder::primary;
namespace cp=primary::version;
namespace data=cuwacunu::data;
using Dataset=data::MemoryMappedDataset<data::kline_t>;

int main(int argc,char** argv) try {
  TORCH_CHECK(argc==3,"usage: latent_market_v1_prepare PLAN.conf NEW_OUTPUT_LEAF");
  torch::set_num_threads(1);torch::set_num_interop_threads(1);
  const auto p=lm::read_plan(argv[1]);const auto origins=lm::read_origins(p.origins);
  lm::require_pinned(p.source_pins,__FILE__);
  lm::require_pinned(p.source_pins,"code/integration/include/cuwacunu/encoder/encoder.h");
  TORCH_CHECK(torch::cuda::is_available(),"Actual CUDA required for native Encoder capture");
  const auto config=primary::read_settings(p.settings);const auto& c=config.level;
  TORCH_CHECK(c.channel_count==3 && c.input_width==9 && c.sampling_interval==60000. && c.device.is_cuda(),
      "v1 market adapter requires declared three-channel, nine-feature, minute CUDA geometry");
  TORCH_CHECK(*std::max_element(p.pairs.horizons_ticks.begin(),p.pairs.horizons_ticks.end())>=c.history_length,
      "Declare a horizon at least as large as the Encoder window");
  const auto channel_ids=cp::implementation::resolved_channel_ids(c);
  std::vector<std::unique_ptr<Dataset>> datasets;
  std::set<std::filesystem::path> distinct_caches;
  std::string cache_identity;
  for(const auto& source:p.sources) {
    auto cache=source;
    if(source.extension()==".csv") {
      cache=data::raw_binary_for_csv(source.string());
      TORCH_CHECK(data::cache_file_is_strictly_newer(cache.string(),source.string()),"Existing fresh raw mmap cache required; this runner never rebuilds");
    }
    lm::require_pinned(p.pins,cache);cache_identity+=lm::sha256_file(cache)+":"+cache.filename().string()+";";
    TORCH_CHECK(distinct_caches.insert(std::filesystem::weakly_canonical(cache)).second,"Three distinct pinned channel caches required");
    datasets.push_back(std::make_unique<Dataset>(cache.string(),c.history_length,0,torch::kFloat64));
    TORCH_CHECK(!datasets.back()->is_normalized(),"Raw caches required");
  }
  auto checked=cuwacunu::artifacts::selected_encoder::load_checked(
      p.checkpoint,p.encoder_sidecar_sha256,config,".");
  TORCH_CHECK(checked.component().model_ref==p.encoder_model_ref,"Encoder ModelRef differs from preparation selection");
  const auto frozen_state=checked.state_identity(),frozen_scaler=checked.scaler_identity();
  lm::EmbeddingTimeline timeline;
  auto& id=timeline.identity;id.artifact_id=p.experiment_id+":timeline:"+p.plan_hash;
  id.dataset_id=p.experiment_id;id.encoder_family="causal_patch_bottleneck_mae";id.encoder_version=cp::kModelTag;
  id.checkpoint_hash=p.checkpoint_hash;id.scaler_hash=frozen_scaler;id.encoder_config_hash=p.settings_hash;
  id.source_hash=p.manifest_hash;id.cache_hash=cache_identity;id.export_layout="complete-native-global";
  id.availability_policy="assumed-completed-candle-latency-ms:"+std::to_string(p.latency_ms);
  id.source_closure_hash=p.closure_hash;id.executable_hash=lm::sha256_file("/proc/self/exe");
  id.encoder_fit_cutoff_ms=p.encoder_fit_cutoff_ms;id.channel_ids=channel_ids;
  for(const auto& feature:data::kKlineFeatureRegistry)id.feature_names.emplace_back(feature.name);
  timeline.interval_ms=60000;timeline.window_ticks=c.history_length;
  timeline.embedding_width=cp::implementation::location_scale_decoder_native_width(config);
  const auto endpoints=lm::required_window_endpoints(origins,p.pairs,timeline.interval_ms);
  lm::new_leaf(argv[2]);const std::filesystem::path output=argv[2];
  std::ofstream unavailable(output/"uncaptured-windows.tsv");unavailable<<"endpoint_ms\treason\n";
  int64_t cuda_calls=0;torch::NoGradGuard no_grad;
  for(const auto endpoint:endpoints) {
    TORCH_CHECK(endpoint<=9007199254740992LL,"Cutoff must be exactly representable as an Encoder float64 timestamp");
    lm::Role role=lm::Role::unassigned;
    for(const auto& r:p.roles)if(endpoint>=r.begin_ms && endpoint<=r.end_ms)role=r.role;
    if(role==lm::Role::unassigned){unavailable<<endpoint<<"\toutside-declared-roles\n";continue;}
    bool found=true;for(const auto& d:datasets)found=found && endpoint>=d->leftmost_key_value_ && endpoint<=d->rightmost_key_value_ && d->can_get_edge_sample_at_anchor_key(endpoint,c.history_length,0);
    if(!found){unavailable<<endpoint<<"\tunavailable-complete-source-window\n";continue;}
    std::vector<torch::Tensor> values,masks,keys;
    for(const auto& d:datasets) {
      const auto sample=d->get_edge_sample_at_anchor_key(endpoint,c.history_length,0);
      if(sample.past_keys[-1].item<int64_t>()!=endpoint){found=false;break;}
      for(int64_t h=0;h<c.history_length;++h)
        if(sample.past_keys[h].item<int64_t>()!=lm::shift_timestamp(endpoint,h-c.history_length+1,timeline.interval_ms)){found=false;break;}
      if(!found)break;
      values.push_back(sample.features);masks.push_back(sample.mask);keys.push_back(sample.past_keys);
    }
    if(!found){unavailable<<endpoint<<"\tnonexact-source-window-grid\n";continue;}
    const auto raw=torch::stack(values).unsqueeze(0),row_masks=torch::stack(masks).unsqueeze(0);
    const auto observed=(row_masks.dim()==3?row_masks.unsqueeze(-1).expand_as(raw):row_masks).contiguous();
    const auto all_keys=torch::stack(keys);
    TORCH_CHECK(torch::equal(all_keys,all_keys[0].unsqueeze(0).expand_as(all_keys)),"Channels must use identical exact candle grid");
    const auto times=all_keys[0].to(torch::kFloat64);
    cp::RawHistory history{raw,observed,torch::tensor(channel_ids,torch::kInt64),times,c.sampling_interval};
    const auto cutoffs=torch::tensor({static_cast<double>(endpoint)},torch::kFloat64);
    const auto encoded=checked.encode_at(history,cutoffs);++cuda_calls;
    TORCH_CHECK(encoded.native.is_cuda() && !encoded.native.requires_grad() && torch::isfinite(encoded.native).all().item<bool>(),"Actual finite CUDA native export required");
    lm::EmbeddingRow row;row.window_id=lm::expected_window_id(id,endpoint);row.endpoint_ms=endpoint;
    TORCH_CHECK(endpoint<=INT64_MAX-p.latency_ms,"Availability overflow");row.available_ms=endpoint+p.latency_ms;
    row.role=role;row.sample_valid=encoded.valid.item<bool>();
    const auto native=encoded.native.to(torch::kCPU,torch::kFloat64).contiguous();row.values.assign(native.data_ptr<double>(),native.data_ptr<double>()+native.numel());
    row.support.channels=3;row.support.features=9;
    const auto support_keys=all_keys[0].to(torch::kInt64).contiguous();
    row.support.timestamps_ms.assign(support_keys.data_ptr<int64_t>(),support_keys.data_ptr<int64_t>()+support_keys.numel());
    const auto mask=observed[0].permute({1,0,2}).to(torch::kUInt8).contiguous();
    row.support.observed.assign(mask.data_ptr<uint8_t>(),mask.data_ptr<uint8_t>()+mask.numel());timeline.rows.push_back(std::move(row));
  }
  checked.validate_frozen();
  TORCH_CHECK(checked.state_identity()==frozen_state && checked.scaler_identity()==frozen_scaler,"Frozen Encoder/scaler changed during capture");
  lm::validate_timeline(timeline);
  const auto pairs=lm::build_pairs(timeline,origins,p.pairs,p.roles);
  std::set<int64_t> label_times;
  for(const auto& o:origins)for(const auto h:p.pairs.future_label_ticks)for(int64_t j=0;j<=h;++j)
    label_times.insert(lm::shift_timestamp(o.endpoint_ms,j,timeline.interval_ms));
  std::vector<lm::AssetCloseSeries> closes;
  for(size_t channel=0;channel<datasets.size();++channel) {
    lm::AssetCloseSeries series;series.channel_id=channel_ids[channel];const auto& d=datasets[channel];
    for(const auto time:label_times) {
      if(time<d->leftmost_key_value_ || time>d->rightmost_key_value_ || !d->can_get_edge_sample_at_anchor_key(time,1,0))continue;
      const auto s=d->get_edge_sample_at_anchor_key(time,1,0);
      if(s.past_keys[-1].item<int64_t>()!=time)continue;
      const auto close=s.features[-1][3].item<double>();const bool visible=s.mask[-1].all().item<bool>();
      TORCH_CHECK(time<=INT64_MAX-p.latency_ms,"Label availability overflow");
      series.rows.push_back({time,time+p.latency_ms,visible?close:0.,visible});
    }
    closes.push_back(std::move(series));
  }
  const auto risk=lm::build_future_risk_targets(id.artifact_id,origins,closes,timeline.interval_ms,p.pairs.future_label_ticks,p.roles,p.manifest_hash);
  lm::verify_plan(p);
  lm::publish_timeline(output/"timeline.bin",timeline);lm::publish_pairs(output/"pairs.bin",pairs);lm::publish_risk_targets(output/"future-risk.bin",risk);
  std::ofstream receipt(output/"preparation.txt");receipt<<"latent-market-preparation-v1\nexperiment "<<p.experiment_id<<"\nplan_sha256 "<<p.plan_hash
    <<"\ncard_sha256 "<<p.card_hash<<"\nunique_cuda_captures "<<cuda_calls<<"\nencoder_updates 0\nscaler_fits 0\nsource_manifest_sha256 "<<p.manifest_hash
    <<"\nsource_closure_sha256 "<<p.closure_hash<<"\nencoder_sidecar_sha256 "<<p.encoder_sidecar_sha256
    <<"\nencoder_model_ref "<<checked.component().model_ref<<"\nbinary_sha256 "<<id.executable_hash<<"\ntimeline_sha256 "<<lm::sha256_file(output/"timeline.bin")
    <<"\npairs_sha256 "<<lm::sha256_file(output/"pairs.bin")<<"\nrisk_sha256 "<<lm::sha256_file(output/"future-risk.bin")<<"\n";
  TORCH_CHECK(receipt.good() && unavailable.good(),"Preparation receipt write failed");
  std::cout<<"Prepared "<<timeline.rows.size()<<" unique native windows; no neural updates or quality fitting.\n";
  return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
